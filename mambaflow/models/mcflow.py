from __future__ import annotations

import math

from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

from mambaflow.backbones.mambavoice import MambaVoiceConditioner
from mambaflow.flow import FlowConfig, euler_sample, project_sources_to_mixture, project_velocity_zero_sum, sample_training_tuple
from mambaflow.audio import (
    STFTConfig,
    stft_waveform,
    istft_waveform,
    complex_to_ri,
    sources_complex_to_ri,
    ri_to_sources_complex,
    project_ri_sources_to_mixture,
    project_ri_velocity_zero_sum,
)
from mambaflow.losses import multi_resolution_stft_loss
from .flow_head import WaveformFlowHead
from .spec_unet_flow_head import SpecUNetFlowHead
from .diffvs_unet_flow_head import DiffVSUNetFlowHead
from .tfc_tdf_unet_flow_head import TFCTDFUNetFlowHead
from .mamba_hybrid_flow_head import MambaHybridFlowHead

def si_sdr_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable per-sample SI-SDR score in dB.

    pred, target: [B, L]. Higher is better. This is used as a training
    anchor in DAVIS target-only mode to discourage leaving the target in the
    residual (mixture - prediction).
    """
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (pred * target).sum(dim=-1, keepdim=True) / (
        target.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    target_scaled = scale * target
    noise = pred - target_scaled
    ratio = target_scaled.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


class MambaVoiceMCFlow(nn.Module):
    """Mixture-consistent rectified flow separator conditioned by MambaVoice encoders.

    Two velocity heads are supported:
      - head.type: waveform   -> legacy 1-D WaveNet-style flow head.
      - head.type: spec_unet  -> legacy 2-D complex-STFT U-Net flow head.
      - head.type: diffvs_unet -> Diff-VS-inspired audio-aware DDPM++ U-Net.
      - head.type: tfc_tdf_unet -> four-level TFC-TDF U-Net (VIST).
      - head.type: mamba_hybrid -> MambaVoice hybrid Mamba-Transformer.

    The spectrogram head is the recommended SOTA path. It predicts velocity over
    target/residual complex STFT channels and then reconstructs the waveform with
    iSTFT, while still using the same MambaVoice audio/video conditioner.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        cond_dim = int(model_cfg.get("cond_dim", 256))
        self.conditioner = MambaVoiceConditioner(
            cfg.get("backbone", {}),
            cond_dim=cond_dim,
            gate_type=str(model_cfg.get("gate_type", "residual_tanh")),
            gate_strength=float(model_cfg.get("gate_strength", 1.0)),
            global_condition_type=str(model_cfg.get("global_condition_type", "av_gated")),
            temporal_token_type=str(model_cfg.get("temporal_token_type", "av_gated")),
            audio_condition_dropout=float(model_cfg.get("audio_condition_dropout", 0.0)),
            visual_token_dropout=float(model_cfg.get("visual_token_dropout", 0.0)),
            visual_token_noise_std=float(model_cfg.get("visual_token_noise_std", 0.0)),
            debug_feature_shapes=bool(model_cfg.get("debug_feature_shapes", False)),
            raw_visual_map=bool(model_cfg.get("raw_visual_map", False)),
            concat_learned_activity=bool(model_cfg.get("concat_learned_activity", True)),
        )

        flow_cfg = cfg.get("flow", {})
        self.spec_target_mode = str(flow_cfg.get("target_mode", "source_pair")).lower()
        if self.spec_target_mode in {"pair", "source_pair", "target_residual", "target+residual"}:
            self.spec_target_mode = "source_pair"
        elif self.spec_target_mode in {"target_only", "target", "davis", "davis_target"}:
            self.spec_target_mode = "target_only"
        else:
            raise ValueError(
                "flow.target_mode must be 'source_pair' or 'target_only', "
                f"got {self.spec_target_mode!r}"
            )
        self.spec_init_mode = str(flow_cfg.get("init_mode", "noise")).lower()
        self.spec_init_noise_scale = float(flow_cfg.get("init_noise_scale", 0.0))

        # Training objective for target-only spectrogram models.
        #   flow          : standard conditional flow matching, integrates x_0 -> x_1.
        #   drift_only    : one-step drift/correction field, pred = z + D_theta(z,c).
        #   residual_flow : mixture-anchored rectified flow.  The underlying
        #                   residual path is r_t=t(S-M), while the network sees
        #                   the equivalent current audio state X_t=M+r_t.
        #   alphaflow     : finite-interval mean velocity on the same M->S path,
        #                   trained with a local FM anchor plus JVP-free
        #                   AlphaFlow teacher-student interval consistency.
        # The drift objective is an experimental separator, not a refinement model:
        # it still starts from the mixture/noisy mixture and directly predicts the target.
        self.spec_objective = str(flow_cfg.get("objective", flow_cfg.get("training_objective", "flow"))).lower()
        if self.spec_objective in {"flow", "cfm", "rectified_flow", "flow_matching"}:
            self.spec_objective = "flow"
        elif self.spec_objective in {"drift", "drift_only", "one_step", "one_step_drift"}:
            self.spec_objective = "drift_only"
        elif self.spec_objective in {"hybrid", "hybrid_flow", "hybrid_flow_matching"}:
            self.spec_objective = "hybrid_flow"
            self.hybrid_endpoint_prob = float(flow_cfg.get("endpoint_prob", 0.5))
        elif self.spec_objective in {
            "residual_flow",
            "residual_rectified_flow",
            "mixture_anchored_flow",
            "mixture_anchored_residual_flow",
        }:
            self.spec_objective = "residual_flow"
        elif self.spec_objective in {
            "alphaflow", "alpha_flow", "alpha-flow", "mean_residual_flow",
            "mixture_to_target_alphaflow",
        }:
            self.spec_objective = "alphaflow"
        elif self.spec_objective in {
            "flowmap_adapter", "flow_map_adapter", "drift_flowmap",
            "drift_flow_map", "shortcut_adapter",
        }:
            # Convert a competent one-step drift separator into a one/few-step
            # flow map without allowing the drift model itself to move.
            self.spec_objective = "flowmap_adapter"
        elif self.spec_objective in {
            "visual_floss", "visual-floss", "floss", "floss_visual",
            "mixture_consistent_floss",
        }:
            # Visual-FLOSS: mixture-consistent target/complement flow matching.
            # Only the target slot is parameterised; the complementary source is
            # always M-target, so mixture consistency is exact by construction.
            self.spec_objective = "visual_floss"
        elif self.spec_objective in {"direct", "direct_unet", "vanilla", "vanilla_unet", "recon", "recon_only"}:
            self.spec_objective = "direct_unet"
        elif self.spec_objective in {"mask_drift", "masked_drift", "mask"}:
            # Bounded complex-ratio mask plus an additive residual correction:
            #     x_hat = m * mixture + r,   m in [0, mask_max]
            # `drift_only` predicts an unconstrained additive correction, which
            # has to represent cancellation of loud accompaniment as a large
            # difference of large numbers.  A mask does that multiplicatively,
            # which matches the dynamic range of a spectrogram far better and is
            # the parameterisation used by essentially all strong music/voice
            # separators.  The additive residual keeps the extra expressiveness
            # that a pure mask lacks (it cannot add energy that is not already
            # in the mixture).
            self.spec_objective = "mask_drift"
        elif self.spec_objective in {
            "meanflow", "mean_flow", "mean-flow", "noise_meanflow",
            "meanflow_noise", "standalone_meanflow",
        }:
            # Noise-anchored MeanFlow with the true JVP identity.  This is the
            # ablation of the mixture-anchored claim, NOT a variant of it: with a
            # random anchor the (eps,S) pairing is no longer deterministic given
            # z_t, so the marginal velocity genuinely differs from the
            # conditional one and the correction term does not vanish.
            self.spec_objective = "meanflow"
        else:
            raise ValueError(
                f"Unknown flow.objective={self.spec_objective!r}; "
                "use 'flow', 'residual_flow', 'alphaflow', 'meanflow', "
                "'drift_only', 'flowmap_adapter', 'visual_floss', "
                "'hybrid_flow', 'mask_drift', "
                "or 'direct_unet'"
            )
        self.mask_max = float(flow_cfg.get("mask_max", 2.0))
        self.mask_residual = bool(flow_cfg.get("mask_residual", True))
        # Offset that makes the bounded mask exactly 1.0 at initialisation.
        # A unit mask is only reachable when mask_max > 1; otherwise start as
        # close to passthrough as the bound allows.
        if self.mask_max > 1.0:
            self._mask_identity_offset = self.mask_max * math.atanh(1.0 / self.mask_max)
        else:
            self._mask_identity_offset = 4.0 * self.mask_max
        # See _separate_spec_target_only for why this matters.
        self.drift_step_mode = str(flow_cfg.get("drift_step_mode", "endpoint")).lower()
        if self.drift_step_mode not in {"endpoint", "euler"}:
            raise ValueError(
                f"flow.drift_step_mode must be 'endpoint' or 'euler', got {self.drift_step_mode!r}"
            )

        head_cfg = dict(cfg.get("head", {}))
        self.head_type = str(head_cfg.pop("type", "waveform")).lower()
        if self.head_type in {"waveform", "wavenet", "conv1d"}:
            self.head = WaveformFlowHead(cond_dim=cond_dim, **head_cfg)
            self.is_spec_head = False
        elif self.head_type in {
            "spec_unet", "spectrogram_unet", "complex_unet", "y_net", "ynet",
            "diffvs_unet", "diff_vs_unet", "diffvs",
            "tfc_tdf_unet", "tfctdf_unet", "tfc_tdf",
            "mamba_hybrid", "mambavoice_hybrid",
        }:
            # DAVIS-style target-only flow predicts only the target complex STFT
            # velocity. The residual is computed afterwards as mixture - target.
            if self.spec_target_mode == "target_only":
                head_cfg.setdefault("in_channels", 4)   # target x_t RI + mixture RI
                if self.spec_objective == "mask_drift" and self.mask_residual:
                    # mask RI + additive residual RI
                    head_cfg.setdefault("out_channels", 4)
                else:
                    head_cfg.setdefault("out_channels", 2)  # target velocity RI only
            else:
                head_cfg.setdefault("in_channels", 6)   # target/residual x_t RI + mixture RI
                head_cfg.setdefault("out_channels", 4)  # target/residual velocity RI

            if self.head_type in {"diffvs_unet", "diff_vs_unet", "diffvs"}:
                self.head = DiffVSUNetFlowHead(cond_dim=cond_dim, **head_cfg)
            elif self.head_type in {"tfc_tdf_unet", "tfctdf_unet", "tfc_tdf"}:
                self.head = TFCTDFUNetFlowHead(cond_dim=cond_dim, **head_cfg)
            elif self.head_type in {"mamba_hybrid", "mambavoice_hybrid"}:
                self.head = MambaHybridFlowHead(cond_dim=cond_dim, **head_cfg)
            else:
                self.head = SpecUNetFlowHead(cond_dim=cond_dim, **head_cfg)
            self.is_spec_head = True
        else:
            raise ValueError(f"Unknown head.type={self.head_type!r}")

        self.stft_cfg = STFTConfig.from_dict(cfg.get("stft", {}))
        self.flow_cfg = FlowConfig(
            consistency=flow_cfg.get("consistency", "every_step"),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            num_steps=int(flow_cfg.get("num_steps", 8)),
        )

        # Optional adaptive drift gate.
        # Disabled: original behavior.
        # Enabled: x_next = x + alpha * drift.
        self.adaptive_drift = bool(flow_cfg.get("adaptive_drift", False))

        # Route the true visual tokens into bottleneck cross-attention as K/V
        # for the drift objective, matching what the AlphaFlow v3 arm does.
        #
        # Without this the drift path passes only temporal_tokens, so the head
        # falls back to the FUSED tokens for attention while the AlphaFlow arm
        # receives the actual visual tokens.  A baseline trained that way is
        # handicapped on the visual pathway rather than on the objective under
        # test, and any win for the transport could be attributed to the better
        # conditioning route instead.
        #
        # Defaults to False so every pre-existing drift config -- including the
        # runs already reported -- behaves exactly as before.
        self.drift_visual_cross_attention = bool(
            flow_cfg.get("visual_cross_attention", False)
        )
        self.drift_alpha_min = float(flow_cfg.get("drift_alpha_min", 0.0))
        self.drift_alpha_max = float(flow_cfg.get("drift_alpha_max", 1.25))
        self.drift_alpha_init = float(flow_cfg.get("drift_alpha_init", 1.0))

        if self.adaptive_drift:
            denom = max(self.drift_alpha_max - self.drift_alpha_min, 1e-6)
            raw_alpha = (self.drift_alpha_init - self.drift_alpha_min) / denom
            raw_alpha = min(max(raw_alpha, 1e-4), 1.0 - 1e-4)
            alpha_bias = math.log(raw_alpha / (1.0 - raw_alpha))

            hidden = max(int(cond_dim), 32)
            self.drift_alpha_head = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.drift_alpha_head[-1].weight)
            nn.init.constant_(self.drift_alpha_head[-1].bias, alpha_bias)
        else:
            self.drift_alpha_head = None
        train_cfg = cfg.get("training", {})
        self.lambda_fm = float(train_cfg.get("lambda_fm", 1.0))
        # One-step drift objective weight. Defaults to lambda_fm so older configs
        # keep behaving sensibly if flow.objective is changed to drift_only.
        self.lambda_drift = float(train_cfg.get("lambda_drift", self.lambda_fm))
        self.lambda_recon = float(train_cfg.get("lambda_recon", 0.25))
        # Explicit endpoint-reconstruction weight for residual_flow.  Keeping
        # this separate from lambda_drift avoids accidentally carrying the
        # one-step drift objective into genuine flow-matching experiments.
        self.lambda_endpoint = float(train_cfg.get("lambda_endpoint", self.lambda_recon))
        self.lambda_consistency = float(train_cfg.get("lambda_consistency", 0.1))
        # DAVIS target-only anchoring losses. These are no-ops unless their
        # weights are > 0. They directly address the measured failure mode where
        # the target is left in the residual and the predicted target RMS collapses.
        self.lambda_target_anchor = float(train_cfg.get("lambda_target_anchor", 0.0))
        self.target_anchor_margin = float(train_cfg.get("target_anchor_margin", 1.0))
        self.lambda_rms = float(train_cfg.get("lambda_rms", 0.0))
        self.lambda_wave = float(train_cfg.get("lambda_wave", 0.0))
        self.lambda_target_gain = float(train_cfg.get("lambda_target_gain", 0.0))
        self.lambda_residual_leak = float(train_cfg.get("lambda_residual_leak", 0.0))
        self.target_gain_floor = float(train_cfg.get("target_gain_floor", 0.85))
        self.lambda_residual_recon= float(train_cfg.get("lambda_residual_recon", 0.0))
        # Complementary anti-leakage: after target allocation is fixed, the
        # predicted target may still contain accompaniment/interferer. Penalize
        # projection of predicted target onto the reference residual
        # (mixture-target), and optionally cap target RMS to avoid mixture-like
        # over-extraction.
        self.lambda_interferer_leak = float(train_cfg.get("lambda_interferer_leak", 0.0))
        self.lambda_target_energy_ceiling = float(train_cfg.get("lambda_target_energy_ceiling", 0.0))
        self.target_energy_ceiling_ratio = float(train_cfg.get("target_energy_ceiling_ratio", 1.10))
        # Multi-resolution log-magnitude STFT loss. Off by default.
        self.lambda_mrstft = float(train_cfg.get("lambda_mrstft", 0.0))
        self.lambda_visual_reliability = float(train_cfg.get("lambda_visual_reliability", 0.0))

        # Dual-head drift-anchored AlphaFlow losses. Defaults to zero (dual-head disabled).
        # Dual-head config explicitly sets these to 0.25, 0.001 to enable.
        self.lambda_base_drift = float(train_cfg.get("lambda_base_drift", 0.0))
        self.lambda_delta_reg = float(train_cfg.get("lambda_delta_reg", 0.0))

        # Frozen-drift flow-map conversion.  The exact deployment objective is
        # evaluated over the whole batch on every update; the sampled map loss
        # then supplies oracle and error-recycled states without diluting that
        # one-step anchor.
        self.lambda_flowmap = float(train_cfg.get("lambda_flowmap", 0.5))
        self.lambda_deployment = float(train_cfg.get("lambda_deployment", 1.0))
        self.lambda_composition = float(train_cfg.get("lambda_composition", 0.05))
        self.lambda_adapter_reg = float(
            train_cfg.get("lambda_adapter_reg", train_cfg.get("lambda_delta_reg", 0.0))
        )

        # Visual-FLOSS keeps the strong one-step drift operating point while
        # adding mixture-preserving noisy transport states.  The normalized-dB
        # term is intentionally separate from the deployment endpoint anchor:
        # the former teaches a field, the latter protects NFE=1 separation.
        self.lambda_floss = float(train_cfg.get("lambda_floss", 0.05))
        vf_cfg = dict(flow_cfg.get("visual_floss", {}))
        self.visual_floss_deployment_ratio = float(
            vf_cfg.get("deployment_ratio", 0.5)
        )
        self.visual_floss_noise_scale = float(vf_cfg.get("noise_scale", 0.15))
        self.visual_floss_curriculum_steps = int(
            vf_cfg.get("curriculum_steps", 8000)
        )
        self.visual_floss_time_sampling = str(
            vf_cfg.get("time_sampling", "logit_normal")
        ).lower()
        self.visual_floss_logit_mu = float(vf_cfg.get("logit_mu", -0.4))
        self.visual_floss_logit_sigma = float(vf_cfg.get("logit_sigma", 1.0))
        self.visual_floss_max_t = float(vf_cfg.get("max_t", 0.95))
        self.visual_floss_loss_eps = float(vf_cfg.get("loss_eps", 1e-7))
        self.visual_floss_db_floor = float(vf_cfg.get("db_floor", -30.0))
        self.visual_floss_db_ceiling = float(vf_cfg.get("db_ceiling", 30.0))
        if not 0.0 <= self.visual_floss_deployment_ratio <= 1.0:
            raise ValueError("flow.visual_floss.deployment_ratio must be in [0,1]")
        if self.visual_floss_noise_scale < 0.0:
            raise ValueError("flow.visual_floss.noise_scale must be >= 0")
        if self.visual_floss_curriculum_steps < 0:
            raise ValueError("flow.visual_floss.curriculum_steps must be >= 0")
        if self.visual_floss_time_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(
                "flow.visual_floss.time_sampling must be 'uniform' or 'logit_normal'"
            )
        if not 0.0 < self.visual_floss_max_t < 1.0:
            raise ValueError("flow.visual_floss.max_t must be in (0,1)")
        if self.visual_floss_loss_eps <= 0.0:
            raise ValueError("flow.visual_floss.loss_eps must be > 0")
        if self.visual_floss_db_floor >= self.visual_floss_db_ceiling:
            raise ValueError(
                "flow.visual_floss.db_floor must be smaller than db_ceiling"
            )

        # JVP-free AlphaFlow / mean-velocity configuration.  These defaults are
        # exposed rather than hidden so audio-specific ablations can be run
        # without changing code.  AlphaFlowTSE uses equal branch probability,
        # lambda_FM=0.6, lambda_MF=0.4, alpha_min=0.1, and 15% long intervals.
        af_cfg = dict(flow_cfg.get("alphaflow", {}))
        self.alphaflow_variant = str(af_cfg.get("variant", "legacy")).lower()
        self.alphaflow_fm_ratio = float(af_cfg.get("fm_ratio", 0.5))
        self.lambda_meanflow = float(train_cfg.get("lambda_meanflow", af_cfg.get("lambda_meanflow", 0.4)))
        self.alphaflow_alpha_min = float(af_cfg.get("alpha_min", 0.1))
        self.alphaflow_schedule_start_frac = float(af_cfg.get("alpha_schedule_start_frac", 0.05))
        self.alphaflow_schedule_end_frac = float(af_cfg.get("alpha_schedule_end_frac", 1.0))
        self.alphaflow_schedule_k = float(af_cfg.get("alpha_schedule_k", 15.0))
        self.alphaflow_large_span_prob = float(af_cfg.get("large_span_prob", 0.15))
        self.alphaflow_large_span_t_max = float(af_cfg.get("large_span_t_max", 0.15))
        self.alphaflow_large_span_r_min = float(af_cfg.get("large_span_r_min", 0.85))
        self.alphaflow_pair_sampling = str(af_cfg.get("pair_sampling", "logit_normal")).lower()
        self.alphaflow_logit_mu = float(af_cfg.get("logit_mu", -0.4))
        self.alphaflow_logit_sigma = float(af_cfg.get("logit_sigma", 1.0))
        self.alphaflow_adaptive_gamma = float(af_cfg.get("adaptive_gamma", 0.5))
        self.alphaflow_adaptive_eps = float(af_cfg.get("adaptive_eps", 1e-3))
        self.alphaflow_bounded_kappa = float(af_cfg.get("bounded_kappa", 1.0))
        self.alphaflow_bounded_eps = float(af_cfg.get("bounded_eps", 1e-6))
        self.alphaflow_loss_eps = float(af_cfg.get("loss_eps", 1e-3))
        self.alphaflow_deployment_interval_prob = float(af_cfg.get("deployment_interval_prob", 0.0))
        # Scope of the ground-truth endpoint anchor.
        #   "deployment"    - only exact (t,r)=(0,1) samples (v3 default).
        #   "all_intervals" - every finite-interval sample.  On the linear
        #       transport z_t=M+t(S-M) we have z_r_true = z_t + (r-t)v exactly,
        #       so the endpoint L1 reduces to span*L1(student, v_target): a
        #       ground-truth anchor that is already span-weighted, which
        #       emphasises the long spans one-NFE inference actually uses.
        self.alphaflow_endpoint_scope = str(af_cfg.get("endpoint_scope", "deployment")).lower()
        if not 0.0 <= self.alphaflow_fm_ratio <= 1.0:
            raise ValueError("flow.alphaflow.fm_ratio must be in [0,1]")
        if self.alphaflow_loss_eps <= 0.0:
            raise ValueError("flow.alphaflow.loss_eps must be > 0")
        if not 0.0 <= self.alphaflow_deployment_interval_prob <= 1.0:
            raise ValueError("flow.alphaflow.deployment_interval_prob must be in [0,1]")
        if self.alphaflow_endpoint_scope not in {"deployment", "all_intervals"}:
            raise ValueError(
                "flow.alphaflow.endpoint_scope must be 'deployment' or 'all_intervals', "
                f"got {self.alphaflow_endpoint_scope!r}"
            )

        flowmap_cfg = dict(flow_cfg.get("flowmap", {}))
        self.flowmap_warmup_steps = int(flowmap_cfg.get("warmup_steps", 8000))
        self.flowmap_deployment_prob = float(flowmap_cfg.get("deployment_prob", 0.5))
        self.flowmap_oracle_prob = float(flowmap_cfg.get("oracle_prob", 0.25))
        self.flowmap_on_policy_prob = float(flowmap_cfg.get("on_policy_prob", 0.25))
        self.flowmap_t_max = float(flowmap_cfg.get("t_max", 0.8))
        self.flowmap_min_span = float(flowmap_cfg.get("min_span", 0.05))
        self.flowmap_composition_start_step = int(
            flowmap_cfg.get("composition_start_step", 40000)
        )
        self.flowmap_composition_prob = float(
            flowmap_cfg.get("composition_prob", 0.25)
        )
        flowmap_prob_sum = (
            self.flowmap_deployment_prob
            + self.flowmap_oracle_prob
            + self.flowmap_on_policy_prob
        )
        if abs(flowmap_prob_sum - 1.0) > 1e-6:
            raise ValueError(
                "flow.flowmap deployment_prob + oracle_prob + on_policy_prob "
                f"must equal 1, got {flowmap_prob_sum:g}"
            )
        if self.flowmap_warmup_steps < 0:
            raise ValueError("flow.flowmap.warmup_steps must be >= 0")
        if not 0.0 < self.flowmap_t_max < 1.0:
            raise ValueError("flow.flowmap.t_max must be in (0,1)")
        if not 0.0 <= self.flowmap_min_span < 1.0:
            raise ValueError("flow.flowmap.min_span must be in [0,1)")
        if not 0.0 <= self.flowmap_composition_prob <= 1.0:
            raise ValueError("flow.flowmap.composition_prob must be in [0,1]")
        # ---- Standalone noise-anchored MeanFlow -------------------------------
        #
        # The (t,r) sampler, the branch split and the adaptive loss deliberately
        # reuse the AlphaFlow code paths (flow.alphaflow.*), so the noise-anchored
        # and mixture-anchored arms differ in exactly two things: where the path
        # starts, and whether the target carries a correction term.  Anything
        # this block adds is specific to the correction itself.
        mf_cfg = dict(flow_cfg.get("meanflow", {}))
        self.meanflow_correction = str(mf_cfg.get("correction", "jvp")).lower()
        self.meanflow_jvp_fp32 = bool(mf_cfg.get("jvp_fp32", True))
        if self.meanflow_correction not in {"jvp", "none"}:
            raise ValueError(
                "flow.meanflow.correction must be 'jvp' (the real MeanFlow "
                "identity) or 'none' (average-velocity regression against the "
                "conditional velocity, which is the control showing the "
                f"correction is load-bearing), got {self.meanflow_correction!r}"
            )
        if self.spec_objective == "meanflow" and self.spec_init_mode not in {
            "noise", "gaussian", "random"
        }:
            raise ValueError(
                "flow.objective=meanflow anchors the path at Gaussian noise, so "
                "flow.init_mode must be 'noise'; training draws eps with "
                "flow.noise_scale and inference must start from the same "
                f"distribution, got init_mode={self.spec_init_mode!r}"
            )

        # ---- Frozen first-stage prior (predictor-anchored flow) --------------
        #
        # The FlowAVSE / AVDiffuSS pattern: a trained direct separator supplies
        # the anchor, and the transport starts from ITS estimate P instead of
        # from the mixture M.
        #
        #     z_t = P + t (S - P),        S_hat = P + u_theta(P, 0, 1, M, V)
        #
        # P is a deterministic function of (M, V), so the pairing is still fixed
        # by the conditioning and the interval-averaged velocity is the constant
        # S - P in closed form.  The MeanFlow correction cancels exactly as it
        # does on the mixture path, which is why only the alpha=1 AlphaFlow
        # objective and the one-shot drift refiner are accepted here.
        #
        # Frozen on purpose: the flow's improvement over its own predictor is
        # then directly measurable -- the ablation FlowAVSE never reports.
        self.prior = None
        self.prior_checkpoint: Optional[str] = None
        self.prior_condition_on = "mixture"
        self.prior_anchor_noise_std = 0.0
        prior_cfg = flow_cfg.get("prior", None)
        if prior_cfg:
            prior_cfg = dict(prior_cfg)
            checkpoint = prior_cfg.get("checkpoint", None)
            if not checkpoint or "REPLACE" in str(checkpoint):
                raise ValueError(
                    "flow.prior.checkpoint is unset. Point it at the trained "
                    "predictor's best.pt (for example the drift_matched run) "
                    "before launching."
                )
            self.prior_condition_on = str(prior_cfg.get("condition_on", "mixture")).lower()
            if self.prior_condition_on not in {"mixture", "anchor"}:
                raise ValueError(
                    "flow.prior.condition_on must be 'mixture' (the real mixture in "
                    "the head's conditioning slot) or 'anchor' (the transport state "
                    f"there, as the AlphaFlow arms do), got {self.prior_condition_on!r}"
                )
            self.prior_anchor_noise_std = float(prior_cfg.get("anchor_noise_std", 0.0))
            if self.prior_anchor_noise_std < 0.0:
                raise ValueError("flow.prior.anchor_noise_std must be >= 0")
            is_v3 = self.alphaflow_variant in {
                "v3", "deployment_v3", "faithful_v3", "av_v3", "alphaflow_v3"
            }
            if not (
                self.spec_objective == "drift_only"
                or (self.spec_objective == "alphaflow" and is_v3)
            ):
                raise ValueError(
                    "flow.prior supports objective=alphaflow with variant faithful_v3, "
                    "or objective=drift_only (the matched one-shot refiner); got "
                    f"objective={self.spec_objective!r}"
                )
            if self.spec_objective == "drift_only" and self.prior_condition_on != "mixture":
                raise ValueError(
                    "flow.prior with objective=drift_only always conditions on the "
                    "mixture; set flow.prior.condition_on to 'mixture'"
                )
            self.prior_checkpoint = str(checkpoint)
            self.prior = self._load_frozen_prior(
                self.prior_checkpoint, use_ema=bool(prior_cfg.get("use_ema", True))
            )

        self._training_progress_step = 0
        self._training_progress_total = 1

        self._warn_duplicate_losses()

        if self.spec_objective == "flowmap_adapter":
            if self.spec_init_mode not in {"mixture", "mixture_stft"}:
                raise ValueError(
                    "flowmap_adapter must start from the observed mixture"
                )
            if not isinstance(self.head, TFCTDFUNetFlowHead):
                raise ValueError(
                    "flowmap_adapter currently requires head.type=tfc_tdf_unet"
                )
            if not self.head.has_flowmap_adapter:
                raise ValueError(
                    "flowmap_adapter requires head.flowmap_adapter_blocks >= 1"
                )
            if self.head.interval_embedding_reference != 1.0:
                raise ValueError(
                    "flowmap_adapter requires head.interval_embedding_reference=1.0 "
                    "so its initial (t,r)=(0,1) output exactly matches drift"
                )
            self._freeze_for_flowmap_adapter()
        if self.spec_objective == "visual_floss":
            if self.spec_target_mode != "target_only":
                raise ValueError("visual_floss requires flow.target_mode=target_only")
            if self.spec_init_mode not in {"mixture", "mixture_stft"}:
                raise ValueError(
                    "visual_floss deployment must start from the observed mixture"
                )
            if self.adaptive_drift:
                raise ValueError(
                    "visual_floss requires flow.adaptive_drift=false so the warm-start "
                    "exactly reproduces the drift checkpoint"
                )

    def _warn_duplicate_losses(self) -> None:
        """Flag loss weights that are algebraically the same term.

        Several of the configured losses are provably identical, so their
        weights simply add.  Reporting them as separate objectives (in configs
        or in a paper's loss table) overstates how many constraints are active.
        """
        import warnings

        if self.lambda_wave > 0.0 and self.lambda_residual_recon > 0.0:
            warnings.warn(
                "lambda_wave and lambda_residual_recon are the same loss: "
                "|(mixture - pred) - (mixture - target)| == |pred - target|. "
                f"Effective waveform-L1 weight is {self.lambda_wave + self.lambda_residual_recon:g}.",
                stacklevel=2,
            )
        if self.spec_objective in {"drift_only", "mask_drift"} and (
            self.lambda_drift > 0.0 and self.lambda_recon > 0.0
        ):
            warnings.warn(
                "lambda_drift and lambda_recon are the same loss for this objective: "
                "|drift - (y - z)| == |(z + drift) - y|. "
                f"Effective spectral-L1 weight is {self.lambda_drift + self.lambda_recon:g}.",
                stacklevel=2,
            )

    def set_training_progress(self, step: int, total_steps: Optional[int]) -> None:
        """Expose optimizer progress to schedules without coupling the model to Trainer."""
        self._training_progress_step = max(0, int(step))
        self._training_progress_total = max(1, int(total_steps or 1))

    def _alphaflow_alpha(self) -> float:
        """Sigmoid curriculum from 1.0 to alpha_min over configured training fractions."""
        frac = self._training_progress_step / max(self._training_progress_total, 1)
        start = self.alphaflow_schedule_start_frac
        end = max(self.alphaflow_schedule_end_frac, start + 1e-8)
        if frac <= start:
            return 1.0
        if frac >= end:
            return self.alphaflow_alpha_min
        p = (frac - start) / (end - start)
        k = self.alphaflow_schedule_k
        # Normalize a logistic curve so p=0 -> 0 and p=1 -> 1 exactly.
        lo = 1.0 / (1.0 + math.exp(k * 0.5))
        hi = 1.0 / (1.0 + math.exp(-k * 0.5))
        cur = 1.0 / (1.0 + math.exp(-k * (p - 0.5)))
        q = (cur - lo) / max(hi - lo, 1e-8)
        return 1.0 - (1.0 - self.alphaflow_alpha_min) * q

    @staticmethod
    def _per_sample_mse(x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).flatten(1).mean(dim=1)

    def _alphaflow_adaptive_loss(self, residual: torch.Tensor) -> torch.Tensor:
        m = self._per_sample_mse(residual)
        weight = (m + self.alphaflow_adaptive_eps).pow(self.alphaflow_adaptive_gamma - 1.0).detach()
        return (weight * m).mean()

    def _alphaflow_bounded_loss(self, residual: torch.Tensor, alpha: float) -> torch.Tensor:
        m = self._per_sample_mse(residual)
        kappa = self.alphaflow_bounded_kappa
        weight = (kappa / (m + float(alpha) * kappa + self.alphaflow_bounded_eps)).detach()
        return (weight * m).mean()

    def _sample_alphaflow_interval(self, batch_size: int, device, dtype):
        """Sample 0<=t<r<=1, with an explicit long-span component for NFE=1."""
        large = torch.rand(batch_size, device=device) < self.alphaflow_large_span_prob
        if self.alphaflow_pair_sampling in {"logit_normal", "logit-normal", "ln"}:
            a = torch.sigmoid(
                self.alphaflow_logit_mu
                + self.alphaflow_logit_sigma * torch.randn(batch_size, device=device, dtype=dtype)
            )
            b = torch.sigmoid(
                self.alphaflow_logit_mu
                + self.alphaflow_logit_sigma * torch.randn(batch_size, device=device, dtype=dtype)
            )
        else:
            a = torch.rand(batch_size, device=device, dtype=dtype)
            b = torch.rand(batch_size, device=device, dtype=dtype)
        t = torch.minimum(a, b)
        r = torch.maximum(a, b)
        # Prevent exactly-zero spans from numerical ties.
        r = torch.maximum(r, (t + 1e-4).clamp_max(1.0))
        if large.any():
            n = int(large.sum().item())
            t_large = self.alphaflow_large_span_t_max * torch.rand(n, device=device, dtype=dtype)
            r_large = self.alphaflow_large_span_r_min + (1.0 - self.alphaflow_large_span_r_min) * torch.rand(n, device=device, dtype=dtype)
            t = t.clone(); r = r.clone()
            t[large] = t_large
            r[large] = torch.maximum(r_large, t_large + 1e-4)
        return t.clamp(0.0, 1.0), r.clamp(0.0, 1.0)

    def _sample_alphaflow_interval_v2(self, batch_size: int, device, dtype):
        """Sample finite intervals and return which samples are near-full-span."""
        if batch_size == 0:
            empty = torch.empty(0, device=device, dtype=dtype)
            return empty, empty, torch.empty(0, device=device, dtype=torch.bool)

        large = torch.rand(batch_size, device=device) < self.alphaflow_large_span_prob
        if self.alphaflow_pair_sampling in {"logit_normal", "logit-normal", "ln"}:
            a = torch.sigmoid(
                self.alphaflow_logit_mu
                + self.alphaflow_logit_sigma
                * torch.randn(batch_size, device=device, dtype=dtype)
            )
            b = torch.sigmoid(
                self.alphaflow_logit_mu
                + self.alphaflow_logit_sigma
                * torch.randn(batch_size, device=device, dtype=dtype)
            )
        else:
            a = torch.rand(batch_size, device=device, dtype=dtype)
            b = torch.rand(batch_size, device=device, dtype=dtype)

        t = torch.minimum(a, b)
        r = torch.maximum(a, b)
        r = torch.maximum(r, (t + 1e-4).clamp_max(1.0))
        if large.any():
            n = int(large.sum().item())
            t_large = self.alphaflow_large_span_t_max * torch.rand(
                n, device=device, dtype=dtype
            )
            r_large = self.alphaflow_large_span_r_min + (
                1.0 - self.alphaflow_large_span_r_min
            ) * torch.rand(n, device=device, dtype=dtype)
            t = t.clone()
            r = r.clone()
            t[large] = t_large
            r[large] = torch.maximum(r_large, t_large + 1e-4)
        return t.clamp(0.0, 1.0), r.clamp(0.0, 1.0), large

    def _sample_alphaflow_interval_v3(self, batch_size: int, device, dtype):
        """Sample finite intervals with an explicit exact deployment component.

        Deployment samples are exactly (t,r)=(0,1), matching one-NFE inference.
        They are drawn *within* the finite-interval branch so diagonal FM coverage
        remains unchanged.
        """
        t, r, large = self._sample_alphaflow_interval_v2(batch_size, device, dtype)
        if batch_size == 0:
            return t, r, large, torch.empty(0, device=device, dtype=torch.bool)
        deploy = torch.rand(batch_size, device=device) < self.alphaflow_deployment_interval_prob
        if deploy.any():
            t = t.clone()
            r = r.clone()
            t[deploy] = 0.0
            r[deploy] = 1.0
        return t, r, large, deploy

    def _alphaflow_v2_fm_count(self, batch_size: int, device) -> int:
        """Choose a low-variance sample count with unbiased stochastic rounding."""
        expected = float(batch_size) * self.alphaflow_fm_ratio
        count = int(math.floor(expected))
        fractional = expected - count
        if fractional > 0.0 and bool(torch.rand((), device=device) < fractional):
            count += 1
        return max(0, min(batch_size, count))

    def _alphaflow_v2_weighted_loss(
        self,
        per_sample_mse: torch.Tensor,
        interval_mask: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample losses and detached AlphaFlow weights."""
        numerator = torch.ones_like(per_sample_mse)
        numerator[interval_mask] = float(alpha)
        weight = numerator / (
            per_sample_mse.detach() + self.alphaflow_loss_eps
        )
        return weight * per_sample_mse, weight

    @staticmethod
    def unpack_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        mixture = batch.get("mixture")
        target = batch.get("target")
        face = batch.get("face")
        if face is None:
            face = batch.get("visuals")
        body = batch.get("body")
        if mixture is None or target is None:
            raise KeyError("Batch must contain 'mixture' and 'target' tensors")
        return {"mixture": mixture, "target": target, "face": face, "body": body}

    def condition(self, mixture: torch.Tensor, face: Optional[torch.Tensor], body: Optional[torch.Tensor]) -> Dict[str, Any]:
        captured = self.conditioner(mixture, face=face, body=body)
        # MambaVoice uses LazyLinear projections.  They cannot be frozen until
        # their first real batch has materialized the parameters, so finish the
        # adapter-only freeze immediately after that forward pass.
        if (
            getattr(self, "spec_objective", None) == "flowmap_adapter"
            and getattr(self, "_flowmap_freeze_pending", False)
        ):
            self._freeze_for_flowmap_adapter()
        return captured

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, mixture: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Waveform-domain velocity for the legacy 1-D head."""
        v = self.head(x_t, mixture, cond, t)
        if self.flow_cfg.consistency in {"final", "every_step"}:
            v = project_velocity_zero_sum(v)
        return v

    def spec_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mixture_ri: torch.Tensor,
        cond: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor] = None,
        visual_activity: Optional[torch.Tensor] = None,
        cross_attention_tokens: Optional[torch.Tensor] = None,
        interval_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Complex-STFT velocity / finite-interval mean velocity for 2-D heads."""
        extra = {}
        if interval_end is not None:
            if not isinstance(self.head, TFCTDFUNetFlowHead):
                raise NotImplementedError(
                    "AlphaFlow interval conditioning is currently implemented for "
                    "head.type=tfc_tdf_unet only"
                )
            extra["interval_end"] = interval_end
        if cross_attention_tokens is not None:
            if isinstance(self.head, (TFCTDFUNetFlowHead, MambaHybridFlowHead)):
                extra["cross_attention_tokens"] = cross_attention_tokens
        v = self.head(
            x_t,
            mixture_ri,
            cond,
            t,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
            **extra,
        )
        # Source-pair flow needs zero-sum velocity so target/residual stay
        # mixture-consistent.  DAVIS-style target-only flow should NOT do this,
        # because it predicts only one source and the residual is mixture-target.
        if self.spec_target_mode == "source_pair" and self.flow_cfg.consistency in {"final", "every_step"}:
            v = project_ri_velocity_zero_sum(v, num_sources=2)
        return v

    def _target_only_initial_state(
        self,
        mixture_ri: torch.Tensor,
        *,
        noise_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Initial state z for DAVIS-style target-only flow.

        The default/recommended setting for the user's current experiments is
        flow.init_mode='half_mixture', i.e. z = 0.5 * mixture_STFT.  This gives
        a deterministic, mixture-grounded starting point and removes the
        target/residual channel-swap ambiguity of source-pair flow.
        """
        mode = self.spec_init_mode
        if mode in {"half_mixture", "0.5_mixture", "half", "mixture_half"}:
            z = 0.5 * mixture_ri
        elif mode in {"mixture", "mixture_stft"}:
            z = mixture_ri.clone()
        elif mode in {"zero", "zeros"}:
            z = torch.zeros_like(mixture_ri)
        elif mode in {"noise", "gaussian", "random"}:
            scale = self.flow_cfg.noise_scale if noise_scale is None else float(noise_scale)
            z = torch.randn_like(mixture_ri) * scale
        else:
            raise ValueError(f"Unknown flow.init_mode={mode!r}")

        extra_noise = self.spec_init_noise_scale if noise_scale is None else float(noise_scale)
        if extra_noise > 0 and mode not in {"noise", "gaussian", "random"}:
            z = z + torch.randn_like(z) * extra_noise
        return z

    def _prep_waveforms(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        b = self.unpack_batch(batch)
        mixture = b["mixture"].float()
        target = b["target"].float()
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        l = min(mixture.shape[-1], target.shape[-1])
        mixture = mixture[..., :l]
        target = target[..., :l]
        return mixture, target, b

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.is_spec_head:
            return self._training_loss_spec(batch)
        return self._training_loss_waveform(batch)

    def _training_loss_waveform(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mixture, target, b = self._prep_waveforms(batch)
        residual = mixture - target
        y = torch.stack([target, residual], dim=1)
        if self.flow_cfg.consistency in {"final", "every_step"}:
            y = project_sources_to_mixture(y, mixture)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        x_t, t, v_target = sample_training_tuple(y, mixture, noise_scale=self.flow_cfg.noise_scale)
        v_pred = self.velocity(x_t, t, mixture, cond)
        loss_fm = F.l1_loss(v_pred, v_target)
        shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
        x1_hat = x_t + (1.0 - t.view(shape)) * v_pred
        if self.flow_cfg.consistency in {"final", "every_step"}:
            x1_hat = project_sources_to_mixture(x1_hat, mixture)
        # Train on the full target/residual pair by default; this is better aligned
        # with mixture-consistent source-pair flow than target-only L1.
        loss_recon = F.l1_loss(x1_hat, y)
        loss_cons = (x1_hat.sum(dim=1) - mixture).abs().mean()
        loss = self.lambda_fm * loss_fm + self.lambda_recon * loss_recon + self.lambda_consistency * loss_cons
        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_consistency": loss_cons.detach(),
        }

    def _training_loss_spec(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.spec_target_mode == "target_only":
            return self._training_loss_spec_target_only(batch)

        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_tokens = captured.get("video_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)  # [B,F,T] complex
        target_spec = stft_waveform(target, self.stft_cfg)
        residual_spec = mix_spec - target_spec
        y_complex = torch.stack([target_spec, residual_spec], dim=1)  # [B,2,F,T]
        y = sources_complex_to_ri(y_complex)  # [B,4,F,T]
        mixture_ri = complex_to_ri(mix_spec)  # [B,2,F,T]
        if self.flow_cfg.consistency in {"final", "every_step"}:
            y = project_ri_sources_to_mixture(y, mixture_ri, num_sources=2)

        z = torch.randn_like(y) * self.flow_cfg.noise_scale
        if self.flow_cfg.consistency in {"final", "every_step"}:
            z = project_ri_sources_to_mixture(z, mixture_ri, num_sources=2)
        batch_size = y.shape[0]
        t = torch.rand(batch_size, device=y.device, dtype=y.dtype)
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        x_t = (1.0 - t.view(shape)) * z + t.view(shape) * y
        v_target = y - z
        if self.flow_cfg.consistency in {"final", "every_step"}:
            v_target = project_ri_velocity_zero_sum(v_target, num_sources=2)

        v_pred = self.spec_velocity(
            x_t,
            t,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
        )
        loss_fm = F.l1_loss(v_pred, v_target)
        x1_hat = x_t + (1.0 - t.view(shape)) * v_pred
        if self.flow_cfg.consistency in {"final", "every_step"}:
            x1_hat = project_ri_sources_to_mixture(x1_hat, mixture_ri, num_sources=2)
        loss_recon = F.l1_loss(x1_hat, y)
        mix_hat = ri_to_sources_complex(x1_hat, num_sources=2).sum(dim=1)
        loss_cons = (mix_hat - mix_spec).abs().mean()
        loss = self.lambda_fm * loss_fm + self.lambda_recon * loss_recon + self.lambda_consistency * loss_cons
        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_consistency": loss_cons.detach(),
        }

    def _training_loss_spec_target_only_direct(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Direct/vanilla target separator.

        This removes the flow/drift parameterization. The TFC-TDF U-Net
        directly predicts the target complex STFT from mixture conditioning:

            pred_y = D_theta(0, t=0, mixture, visual)

        This is the direct U-Net baseline against the one-step drift model.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)       # [B,2,F,T]
        mixture_ri = complex_to_ri(mix_spec) # [B,2,F,T]

        x_in = torch.zeros_like(mixture_ri)
        batch_size = y.shape[0]
        t_zero = torch.zeros(batch_size, device=y.device, dtype=y.dtype)

        x1_hat = self.spec_velocity(
            x_in,
            t_zero,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
        )

        loss_drift = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_recon = F.l1_loss(x1_hat, y)
        loss_fm = loss_drift  # zero for direct_unet; kept for logging compatibility
        loss_endpoint_boundary = loss_recon

        # Target-only mode defines the residual deterministically.
        loss_cons = torch.zeros((), device=y.device, dtype=y.dtype)

        loss_target_anchor = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_rms = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_wave = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_gain = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_interferer_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_energy_ceiling = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_recon = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_mrstft = torch.zeros((), device=y.device, dtype=y.dtype)

        need_wave_endpoint = (
            self.lambda_target_anchor > 0.0
            or self.lambda_rms > 0.0
            or self.lambda_wave > 0.0
            or self.lambda_target_gain > 0.0
            or self.lambda_residual_leak > 0.0
            or self.lambda_interferer_leak > 0.0
            or self.lambda_target_energy_ceiling > 0.0
            or self.lambda_residual_recon > 0.0
            or self.lambda_mrstft > 0.0
        )
        if need_wave_endpoint:
            pred_target_complex = ri_to_sources_complex(x1_hat, num_sources=1)[:, 0]
            pred_target_wave = istft_waveform(pred_target_complex, self.stft_cfg, length=target.shape[-1])
            pred_residual_wave = mixture - pred_target_wave

            if self.lambda_mrstft > 0.0:
                loss_mrstft = multi_resolution_stft_loss(pred_target_wave, target)

            if self.lambda_target_gain > 0.0 or self.lambda_residual_leak > 0.0:
                target_energy = target.pow(2).sum(dim=-1) + 1e-8
                target_gain = (pred_target_wave * target).sum(dim=-1) / target_energy
                residual_gain = (pred_residual_wave * target).sum(dim=-1) / target_energy
                loss_target_gain = torch.relu(self.target_gain_floor - target_gain).pow(2).mean()
                loss_residual_leak = residual_gain.abs().mean()

            if self.lambda_residual_recon > 0.0:
                ref_residual_wave = mixture - target
                loss_residual_recon = F.l1_loss(pred_residual_wave, ref_residual_wave)

            if self.lambda_interferer_leak > 0.0:
                ref_residual = mixture - target
                ref_residual_energy = ref_residual.pow(2).sum(dim=-1) + 1e-8
                interferer_gain = (pred_target_wave * ref_residual).sum(dim=-1) / ref_residual_energy
                loss_interferer_leak = interferer_gain.abs().mean()

            if self.lambda_target_energy_ceiling > 0.0:
                pred_rms_for_ceiling = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms_for_ceiling = target.pow(2).mean(dim=-1).sqrt()
                loss_target_energy_ceiling = torch.relu(
                    pred_rms_for_ceiling - self.target_energy_ceiling_ratio * target_rms_for_ceiling
                ).mean()

            if self.lambda_target_anchor > 0.0:
                sisdr_t = si_sdr_score(pred_target_wave, target)
                sisdr_r = si_sdr_score(pred_residual_wave, target)
                loss_target_anchor = torch.relu(
                    sisdr_r - sisdr_t + self.target_anchor_margin
                ).mean()

            if self.lambda_rms > 0.0:
                pred_rms = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms = target.pow(2).mean(dim=-1).sqrt()
                loss_rms = (pred_rms - target_rms).abs().mean()

            if self.lambda_wave > 0.0:
                loss_wave = F.l1_loss(pred_target_wave, target)

        loss = (
            0.0 * loss_drift
            + self.lambda_recon * loss_recon
            + self.lambda_consistency * loss_cons
            + self.lambda_target_anchor * loss_target_anchor
            + self.lambda_rms * loss_rms
            + self.lambda_wave * loss_wave
            + self.lambda_target_gain * loss_target_gain
            + self.lambda_residual_leak * loss_residual_leak
            + self.lambda_interferer_leak * loss_interferer_leak
            + self.lambda_target_energy_ceiling * loss_target_energy_ceiling
            + self.lambda_residual_recon * loss_residual_recon
            + self.lambda_mrstft * loss_mrstft
        )
        return {
            "loss": loss,
            "loss_drift": loss_drift.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_endpoint_boundary": loss_endpoint_boundary.detach(),
            "loss_consistency": loss_cons.detach(),
            "loss_target_anchor": loss_target_anchor.detach(),
            "loss_rms": loss_rms.detach(),
            "loss_wave": loss_wave.detach(),
            "loss_target_gain": loss_target_gain.detach(),
            "loss_residual_leak": loss_residual_leak.detach(),
            "loss_interferer_leak": loss_interferer_leak.detach(),
            "loss_target_energy_ceiling": loss_target_energy_ceiling.detach(),
            "loss_residual_reconstruction": loss_residual_recon.detach(),
            "loss_mrstft": loss_mrstft.detach(),
        }

    def _apply_mask(
        self,
        head_out: torch.Tensor,
        mixture_ri: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a raw head output into a masked target estimate.

        head_out: [B,4,F,T] (mask RI + residual RI) or [B,2,F,T] (mask only).
        Returns (x_hat_ri, mask_magnitude) where x_hat = m * mixture + r.
        """
        eps = 1e-8
        if self.mask_residual:
            mask_ri, resid_ri = head_out[:, :2], head_out[:, 2:4]
        else:
            mask_ri, resid_ri = head_out[:, :2], None

        # The head's final conv is initialised near zero, so mask_ri starts at
        # ~0.  Offsetting the real part makes the model an identity
        # (x_hat = mixture) at initialisation, matching the drift model's
        # starting point instead of starting from silence.
        #
        # The offset is pre-compensated for the tanh bound below: we need
        # mask_max * tanh(c / mask_max) == 1, i.e. c = mask_max * atanh(1/mask_max).
        # Using a plain +1.0 would leave the initial mask at
        # mask_max * tanh(1/mask_max) (0.924 for mask_max=2), a silent 8%
        # attenuation of the passthrough.
        m_re = mask_ri[:, 0] + self._mask_identity_offset
        m_im = mask_ri[:, 1]
        mag = torch.sqrt(m_re.pow(2) + m_im.pow(2) + eps)
        # Smooth magnitude bound: ~identity for small magnitudes, saturating at
        # mask_max.  Phase is preserved, so this stays a complex ratio mask.
        bounded = self.mask_max * torch.tanh(mag / self.mask_max)
        gain = bounded / (mag + eps)
        m_re, m_im = m_re * gain, m_im * gain

        x_re, x_im = mixture_ri[:, 0], mixture_ri[:, 1]
        # Complex multiply m * mixture.
        out_re = m_re * x_re - m_im * x_im
        out_im = m_re * x_im + m_im * x_re
        x_hat = torch.stack([out_re, out_im], dim=1)
        if resid_ri is not None:
            x_hat = x_hat + resid_ri
        return x_hat, bounded

    def _apply_adaptive_drift(
        self,
        drift: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Optionally scale the predicted drift by a learned alpha gate."""
        if not bool(getattr(self, "adaptive_drift", False)) or self.drift_alpha_head is None:
            return drift, None

        if cond is None:
            alpha = drift.new_full((drift.shape[0], 1, 1, 1), float(self.drift_alpha_init))
            return drift * alpha, alpha

        cond_feat = cond
        if cond_feat.ndim == 1:
            cond_feat = cond_feat.unsqueeze(0)

        # Accept [B,D] or token-like [B,T,D] / [B,...,D].
        while cond_feat.ndim > 2:
            cond_feat = cond_feat.mean(dim=1)

        alpha_raw = torch.sigmoid(self.drift_alpha_head(cond_feat.float()))
        alpha = self.drift_alpha_min + (self.drift_alpha_max - self.drift_alpha_min) * alpha_raw
        alpha = alpha.to(dtype=drift.dtype, device=drift.device).view(drift.shape[0], 1, 1, 1)
        return drift * alpha, alpha

    def _training_loss_spec_target_only_drift(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """One-step drift-only target separator.

        This experimental objective removes the sampled flow trajectory and
        trains the SpecUNet to predict the complete correction field from the
        configured initial state z to the clean target y:

            D_target = y - z
            pred_y   = z + D_theta(z, t=0, mixture, visual)

        It is still a separation model: z is built from the mixture/noisy
        mixture, and the conditioning remains mixture + visual cue.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)
        # Only consumed when flow.visual_cross_attention /
        # lambda_visual_reliability are enabled; see __init__ for why both
        # default off.
        visual_tokens = captured.get("video_tokens", None)
        reliability_target = captured.get("visual_reliability_target", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)       # [B,2,F,T]
        mixture_ri = complex_to_ri(mix_spec) # [B,2,F,T]

        # Sample z_0 early for noise initialization consistency.
        z = self._target_only_initial_state(mixture_ri)
        if self.prior is not None:
            # Stage-2 direct refiner: predict S - P in one shot from a frozen
            # prior's estimate.  The matched control for the prior-anchored
            # flow -- same prior, same conditioning, one-shot objective.
            z = self._flow_anchor(mixture, mixture_ri, b["face"], b["body"], init_z=z)
        batch_size = y.shape[0]
        t_zero = torch.zeros(batch_size, device=y.device, dtype=y.dtype)

        head_out = self.spec_velocity(
            z,
            t_zero,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
            cross_attention_tokens=(
                visual_tokens if self.drift_visual_cross_attention else None
            ),
        )

        if self.spec_objective == "mask_drift":
            x1_hat, mask_mag = self._apply_mask(head_out, mixture_ri)
            drift_alpha = mask_mag.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        else:
            drift_pred, drift_alpha = self._apply_adaptive_drift(head_out, cond)
            x1_hat = z + drift_pred

        # NOTE: for the additive drift objective, F.l1_loss(drift_pred, y - z)
        # and F.l1_loss(z + drift_pred, y) are the *same number* elementwise, so
        # lambda_drift and lambda_recon are not two objectives -- they add up to
        # a single weight on one term.  A single endpoint L1 is used here and
        # both weights are applied to it, which is exactly what the previous code
        # computed but without implying two independent losses.
        loss_recon = F.l1_loss(x1_hat, y)
        loss_drift = loss_recon
        loss_fm = loss_drift  # backward-compatible logging name
        loss_endpoint_boundary = loss_recon
        loss_drift_alpha = (
            drift_alpha.mean()
            if drift_alpha is not None
            else y.new_tensor(1.0)
        )

        # Target-only mode defines the residual deterministically.
        loss_cons = torch.zeros((), device=y.device, dtype=y.dtype)

        loss_target_anchor = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_rms = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_wave = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_gain = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_interferer_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_energy_ceiling = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_recon = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_mrstft = torch.zeros((), device=y.device, dtype=y.dtype)

        need_wave_endpoint = (
            self.lambda_target_anchor > 0.0
            or self.lambda_rms > 0.0
            or self.lambda_wave > 0.0
            or self.lambda_target_gain > 0.0
            or self.lambda_residual_leak > 0.0
            or self.lambda_interferer_leak > 0.0
            or self.lambda_target_energy_ceiling > 0.0
            or self.lambda_residual_recon > 0.0
            or self.lambda_mrstft > 0.0
        )
        if need_wave_endpoint:
            pred_target_complex = ri_to_sources_complex(x1_hat, num_sources=1)[:, 0]
            pred_target_wave = istft_waveform(pred_target_complex, self.stft_cfg, length=target.shape[-1])
            pred_residual_wave = mixture - pred_target_wave

            if self.lambda_mrstft > 0.0:
                loss_mrstft = multi_resolution_stft_loss(pred_target_wave, target)

            if self.lambda_target_gain > 0.0 or self.lambda_residual_leak > 0.0:
                target_energy = target.pow(2).sum(dim=-1) + 1e-8
                target_gain = (pred_target_wave * target).sum(dim=-1) / target_energy
                residual_gain = (pred_residual_wave * target).sum(dim=-1) / target_energy
                loss_target_gain = torch.relu(self.target_gain_floor - target_gain).pow(2).mean()
                loss_residual_leak = residual_gain.abs().mean()

            if self.lambda_residual_recon > 0.0:
                ref_residual_wave = mixture - target
                loss_residual_recon = F.l1_loss(pred_residual_wave, ref_residual_wave)

            if self.lambda_interferer_leak > 0.0:
                ref_residual = mixture - target
                ref_residual_energy = ref_residual.pow(2).sum(dim=-1) + 1e-8
                interferer_gain = (pred_target_wave * ref_residual).sum(dim=-1) / ref_residual_energy
                loss_interferer_leak = interferer_gain.abs().mean()

            if self.lambda_target_energy_ceiling > 0.0:
                pred_rms_for_ceiling = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms_for_ceiling = target.pow(2).mean(dim=-1).sqrt()
                loss_target_energy_ceiling = torch.relu(
                    pred_rms_for_ceiling - self.target_energy_ceiling_ratio * target_rms_for_ceiling
                ).mean()

            if self.lambda_target_anchor > 0.0:
                sisdr_t = si_sdr_score(pred_target_wave, target)
                sisdr_r = si_sdr_score(pred_residual_wave, target)
                loss_target_anchor = torch.relu(
                    sisdr_r - sisdr_t + self.target_anchor_margin
                ).mean()

            if self.lambda_rms > 0.0:
                pred_rms = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms = target.pow(2).mean(dim=-1).sqrt()
                loss_rms = (pred_rms - target_rms).abs().mean()

            if self.lambda_wave > 0.0:
                loss_wave = F.l1_loss(pred_target_wave, target)

        # Visual-reliability supervision.
        #
        # The AlphaFlow arm supervises the conditioner's reliability head and
        # the drift objective historically did not, so a baseline built to
        # compare against it was handicapped on the visual pathway rather than
        # on the objective under test.  Gated on lambda_visual_reliability,
        # which defaults to 0.0, so pre-existing drift configs are unaffected.
        loss_visual_reliability = torch.zeros((), device=y.device, dtype=y.dtype)
        if (
            self.lambda_visual_reliability > 0.0
            and visual_activity is not None
            and reliability_target is not None
        ):
            pred_rel = visual_activity
            if pred_rel.ndim == 2:
                pred_rel = pred_rel.unsqueeze(-1)
            if pred_rel.shape[-1] != 1:
                pred_rel = pred_rel[..., -1:]
            target_rel = reliability_target.to(device=pred_rel.device, dtype=pred_rel.dtype)
            if target_rel.shape[1] != pred_rel.shape[1]:
                target_rel = F.interpolate(
                    target_rel.transpose(1, 2),
                    size=pred_rel.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            # fp32 with autocast disabled: F.binary_cross_entropy is banned
            # inside a CUDA autocast region.  Same treatment as the v3 arm.
            with torch.autocast(device_type=pred_rel.device.type, enabled=False):
                loss_visual_reliability = F.binary_cross_entropy(
                    pred_rel.float().clamp(1e-5, 1.0 - 1e-5),
                    target_rel.float(),
                )

        loss = (
            self.lambda_drift * loss_drift
            + self.lambda_recon * loss_recon
            + self.lambda_consistency * loss_cons
            + self.lambda_target_anchor * loss_target_anchor
            + self.lambda_rms * loss_rms
            + self.lambda_wave * loss_wave
            + self.lambda_target_gain * loss_target_gain
            + self.lambda_residual_leak * loss_residual_leak
            + self.lambda_interferer_leak * loss_interferer_leak
            + self.lambda_target_energy_ceiling * loss_target_energy_ceiling
            + self.lambda_residual_recon * loss_residual_recon
            + self.lambda_mrstft * loss_mrstft
            + self.lambda_visual_reliability * loss_visual_reliability
        )
        return {
            "loss": loss,
            "loss_visual_reliability": loss_visual_reliability.detach(),
            "loss_drift": loss_drift.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_endpoint_boundary": loss_endpoint_boundary.detach(),
            "loss_drift_alpha": loss_drift_alpha.detach(),
            "loss_consistency": loss_cons.detach(),
            "loss_target_anchor": loss_target_anchor.detach(),
            "loss_rms": loss_rms.detach(),
            "loss_wave": loss_wave.detach(),
            "loss_target_gain": loss_target_gain.detach(),
            "loss_residual_leak": loss_residual_leak.detach(),
            "loss_interferer_leak": loss_interferer_leak.detach(),
            "loss_target_energy_ceiling": loss_target_energy_ceiling.detach(),
            "loss_residual_reconstruction": loss_residual_recon.detach(),
            "loss_mrstft": loss_mrstft.detach(),
        }

    def _training_loss_spec_target_only_residual_flow(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Mixture-anchored rectified flow in residual coordinates.

        Let M be the mixture complex STFT and S the target.  Define the target
        correction R* = S - M and the straight residual path

            R_t = t R*,                 t ~ U[0,1]
            X_t = M + R_t = (1-t)M + tS.

        The network receives the current audio state X_t, the fixed mixture M,
        the AV condition, and t, and predicts the residual-space velocity

            v*(R_t,t) = dR_t/dt = S - M.

        The implied endpoint from any sampled t is

            S_hat_t = X_t + (1-t) v_theta(X_t, M, V, t).

        This is deliberately different from ``hybrid_flow``: there is no 50/50
        endpoint oversampling and no drift loss mixed into the FM objective.
        ``lambda_fm`` supervises the velocity field, ``lambda_endpoint`` can
        optionally supervise the complex-STFT endpoint, and MR-STFT/other
        waveform losses are applied to the implied endpoint.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)       # [B,2,F,T]
        mixture_ri = complex_to_ri(mix_spec) # [B,2,F,T]

        # Residual-space straight path: r_t=t(S-M), represented to the head as
        # the equivalent anchored audio state X_t=M+r_t.  At t=0 this exactly
        # matches the working drift model's input operating point (X_0=M), which
        # keeps one-step inference and checkpoint warm-starts well defined.
        residual_target = y - mixture_ri
        batch_size = y.shape[0]
        t = torch.rand(batch_size, device=y.device, dtype=y.dtype)
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        t_b = t.view(shape)
        residual_t = t_b * residual_target
        x_t = mixture_ri + residual_t
        v_target = residual_target

        # Do not expose raw M beside X_t: (X_t-M)/t leaks the exact target
        # velocity during teacher-forced residual-flow training.  The original
        # mixture and vision remain available through C(M,V).
        v_pred = self.spec_velocity(
            x_t,
            t,
            x_t,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
        )

        # Velocity-field objective and endpoint implied from the sampled state.
        loss_fm = F.l1_loss(v_pred, v_target)
        residual_1_hat = residual_t + (1.0 - t_b) * v_pred
        x1_hat = mixture_ri + residual_1_hat
        loss_endpoint = F.l1_loss(x1_hat, y)

        # Target-only mode defines the complementary source deterministically.
        loss_cons = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_anchor = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_rms = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_wave = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_gain = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_interferer_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_energy_ceiling = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_recon = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_mrstft = torch.zeros((), device=y.device, dtype=y.dtype)

        need_wave_endpoint = (
            self.lambda_target_anchor > 0.0
            or self.lambda_rms > 0.0
            or self.lambda_wave > 0.0
            or self.lambda_target_gain > 0.0
            or self.lambda_residual_leak > 0.0
            or self.lambda_interferer_leak > 0.0
            or self.lambda_target_energy_ceiling > 0.0
            or self.lambda_residual_recon > 0.0
            or self.lambda_mrstft > 0.0
        )
        if need_wave_endpoint:
            pred_target_complex = ri_to_sources_complex(x1_hat, num_sources=1)[:, 0]
            pred_target_wave = istft_waveform(
                pred_target_complex, self.stft_cfg, length=target.shape[-1]
            )
            pred_residual_wave = mixture - pred_target_wave

            if self.lambda_mrstft > 0.0:
                loss_mrstft = multi_resolution_stft_loss(pred_target_wave, target)

            if self.lambda_target_gain > 0.0 or self.lambda_residual_leak > 0.0:
                target_energy = target.pow(2).sum(dim=-1) + 1e-8
                target_gain = (pred_target_wave * target).sum(dim=-1) / target_energy
                residual_gain = (pred_residual_wave * target).sum(dim=-1) / target_energy
                loss_target_gain = torch.relu(self.target_gain_floor - target_gain).pow(2).mean()
                loss_residual_leak = residual_gain.abs().mean()

            if self.lambda_residual_recon > 0.0:
                ref_residual_wave = mixture - target
                loss_residual_recon = F.l1_loss(pred_residual_wave, ref_residual_wave)

            if self.lambda_interferer_leak > 0.0:
                ref_residual = mixture - target
                ref_residual_energy = ref_residual.pow(2).sum(dim=-1) + 1e-8
                interferer_gain = (
                    (pred_target_wave * ref_residual).sum(dim=-1) / ref_residual_energy
                )
                loss_interferer_leak = interferer_gain.abs().mean()

            if self.lambda_target_energy_ceiling > 0.0:
                pred_rms_for_ceiling = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms_for_ceiling = target.pow(2).mean(dim=-1).sqrt()
                loss_target_energy_ceiling = torch.relu(
                    pred_rms_for_ceiling
                    - self.target_energy_ceiling_ratio * target_rms_for_ceiling
                ).mean()

            if self.lambda_target_anchor > 0.0:
                sisdr_t = si_sdr_score(pred_target_wave, target)
                sisdr_r = si_sdr_score(pred_residual_wave, target)
                loss_target_anchor = torch.relu(
                    sisdr_r - sisdr_t + self.target_anchor_margin
                ).mean()

            if self.lambda_rms > 0.0:
                pred_rms = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms = target.pow(2).mean(dim=-1).sqrt()
                loss_rms = (pred_rms - target_rms).abs().mean()

            if self.lambda_wave > 0.0:
                loss_wave = F.l1_loss(pred_target_wave, target)

        loss = (
            self.lambda_fm * loss_fm
            + self.lambda_endpoint * loss_endpoint
            + self.lambda_consistency * loss_cons
            + self.lambda_target_anchor * loss_target_anchor
            + self.lambda_rms * loss_rms
            + self.lambda_wave * loss_wave
            + self.lambda_target_gain * loss_target_gain
            + self.lambda_residual_leak * loss_residual_leak
            + self.lambda_interferer_leak * loss_interferer_leak
            + self.lambda_target_energy_ceiling * loss_target_energy_ceiling
            + self.lambda_residual_recon * loss_residual_recon
            + self.lambda_mrstft * loss_mrstft
        )
        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            # Keep the historical logging keys present so existing dashboards
            # and parsing scripts do not need special cases.
            "loss_recon": loss_endpoint.detach(),
            "loss_consistency": loss_cons.detach(),
            "loss_target_anchor": loss_target_anchor.detach(),
            "loss_rms": loss_rms.detach(),
            "loss_wave": loss_wave.detach(),
            "loss_target_gain": loss_target_gain.detach(),
            "loss_residual_leak": loss_residual_leak.detach(),
            "loss_interferer_leak": loss_interferer_leak.detach(),
            "loss_target_energy_ceiling": loss_target_energy_ceiling.detach(),
            "loss_residual_reconstruction": loss_residual_recon.detach(),
            "loss_mrstft": loss_mrstft.detach(),
        }

    def _training_loss_spec_target_only_alphaflow_legacy(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """JVP-free AlphaFlow on the deterministic mixture-to-target path.

        z_t=(1-t)M+tS, v=S-M.  A batch-level Monte Carlo switch samples either
        (1) the diagonal local-FM anchor r=t, or (2) finite-interval
        teacher-student consistency.  Using one branch per batch is an unbiased
        estimator of the decoupled expected objective and avoids evaluating both
        branches every step.

        The raw mixture STFT is deliberately *not* passed beside z_t to the head:
        doing so leaks v=(z_t-M)/t.  M and vision remain available through the
        learned AV conditioner C(M,V).
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)
        mixture_ri = complex_to_ri(mix_spec)
        v_target = y - mixture_ri
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        alpha = self._alphaflow_alpha()

        zero = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_fm = zero
        loss_mf = zero
        use_fm = bool(torch.rand((), device=y.device) < self.alphaflow_fm_ratio)

        if use_fm:
            t = torch.rand(batch_size, device=y.device, dtype=y.dtype)
            r = t
            t_b = t.view(shape)
            z_t = mixture_ri + t_b * v_target
            u = self.spec_velocity(
                z_t, t, z_t, cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
                interval_end=r,
            )
            loss_fm = self._alphaflow_adaptive_loss(u - v_target)
            student = u
            student_t, student_r = t, r
        else:
            t, r = self._sample_alphaflow_interval(batch_size, y.device, y.dtype)
            s_mid = float(alpha) * r + (1.0 - float(alpha)) * t
            z_t = mixture_ri + t.view(shape) * v_target
            z_s = mixture_ri + s_mid.view(shape) * v_target
            student = self.spec_velocity(
                z_t, t, z_t, cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
                interval_end=r,
            )
            with torch.no_grad():
                teacher = self.spec_velocity(
                    z_s, s_mid, z_s, cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                    interval_end=r,
                )
            target_alpha = float(alpha) * v_target + (1.0 - float(alpha)) * teacher
            loss_mf = self._alphaflow_bounded_loss(student - target_alpha, alpha)
            student_t, student_r = t, r

        # Decoupled branch objective as in AlphaFlowTSE.  lambda_fm defaults to
        # 0.6 in the supplied config; lambda_meanflow defaults to 0.4.
        loss = self.lambda_fm * loss_fm + self.lambda_meanflow * loss_mf

        # Log a direct finite-interval endpoint error for interpretability only;
        # it is not optimized unless future experiments explicitly add a weight.
        span = (student_r - student_t).view(shape)
        z_student = mixture_ri + student_t.view(shape) * v_target
        z_r_hat = z_student + span * student
        z_r_true = mixture_ri + student_r.view(shape) * v_target
        loss_endpoint = F.l1_loss(z_r_hat, z_r_true)

        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_meanflow": loss_mf.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            "alphaflow_alpha": torch.tensor(float(alpha), device=y.device, dtype=y.dtype),
            "alphaflow_branch_fm": torch.tensor(float(use_fm), device=y.device, dtype=y.dtype),
            "alphaflow_span": (student_r - student_t).mean().detach(),
        }

    def _training_loss_spec_target_only_alphaflow_v2(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Low-variance finite-alpha AlphaFlow for mixture-to-target AV-SVS.

        Each minibatch contains both diagonal trajectory-FM samples and
        finite-interval teacher-consistency samples. Per-sample weighting is
        1/(E+eps) on the boundary and alpha/(E+eps) on finite intervals.
        Branch frequency alone controls their balance; the legacy 0.6/0.4
        coefficients are intentionally not applied.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)
        mixture_ri = complex_to_ri(mix_spec)
        v_target = y - mixture_ri
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        alpha = float(self._alphaflow_alpha())

        n_fm = self._alphaflow_v2_fm_count(batch_size, y.device)
        permutation = torch.randperm(batch_size, device=y.device)
        fm_index = permutation[:n_fm]
        interval_index = permutation[n_fm:]
        fm_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)
        fm_mask[fm_index] = True
        interval_mask = ~fm_mask

        t = torch.empty(batch_size, device=y.device, dtype=y.dtype)
        r = torch.empty_like(t)
        large_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)

        if n_fm > 0:
            t_fm = torch.rand(n_fm, device=y.device, dtype=y.dtype)
            t[fm_index] = t_fm
            r[fm_index] = t_fm

        n_interval = batch_size - n_fm
        if n_interval > 0:
            t_interval, r_interval, large_interval = self._sample_alphaflow_interval_v2(
                n_interval, y.device, y.dtype
            )
            t[interval_index] = t_interval
            r[interval_index] = r_interval
            large_mask[interval_index] = large_interval

        z_t = mixture_ri + t.view(shape) * v_target
        student = self.spec_velocity(
            z_t, t, z_t, cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
            interval_end=r,
        )
        target_velocity = v_target.detach().clone()

        if n_interval > 0:
            t_interval = t[interval_index]
            r_interval = r[interval_index]
            s_interval = alpha * r_interval + (1.0 - alpha) * t_interval
            v_interval = v_target[interval_index]
            interval_shape = (n_interval,) + (1,) * (y.ndim - 1)
            z_s = (
                mixture_ri[interval_index]
                + s_interval.view(interval_shape) * v_interval
            )
            interval_tokens = (
                temporal_tokens[interval_index]
                if temporal_tokens is not None else None
            )
            interval_activity = (
                visual_activity[interval_index]
                if visual_activity is not None else None
            )
            with torch.no_grad():
                teacher = self.spec_velocity(
                    z_s,
                    s_interval,
                    z_s,
                    cond[interval_index],
                    temporal_tokens=interval_tokens,
                    visual_activity=interval_activity,
                    interval_end=r_interval,
                )
            target_velocity[interval_index] = (
                alpha * v_interval + (1.0 - alpha) * teacher
            )

        per_sample_mse = self._per_sample_mse(student - target_velocity)
        per_sample_loss, weight = self._alphaflow_v2_weighted_loss(
            per_sample_mse, interval_mask, alpha
        )
        loss = per_sample_loss.mean()

        zero = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_fm = per_sample_loss[fm_mask].mean() if n_fm > 0 else zero
        loss_interval = (
            per_sample_loss[interval_mask].mean() if n_interval > 0 else zero
        )
        mse_fm = per_sample_mse[fm_mask].mean() if n_fm > 0 else zero
        mse_interval = (
            per_sample_mse[interval_mask].mean() if n_interval > 0 else zero
        )

        span = (r - t).view(shape)
        z_r_hat = z_t + span * student
        z_r_true = mixture_ri + r.view(shape) * v_target
        loss_endpoint = F.l1_loss(z_r_hat, z_r_true)
        interval_span = (
            (r[interval_mask] - t[interval_mask]).mean()
            if n_interval > 0 else zero
        )

        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_meanflow": loss_interval.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            "alphaflow_alpha": torch.tensor(
                alpha, device=y.device, dtype=y.dtype
            ),
            # Historical key retained; for v2 it reports the sample fraction.
            "alphaflow_branch_fm": fm_mask.float().mean().detach(),
            "alphaflow_fm_fraction": fm_mask.float().mean().detach(),
            "alphaflow_large_span": large_mask.float().mean().detach(),
            "alphaflow_span": (r - t).mean().detach(),
            "alphaflow_interval_span": interval_span.detach(),
            "alphaflow_mse_fm": mse_fm.detach(),
            "alphaflow_mse_interval": mse_interval.detach(),
            "alphaflow_weight_mean": weight.mean().detach(),
        }

    def _training_loss_spec_target_only_alphaflow_v3(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """AlphaFlow v3: faithful branch losses + deployment-matched AV anchors.

        This keeps the mixture-to-target path z_t=M+t(S-M) and JVP-free
        stop-gradient self-teacher, but fixes the v2 interval weighting so the
        consistency branch is not suppressed as alpha decreases.  A subset of
        interval samples exactly matches one-NFE deployment (t,r)=(0,1).

        FiLM uses the selected temporal AV tokens while bottleneck cross-attention
        receives the visual tokens explicitly as K/V.  If train-time visual token
        corruption is enabled, the learned visual reliability is supervised and
        also gates the cross-attention residual.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_tokens = captured.get("video_tokens", None)
        visual_activity = captured.get("visual_activity", None)
        reliability_target = captured.get("visual_reliability_target", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)
        mixture_ri = complex_to_ri(mix_spec)
        # Transport start: sample z_0 early so noise initialization uses the same
        # sample throughout training and conditioner both see the real mixture.
        init_z = self._target_only_initial_state(mixture_ri)
        anchor = self._flow_anchor(mixture, mixture_ri, b["face"], b["body"], init_z=init_z)
        # With a prior the head's conditioning slot carries the REAL mixture:
        # the prior's errors are mostly under-recovery, and a refiner that sees
        # only P cannot restore energy P already discarded.  With noise initialization,
        # the evolving state has no mixture phase, so supply it explicitly.
        noise_init = self.spec_init_mode in {"noise", "gaussian", "random"}
        use_mixture_slot = noise_init or (
            self.prior is not None and self.prior_condition_on == "mixture"
        )
        v_target = y - anchor
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        alpha = float(self._alphaflow_alpha())

        n_fm = self._alphaflow_v2_fm_count(batch_size, y.device)
        permutation = torch.randperm(batch_size, device=y.device)
        fm_index = permutation[:n_fm]
        interval_index = permutation[n_fm:]
        fm_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)
        fm_mask[fm_index] = True
        interval_mask = ~fm_mask

        t = torch.empty(batch_size, device=y.device, dtype=y.dtype)
        r = torch.empty_like(t)
        large_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)
        deploy_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)

        if n_fm > 0:
            t_fm = torch.rand(n_fm, device=y.device, dtype=y.dtype)
            t[fm_index] = t_fm
            r[fm_index] = t_fm

        n_interval = batch_size - n_fm
        if n_interval > 0:
            t_i, r_i, large_i, deploy_i = self._sample_alphaflow_interval_v3(
                n_interval, y.device, y.dtype
            )
            t[interval_index] = t_i
            r[interval_index] = r_i
            large_mask[interval_index] = large_i
            deploy_mask[interval_index] = deploy_i

        z_t = anchor + t.view(shape) * v_target

        # Dual-head forward if enabled and available, else standard single-head
        has_dual_head = (
            getattr(self.head, "dual_head", False)
            and self.head.delta_conv is not None
        )
        if has_dual_head:
            u_base, u_delta, student = self.head.forward_dual_head(
                z_t,
                mixture_ri if use_mixture_slot else z_t,
                cond,
                t,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
                cross_attention_tokens=visual_tokens,
                interval_end=r,
            )
        else:
            student = self.spec_velocity(
                z_t,
                t,
                mixture_ri if use_mixture_slot else z_t,
                cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
                cross_attention_tokens=visual_tokens,
                interval_end=r,
            )

        target_velocity = v_target.detach().clone()

        if n_interval > 0:
            t_i = t[interval_index]
            r_i = r[interval_index]
            s_i = alpha * r_i + (1.0 - alpha) * t_i
            v_i = v_target[interval_index]
            interval_shape = (n_interval,) + (1,) * (y.ndim - 1)
            z_s = anchor[interval_index] + s_i.view(interval_shape) * v_i
            interval_tokens = temporal_tokens[interval_index] if temporal_tokens is not None else None
            interval_visual_tokens = visual_tokens[interval_index] if visual_tokens is not None else None
            interval_activity = visual_activity[interval_index] if visual_activity is not None else None
            with torch.no_grad():
                teacher = self.spec_velocity(
                    z_s,
                    s_i,
                    mixture_ri[interval_index] if use_mixture_slot else z_s,
                    cond[interval_index],
                    temporal_tokens=interval_tokens,
                    visual_activity=interval_activity,
                    cross_attention_tokens=interval_visual_tokens,
                    interval_end=r_i,
                )
            target_velocity[interval_index] = alpha * v_i + (1.0 - alpha) * teacher

        residual = student - target_velocity
        zero = torch.zeros((), device=y.device, dtype=y.dtype)
        if n_fm > 0:
            loss_fm = self._alphaflow_adaptive_loss(residual[fm_mask])
        else:
            loss_fm = zero
        if n_interval > 0:
            loss_mf = self._alphaflow_bounded_loss(residual[interval_mask], alpha)
        else:
            loss_mf = zero

        # Faithful AlphaFlowTSE-style decoupled branch weighting.
        loss = self.lambda_fm * loss_fm + self.lambda_meanflow * loss_mf

        # Ground-truth endpoint anchor.
        #
        # With endpoint_scope="deployment" only exact (0,1) samples are used, so
        # the anchor cannot replace interval consistency with a generic direct
        # separator objective.  With "all_intervals" every finite-interval sample
        # contributes: because the transport is linear, z_r_true = z_t + span*v
        # holds exactly, so the L1 below equals span*L1(student, v_target).  The
        # span factor is the weighting -- near-full spans (what one-NFE inference
        # queries) dominate and short spans contribute almost nothing -- so this
        # densifies the ground-truth signal without flattening the (t,r) domain.
        span = (r - t).view(shape)
        z_r_hat = z_t + span * student
        z_r_true = anchor + r.view(shape) * v_target
        if self.alphaflow_endpoint_scope == "all_intervals":
            # FM samples have t==r, hence span==0 and an identically zero
            # contribution, so they are excluded rather than averaged in as zeros.
            endpoint_mask = interval_mask
        else:
            endpoint_mask = deploy_mask
        if endpoint_mask.any():
            loss_endpoint = F.l1_loss(z_r_hat[endpoint_mask], z_r_true[endpoint_mask])
        else:
            loss_endpoint = zero
        loss = loss + self.lambda_endpoint * loss_endpoint

        loss_mrstft = zero
        if self.lambda_mrstft > 0.0 and deploy_mask.any():
            pred_complex = ri_to_sources_complex(z_r_hat[deploy_mask], num_sources=1)[:, 0]
            true_complex = ri_to_sources_complex(z_r_true[deploy_mask], num_sources=1)[:, 0]
            pred_wave = istft_waveform(pred_complex, self.stft_cfg, length=mixture.shape[-1])
            true_wave = istft_waveform(true_complex, self.stft_cfg, length=mixture.shape[-1])
            loss_mrstft = multi_resolution_stft_loss(pred_wave, true_wave)
            loss = loss + self.lambda_mrstft * loss_mrstft

        loss_visual_reliability = zero
        if (
            self.lambda_visual_reliability > 0.0
            and visual_activity is not None
            and reliability_target is not None
        ):
            pred_rel = visual_activity
            if pred_rel.ndim == 2:
                pred_rel = pred_rel.unsqueeze(-1)
            if pred_rel.shape[-1] != 1:
                # When raw visual curves are concatenated, the learned reliability
                # channel is appended last by the conditioner.
                pred_rel = pred_rel[..., -1:]
            target_rel = reliability_target.to(device=pred_rel.device, dtype=pred_rel.dtype)
            if target_rel.shape[1] != pred_rel.shape[1]:
                target_rel = F.interpolate(
                    target_rel.transpose(1, 2),
                    size=pred_rel.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            # F.binary_cross_entropy is banned inside a CUDA autocast region
            # (unsafe in reduced precision), and v3 is the first config to pair
            # amp: bf16 with lambda_visual_reliability > 0 -- so this raised at
            # step 1 on GPU while every CPU test passed, because the CPU
            # autocast policy does not ban it.
            #
            # binary_cross_entropy_with_logits is not a drop-in: the conditioner
            # emits visual_activity already sigmoid-ed and never carries the
            # pre-sigmoid logits through the bundle, and that probability is
            # reused downstream as the cross-attention gate.  Computing the term
            # in float32 with autocast disabled keeps the loss identical.  The
            # clamp bounds it at -log(1e-5), so fp32 BCE here is stable.
            with torch.autocast(device_type=pred_rel.device.type, enabled=False):
                loss_visual_reliability = F.binary_cross_entropy(
                    pred_rel.float().clamp(1e-5, 1.0 - 1e-5),
                    target_rel.float(),
                )
            loss = loss + self.lambda_visual_reliability * loss_visual_reliability

        per_sample_mse = self._per_sample_mse(residual)
        mse_fm = per_sample_mse[fm_mask].mean() if n_fm > 0 else zero
        mse_interval = per_sample_mse[interval_mask].mean() if n_interval > 0 else zero
        weights = torch.zeros_like(per_sample_mse)
        if n_fm > 0:
            m = per_sample_mse[fm_mask]
            weights[fm_mask] = (m + self.alphaflow_adaptive_eps).pow(
                self.alphaflow_adaptive_gamma - 1.0
            )
        if n_interval > 0:
            m = per_sample_mse[interval_mask]
            weights[interval_mask] = self.alphaflow_bounded_kappa / (
                m + alpha * self.alphaflow_bounded_kappa + self.alphaflow_bounded_eps
            )

        interval_span = (
            (r[interval_mask] - t[interval_mask]).mean() if n_interval > 0 else zero
        )

        # Dual-head base and delta losses
        loss_base_drift = zero
        loss_delta_reg = zero
        base_velocity_rms = zero
        delta_velocity_rms = zero
        delta_to_base_rms_ratio = zero

        if has_dual_head:
            # Base head: L1 loss on exact residuals (S - M)
            residual_target = v_target.detach().clone()
            loss_base_drift = F.l1_loss(u_base, residual_target)
            base_velocity_rms = u_base.std()

            # Delta head: normalized L2 regularization (prevents unbounded growth)
            delta_energy = u_delta.square().flatten(1).mean(1)
            target_energy = residual_target.square().flatten(1).mean(1).detach()
            normalized_delta_energy = delta_energy / (target_energy + 1e-6)
            loss_delta_reg = normalized_delta_energy.mean()
            delta_velocity_rms = u_delta.std()

            if base_velocity_rms > 1e-8:
                delta_to_base_rms_ratio = (delta_velocity_rms / base_velocity_rms).detach()

            # Accumulate dual-head losses
            loss = loss + self.lambda_base_drift * loss_base_drift + self.lambda_delta_reg * loss_delta_reg

        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_meanflow": loss_mf.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            "loss_mrstft": loss_mrstft.detach(),
            "loss_visual_reliability": loss_visual_reliability.detach(),
            "loss_base_drift": loss_base_drift.detach() if has_dual_head else zero,
            "loss_delta_reg": loss_delta_reg.detach() if has_dual_head else zero,
            "base_velocity_rms": base_velocity_rms.detach() if has_dual_head else zero,
            "delta_velocity_rms": delta_velocity_rms.detach() if has_dual_head else zero,
            "delta_to_base_rms_ratio": delta_to_base_rms_ratio.detach() if has_dual_head else zero,
            "alphaflow_alpha": torch.tensor(alpha, device=y.device, dtype=y.dtype),
            "alphaflow_branch_fm": fm_mask.float().mean().detach(),
            "alphaflow_fm_fraction": fm_mask.float().mean().detach(),
            "alphaflow_large_span": large_mask.float().mean().detach(),
            "alphaflow_deployment_fraction": deploy_mask.float().mean().detach(),
            "alphaflow_endpoint_fraction": endpoint_mask.float().mean().detach(),
            "alphaflow_span": (r - t).mean().detach(),
            "alphaflow_interval_span": interval_span.detach(),
            "alphaflow_mse_fm": mse_fm.detach(),
            "alphaflow_mse_interval": mse_interval.detach(),
            "alphaflow_weight_mean": weights.mean().detach(),
        }

    def _training_loss_spec_target_only_alphaflow(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.alphaflow_variant in {
            "v3", "deployment_v3", "faithful_v3", "av_v3", "alphaflow_v3"
        }:
            return self._training_loss_spec_target_only_alphaflow_v3(batch)
        if self.alphaflow_variant in {
            "v2", "faithful", "faithful_v2", "sample_level", "sample-level"
        }:
            return self._training_loss_spec_target_only_alphaflow_v2(batch)
        return self._training_loss_spec_target_only_alphaflow_legacy(batch)

    def _training_loss_spec_target_only_flowmap_adapter(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Deployment-anchored flow-map training over a frozen drift separator.

        Every update contains a full-batch one-step deployment loss.  After an
        adapter-only warm-up, a second batch of states is sampled from three
        regimes: exact deployment, oracle path states, and states lying on the
        current student's erroneous path.  The latter closes the train/test gap
        that makes multi-step AlphaFlow deteriorate when it is trained only on
        the clean line from mixture to target.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_tokens = captured.get("video_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        mixture_ri = complex_to_ri(mix_spec)
        y = complex_to_ri(target_spec)
        v_target = y - mixture_ri
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        zero = torch.zeros((), device=y.device, dtype=y.dtype)

        def adapter_forward(
            state: torch.Tensor,
            t: torch.Tensor,
            r: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # The second slot is deliberately the original mixture for every
            # training regime and every rollout step.
            return self.head.forward_dual_head(
                state,
                mixture_ri,
                cond,
                t,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
                cross_attention_tokens=visual_tokens,
                interval_end=r,
            )

        # The deployment query is present over the complete batch on every
        # optimizer update, independent of auxiliary-state sampling ratios.
        t0 = torch.zeros(batch_size, device=y.device, dtype=y.dtype)
        r1 = torch.ones_like(t0)
        u_base_dep, u_delta_dep, u_dep = adapter_forward(mixture_ri, t0, r1)
        base_endpoint = mixture_ri + u_base_dep
        deployment_endpoint = mixture_ri + u_dep
        loss_base_deployment = F.l1_loss(base_endpoint, y)
        loss_deployment = F.l1_loss(deployment_endpoint, y)
        loss = self.lambda_deployment * loss_deployment

        warmup = self._training_progress_step < self.flowmap_warmup_steps
        loss_map = zero
        loss_composition = zero
        loss_map_deployment = zero
        loss_map_oracle = zero
        loss_map_on_policy = zero
        deployment_mask = torch.ones(batch_size, device=y.device, dtype=torch.bool)
        oracle_mask = torch.zeros_like(deployment_mask)
        on_policy_mask = torch.zeros_like(deployment_mask)
        sampled_delta = u_delta_dep

        if not warmup:
            probabilities = torch.tensor(
                [
                    self.flowmap_deployment_prob,
                    self.flowmap_oracle_prob,
                    self.flowmap_on_policy_prob,
                ],
                device=y.device,
                dtype=torch.float32,
            )
            regime = torch.multinomial(
                probabilities.expand(batch_size, -1), 1
            ).squeeze(1)
            deployment_mask = regime == 0
            oracle_mask = regime == 1
            on_policy_mask = regime == 2

            t = torch.zeros(batch_size, device=y.device, dtype=y.dtype)
            r = torch.ones_like(t)
            state = mixture_ri.clone()
            map_target = y.clone()

            non_deployment = oracle_mask | on_policy_mask
            if non_deployment.any():
                t[non_deployment] = (
                    torch.rand(
                        int(non_deployment.sum().item()),
                        device=y.device,
                        dtype=y.dtype,
                    )
                    * self.flowmap_t_max
                )

            if oracle_mask.any():
                t_oracle = t[oracle_mask]
                r_floor = (t_oracle + self.flowmap_min_span).clamp_max(1.0)
                r_oracle = r_floor + (1.0 - r_floor) * torch.rand_like(t_oracle)
                r[oracle_mask] = r_oracle
                oracle_shape = (int(oracle_mask.sum().item()),) + (1,) * (y.ndim - 1)
                state[oracle_mask] = (
                    mixture_ri[oracle_mask]
                    + t_oracle.view(oracle_shape) * v_target[oracle_mask]
                )
                map_target[oracle_mask] = (
                    mixture_ri[oracle_mask]
                    + r_oracle.view(oracle_shape) * v_target[oracle_mask]
                )

            if on_policy_mask.any():
                on_policy_shape = (int(on_policy_mask.sum().item()),) + (1,) * (y.ndim - 1)
                # Error recycling: the state follows the student's current
                # deployment prediction, not the oracle mixture-target line.
                state[on_policy_mask] = (
                    mixture_ri[on_policy_mask]
                    + t[on_policy_mask].view(on_policy_shape)
                    * u_dep[on_policy_mask].detach()
                )

            _, sampled_delta, sampled_velocity = adapter_forward(state, t, r)
            span = (r - t).view(shape)
            map_endpoint = state + span * sampled_velocity
            loss_map = F.l1_loss(map_endpoint, map_target)
            loss = loss + self.lambda_flowmap * loss_map

            if deployment_mask.any():
                loss_map_deployment = F.l1_loss(
                    map_endpoint[deployment_mask], map_target[deployment_mask]
                )
            if oracle_mask.any():
                loss_map_oracle = F.l1_loss(
                    map_endpoint[oracle_mask], map_target[oracle_mask]
                )
            if on_policy_mask.any():
                loss_map_on_policy = F.l1_loss(
                    map_endpoint[on_policy_mask], map_target[on_policy_mask]
                )

            composition_active = (
                self.lambda_composition > 0.0
                and self._training_progress_step >= self.flowmap_composition_start_step
                and bool(
                    torch.rand((), device=y.device)
                    < self.flowmap_composition_prob
                )
            )
            if composition_active:
                midpoint_t = 0.5 * (t + r)
                first_span = (midpoint_t - t).view(shape)
                second_span = (r - midpoint_t).view(shape)
                # The first predicted state is detached on purpose: the second
                # call learns to recover from a realistic student state without
                # using a high-memory gradient path through the rollout.
                with torch.no_grad():
                    _, _, first_velocity = adapter_forward(state, t, midpoint_t)
                    midpoint_state = state + first_span * first_velocity
                _, _, second_velocity = adapter_forward(
                    midpoint_state.detach(), midpoint_t, r
                )
                composed_endpoint = (
                    midpoint_state.detach() + second_span * second_velocity
                )
                loss_composition = F.l1_loss(
                    composed_endpoint, map_endpoint.detach()
                )
                loss = loss + self.lambda_composition * loss_composition

        # Trust region for the correction.  Normalizing by the frozen base
        # prediction makes this stable across clips with different loudness.
        delta_energy = u_delta_dep.square().flatten(1).mean(1)
        base_energy = u_base_dep.detach().square().flatten(1).mean(1)
        loss_adapter_reg = (delta_energy / (base_energy + 1e-6)).mean()
        loss = loss + self.lambda_adapter_reg * loss_adapter_reg

        loss_mrstft = zero
        if self.lambda_mrstft > 0.0:
            pred_complex = ri_to_sources_complex(
                deployment_endpoint, num_sources=1
            )[:, 0]
            pred_wave = istft_waveform(
                pred_complex, self.stft_cfg, length=mixture.shape[-1]
            )
            loss_mrstft = multi_resolution_stft_loss(pred_wave, target)
            loss = loss + self.lambda_mrstft * loss_mrstft

        base_rms = u_base_dep.square().mean().sqrt()
        delta_rms = u_delta_dep.square().mean().sqrt()
        sampled_delta_rms = sampled_delta.square().mean().sqrt()
        delta_ratio = delta_rms / (base_rms.detach() + 1e-8)
        correction_gain_l1 = loss_base_deployment.detach() - loss_deployment.detach()

        return {
            "loss": loss,
            "loss_deployment": loss_deployment.detach(),
            "loss_flowmap": loss_map.detach(),
            "loss_composition": loss_composition.detach(),
            "loss_mrstft": loss_mrstft.detach(),
            "loss_adapter_reg": loss_adapter_reg.detach(),
            "loss_base_deployment": loss_base_deployment.detach(),
            "loss_map_deployment": loss_map_deployment.detach(),
            "loss_map_oracle": loss_map_oracle.detach(),
            "loss_map_on_policy": loss_map_on_policy.detach(),
            "flowmap_warmup": torch.tensor(
                float(warmup), device=y.device, dtype=y.dtype
            ),
            "flowmap_deployment_fraction": deployment_mask.float().mean().detach(),
            "flowmap_oracle_fraction": oracle_mask.float().mean().detach(),
            "flowmap_on_policy_fraction": on_policy_mask.float().mean().detach(),
            "base_velocity_rms": base_rms.detach(),
            "delta_velocity_rms": delta_rms.detach(),
            "sampled_delta_velocity_rms": sampled_delta_rms.detach(),
            "delta_to_base_rms_ratio": delta_ratio.detach(),
            "correction_gain_l1": correction_gain_l1,
        }

    def _training_loss_spec_target_only_visual_floss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Mixture-consistent visual flow matching around a drift warm start.

        The target slot starts at ``z0 = mixture + noise`` and the unmodelled
        background slot is defined as ``mixture - z0``.  Along the complete
        path the background remains ``mixture - target_state`` and therefore
        has velocity ``-v``.  We only predict the target velocity; this makes
        target + background == mixture an algebraic invariant rather than a
        soft loss.

        Half of a typical batch is evaluated at the deployment point
        ``(noise,t)=(0,0)``.  In addition, the whole batch receives an explicit
        one-step deployment anchor so fine-tuning cannot silently destroy the
        strong drift checkpoint while learning noisy intermediate states.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_tokens = captured.get("video_tokens", None)
        visual_activity = captured.get("visual_activity", None)
        reliability_target = captured.get("visual_reliability_target", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        mixture_ri = complex_to_ri(mix_spec)
        y = complex_to_ri(target_spec)
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        zero = torch.zeros((), device=y.device, dtype=y.dtype)

        # Use waveform-shaped noise so every perturbed complex STFT corresponds
        # to a valid real signal.  Scale it per example by mixture RMS rather
        # than by an absolute STFT magnitude.
        mixture_rms = mixture.square().mean(dim=-1, keepdim=True).sqrt()
        if self.visual_floss_curriculum_steps > 0:
            curriculum = min(
                1.0,
                self._training_progress_step / self.visual_floss_curriculum_steps,
            )
        else:
            curriculum = 1.0
        noise_wave = (
            torch.randn_like(mixture)
            * mixture_rms
            * self.visual_floss_noise_scale
            * curriculum
        )
        noise_ri = complex_to_ri(stft_waveform(noise_wave, self.stft_cfg))

        deployment_count = int(round(batch_size * self.visual_floss_deployment_ratio))
        deployment_count = max(0, min(batch_size, deployment_count))
        deployment_mask = torch.zeros(
            batch_size, device=y.device, dtype=torch.bool
        )
        if deployment_count:
            indices = torch.randperm(batch_size, device=y.device)[:deployment_count]
            deployment_mask[indices] = True
            noise_ri = noise_ri.clone()
            noise_ri[deployment_mask] = 0

        z0 = mixture_ri + noise_ri
        if self.visual_floss_time_sampling == "logit_normal":
            logits = (
                torch.randn(batch_size, device=y.device, dtype=y.dtype)
                * self.visual_floss_logit_sigma
                + self.visual_floss_logit_mu
            )
            t = torch.sigmoid(logits) * self.visual_floss_max_t * curriculum
        else:
            t = (
                torch.rand(batch_size, device=y.device, dtype=y.dtype)
                * self.visual_floss_max_t
                * curriculum
            )
        t = t.masked_fill(deployment_mask, 0.0)

        v_target = y - z0
        x_t = z0 + t.view(shape) * v_target
        v_pred = self.spec_velocity(
            x_t,
            t,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
            cross_attention_tokens=visual_tokens,
        )

        # FLOSS's scale-normalized dB objective.  Compute in fp32 outside AMP;
        # a floor prevents nearly solved drift examples from producing enormous
        # gradients through log(error) on older fp16 GPUs.
        with torch.autocast(device_type=y.device.type, enabled=False):
            error_energy = (
                (v_pred.float() - v_target.float()).square().flatten(1).mean(1)
            )
            target_energy = v_target.float().square().flatten(1).mean(1)
            relative_mse = (
                (error_energy + self.visual_floss_loss_eps)
                / (target_energy + self.visual_floss_loss_eps)
            )
            per_sample_db = 10.0 * torch.log10(
                relative_mse.clamp_min(self.visual_floss_loss_eps)
            )
            clipped_db = per_sample_db.clamp(
                min=self.visual_floss_db_floor,
                max=self.visual_floss_db_ceiling,
            )
            loss_floss_db = clipped_db.mean()
            # Adding a constant does not change the gradient, but keeps the
            # combined training loss non-negative and easier to monitor.
            loss_floss = (clipped_db - self.visual_floss_db_floor).mean()

        sampled_endpoint = x_t + (1.0 - t).view(shape) * v_pred
        loss_endpoint = F.l1_loss(sampled_endpoint, y)

        # Exact one-step deployment query for every example.  With zero noise
        # and t=0 this is precisely the pre-trained residual predictor's call.
        t0 = torch.zeros(batch_size, device=y.device, dtype=y.dtype)
        deployment_velocity = self.spec_velocity(
            mixture_ri,
            t0,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
            cross_attention_tokens=visual_tokens,
        )
        deployment_endpoint = mixture_ri + deployment_velocity
        loss_deployment = F.l1_loss(deployment_endpoint, y)

        loss_mrstft = zero
        if self.lambda_mrstft > 0.0:
            pred_complex = ri_to_sources_complex(
                deployment_endpoint, num_sources=1
            )[:, 0]
            pred_wave = istft_waveform(
                pred_complex, self.stft_cfg, length=mixture.shape[-1]
            )
            loss_mrstft = multi_resolution_stft_loss(pred_wave, target)

        loss_visual_reliability = zero
        if (
            self.lambda_visual_reliability > 0.0
            and visual_activity is not None
            and reliability_target is not None
        ):
            pred_rel = visual_activity
            if pred_rel.ndim == 2:
                pred_rel = pred_rel.unsqueeze(-1)
            if pred_rel.shape[-1] != 1:
                pred_rel = pred_rel[..., -1:]
            target_rel = reliability_target.to(
                device=pred_rel.device, dtype=pred_rel.dtype
            )
            if target_rel.shape[1] != pred_rel.shape[1]:
                target_rel = F.interpolate(
                    target_rel.transpose(1, 2),
                    size=pred_rel.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            with torch.autocast(device_type=pred_rel.device.type, enabled=False):
                loss_visual_reliability = F.binary_cross_entropy(
                    pred_rel.float().clamp(1e-5, 1.0 - 1e-5),
                    target_rel.float(),
                )

        loss = (
            self.lambda_deployment * loss_deployment
            + self.lambda_floss * loss_floss
            + self.lambda_endpoint * loss_endpoint
            + self.lambda_mrstft * loss_mrstft
            + self.lambda_visual_reliability * loss_visual_reliability
        )

        # The complement is implicit, but compute the invariant as a regression
        # diagnostic.  It should stay at numerical zero for every checkpoint.
        background_endpoint = mixture_ri - sampled_endpoint
        mixture_error = (
            sampled_endpoint + background_endpoint - mixture_ri
        ).abs().mean()
        velocity_sum_error = (v_pred + (-v_pred)).abs().mean()

        return {
            "loss": loss,
            "loss_deployment": loss_deployment.detach(),
            "loss_floss": loss_floss.detach(),
            "loss_floss_db": loss_floss_db.detach(),
            "loss_floss_endpoint": loss_endpoint.detach(),
            "loss_mrstft": loss_mrstft.detach(),
            "loss_visual_reliability": loss_visual_reliability.detach(),
            "visual_floss_relative_mse": relative_mse.mean().detach(),
            "visual_floss_deployment_fraction": deployment_mask.float().mean().detach(),
            "visual_floss_curriculum": torch.tensor(
                curriculum, device=y.device, dtype=y.dtype
            ),
            "visual_floss_t_mean": t.mean().detach(),
            "visual_floss_noise_rms": noise_wave.square().mean().sqrt().detach(),
            "visual_floss_velocity_rms": v_target.square().mean().sqrt().detach(),
            "visual_floss_pred_velocity_rms": v_pred.square().mean().sqrt().detach(),
            "visual_floss_mixture_error": mixture_error.detach(),
            "visual_floss_velocity_sum_error": velocity_sum_error.detach(),
        }

    def _load_frozen_prior(self, checkpoint: str, *, use_ema: bool) -> "torch.nn.Module":
        """Load a trained separator as a frozen, eval-pinned first stage.

        Loaded strictly on purpose.  The trainer's own loader uses strict=False
        so older checkpoints survive head changes, but a silently half-random
        prior would corrupt every anchor with no visible error -- the worst
        failure mode this path has.
        """
        import os

        path = os.path.expanduser(os.path.expandvars(checkpoint))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"flow.prior.checkpoint does not exist: {path!r}. Point it at the "
                "trained predictor's best.pt."
            )
        blob = torch.load(path, map_location="cpu")
        prior_model_cfg = blob.get("cfg")
        if prior_model_cfg is None or "model" not in blob:
            raise ValueError(f"{path!r} is not a training checkpoint (needs 'cfg' and 'model')")
        if (prior_model_cfg.get("flow") or {}).get("prior"):
            raise ValueError(
                "Chained priors are not supported: the prior checkpoint was itself "
                "trained with flow.prior. Point flow.prior.checkpoint at a "
                "single-stage model."
            )
        prior = type(self)(prior_model_cfg)
        prior.load_state_dict(blob["model"], strict=True)
        # best.pt is written with EMA weights already swapped in; raw checkpoints
        # (last.pt, best_raw.pt) carry them separately as a shadow.
        if use_ema and not blob.get("is_ema_model", False):
            if "ema" not in blob:
                raise KeyError(
                    f"flow.prior.use_ema=true but {path!r} holds no EMA weights; "
                    "point at best.pt or set use_ema to false"
                )
            shadow = blob["ema"].get("shadow", blob["ema"])
            params = dict(prior.named_parameters())
            with torch.no_grad():
                for name, avg in shadow.items():
                    if name in params:
                        params[name].copy_(avg.to(dtype=params[name].dtype))
        for param in prior.parameters():
            param.requires_grad_(False)
        prior.eval()
        return prior

    def train(self, mode: bool = True):
        """Keep frozen components in eval mode whatever the trainer asks for.

        The trainer calls model.train() every step, and that recurses into
        submodules.  For the prior it would switch its conditioner's train-time
        augmentation back on -- visual token dropout and noise, audio condition
        dropout -- and hand the flow a different, noisier anchor on every call.
        """
        super().train(mode)
        if getattr(self, "prior", None) is not None:
            self.prior.eval()
        if mode and getattr(self, "spec_objective", None) == "flowmap_adapter":
            self._freeze_for_flowmap_adapter()
        return self

    def _freeze_for_flowmap_adapter(self) -> None:
        """Make the pre-trained drift estimator immutable in weights *and* state.

        ``requires_grad=False`` alone is insufficient because BatchNorm running
        statistics and train-time condition corruption can still change a
        supposedly frozen model.  Every subtree is therefore pinned to eval,
        then only the standalone correction adapter is activated.
        """
        if not isinstance(self.head, TFCTDFUNetFlowHead):
            raise TypeError("flowmap adapter freezing requires TFCTDFUNetFlowHead")
        adapter = self.head.flowmap_adapter
        if adapter is None:
            raise RuntimeError("flowmap correction adapter is missing")
        pending_lazy = False
        for param in self.parameters():
            if isinstance(param, UninitializedParameter):
                pending_lazy = True
                continue
            param.requires_grad_(False)
        for child in self.children():
            child.eval()
        for param in adapter.parameters():
            param.requires_grad_(True)
        adapter.train(self.training)
        self._flowmap_freeze_pending = pending_lazy

    def set_training_stage(self, epoch: int, stage_config: Optional[list] = None) -> None:
        """Freeze/unfreeze modules based on training stage (epoch).

        stage_config is a list of (epoch_threshold, module_name) tuples.

        CRITICAL DESIGN: ROOT MODEL STAYS IN TRAINING MODE.
        Only frozen child subtrees go to eval() to prevent BatchNorm drift.
        This design is clean, automatic, and avoids validation restoration races.
        """
        if self.spec_objective == "flowmap_adapter":
            if stage_config:
                raise ValueError(
                    "flowmap_adapter forbids training.stage_unfreezes; the drift "
                    "conditioner and U-Net must remain frozen for the whole run"
                )
            self._freeze_for_flowmap_adapter()
            return
        if stage_config is None:
            return

        # Freeze all parameters (resume-safe: reapplied every epoch)
        for param in self.parameters():
            param.requires_grad = False

        # Put all TOP-LEVEL CHILDREN in eval mode (frozen subtrees)
        # ROOT MODEL STAYS IN TRAINING MODE
        for child in self.children():
            child.eval()

        # Unfreeze and activate modules specified at or below current epoch
        for threshold_epoch, module_name in stage_config:
            if epoch >= threshold_epoch:
                module = self.get_submodule(module_name)
                for param in module.parameters():
                    param.requires_grad = True
                # Put this module back in train mode
                module.train()

    def _flow_anchor(
        self,
        mixture: torch.Tensor,
        mixture_ri: torch.Tensor,
        face: Optional[torch.Tensor],
        body: Optional[torch.Tensor],
        *,
        init_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Start of the transport: the mixture, noise sample, or a frozen prior's estimate.

        Priority: prior > noise > mixture. With flow.prior, uses the prior's one-step
        estimate. With init_mode='noise' and no prior, uses the noise sample. Without
        both, returns mixture_ri itself -- the same tensor -- so every objective
        behaves exactly as before.
        """
        if self.prior is None and self.spec_init_mode in {"noise", "gaussian", "random"} and init_z is not None:
            return init_z
        if self.prior is None:
            return mixture_ri
        with torch.no_grad():
            anchor = self.prior._separate_spec_target_only(
                mixture, face=face, body=body, num_steps=1
            )["target_stft_ri"]
        anchor = anchor.detach().to(device=mixture_ri.device, dtype=mixture_ri.dtype)
        if anchor.shape != mixture_ri.shape:
            raise RuntimeError(
                f"prior anchor shape {tuple(anchor.shape)} does not match the mixture "
                f"STFT {tuple(mixture_ri.shape)}; the prior and the flow must share "
                "one stft config"
            )
        # Training-only regulariser.  The prior is better on training data than
        # on held-out data, so a refiner trained on clean training-time anchors
        # meets systematically worse ones at test; jittering the anchor narrows
        # that gap.  Inference always uses the clean estimate.
        if self.training and self.prior_anchor_noise_std > 0.0:
            anchor = anchor + torch.randn_like(anchor) * self.prior_anchor_noise_std
        return anchor

    @staticmethod
    def _math_attention_ctx():
        """Force SDPA onto the math kernel, which supports forward-mode AD.

        nn.MultiheadAttention dispatches to _scaled_dot_product_flash_attention,
        which has no forward-AD rule, so torch.func.jvp raises
        NotImplementedError inside the bottleneck cross-attention.  The math
        backend computes the same function with plain matmul+softmax and is
        forward-AD capable; the numerics are equivalent, only the kernel differs.

        Deliberately scoped to the dual pass alone: ordinary training and the
        AlphaFlow arm keep the fused kernel and are not affected.

        torch>=2.3 spells this torch.nn.attention.sdpa_kernel; 2.2 only has
        torch.backends.cuda.sdp_kernel, which despite the `cuda` namespace is
        consulted by the CPU dispatcher as well.  Both are supported so the
        cluster's torch version does not have to match the development one.
        """
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.MATH)
        except ImportError:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )

    def _training_loss_spec_target_only_meanflow(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Standalone noise-anchored MeanFlow for audiovisual separation.

        Transport:  z_t = (1-t) * eps + t * S,   eps ~ N(0, noise_scale^2 I)
        Velocity:   v = S - eps                  (conditional, per sampled pair)
        Network:    u_theta(z_t, t, r, M, V)     mean velocity over [t, r]
        Identity:   u_tgt = v + (r - t) * (v . grad_z u + d_t u)
        Inference:  S_hat = eps + u_theta(eps, t=0, r=1, M, V)

        Why this arm exists.  On the mixture-anchored path the pairing is
        deterministic given (z_t, M): S = M + (z_t - M)/t, so the true mean
        velocity is the constant S-M and the correction term cancels along the
        trajectory (v.grad_z u* = (S-M)/t and d_t u* = -(S-M)/t).  Anchoring at
        noise destroys that: many (eps, S) pairs reach the same z_t, the target
        becomes a genuine marginal E[S-eps | z_t], and the correction is
        load-bearing.  This is therefore the controlled ablation of the paper's
        central claim rather than a reformulation of it.

        The orientation matches the rest of the repo -- t=0 is the anchor, t=1
        the target, r>t -- so `interval_end` and the one-NFE inference form are
        shared with the AlphaFlow arm unchanged.
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_tokens = captured.get("video_tokens", None)
        visual_activity = captured.get("visual_activity", None)
        reliability_target = captured.get("visual_reliability_target", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)
        mixture_ri = complex_to_ri(mix_spec)
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)

        # Same draw the sampler uses at inference, so train and test anchors
        # come from one distribution.
        eps = torch.randn_like(y) * float(self.flow_cfg.noise_scale)
        v_target = y - eps

        # Identical branch split and (t,r) distribution to the AlphaFlow arm.
        n_fm = self._alphaflow_v2_fm_count(batch_size, y.device)
        permutation = torch.randperm(batch_size, device=y.device)
        fm_index = permutation[:n_fm]
        interval_index = permutation[n_fm:]
        fm_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)
        fm_mask[fm_index] = True
        interval_mask = ~fm_mask

        t = torch.empty(batch_size, device=y.device, dtype=y.dtype)
        r = torch.empty_like(t)
        large_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)
        deploy_mask = torch.zeros(batch_size, device=y.device, dtype=torch.bool)

        if n_fm > 0:
            t_fm = torch.rand(n_fm, device=y.device, dtype=y.dtype)
            t[fm_index] = t_fm
            r[fm_index] = t_fm

        n_interval = batch_size - n_fm
        if n_interval > 0:
            t_i, r_i, large_i, deploy_i = self._sample_alphaflow_interval_v3(
                n_interval, y.device, y.dtype
            )
            t[interval_index] = t_i
            r[interval_index] = r_i
            large_mask[interval_index] = large_i
            deploy_mask[interval_index] = deploy_i

        z_t = (1.0 - t.view(shape)) * eps + t.view(shape) * y
        span = (r - t).view(shape)

        def _u(z: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
            # `r` is closed over, so the derivative below is taken at FIXED r.
            # The head is parameterised on (t, delta=r-t), so differentiating
            # through `delta` supplies the -1 tangent automatically; no
            # reparameterisation is needed to get d/dt at constant r.
            return self.spec_velocity(
                z,
                tt,
                mixture_ri,
                cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
                cross_attention_tokens=visual_tokens,
                interval_end=r,
            )

        if self.meanflow_correction == "jvp":
            # jvp returns the primal and the directional derivative from one
            # dual pass, so the student prediction costs no extra forward.  The
            # primal keeps its graph, so parameter gradients still flow.
            #
            # bf16 makes a directional derivative unreliable -- it is a
            # difference of nearly equal quantities -- so the dual pass is taken
            # in fp32 with autocast off, the same reasoning as the reliability
            # BCE below.  This is the real compute cost of the noise anchor.
            with self._math_attention_ctx():
                if self.meanflow_jvp_fp32:
                    with torch.autocast(device_type=z_t.device.type, enabled=False):
                        student, dudt = torch.func.jvp(
                            _u, (z_t, t), (v_target, torch.ones_like(t))
                        )
                else:
                    student, dudt = torch.func.jvp(
                        _u, (z_t, t), (v_target, torch.ones_like(t))
                    )
            u_target = (v_target + span * dudt).detach()
        else:
            # Control arm: average-velocity regression with no correction.  On
            # the noise path this target is biased, and the gap to the jvp arm
            # is the measurement that shows the correction is load-bearing.
            student = _u(z_t, t)
            dudt = torch.zeros_like(student)
            u_target = v_target.detach()

        residual = student - u_target
        zero = torch.zeros((), device=y.device, dtype=y.dtype)
        if n_fm > 0:
            # t==r, so span==0 and the target is exactly v: plain flow matching.
            loss_fm = self._alphaflow_adaptive_loss(residual[fm_mask])
        else:
            loss_fm = zero
        if n_interval > 0:
            loss_mf = self._alphaflow_adaptive_loss(residual[interval_mask])
        else:
            loss_mf = zero
        loss = self.lambda_fm * loss_fm + self.lambda_meanflow * loss_mf

        # Endpoint and MR-STFT anchors are available but OFF by default for this
        # objective, and the config ships them at 0.0.  Reason: both regress the
        # prediction toward the *conditional* velocity (z_r_hat - z_r_true
        # reduces to span*(student - v) on this linear path, exactly as on the
        # mixture path), which is precisely the quantity the correction term is
        # supposed to move away from.  Enabling them re-imposes the anchored
        # target and quietly turns this arm back into the other one.
        z_r_hat = z_t + span * student
        z_r_true = (1.0 - r.view(shape)) * eps + r.view(shape) * y
        loss_endpoint = zero
        if self.lambda_endpoint > 0.0:
            endpoint_mask = (
                interval_mask
                if self.alphaflow_endpoint_scope == "all_intervals"
                else deploy_mask
            )
            if endpoint_mask.any():
                loss_endpoint = F.l1_loss(z_r_hat[endpoint_mask], z_r_true[endpoint_mask])
                loss = loss + self.lambda_endpoint * loss_endpoint

        loss_mrstft = zero
        if self.lambda_mrstft > 0.0 and deploy_mask.any():
            pred_complex = ri_to_sources_complex(z_r_hat[deploy_mask], num_sources=1)[:, 0]
            true_complex = ri_to_sources_complex(z_r_true[deploy_mask], num_sources=1)[:, 0]
            pred_wave = istft_waveform(pred_complex, self.stft_cfg, length=mixture.shape[-1])
            true_wave = istft_waveform(true_complex, self.stft_cfg, length=mixture.shape[-1])
            loss_mrstft = multi_resolution_stft_loss(pred_wave, true_wave)
            loss = loss + self.lambda_mrstft * loss_mrstft

        # Visual reliability supervision is a property of the conditioner, not
        # of the transport, so it is kept identical to the AlphaFlow arm.
        loss_visual_reliability = zero
        if (
            self.lambda_visual_reliability > 0.0
            and visual_activity is not None
            and reliability_target is not None
        ):
            pred_rel = visual_activity
            if pred_rel.ndim == 2:
                pred_rel = pred_rel.unsqueeze(-1)
            if pred_rel.shape[-1] != 1:
                pred_rel = pred_rel[..., -1:]
            target_rel = reliability_target.to(device=pred_rel.device, dtype=pred_rel.dtype)
            if target_rel.shape[1] != pred_rel.shape[1]:
                target_rel = F.interpolate(
                    target_rel.transpose(1, 2),
                    size=pred_rel.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            with torch.autocast(device_type=pred_rel.device.type, enabled=False):
                loss_visual_reliability = F.binary_cross_entropy(
                    pred_rel.float().clamp(1e-5, 1.0 - 1e-5),
                    target_rel.float(),
                )
            loss = loss + self.lambda_visual_reliability * loss_visual_reliability

        # Headline diagnostic: how large the MeanFlow correction actually is,
        # as a fraction of the conditional velocity it corrects.  On the
        # mixture-anchored path this quantity is ~0 by construction; here it is
        # the empirical answer to "is the identity doing any work?".
        correction_rel = zero
        if self.meanflow_correction == "jvp":
            corr = (span * dudt).detach()
            correction_rel = (
                corr.flatten(1).norm(dim=1)
                / v_target.detach().flatten(1).norm(dim=1).clamp_min(1e-8)
            ).mean()

        per_sample_mse = self._per_sample_mse(residual)
        mse_fm = per_sample_mse[fm_mask].mean() if n_fm > 0 else zero
        mse_interval = per_sample_mse[interval_mask].mean() if n_interval > 0 else zero
        interval_span = (
            (r[interval_mask] - t[interval_mask]).mean() if n_interval > 0 else zero
        )
        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_meanflow": loss_mf.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            "loss_mrstft": loss_mrstft.detach(),
            "loss_visual_reliability": loss_visual_reliability.detach(),
            "meanflow_correction_rel": correction_rel.detach(),
            "meanflow_fm_fraction": fm_mask.float().mean().detach(),
            "meanflow_deployment_fraction": deploy_mask.float().mean().detach(),
            "meanflow_large_span": large_mask.float().mean().detach(),
            "meanflow_span": (r - t).mean().detach(),
            "meanflow_interval_span": interval_span.detach(),
            "meanflow_mse_fm": mse_fm.detach(),
            "meanflow_mse_interval": mse_interval.detach(),
        }

    def _training_loss_spec_target_only_hybrid_flow(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Hybrid flow matching: 50% endpoint (t=0), 50% uniform t ∈ [0,1].

        This combines the strong one-step endpoint performance of drift_only with
        genuine flow matching by learning the velocity field across the entire trajectory.

        Training:
          - 50% batches: t=0 (pure drift correction, loss_recon weight dominates)
          - 50% batches: t~U[0,1] (flow matching, loss_fm weight on velocity matching)

        Inference (one-step):
          Ŝ = M + v_θ(M, 0, M, V)

        Inference (multi-step, if desired):
          x_0 = M
          for k in range(N):
              x_{k+1} = x_k + (1/N) * v_θ(x_k, k/N, M, V)
        """
        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)       # [B,2,F,T]
        mixture_ri = complex_to_ri(mix_spec) # [B,2,F,T]

        z = self._target_only_initial_state(mixture_ri)
        batch_size = y.shape[0]

        # Hybrid sampling: 50% endpoint, 50% uniform trajectory
        use_endpoint = torch.rand(batch_size, device=y.device) < self.hybrid_endpoint_prob
        t = torch.where(
            use_endpoint,
            torch.zeros(batch_size, device=y.device, dtype=y.dtype),
            torch.rand(batch_size, device=y.device, dtype=y.dtype)
        )

        shape = (batch_size,) + (1,) * (y.ndim - 1)
        x_t = (1.0 - t.view(shape)) * z + t.view(shape) * y
        v_target = y - z

        head_out = self.spec_velocity(
            x_t,
            t,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
        )

        drift_pred, drift_alpha = self._apply_adaptive_drift(head_out, cond)
        x1_hat = x_t + (1.0 - t.view(shape)) * drift_pred

        # Endpoint reconstruction loss (all samples, but t-weighted by the interpolation)
        loss_recon = F.l1_loss(x1_hat, y)

        # Velocity matching loss (all samples, including t=0, to force real separation)
        loss_fm = F.l1_loss(drift_pred, v_target)

        loss_drift = loss_recon  # For backward-compatible logging
        loss_endpoint_boundary = loss_recon
        loss_drift_alpha = (
            drift_alpha.mean()
            if drift_alpha is not None
            else y.new_tensor(1.0)
        )

        # Target-only mode defines the residual deterministically.
        loss_cons = torch.zeros((), device=y.device, dtype=y.dtype)

        loss_target_anchor = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_rms = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_wave = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_gain = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_interferer_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_energy_ceiling = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_recon = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_mrstft = torch.zeros((), device=y.device, dtype=y.dtype)

        need_wave_endpoint = (
            self.lambda_target_anchor > 0.0
            or self.lambda_rms > 0.0
            or self.lambda_wave > 0.0
            or self.lambda_target_gain > 0.0
            or self.lambda_residual_leak > 0.0
            or self.lambda_interferer_leak > 0.0
            or self.lambda_target_energy_ceiling > 0.0
            or self.lambda_residual_recon > 0.0
            or self.lambda_mrstft > 0.0
        )
        if need_wave_endpoint:
            pred_target_complex = ri_to_sources_complex(x1_hat, num_sources=1)[:, 0]
            pred_target_wave = istft_waveform(pred_target_complex, self.stft_cfg, length=target.shape[-1])
            pred_residual_wave = mixture - pred_target_wave

            if self.lambda_mrstft > 0.0:
                loss_mrstft = multi_resolution_stft_loss(pred_target_wave, target)

            if self.lambda_target_gain > 0.0 or self.lambda_residual_leak > 0.0:
                target_energy = target.pow(2).sum(dim=-1) + 1e-8
                target_gain = (pred_target_wave * target).sum(dim=-1) / target_energy
                residual_gain = (pred_residual_wave * target).sum(dim=-1) / target_energy
                loss_target_gain = torch.relu(self.target_gain_floor - target_gain).pow(2).mean()
                loss_residual_leak = residual_gain.abs().mean()

            if self.lambda_residual_recon > 0.0:
                ref_residual_wave = mixture - target
                loss_residual_recon = F.l1_loss(pred_residual_wave, ref_residual_wave)

            if self.lambda_interferer_leak > 0.0:
                ref_residual = mixture - target
                ref_residual_energy = ref_residual.pow(2).sum(dim=-1) + 1e-8
                interferer_gain = (pred_target_wave * ref_residual).sum(dim=-1) / ref_residual_energy
                loss_interferer_leak = interferer_gain.abs().mean()

            if self.lambda_target_energy_ceiling > 0.0:
                pred_rms_for_ceiling = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms_for_ceiling = target.pow(2).mean(dim=-1).sqrt()
                loss_target_energy_ceiling = torch.relu(
                    pred_rms_for_ceiling - self.target_energy_ceiling_ratio * target_rms_for_ceiling
                ).mean()

            if self.lambda_target_anchor > 0.0:
                sisdr_t = si_sdr_score(pred_target_wave, target)
                sisdr_r = si_sdr_score(pred_residual_wave, target)
                loss_target_anchor = torch.relu(
                    sisdr_r - sisdr_t + self.target_anchor_margin
                ).mean()

            if self.lambda_rms > 0.0:
                pred_rms = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms = target.pow(2).mean(dim=-1).sqrt()
                loss_rms = (pred_rms - target_rms).abs().mean()

            if self.lambda_wave > 0.0:
                loss_wave = F.l1_loss(pred_target_wave, target)

        # Main loss: endpoint reconstruction + velocity matching on non-endpoint samples
        loss = (
            self.lambda_drift * loss_recon
            + self.lambda_fm * loss_fm
            + self.lambda_recon * loss_recon
            + self.lambda_consistency * loss_cons
            + self.lambda_target_anchor * loss_target_anchor
            + self.lambda_rms * loss_rms
            + self.lambda_wave * loss_wave
            + self.lambda_target_gain * loss_target_gain
            + self.lambda_residual_leak * loss_residual_leak
            + self.lambda_interferer_leak * loss_interferer_leak
            + self.lambda_target_energy_ceiling * loss_target_energy_ceiling
            + self.lambda_residual_recon * loss_residual_recon
            + self.lambda_mrstft * loss_mrstft
        )
        return {
            "loss": loss,
            "loss_drift": loss_drift.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_endpoint_boundary": loss_endpoint_boundary.detach(),
            "loss_drift_alpha": loss_drift_alpha.detach(),
            "loss_consistency": loss_cons.detach(),
            "loss_target_anchor": loss_target_anchor.detach(),
            "loss_rms": loss_rms.detach(),
            "loss_wave": loss_wave.detach(),
            "loss_target_gain": loss_target_gain.detach(),
            "loss_residual_leak": loss_residual_leak.detach(),
            "loss_interferer_leak": loss_interferer_leak.detach(),
            "loss_target_energy_ceiling": loss_target_energy_ceiling.detach(),
            "loss_residual_reconstruction": loss_residual_recon.detach(),
            "loss_mrstft": loss_mrstft.detach(),
        }

    def _training_loss_spec_target_only(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """DAVIS-style target-only conditional flow loss.

        Flow state: target complex STFT only, [B,2,F,T].
        Conditioning: mixture STFT + temporal visual/AV tokens.
        Start state: configured by flow.init_mode; default for this repo variant
        is 0.5 * mixture STFT.
        """
        if self.spec_objective == "direct_unet":
            return self._training_loss_spec_target_only_direct(batch)
        if self.spec_objective in {"drift_only", "mask_drift"}:
            return self._training_loss_spec_target_only_drift(batch)
        if self.spec_objective == "residual_flow":
            return self._training_loss_spec_target_only_residual_flow(batch)
        if self.spec_objective == "alphaflow":
            return self._training_loss_spec_target_only_alphaflow(batch)
        if self.spec_objective == "flowmap_adapter":
            return self._training_loss_spec_target_only_flowmap_adapter(batch)
        if self.spec_objective == "visual_floss":
            return self._training_loss_spec_target_only_visual_floss(batch)
        if self.spec_objective == "meanflow":
            return self._training_loss_spec_target_only_meanflow(batch)
        if self.spec_objective == "hybrid_flow":
            return self._training_loss_spec_target_only_hybrid_flow(batch)

        mixture, target, b = self._prep_waveforms(batch)
        captured = self.condition(mixture, b["face"], b["body"])
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        target_spec = stft_waveform(target, self.stft_cfg)
        y = complex_to_ri(target_spec)      # [B,2,F,T]
        mixture_ri = complex_to_ri(mix_spec) # [B,2,F,T]

        z = self._target_only_initial_state(mixture_ri)
        batch_size = y.shape[0]
        t = torch.rand(batch_size, device=y.device, dtype=y.dtype)
        shape = (batch_size,) + (1,) * (y.ndim - 1)
        x_t = (1.0 - t.view(shape)) * z + t.view(shape) * y
        v_target = y - z

        v_pred = self.spec_velocity(
            x_t,
            t,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
        )

        captured = self.conditioner(mixture, b["face"], b["body"])

        cond = captured.get("conditioning", None)
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        if not hasattr(self, "_debug_flow_inputs_count"):
            self._debug_flow_inputs_count = 0

        if self._debug_flow_inputs_count < 5:
            print("=== FLOW INPUT DEBUG ===")
            print("global_condition_type:", captured.get("global_condition_type"))
            print("temporal_token_type:", captured.get("temporal_token_type"))

            print("cond:", None if cond is None else (
                cond.shape,
                cond.mean().item(),
                cond.std().item(),
                cond.min().item(),
                cond.max().item(),
            ))

            print("temporal_tokens:", None if temporal_tokens is None else (
                temporal_tokens.shape,
                temporal_tokens.mean().item(),
                temporal_tokens.std().item(),
                temporal_tokens.min().item(),
                temporal_tokens.max().item(),
            ))

            if temporal_tokens is not None:
                if temporal_tokens.shape[1] > 1:
                    print(
                        "temporal token temporal diff:",
                        (temporal_tokens[:, 1:] - temporal_tokens[:, :-1])
                        .abs()
                        .mean()
                        .item(),
                    )
                else:
                    print("temporal token temporal diff: skipped (sequence length < 2)")

                if temporal_tokens.shape[0] > 1:
                    print(
                        "temporal token batch diff:",
                        (temporal_tokens[0] - temporal_tokens[1])
                        .abs()
                        .mean()
                        .item(),
                    )
                else:
                    print("temporal token batch diff: skipped (batch size < 2)")

            for name in ["audio_tokens", "video_tokens", "av_tokens"]:
                tok = captured.get(name)
                if tok is not None and temporal_tokens is not None and tok.shape == temporal_tokens.shape:
                    print(f"diff temporal vs {name}:",
                        (temporal_tokens - tok).abs().max().item())

            self._debug_flow_inputs_count += 1
            
        loss_fm = F.l1_loss(v_pred, v_target)
        x1_hat = x_t + (1.0 - t.view(shape)) * v_pred
        loss_recon = F.l1_loss(x1_hat, y)

        # =========================================================================
        # DIRECT ENDPOINT RECONSTRUCTION ANCHOR AT t=0.0
        # Forces velocity modeling from pure z (initial half-mixture) to full clean 
        # target y, keeping (1.0 - t) at exactly 1.0 to maximize gradient energy.
        t_zero = torch.zeros_like(t)
        v_pred_at_zero = self.spec_velocity(
            z,  # Input is the pure starting state sequence
            t_zero,
            mixture_ri,
            cond,
            temporal_tokens=temporal_tokens,
            visual_activity=visual_activity,
        )
        x1_hat_at_zero = z + v_pred_at_zero
        loss_endpoint_boundary = F.l1_loss(x1_hat_at_zero, y)
        # =========================================================================

        # In target-only mode, residual is defined as mixture-target. The
        # target+residual consistency is therefore exact by construction.
        loss_cons = torch.zeros((), device=y.device, dtype=y.dtype)

        # Optional waveform-domain anchoring losses. These are computed from the
        # endpoint estimate x1_hat so they directly shape the separated target,
        # not only the velocity field in complex-STFT L1 space.
        loss_target_anchor = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_rms = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_wave = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_gain = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_interferer_leak = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_target_energy_ceiling = torch.zeros((), device=y.device, dtype=y.dtype)
        loss_residual_recon= torch.zeros((), device=y.device, dtype=y.dtype)
        loss_mrstft = torch.zeros((), device=y.device, dtype=y.dtype)

        need_wave_endpoint = (
            self.lambda_target_anchor > 0.0
            or self.lambda_rms > 0.0
            or self.lambda_wave > 0.0
            or self.lambda_target_gain > 0.0
            or self.lambda_residual_leak > 0.0
            or self.lambda_interferer_leak > 0.0
            or self.lambda_target_energy_ceiling > 0.0
            or self.lambda_residual_recon>0.0
            or self.lambda_mrstft > 0.0
        )
        if need_wave_endpoint:
            # x1_hat is target-only RI: [B,2,F,T]. Convert to waveform.
            pred_target_complex = ri_to_sources_complex(x1_hat, num_sources=1)[:, 0]
            pred_target_wave = istft_waveform(pred_target_complex, self.stft_cfg, length=target.shape[-1])
            pred_residual_wave = mixture - pred_target_wave
            ref_residual=mixture-target

            if self.lambda_mrstft > 0.0:
                loss_mrstft = multi_resolution_stft_loss(pred_target_wave, target)

            if self.lambda_target_gain > 0.0 or self.lambda_residual_leak > 0.0:
                # Scale-sensitive target allocation losses.  These directly
                # measure how much of the reference target is projected into the
                # predicted target and residual. Unlike SI-SDR, this cannot be
                # satisfied by a very quiet but correlated prediction.
                target_energy = target.pow(2).sum(dim=-1) + 1e-8
                target_gain = (pred_target_wave * target).sum(dim=-1) / target_energy
                residual_gain = (pred_residual_wave * target).sum(dim=-1) / target_energy
                loss_target_gain = torch.relu(self.target_gain_floor - target_gain).pow(2).mean()
                loss_residual_leak = residual_gain.abs().mean()

            if self.lambda_residual_recon>0.0:
                ref_residual_wave=mixture-target
                pred_residual_wave=mixture - pred_target_wave
                loss_residual_recon=F.l1_loss(pred_residual_wave,ref_residual_wave)

            if self.lambda_interferer_leak > 0.0:
                # Complementary source-selectivity loss: penalize non-target
                # material in the predicted target.  The reference non-target is
                # exactly mixture - target in this synthetic/remixed setting.
                # This is scale-sensitive and catches the current failure mode
                # where the target branch becomes too mixture-like.
                ref_residual = mixture - target
                ref_residual_energy = ref_residual.pow(2).sum(dim=-1) + 1e-8
                interferer_gain = (pred_target_wave * ref_residual).sum(dim=-1) / ref_residual_energy
                loss_interferer_leak = interferer_gain.abs().mean()

            if self.lambda_target_energy_ceiling > 0.0:
                pred_rms_for_ceiling = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms_for_ceiling = target.pow(2).mean(dim=-1).sqrt()
                loss_target_energy_ceiling = torch.relu(
                    pred_rms_for_ceiling - self.target_energy_ceiling_ratio * target_rms_for_ceiling
                ).mean()

            if self.lambda_target_anchor > 0.0:
                sisdr_t = si_sdr_score(pred_target_wave, target)
                sisdr_r = si_sdr_score(pred_residual_wave, target)
                # Penalize the measured failure mode: residual is more target-like
                # than the predicted target.  Margin is in dB.
                loss_target_anchor = torch.relu(
                    sisdr_r - sisdr_t + self.target_anchor_margin
                ).mean()

            if self.lambda_rms > 0.0:
                pred_rms = pred_target_wave.pow(2).mean(dim=-1).sqrt()
                target_rms = target.pow(2).mean(dim=-1).sqrt()
                loss_rms = (pred_rms - target_rms).abs().mean()

            if self.lambda_wave > 0.0:
                loss_wave = F.l1_loss(pred_target_wave, target)

        loss = (
            self.lambda_fm * loss_fm
            + self.lambda_recon * loss_recon
            + (0.5 * loss_endpoint_boundary)  # Pins down phase boundaries at t=0
            + self.lambda_consistency * loss_cons
            + self.lambda_target_anchor * loss_target_anchor
            + self.lambda_rms * loss_rms
            + self.lambda_wave * loss_wave
            + self.lambda_target_gain * loss_target_gain
            + self.lambda_residual_leak * loss_residual_leak
            + self.lambda_interferer_leak * loss_interferer_leak
            + self.lambda_target_energy_ceiling * loss_target_energy_ceiling
            + self.lambda_residual_recon * loss_residual_recon
            + self.lambda_mrstft * loss_mrstft
        )
        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_consistency": loss_cons.detach(),
            "loss_target_anchor": loss_target_anchor.detach(),
            "loss_rms": loss_rms.detach(),
            "loss_wave": loss_wave.detach(),
            "loss_target_gain": loss_target_gain.detach(),
            "loss_residual_leak": loss_residual_leak.detach(),
            "loss_interferer_leak": loss_interferer_leak.detach(),
            "loss_target_energy_ceiling": loss_target_energy_ceiling.detach(),
            "loss_residual_reconstruction": loss_residual_recon.detach(),
            "loss_mrstft": loss_mrstft.detach(),
        }

    @torch.no_grad()
    def separate(
        self,
        mixture: torch.Tensor,
        face: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.is_spec_head:
            return self._separate_spec(mixture, face=face, body=body, num_steps=num_steps)
        return self._separate_waveform(mixture, face=face, body=body, num_steps=num_steps)

    @torch.no_grad()
    def _separate_waveform(
        self,
        mixture: torch.Tensor,
        face: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        mixture = mixture.float()
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        captured = self.condition(mixture, face, body)
        cond = captured["conditioning"]
        cfg = self.flow_cfg
        if num_steps is not None:
            cfg = FlowConfig(consistency=cfg.consistency, noise_scale=cfg.noise_scale, num_steps=int(num_steps))

        def vf(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.velocity(x, t, mixture, cond)

        sources = euler_sample(vf, mixture, torch.Size((mixture.shape[0], 2, mixture.shape[-1])), cfg, mixture.device, mixture.dtype)
        return {
            "target": sources[:, 0],
            "residual": sources[:, 1],
            "sources": sources,
            "mixture_error": sources.sum(dim=1) - mixture,
        }

    @torch.no_grad()
    def _separate_spec_target_only(
        self,
        mixture: torch.Tensor,
        face: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference for DAVIS-style target-only flow.

        Starts from z = 0.5 * mixture STFT by default, integrates the learned
        target velocity field, then defines residual as mixture - target.
        """
        mixture = mixture.float()
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        length = mixture.shape[-1]
        captured = self.condition(mixture, face, body)
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_tokens = captured.get("video_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        cfg = self.flow_cfg
        if num_steps is not None:
            cfg = FlowConfig(consistency=cfg.consistency, noise_scale=cfg.noise_scale, num_steps=int(num_steps))

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        mixture_ri = complex_to_ri(mix_spec)
        x = self._target_only_initial_state(mixture_ri)
        prior_anchor = None
        if self.prior is not None:
            # Always the clean estimate here: anchor_noise_std is a training-time
            # regulariser, and _flow_anchor only applies it in training mode.
            prior_anchor = self._flow_anchor(mixture, mixture_ri, face, body, init_z=x)
            x = prior_anchor

        if self.spec_objective == "direct_unet":
            # Direct separator: x = D_theta(0, t=0, mixture, visual).
            t = torch.zeros((x.shape[0],), device=mixture.device, dtype=mixture.dtype)
            x_in = torch.zeros_like(mixture_ri)
            x = self.spec_velocity(
                x_in,
                t,
                mixture_ri,
                cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
            )
        elif self.spec_objective == "mask_drift":
            # Masked separator: x = m * mixture + r, evaluated once.  Iterating
            # is not defined for this parameterisation because the mask is
            # always applied to the original mixture.
            t = torch.zeros((x.shape[0],), device=mixture.device, dtype=mixture.dtype)
            head_out = self.spec_velocity(
                x,
                t,
                mixture_ri,
                cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
            )
            x, _mask_mag = self._apply_mask(head_out, mixture_ri)
        elif self.spec_objective == "drift_only":
            # Drift-only sampler.
            #
            # IMPORTANT: this network is trained at exactly one operating point
            # -- input x = z (the mixture) and t = 0 -- and its output is the
            # *whole* displacement y - z, not a velocity along a path.  So the
            # Euler-style update `x += D(x, t=i/n) / n` used previously is not a
            # refinement of a learned vector field: at step 2 it queries the net
            # at t=0.5 and at an input halfway to the target, neither of which
            # the drift objective ever produced.  Whatever number that yields is
            # dominated by how the frozen time embedding extrapolates.
            #
            # `drift_step_mode` makes the choice explicit:
            #   "endpoint" (default) - re-anchor at t=0 and re-predict the full
            #       displacement from the current estimate, which is the only
            #       iteration consistent with how the model was trained:
            #           x <- x + D(x, t=0, mixture, visual)
            #       n=1 reproduces the original one-shot behaviour exactly.
            #   "euler" - the previous behaviour, kept so the earlier 2-step
            #       ablation can still be reproduced.  Only meaningful for a
            #       model trained with flow.objective=flow.
            n = max(1, int(cfg.num_steps))
            mode = str(getattr(self, "drift_step_mode", "endpoint")).lower()

            for i in range(n):
                if mode == "euler":
                    t_val = i / n
                    step = 1.0 / n
                else:
                    t_val = 0.0
                    step = 1.0
                t = torch.full(
                    (x.shape[0],),
                    t_val,
                    device=mixture.device,
                    dtype=mixture.dtype,
                )
                v_raw = self.spec_velocity(
                    x,
                    t,
                    mixture_ri,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                    # Must mirror the training call exactly: feeding attention
                    # different K/V at inference than at training would make the
                    # baseline's test-time behaviour diverge from what it learned.
                    cross_attention_tokens=(
                        visual_tokens if self.drift_visual_cross_attention else None
                    ),
                )
                v, _alpha = self._apply_adaptive_drift(v_raw, cond)
                x = x + v * step
        elif self.spec_objective == "residual_flow":
            # Mixture-anchored residual flow.  Internally r_0=0 and
            # X_k=M+r_k; the head sees X_k plus the fixed mixture anchor and
            # predicts dr/dt.  Euler integration gives
            #
            #   r_{k+1} = r_k + (1/N) v_theta(M+r_k, k/N, M, V)
            #   S_hat   = M + r_N.
            #
            # Keeping x as the anchored audio state avoids an unnecessary
            # conversion and makes N=1 exactly the same inference form as the
            # working drift separator: S_hat=M+v_theta(M,0,M,V).
            n = max(1, int(cfg.num_steps))
            step = 1.0 / n
            x = mixture_ri.clone()
            for i in range(n):
                t_val = i / n
                t = torch.full(
                    (x.shape[0],),
                    t_val,
                    device=mixture.device,
                    dtype=mixture.dtype,
                )
                v = self.spec_velocity(
                    x,
                    t,
                    x,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                )
                x = x + step * v
        elif self.spec_objective == "alphaflow":
            # Finite-interval mean velocity.  NFE=1 performs the intended
            # direct mixture->target jump; n>1 is retained as a diagnostic.
            # With flow.prior the jump starts from the prior's estimate, and the
            # conditioning slot matches what training used.  With init_mode='noise'
            # and no prior, the jump starts from the sampled noise, matching training.
            n = max(1, int(cfg.num_steps))
            if prior_anchor is not None:
                x = prior_anchor.clone()
            elif self.spec_init_mode not in {"noise", "gaussian", "random"}:
                # Mixture or half-mixture: override the sampled state
                x = mixture_ri.clone()
            # else: x holds the sampled noise from _target_only_initial_state, keep it
            noise_init = self.spec_init_mode in {"noise", "gaussian", "random"}
            use_mixture_slot = noise_init or (
                prior_anchor is not None and self.prior_condition_on == "mixture"
            )
            for i in range(n):
                t_val = i / n
                r_val = (i + 1) / n
                t = torch.full((x.shape[0],), t_val, device=mixture.device, dtype=mixture.dtype)
                r = torch.full((x.shape[0],), r_val, device=mixture.device, dtype=mixture.dtype)
                if self.alphaflow_variant in {
                    "v3", "deployment_v3", "faithful_v3", "av_v3", "alphaflow_v3"
                }:
                    u = self.spec_velocity(
                        x, t, mixture_ri if use_mixture_slot else x, cond,
                        temporal_tokens=temporal_tokens,
                        visual_activity=visual_activity,
                        cross_attention_tokens=visual_tokens,
                        interval_end=r,
                    )
                else:
                    u = self.spec_velocity(
                        x, t, x, cond,
                        temporal_tokens=temporal_tokens,
                        visual_activity=visual_activity,
                        interval_end=r,
                    )
                x = x + (r_val - t_val) * u
        elif self.spec_objective == "visual_floss":
            # Deterministic deployment starts at the observed mixture (zero
            # training noise).  The second head input remains that original
            # mixture at every NFE; the complementary source is reconstructed
            # below as mixture-target, preserving mixture consistency exactly.
            n = max(1, int(cfg.num_steps))
            x = mixture_ri.clone()
            step = 1.0 / n
            for i in range(n):
                t = torch.full(
                    (x.shape[0],),
                    i / n,
                    device=mixture.device,
                    dtype=mixture.dtype,
                )
                velocity = self.spec_velocity(
                    x,
                    t,
                    mixture_ri,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                    cross_attention_tokens=visual_tokens,
                )
                x = x + step * velocity
        elif self.spec_objective == "flowmap_adapter":
            # One/few-step map over an immutable drift base.  The evolving
            # state occupies the first input slot and the observed mixture is
            # fixed in the second slot at every NFE, exactly as in training.
            n = max(1, int(cfg.num_steps))
            x = mixture_ri.clone()
            for i in range(n):
                t_val = i / n
                r_val = (i + 1) / n
                t = torch.full(
                    (x.shape[0],), t_val,
                    device=mixture.device, dtype=mixture.dtype,
                )
                r = torch.full(
                    (x.shape[0],), r_val,
                    device=mixture.device, dtype=mixture.dtype,
                )
                u = self.spec_velocity(
                    x,
                    t,
                    mixture_ri,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                    cross_attention_tokens=visual_tokens,
                    interval_end=r,
                )
                x = x + (r_val - t_val) * u
        elif self.spec_objective == "meanflow":
            # Noise-anchored mean velocity.  x already holds eps because
            # flow.init_mode is required to be 'noise' for this objective, so
            # the inference anchor matches the training draw.
            #
            # NFE=1 is the intended operating point:
            #     S_hat = eps + u_theta(eps, t=0, r=1, M, V)
            # n>1 walks the same path in equal sub-intervals, as a diagnostic.
            #
            # Unlike the mixture-anchored arm this is genuinely stochastic --
            # each call draws a different eps, so repeated evaluation of one
            # clip gives different outputs.  Evaluation must therefore fix the
            # seed to be reproducible.
            n = max(1, int(cfg.num_steps))
            for i in range(n):
                t_val = i / n
                r_val = (i + 1) / n
                t = torch.full((x.shape[0],), t_val, device=mixture.device, dtype=mixture.dtype)
                r = torch.full((x.shape[0],), r_val, device=mixture.device, dtype=mixture.dtype)
                u = self.spec_velocity(
                    x,
                    t,
                    mixture_ri,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                    cross_attention_tokens=visual_tokens,
                    interval_end=r,
                )
                x = x + (r_val - t_val) * u
        elif self.spec_objective == "hybrid_flow":
            # Hybrid flow matching sampler: uses Euler stepping with learned velocity field.
            #
            # Unlike drift_only (trained at t=0 only), hybrid_flow is trained across
            # the full trajectory [0,1], so the velocity field is meaningful at all t.
            # Euler stepping naturally integrates this field:
            #   x_{k+1} = x_k + (1/n) * v_θ(x_k, k/n, mixture, visual)
            #
            # One-step (n=1): evaluates at t=0, recovering drift-like behavior
            # Multi-step (n>1): iteratively refines using the learned field
            n = max(1, int(cfg.num_steps))
            step = 1.0 / n

            for i in range(n):
                t_val = i / n
                t = torch.full(
                    (x.shape[0],),
                    t_val,
                    device=mixture.device,
                    dtype=mixture.dtype,
                )
                v_raw = self.spec_velocity(
                    x,
                    t,
                    mixture_ri,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                )
                v, _alpha = self._apply_adaptive_drift(v_raw, cond)
                x = x + v * step
        else:
            n = int(cfg.num_steps)
            for i in range(n):
                t = torch.full((x.shape[0],), i / max(n, 1), device=mixture.device, dtype=mixture.dtype)
                v = self.spec_velocity(
                    x,
                    t,
                    mixture_ri,
                    cond,
                    temporal_tokens=temporal_tokens,
                    visual_activity=visual_activity,
                )
                x = x + v / max(n, 1)

        target_complex = ri_to_sources_complex(x, num_sources=1)[:, 0]
        target = istft_waveform(target_complex, self.stft_cfg, length=length)
        # Defining residual in waveform domain guarantees output mixture consistency
        # and removes the target/residual permutation ambiguity.
        residual = mixture - target
        sources = torch.stack([target, residual], dim=1)

        residual_ri = mixture_ri - x
        sources_stft_ri = torch.cat([x, residual_ri], dim=1)
        return {
            "target": target,
            "residual": residual,
            "sources": sources,
            "mixture_error": sources.sum(dim=1) - mixture,
            "sources_stft_ri": sources_stft_ri,
            "target_stft_ri": x,
        }

    @torch.no_grad()
    def _separate_spec(
        self,
        mixture: torch.Tensor,
        face: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.spec_target_mode == "target_only":
            return self._separate_spec_target_only(mixture, face=face, body=body, num_steps=num_steps)

        mixture = mixture.float()
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        length = mixture.shape[-1]
        captured = self.condition(mixture, face, body)
        cond = captured["conditioning"]
        temporal_tokens = captured.get("temporal_tokens", None)
        visual_activity = captured.get("visual_activity", None)

        cfg = self.flow_cfg
        if num_steps is not None:
            cfg = FlowConfig(consistency=cfg.consistency, noise_scale=cfg.noise_scale, num_steps=int(num_steps))

        mix_spec = stft_waveform(mixture, self.stft_cfg)
        mixture_ri = complex_to_ri(mix_spec)
        shape = torch.Size((mixture.shape[0], 4, mixture_ri.shape[-2], mixture_ri.shape[-1]))
        x = torch.randn(shape, device=mixture.device, dtype=mixture.dtype) * cfg.noise_scale
        if cfg.consistency in {"final", "every_step"}:
            x = project_ri_sources_to_mixture(x, mixture_ri, num_sources=2)
        n = int(cfg.num_steps)
        for i in range(n):
            t = torch.full((shape[0],), i / max(n, 1), device=mixture.device, dtype=mixture.dtype)
            v = self.spec_velocity(
                x,
                t,
                mixture_ri,
                cond,
                temporal_tokens=temporal_tokens,
                visual_activity=visual_activity,
            )
            if cfg.consistency in {"final", "every_step"}:
                v = project_ri_velocity_zero_sum(v, num_sources=2)
            x = x + v / max(n, 1)
            if cfg.consistency == "every_step":
                x = project_ri_sources_to_mixture(x, mixture_ri, num_sources=2)
        if cfg.consistency in {"final", "every_step"}:
            x = project_ri_sources_to_mixture(x, mixture_ri, num_sources=2)

        sources_complex = ri_to_sources_complex(x, num_sources=2)
        target = istft_waveform(sources_complex[:, 0], self.stft_cfg, length=length)
        residual = istft_waveform(sources_complex[:, 1], self.stft_cfg, length=length)
        sources = torch.stack([target, residual], dim=1)
        if cfg.consistency in {"final", "every_step"}:
            sources = project_sources_to_mixture(sources, mixture)
        return {
            "target": sources[:, 0],
            "residual": sources[:, 1],
            "sources": sources,
            "mixture_error": sources.sum(dim=1) - mixture,
            "sources_stft_ri": x,
        }
