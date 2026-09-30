"""VIST: visually indexed, mixture-anchored source transport (paper Sec. 3)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from vist.conditioner import AVConditioner
from vist.losses import multi_resolution_stft_loss
from vist.stft import STFTConfig, complex_to_ri, istft_waveform, ri_to_complex, stft_waveform
from vist.unet import TFCTDFUNet


def _transport_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read the ``transport`` section, falling back to the pre-release layout.

    Checkpoints trained before the rename store these settings under
    ``flow.visual_floss`` with ``deployment_ratio`` / ``curriculum_steps``.
    """
    if "transport" in cfg:
        return dict(cfg["transport"])
    flow = dict(cfg.get("flow", {}))
    objective = str(flow.get("objective", "vist")).lower()
    if objective not in {"vist", "visual_floss"}:
        raise ValueError(f"Only the VIST objective is implemented, got flow.objective={objective!r}")
    for key, expected in (("target_mode", "target_only"), ("init_mode", "mixture")):
        if key in flow and str(flow[key]).lower() != expected:
            raise ValueError(f"VIST requires flow.{key}={expected!r}, got {flow[key]!r}")
    legacy = dict(flow.get("visual_floss", {}))
    out = {
        "inference_state_ratio": legacy.get("deployment_ratio", 0.5),
        "ramp_steps": legacy.get("curriculum_steps", 16000),
        "num_steps": flow.get("num_steps", 1),
    }
    for key in ("noise_scale", "time_sampling", "logit_mu", "logit_sigma", "max_t",
                "loss_eps", "db_floor", "db_ceiling"):
        if key in legacy:
            out[key] = legacy[key]
    return out


def _loss_weights(train_cfg: Dict[str, Any]) -> tuple[float, float, float]:
    """(lambda_vel, lambda_mr, lambda_rel) of Eq. (4); pre-release names accepted."""
    for key in ("lambda_deployment", "lambda_endpoint", "lambda_drift", "lambda_recon", "lambda_fm"):
        if float(train_cfg.get(key, 0.0)) != 0.0:
            raise ValueError(f"training.{key} is not part of the VIST objective; remove it or set it to 0")
    lambda_vel = float(train_cfg.get("lambda_vel", train_cfg.get("lambda_floss", 0.05)))
    lambda_mr = float(train_cfg.get("lambda_mrstft", 0.10))
    lambda_rel = float(train_cfg.get("lambda_rel", train_cfg.get("lambda_visual_reliability", 0.05)))
    return lambda_vel, lambda_mr, lambda_rel


def _head_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    head_cfg = dict(cfg.get("head", {}))
    # Keys written by pre-release configs; only their VIST values are accepted.
    head_type = str(head_cfg.pop("type", "tfc_tdf_unet")).lower()
    if head_type not in {"tfc_tdf_unet", "tfctdf_unet", "tfc_tdf"}:
        raise ValueError(f"VIST uses the TFC-TDF U-Net, got head.type={head_type!r}")
    conditioning = str(head_cfg.pop("temporal_conditioning", "multiscale_film_cross_attn")).lower()
    if conditioning not in {"multiscale_film_cross_attn", "film_cross_attn", "cross_attn_film"}:
        raise ValueError(f"VIST uses FiLM + cross-attention conditioning, got {conditioning!r}")
    for key in ("dual_head", "flowmap_adapter_blocks", "interval_embedding_reference"):
        if head_cfg.pop(key, None):
            raise ValueError(f"head.{key} is not part of VIST")
    for key in ("flowmap_adapter_channels", "flowmap_adapter_dropout"):
        head_cfg.pop(key, None)
    return head_cfg


class VIST(nn.Module):
    """Single velocity network trained on mixture-anchored target paths.

    Training (Eqs. 1-6): along ``z_t = (1 - t)(M + n) + t S`` with constant
    velocity ``v* = S - M - n``, half of each batch is placed at the inference
    state ``(n, t) = (0, 0)``.  The loss is

        L = lambda_vel L_vel + lambda_mr L_MR + lambda_rel L_rel.

    Inference (Eq. 3): ``S_hat = M + u(M, 0 | M, V)`` and ``B_hat = M - S_hat``.
    The complement is never predicted, so mixture consistency is exact.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = dict(cfg.get("model", {}))
        for key, expected in (
            ("gate_type", "film"),
            ("global_condition_type", "fused"),
            ("temporal_token_type", "fused"),
            ("raw_visual_map", False),
        ):
            if key in model_cfg and model_cfg[key] != expected:
                raise ValueError(f"VIST requires model.{key}={expected!r}, got {model_cfg[key]!r}")
        cond_dim = int(model_cfg.get("cond_dim", 512))
        self.conditioner = AVConditioner(
            dict(cfg.get("backbone", {})),
            cond_dim=cond_dim,
            audio_condition_dropout=float(model_cfg.get("audio_condition_dropout", 0.0)),
            visual_token_dropout=float(model_cfg.get("visual_token_dropout", 0.10)),
            visual_token_noise_std=float(model_cfg.get("visual_token_noise_std", 0.0)),
        )
        # Input [z_t; M] and output v are real/imag STFT channels.
        self.head = TFCTDFUNet(cond_dim=cond_dim, in_channels=4, out_channels=2, **_head_cfg(cfg))
        self.stft_cfg = STFTConfig.from_dict(cfg.get("stft", {}))

        tr = _transport_cfg(cfg)
        self.inference_state_ratio = float(tr.get("inference_state_ratio", 0.5))
        self.noise_scale = float(tr.get("noise_scale", 0.10))
        self.ramp_steps = int(tr.get("ramp_steps", 16000))
        self.time_sampling = str(tr.get("time_sampling", "logit_normal")).lower()
        self.logit_mu = float(tr.get("logit_mu", -0.4))
        self.logit_sigma = float(tr.get("logit_sigma", 1.0))
        self.max_t = float(tr.get("max_t", 0.95))
        self.loss_eps = float(tr.get("loss_eps", 1e-7))
        self.db_floor = float(tr.get("db_floor", -20.0))
        self.db_ceiling = float(tr.get("db_ceiling", 30.0))
        self.num_steps = int(tr.get("num_steps", 1))
        if not 0.0 <= self.inference_state_ratio <= 1.0:
            raise ValueError("transport.inference_state_ratio must be in [0,1]")
        if self.noise_scale < 0.0 or self.ramp_steps < 0:
            raise ValueError("transport.noise_scale and transport.ramp_steps must be >= 0")
        if self.time_sampling not in {"uniform", "logit_normal"}:
            raise ValueError("transport.time_sampling must be 'uniform' or 'logit_normal'")
        if not 0.0 < self.max_t < 1.0:
            raise ValueError("transport.max_t must be in (0,1)")
        if self.loss_eps <= 0.0 or self.db_floor >= self.db_ceiling:
            raise ValueError("need transport.loss_eps > 0 and db_floor < db_ceiling")

        self.lambda_vel, self.lambda_mr, self.lambda_rel = _loss_weights(dict(cfg.get("training", {})))
        self._training_progress_step = 0

    def set_training_progress(self, step: int, total_steps: Optional[int] = None) -> None:
        """Optimizer step count, used to ramp up the noise level and t."""
        self._training_progress_step = max(0, int(step))

    def velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mixture_ri: torch.Tensor,
        captured: Dict[str, Any],
    ) -> torch.Tensor:
        """u_theta(z_t, t | M, V)."""
        return self.head(
            z_t,
            mixture_ri,
            captured["conditioning"],
            t,
            temporal_tokens=captured["temporal_tokens"],
            visual_activity=captured["visual_activity"],
            cross_attention_tokens=captured["video_tokens"],
        )

    @staticmethod
    def _prep_waveforms(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mixture = batch["mixture"].float()
        target = batch["target"].float()
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        length = min(mixture.shape[-1], target.shape[-1])
        return mixture[..., :length], target[..., :length]

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mixture, target = self._prep_waveforms(batch)
        captured = self.conditioner(mixture, face=batch.get("face"))

        mixture_ri = complex_to_ri(stft_waveform(mixture, self.stft_cfg))
        y = complex_to_ri(stft_waveform(target, self.stft_cfg))
        batch_size = y.shape[0]
        shape = (batch_size,) + (1,) * (y.ndim - 1)

        # Noise level and t are ramped up from zero over the first updates.
        ramp = min(1.0, self._training_progress_step / self.ramp_steps) if self.ramp_steps > 0 else 1.0

        # n: STFT of white noise with std noise_scale x mixture RMS (per example),
        # so every perturbed state is the STFT of a real signal.
        mixture_rms = mixture.square().mean(dim=-1, keepdim=True).sqrt()
        noise_wave = torch.randn_like(mixture) * mixture_rms * self.noise_scale * ramp
        noise_ri = complex_to_ri(stft_waveform(noise_wave, self.stft_cfg))

        # A fixed share of every batch sits at the inference state (n, t) = (0, 0).
        inference_count = max(0, min(batch_size, int(round(batch_size * self.inference_state_ratio))))
        at_inference = torch.zeros(batch_size, device=y.device, dtype=torch.bool)
        if inference_count:
            at_inference[torch.randperm(batch_size, device=y.device)[:inference_count]] = True
            noise_ri = noise_ri.clone()
            noise_ri[at_inference] = 0

        if self.time_sampling == "logit_normal":
            logits = torch.randn(batch_size, device=y.device, dtype=y.dtype) * self.logit_sigma + self.logit_mu
            t = torch.sigmoid(logits) * self.max_t * ramp
        else:
            t = torch.rand(batch_size, device=y.device, dtype=y.dtype) * self.max_t * ramp
        t = t.masked_fill(at_inference, 0.0)

        # Eqs. (1)-(2): z_t = z_0 + t v*,  z_0 = M + n,  v* = S - z_0.
        z0 = mixture_ri + noise_ri
        v_target = y - z0
        z_t = z0 + t.view(shape) * v_target
        v_pred = self.velocity(z_t, t, mixture_ri, captured)

        # Eqs. (5)-(6), in fp32 outside autocast.
        with torch.autocast(device_type=y.device.type, enabled=False):
            error_energy = (v_pred.float() - v_target.float()).square().flatten(1).mean(1)
            target_energy = v_target.float().square().flatten(1).mean(1)
            relative_mse = (error_energy + self.loss_eps) / (target_energy + self.loss_eps)
            per_sample_db = 10.0 * torch.log10(relative_mse.clamp_min(self.loss_eps))
            clipped_db = per_sample_db.clamp(min=self.db_floor, max=self.db_ceiling)
            loss_vel_db = clipped_db.mean()
            # Offsetting by the floor leaves the gradient unchanged and keeps the
            # logged total non-negative.
            loss_vel = (clipped_db - self.db_floor).mean()

        # L_MR on the waveform of the one-step estimate, from a second pass at
        # the inference state for the whole batch.
        t0 = torch.zeros(batch_size, device=y.device, dtype=y.dtype)
        s_hat = mixture_ri + self.velocity(mixture_ri, t0, mixture_ri, captured)
        s_hat_wave = istft_waveform(ri_to_complex(s_hat), self.stft_cfg, length=mixture.shape[-1])
        loss_mr = multi_resolution_stft_loss(s_hat_wave, target)

        # L_rel: binary cross-entropy training r to detect dropped visual frames.
        pred_rel = captured["visual_activity"]
        target_rel = captured["visual_reliability_target"].to(device=pred_rel.device, dtype=pred_rel.dtype)
        with torch.autocast(device_type=pred_rel.device.type, enabled=False):
            loss_rel = F.binary_cross_entropy(pred_rel.float().clamp(1e-5, 1.0 - 1e-5), target_rel.float())

        loss = self.lambda_vel * loss_vel + self.lambda_mr * loss_mr + self.lambda_rel * loss_rel
        return {
            "loss": loss,
            "loss_vel": loss_vel.detach(),
            "loss_vel_db": loss_vel_db.detach(),
            "loss_mr": loss_mr.detach(),
            "loss_rel": loss_rel.detach(),
            "ramp": torch.tensor(ramp, device=y.device, dtype=y.dtype),
            "t_mean": t.mean().detach(),
            "noise_rms": noise_wave.square().mean().sqrt().detach(),
        }

    @torch.no_grad()
    def separate(
        self,
        mixture: torch.Tensor,
        face: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Estimate the target with ``num_steps`` Euler steps from the mixture.

        ``num_steps=1`` (the default) is Eq. (3).  The mixture stays in the
        conditioning slot at every step.
        """
        mixture = mixture.float()
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        length = mixture.shape[-1]
        captured = self.conditioner(mixture, face=face)
        mixture_ri = complex_to_ri(stft_waveform(mixture, self.stft_cfg))

        n = max(1, int(self.num_steps if num_steps is None else num_steps))
        step = 1.0 / n
        z = mixture_ri.clone()
        for i in range(n):
            t = torch.full((z.shape[0],), i / n, device=mixture.device, dtype=mixture.dtype)
            z = z + step * self.velocity(z, t, mixture_ri, captured)

        target = istft_waveform(ri_to_complex(z), self.stft_cfg, length=length)
        return {"target": target, "residual": mixture - target, "target_stft_ri": z}
