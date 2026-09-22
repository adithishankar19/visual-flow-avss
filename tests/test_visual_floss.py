"""Regression tests for mixture-consistent Visual-FLOSS training."""

from __future__ import annotations

from pathlib import Path
import types

import torch
import yaml

from mambaflow.models import MambaVoiceMCFlow


ROOT = Path(__file__).parents[1]


def _cfg(objective: str):
    flow = {
        "target_mode": "target_only",
        "init_mode": "mixture",
        "objective": objective,
        "visual_cross_attention": True,
        "adaptive_drift": False,
        "consistency": "none",
        "num_steps": 1,
    }
    training = {
        "lambda_mrstft": 0.0,
        "lambda_visual_reliability": 0.0,
    }
    if objective == "visual_floss":
        flow["visual_floss"] = {
            "deployment_ratio": 0.5,
            "noise_scale": 0.1,
            "curriculum_steps": 10,
            "time_sampling": "logit_normal",
            "max_t": 0.95,
            "loss_eps": 1e-7,
            "db_floor": -20.0,
            "db_ceiling": 30.0,
        }
        training.update(
            lambda_deployment=1.0,
            lambda_floss=0.05,
            lambda_endpoint=0.25,
        )
    else:
        training.update(lambda_drift=1.0, lambda_recon=0.0)
    return {
        "data": {"kind": "synthetic"},
        "backbone": {
            "class_path": "mambaflow.backbones.mambavoice.DummyMambaVoiceLike",
            "init_kwargs": {},
            "audio_module": "audio_encoder",
            "video_module": "stgcn",
            "freeze": False,
            "strict": True,
        },
        "model": {
            "cond_dim": 64,
            "gate_type": "film",
            "global_condition_type": "fused",
            "temporal_token_type": "fused",
            "visual_token_dropout": 0.0,
            "visual_token_noise_std": 0.0,
        },
        "stft": {
            "n_fft": 256,
            "hop_length": 64,
            "win_length": 256,
            "center": True,
        },
        "head": {
            "type": "tfc_tdf_unet",
            "model_channels": 8,
            "channel_mult": [1, 2, 3, 4],
            "blocks_per_level": 1,
            "embedding_dim": 64,
            "time_dim": 32,
            "input_freq_bins": 129,
            "norm_groups": 4,
            "dropout": 0.0,
            "temporal_conditioning": "multiscale_film_cross_attn",
            "temporal_num_heads": 2,
            "visual_reliability_floor": 0.25,
            "exact_resample": True,
            "residual_scale": "unit",
        },
        "flow": flow,
        "training": training,
    }


def _batch(length: int = 4096, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "mixture": torch.randn(2, length, generator=generator) * 0.1,
        "target": torch.randn(2, length, generator=generator) * 0.1,
        "face": torch.randn(2, 25, 3, 68, generator=generator),
    }


def _warm_started_pair():
    torch.manual_seed(0)
    drift = MambaVoiceMCFlow(_cfg("drift_only"))
    drift.training_loss(_batch())
    drift.eval()

    torch.manual_seed(1)
    floss = MambaVoiceMCFlow(_cfg("visual_floss"))
    floss.training_loss(_batch())
    floss.load_state_dict(drift.state_dict(), strict=True)
    floss.eval()
    return drift, floss


def test_visual_floss_step_zero_is_exact_drift_nfe1():
    drift, floss = _warm_started_pair()
    batch = _batch(seed=4)
    with torch.no_grad():
        expected = drift.separate(
            batch["mixture"], face=batch["face"], num_steps=1
        )["target"]
        actual = floss.separate(
            batch["mixture"], face=batch["face"], num_steps=1
        )["target"]
    assert torch.equal(actual, expected)


def test_visual_floss_loss_is_finite_and_mixture_consistent():
    model = MambaVoiceMCFlow(_cfg("visual_floss"))
    model.set_training_progress(10, 100)
    losses = model.training_loss(_batch(seed=7))
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["loss_floss_db"])
    assert losses["visual_floss_deployment_fraction"].item() == 0.5
    # The complement is algebraically exact, but the diagnostic evaluates
    # (target + (mixture - target)) - mixture in floating point.  Allow only
    # machine-roundoff-scale error rather than requiring bitwise cancellation.
    assert losses["visual_floss_mixture_error"].item() < 1e-7
    assert losses["visual_floss_velocity_sum_error"].item() < 1e-7
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_visual_floss_curriculum_starts_at_drift_task():
    model = MambaVoiceMCFlow(_cfg("visual_floss"))
    model.set_training_progress(0, 100)
    losses = model.training_loss(_batch(seed=8))
    assert losses["visual_floss_curriculum"].item() == 0.0
    assert losses["visual_floss_noise_rms"].item() == 0.0
    assert losses["visual_floss_t_mean"].item() == 0.0


def test_visual_floss_nfe2_keeps_original_mixture_condition():
    model = MambaVoiceMCFlow(_cfg("visual_floss")).eval()
    batch = _batch(seed=11)
    records = []
    original = model.head.forward

    def wrapped(self, state, mixture, *args, **kwargs):
        records.append((state.detach().clone(), mixture.detach().clone()))
        return original(state, mixture, *args, **kwargs)

    model.head.forward = types.MethodType(wrapped, model.head)
    with torch.no_grad():
        model.separate(batch["mixture"], face=batch["face"], num_steps=2)
    assert len(records) == 2
    assert torch.equal(records[0][1], records[1][1])
    assert torch.equal(records[0][0], records[0][1])


def test_canonical_config_has_exact_released_objective():
    cfg = yaml.safe_load(
        (ROOT / "configs/visual_floss_mrstft.yaml").read_text()
    )
    training = cfg["training"]
    assert cfg["flow"]["objective"] == "visual_floss"
    assert training["require_init_from"] is False
    assert training["lambda_floss"] == 0.05
    assert training["lambda_mrstft"] == 0.10
    for key in (
        "lambda_deployment",
        "lambda_endpoint",
        "lambda_visual_reliability",
        "lambda_drift",
        "lambda_recon",
        "lambda_fm",
        "lambda_meanflow",
        "lambda_consistency",
        "lambda_base_drift",
        "lambda_delta_reg",
        "lambda_flowmap",
        "lambda_composition",
        "lambda_adapter_reg",
    ):
        assert training[key] == 0.0, key
