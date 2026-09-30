"""Tests for the VIST objective, sampler and released configuration."""

from __future__ import annotations

import copy
from pathlib import Path
import types

import pytest
import torch
import yaml

from vist import VIST


ROOT = Path(__file__).parents[1]


def _released_cfg():
    return yaml.safe_load((ROOT / "configs/vist.yaml").read_text())


def _small_cfg():
    """The released config with a small U-Net and STFT, for fast CPU tests."""
    cfg = _released_cfg()
    cfg["stft"] = {"n_fft": 256, "hop_length": 64, "win_length": 256, "center": True}
    cfg["head"].update(
        model_channels=8, blocks_per_level=1, embedding_dim=64, time_dim=32,
        input_freq_bins=129, norm_groups=4, temporal_num_heads=2,
    )
    cfg["transport"]["ramp_steps"] = 10
    return cfg


def _batch(length: int = 4096, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "mixture": torch.randn(2, length, generator=generator) * 0.1,
        "target": torch.randn(2, length, generator=generator) * 0.1,
        "face": torch.randn(2, 25, 2, 68, generator=generator),
    }


def _built(cfg):
    model = VIST(cfg)
    model.training_loss(_batch())  # materialise the lazy projections
    return model


def test_released_config_matches_paper():
    cfg = _released_cfg()
    training, transport, head = cfg["training"], cfg["transport"], cfg["head"]
    # Eq. (4)
    assert (training["lambda_vel"], training["lambda_mrstft"], training["lambda_rel"]) == (0.05, 0.10, 0.05)
    # Secs. 3.2-3.3
    assert transport["inference_state_ratio"] == 0.5
    assert transport["noise_scale"] == 0.1
    assert (transport["logit_mu"], transport["logit_sigma"], transport["max_t"]) == (-0.4, 1.0, 0.95)
    assert transport["ramp_steps"] == 16000
    assert (transport["db_floor"], transport["db_ceiling"]) == (-20.0, 30.0)
    assert transport["num_steps"] == 1
    # Sec. 3.4
    assert [head["model_channels"] * m for m in head["channel_mult"]] == [60, 120, 180, 240]
    assert head["blocks_per_level"] == 2 and head["visual_reliability_floor"] == 0.25
    assert cfg["model"]["visual_token_dropout"] == 0.10
    # Sec. 4.4
    assert training["lr"] == 3e-5 and training["warmup_steps"] == 12000 and training["min_lr"] == 1e-6
    assert training["batch_size"] * training["grad_accum_steps"] == 8
    assert training["ema_decay"] == 0.999 and training["best_metric"] == "val_si_sdr"


def test_released_model_has_paper_parameter_count():
    torch.manual_seed(0)
    model = VIST(_released_cfg())
    batch = {
        "mixture": torch.randn(1, 16384) * 0.1,
        "target": torch.randn(1, 16384) * 0.1,
        "face": torch.randn(1, 25, 2, 68),
    }
    with torch.no_grad():
        model.training_loss(batch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert round(n_params / 1e6, 1) == 23.2


def test_one_step_inference_is_eq3():
    model = _built(_small_cfg()).eval()
    batch = _batch(seed=3)
    with torch.no_grad():
        out = model.separate(batch["mixture"], face=batch["face"], num_steps=1)
        from vist.stft import complex_to_ri, stft_waveform

        mixture_ri = complex_to_ri(stft_waveform(batch["mixture"], model.stft_cfg))
        captured = model.conditioner(batch["mixture"], face=batch["face"])
        t0 = torch.zeros(2)
        expected = mixture_ri + model.velocity(mixture_ri, t0, mixture_ri, captured)
    assert torch.equal(out["target_stft_ri"], expected)
    assert torch.equal(out["residual"], batch["mixture"] - out["target"])


def test_loss_is_finite_and_trains_every_used_parameter():
    model = _built(_small_cfg())
    model.set_training_progress(10)
    losses = model.training_loss(_batch(seed=7))
    for key in ("loss", "loss_vel", "loss_vel_db", "loss_mr", "loss_rel"):
        assert torch.isfinite(losses[key]), key
    expected = 0.05 * losses["loss_vel"] + 0.10 * losses["loss_mr"] + 0.05 * losses["loss_rel"]
    assert torch.allclose(losses["loss"].detach(), expected)
    losses["loss"].backward()
    unused = {"video_gate", "video_token_gate", "temporal_token_norm", "temporal_fuse"}
    for name, param in model.named_parameters():
        if name.split(".")[1] in unused:
            continue
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_ramp_starts_at_the_inference_state():
    model = _built(_small_cfg())
    model.set_training_progress(0)
    losses = model.training_loss(_batch(seed=8))
    assert losses["ramp"].item() == 0.0
    assert losses["noise_rms"].item() == 0.0
    assert losses["t_mean"].item() == 0.0


def test_multistep_keeps_the_mixture_in_the_conditioning_slot():
    model = _built(_small_cfg()).eval()
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


def test_pre_release_config_layout_builds_the_same_model():
    """Checkpoints trained before the rename store the old config layout."""
    new = _small_cfg()
    old = copy.deepcopy(new)
    transport = old.pop("transport")
    old["flow"] = {
        "target_mode": "target_only",
        "init_mode": "mixture",
        "objective": "visual_floss",
        "num_steps": transport["num_steps"],
        "visual_floss": {
            "deployment_ratio": transport["inference_state_ratio"],
            "curriculum_steps": transport["ramp_steps"],
            **{k: transport[k] for k in ("noise_scale", "time_sampling", "logit_mu", "logit_sigma",
                                         "max_t", "loss_eps", "db_floor", "db_ceiling")},
        },
    }
    training = old["training"]
    training["lambda_floss"] = training.pop("lambda_vel")
    training["lambda_visual_reliability"] = training.pop("lambda_rel")
    training["lambda_deployment"] = 0.0
    old["head"]["type"] = "tfc_tdf_unet"
    old["head"]["temporal_conditioning"] = "multiscale_film_cross_attn"
    old["model"].update(gate_type="film", global_condition_type="fused", temporal_token_type="fused")

    torch.manual_seed(0)
    a = _built(new)
    torch.manual_seed(0)
    b = _built(old)
    assert a.state_dict().keys() == b.state_dict().keys()
    b.load_state_dict(a.state_dict(), strict=True)
    assert (b.lambda_vel, b.lambda_mr, b.lambda_rel) == (a.lambda_vel, a.lambda_mr, a.lambda_rel)
    assert b.inference_state_ratio == a.inference_state_ratio and b.ramp_steps == a.ramp_steps


def test_objectives_outside_vist_are_rejected():
    cfg = _small_cfg()
    cfg["training"]["lambda_deployment"] = 1.0
    with pytest.raises(ValueError):
        VIST(cfg)
