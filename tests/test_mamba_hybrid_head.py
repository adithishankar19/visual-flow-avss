"""Tests for the MambaVoice hybrid Mamba-Transformer velocity head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from mambaflow.models import MambaVoiceMCFlow
from mambaflow.models import mamba_hybrid_flow_head as mh

from test_visual_floss import _batch, _cfg


ROOT = Path(__file__).parents[1]


def _small_head(**kwargs):
    defaults = dict(
        cond_dim=16,
        input_freq_bins=129,
        band_dim=16,
        band_heads=2,
        d_model=32,
        num_heads=4,
        mlp_ratio=2.0,
        d_state=4,
        embedding_dim=24,
        time_dim=16,
        temporal_num_heads=2,
        visual_reliability_floor=0.25,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return mh.MambaHybridFlowHead(**defaults)


def _head_inputs(frames: int = 17, tokens: int = 16):
    g = torch.Generator().manual_seed(0)
    return dict(
        x_t=torch.randn(2, 2, 129, frames, generator=g),
        mixture=torch.randn(2, 2, 129, frames, generator=g),
        cond=torch.randn(2, 16, generator=g),
        t=torch.tensor([0.0, 0.5]),
        temporal_tokens=torch.randn(2, tokens, 16, generator=g),
        visual_activity=torch.rand(2, tokens, 1, generator=g),
        cross_attention_tokens=torch.randn(2, tokens, 16, generator=g),
    )


def test_head_shape_and_near_zero_init():
    torch.manual_seed(0)
    head = _small_head().eval()
    inputs = _head_inputs()
    v = head(**inputs)
    assert v.shape == inputs["x_t"].shape
    # One-step estimate starts at the mixture, as with the U-Net head.
    assert v.abs().max() < 1e-2 * inputs["x_t"].abs().max()


def test_mask_residual_is_complex_product_plus_residual():
    torch.manual_seed(0)
    head = _small_head().eval()
    with torch.no_grad():
        head.decoder[-1].weight.normal_(std=0.1)
    inputs = _head_inputs()
    captured = {}
    head.decoder.register_forward_hook(lambda m, i, o: captured.setdefault("out", o))
    v = head(**inputs)
    b, frames = 2, inputs["x_t"].shape[-1]
    out = captured["out"].view(b, frames, 2, 2, 129).permute(0, 2, 3, 4, 1)
    mask = torch.complex(out[:, 0, 0], out[:, 0, 1])
    res = torch.complex(out[:, 1, 0], out[:, 1, 1])
    z = torch.complex(inputs["x_t"][:, 0], inputs["x_t"][:, 1])
    expected = mask * z + res
    torch.testing.assert_close(torch.complex(v[:, 0], v[:, 1]), expected)


def test_reference_scan_is_recurrent():
    # The MambaVoice fallback (x * sigmoid(dt)) is pointwise in time; the
    # reference scan must propagate an impulse forward and never backward.
    torch.manual_seed(0)
    d, n, length = 3, 4, 10
    u = torch.zeros(1, d, length)
    u[..., 4] = 1.0
    delta = torch.zeros(1, d, length)
    A = -torch.rand(d, n) - 0.1
    B = torch.ones(1, n, length)
    C = torch.ones(1, n, length)
    y = mh.selective_scan_reference(u, delta, A, B, C, torch.zeros(d))
    assert torch.all(y[..., :4] == 0)
    assert torch.all(y[..., 5:] > 0)


@pytest.mark.skipif(
    mh.selective_scan_fn is None or not torch.cuda.is_available(),
    reason="needs mamba_ssm and CUDA",
)
def test_reference_scan_matches_cuda_kernel():
    torch.manual_seed(0)
    mixer = mh.MambaVisionMixer(64, scan_backend="auto").cuda()
    x = torch.randn(2, 50, 64, device="cuda")
    kernel = mixer(x)
    mixer.scan_backend = "reference"
    reference = mixer(x)
    torch.testing.assert_close(kernel, reference, rtol=1e-3, atol=1e-4)


def _mamba_cfg():
    cfg = _cfg("visual_floss")
    cfg["head"] = {
        "type": "mamba_hybrid",
        "input_freq_bins": 129,
        "band_dim": 16,
        "band_heads": 2,
        "d_model": 32,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "d_state": 4,
        "embedding_dim": 32,
        "time_dim": 32,
        "temporal_num_heads": 2,
        "visual_reliability_floor": 0.25,
        "dropout": 0.0,
    }
    return cfg


def test_visual_floss_with_mamba_head_is_finite_and_trains_every_parameter():
    torch.manual_seed(0)
    model = MambaVoiceMCFlow(_mamba_cfg())
    model.train()
    model.set_training_progress(10, 100)
    out = model.training_loss(_batch())
    assert torch.isfinite(out["loss"])
    assert out["visual_floss_mixture_error"] < 1e-8
    out["loss"].backward()
    missing = [k for k, p in model.head.named_parameters() if p.grad is None]
    assert not missing, missing


def test_mamba_config_differs_from_canonical_only_in_head():
    canonical = yaml.safe_load((ROOT / "configs/visual_floss_mrstft.yaml").read_text())
    mamba = yaml.safe_load(
        (ROOT / "configs/visual_floss_mrstft_mamba_hybrid.yaml").read_text()
    )
    assert mamba["head"]["type"] == "mamba_hybrid"
    for key in canonical:
        if key == "head":
            continue
        if key == "training":
            a = dict(canonical[key], out_dir=None)
            b = dict(mamba[key], out_dir=None)
            assert a == b
        else:
            assert canonical[key] == mamba[key], key
    head = mh.MambaHybridFlowHead(
        cond_dim=canonical["model"]["cond_dim"],
        **{k: v for k, v in mamba["head"].items() if k not in {"type", "scan_backend"}},
    )
    params = sum(p.numel() for p in head.parameters()) / 1e6
    assert 15.0 < params < 17.0
