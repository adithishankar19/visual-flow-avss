import torch

from mambaflow.models import MambaVoiceMCFlow


def _base_cfg():
    return {
        "data": {"kind": "synthetic"},
        "backbone": {
            "class_path": "mambaflow.backbones.mambavoice.DummyMambaVoiceLike",
            "init_kwargs": {},
            "audio_module": "audio_encoder",
            "video_module": "stgcn",
            "freeze": True,
            "strict": True,
        },
        "model": {"cond_dim": 64},
        "flow": {"consistency": "every_step", "noise_scale": 1.0, "num_steps": 2},
    }


def _waveform_cfg():
    cfg = _base_cfg()
    cfg["head"] = {"hidden": 32, "depth": 2, "time_dim": 32}
    return cfg


def _specunet_cfg():
    cfg = _base_cfg()
    cfg["model"] = {"cond_dim": 64, "gate_type": "residual_tanh", "gate_strength": 1.0}
    cfg["stft"] = {"n_fft": 256, "hop_length": 64, "win_length": 256, "center": True}
    cfg["head"] = {
        "type": "spec_unet",
        "channels": [16, 32, 32],
        "downsample_modes": ["spatial", "frequency", "frequency"],
        "time_dim": 32,
        "kernel_size": 3,
    }
    cfg["training"] = {"lambda_fm": 1.0, "lambda_recon": 1.0, "lambda_consistency": 0.0}
    return cfg


def _diffvs_cfg():
    cfg = _base_cfg()
    cfg["model"] = {"cond_dim": 64, "gate_type": "residual_tanh", "gate_strength": 1.0}
    cfg["stft"] = {"n_fft": 256, "hop_length": 64, "win_length": 256, "center": True}
    cfg["flow"] = {
        "target_mode": "target_only",
        "init_mode": "mixture",
        "init_noise_scale": 0.0,
        "objective": "drift_only",
        "consistency": "none",
        "noise_scale": 1.0,
        "num_steps": 1,
    }
    cfg["head"] = {
        "type": "diffvs_unet",
        "model_channels": 16,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "embedding_dim": 64,
        "time_dim": 32,
        "band_splits": 4,
        "norm_groups": 8,
        "roformer_depth": 1,
        "roformer_heads": 4,
        "roformer_ff_mult": 2.0,
        "encoder_roformer": True,
        "decoder_roformer_levels": [0],
        "bottleneck_roformer": True,
        "gradient_checkpointing": False,
        "temporal_conditioning": "multiscale_film_cross_attn",
        "temporal_num_heads": 4,
    }
    cfg["training"] = {
        "lambda_drift": 1.0,
        "lambda_recon": 1.0,
        "lambda_consistency": 0.0,
    }
    return cfg


def _batch():
    return {
        "mixture": torch.randn(2, 2048),
        "target": torch.randn(2, 2048),
        "face": torch.randn(2, 25, 3, 68),
    }


def _assert_model_shapes(model, batch):
    losses = model.training_loss(batch)
    assert losses["loss"].ndim == 0
    out = model.separate(batch["mixture"], face=batch["face"], num_steps=2)
    assert out["target"].shape == batch["mixture"].shape
    assert out["residual"].shape == batch["mixture"].shape
    assert out["mixture_error"].abs().max() < 1e-5


def test_waveform_training_and_sampling_shapes():
    _assert_model_shapes(MambaVoiceMCFlow(_waveform_cfg()), _batch())


def test_specunet_training_and_sampling_shapes():
    _assert_model_shapes(MambaVoiceMCFlow(_specunet_cfg()), _batch())


def test_diffvs_unet_training_and_sampling_shapes():
    _assert_model_shapes(MambaVoiceMCFlow(_diffvs_cfg()), _batch())
