"""Audio-visual conditioner (paper Sec. 3.4).

The mixture is encoded with the attention-based band-split encoder of
MambaVoice and the target singer's facial landmarks with an ST-GCN.  Visual
tokens modulate audio tokens through FiLM; the result conditions the U-Net.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from vist.encoders import LocalMambaVoiceFeatureExtractor


def _pool_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C] tokens -> [B, 2C] vector of their mean and std over time."""
    x = x.float()
    return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)


class EncoderBundle(nn.Module):
    """Band-split audio encoder and ST-GCN visual encoder."""

    def __init__(
        self,
        local_feature_cfg: Optional[Dict[str, Any]] = None,
        freeze: bool = False,
        use_local_feature_extractor: bool = True,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if not use_local_feature_extractor:
            raise ValueError("VIST uses the bundled encoders; set backbone.use_local_feature_extractor=true")
        self.freeze = bool(freeze)
        self.model = LocalMambaVoiceFeatureExtractor(**dict(local_feature_cfg or {}))
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, mixture: torch.Tensor, face: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            feats = self.model(mixture, face=face)
        return {"audio": feats.get("audio"), "video": feats.get("video")}


class AVConditioner(nn.Module):
    """Fuse audio and visual tokens with FiLM and expose the U-Net conditioning.

    Returns a global vector ``conditioning`` [B, D], fused temporal tokens
    ``temporal_tokens`` [B, T, D], the visual tokens ``video_tokens`` used as
    keys/values at the bottleneck, and the learned per-frame reliability
    ``visual_activity`` [B, T, 1] with its training target.
    """

    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        cond_dim: int = 512,
        audio_condition_dropout: float = 0.0,
        visual_token_dropout: float = 0.0,
        visual_token_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.bundle = EncoderBundle(**backbone_cfg)
        self.cond_dim = cond_dim
        self.audio_condition_dropout = float(audio_condition_dropout)
        self.visual_token_dropout = float(visual_token_dropout)
        self.visual_token_noise_std = float(visual_token_noise_std)

        # Modules are created and initialised in the order of the released
        # training run, so a seeded run from scratch draws the same weights.
        #
        # video_gate, video_token_gate, temporal_token_norm and temporal_fuse
        # are not used in the forward pass.  They are part of the released
        # checkpoints and of the 23.2M trainable parameters reported in the
        # paper, so they are kept for exact state-dict compatibility.
        self.audio_proj = nn.LazyLinear(cond_dim)
        self.video_proj = nn.LazyLinear(cond_dim)
        self.video_gate = nn.Linear(cond_dim, cond_dim)
        nn.init.normal_(self.video_gate.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.video_gate.bias, mean=0.0, std=1e-3)

        # Global path: pooled audio vector modulated by pooled visual vector.
        self.global_film_gamma = nn.Linear(cond_dim, cond_dim)
        self.global_film_beta = nn.Linear(cond_dim, cond_dim)
        nn.init.normal_(self.global_film_gamma.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.global_film_beta.weight, mean=0.0, std=1e-3)
        nn.init.ones_(self.global_film_gamma.bias)
        nn.init.zeros_(self.global_film_beta.bias)
        self.fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Temporal path: per-frame audio tokens modulated by visual tokens.
        self.audio_token_proj = nn.LazyLinear(cond_dim)
        self.video_token_proj = nn.LazyLinear(cond_dim)
        self.audio_token_norm = nn.LayerNorm(cond_dim)
        self.video_token_norm = nn.LayerNorm(cond_dim)
        self.temporal_token_norm = nn.LayerNorm(cond_dim)
        self.video_token_gate = nn.Linear(cond_dim, cond_dim)
        nn.init.normal_(self.video_token_gate.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.video_token_gate.bias, mean=0.0, std=1e-3)
        self.temporal_film_gamma = nn.Linear(cond_dim, cond_dim)
        self.temporal_film_beta = nn.Linear(cond_dim, cond_dim)
        nn.init.normal_(self.temporal_film_gamma.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.temporal_film_beta.weight, mean=0.0, std=1e-3)
        nn.init.ones_(self.temporal_film_gamma.bias)
        nn.init.zeros_(self.temporal_film_beta.bias)
        self.temporal_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Reliability r: detects visual frames that were dropped in training.
        self.video_activity_head = nn.Linear(cond_dim, 1)
        nn.init.normal_(self.video_activity_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.video_activity_head.bias, mean=0.0, std=1e-3)

    def _corrupt_video_tokens(self, video_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero a random 10% of visual frames in training; return the reliability target.

        The target is 1 for an intact frame and 0 for a dropped one.  Gaussian
        jitter is added as a nuisance perturbation and does not change it.
        """
        reliability = torch.ones(
            video_tokens.shape[0], video_tokens.shape[1], 1,
            device=video_tokens.device, dtype=video_tokens.dtype,
        )
        if not self.training:
            return video_tokens, reliability
        out = video_tokens
        p = max(0.0, min(1.0, self.visual_token_dropout))
        if p > 0.0:
            dropped = torch.rand(
                video_tokens.shape[0], video_tokens.shape[1], 1,
                device=video_tokens.device,
            ) < p
            reliability = (~dropped).to(video_tokens.dtype)
            out = out.masked_fill(dropped, 0.0)
        if self.visual_token_noise_std > 0.0:
            out = out + torch.randn_like(out) * self.visual_token_noise_std
        return out, reliability

    def _drop_audio_condition(
        self, audio_vec: torch.Tensor, audio_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = max(0.0, min(1.0, self.audio_condition_dropout))
        if not self.training or p <= 0.0:
            return audio_vec, audio_tokens
        drop = torch.rand(audio_vec.shape[0], 1, device=audio_vec.device) < p
        return audio_vec.masked_fill(drop, 0.0), audio_tokens.masked_fill(drop[:, :, None], 0.0)

    def forward(self, mixture: torch.Tensor, face: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        feats = self.bundle(mixture, face=face)
        audio, video = feats["audio"], feats["video"]
        if audio is None or video is None:
            raise ValueError("VIST needs both the mixture and the target singer's landmarks")

        audio_vec = self.audio_proj(_pool_tokens(audio).to(mixture.device))
        video_vec = self.video_proj(_pool_tokens(video).to(mixture.device))

        audio_tokens = self.audio_token_norm(self.audio_token_proj(audio.float().to(mixture.device)))
        video_tokens = self.video_token_norm(self.video_token_proj(video.float().to(mixture.device)))
        if video_tokens.shape[1] != audio_tokens.shape[1]:
            video_tokens = F.interpolate(
                video_tokens.transpose(1, 2),
                size=audio_tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        video_tokens, reliability_target = self._corrupt_video_tokens(video_tokens)
        audio_vec, audio_tokens = self._drop_audio_condition(audio_vec, audio_tokens)

        # FiLM fusion: audio * gamma(video) + beta(video), globally and per frame.
        gated_audio_vec = self.global_film_gamma(video_vec) * audio_vec + self.global_film_beta(video_vec)
        cond = self.fuse(torch.cat([gated_audio_vec, video_vec], dim=-1))
        av_tokens = (
            self.temporal_film_gamma(video_tokens) * audio_tokens
            + self.temporal_film_beta(video_tokens)
        )
        visual_activity = torch.sigmoid(self.video_activity_head(video_tokens))

        return {
            "conditioning": cond,
            "temporal_tokens": av_tokens,
            "video_tokens": video_tokens,
            "visual_activity": visual_activity,
            "visual_reliability_target": reliability_target,
        }
