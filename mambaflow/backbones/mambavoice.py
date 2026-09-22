from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from mambaflow.utils.imports import add_repo_to_path, get_module_by_path, import_from_dotted_path
from mambaflow.backbones.local_vovit import LocalMambaVoiceFeatureExtractor


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(x):
        return x
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _feature_to_vector1(x: torch.Tensor, max_flat: int = 1024) -> torch.Tensor:
    """Convert arbitrary encoder output to a compact [B, D] vector.

    This keeps the original MambaVoice encoders untouched while making their hidden
    states usable as global flow-conditioning features.
    """
    if x.is_complex():
        x = torch.view_as_real(x)
    x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim == 2:
        return x

    b = x.shape[0]
    # Prefer preserving a plausible channel dimension.
    if x.ndim >= 3 and x.shape[1] <= 4096:
        y = x.reshape(b, x.shape[1], -1)
        return torch.cat([y.mean(dim=-1), y.std(dim=-1, unbiased=False)], dim=1)
    if x.ndim >= 3 and x.shape[-1] <= 4096:
        y = x.reshape(b, -1, x.shape[-1]).transpose(1, 2)
        return torch.cat([y.mean(dim=-1), y.std(dim=-1, unbiased=False)], dim=1)

    flat = x.reshape(b, 1, -1)
    pooled = F.adaptive_avg_pool1d(flat, max_flat).squeeze(1)
    return pooled

def _feature_to_vector(x: torch.Tensor,
    *,
    prefer_btc: bool = True,
    max_flat: int = 1024,) -> torch.Tensor:
    if x.is_complex():
        x = torch.view_as_real(x)

    x = x.float()

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim == 2:
        return x

    b = x.shape[0]

    if x.ndim == 3:
        if prefer_btc:
            # x is [B,T,C]. Pool over time.
            return torch.cat(
                [x.mean(dim=1), x.std(dim=1, unbiased=False)],
                dim=-1,
            )

        # fallback: assume [B,C,T]. Pool over time/frequency dims.
        y = x.reshape(b, x.shape[1], -1)
        return torch.cat(
            [y.mean(dim=-1), y.std(dim=-1, unbiased=False)],
            dim=1,
        )

    # For [B,C,...] tensors.
    y = x.reshape(b, x.shape[1], -1)
    return torch.cat(
        [y.mean(dim=-1), y.std(dim=-1, unbiased=False)],
        dim=1,
    )



def _feature_to_tokens(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Convert captured MambaVoice features to temporal tokens [B, T, C].

    The current global conditioner pools away time. This helper keeps a simple
    temporal path for the SpecUNet flow head:
      - audio examples like [B, C, K, T] become [B, T, C] by averaging K
      - video examples like [B, C, T, J] become [B, T, C] by averaging J
      - [B, C, T] is treated as channel-first temporal and becomes [B, T, C]
      - [B, T, C] can be handled by setting kind="tokens" only if already known
    """
    if x.is_complex():
        x = torch.view_as_real(x)
    x = x.float()
    kind = str(kind).lower()

    if x.ndim == 2:
        # No temporal dimension. Return a single token.
        return x.unsqueeze(1)

    if x.ndim == 3:
        # Handle both [B,C,T] and already-tokenized [B,T,C].
        if kind in {"tokens", "audio_tokens", "video_tokens"}:
            return x.contiguous()
        if kind == "audio" and x.shape[-1] in {128, 256, 512, 768, 1024}:
            return x.contiguous()
        if kind == "video" and x.shape[-1] in {64, 100, 128, 256, 512, 768, 1024}:
            return x.contiguous()
        return x.transpose(1, 2).contiguous()

    if x.ndim == 4:
        if kind == "audio":
            # Expected audio feature from band-split path: [B, C, K, T].
            x = x.mean(dim=2)          # [B, C, T]
            return x.transpose(1, 2).contiguous()
        if kind == "video":
            # Expected ST-GCN/keypoint feature: [B, C, T, J].
            x = x.mean(dim=3)          # [B, C, T]
            return x.transpose(1, 2).contiguous()
        
        # Fallback: preserve the last axis as time and average all other non-channel axes.
        b = x.shape[0]
        x = x.reshape(b, x.shape[1], -1)
        return x.transpose(1, 2).contiguous()
    

    # Generic fallback for higher dimensional tensors: keep channel dim 1 and flatten the rest as time.
    b = x.shape[0]
    x = x.reshape(b, x.shape[1], -1)
    return x.transpose(1, 2).contiguous()


@dataclass
class HookSpec:
    name: str
    module: nn.Module


class MambaVoiceEncoderBundle(nn.Module):
    """Use the exact MambaVoice/VoViT audio and video encoders via forward hooks."""

    DEFAULT_AUDIO_KEYWORDS = (
        "audio_encoder", "audio_enc", "aud_encoder", "spectrogram", "band", "bandsplit",
        "ap", "encoder_audio", "audio", "mamba"
    )
    DEFAULT_VIDEO_KEYWORDS = (
        "stgcn", "st_gcn", "gcn", "visual_encoder", "video_encoder", "landmark",
        "face", "visual", "video", "body"
    )

    def __init__(
        self,
        class_path: str = "vovit.VoViT_f",
        init_kwargs: Optional[Dict[str, Any]] = None,
        audio_module: Optional[str] = None,
        video_module: Optional[str] = None,
        freeze: bool = True,
        strict: bool = True,
        audio_keywords: Optional[Iterable[str]] = None,
        video_keywords: Optional[Iterable[str]] = None,
        env_repo_var: str = "MAMBAVOICE_REPO",
        use_local_feature_extractor: bool = False,
        local_feature_cfg: Optional[Dict[str, Any]] = None,
        debug_feature_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.use_local_feature_extractor = bool(use_local_feature_extractor)
        self.freeze = freeze
        self.strict = strict
        self.audio_module_name = audio_module
        self.video_module_name = video_module
        self.audio_keywords = tuple(audio_keywords or self.DEFAULT_AUDIO_KEYWORDS)
        self.video_keywords = tuple(video_keywords or self.DEFAULT_VIDEO_KEYWORDS)
        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

        if self.use_local_feature_extractor:
            local_feature_cfg = dict(local_feature_cfg or {})
            local_feature_cfg.setdefault("debug_shapes", bool(debug_feature_shapes))
            self.model = LocalMambaVoiceFeatureExtractor(**local_feature_cfg)
            if self.freeze:
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad_(False)
            self.audio_spec = None
            self.video_spec = None
            return

        add_repo_to_path(env_repo_var)
        init_kwargs = dict(init_kwargs or {})
        cls = import_from_dotted_path(class_path)
        self.model = cls(**init_kwargs)

        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.audio_spec = self._resolve_hook("audio", audio_module, self.audio_keywords)
        self.video_spec = self._resolve_hook("video", video_module, self.video_keywords)
        self._register_hooks()

    def _score_name(self, name: str, keywords: Iterable[str]) -> int:
        lname = name.lower()
        score = 0
        for kw in keywords:
            kw = kw.lower()
            if lname.endswith(kw):
                score += 5
            if kw in lname:
                score += 2
        score += min(name.count("."), 5)
        return score

    def _resolve_hook(self, kind: str, explicit_name: Optional[str], keywords: Iterable[str]) -> Optional[HookSpec]:
        if explicit_name:
            module = get_module_by_path(self.model, explicit_name)
            if not isinstance(module, nn.Module):
                raise TypeError(f"Configured {kind}_module={explicit_name!r} is not an nn.Module")
            return HookSpec(explicit_name, module)

        candidates: List[Tuple[int, str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            score = self._score_name(name, keywords)
            if score > 0:
                candidates.append((score, name, module))
        candidates.sort(key=lambda z: (z[0], z[1].count(".")), reverse=True)
        if candidates:
            _, name, module = candidates[0]
            return HookSpec(name, module)
        if self.strict:
            raise RuntimeError(
                f"Could not auto-discover a MambaVoice {kind} encoder. Run scripts/inspect_mambavoice.py "
                f"and set backbone.{kind}_module in the config."
            )
        return None

    def _register_hooks(self) -> None:
        def make_hook(key: str):
            def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                t = _first_tensor(output)
                if t is not None:
                    self._captured[key] = t
            return hook

        if self.audio_spec is not None:
            self._hooks.append(self.audio_spec.module.register_forward_hook(make_hook("audio")))
        if self.video_spec is not None:
            self._hooks.append(self.video_spec.module.register_forward_hook(make_hook("video")))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, mixture: torch.Tensor, face: Optional[torch.Tensor] = None, body: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        self._captured = {}
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)

        if self.use_local_feature_extractor:
            ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
            with ctx:
                feats = self.model(mixture, face=face, body=body)
            return {
                "audio": feats.get("audio"),
                "video": feats.get("video"),
                "output": feats,
                "audio_hook": "local_audio_encoder",
                "video_hook": "local_forward_visual_stgcn",
            }

        def call_model() -> Any:
            if face is not None and body is not None:
                return self.model(mixture, face, body)
            if face is not None:
                return self.model(mixture, face)
            if body is not None:
                return self.model(mixture, body)
            return self.model(mixture)

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            output = call_model()

        audio_t = self._captured.get("audio")
        video_t = self._captured.get("video")
        if self.strict and audio_t is None:
            raise RuntimeError(
                f"Audio hook {self.audio_spec.name if self.audio_spec else None!r} did not capture a tensor. "
                "Set backbone.audio_module to the exact audio encoder module name."
            )
        if self.strict and video_t is None:
            raise RuntimeError(
                f"Video hook {self.video_spec.name if self.video_spec else None!r} did not capture a tensor. "
                "Set backbone.video_module to the exact ST-GCN/video encoder module name."
            )
        return {
            "audio": audio_t,
            "video": video_t,
            "output": output,
            "audio_hook": self.audio_spec.name if self.audio_spec else None,
            "video_hook": self.video_spec.name if self.video_spec else None,
        }


class MambaVoiceConditioner(nn.Module):
    """Project captured MambaVoice encoder states to a fixed conditioning vector.

    gate_type controls how the visual/ST-GCN feature modulates the audio feature:
      - "none" / "concat": original concat fusion.
      - "residual_tanh": audio * (1 + gate_strength * tanh(W_video video)).
      - "film": Feature-wise Linear Modulation (audio * gamma(video) + beta(video)).
    """

    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        cond_dim: int = 256,
        gate_type: str = "film",
        gate_strength: float = 1.0,
        global_condition_type: str = "av_gated",
        temporal_token_type: str = "av_gated",
        audio_condition_dropout: float = 0.0,
        visual_token_dropout: float = 0.0,
        visual_token_noise_std: float = 0.0,
        debug_feature_shapes: bool = False,
        raw_visual_map: bool = False,
        concat_learned_activity: bool = True,
    ) -> None:
        super().__init__()
        self.bundle = MambaVoiceEncoderBundle(**backbone_cfg)
        self.cond_dim = cond_dim
        self.gate_type = str(gate_type).lower()
        self.gate_strength = float(gate_strength)
        self.global_condition_type = str(global_condition_type).lower()
        self.temporal_token_type = str(temporal_token_type).lower()
        self.audio_condition_dropout = float(audio_condition_dropout)
        self.visual_token_dropout = float(visual_token_dropout)
        self.visual_token_noise_std = float(visual_token_noise_std)
        self.debug_feature_shapes = bool(debug_feature_shapes)
        self.raw_visual_map = bool(raw_visual_map)
        self.concat_learned_activity = bool(concat_learned_activity)
        self.audio_proj = nn.LazyLinear(cond_dim)
        self.video_proj = nn.LazyLinear(cond_dim)
        
        # Original Gating Layer
        self.video_gate = nn.Linear(cond_dim, cond_dim)
        nn.init.normal_(self.video_gate.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.video_gate.bias, mean=0.0, std=1e-3)
        
        # NEW: Global FiLM Projection Branches
        self.global_film_gamma = nn.Linear(cond_dim, cond_dim)
        self.global_film_beta = nn.Linear(cond_dim, cond_dim)
        # Initialize FiLM weights to close to zero so it initially passes features cleanly
        nn.init.normal_(self.global_film_gamma.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.global_film_beta.weight, mean=0.0, std=1e-3)
        nn.init.ones_(self.global_film_gamma.bias) # Gamma should start around 1.0
        nn.init.zeros_(self.global_film_beta.bias) # Beta should start around 0.0

        self.fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.audio_token_proj = nn.LazyLinear(cond_dim)
        self.video_token_proj = nn.LazyLinear(cond_dim)
        self.audio_token_norm = nn.LayerNorm(cond_dim)
        self.video_token_norm = nn.LayerNorm(cond_dim)
        self.temporal_token_norm = nn.LayerNorm(cond_dim)
        
        # Original Temporal Gate
        self.video_token_gate = nn.Linear(cond_dim, cond_dim)
        nn.init.normal_(self.video_token_gate.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.video_token_gate.bias, mean=0.0, std=1e-3)
        
        # NEW: Temporal FiLM Projection Branches
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

        self.video_activity_head = nn.Linear(cond_dim, 1)
        nn.init.normal_(self.video_activity_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.video_activity_head.bias, mean=0.0, std=1e-3)

    def _maybe_corrupt_video_tokens(
        self,
        video_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply train-time visual corruption and return a reliability target.

        The target is 1 for an intact visual token and 0 for a token that was
        deliberately dropped.  Gaussian jitter is also supported as a softer
        nuisance perturbation but does not change the binary target.
        """
        reliability = torch.ones(
            video_tokens.shape[0],
            video_tokens.shape[1],
            1,
            device=video_tokens.device,
            dtype=video_tokens.dtype,
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

    def _debug_tensor(self, name: str, x: Optional[torch.Tensor]) -> None:
        if not self.debug_feature_shapes or x is None:
            return
        with torch.no_grad():
            xf = x.detach().float()
            mean = float(xf.mean())
            std = float(xf.std(unbiased=False))
        print(f"[mcflow_conditioner] {name}: shape={tuple(x.shape)} mean={mean:.6g} std={std:.6g}")

    def _apply_gate(self, audio_vec: torch.Tensor, video_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate_type in {"none", "concat", "off", "false"}:
            gate = torch.ones_like(audio_vec)
            return audio_vec, gate
        if self.gate_type in {"residual_tanh", "tanh_residual", "res_tanh"}:
            gate_delta = torch.tanh(self.video_gate(video_vec))
            gate = 1.0 + self.gate_strength * gate_delta
            return audio_vec * gate, gate
        if self.gate_type in {"film"}:
            # Compute scales (gamma) and shifts (beta) via video conditioning
            gamma = self.global_film_gamma(video_vec)
            beta = self.global_film_beta(video_vec)
            # Apply linear modulation transform to the audio feature vector
            gated_audio = (gamma * audio_vec) + beta
            return gated_audio, gamma # Return gamma as pseudo gate tracking statistic
        raise ValueError(f"Unknown conditioner gate_type={self.gate_type!r}")

    def _maybe_drop_audio_condition(
        self,
        audio_vec: torch.Tensor,
        audio_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (not self.training) or self.audio_condition_dropout <= 0.0:
            return audio_vec, audio_tokens
        p = max(0.0, min(1.0, self.audio_condition_dropout))
        if p <= 0.0:
            return audio_vec, audio_tokens
        drop = (torch.rand(audio_vec.shape[0], 1, device=audio_vec.device) < p)
        audio_vec = audio_vec.masked_fill(drop, 0.0)
        if audio_tokens is not None:
            audio_tokens = audio_tokens.masked_fill(drop[:, :, None], 0.0)
        return audio_vec, audio_tokens

    def _select_global_condition(
        self,
        audio_vec: torch.Tensor,
        video_vec: torch.Tensor,
        gated_audio_vec: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.global_condition_type
        if mode in {"av_gated", "av", "fused", "concat", "audio_video", "video_audio"}:
            return self.fuse(torch.cat([gated_audio_vec, video_vec], dim=-1))
        if mode in {"video_only", "video", "visual_only", "visual"}:
            return video_vec
        if mode in {"audio_only", "audio"}:
            return audio_vec
        if mode in {"none", "zero", "zeros", "off", "false"}:
            return torch.zeros_like(video_vec)
        raise ValueError(f"Unknown global_condition_type={self.global_condition_type!r}")

    def _select_temporal_tokens(
        self,
        token_bundle: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if token_bundle is None:
            return None
        mode = self.temporal_token_type
        if mode in {"av_gated", "av", "fused", "concat", "audio_video", "video_audio"}:
            return token_bundle["av_tokens"]
        if mode in {"video_only", "video", "visual_only", "visual"}:
            return token_bundle["video_tokens"]
        if mode in {"audio_only", "audio"}:
            return token_bundle["audio_tokens"]
        if mode in {"none", "zero", "zeros", "off", "false"}:
            return None
        raise ValueError(f"Unknown temporal_token_type={self.temporal_token_type!r}")

    def _fuse_temporal_tokens(
        self,
        audio_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        visual_reliability_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create gated or FiLM-modulated AV temporal tokens from aligned audio/video tokens."""
        if self.gate_type in {"none", "concat", "off", "false"}:
            gated_audio_tokens = audio_tokens
            token_gate = torch.ones_like(audio_tokens)
        elif self.gate_type == "film":
            # Apply sequence-aligned linear scaling and bias shifts
            gamma = self.temporal_film_gamma(video_tokens)
            beta = self.temporal_film_beta(video_tokens)
            gated_audio_tokens = (gamma * audio_tokens) + beta
            token_gate = gamma # Trace gamma instead for verification prints
        else:
            token_gate = 2.0 * torch.sigmoid(self.video_token_gate(video_tokens))
            gated_audio_tokens = audio_tokens * token_gate

        av_tokens = gated_audio_tokens
        visual_activity = torch.sigmoid(self.video_activity_head(video_tokens))
        return {
            "audio_tokens": audio_tokens,
            "video_tokens": video_tokens,
            "av_tokens": av_tokens,
            "token_gate": token_gate,
            "visual_activity": visual_activity,
            "visual_reliability_target": (
                visual_reliability_target
                if visual_reliability_target is not None
                else torch.ones_like(visual_activity)
            ),
        }

    def _canonical_face_landmarks(self, face: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if face is None:
            return None
        x = face.float()
        if x.ndim == 5:
            if x.shape[1] == 2:
                x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()
            elif x.shape[2] == 2:
                x = x[:, :, :, :, 0].permute(0, 1, 3, 2).contiguous()
            else:
                return None
        elif x.ndim == 4:
            if x.shape[1] == 2:
                x = x.permute(0, 2, 3, 1).contiguous()
            elif x.shape[2] == 2:
                x = x.permute(0, 1, 3, 2).contiguous()
            elif x.shape[-1] == 2:
                pass
            else:
                return None
        else:
            return None
        if x.shape[-2] < 68 or x.shape[-1] != 2:
            return None
        return x[:, :, :68, :]

    def _standardize_visual_curves(self, curves: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = curves.mean(dim=1, keepdim=True)
        std = curves.std(dim=1, keepdim=True, unbiased=False)
        return torch.tanh((curves - mean) / (std + eps))

    def _make_raw_visual_map(self, face: Optional[torch.Tensor], mixture: torch.Tensor) -> Optional[torch.Tensor]:
        lm = self._canonical_face_landmarks(face)
        if lm is None:
            return None
        lm = lm.to(device=mixture.device, dtype=mixture.dtype)
        b, t, _, _ = lm.shape

        def dist(a: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
            return (a - b_).pow(2).sum(dim=-1).sqrt()

        mouth = lm[:, :, 48:68, :]
        mouth_outer = lm[:, :, 48:60, :]
        upper = lm[:, :, [50, 51, 52, 61, 62, 63], :].mean(dim=2)
        lower = lm[:, :, [56, 57, 58, 65, 66, 67], :].mean(dim=2)
        left = lm[:, :, 48, :]
        right = lm[:, :, 54, :]
        chin = lm[:, :, 8, :]

        mouth_open = dist(upper, lower)
        mouth_width = dist(left, right)
        mouth_area_proxy = mouth_open * mouth_width

        d_lm = torch.zeros_like(lm)
        if t > 1:
            d_lm[:, 1:] = lm[:, 1:] - lm[:, :-1]
        d_mouth = d_lm[:, :, 48:68, :]
        mouth_motion = d_mouth.pow(2).sum(dim=-1).sqrt().mean(dim=-1)
        jaw_motion = d_lm[:, :, [6, 7, 8, 9, 10], :].pow(2).sum(dim=-1).sqrt().mean(dim=-1)
        face_motion = d_lm.pow(2).sum(dim=-1).sqrt().mean(dim=-1)
        mouth_open_delta = torch.zeros_like(mouth_open)
        if t > 1:
            mouth_open_delta[:, 1:] = mouth_open[:, 1:] - mouth_open[:, :-1]

        curves = torch.stack(
            [mouth_open, mouth_open_delta.abs(), mouth_motion, jaw_motion, face_motion, mouth_area_proxy],
            dim=-1,
        )
        return self._standardize_visual_curves(curves)

    def _build_temporal_tokens(
        self,
        audio: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        mixture: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if audio is None or video is None:
            return None

        audio_tokens = _feature_to_tokens(audio, kind="audio").to(mixture.device)
        video_tokens = _feature_to_tokens(video, kind="video").to(mixture.device)

        audio_tokens = self.audio_token_norm(self.audio_token_proj(audio_tokens))
        video_tokens = self.video_token_norm(self.video_token_proj(video_tokens))

        if video_tokens.shape[1] != audio_tokens.shape[1]:
            video_tokens = F.interpolate(
                video_tokens.transpose(1, 2),
                size=audio_tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        video_tokens, reliability_target = self._maybe_corrupt_video_tokens(video_tokens)
        return self._fuse_temporal_tokens(
            audio_tokens,
            video_tokens,
            visual_reliability_target=reliability_target,
        )

    def forward(self, mixture: torch.Tensor, face: Optional[torch.Tensor] = None, body: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        captured = self.bundle(mixture, face=face, body=body)
        audio = captured.get("audio")
        video = captured.get("video")
        
        if audio is None:
            audio_vec = torch.zeros(mixture.shape[0], self.cond_dim, device=mixture.device, dtype=mixture.dtype)
        else:
            audio_vec = self.audio_proj(_feature_to_vector(audio).to(mixture.device))
        if video is None:
            video_vec = torch.zeros(mixture.shape[0], self.cond_dim, device=mixture.device, dtype=mixture.dtype)
        else:
            video_vec = self.video_proj(_feature_to_vector(video).to(mixture.device))

        self._debug_tensor("captured_audio", audio)
        self._debug_tensor("captured_video", video)
        self._debug_tensor("audio_vec", audio_vec)
        self._debug_tensor("video_vec", video_vec)

        token_bundle = self._build_temporal_tokens(audio, video, mixture)
        raw_visual_map = self._make_raw_visual_map(face, mixture) if self.raw_visual_map else None
        if token_bundle is not None:
            self._debug_tensor("audio_tokens", token_bundle.get("audio_tokens"))
            self._debug_tensor("video_tokens", token_bundle.get("video_tokens"))
            self._debug_tensor("av_tokens", token_bundle.get("av_tokens"))
            self._debug_tensor("learned_visual_activity", token_bundle.get("visual_activity"))
        self._debug_tensor("raw_visual_map", raw_visual_map)
        audio_tokens = token_bundle["audio_tokens"] if token_bundle is not None else None
        audio_vec, audio_tokens = self._maybe_drop_audio_condition(audio_vec, audio_tokens)
        if token_bundle is not None and audio_tokens is not None:
            token_bundle = self._fuse_temporal_tokens(
                audio_tokens,
                token_bundle["video_tokens"],
                visual_reliability_target=token_bundle.get("visual_reliability_target"),
            )

        gated_audio_vec, av_gate = self._apply_gate(audio_vec, video_vec)
        cond = self._select_global_condition(audio_vec, video_vec, gated_audio_vec)
        temporal_tokens = self._select_temporal_tokens(token_bundle)
        self._debug_tensor("conditioning", cond)
        self._debug_tensor("selected_temporal_tokens", temporal_tokens)

        learned_activity = token_bundle.get("visual_activity") if token_bundle is not None else None
        if raw_visual_map is not None and learned_activity is not None and self.concat_learned_activity:
            learned = learned_activity.to(raw_visual_map.device, raw_visual_map.dtype)
            if learned.ndim == 2:
                learned = learned.unsqueeze(-1)
            if learned.ndim != 3:
                raise ValueError(f"learned visual activity must be [B,T,C], got {tuple(learned.shape)}")
            if learned.shape[1] != raw_visual_map.shape[1]:
                learned = F.interpolate(
                    learned.transpose(1, 2),
                    size=raw_visual_map.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            visual_activity = torch.cat([raw_visual_map, learned], dim=-1)
        elif raw_visual_map is not None:
            visual_activity = raw_visual_map
        else:
            visual_activity = learned_activity

        captured["conditioning"] = cond
        captured["temporal_tokens"] = temporal_tokens
        captured["visual_activity"] = visual_activity
        captured["audio_vec"] = audio_vec
        captured["video_vec"] = video_vec
        captured["gated_audio_vec"] = gated_audio_vec
        captured["av_gate"] = av_gate
        captured["gate_type"] = self.gate_type
        captured["global_condition_type"] = self.global_condition_type
        captured["temporal_token_type"] = self.temporal_token_type
        if token_bundle is not None:
            captured.update(token_bundle)
        return captured


class DummyMambaVoiceLike(nn.Module):
    """Small test-only module with audio_encoder and stgcn names."""

    def __init__(self, audio_dim: int = 32, video_dim: int = 32) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, audio_dim, 9, padding=4), nn.SiLU(),
            nn.Conv1d(audio_dim, audio_dim, 9, padding=4), nn.SiLU(),
        )
        self.stgcn = nn.Sequential(
            nn.Conv1d(204, video_dim, 3, padding=1), nn.SiLU(),
            nn.Conv1d(video_dim, video_dim, 3, padding=1), nn.SiLU(),
        )

    def forward(self, mixture: torch.Tensor, face: torch.Tensor) -> Dict[str, torch.Tensor]:
        if mixture.ndim == 2:
            a = self.audio_encoder(mixture.unsqueeze(1))
        else:
            a = self.audio_encoder(mixture)
        if face.ndim == 4:
            b, t, c, j = face.shape
            v = face.reshape(b, t, c * j).transpose(1, 2)
        else:
            v = face.transpose(1, 2)
        v = self.stgcn(v.float())
        return {"estimated_wav": mixture, "audio_latent": a, "video_latent": v}