from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .modules.audio_encoder_attention import AudioEncoder
from .modules.st_gcn import ST_GCN


class LocalMambaVoiceFeatureExtractor(nn.Module):
    """Local MambaVoice-style audio/visual feature extractor.

    This is intentionally only the feature-extraction part needed by MCFlow:
      - facial keypoints -> ST-GCN -> interpolated visual tokens [B,T,256]
      - mixture waveform -> STFT magnitude -> BandSplit AudioEncoder -> [B,T,512]

    It mirrors the relevant `forward_visual` and audio feature path from the
    gesture-guided/MambaVoice repo so the flow repo does not need to import that
    repo just to build temporal conditioning features.
    """

    DEFAULT_BANDS: Tuple[Tuple[int, int], ...] = (
        (0, 4), (4, 12), (12, 28), (28, 52),
        (52, 84), (84, 128), (128, 192), (192, 256),
    )

    DEFAULT_GRAPH_KWARGS: Dict[str, Any] = {
        "graph_cfg": {
            "layout": "acappella",
            "strategy": "spatial",
            "max_hop": 1,
            "dilation": 1,
        },
        "edge_importance_weighting": "dynamic",
        "dropout": False,
        "dilated": False,
    }

    def __init__(
        self,
        sample_rate: int = 16384,
        n_fft: int = 1022,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        downsample_coarse: bool = True,
        n_sources: int = 2,
        visual_in_channels: int = 2,
        visual_dim: int = 256,
        audio_dim: int = 512,
        video_temporal_features: int = 256,
        n: int = 1,
        bands: Optional[Sequence[Tuple[int, int]]] = None,
        graph_kwargs: Optional[Dict[str, Any]] = None,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.downsample_coarse = bool(downsample_coarse)
        self.n_sources = int(n_sources)
        self.visual_dim = int(visual_dim)
        self.audio_dim = int(audio_dim)
        self.video_temporal_features = int(video_temporal_features)
        self._n = int(n)
        self.debug_shapes = bool(debug_shapes)

        graph_kwargs = dict(graph_kwargs or self.DEFAULT_GRAPH_KWARGS)
        self.graph_net = ST_GCN(
            in_channels=visual_in_channels,
            temporal_downsample=False,
            **graph_kwargs,
        )
        self.audio_net = AudioEncoder(
            bands=tuple(bands or self.DEFAULT_BANDS),
            embed_dim=128,
            output_dim=audio_dim,
        )
        self.register_buffer("_window", torch.hann_window(self.win_length), persistent=False)

    def _log_shape(self, name: str, x: Optional[torch.Tensor]) -> None:
        if self.debug_shapes and x is not None:
            with torch.no_grad():
                mean = float(x.detach().float().mean())
                std = float(x.detach().float().std(unbiased=False))
            print(f"[local_mambavoice] {name}: shape={tuple(x.shape)} mean={mean:.6g} std={std:.6g}")

    def _canonical_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Return landmarks in ST-GCN format [B, C=2, T, V=68, M=1]."""
        if landmarks is None:
            raise ValueError("LocalMambaVoiceFeatureExtractor.forward_visual requires landmarks")
        x = landmarks.float()
        if x.ndim == 5:
            # Expected [B,2,T,68,1]. Also handle [B,T,2,68,1].
            if x.shape[1] == 2:
                return x
            if x.shape[2] == 2:
                return x.permute(0, 2, 1, 3, 4).contiguous()
        if x.ndim == 4:
            # [B,T,2,68] or [B,2,T,68]
            if x.shape[2] == 2:
                return x.permute(0, 2, 1, 3).unsqueeze(-1).contiguous()
            if x.shape[1] == 2:
                return x.unsqueeze(-1).contiguous()
        raise ValueError(
            "Expected facial landmarks as [B,2,T,68,1], [B,T,2,68,1], "
            f"[B,2,T,68], or [B,T,2,68], got {tuple(x.shape)}"
        )

    def forward_visual(self, landmarks: torch.Tensor) -> torch.Tensor:
        """MambaVoice-style visual path: landmarks -> ST-GCN -> [B,T,256]."""
        landmarks = self._canonical_landmarks(landmarks)
        self._log_shape("landmarks", landmarks)
        sk_features = self.graph_net(landmarks)  # [B,256,T_in]
        self._log_shape("stgcn_raw", sk_features)
        target_t = self.video_temporal_features * self._n
        sk_features = F.interpolate(sk_features, size=target_t, mode="linear", align_corners=False)
        sk_features = sk_features.transpose(1, 2).contiguous()  # [B,T,256]
        self._log_shape("visual_tokens", sk_features)
        return sk_features

    def _stft_magnitude(self, mixture: torch.Tensor) -> torch.Tensor:
        """Return coarse STFT magnitude [B,256,T] like the MambaVoice bandsplit path."""
        if mixture.ndim == 3 and mixture.shape[1] == 1:
            mixture = mixture[:, 0]
        if mixture.ndim != 2:
            raise ValueError(f"Expected mixture [B,L] or [B,1,L], got {tuple(mixture.shape)}")
        window = self._window.to(device=mixture.device, dtype=torch.float32)
        spec = torch.stft(
            mixture.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        spec = spec / float(max(self.n_sources, 1))
        mag = spec.abs()  # [B,F,T]
        if self.downsample_coarse:
            mag = mag[:, ::2, :].contiguous()
        # The copied AudioEncoder bands expect at least 256 frequency bins and use bins 0:256.
        if mag.shape[1] < 256:
            mag = F.pad(mag, (0, 0, 0, 256 - mag.shape[1]))
        if mag.shape[1] > 256:
            mag = mag[:, :256, :].contiguous()
        self._log_shape("stft_mag", mag)
        return mag

    def forward_audio(self, mixture: torch.Tensor, target_t: Optional[int] = None) -> torch.Tensor:
        """MambaVoice-style audio feature path: waveform -> [B,T,512]."""
        mag = self._stft_magnitude(mixture)
        audio_feats = self.audio_net(mag)  # [B,T,512]
        if target_t is None:
            target_t = self.video_temporal_features * self._n
        if audio_feats.shape[1] != target_t:
            audio_feats = F.interpolate(
                audio_feats.transpose(1, 2),
                size=target_t,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        self._log_shape("audio_tokens", audio_feats)
        return audio_feats

    def forward(
        self,
        mixture: torch.Tensor,
        face: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        video = self.forward_visual(face) if face is not None else None
        target_t = video.shape[1] if video is not None else self.video_temporal_features * self._n
        audio = self.forward_audio(mixture, target_t=target_t)
        return {"audio": audio, "video": video}
