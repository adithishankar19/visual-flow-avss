from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .flow_head import sinusoidal_embedding


def _group_count(channels: int, requested: int = 8) -> int:
    for g in (requested, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FiLM2d(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class TemporalFiLM2d(nn.Module):
    """Time-resolved FiLM modulation for U-Net feature maps.

    temporal_tokens are [B, T_tok, cond_dim]. They are linearly interpolated
    to the current feature-map time length and projected to per-frame gamma/beta.
    The modulation is broadcast over frequency: [B, C, 1, T].

    The projection is zero-initialized, so enabling this module starts as a
    no-op and can be safely compared against the previous architecture.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, temporal_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if temporal_tokens is None:
            return x
        if temporal_tokens.ndim != 3:
            raise ValueError(
                f"temporal_tokens must be [B,T,C], got {tuple(temporal_tokens.shape)}"
            )
        tokens = temporal_tokens.to(device=x.device, dtype=x.dtype)
        time_len = x.shape[-1]
        if tokens.shape[1] != time_len:
            tokens = F.interpolate(
                tokens.transpose(1, 2),
                size=time_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        gamma, beta = self.proj(tokens).chunk(2, dim=-1)  # [B,T,C], [B,T,C]
        gamma = gamma.transpose(1, 2).unsqueeze(2)        # [B,C,1,T]
        beta = beta.transpose(1, 2).unsqueeze(2)
        return x * (1.0 + gamma) + beta


class ResConv2dBlock(nn.Module):
    """Residual 2-D conv block with FiLM conditioning.

    The FiLM condition should already include the flow-time embedding and the
    MambaVoice audio-visual conditioning vector.
    """

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.film = FiLM2d(cond_dim, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Downsample2d(nn.Module):
    def __init__(self, channels: int, mode: str = "spatial") -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode in {"spatial", "both", "ft", "time_freq"}:
            stride = (2, 2)
        elif mode in {"frequency", "freq", "f"}:
            stride = (2, 1)
        elif mode in {"time", "temporal", "t"}:
            stride = (1, 2)
        elif mode in {"none", "identity", "no"}:
            stride = (1, 1)
        else:
            raise ValueError(f"Unknown downsample mode {mode!r}")
        self.stride = stride
        if stride == (1, 1):
            self.op = nn.Identity()
        else:
            self.op = nn.Conv2d(channels, channels, 4, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int, down_mode: str, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, out_ch, 3, padding=1) if in_ch != out_ch else nn.Identity()
        self.block1 = ResConv2dBlock(out_ch, cond_dim, kernel_size, dropout)
        self.block2 = ResConv2dBlock(out_ch, cond_dim, kernel_size, dropout)
        self.down = Downsample2d(out_ch, down_mode)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        h = self.block1(h, cond)
        h = self.block2(h, cond)
        skip = h
        h = self.down(h)
        return h, skip


class DecoderLevel(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.block1 = ResConv2dBlock(out_ch, cond_dim, kernel_size, dropout)
        self.block2 = ResConv2dBlock(out_ch, cond_dim, kernel_size, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([x, skip], dim=1)
        h = self.in_proj(h)
        h = self.block1(h, cond)
        h = self.block2(h, cond)
        return h


class SpecUNetFlowHead(nn.Module):
    """Complex-spectrogram 2-D U-Net velocity head.

    Inputs:
      x_t: [B, 4, F, T] current source-pair state in real channels
           channels are target_real, target_imag, residual_real, residual_imag
      mixture: [B, 2, F, T] complex mixture in real channels
               channels are mixture_real, mixture_imag
      cond: [B, D] MambaVoice audio+video conditioning
      t: [B] rectified-flow time

    Output:
      velocity: [B, 4, F, T]
    """

    def __init__(
        self,
        cond_dim: int = 256,
        channels: Sequence[int] = (64, 128, 256, 256, 256, 256),
        downsample_modes: Sequence[str] = ("spatial", "spatial", "spatial", "spatial", "frequency", "frequency"),
        time_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.0,
        in_channels: int = 6,
        out_channels: int = 4,
        temporal_conditioning: str = "none",
        temporal_num_heads: int = 4,
        visual_activity_input: bool = False,
        visual_activity_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("SpecUNetFlowHead needs at least two channel levels")
        if len(downsample_modes) != len(channels):
            raise ValueError("downsample_modes must have the same length as channels")
        self.time_dim = int(time_dim)
        self.cond_dim = int(cond_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.visual_activity_input = bool(visual_activity_input)
        self.visual_activity_channels = int(visual_activity_channels)
        if self.visual_activity_channels < 1:
            raise ValueError("visual_activity_channels must be >= 1")
        self.temporal_conditioning = str(temporal_conditioning).lower()
        self.temporal_num_heads = int(temporal_num_heads)
        self.use_multiscale_temporal_film = False
        self.use_bottleneck_cross_attn = False
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        chs = list(map(int, channels))
        actual_in_channels = self.in_channels + (self.visual_activity_channels if self.visual_activity_input else 0)
        self.in_conv = nn.Conv2d(actual_in_channels, chs[0], 3, padding=1)
        if self.visual_activity_input:
            # Start from the old behavior: the new visual-prior input channel
            # initially has no effect, but its weights can learn immediately.
            with torch.no_grad():
                #self.in_conv.weight[:, self.in_channels:, :, :].zero_()
                nn.init.normal_(self.in_conv.weight[:, self.in_channels:, :, :], mean=0.0, std=1e-3)
        encoders = []
        prev = chs[0]
        for ch, mode in zip(chs, downsample_modes):
            encoders.append(EncoderLevel(prev, ch, cond_dim, kernel_size, mode, dropout))
            prev = ch
        self.encoders = nn.ModuleList(encoders)
        self.bottleneck = nn.Sequential(
            ResConv2dBlock(chs[-1], cond_dim, kernel_size, dropout),
            ResConv2dBlock(chs[-1], cond_dim, kernel_size, dropout),
        )

        # Optional time-resolved AV/visual conditioning.
        # Modes:
        #   none:                         old global-only behavior
        #   bottleneck_bias:              add temporal bias at the bottleneck
        #   bottleneck_cross_attn:        bottleneck queries temporal tokens
        #   multiscale_film:              temporal FiLM from the first U-Net features onward
        #   multiscale_film_cross_attn:   temporal FiLM + bottleneck cross-attention
        if self.temporal_conditioning in {"none", "off", "false"}:
            self.temporal_conditioning = "none"
        elif self.temporal_conditioning in {"bottleneck_bias", "bias", "temporal_bias"}:
            self.temporal_conditioning = "bottleneck_bias"
            self.temporal_bottleneck_proj = nn.Linear(cond_dim, chs[-1])
            nn.init.zeros_(self.temporal_bottleneck_proj.weight)
            nn.init.zeros_(self.temporal_bottleneck_proj.bias)
        elif self.temporal_conditioning in {"bottleneck_cross_attn", "cross_attn", "attention", "attn"}:
            self.temporal_conditioning = "bottleneck_cross_attn"
            self.use_bottleneck_cross_attn = True
        elif self.temporal_conditioning in {"multiscale_film", "temporal_film", "film"}:
            self.temporal_conditioning = "multiscale_film"
            self.use_multiscale_temporal_film = True
        elif self.temporal_conditioning in {
            "multiscale_film_cross_attn",
            "film_cross_attn",
            "cross_attn_film",
            "multiscale_cross_attn",
        }:
            self.temporal_conditioning = "multiscale_film_cross_attn"
            self.use_multiscale_temporal_film = True
            self.use_bottleneck_cross_attn = True
        else:
            raise ValueError(f"Unknown temporal_conditioning={self.temporal_conditioning!r}")

        if self.use_multiscale_temporal_film:
            self.temporal_input_film = TemporalFiLM2d(cond_dim, chs[0])
            self.temporal_encoder_skip_films = nn.ModuleList([
                TemporalFiLM2d(cond_dim, ch) for ch in chs
            ])
            self.temporal_encoder_down_films = nn.ModuleList([
                TemporalFiLM2d(cond_dim, ch) for ch in chs
            ])
            self.temporal_bottleneck_film = TemporalFiLM2d(cond_dim, chs[-1])
        else:
            self.temporal_input_film = None
            self.temporal_encoder_skip_films = nn.ModuleList()
            self.temporal_encoder_down_films = nn.ModuleList()
            self.temporal_bottleneck_film = None

        if self.use_bottleneck_cross_attn:
            self.temporal_kv_proj = nn.Linear(cond_dim, chs[-1])
            heads = max(1, self.temporal_num_heads)
            if chs[-1] % heads != 0:
                heads = _group_count(chs[-1], heads)
            self.temporal_attn = nn.MultiheadAttention(chs[-1], heads, batch_first=True)
            self.temporal_attn_norm = nn.LayerNorm(chs[-1])
            # Start close to the previous model: attention output is initially zero.
            nn.init.zeros_(self.temporal_attn.out_proj.weight)
            nn.init.zeros_(self.temporal_attn.out_proj.bias)

        decoders = []
        cur = chs[-1]
        for skip_ch in reversed(chs):
            out_ch = skip_ch
            decoders.append(DecoderLevel(cur, skip_ch, out_ch, cond_dim, kernel_size, dropout))
            cur = out_ch
        self.decoders = nn.ModuleList(decoders)
        if self.use_multiscale_temporal_film:
            self.temporal_decoder_films = nn.ModuleList([
                TemporalFiLM2d(cond_dim, ch) for ch in reversed(chs)
            ])
        else:
            self.temporal_decoder_films = nn.ModuleList()
        self.out_norm = nn.GroupNorm(_group_count(cur), cur)
        self.out_conv = nn.Conv2d(cur, self.out_channels, 3, padding=1)

        # Small output init helps avoid unstable initial Euler steps.
        nn.init.normal_(self.out_conv.weight,mean=0.0,std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def _align_temporal_tokens(self, temporal_tokens: torch.Tensor, time_len: int) -> torch.Tensor:
        """Return temporal tokens aligned to the current U-Net time length."""
        if temporal_tokens.ndim != 3:
            raise ValueError(
                f"temporal_tokens must be [B,T,C], got {tuple(temporal_tokens.shape)}"
            )
        if temporal_tokens.shape[1] != time_len:
            temporal_tokens = F.interpolate(
                temporal_tokens.transpose(1, 2),
                size=time_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return temporal_tokens

    def _inject_temporal_tokens(
        self,
        h: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Inject temporal AV tokens into bottleneck features h [B,C,F,T]."""
        if temporal_tokens is None or self.temporal_conditioning == "none":
            return h

        b, c, freq_len, time_len = h.shape
        temporal_tokens = self._align_temporal_tokens(temporal_tokens, time_len)

        if self.temporal_conditioning == "bottleneck_bias":
            temp = self.temporal_bottleneck_proj(temporal_tokens)  # [B,T,C]
            temp = temp.transpose(1, 2).unsqueeze(2)              # [B,C,1,T]
            return h + temp

        if self.use_bottleneck_cross_attn:
            q = h.permute(0, 2, 3, 1).reshape(b, freq_len * time_len, c)
            kv = self.temporal_kv_proj(temporal_tokens)
            attn_out, _ = self.temporal_attn(q, kv, kv, need_weights=False)
            q = self.temporal_attn_norm(q + attn_out)
            return q.reshape(b, freq_len, time_len, c).permute(0, 3, 1, 2).contiguous()

        return h

    def _apply_temporal_film(
        self,
        film: Optional[TemporalFiLM2d],
        x: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if film is None or temporal_tokens is None:
            return x
        return film(x, temporal_tokens)

    def _make_visual_activity_channel(
        self,
        visual_activity: Optional[torch.Tensor],
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Return a [B,1,F,T] visual activity prior channel.

        Accepted inputs:
          - [B,T,C] from the conditioner, including raw landmark-derived maps
          - [B,T], [B,1,T], or [B,C,T]
          - [B,C,F,T] already expanded

        The output is interpolated to the SpecUNet time resolution and broadcast
        across frequency.  Unlike the original one-channel activity prior, this
        version preserves multiple visual map channels.  Extra/missing channels
        are truncated/padded to self.visual_activity_channels so configs remain
        explicit and checkpoints have stable input shapes.
        """
        b, _, freq_len, time_len = x_t.shape
        c_expected = self.visual_activity_channels
        if visual_activity is None:
            return torch.zeros(b, c_expected, freq_len, time_len, device=x_t.device, dtype=x_t.dtype)

        a = visual_activity.to(device=x_t.device, dtype=x_t.dtype)
        if a.ndim == 4:
            # [B,C,F,T].
            if a.shape[-2:] != x_t.shape[-2:]:
                a = F.interpolate(a, size=x_t.shape[-2:], mode="bilinear", align_corners=False)
            if a.shape[1] < c_expected:
                pad = torch.zeros(b, c_expected - a.shape[1], freq_len, time_len, device=x_t.device, dtype=x_t.dtype)
                a = torch.cat([a, pad], dim=1)
            return a[:, :c_expected]

        if a.ndim == 2:
            a = a.unsqueeze(-1)  # [B,T,1]
        if a.ndim == 3:
            # Prefer [B,T,C] from the conditioner. Also support [B,C,T].
            if a.shape[1] == c_expected and a.shape[-1] != c_expected:
                pass  # [B,C,T]
            else:
                a = a.transpose(1, 2)  # [B,C,T]
        else:
            raise ValueError(f"Unsupported visual_activity shape: {tuple(a.shape)}")

        if a.shape[-1] != time_len:
            a = F.interpolate(a, size=time_len, mode="linear", align_corners=False)
        if a.shape[1] < c_expected:
            pad = torch.zeros(b, c_expected - a.shape[1], time_len, device=x_t.device, dtype=x_t.dtype)
            a = torch.cat([a, pad], dim=1)
        a = a[:, :c_expected]
        return a.unsqueeze(2).expand(-1, -1, freq_len, -1)

    def forward(
        self,
        x_t: torch.Tensor,
        mixture: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor] = None,
        visual_activity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"SpecUNetFlowHead expected x_t [B,4,F,T], got {tuple(x_t.shape)}")
        if mixture.ndim != 4:
            raise ValueError(f"SpecUNetFlowHead expected mixture [B,2,F,T], got {tuple(mixture.shape)}")
        if mixture.shape[-2:] != x_t.shape[-2:]:
            f = min(mixture.shape[-2], x_t.shape[-2])
            tt = min(mixture.shape[-1], x_t.shape[-1])
            mixture = mixture[..., :f, :tt]
            x_t = x_t[..., :f, :tt]

        temb = sinusoidal_embedding(t, self.time_dim)
        c = self.cond_proj(torch.cat([cond, temb], dim=-1))
        unet_inputs = [x_t, mixture]
        if self.visual_activity_input:
            unet_inputs.append(self._make_visual_activity_channel(visual_activity, x_t))
        h = self.in_conv(torch.cat(unet_inputs, dim=1))
        h = self._apply_temporal_film(self.temporal_input_film, h, temporal_tokens)
        skips = []
        for idx, enc in enumerate(self.encoders):
            h, skip = enc(h, c)
            if self.use_multiscale_temporal_film:
                skip = self.temporal_encoder_skip_films[idx](skip, temporal_tokens)
                h = self.temporal_encoder_down_films[idx](h, temporal_tokens)
            skips.append(skip)
        for block in self.bottleneck:
            h = block(h, c)
        h = self._apply_temporal_film(self.temporal_bottleneck_film, h, temporal_tokens)
        h = self._inject_temporal_tokens(h, temporal_tokens)
        for idx, (dec, skip) in enumerate(zip(self.decoders, reversed(skips))):
            h = dec(h, skip, c)
            if self.use_multiscale_temporal_film:
                h = self.temporal_decoder_films[idx](h, temporal_tokens)
        return self.out_conv(F.silu(self.out_norm(h)))
