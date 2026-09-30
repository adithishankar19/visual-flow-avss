"""Four-level TFC-TDF U-Net velocity network (paper Sec. 3.4, Fig. 1b)."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device, dtype=t.dtype) * -1)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def _group_count(channels: int, requested: int = 32) -> int:
    """Choose a GroupNorm group count that divides channels."""
    requested = max(1, min(int(requested), int(channels)))
    for groups in range(requested, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _head_count(channels: int, requested: int) -> int:
    """Choose an attention head count that divides channels."""
    requested = max(1, min(int(requested), int(channels)))
    for heads in range(requested, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class TemporalFiLM2d(nn.Module):
    """Time-resolved FiLM conditioning broadcast along frequency."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if temporal_tokens is None:
            return x
        if temporal_tokens.ndim != 3:
            raise ValueError(
                f"temporal_tokens must be [B,T,C], got {tuple(temporal_tokens.shape)}"
            )
        tokens = temporal_tokens.to(device=x.device, dtype=x.dtype)
        if tokens.shape[1] != x.shape[-1]:
            tokens = F.interpolate(
                tokens.transpose(1, 2),
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        gamma, beta = self.proj(tokens).chunk(2, dim=-1)
        gamma = gamma.transpose(1, 2).unsqueeze(2)
        beta = beta.transpose(1, 2).unsqueeze(2)
        return x * (1.0 + gamma) + beta


class BottleneckTemporalCrossAttention(nn.Module):
    """Bottleneck cross-attention to the visual tokens, gated by reliability r."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        heads: int = 4,
        reliability_floor: float = 0.0,
    ) -> None:
        super().__init__()
        heads = _head_count(channels, heads)
        self.kv_proj = nn.Linear(cond_dim, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.reliability_floor = float(reliability_floor)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
        visual_reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if temporal_tokens is None:
            return x
        b, c, freq, time = x.shape
        tokens = temporal_tokens.to(device=x.device, dtype=x.dtype)
        if tokens.ndim != 3:
            raise ValueError(
                f"temporal_tokens must be [B,T,C], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] != time:
            tokens = F.interpolate(
                tokens.transpose(1, 2),
                size=time,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        q = x.permute(0, 2, 3, 1).reshape(b, freq * time, c)
        kv = self.kv_proj(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        if visual_reliability is not None:
            rel = visual_reliability.to(device=x.device, dtype=x.dtype)
            if rel.ndim == 2:
                rel = rel.unsqueeze(-1)
            if rel.ndim != 3:
                raise ValueError(
                    "visual_reliability must be [B,T] or [B,T,C], "
                    f"got {tuple(rel.shape)}"
                )
            if rel.shape[-1] != 1:
                rel = rel.mean(dim=-1, keepdim=True)
            if rel.shape[1] != time:
                rel = F.interpolate(
                    rel.transpose(1, 2),
                    size=time,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            rel = rel.clamp(0.0, 1.0)
            # r in [floor, 1]; the paper uses floor = 0.25.
            if self.reliability_floor > 0.0:
                rel = self.reliability_floor + (1.0 - self.reliability_floor) * rel
            rel = rel[:, None, :, :].expand(b, freq, time, 1).reshape(b, freq * time, 1)
            out = out * rel
        q = self.norm(q + out)
        return q.reshape(b, freq, time, c).permute(0, 3, 1, 2).contiguous()


class TDFBlock2d(nn.Module):
    """Low-rank time-distributed fully-connected frequency mixer.

    The same frequency MLP is applied independently to every channel and time
    frame. This is the TDF component used to complement local 2-D convolutions.
    """

    def __init__(
        self,
        freq_bins: int,
        *,
        bottleneck_factor: int = 16,
        dropout: float = 0.0,
        residual_scale: float = 2.0 ** -0.5,
    ) -> None:
        super().__init__()
        self.freq_bins = int(freq_bins)
        if self.freq_bins < 1:
            raise ValueError("freq_bins must be positive")
        self.residual_scale = float(residual_scale)
        hidden = max(8, self.freq_bins // max(1, int(bottleneck_factor)))
        self.norm = nn.LayerNorm(self.freq_bins)
        self.fc1 = nn.Linear(self.freq_bins, hidden)
        self.fc2 = nn.Linear(hidden, self.freq_bins)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1e-2)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.freq_bins:
            raise ValueError(
                f"TDFBlock2d expected {self.freq_bins} frequency bins, "
                f"got {x.shape[-2]}"
            )
        # [B,C,F,T] -> [B,C,T,F], apply shared dense frequency transform.
        h = x.permute(0, 1, 3, 2)
        h = self.norm(h)
        h = self.fc2(self.dropout(F.gelu(self.fc1(h), approximate="tanh")))
        h = h.permute(0, 1, 3, 2)
        return (x + h) * self.residual_scale


class TFCTDFBlock2d(nn.Module):
    """Residual TFC-TDF block with global flow/AV FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        freq_bins: int,
        emb_dim: int,
        *,
        norm_groups: int = 16,
        tdf_bottleneck_factor: int = 16,
        dropout: float = 0.0,
        residual_scale: float = 2.0 ** -0.5,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.residual_scale = float(residual_scale)

        self.norm0 = nn.GroupNorm(
            _group_count(self.in_channels, norm_groups), self.in_channels, eps=1e-6
        )
        self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.affine = nn.Linear(emb_dim, self.out_channels * 2)
        self.norm1 = nn.GroupNorm(
            _group_count(self.out_channels, norm_groups), self.out_channels, eps=1e-6
        )
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.tdf = TDFBlock2d(
            freq_bins,
            bottleneck_factor=tdf_bottleneck_factor,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        self.skip = (
            nn.Conv2d(self.in_channels, self.out_channels, 1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

        nn.init.xavier_uniform_(self.conv0.weight)
        nn.init.zeros_(self.conv0.bias)
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1e-2)
        nn.init.zeros_(self.conv1.bias)
        if isinstance(self.skip, nn.Conv2d):
            nn.init.xavier_uniform_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv0(F.silu(self.norm0(x)))
        scale, shift = self.affine(emb).to(h.dtype).chunk(2, dim=-1)
        h = self.norm1(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(self.dropout(F.silu(h)))
        h = (residual + h) * self.residual_scale
        return self.tdf(h)


class TFCTDFUNet(nn.Module):
    """Four-level TFC-TDF U-Net predicting the target velocity.

    Input is the path state stacked with the mixture, ``[z_t; M]``, as real and
    imaginary STFT channels.  The fused AV tokens modulate every level through
    temporal FiLM; pooled and concatenated with an embedding of ``t`` they set a
    scale and shift in every block.  At the bottleneck the visual tokens enter a
    cross-attention layer gated by the learned reliability ``r``.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        in_channels: int = 4,
        out_channels: int = 2,
        model_channels: int = 60,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        blocks_per_level: int = 2,
        embedding_dim: int = 384,
        time_dim: int = 128,
        input_freq_bins: int = 513,
        norm_groups: int = 16,
        tdf_bottleneck_factor: int = 16,
        dropout: float = 0.0,
        temporal_num_heads: int = 4,
        visual_reliability_floor: float = 0.25,
        residual_scale: str | float = "unit",
        exact_resample: bool = True,
    ) -> None:
        super().__init__()
        if len(channel_mult) != 4:
            raise ValueError("TFCTDFUNet requires exactly four levels")
        if blocks_per_level < 1:
            raise ValueError("blocks_per_level must be >= 1")

        self.cond_dim = int(cond_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_dim = int(time_dim)
        self.embedding_dim = int(embedding_dim)
        self.input_freq_bins = int(input_freq_bins)

        # Residual weighting inside every TFC-TDF block.
        #   "unit": plain (x + h); conv1/fc2 are initialised near zero, which is
        #       what keeps activations bounded at initialisation.
        #   "half": (x + h) * 2^-0.5, applied twice per block.
        if isinstance(residual_scale, str):
            key = residual_scale.lower()
            if key in {"half", "sqrt2", "legacy"}:
                res_scale = 2.0 ** -0.5
            elif key in {"unit", "one", "plain", "none"}:
                res_scale = 1.0
            else:
                raise ValueError(f"Unknown residual_scale={residual_scale!r}")
        else:
            res_scale = float(residual_scale)
        self.residual_scale = res_scale

        # With F=513 the three stride-2 encoder convs give 513->256->128->64 but
        # the matching transposed convs give 64->128->256->512.  `exact_resample`
        # pads the input to a multiple of 8 in both axes and crops the output
        # back, so every skip connection lines up without resampling.
        self.exact_resample = bool(exact_resample)
        if self.exact_resample:
            padded_bins = ((self.input_freq_bins + 7) // 8) * 8
        else:
            padded_bins = self.input_freq_bins
        self._padded_freq_bins = padded_bins

        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim + self.time_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
        )
        for module in self.cond_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        channels = [int(model_channels) * int(m) for m in channel_mult]
        freq_bins = [padded_bins]
        for _ in range(1, 4):
            freq_bins.append(freq_bins[-1] // 2)

        self.in_conv = nn.Conv2d(self.in_channels, channels[0], 3, padding=1)
        nn.init.xavier_uniform_(self.in_conv.weight)
        nn.init.zeros_(self.in_conv.bias)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_films = nn.ModuleList()
        self.downs = nn.ModuleList()

        current = channels[0]
        for level, (out_ch, bins) in enumerate(zip(channels, freq_bins)):
            blocks = nn.ModuleList()
            films = nn.ModuleList()
            for block_idx in range(int(blocks_per_level)):
                block_in = current if block_idx == 0 else out_ch
                blocks.append(
                    TFCTDFBlock2d(
                        block_in,
                        out_ch,
                        bins,
                        self.embedding_dim,
                        norm_groups=norm_groups,
                        tdf_bottleneck_factor=tdf_bottleneck_factor,
                        dropout=dropout,
                        residual_scale=res_scale,
                    )
                )
                films.append(TemporalFiLM2d(self.cond_dim, out_ch))
                current = out_ch
            self.encoder_blocks.append(blocks)
            self.encoder_films.append(films)
            if level < 3:
                self.downs.append(
                    nn.Conv2d(current, channels[level + 1], 4, stride=2, padding=1)
                )
                nn.init.xavier_uniform_(self.downs[-1].weight)
                nn.init.zeros_(self.downs[-1].bias)
                current = channels[level + 1]

        bottleneck_bins = freq_bins[-1]
        self.bottleneck0 = TFCTDFBlock2d(
            current,
            current,
            bottleneck_bins,
            self.embedding_dim,
            norm_groups=norm_groups,
            tdf_bottleneck_factor=tdf_bottleneck_factor,
            dropout=dropout,
            residual_scale=res_scale,
        )
        self.bottleneck1 = TFCTDFBlock2d(
            current,
            current,
            bottleneck_bins,
            self.embedding_dim,
            norm_groups=norm_groups,
            tdf_bottleneck_factor=tdf_bottleneck_factor,
            dropout=dropout,
            residual_scale=res_scale,
        )
        self.bottleneck_film = TemporalFiLM2d(self.cond_dim, current)
        self.bottleneck_cross_attn = BottleneckTemporalCrossAttention(
            current,
            self.cond_dim,
            temporal_num_heads,
            reliability_floor=visual_reliability_floor,
        )

        self.ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        for level in reversed(range(3)):
            out_ch = channels[level]
            self.ups.append(
                nn.ConvTranspose2d(current, out_ch, 4, stride=2, padding=1)
            )
            nn.init.xavier_uniform_(self.ups[-1].weight)
            nn.init.zeros_(self.ups[-1].bias)
            current = out_ch

            blocks = nn.ModuleList()
            films = nn.ModuleList()
            for block_idx in range(int(blocks_per_level)):
                block_in = current + out_ch if block_idx == 0 else out_ch
                blocks.append(
                    TFCTDFBlock2d(
                        block_in,
                        out_ch,
                        freq_bins[level],
                        self.embedding_dim,
                        norm_groups=norm_groups,
                        tdf_bottleneck_factor=tdf_bottleneck_factor,
                        dropout=dropout,
                        residual_scale=res_scale,
                    )
                )
                films.append(TemporalFiLM2d(self.cond_dim, out_ch))
                current = out_ch
            self.decoder_blocks.append(blocks)
            self.decoder_films.append(films)

        self.out_norm = nn.GroupNorm(
            _group_count(current, norm_groups), current, eps=1e-6
        )
        self.out_conv = nn.Conv2d(current, self.out_channels, 3, padding=1)
        nn.init.xavier_uniform_(self.out_conv.weight, gain=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _align(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        mixture: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor] = None,
        visual_activity: Optional[torch.Tensor] = None,
        cross_attention_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.ndim != 4 or mixture.ndim != 4:
            raise ValueError("x_t and mixture must both be [B,C,F,T]")
        if cond.ndim != 2:
            raise ValueError("cond must be [B,D]")
        if x_t.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("x_t and mixture must have matching STFT dimensions")
        if x_t.shape[-2] != self.input_freq_bins:
            raise ValueError(
                f"Configured input_freq_bins={self.input_freq_bins}, "
                f"but input has {x_t.shape[-2]} bins"
            )

        h = torch.cat([x_t, mixture], dim=1)
        if h.shape[1] != self.in_channels:
            raise ValueError(
                f"Configured in_channels={self.in_channels}, got {h.shape[1]}"
            )

        pad_f = pad_t = 0
        if self.exact_resample:
            orig_f, orig_t = h.shape[-2], h.shape[-1]
            pad_f = self._padded_freq_bins - orig_f
            pad_t = (-orig_t) % 8
            if pad_f or pad_t:
                # Replicate rather than zero-pad: a hard zero band at the top of
                # the spectrum is an edge the convolutions would have to learn
                # around.
                h = F.pad(h, (0, pad_t, 0, pad_f), mode="replicate")

        emb = self.cond_proj(torch.cat([cond, sinusoidal_embedding(t, self.time_dim)], dim=-1))
        h = self.in_conv(h)
        skips = []

        for level, blocks in enumerate(self.encoder_blocks):
            for block, film in zip(blocks, self.encoder_films[level]):
                h = block(h, emb)
                h = film(h, temporal_tokens)
            skips.append(h)
            if level < 3:
                h = self.downs[level](h)

        h = self.bottleneck0(h, emb)
        h = self.bottleneck1(h, emb)
        h = self.bottleneck_film(h, temporal_tokens)
        attn_tokens = cross_attention_tokens if cross_attention_tokens is not None else temporal_tokens
        h = self.bottleneck_cross_attn(h, attn_tokens, visual_activity)

        # The deepest encoder output is the bottleneck input, so decode against
        # the three shallower skips only.
        for decoder_index, level in enumerate(reversed(range(3))):
            h = self.ups[decoder_index](h)
            skip = skips[level]
            h = self._align(h, skip)
            h = torch.cat([h, skip], dim=1)
            for block, film in zip(
                self.decoder_blocks[decoder_index],
                self.decoder_films[decoder_index],
            ):
                h = block(h, emb)
                h = film(h, temporal_tokens)

        out = self.out_conv(F.silu(self.out_norm(h)))
        if pad_f or pad_t:
            out = out[..., : out.shape[-2] - pad_f, : out.shape[-1] - pad_t]
        return out
