from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .flow_head import sinusoidal_embedding


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


def _checkpoint(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Checkpoint a one-input module across supported PyTorch versions."""
    from torch.utils.checkpoint import checkpoint

    try:
        return checkpoint(module, x, use_reentrant=False)
    except TypeError:  # PyTorch < 2.0
        return checkpoint(module, x)


class EqualBandSplit2d(nn.Module):
    """Move equal contiguous frequency bands into the channel dimension.

    [B, C, F, T] -> [B, C * N, ceil(F / N), T]

    Diff-VS uses four equal frequency splits before its DDPM++ U-Net. Padding is
    applied only at the upper-frequency edge and removed after the output merge.
    """

    def __init__(self, num_splits: int = 4) -> None:
        super().__init__()
        self.num_splits = int(num_splits)
        if self.num_splits < 1:
            raise ValueError("num_splits must be >= 1")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        b, c, freq, time = x.shape
        pad_freq = (-freq) % self.num_splits
        if pad_freq:
            x = F.pad(x, (0, 0, 0, pad_freq))
        padded_freq = freq + pad_freq
        band_freq = padded_freq // self.num_splits
        x = x.reshape(b, c, self.num_splits, band_freq, time)
        x = x.permute(0, 2, 1, 3, 4).reshape(
            b, self.num_splits * c, band_freq, time
        )
        return x, freq, pad_freq

    def merge(self, x: torch.Tensor, logical_channels: int, original_freq: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        b, channels, band_freq, time = x.shape
        expected = int(logical_channels) * self.num_splits
        if channels != expected:
            raise ValueError(
                f"Band merge expected {expected} channels, got {channels}"
            )
        x = x.reshape(b, self.num_splits, logical_channels, band_freq, time)
        x = x.permute(0, 2, 1, 3, 4).reshape(
            b, logical_channels, self.num_splits * band_freq, time
        )
        return x[..., :original_freq, :]


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


class DiffVSResidualBlock(nn.Module):
    """DDPM++-style residual block with frequency-only resampling.

    The Diff-VS paper removes time-axis down/up-sampling. Frequency resizing is
    performed with pooling/interpolation rather than strided transpose
    convolutions to avoid temporal aliasing and preserve AV alignment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        *,
        dropout: float = 0.0,
        resample: str = "none",
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.resample = str(resample).lower()
        if self.resample not in {"none", "down", "up"}:
            raise ValueError(f"Unknown resample mode {resample!r}")

        self.norm0 = nn.GroupNorm(
            _group_count(self.in_channels, groups), self.in_channels, eps=1e-6
        )
        self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.affine = nn.Linear(emb_dim, self.out_channels * 2)
        self.norm1 = nn.GroupNorm(
            _group_count(self.out_channels, groups), self.out_channels, eps=1e-6
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(self.in_channels, self.out_channels, 1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

        nn.init.xavier_uniform_(self.conv0.weight)
        nn.init.zeros_(self.conv0.bias)
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1e-3)
        nn.init.zeros_(self.conv1.bias)
        if isinstance(self.skip, nn.Conv2d):
            nn.init.xavier_uniform_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample == "down":
            return F.avg_pool2d(
                x, kernel_size=(2, 1), stride=(2, 1), ceil_mode=True
            )
        if self.resample == "up":
            return F.interpolate(x, scale_factor=(2.0, 1.0), mode="nearest")
        return x

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(self._resize(x))
        h = self._resize(x)
        h = self.conv0(F.silu(self.norm0(h)))
        scale, shift = self.affine(emb).to(h.dtype).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = F.silu(self.norm1(h) * (1.0 + scale) + shift)
        h = self.conv1(self.dropout(h))
        return (residual + h) * (2.0 ** -0.5)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.scale = dim ** 0.5
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(x.float(), dim=-1, eps=self.eps).to(x.dtype)
        return normed * self.scale * self.weight.to(x.dtype)


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply 1-D RoPE to [B,H,N,D], computing rotations in FP32."""
    head_dim = x.shape[-1]
    rotary_dim = head_dim - (head_dim % 2)
    if rotary_dim == 0:
        return x

    x_rot = x[..., :rotary_dim].float()
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, half, device=x.device, dtype=torch.float32)
            / max(1, half)
        )
    )
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    even = x_rot[..., 0::2]
    odd = x_rot[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    rotated = rotated.to(x.dtype)
    if x_pass.numel() == 0:
        return rotated
    return torch.cat([rotated, x_pass], dim=-1)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = _head_count(self.dim, heads)
        self.head_dim = self.dim // self.heads
        self.norm = RMSNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.out = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()
        self.attn_dropout = float(attn_dropout)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = _apply_rope(q)
        k = _apply_rope(k)

        dropout_p = self.attn_dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=False
            )
        else:  # pragma: no cover - compatibility path for old torch
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            weights = scores.softmax(dim=-1).to(v.dtype)
            if dropout_p > 0:
                weights = F.dropout(weights, p=dropout_p, training=True)
            attended = torch.matmul(weights, v)

        attended = attended.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj_dropout(self.out(attended))


class RoPETransformerLayer(nn.Module):
    """Pre-norm RoFormer layer with the stabilization choices from Diff-VS."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = RoPEAttention(
            dim,
            heads=heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )
        hidden = max(dim, int(round(dim * ff_mult)))
        self.ff_norm = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        nn.init.xavier_uniform_(self.ff[0].weight)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.xavier_uniform_(self.ff[3].weight)
        nn.init.zeros_(self.ff[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(self.ff_norm(x))
        return x


class DualPathRoFormer2d(nn.Module):
    """Alternating RoPE attention along time and frequency."""

    def __init__(
        self,
        channels: int,
        *,
        depth: int = 1,
        heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        axis_batch_chunk: int = 0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.axis_batch_chunk = int(axis_batch_chunk)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.time_layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    self.channels, heads=heads, ff_mult=ff_mult, dropout=dropout
                )
                for _ in range(int(depth))
            ]
        )
        self.freq_layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    self.channels, heads=heads, ff_mult=ff_mult, dropout=dropout
                )
                for _ in range(int(depth))
            ]
        )

    def _run_layers(self, x: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        chunk = self.axis_batch_chunk
        if chunk <= 0 or x.shape[0] <= chunk:
            for layer in layers:
                x = layer(x)
            return x
        outputs = []
        for start in range(0, x.shape[0], chunk):
            part = x[start : start + chunk]
            for layer in layers:
                part = layer(part)
            outputs.append(part)
        return torch.cat(outputs, dim=0)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        b, c, freq, time = x.shape
        time_seq = x.permute(0, 2, 3, 1).reshape(b * freq, time, c)
        time_seq = self._run_layers(time_seq, self.time_layers)
        x = time_seq.reshape(b, freq, time, c).permute(0, 3, 1, 2).contiguous()

        freq_seq = x.permute(0, 3, 2, 1).reshape(b * time, freq, c)
        freq_seq = self._run_layers(freq_seq, self.freq_layers)
        return freq_seq.reshape(b, time, freq, c).permute(0, 3, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return _checkpoint(self._forward_impl, x)
        return self._forward_impl(x)


class BottleneckTemporalCrossAttention(nn.Module):
    """Let bottleneck spectrogram cells attend to time-resolved AV tokens."""

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
            if self.reliability_floor > 0.0:
                rel = self.reliability_floor + (1.0 - self.reliability_floor) * rel
            rel = rel[:, None, :, :].expand(b, freq, time, 1).reshape(b, freq * time, 1)
            out = out * rel
        q = self.norm(q + out)
        return q.reshape(b, freq, time, c).permute(0, 3, 1, 2).contiguous()


class DiffVSUNetFlowHead(nn.Module):
    """Diff-VS-inspired audio-aware DDPM++ U-Net used as a flow/drift head.

    This ports the *architecture* of Diff-VS into the existing MambaVoice
    rectified-flow project. It intentionally does not replace the surrounding
    training objective with EDM: the current one-step drift/flow loss and AV
    conditioner remain unchanged.

    Adapted Diff-VS components:
      * equal-width frequency band split at the input/output;
      * DDPM++ residual U-Net with sinusoidal flow-time conditioning;
      * no time-axis down/up-sampling;
      * dual-path RoFormer blocks over time and frequency;
      * Xavier initialization, FP32 RoPE, tanh-approximate GELU;
      * optional Diff-VS complex-magnitude input compression.

    Inputs and outputs use the same contract as SpecUNetFlowHead.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        in_channels: int = 6,
        out_channels: int = 4,
        model_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 4,
        embedding_dim: int = 1024,
        time_dim: int = 256,
        band_splits: int = 4,
        dropout: float = 0.0,
        norm_groups: int = 32,
        roformer_depth: int = 1,
        roformer_heads: int = 8,
        roformer_ff_mult: float = 4.0,
        roformer_dropout: float = 0.0,
        encoder_roformer: bool = True,
        decoder_roformer_levels: Sequence[int] = (0,),
        bottleneck_roformer: bool = True,
        axis_batch_chunk: int = 0,
        gradient_checkpointing: bool = True,
        temporal_conditioning: str = "none",
        temporal_num_heads: int = 4,
        visual_activity_input: bool = False,
        visual_activity_channels: int = 1,
        input_power_compression: bool = False,
        compression_alpha: float = 0.667,
        compression_beta: float = 0.065,
    ) -> None:
        super().__init__()
        if len(channel_mult) < 2:
            raise ValueError("DiffVSUNetFlowHead needs at least two levels")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        if in_channels % 2 != 0:
            raise ValueError(
                "DiffVS audio input channels must be real/imag pairs (an even count)"
            )

        self.cond_dim = int(cond_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_dim = int(time_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.band_split = EqualBandSplit2d(band_splits)
        self.band_splits = int(band_splits)
        self.visual_activity_input = bool(visual_activity_input)
        self.visual_activity_channels = int(visual_activity_channels)
        self.input_power_compression = bool(input_power_compression)
        self.compression_alpha = float(compression_alpha)
        self.compression_beta = float(compression_beta)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        temporal_mode = str(temporal_conditioning).lower()
        if temporal_mode in {"none", "off", "false"}:
            temporal_mode = "none"
        elif temporal_mode in {"bottleneck_bias", "bias", "temporal_bias"}:
            temporal_mode = "bottleneck_bias"
        elif temporal_mode in {"bottleneck_cross_attn", "cross_attn", "attention", "attn"}:
            temporal_mode = "bottleneck_cross_attn"
        elif temporal_mode in {"multiscale_film", "temporal_film", "film"}:
            temporal_mode = "multiscale_film"
        elif temporal_mode in {
            "multiscale_film_cross_attn",
            "film_cross_attn",
            "cross_attn_film",
            "multiscale_cross_attn",
        }:
            temporal_mode = "multiscale_film_cross_attn"
        else:
            raise ValueError(f"Unknown temporal_conditioning={temporal_mode!r}")
        self.temporal_conditioning = temporal_mode
        self.use_temporal_film = temporal_mode in {
            "multiscale_film",
            "multiscale_film_cross_attn",
        }
        self.use_temporal_cross_attn = temporal_mode in {
            "bottleneck_cross_attn",
            "multiscale_film_cross_attn",
        }
        self.use_temporal_bias = temporal_mode == "bottleneck_bias"

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

        internal_in = self.in_channels * self.band_splits
        if self.visual_activity_input:
            internal_in += self.visual_activity_channels
        base_channels = int(model_channels)
        self.in_conv = nn.Conv2d(internal_in, base_channels, 3, padding=1)
        nn.init.xavier_uniform_(self.in_conv.weight)
        nn.init.zeros_(self.in_conv.bias)

        rope_kwargs = dict(
            depth=int(roformer_depth),
            heads=int(roformer_heads),
            ff_mult=float(roformer_ff_mult),
            dropout=float(roformer_dropout),
            axis_batch_chunk=int(axis_batch_chunk),
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # Encoder. The skip bookkeeping follows SongUNet/DDPM++ exactly:
        # input conv + each downsample block + every residual block.
        self.encoder_downs = nn.ModuleList()
        self.encoder_down_films = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        self.encoder_ropes = nn.ModuleList()
        self.encoder_films = nn.ModuleList()

        skip_channels = [base_channels]
        current = base_channels
        level_channels = [base_channels * int(mult) for mult in channel_mult]
        for level, out_ch in enumerate(level_channels):
            if level > 0:
                self.encoder_downs.append(
                    DiffVSResidualBlock(
                        current,
                        current,
                        self.embedding_dim,
                        dropout=dropout,
                        resample="down",
                        groups=norm_groups,
                    )
                )
                self.encoder_down_films.append(
                    TemporalFiLM2d(self.cond_dim, current)
                    if self.use_temporal_film
                    else nn.Identity()
                )
                skip_channels.append(current)

            blocks = nn.ModuleList()
            ropes = nn.ModuleList()
            films = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(
                    DiffVSResidualBlock(
                        current,
                        out_ch,
                        self.embedding_dim,
                        dropout=dropout,
                        groups=norm_groups,
                    )
                )
                current = out_ch
                ropes.append(
                    DualPathRoFormer2d(current, **rope_kwargs)
                    if encoder_roformer
                    else nn.Identity()
                )
                films.append(
                    TemporalFiLM2d(self.cond_dim, current)
                    if self.use_temporal_film
                    else nn.Identity()
                )
                skip_channels.append(current)
            self.encoder_blocks.append(blocks)
            self.encoder_ropes.append(ropes)
            self.encoder_films.append(films)

        # DDPM++ bottleneck: residual -> attention/RoFormer -> residual.
        self.bottleneck0 = DiffVSResidualBlock(
            current,
            current,
            self.embedding_dim,
            dropout=dropout,
            groups=norm_groups,
        )
        self.bottleneck_rope = (
            DualPathRoFormer2d(current, **rope_kwargs)
            if bottleneck_roformer
            else nn.Identity()
        )
        self.bottleneck1 = DiffVSResidualBlock(
            current,
            current,
            self.embedding_dim,
            dropout=dropout,
            groups=norm_groups,
        )
        self.bottleneck_film = (
            TemporalFiLM2d(self.cond_dim, current)
            if self.use_temporal_film
            else nn.Identity()
        )
        self.bottleneck_cross_attn = (
            BottleneckTemporalCrossAttention(current, self.cond_dim, temporal_num_heads)
            if self.use_temporal_cross_attn
            else nn.Identity()
        )
        self.bottleneck_bias = (
            nn.Linear(self.cond_dim, current) if self.use_temporal_bias else None
        )
        if self.bottleneck_bias is not None:
            nn.init.zeros_(self.bottleneck_bias.weight)
            nn.init.zeros_(self.bottleneck_bias.bias)

        # Decoder consumes all stored skips, num_res_blocks + 1 per level.
        decoder_roformer_levels = {int(v) for v in decoder_roformer_levels}
        remaining_skips = list(skip_channels)
        self.decoder_ups = nn.ModuleList()
        self.decoder_up_films = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ropes = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        self.decoder_level_ids: list[int] = []

        reversed_levels = list(reversed(range(len(level_channels))))
        for decoder_position, level in enumerate(reversed_levels):
            out_ch = level_channels[level]
            if decoder_position > 0:
                self.decoder_ups.append(
                    DiffVSResidualBlock(
                        current,
                        current,
                        self.embedding_dim,
                        dropout=dropout,
                        resample="up",
                        groups=norm_groups,
                    )
                )
                self.decoder_up_films.append(
                    TemporalFiLM2d(self.cond_dim, current)
                    if self.use_temporal_film
                    else nn.Identity()
                )

            blocks = nn.ModuleList()
            ropes = nn.ModuleList()
            films = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                if not remaining_skips:
                    raise RuntimeError("Internal Diff-VS skip-channel bookkeeping error")
                skip_ch = remaining_skips.pop()
                blocks.append(
                    DiffVSResidualBlock(
                        current + skip_ch,
                        out_ch,
                        self.embedding_dim,
                        dropout=dropout,
                        groups=norm_groups,
                    )
                )
                current = out_ch
                ropes.append(
                    DualPathRoFormer2d(current, **rope_kwargs)
                    if level in decoder_roformer_levels
                    else nn.Identity()
                )
                films.append(
                    TemporalFiLM2d(self.cond_dim, current)
                    if self.use_temporal_film
                    else nn.Identity()
                )
            self.decoder_blocks.append(blocks)
            self.decoder_ropes.append(ropes)
            self.decoder_films.append(films)
            self.decoder_level_ids.append(level)

        if remaining_skips:
            raise RuntimeError(
                f"Internal Diff-VS skip bookkeeping left {len(remaining_skips)} skips"
            )

        self.out_norm = nn.GroupNorm(
            _group_count(current, norm_groups), current, eps=1e-6
        )
        self.out_conv = nn.Conv2d(
            current, self.out_channels * self.band_splits, 3, padding=1
        )
        nn.init.xavier_uniform_(self.out_conv.weight, gain=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def _compress_complex_pairs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_power_compression:
            return x
        b, channels, freq, time = x.shape
        if channels % 2 != 0:
            raise ValueError("Complex power compression requires real/imag channel pairs")
        pairs = x.reshape(b, channels // 2, 2, freq, time)
        real = pairs[:, :, 0]
        imag = pairs[:, :, 1]
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-12)
        compressed_magnitude = self.compression_beta * magnitude.pow(
            self.compression_alpha
        )
        scale = compressed_magnitude / magnitude.clamp_min(1e-8)
        pairs = torch.stack((real * scale, imag * scale), dim=2)
        return pairs.reshape(b, channels, freq, time)

    def _apply_film(
        self,
        module: nn.Module,
        x: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(module, TemporalFiLM2d):
            return module(x, temporal_tokens)
        return x

    def _apply_rope(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return module(x) if isinstance(module, DualPathRoFormer2d) else x

    def _align_for_skip(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _make_visual_activity(
        self,
        visual_activity: Optional[torch.Tensor],
        *,
        batch: int,
        freq: int,
        time: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected = self.visual_activity_channels
        if visual_activity is None:
            return torch.zeros(batch, expected, freq, time, device=device, dtype=dtype)
        activity = visual_activity.to(device=device, dtype=dtype)
        if activity.ndim == 4:
            activity = F.interpolate(
                activity, size=(freq, time), mode="bilinear", align_corners=False
            )
        else:
            if activity.ndim == 2:
                activity = activity.unsqueeze(-1)
            if activity.ndim != 3:
                raise ValueError(
                    f"Unsupported visual_activity shape {tuple(activity.shape)}"
                )
            # Prefer conditioner output [B,T,C], but accept [B,C,T].
            if not (activity.shape[1] == expected and activity.shape[-1] != expected):
                activity = activity.transpose(1, 2)
            activity = F.interpolate(
                activity, size=time, mode="linear", align_corners=False
            )
            activity = activity.unsqueeze(2).expand(-1, -1, freq, -1)
        if activity.shape[1] < expected:
            padding = torch.zeros(
                batch,
                expected - activity.shape[1],
                freq,
                time,
                device=device,
                dtype=dtype,
            )
            activity = torch.cat([activity, padding], dim=1)
        return activity[:, :expected]

    def _inject_bottleneck_bias(
        self,
        x: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.bottleneck_bias is None or temporal_tokens is None:
            return x
        tokens = temporal_tokens.to(device=x.device, dtype=x.dtype)
        if tokens.shape[1] != x.shape[-1]:
            tokens = F.interpolate(
                tokens.transpose(1, 2),
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        bias = self.bottleneck_bias(tokens).transpose(1, 2).unsqueeze(2)
        return x + bias

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
            raise ValueError(f"DiffVSUNetFlowHead expected x_t [B,C,F,T], got {tuple(x_t.shape)}")
        if mixture.ndim != 4:
            raise ValueError(
                f"DiffVSUNetFlowHead expected mixture [B,2,F,T], got {tuple(mixture.shape)}"
            )
        if cond.ndim != 2:
            raise ValueError(f"cond must be [B,D], got {tuple(cond.shape)}")
        if mixture.shape[-2:] != x_t.shape[-2:]:
            freq = min(mixture.shape[-2], x_t.shape[-2])
            time = min(mixture.shape[-1], x_t.shape[-1])
            mixture = mixture[..., :freq, :time]
            x_t = x_t[..., :freq, :time]

        audio_input = torch.cat([x_t, mixture], dim=1)
        if audio_input.shape[1] != self.in_channels:
            raise ValueError(
                f"Configured in_channels={self.in_channels}, but x_t+mixture produced "
                f"{audio_input.shape[1]} channels"
            )
        audio_input = self._compress_complex_pairs(audio_input)
        h, original_freq, _ = self.band_split(audio_input)
        if self.visual_activity_input:
            activity = self._make_visual_activity(
                visual_activity,
                batch=h.shape[0],
                freq=h.shape[-2],
                time=h.shape[-1],
                device=h.device,
                dtype=h.dtype,
            )
            h = torch.cat([h, activity], dim=1)

        time_embedding = sinusoidal_embedding(t, self.time_dim)
        emb = self.cond_proj(torch.cat([cond, time_embedding], dim=-1))
        h = self.in_conv(h)
        skips = [h]

        down_index = 0
        for level, blocks in enumerate(self.encoder_blocks):
            if level > 0:
                h = self.encoder_downs[down_index](h, emb)
                h = self._apply_film(
                    self.encoder_down_films[down_index], h, temporal_tokens
                )
                skips.append(h)
                down_index += 1
            for block, rope, film in zip(
                blocks, self.encoder_ropes[level], self.encoder_films[level]
            ):
                h = block(h, emb)
                h = self._apply_rope(rope, h)
                h = self._apply_film(film, h, temporal_tokens)
                skips.append(h)

        h = self.bottleneck0(h, emb)
        h = self._apply_rope(self.bottleneck_rope, h)
        h = self.bottleneck1(h, emb)
        h = self._apply_film(self.bottleneck_film, h, temporal_tokens)
        h = self._inject_bottleneck_bias(h, temporal_tokens)
        if isinstance(self.bottleneck_cross_attn, BottleneckTemporalCrossAttention):
            h = self.bottleneck_cross_attn(h, temporal_tokens)

        up_index = 0
        for decoder_position, blocks in enumerate(self.decoder_blocks):
            if decoder_position > 0:
                h = self.decoder_ups[up_index](h, emb)
                h = self._apply_film(
                    self.decoder_up_films[up_index], h, temporal_tokens
                )
                up_index += 1
            for block, rope, film in zip(
                blocks,
                self.decoder_ropes[decoder_position],
                self.decoder_films[decoder_position],
            ):
                skip = skips.pop()
                h = self._align_for_skip(h, skip)
                h = block(torch.cat([h, skip], dim=1), emb)
                h = self._apply_rope(rope, h)
                h = self._apply_film(film, h, temporal_tokens)

        if skips:
            raise RuntimeError(f"Diff-VS decoder left {len(skips)} unused skips")
        h = self.out_conv(F.silu(self.out_norm(h)))
        return self.band_split.merge(h, self.out_channels, original_freq)
