from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .flow_head import sinusoidal_embedding
from .diffvs_unet_flow_head import (
    BottleneckTemporalCrossAttention,
    TemporalFiLM2d,
    _group_count,
)


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


class FlowMapCorrectionAdapter2d(nn.Module):
    """Small zero-initialized correction network for drift-to-flow-map conversion.

    The pre-trained U-Net remains an immutable feature extractor.  This adapter
    receives its final feature map and base drift prediction, while time ``t``
    and interval length ``r-t`` are embedded *separately*.  Only the last
    convolution is zero initialized, so the complete model is exactly the
    pre-trained drift estimator before the first optimizer update while all
    preceding adapter layers can receive a useful gradient immediately.
    """

    def __init__(
        self,
        feature_channels: int,
        out_channels: int,
        time_dim: int,
        *,
        hidden_channels: Optional[int] = None,
        blocks: int = 3,
        norm_groups: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("flowmap_adapter_blocks must be >= 1")
        hidden = int(hidden_channels or feature_channels)
        self.time_dim = int(time_dim)
        self.input_proj = nn.Conv2d(
            int(feature_channels) + int(out_channels), hidden, 1
        )
        # Concatenation, rather than addition, preserves the distinction between
        # absolute position and requested step length.
        self.interval_proj = nn.Sequential(
            nn.Linear(2 * self.time_dim, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden),
        )
        self.blocks = nn.ModuleList()
        for _ in range(int(blocks)):
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(_group_count(hidden, norm_groups), hidden, eps=1e-6),
                    nn.SiLU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.GroupNorm(_group_count(hidden, norm_groups), hidden, eps=1e-6),
                    nn.SiLU(),
                    nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                )
            )
        self.output_norm = nn.GroupNorm(
            _group_count(hidden, norm_groups), hidden, eps=1e-6
        )
        self.output = nn.Conv2d(hidden, int(out_channels), 3, padding=1)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for module in self.interval_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            convs = [module for module in block if isinstance(module, nn.Conv2d)]
            nn.init.xavier_uniform_(convs[0].weight)
            nn.init.zeros_(convs[0].bias)
            nn.init.xavier_uniform_(convs[1].weight, gain=1e-2)
            nn.init.zeros_(convs[1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        base: torch.Tensor,
        t: torch.Tensor,
        interval_end: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if interval_end is None:
            interval_end = torch.ones_like(t)
        delta = (interval_end - t).clamp_min(0.0)
        interval_condition = torch.cat(
            [
                sinusoidal_embedding(t, self.time_dim),
                sinusoidal_embedding(delta, self.time_dim),
            ],
            dim=-1,
        )
        scale, shift = self.interval_proj(interval_condition).chunk(2, dim=-1)
        h = self.input_proj(torch.cat([features, base.detach()], dim=1))
        h = h * (1.0 + scale[:, :, None, None].to(h.dtype))
        h = h + shift[:, :, None, None].to(h.dtype)
        for block in self.blocks:
            h = h + block(h)
        return self.output(F.silu(self.output_norm(h)))


class TFCTDFUNetFlowHead(nn.Module):
    """Compact four-level TFC-TDF U-Net for complex-STFT drift prediction.

    The head keeps the same call signature as the existing spectrogram heads:
    it receives the current target state, mixture, global condition, flow time,
    and optional temporal AV tokens, then predicts a complex-STFT velocity.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        in_channels: int = 4,
        out_channels: int = 2,
        model_channels: int = 48,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        blocks_per_level: int = 2,
        embedding_dim: int = 384,
        time_dim: int = 128,
        input_freq_bins: int = 513,
        norm_groups: int = 16,
        tdf_bottleneck_factor: int = 16,
        dropout: float = 0.0,
        temporal_conditioning: str = "multiscale_film_cross_attn",
        temporal_num_heads: int = 4,
        visual_reliability_floor: float = 0.0,
        residual_scale: str | float = "half",
        exact_resample: bool = False,
        interval_embedding_reference: Optional[float] = None,
        dual_head: bool = False,
        flowmap_adapter_blocks: int = 0,
        flowmap_adapter_channels: Optional[int] = None,
        flowmap_adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channel_mult) != 4:
            raise ValueError("TFCTDFUNetFlowHead requires exactly four levels")
        if blocks_per_level < 1:
            raise ValueError("blocks_per_level must be >= 1")

        self.cond_dim = int(cond_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_dim = int(time_dim)
        self.embedding_dim = int(embedding_dim)
        self.input_freq_bins = int(input_freq_bins)

        # Reference interval length for warm-starting a mean-velocity model from
        # a direct (drift) checkpoint.  None, the default, keeps the original
        # conditioning emb(t) + emb(r - t).  A value d uses
        # emb(t) + [emb(r - t) - emb(d)], which is exactly emb(t) when r - t == d.
        # With d = 1 the one-step query (0, 1) feeds the head the embedding a
        # drift model was trained on, emb(0), so drift weights reproduce the drift
        # prediction before any update.  No parameters are added and no (t, r)
        # information is lost: the offset is a constant vector.
        self.interval_embedding_reference = (
            None if interval_embedding_reference is None else float(interval_embedding_reference)
        )

        # Residual weighting inside every TFC-TDF block.
        #   "half" (default, legacy): (x + h) * 2^-0.5, applied twice per block,
        #       so each block multiplies its identity path by exactly 0.5.
        #   "unit": plain (x + h).  conv1/fc2 are already initialised near zero
        #       (gain 1e-2), which is what actually keeps activations bounded at
        #       init, so the extra 2^-0.5 only suppresses the skip pathway.
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
        # the matching transposed convs give 64->128->256->512, so the finest
        # decoder level used to bilinearly resample 512->513.  That resamples the
        # highest-resolution complex-STFT features on every forward pass.
        # `exact_resample` instead pads the input to a multiple of 8 in both axes
        # and crops the output back, so every skip connection lines up exactly.
        self.exact_resample = bool(exact_resample)
        if self.exact_resample:
            padded_bins = ((self.input_freq_bins + 7) // 8) * 8
        else:
            padded_bins = self.input_freq_bins
        self._padded_freq_bins = padded_bins

        temporal_mode = str(temporal_conditioning).lower()
        if temporal_mode in {"none", "off", "false"}:
            temporal_mode = "none"
        elif temporal_mode in {"multiscale_film", "film", "temporal_film"}:
            temporal_mode = "multiscale_film"
        elif temporal_mode in {
            "multiscale_film_cross_attn",
            "film_cross_attn",
            "cross_attn_film",
        }:
            temporal_mode = "multiscale_film_cross_attn"
        else:
            raise ValueError(f"Unknown temporal_conditioning={temporal_mode!r}")
        self.use_temporal_film = temporal_mode != "none"
        self.use_cross_attn = temporal_mode == "multiscale_film_cross_attn"

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
                films.append(
                    TemporalFiLM2d(self.cond_dim, out_ch)
                    if self.use_temporal_film
                    else nn.Identity()
                )
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
        self.bottleneck_film = (
            TemporalFiLM2d(self.cond_dim, current)
            if self.use_temporal_film
            else nn.Identity()
        )
        self.bottleneck_cross_attn = (
            BottleneckTemporalCrossAttention(
                current,
                self.cond_dim,
                temporal_num_heads,
                reliability_floor=visual_reliability_floor,
            )
            if self.use_cross_attn
            else nn.Identity()
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
                films.append(
                    TemporalFiLM2d(self.cond_dim, out_ch)
                    if self.use_temporal_film
                    else nn.Identity()
                )
                current = out_ch
            self.decoder_blocks.append(blocks)
            self.decoder_films.append(films)

        self.out_norm = nn.GroupNorm(
            _group_count(current, norm_groups), current, eps=1e-6
        )
        self.out_conv = nn.Conv2d(current, self.out_channels, 3, padding=1)
        nn.init.xavier_uniform_(self.out_conv.weight, gain=1e-3)
        nn.init.zeros_(self.out_conv.bias)

        # Optional: zero-initialized correction head for dual-head AlphaFlow refinement.
        # Only created if dual_head=true in config. If used, forward_dual_head()
        # returns (base=out_conv, delta, total). Kept separate from out_conv so
        # checkpoint loading works transparently: loading an old checkpoint populates
        # out_conv, delta_conv stays zero-initialized or None.
        self.dual_head = bool(dual_head)
        if self.dual_head and int(flowmap_adapter_blocks) > 0:
            raise ValueError(
                "Use either head.dual_head or head.flowmap_adapter_blocks, not both"
            )
        if self.dual_head:
            self.delta_conv = nn.Conv2d(current, self.out_channels, 3, padding=1)
            nn.init.zeros_(self.delta_conv.weight)
            nn.init.zeros_(self.delta_conv.bias)
        else:
            self.delta_conv = None
        if int(flowmap_adapter_blocks) > 0:
            self.flowmap_adapter = FlowMapCorrectionAdapter2d(
                current,
                self.out_channels,
                self.time_dim,
                hidden_channels=flowmap_adapter_channels,
                blocks=int(flowmap_adapter_blocks),
                norm_groups=norm_groups,
                dropout=float(flowmap_adapter_dropout),
            )
        else:
            self.flowmap_adapter = None

    @property
    def has_flowmap_adapter(self) -> bool:
        return self.flowmap_adapter is not None

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

    @staticmethod
    def _apply_film(
        module: nn.Module,
        x: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(module, TemporalFiLM2d):
            return module(x, temporal_tokens)
        return x

    def forward_dual_head(
        self,
        x_t: torch.Tensor,
        mixture: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor] = None,
        visual_activity: Optional[torch.Tensor] = None,
        cross_attention_tokens: Optional[torch.Tensor] = None,
        interval_end: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning base, delta, and combined outputs separately.

        Used during training to supervise the base and delta heads independently.
        Returns (u_base, delta_u, u_total) where u_total = u_base + delta_u.
        """
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
                h = F.pad(h, (0, pad_t, 0, pad_f), mode="replicate")

        time_embedding = sinusoidal_embedding(t, self.time_dim)
        if interval_end is not None:
            delta = (interval_end - t).clamp_min(0.0)
            time_embedding = time_embedding + self._interval_embedding(delta)
        emb = self.cond_proj(torch.cat([cond, time_embedding], dim=-1))
        h = self.in_conv(h)
        skips = []

        for level, blocks in enumerate(self.encoder_blocks):
            for block, film in zip(blocks, self.encoder_films[level]):
                h = block(h, emb)
                h = self._apply_film(film, h, temporal_tokens)
            skips.append(h)
            if level < 3:
                h = self.downs[level](h)

        h = self.bottleneck0(h, emb)
        h = self.bottleneck1(h, emb)
        h = self._apply_film(self.bottleneck_film, h, temporal_tokens)
        if isinstance(self.bottleneck_cross_attn, BottleneckTemporalCrossAttention):
            attn_tokens = cross_attention_tokens if cross_attention_tokens is not None else temporal_tokens
            h = self.bottleneck_cross_attn(h, attn_tokens, visual_activity)

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
                h = self._apply_film(film, h, temporal_tokens)

        h_norm = F.silu(self.out_norm(h))
        out_base = self.out_conv(h_norm)
        if self.flowmap_adapter is not None:
            out_delta = self.flowmap_adapter(h_norm, out_base, t, interval_end)
        elif self.delta_conv is not None:
            out_delta = self.delta_conv(h_norm)
        else:
            out_delta = torch.zeros_like(out_base)

        if pad_f or pad_t:
            out_base = out_base[..., : out_base.shape[-2] - pad_f, : out_base.shape[-1] - pad_t]
            out_delta = out_delta[..., : out_delta.shape[-2] - pad_f, : out_delta.shape[-1] - pad_t]

        return out_base, out_delta, out_base + out_delta

    def _interval_embedding(self, delta: torch.Tensor) -> torch.Tensor:
        """emb(r - t), offset so that r - t == interval_embedding_reference gives 0.

        The difference is formed before it is added to emb(t), so at the
        reference length it is exactly zero and emb(t) passes through unchanged.
        Adding emb(r - t) and then subtracting emb(reference) would not
        round-trip exactly in floating point.
        """
        emb = sinusoidal_embedding(delta, self.time_dim)
        if self.interval_embedding_reference is None:
            return emb
        reference = torch.full_like(delta, self.interval_embedding_reference)
        return emb - sinusoidal_embedding(reference, self.time_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        mixture: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor] = None,
        visual_activity: Optional[torch.Tensor] = None,
        cross_attention_tokens: Optional[torch.Tensor] = None,
        interval_end: Optional[torch.Tensor] = None,
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

        # Pad to a multiple of 8 so all three down/up stages are exactly
        # invertible and no skip connection needs resampling.
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

        time_embedding = sinusoidal_embedding(t, self.time_dim)
        if interval_end is not None:
            # AlphaFlow/mean-velocity conditioning: the finite-interval model
            # is conditioned on absolute start time t and interval length
            # delta=r-t.  Summing their sinusoidal embeddings follows the
            # c_{t,delta}=emb(t)+emb(delta) parameterization used by
            # AlphaFlowTSE while preserving the existing cond_proj shape.
            delta = (interval_end - t).clamp_min(0.0)
            time_embedding = time_embedding + self._interval_embedding(delta)
        emb = self.cond_proj(torch.cat([cond, time_embedding], dim=-1))
        h = self.in_conv(h)
        skips = []

        for level, blocks in enumerate(self.encoder_blocks):
            for block, film in zip(blocks, self.encoder_films[level]):
                h = block(h, emb)
                h = self._apply_film(film, h, temporal_tokens)
            skips.append(h)
            if level < 3:
                h = self.downs[level](h)

        h = self.bottleneck0(h, emb)
        h = self.bottleneck1(h, emb)
        h = self._apply_film(self.bottleneck_film, h, temporal_tokens)
        if isinstance(self.bottleneck_cross_attn, BottleneckTemporalCrossAttention):
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
                h = self._apply_film(film, h, temporal_tokens)

        # Standard forward: out_conv (base residual) + optional delta_conv correction.
        h_norm = F.silu(self.out_norm(h))
        out = self.out_conv(h_norm)
        if self.flowmap_adapter is not None:
            out = out + self.flowmap_adapter(h_norm, out, t, interval_end)
        elif hasattr(self, 'delta_conv') and self.delta_conv is not None:
            out = out + self.delta_conv(h_norm)
        if pad_f or pad_t:
            out = out[..., : out.shape[-2] - pad_f, : out.shape[-1] - pad_t]
        return out
