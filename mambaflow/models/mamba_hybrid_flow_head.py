from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .flow_head import sinusoidal_embedding
from .diffvs_unet_flow_head import _head_count

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:  # CPU-only / macOS environments
    selective_scan_fn = None


# MambaVoice splits its 256 coarse bins at these edges.  The flow head works on
# the full 513-bin STFT, so the same bands are used at twice the resolution.
MAMBAVOICE_BAND_EDGES_513 = (0, 8, 24, 56, 104, 168, 256, 384, 513)


def _resample_time(tokens: torch.Tensor, frames: int) -> torch.Tensor:
    """[B,T',C] -> [B,frames,C], linear in time (as TemporalFiLM2d does)."""
    if tokens.shape[1] == frames:
        return tokens
    return F.interpolate(
        tokens.transpose(1, 2), size=frames, mode="linear", align_corners=False
    ).transpose(1, 2)


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Sequential selective scan with the semantics of ``mamba_ssm``.

    u, delta: [B,D,L]; A: [D,N]; B, C: [B,N,L]; D: [D].  ``delta`` is passed
    through softplus, matching ``selective_scan_fn(..., delta_softplus=True)``.
    Slow but exact; used for CPU tests and when the CUDA kernel is absent.
    MambaVoice's own fallback (``x * sigmoid(dt)``) has no recurrence at all,
    so it is not used here.
    """
    dtype = u.dtype
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    delta = F.softplus(delta)
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for i in range(u.shape[-1]):
        state = delta_a[:, :, i] * state + delta_b_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", state, C[:, :, i]))
    y = torch.stack(ys, dim=-1) + u * D.float()[None, :, None]
    return y.to(dtype)


class MambaVisionMixer(nn.Module):
    """MambaVoice's mixer: a selective-scan half and a gated-conv half.

    Parameterization is identical to
    ``mamba/core/models/transformers/mambavoice_multiplicative.py`` so weights
    and behaviour carry over; only the no-kernel fallback differs.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        kernel_size: int = 3,
        scan_backend: str = "auto",
    ) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("MambaVisionMixer needs an even dim")
        self.dim = int(dim)
        self.d_state = int(d_state)
        self.dt_rank = math.ceil(self.dim / 16)
        self.scan_backend = str(scan_backend).lower()
        if self.scan_backend not in {"auto", "cuda", "reference"}:
            raise ValueError(f"Unknown scan_backend={scan_backend!r}")
        if self.scan_backend == "cuda" and selective_scan_fn is None:
            raise ImportError(
                "head.scan_backend=cuda requires mamba_ssm "
                "(pip install mamba-ssm causal-conv1d)"
            )
        half = self.dim // 2
        self.in_proj = nn.Linear(self.dim, self.dim)
        self.conv1d_ssm = nn.Conv1d(half, half, kernel_size, padding="same", groups=half)
        self.conv1d_sym = nn.Conv1d(half, half, kernel_size, padding="same", groups=half)
        self.x_proj = nn.Linear(half, self.dt_rank + self.d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, half)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(half, 1)
        self.A_log = nn.Parameter(torch.log(A))
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        nn.init.constant_(self.dt_proj.bias, -1.0)
        self.D = nn.Parameter(torch.ones(half))
        self.out_proj = nn.Linear(self.dim, self.dim)

    def _scan(self, u, delta, A, B, C):
        use_kernel = selective_scan_fn is not None and u.is_cuda
        if self.scan_backend == "reference":
            use_kernel = False
        if self.scan_backend == "cuda" and not use_kernel:
            raise RuntimeError("scan_backend=cuda but input is not on a CUDA device")
        if use_kernel:
            return selective_scan_fn(
                u.contiguous(),
                delta.contiguous(),
                A,
                B.contiguous(),
                C.contiguous(),
                self.D.float(),
                z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
        if u.is_cuda and not getattr(MambaVisionMixer, "_warned", False):
            warnings.warn(
                "mamba_ssm is not installed: using the sequential reference "
                "selective scan on GPU, which is exact but slow. Install "
                "mamba-ssm or set head.scan_backend=cuda to fail instead."
            )
            MambaVisionMixer._warned = True
        return selective_scan_reference(u, delta, A, B, C, self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in, z_in = self.in_proj(x).chunk(2, dim=-1)
        x_sym = F.silu(self.conv1d_sym(z_in.transpose(1, 2))).transpose(1, 2)
        x_conv = F.silu(self.conv1d_ssm(x_in.transpose(1, 2))).transpose(1, 2)
        dt, B_s, C_s = torch.split(
            self.x_proj(x_conv), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)
        x_ssm = self._scan(
            x_conv.transpose(1, 2),
            dt.transpose(1, 2),
            -torch.exp(self.A_log.float()),
            B_s.transpose(1, 2),
            C_s.transpose(1, 2),
        ).transpose(1, 2)
        return self.out_proj(torch.cat([x_ssm.to(x_sym.dtype), x_sym], dim=-1))


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class _AdaLNBlock(nn.Module):
    """Pre-norm residual block whose two norms are modulated per frame.

    The modulation comes from the flow time, the global AV condition and the
    fused AV tokens (see MambaHybridFlowHead).  It is zero-initialized, so at
    initialization each block is exactly the unconditioned MambaVoice block.
    """

    def __init__(self, dim: int, emb_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.ada = nn.Linear(emb_dim, 4 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def mix(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        s1, b1, s2, b2 = self.ada(emb).to(x.dtype).chunk(4, dim=-1)
        x = x + self.mix(_modulate(self.norm1(x), s1, b1))
        return x + self.mlp(_modulate(self.norm2(x), s2, b2))


class MambaBlock(_AdaLNBlock):
    def __init__(self, dim, emb_dim, *, mlp_ratio, dropout, d_state, kernel_size, scan_backend):
        super().__init__(dim, emb_dim, mlp_ratio, dropout)
        self.mixer = MambaVisionMixer(
            dim, d_state=d_state, kernel_size=kernel_size, scan_backend=scan_backend
        )

    def mix(self, h):
        return self.mixer(h)


class TransformerBlock(_AdaLNBlock):
    def __init__(self, dim, emb_dim, *, mlp_ratio, dropout, num_heads):
        super().__init__(dim, emb_dim, mlp_ratio, dropout)
        self.attn = nn.MultiheadAttention(
            dim, _head_count(dim, num_heads), dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def mix(self, h):
        return self.drop(self.attn(h, h, h, need_weights=False)[0])


class VisualCrossAttention(nn.Module):
    """Frame tokens attend to visual tokens; output scaled by reliability r.

    Sequence version of BottleneckTemporalCrossAttention, pre-norm so that the
    zero-initialized output projection makes it an identity at initialization.
    """

    def __init__(self, dim: int, cond_dim: int, heads: int, reliability_floor: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kv_proj = nn.Linear(cond_dim, dim)
        self.attn = nn.MultiheadAttention(dim, _head_count(dim, heads), batch_first=True)
        self.reliability_floor = float(reliability_floor)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor],
        reliability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if tokens is None:
            return x
        frames = x.shape[1]
        kv = self.kv_proj(_resample_time(tokens.to(x.dtype), frames))
        out = self.attn(self.norm(x), kv, kv, need_weights=False)[0]
        if reliability is not None:
            rel = reliability.to(device=x.device, dtype=x.dtype)
            if rel.ndim == 2:
                rel = rel.unsqueeze(-1)
            if rel.shape[-1] != 1:
                rel = rel.mean(dim=-1, keepdim=True)
            rel = _resample_time(rel, frames).clamp(0.0, 1.0)
            if self.reliability_floor > 0.0:
                rel = self.reliability_floor + (1.0 - self.reliability_floor) * rel
            out = out * rel
        return x + out


class MambaHybridFlowHead(nn.Module):
    """MambaVoice's hybrid Mamba-Transformer as a complex-STFT velocity network.

    Drop-in replacement for TFCTDFUNetFlowHead (same call signature).  The
    layout follows MambaVoice (band-split encoder with band self-attention ->
    one token per STFT frame -> multiplicative visual gate -> Mamba blocks then
    Transformer blocks -> per-frame spectral decoder), with three additions a
    velocity network needs:

      * the encoder sees the full-resolution complex path state and mixture
        ``[z_t; M]`` rather than MambaVoice's averaged, pooled coarse bins;
      * every block is modulated per frame by the flow time, the global AV
        condition and the fused AV tokens (the role FiLM plays in the U-Net);
      * the reliability-gated visual cross-attention of the U-Net bottleneck
        sits between the Mamba and the Transformer stages.

    ``output_mode="mask_residual"`` predicts v = m * z_t + r (complex product),
    the velocity analogue of MambaVoice's complex mask; ``"direct"`` predicts v.
    Both are near zero at initialization, so the one-step estimate starts at M.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        in_channels: int = 4,
        out_channels: int = 2,
        input_freq_bins: int = 513,
        band_edges: Optional[Sequence[int]] = None,
        band_dim: int = 128,
        band_heads: int = 8,
        d_model: int = 512,
        num_mamba_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        d_state: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1,
        embedding_dim: int = 384,
        time_dim: int = 128,
        visual_gate: bool = True,
        visual_cross_attention: bool = True,
        temporal_num_heads: int = 8,
        visual_reliability_floor: float = 0.0,
        output_mode: str = "mask_residual",
        scan_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_freq_bins = int(input_freq_bins)
        self.time_dim = int(time_dim)
        self.d_model = int(d_model)
        self.output_mode = str(output_mode).lower()
        if self.output_mode not in {"mask_residual", "direct"}:
            raise ValueError(f"Unknown output_mode={output_mode!r}")
        if self.output_mode == "mask_residual" and self.in_channels != 2 * self.out_channels:
            raise ValueError(
                "output_mode=mask_residual needs in_channels == 2 * out_channels "
                "(state and mixture stacked, mask applied to the state)"
            )

        if band_edges is None:
            if self.input_freq_bins == 513:
                band_edges = MAMBAVOICE_BAND_EDGES_513
            else:
                # Same relative layout for other STFT sizes (used by tests).
                band_edges = sorted(
                    {round(e * self.input_freq_bins / 513) for e in MAMBAVOICE_BAND_EDGES_513}
                )
        edges = [int(e) for e in band_edges]
        if edges[0] != 0 or edges[-1] != self.input_freq_bins or edges != sorted(set(edges)):
            raise ValueError(
                f"band_edges must increase from 0 to input_freq_bins={self.input_freq_bins}"
            )
        self.band_widths = [b - a for a, b in zip(edges[:-1], edges[1:])]
        n_bands = len(self.band_widths)

        # --- band-split encoder (MambaVoice AudioEncoder on RI inputs) -------
        self.band_in = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.in_channels * w),
                nn.Linear(self.in_channels * w, band_dim),
            )
            for w in self.band_widths
        )
        self.band_pos = nn.Parameter(torch.randn(1, n_bands, band_dim) * 0.02)
        self.band_attn = nn.MultiheadAttention(
            band_dim, _head_count(band_dim, band_heads), batch_first=True
        )
        self.band_norm = nn.LayerNorm(band_dim)
        self.aggregate = nn.Sequential(
            nn.Linear(n_bands * band_dim, 2 * self.d_model),
            nn.SiLU(),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # --- conditioning ------------------------------------------------------
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim + self.time_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.token_proj = nn.Linear(self.cond_dim, embedding_dim)
        nn.init.zeros_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        self.emb_act = nn.SiLU()

        # MambaVoice fusion: audio * sigmoid(W v).
        self.visual_gate = (
            nn.Sequential(nn.LayerNorm(self.cond_dim), nn.Linear(self.cond_dim, self.d_model))
            if visual_gate
            else None
        )

        # --- backbone: MambaVoice "mamba_first" stage --------------------------
        self.mamba_blocks = nn.ModuleList(
            MambaBlock(
                self.d_model,
                embedding_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                d_state=d_state,
                kernel_size=kernel_size,
                scan_backend=scan_backend,
            )
            for _ in range(int(num_mamba_layers))
        )
        self.cross_attn = (
            VisualCrossAttention(
                self.d_model, self.cond_dim, temporal_num_heads, visual_reliability_floor
            )
            if visual_cross_attention
            else None
        )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(
                self.d_model,
                embedding_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                num_heads=num_heads,
            )
            for _ in range(int(num_transformer_layers))
        )

        # --- decoder (MambaVoice feats2mask, without the tanh bound) ----------
        n_out = self.out_channels * self.input_freq_bins
        if self.output_mode == "mask_residual":
            n_out *= 2
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, n_out),
        )
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)

    def _encode(self, h: torch.Tensor) -> torch.Tensor:
        # [B,C,F,T] -> per-band [B,T,C*w] -> [B,T,d_model]
        b, c, _, frames = h.shape
        bands = []
        for proj, band in zip(self.band_in, h.split(self.band_widths, dim=2)):
            band = band.permute(0, 3, 1, 2).reshape(b, frames, -1)
            bands.append(proj(band))
        x = torch.stack(bands, dim=2) + self.band_pos[:, None]
        n_bands, band_dim = x.shape[2], x.shape[3]
        x = x.reshape(b * frames, n_bands, band_dim)
        x = self.band_norm(x + self.band_attn(x, x, x, need_weights=False)[0])
        return self.aggregate(x.reshape(b, frames, n_bands * band_dim))

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
        if interval_end is not None:
            raise NotImplementedError("MambaHybridFlowHead has no interval conditioning")
        if x_t.ndim != 4 or mixture.ndim != 4:
            raise ValueError("x_t and mixture must both be [B,C,F,T]")
        if x_t.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("x_t and mixture must have matching STFT dimensions")
        if x_t.shape[-2] != self.input_freq_bins:
            raise ValueError(
                f"Configured input_freq_bins={self.input_freq_bins}, "
                f"but input has {x_t.shape[-2]} bins"
            )
        h = torch.cat([x_t, mixture], dim=1)
        if h.shape[1] != self.in_channels:
            raise ValueError(f"Configured in_channels={self.in_channels}, got {h.shape[1]}")
        b, _, freq, frames = h.shape

        x = self._encode(h)

        # Per-frame conditioning vector: t and global AV condition, plus the
        # time-resolved fused AV tokens (zero-initialized projection).
        emb = self.cond_proj(torch.cat([cond, sinusoidal_embedding(t, self.time_dim)], dim=-1))
        emb = emb[:, None, :].expand(b, frames, emb.shape[-1])
        if temporal_tokens is not None:
            emb = emb + self.token_proj(_resample_time(temporal_tokens.to(emb.dtype), frames))
        emb = self.emb_act(emb)

        visual = cross_attention_tokens if cross_attention_tokens is not None else temporal_tokens
        if self.visual_gate is not None and visual is not None:
            gate = torch.sigmoid(self.visual_gate(_resample_time(visual.to(x.dtype), frames)))
            x = x * gate

        for block in self.mamba_blocks:
            x = block(x, emb)
        if self.cross_attn is not None:
            x = self.cross_attn(x, visual, visual_activity)
        for block in self.transformer_blocks:
            x = block(x, emb)

        out = self.decoder(x)  # [B,T,K*C*F]
        out = out.view(b, frames, -1, self.out_channels, freq).permute(0, 2, 3, 4, 1)
        if self.output_mode == "direct":
            return out[:, 0]
        mask, residual = out[:, 0], out[:, 1]
        # Complex product m * z_t over RI channel pairs.
        mr, mi = mask[:, 0::2], mask[:, 1::2]
        xr, xi = x_t[:, 0::2], x_t[:, 1::2]
        masked = torch.stack([mr * xr - mi * xi, mr * xi + mi * xr], dim=2)
        return masked.flatten(1, 2) + residual
