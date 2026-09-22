import torch
import torch.nn as nn
from .bandsplit import BandSplit


class AudioEncoder(nn.Module):
    def __init__(self, bands, embed_dim, output_dim, num_heads=8):
        super().__init__()
        self.bands = bands
        self.dim_inputs = [b[1] - b[0] for b in bands]
        self.n_bands = len(bands)

        self.bandsplit = BandSplit(dim=embed_dim, dim_inputs=tuple(self.dim_inputs))

        # NEW: Learnable Position Embeddings for the bands
        self.pos_emb = nn.Parameter(torch.randn(1, 1, self.n_bands, embed_dim) * 0.02)

        # NEW: Global Band Interaction (Self-Attention)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Better Aggregator
        self.aggregate = nn.Sequential(
            nn.Linear(self.n_bands * embed_dim, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, mag):
        # 1. Pre-processing
        if mag.dim() == 4: # Handle (B, C, F, T)
            mag = mag.mean(dim=1)

        # log1p is critical for SDR stability; uncomment it if your data is raw magnitude
        # mag = torch.log1p(mag)

        # 2. BandSplit -> (B, T, N, D)
        # Note: We transpose to (B, T, F) for BandSplit
        x = self.bandsplit(mag.transpose(1, 2))

        # 3. Add Positional Info so model knows which band is which
        x = x + self.pos_emb

        # 4. Global Context (Bands talk to each other)
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.attn_norm(x_flat + attn_out).view(B, T, N, D)

        # 5. Collapse & Project
        x = x.reshape(B, T, N * D)
        return self.aggregate(x)
