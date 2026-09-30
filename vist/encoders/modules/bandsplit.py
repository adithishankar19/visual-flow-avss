import torch
import torch.nn as nn
from torch.nn import Module, ModuleList

class BandSplit(Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim = dim
        self.dim_inputs = dim_inputs

        self.to_features = ModuleList([
            nn.Sequential(
                # Process raw bins first
                nn.Conv1d(1, 8, kernel_size=3, padding=1),
                nn.SiLU(),
                # Use a strided conv or pooling to reduce length instead of Flattening
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten(),
                nn.Linear(8 * 16, dim),
                nn.LayerNorm(dim) # Final normalization before Mamba
            )
            for dim_in in dim_inputs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        bands = x.split(self.dim_inputs, dim=-1)

        outs = []
        for band_x, proj in zip(bands, self.to_features):
            # (B*T, 1, band_width)
            b_t_x = band_x.reshape(-1, 1, band_x.shape[-1])

            # Feature extraction
            feat = proj(b_t_x)

            # Reshape back to (B, T, dim)
            outs.append(feat.view(*orig_shape, self.dim))

        return torch.stack(outs, dim=-2).contiguous()
