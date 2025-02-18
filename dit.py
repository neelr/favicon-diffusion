import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DiTConfig:
    """Configuration for DiT model"""
    input_size: int = 64          # Input image size
    patch_size: int = 4           # Patch size
    in_channels: int = 3          # Number of input channels
    dim: int = 768               # Hidden dimension
    depth: int = 12              # Number of transformer blocks
    dim_head: int = 192          # Dimension per attention head
    mlp_mult: int = 4            # MLP expansion factor
    time_emb_dim: int = 768      # Time embedding dimension


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Position Embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Fixed2DPosEmb(nn.Module):
    """2D Fixed Positional Embeddings using sinusoidal embeddings"""

    def __init__(self, dim, h, w):
        super().__init__()
        self.h = h
        self.w = w

        # Create position indices
        pos_h = torch.arange(h)
        pos_w = torch.arange(w)

        # Create 2D position mesh
        grid_y, grid_x = torch.meshgrid(pos_h, pos_w, indexing='ij')
        grid_y = grid_y.flatten()
        grid_x = grid_x.flatten()

        # Use existing sinusoidal embeddings for both dimensions
        self.pos_emb_h = SinusoidalPosEmb(dim // 2)
        self.pos_emb_w = SinusoidalPosEmb(dim // 2)

        # Pre-compute the position embeddings
        with torch.no_grad():
            emb_h = self.pos_emb_h(grid_y)
            emb_w = self.pos_emb_w(grid_x)
            # Combine both dimensions
            pos_emb = torch.cat([emb_h, emb_w], dim=-1)
            self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        return x + self.pos_emb


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with proper mean/variance normalization"""

    def __init__(self, dim, time_emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale_proj = nn.Linear(time_emb_dim, dim, bias=False)
        self.shift_proj = nn.Linear(time_emb_dim, dim, bias=False)

    def forward(self, x, time_emb):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        norm_x = (x - mean) / std

        scale = 1 + self.scale_proj(time_emb).unsqueeze(1)
        shift = self.shift_proj(time_emb).unsqueeze(1)

        return norm_x * scale + shift


class Attention(nn.Module):
    """Single-head Self-attention"""

    def __init__(self, dim, dim_head):
        super().__init__()
        self.scale = dim_head ** -0.5

        # Project input to lower dimension for q, k, v
        self.to_q = nn.Linear(dim, dim_head, bias=False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)

        # Project back to original dimension
        self.to_out = nn.Linear(dim_head, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape

        # Project to queries, keys and values
        q = self.to_q(x)  # b n dim_head
        k = self.to_k(x)  # b n dim_head
        v = self.to_v(x)  # b n dim_head

        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return self.to_out(out)


class FeedForward(nn.Module):
    """MLP with GeLU activation"""

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult, bias=False),
            nn.SiLU(),
            nn.Linear(dim * mult, dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """DiT Transformer Block"""

    def __init__(self, dim, time_emb_dim, dim_head, mlp_mult):
        super().__init__()
        self.attn = Attention(dim, dim_head)
        self.ff = FeedForward(dim, mlp_mult)
        self.norm1 = AdaLN(dim, time_emb_dim)
        self.norm2 = AdaLN(dim, time_emb_dim)

    def forward(self, x, time_emb):
        x = x + self.attn(self.norm1(x, time_emb))
        x = x + self.ff(self.norm2(x, time_emb))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer Model with 2D positional embeddings
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        # Calculate patch dimensions
        self.num_patches = (config.input_size // config.patch_size) ** 2
        self.patch_dim = config.in_channels * config.patch_size * config.patch_size

        # Grid size for positional embeddings
        self.grid_size = config.input_size // config.patch_size

        # Patch embedding (removed bias)
        self.patch_embed = nn.Linear(self.patch_dim, config.dim, bias=False)

        # Add 2D positional embeddings
        self.pos_embed = Fixed2DPosEmb(
            config.dim,
            h=self.grid_size,
            w=self.grid_size
        )

        # Time embedding MLP (removed biases)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.dim // 2),
            nn.Linear(config.dim // 2, config.time_emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.time_emb_dim, config.time_emb_dim, bias=False)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                time_emb_dim=config.time_emb_dim,
                dim_head=config.dim_head,
                mlp_mult=config.mlp_mult
            ) for _ in range(config.depth)
        ])

        self.final_norm = AdaLN(config.dim, config.time_emb_dim)
        self.proj_fin = nn.Linear(config.dim, self.patch_dim)
        self.to_pixels = nn.Linear(
            config.dim, self.patch_dim, bias=False)  # Removed bias

    def patchify(self, x):
        """Convert image to patches"""
        p = self.config.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//p, p, w//p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h//p)*(w//p), c*p*p)
        return x

    def unpatchify(self, x):
        """Convert patches back to image"""
        p = self.config.patch_size
        h = w = self.config.input_size
        c = self.config.in_channels

        x = x.reshape(x.shape[0], h//p, w//p, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], c, h, w)
        return x

    def forward(self, x, time):
        """
        Forward pass
        x: [B, C, H, W] - Input image
        time: [B] - Time values
        """
        # Patchify and embed
        x = self.patchify(x)
        x = self.patch_embed(x)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Time embedding
        time_emb = self.time_mlp(time)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, time_emb)

        # Final norm and projection
        latent = self.final_norm(x, time_emb)
        x = self.to_pixels(latent)
        fin = self.proj_fin(latent)
        # Unpatchify to image
        x = self.unpatchify(x)
        return x, self.unpatchify(fin)

    def export_matrices(self) -> Dict[str, torch.Tensor]:
        """Export all learnable matrices for comparison with WebGPU implementation"""
        matrices = {}

        # Patch embedding matrices
        matrices['patch_embed.weight'] = self.patch_embed.weight.data

        # Time MLP matrices
        matrices['time_mlp.1.weight'] = self.time_mlp[1].weight.data
        matrices['time_mlp.3.weight'] = self.time_mlp[3].weight.data

        # Transformer block matrices
        for i, block in enumerate(self.blocks):
            prefix = f'block_{i}.'

            # Attention matrices
            matrices[prefix + 'attn.to_q.weight'] = block.attn.to_q.weight.data
            matrices[prefix + 'attn.to_k.weight'] = block.attn.to_k.weight.data
            matrices[prefix + 'attn.to_v.weight'] = block.attn.to_v.weight.data
            matrices[prefix + 'attn.to_out.weight'] = block.attn.to_out.weight.data

            # Feed-forward matrices
            matrices[prefix + 'ff.net.0.weight'] = block.ff.net[0].weight.data
            matrices[prefix + 'ff.net.2.weight'] = block.ff.net[2].weight.data

            # AdaLN matrices
            matrices[prefix +
                     'norm1.scale.weight'] = block.norm1.scale_proj.weight.data
            matrices[prefix +
                     'norm1.shift.weight'] = block.norm1.shift_proj.weight.data
            matrices[prefix +
                     'norm2.scale.weight'] = block.norm2.scale_proj.weight.data
            matrices[prefix +
                     'norm2.shift.weight'] = block.norm2.shift_proj.weight.data

        # Final layers matrices
        matrices['final_norm.scale.weight'] = self.final_norm.scale_proj.weight.data
        matrices['final_norm.shift.weight'] = self.final_norm.shift_proj.weight.data
        matrices['to_pixels.weight'] = self.to_pixels.weight.data

        return matrices


def load_matrices(model: DiT, matrices: Dict[str, torch.Tensor]):
    """Load matrices into the model"""
    state_dict = model.state_dict()
    for name, matrix in matrices.items():
        if name in state_dict:
            state_dict[name].copy_(matrix)
    model.load_state_dict(state_dict)


# Example usage:
if __name__ == "__main__":
    # Create model with default config
    config = DiTConfig()
    model = DiT(config)

    # Example input
    batch_size = 1
    x = torch.randn(batch_size, config.in_channels,
                    config.input_size, config.input_size)
    time = torch.zeros(batch_size)

    # Forward pass
    output = model(x, time)
    print(f"Output shape: {output.shape}")

    # Export matrices
    matrices = model.export_matrices()
    print("\nExported matrices:")
    for name, matrix in matrices.items():
        print(f"{name}: {matrix.shape}")
