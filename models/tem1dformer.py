from __future__ import annotations

import math

import torch

from torch import nn

from torch.nn import functional as F

class ConvSelfAttention1D(nn.Module):
    """Multi-head self-attention with pointwise/depthwise QKV projections."""

    def __init__(self, dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_pointwise = nn.Conv1d(dim, 3 * dim, kernel_size=1, bias=True)
        self.qkv_depthwise = nn.Conv1d(
            3 * dim, 3 * dim, kernel_size=3, padding=1, groups=3 * dim, bias=True
        )
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, tokens, channels]
        batch, tokens, channels = x.shape
        qkv = self.qkv_depthwise(self.qkv_pointwise(x.transpose(1, 2)))
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, tokens)
        q, k, v = qkv.unbind(dim=1)
        q, k, v = (z.transpose(-2, -1) for z in (q, k, v))
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            y = attn.softmax(dim=-1) @ v
        y = y.transpose(-2, -1).reshape(batch, channels, tokens)
        return self.proj(y).transpose(1, 2)

class TransformerEncoder1D(nn.Module):
    def __init__(self, dim: int = 32, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvSelfAttention1D(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class ResidualBlock1D(nn.Module):
    """Two 3x1 convolutions with ReLU and a local skip connection."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        return x + residual

class TEM1Dformer(nn.Module):
    """Length-agnostic TEM1Dformer reproduction.

    ``max_tokens`` is an engineering adaptation for the four supplied datasets.
    The original TEM sequences are much shorter, whereas the supplied signals
    contain up to 16,384 samples. Adaptive pooling only limits the global
    attention sequence. The pooled branch is interpolated back before local
    residual blocks; there is no unpooled full-resolution feature bypass.
    """

    def __init__(
        self,
        channels: int = 32,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        num_residual_blocks: int = 10,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.feature_conv1 = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.feature_conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.transformer = TransformerEncoder1D(channels, num_heads, mlp_ratio)
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock1D(channels) for _ in range(num_residual_blocks))
        )
        self.recovery_conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.recovery_conv2 = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        features = self.act(self.feature_conv1(noisy))
        features = self.act(self.feature_conv2(features))
        original_length = features.shape[-1]
        token_count = min(original_length, self.max_tokens)
        tokens = F.adaptive_avg_pool1d(features, token_count)
        tokens = self.transformer(tokens.transpose(1, 2)).transpose(1, 2)
        if token_count != original_length:
            tokens = F.interpolate(tokens, size=original_length, mode="linear", align_corners=False)
        features = self.residual_blocks(tokens)
        features = self.act(self.recovery_conv1(features))
        return self.recovery_conv2(features)

    def denoise(self, noisy: torch.Tensor) -> torch.Tensor:
        """Clean estimate for n-pred weights only; not for x-pred weights."""
        return noisy - self(noisy)

def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
