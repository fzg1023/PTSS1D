import torch

import torch.nn as nn

import torch.nn.functional as F

import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))

class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        freqs_cos = freqs.cos().repeat(1, 2)
        freqs_sin = freqs.sin().repeat(1, 2)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        
    def forward(self, x):
        seq_len = x.shape[1]
        return x * self.freqs_cos[:seq_len] + self._rotate_half(x) * self.freqs_sin[:seq_len]
        
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, rope_embed=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if rope_embed is not None:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            cos = rope_embed.freqs_cos[:N].unsqueeze(0).unsqueeze(2)
            sin = rope_embed.freqs_sin[:N].unsqueeze(0).unsqueeze(2)
            
            def apply_rotary(x_in, cos_in, sin_in):
                x1, x2 = x_in.chunk(2, dim=-1)
                return (x_in * cos_in) + (torch.cat((-x2, x1), dim=-1) * sin_in)
                
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        # 💡 彻底移除 adaLN，换成最原汁原味的 ViT Block
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qk_norm=True)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * 4))
        
    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x

class UnconditionalJiT1D(nn.Module):
    def __init__(self, signal_length=1024, patch_size=32, in_channels=1, hidden_size=128, depth=4, num_heads=8):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.out_channels = in_channels
        self.num_patches = signal_length // patch_size
        
        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.rope = RotaryEmbedding1D(hidden_size // num_heads, max_seq_len=self.num_patches)
        
        self.blocks = nn.ModuleList([JiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_norm = RMSNorm(hidden_size)
        self.final_linear = nn.Linear(hidden_size, patch_size * in_channels)
        
    def forward(self, x):
        # 💡 参数里彻底没有 t 了
        x = self.x_embedder(x).transpose(1, 2)
        
        for block in self.blocks:
            x = block(x, self.rope)
            
        x = self.final_norm(x)
        x = self.final_linear(x)
        
        B, N, _ = x.shape
        x = x.reshape(B, N, self.out_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.out_channels, self.signal_length)
        
        # Return the trained prediction target (clean signal or noise).
        return x
