from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

DB4_DEC_LO = (
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171909309,
    -0.027983769416859854,
    0.6308807679298587,
    0.7148465705529154,
    0.2303778133088965,
)

def _analysis_pair(parent_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor(DB4_DEC_LO, dtype=torch.float32).view(1, 1, -1)
    cmf = torch.tensor([(-1.0) ** i for i in range(len(DB4_DEC_LO))]).view(1, 1, -1)
    high = torch.flip(low, dims=(-1,)) * cmf
    return (low, high) if parent_index % 2 == 0 else (high, low)

def _synthesis_pair(parent_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.tensor(DB4_DEC_LO, dtype=torch.float32).view(1, 1, -1)
    cmf = torch.tensor([(-1.0) ** i for i in range(len(DB4_DEC_LO))]).view(1, 1, -1)
    low = torch.flip(base, dims=(-1,))
    high = base * (-cmf)
    return (low, high) if parent_index % 2 == 0 else (high, low)

class SmoothHardThreshold(nn.Module):
    """Paper Eq. (6), with one non-negative threshold per tree node."""

    def __init__(self, nodes: int, initial_threshold: float = 0.0) -> None:
        super().__init__()
        self.thresholds = nn.Parameter(torch.full((nodes,), initial_threshold))

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        gamma = self.thresholds[index]
        return x * (torch.sigmoid(10.0 * (x - gamma)) + torch.sigmoid(-10.0 * (x + gamma)))

class LearnableWaveletPacketDenoiser(nn.Module):
    """Five-level learnable WPT analysis-threshold-synthesis network."""

    def __init__(self, levels: int = 5) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.levels = levels
        self.kernel_size = len(DB4_DEC_LO)
        self.analysis = nn.ParameterList()
        self.synthesis = nn.ParameterList()
        self.thresholds = nn.ParameterList()
        for level in range(levels):
            analysis_level = []
            synthesis_level = []
            for parent in range(2**level):
                a_low, a_high = _analysis_pair(parent)
                s_low, s_high = _synthesis_pair(parent)
                analysis_level.extend([a_low, a_high])
                synthesis_level.append(torch.cat([s_low, s_high], dim=1))
            self.analysis.append(nn.Parameter(torch.cat(analysis_level, dim=0)))
            self.synthesis.append(nn.Parameter(torch.cat(synthesis_level, dim=0)))
            self.thresholds.append(nn.Parameter(torch.zeros(2 ** (level + 1))))

    @staticmethod
    def _pad_for_analysis(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (3, 4))

    @staticmethod
    def _upsample_reference(x: torch.Tensor, parent_length: int) -> torch.Tensor:
        # Reference implementation inserts even-length coefficients at odd indices.
        up = x.new_zeros(x.shape[0], x.shape[1], parent_length)
        if parent_length % 2 == 0:
            up[..., 1::2] = x
        else:
            up[..., 0::2] = x
        return up

    def forward(self, x: torch.Tensor, threshold_scale: float | torch.Tensor = 1.0) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected [batch, 1, length], got {tuple(x.shape)}")
        nodes = x
        parent_lengths = []
        for level in range(self.levels):
            parent_lengths.append(nodes.shape[-1])
            groups = 2**level
            nodes = F.conv1d(self._pad_for_analysis(nodes), self.analysis[level], stride=2, groups=groups)
            # Match the authors' implementation: thresholds are unconstrained
            # parameters initialized at zero. Applying abs() here would give a
            # zero derivative at initialization and freeze every threshold.
            gamma = self.thresholds[level].view(1, -1, 1) * threshold_scale
            nodes = nodes * (torch.sigmoid(10.0 * (nodes - gamma)) + torch.sigmoid(-10.0 * (nodes + gamma)))

        for level in range(self.levels - 1, -1, -1):
            parent_length = parent_lengths[level]
            pad = (3, 4) if parent_length % 2 == 0 else (4, 3)
            up = nodes.new_zeros(nodes.shape[0], nodes.shape[1], parent_length)
            if parent_length % 2 == 0:
                up[..., 1::2] = nodes
            else:
                up[..., 0::2] = nodes
            nodes = F.conv1d(F.pad(up, pad), self.synthesis[level], groups=2**level)
        return nodes

def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
