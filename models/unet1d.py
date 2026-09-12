from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
MAX_GROUPS = 8
def get_norm(norm_type: str, num_channels: int):
    if norm_type == "gn":
        groups = min(MAX_GROUPS, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    elif norm_type == "bn":
        return nn.BatchNorm1d(num_channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

def get_activation(act_type: str):
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported act_type: {act_type}")

class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=2, kernel_size=3, norm_type="gn", act_type="gelu"):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        for i in range(num_convs):
            cur_in = in_ch if i == 0 else out_ch
            layers.append(nn.Conv1d(cur_in, out_ch, kernel_size=kernel_size, padding=padding))
            layers.append(get_norm(norm_type, out_ch))
            layers.append(get_activation(act_type))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PureUNet1D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, stages=4, conv_per_block=2, kernel_size=3, norm_type="gn", act_type="gelu"):
        super().__init__()
        if stages != 4:
            raise ValueError("NUM_STAGES must be 4 for this implementation.")
        if conv_per_block not in [1, 2]:
            raise ValueError("CONV_PER_BLOCK should be 1 or 2.")

        self.inc = DoubleConv1D(in_ch, base_ch, conv_per_block, kernel_size, norm_type, act_type)
        self.down1 = nn.MaxPool1d(2)
        self.enc1 = DoubleConv1D(base_ch, base_ch * 2, conv_per_block, kernel_size, norm_type, act_type)
        self.down2 = nn.MaxPool1d(2)
        self.enc2 = DoubleConv1D(base_ch * 2, base_ch * 4, conv_per_block, kernel_size, norm_type, act_type)
        self.down3 = nn.MaxPool1d(2)
        self.enc3 = DoubleConv1D(base_ch * 4, base_ch * 8, conv_per_block, kernel_size, norm_type, act_type)

        self.down4 = nn.MaxPool1d(2)
        self.bot = DoubleConv1D(base_ch * 8, base_ch * 8, conv_per_block, kernel_size, norm_type, act_type)

        self.up1 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv1D(base_ch * 12, base_ch * 4, conv_per_block, kernel_size, norm_type, act_type)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv1D(base_ch * 6, base_ch * 2, conv_per_block, kernel_size, norm_type, act_type)

        self.up3 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec3 = DoubleConv1D(base_ch * 3, base_ch, conv_per_block, kernel_size, norm_type, act_type)

        self.up4 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec4 = DoubleConv1D(base_ch * 2, base_ch, conv_per_block, kernel_size, norm_type, act_type)

        self.outc = nn.Conv1d(base_ch, in_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.enc1(self.down1(x1))
        x3 = self.enc2(self.down2(x2))
        x4 = self.enc3(self.down3(x3))
        xb = self.bot(self.down4(x4))

        d1 = self.up1(xb)
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.dec4(d4)

        return self.outc(d4)
