"""Compact time-conditioned U-Net for DDPM / Flow Matching.

Minimal but correct: a 3-level encoder/decoder with sinusoidal timestep
conditioning and optional class embedding. Designed for shape-test friendliness;
swap for ADM/EDM/DiT for production research.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal embedding (Vaswani et al., 2017). t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        # x: (B, C, H, W), temb: (B, D)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """3-level U-Net.

    Args:
        in_channels: input image channels.
        base_channels: stem channel width.
        channel_mults: per-resolution multipliers; len defines depth.
        num_classes: enables class-conditional embedding when set.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        temb_dim = base_channels * 4
        self.temb_dim = temb_dim
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.label_emb = nn.Embedding(num_classes, temb_dim) if num_classes else None

        chs = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for i, ch in enumerate(chs):
            self.enc_blocks.append(ResBlock(prev, ch, temb_dim))
            prev = ch
            if i < len(chs) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        # Mid
        self.mid1 = ResBlock(prev, prev, temb_dim)
        self.mid2 = ResBlock(prev, prev, temb_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, ch in enumerate(reversed(chs)):
            self.dec_blocks.append(ResBlock(prev + ch, ch, temb_dim))
            prev = ch
            if i < len(chs) - 1:
                self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # x: (B, C, H, W), t: (B,), y: (B,) or None
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))
        if y is not None and self.label_emb is not None:
            temb = temb + self.label_emb(y)

        h = self.in_conv(x)
        skips: list[Tensor] = []
        for i, block in enumerate(self.enc_blocks):
            h = block(h, temb)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for i, block in enumerate(self.dec_blocks):
            skip = skips.pop()
            if skip.shape[-1] != h.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, temb)
            if i < len(self.ups):
                h = self.ups[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))
