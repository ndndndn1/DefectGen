"""Cutout augmentation (DeVries & Taylor, 2017)."""
from __future__ import annotations

import random

import torch
from torch import Tensor


class Cutout:
    def __init__(self, size_ratio: tuple[float, float] = (0.1, 0.3), p: float = 0.5, fill: float = 0.0) -> None:
        self.size_ratio = size_ratio
        self.p = p
        self.fill = fill

    def __call__(self, x: Tensor) -> Tensor:
        # x: (C, H, W)
        if random.random() > self.p:
            return x
        _, H, W = x.shape
        ph = int(random.uniform(*self.size_ratio) * H)
        pw = int(random.uniform(*self.size_ratio) * W)
        y = random.randint(0, H - ph)
        xv = random.randint(0, W - pw)
        out = x.clone()
        out[:, y:y + ph, xv:xv + pw] = self.fill
        return out
