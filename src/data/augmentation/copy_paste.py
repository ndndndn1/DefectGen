"""Defect copy-paste augmentation (Ghiasi et al., 2021, adapted)."""
from __future__ import annotations

import random

import torch
from torch import Tensor


class CopyPaste:
    """Copy a random patch from a source image and paste into a destination.

    Args:
        patch_ratio: range of patch size as fraction of image side.
        p: probability of applying.
    """

    def __init__(self, patch_ratio: tuple[float, float] = (0.1, 0.3), p: float = 0.5) -> None:
        self.patch_ratio = patch_ratio
        self.p = p

    def __call__(self, dst: Tensor, src: Tensor) -> Tensor:
        # dst, src: (C, H, W)
        if random.random() > self.p:
            return dst
        _, H, W = dst.shape
        ph = int(random.uniform(*self.patch_ratio) * H)
        pw = int(random.uniform(*self.patch_ratio) * W)
        sy = random.randint(0, H - ph)
        sx = random.randint(0, W - pw)
        dy = random.randint(0, H - ph)
        dx = random.randint(0, W - pw)
        out = dst.clone()
        out[:, dy:dy + ph, dx:dx + pw] = src[:, sy:sy + ph, sx:sx + pw]
        return out
