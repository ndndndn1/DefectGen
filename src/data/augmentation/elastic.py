"""Elastic deformation (Simard et al., 2003)."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class ElasticDeformation:
    """Smooth random displacement field warping.

    Args:
        alpha: displacement magnitude in pixels.
        sigma: gaussian kernel std for smoothing the field.
        p: probability of applying.
    """

    def __init__(self, alpha: float = 8.0, sigma: float = 4.0, p: float = 0.5) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        # x: (C, H, W)
        if torch.rand(1).item() > self.p:
            return x
        C, H, W = x.shape
        # Random displacement in [-1, 1] then smoothed.
        dx = torch.randn(1, 1, H, W) * 2 - 1
        dy = torch.randn(1, 1, H, W) * 2 - 1
        k = max(3, int(self.sigma) * 2 + 1)
        kernel = torch.ones(1, 1, k, k) / (k * k)
        dx = F.conv2d(dx, kernel, padding=k // 2) * self.alpha
        dy = F.conv2d(dy, kernel, padding=k // 2) * self.alpha

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        grid = torch.stack([xs + dx[0, 0] * 2 / W, ys + dy[0, 0] * 2 / H], dim=-1)
        return F.grid_sample(x.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
