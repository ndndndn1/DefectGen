"""DefectCoverage: fraction of real defect classes "covered" by generated set.

A real sample is considered covered if at least one generated sample falls
within ``radius`` (L2 distance, normalized) in feature space.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import BaseMetric


class DefectCoverage(BaseMetric):
    name = "defect_coverage"

    def __init__(self, radius: float = 0.5) -> None:
        self.radius = radius
        self.real_chunks: list[Tensor] = []
        self.fake_chunks: list[Tensor] = []

    def update(self, real: Tensor, fake: Tensor) -> None:
        # Use mean pooled pixels as a cheap "feature" — swap for inception in practice.
        self.real_chunks.append(real.flatten(1))
        self.fake_chunks.append(fake.flatten(1))

    def compute(self) -> float:
        real = torch.cat(self.real_chunks, 0)
        fake = torch.cat(self.fake_chunks, 0)
        # Normalize for scale-invariance
        real = real / (real.norm(dim=1, keepdim=True) + 1e-8)
        fake = fake / (fake.norm(dim=1, keepdim=True) + 1e-8)
        d = torch.cdist(real, fake)  # (Nr, Nf)
        covered = (d.min(dim=1).values < self.radius).float().mean()
        return float(covered)

    def reset(self) -> None:
        self.real_chunks.clear()
        self.fake_chunks.clear()
