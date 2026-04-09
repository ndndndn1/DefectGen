"""Intra-class diversity: mean pairwise L2 distance among generated samples."""
from __future__ import annotations

import torch
from torch import Tensor

from .base import BaseMetric


class IntraClassDiversity(BaseMetric):
    name = "intra_class_diversity"

    def __init__(self) -> None:
        self.fake_chunks: list[Tensor] = []

    def update(self, real: Tensor, fake: Tensor) -> None:  # noqa: ARG002
        self.fake_chunks.append(fake.flatten(1))

    def compute(self) -> float:
        fake = torch.cat(self.fake_chunks, 0)
        if fake.shape[0] < 2:
            return 0.0
        d = torch.cdist(fake, fake)
        n = fake.shape[0]
        # Average over off-diagonal entries
        return float(d.sum() / (n * (n - 1)))

    def reset(self) -> None:
        self.fake_chunks.clear()
