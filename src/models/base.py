"""Abstract base class for all generative models in DefectGen."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class BaseGenerator(nn.Module, ABC):
    """Abstract base for generative models (Diffusion, Flow Matching, GAN, ...).

    Subclasses must implement :meth:`compute_loss` (training step) and
    :meth:`sample` (inference). They are expected to register the underlying
    denoising / vector-field network as ``self.net``.
    """

    def __init__(self, image_size: int, in_channels: int, num_classes: Optional[int] = None) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

    @abstractmethod
    def compute_loss(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Return scalar training loss.

        Args:
            x: clean images, shape (B, C, H, W).
            y: optional class / defect-type labels, shape (B,).
        """

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        y: Optional[Tensor] = None,
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Generate samples. Returns (N, C, H, W) tensor in [-1, 1]."""

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:  # noqa: D401
        # x: (B, C, H, W), y: (B,) or None
        return self.compute_loss(x, y)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
