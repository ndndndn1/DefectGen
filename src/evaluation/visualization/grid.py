"""Visualization utilities: grid, trajectory, real-vs-fake comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image


def _to_unit(x: Tensor) -> Tensor:
    return ((x.clamp(-1, 1) + 1) / 2)


def save_grid(images: Tensor, path: str, nrow: int = 8) -> None:
    """Save a grid; rows correspond to defect_type when batched as such."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(_to_unit(images), nrow=nrow, padding=2)
    save_image(grid, path)


def save_real_vs_generated(real: Tensor, fake: Tensor, path: str, n: int = 8) -> None:
    """Top row real, bottom row generated, ``n`` columns."""
    real = _to_unit(real[:n])
    fake = _to_unit(fake[:n])
    pair = torch.cat([real, fake], dim=0)
    save_image(make_grid(pair, nrow=n, padding=2), path)


def save_trajectory(traj: Tensor, path: str, sample_idx: int = 0, n: int = 8) -> None:
    """Visualize sampling trajectory of a single sample over evenly spaced steps.

    Args:
        traj: shape (T, B, C, H, W).
        sample_idx: which sample in the batch to visualize.
        n: number of evenly spaced steps to show.
    """
    T = traj.shape[0]
    idxs = torch.linspace(0, T - 1, n).long()
    frames = _to_unit(traj[idxs, sample_idx])
    save_image(make_grid(frames, nrow=n, padding=2), path)
