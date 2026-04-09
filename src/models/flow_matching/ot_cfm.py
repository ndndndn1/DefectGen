"""Optimal-Transport Conditional Flow Matching (Tong et al., 2023; Lipman et al., 2023).

We implement OT-CFM with minibatch optimal-transport coupling between source
noise samples and target images, then regress the network onto the conditional
vector field of the straight-line interpolation.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from ..base import BaseGenerator
from ..unet import UNet


def minibatch_ot_coupling(x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
    """Solve a minibatch entropic-free 2-Wasserstein assignment via Hungarian.

    Args:
        x0: source samples, (B, C, H, W).
        x1: target samples, (B, C, H, W).

    Returns:
        (x0, x1_perm) where x1_perm is x1 permuted by the optimal assignment.

    Notes:
        Falls back to the identity coupling if scipy is unavailable. The
        squared-cost matrix is computed in flattened pixel space.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # pragma: no cover
        return x0, x1

    B = x0.shape[0]
    a = x0.reshape(B, -1)
    b = x1.reshape(B, -1)
    # Eq. (analogue) ||x0_i - x1_j||^2 cost matrix; Tong et al. (2023), Sec. 3.2
    cost = torch.cdist(a, b, p=2).pow(2).detach().cpu().numpy()
    row, col = linear_sum_assignment(cost)
    perm = torch.as_tensor(col, device=x1.device, dtype=torch.long)
    return x0[torch.as_tensor(row, device=x0.device, dtype=torch.long)], x1[perm]


class FlowMatchingGenerator(BaseGenerator):
    """OT-Conditional Flow Matching generator.

    Args:
        image_size: spatial resolution.
        in_channels: image channels.
        sigma_min: minimum interpolation noise (Lipman et al., 2023, Eq. 22).
        use_ot_coupling: enable minibatch OT coupling (Tong et al., 2023).
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        sigma_min: float = 1e-4,
        use_ot_coupling: bool = True,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__(image_size=image_size, in_channels=in_channels, num_classes=num_classes)
        self.sigma_min = sigma_min
        self.use_ot_coupling = use_ot_coupling
        self.net = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=tuple(channel_mults),
            num_classes=num_classes,
        )

    # ------------------------------------------------------------------ train
    def compute_loss(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # x: (B, C, H, W) target image, y: (B,) optional label
        B = x.shape[0]
        x0 = torch.randn_like(x)
        x1 = x
        if self.use_ot_coupling:
            x0, x1 = minibatch_ot_coupling(x0, x1)

        t = torch.rand(B, device=x.device)
        t_ = t.view(-1, 1, 1, 1)

        # Eq. (22)/(23) in Lipman et al. (2023): straight-line OT path
        #   x_t = (1 - (1 - sigma_min) t) * x0 + t * x1
        #   u_t = x1 - (1 - sigma_min) * x0
        x_t = (1 - (1 - self.sigma_min) * t_) * x0 + t_ * x1
        target_v = x1 - (1 - self.sigma_min) * x0

        v_pred = self.net(x_t, t, y)
        return F.mse_loss(v_pred, target_v)

    # ------------------------------------------------------------------ sample
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        y: Optional[Tensor] = None,
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Integrate the learned vector field with explicit Euler from t=0→1."""
        shape = (num_samples, self.in_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        traj = [x] if return_trajectory else None
        for k in range(num_steps):
            t = torch.full((num_samples,), k * dt, device=device)
            v = self.net(x, t, y)
            x = x + dt * v
            if traj is not None:
                traj.append(x)
        return torch.stack(traj) if return_trajectory else x
