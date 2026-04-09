"""DDPM generator (Ho et al., NeurIPS 2020).

Implements ε-prediction training and ancestral sampling. Equations referenced
in comments use the numbering of the original DDPM paper.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from ..base import BaseGenerator
from ..unet import UNet


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    """Linear β schedule, Eq. (1) discussion in Ho et al., DDPM (NeurIPS 2020)."""
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


class DDPMGenerator(BaseGenerator):
    """Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Args:
        image_size: spatial resolution of training images.
        in_channels: image channels.
        num_steps: number of diffusion timesteps T.
        base_channels / channel_mults: U-Net width / depth.
        num_classes: optional class-conditional labels.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        num_steps: int = 1000,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__(image_size=image_size, in_channels=in_channels, num_classes=num_classes)
        self.num_steps = num_steps
        self.net = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=tuple(channel_mults),
            num_classes=num_classes,
        )

        # Forward process schedule
        betas = linear_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    # ------------------------------------------------------------------ train
    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion q(x_t | x_0).

        Eq. (4) in Ho et al., DDPM (NeurIPS 2020):
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        sab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sob = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return sab * x_0 + sob * noise

    def compute_loss(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # x: (B, C, H, W), y: (B,) or None
        B = x.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)
        # Eq. (14) in Ho et al., DDPM (NeurIPS 2020): simple ε-prediction loss
        eps_pred = self.net(x_t, t.float(), y)
        return F.mse_loss(eps_pred, noise)

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
        """Ancestral sampling using a uniform sub-schedule of length ``num_steps``.

        Eq. (11) in Ho et al., DDPM (NeurIPS 2020):
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
        """
        shape = (num_samples, self.in_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_steps - 1, 0, num_steps, device=device).long()
        traj = [x] if return_trajectory else None

        for i, t in enumerate(timesteps):
            t_batch = t.expand(num_samples)
            eps = self.net(x, t_batch.float(), y)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            mean = (1.0 / alpha_t.sqrt()) * (x - beta_t / (1.0 - alpha_bar_t).sqrt() * eps)
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
            if traj is not None:
                traj.append(x)
        return torch.stack(traj) if return_trajectory else x
