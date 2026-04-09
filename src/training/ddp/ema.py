"""Exponential moving average of model parameters."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    """Maintain a shadow copy of model weights with EMA decay.

    Args:
        model: source model.
        decay: EMA decay (typically 0.999~0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)
