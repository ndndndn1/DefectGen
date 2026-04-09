"""Forward/backward shape tests for both generators."""
from __future__ import annotations

import torch

from src.models import DDPMGenerator, FlowMatchingGenerator


def _shape_check(model, b: int = 2) -> None:
    x = torch.randn(b, model.in_channels, model.image_size, model.image_size)
    y = torch.zeros(b, dtype=torch.long) if model.num_classes else None
    loss = model.compute_loss(x, y)
    assert loss.ndim == 0, "loss must be scalar"
    loss.backward()
    samples = model.sample(num_samples=b, device=torch.device("cpu"), y=y, num_steps=2)
    assert samples.shape == (b, model.in_channels, model.image_size, model.image_size)


def test_ddpm_shapes():
    model = DDPMGenerator(image_size=16, in_channels=1, num_steps=10, base_channels=16, num_classes=3)
    _shape_check(model)


def test_flow_matching_shapes():
    model = FlowMatchingGenerator(image_size=16, in_channels=3, base_channels=16, num_classes=4)
    _shape_check(model)
