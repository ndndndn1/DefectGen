"""Batched generation pipeline + ONNX export of the underlying denoiser."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ..models.base import BaseGenerator


class GenerationPipeline:
    """High-level batched sampler.

    Args:
        model: a trained :class:`BaseGenerator`.
        device: target device for sampling.
    """

    def __init__(self, model: BaseGenerator, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        batch_size: int = 16,
        defect_type: Optional[int] = None,
        num_steps: int = 50,
    ) -> Tensor:
        out: list[Tensor] = []
        remaining = num_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            y = (
                torch.full((b,), defect_type, device=self.device, dtype=torch.long)
                if defect_type is not None and self.model.num_classes
                else None
            )
            samples = self.model.sample(b, self.device, y=y, num_steps=num_steps)
            out.append(samples.cpu())
            remaining -= b
        return torch.cat(out, dim=0)


def export_onnx(model: BaseGenerator, path: str, opset: int = 17) -> None:
    """Export the underlying denoising network ``model.net`` to ONNX.

    Note: only the per-step network is exported; the sampling loop must be
    re-implemented in the deployment runtime.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    B, C, H = 1, model.in_channels, model.image_size
    dummy_x = torch.randn(B, C, H, H)
    dummy_t = torch.zeros(B)
    inputs = (dummy_x, dummy_t)
    input_names = ["x", "t"]
    dynamic_axes = {"x": {0: "batch"}, "t": {0: "batch"}, "out": {0: "batch"}}
    torch.onnx.export(
        model.net,
        inputs,
        path,
        input_names=input_names,
        output_names=["out"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
