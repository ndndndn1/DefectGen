"""Smoke test for the DDPTrainer in single-process mode.

A real DDP test would require ``torchrun --nproc_per_node=2 -m pytest ...``;
here we exercise the single-GPU/CPU path that the trainer falls back to.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.models import DDPMGenerator
from src.training.ddp import DDPTrainer


class _ToyDataset(Dataset):
    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {"image": torch.randn(1, 16, 16), "label": torch.tensor(0)}


def test_trainer_single_process(tmp_path):
    cfg = OmegaConf.create(
        {
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "lr": 1e-4,
            "amp": False,
            "grad_accum_steps": 1,
            "grad_clip": 0.0,
            "ema_decay": 0.0,
            "wandb": False,
            "log_every": 1,
            "run_dir": str(tmp_path / "run"),
        }
    )
    model = DDPMGenerator(image_size=16, in_channels=1, num_steps=10, base_channels=16)
    trainer = DDPTrainer(model, _ToyDataset(), cfg, device=torch.device("cpu"))
    trainer.fit()
    assert (tmp_path / "run" / "last.pt").exists()
