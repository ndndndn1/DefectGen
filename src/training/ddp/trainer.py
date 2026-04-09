"""DDP-aware training engine.

Supports single GPU and multi-GPU (torchrun) execution with:
- Mixed precision (AMP)
- Gradient accumulation
- EMA shadow weights
- Rank-0 only checkpointing and wandb logging
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from ...models.base import BaseGenerator
from .ema import EMA
from .utils import get_rank, get_world_size, is_main_process


class DDPTrainer:
    """End-to-end training loop for a :class:`BaseGenerator`.

    Args:
        model: generative model (must inherit BaseGenerator).
        dataset: training dataset returning ``{"image": Tensor, "label": Tensor}``.
        cfg: OmegaConf-like training config (see configs/*.yaml).
        device: torch device.
        local_rank: process-local CUDA index for DDP.
    """

    def __init__(
        self,
        model: BaseGenerator,
        dataset: Dataset,
        cfg: Any,
        device: torch.device,
        local_rank: int = 0,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank
        self.world_size = get_world_size()
        self.rank = get_rank()

        model = model.to(device)
        self.raw_model = model
        if self.world_size > 1:
            self.model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        self.use_amp = bool(cfg.get("amp", False)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        self.accum_steps: int = int(cfg.get("grad_accum_steps", 1))

        self.ema: Optional[EMA] = None
        if cfg.get("ema_decay", 0) > 0 and is_main_process():
            self.ema = EMA(self.raw_model, decay=float(cfg.ema_decay))

        # Data
        sampler = (
            DistributedSampler(dataset, shuffle=True, drop_last=True)
            if self.world_size > 1
            else None
        )
        self.sampler = sampler
        self.loader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        self.run_dir = Path(cfg.get("run_dir", "runs/exp"))
        if is_main_process():
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        if is_main_process() and cfg.get("wandb", False):
            try:
                import wandb

                wandb.init(
                    project=cfg.get("wandb_project", "defectgen"),
                    name=cfg.get("run_name", self.run_dir.name),
                    config=dict(cfg) if hasattr(cfg, "items") else None,
                )
                self._wandb = wandb
            except Exception:
                self._wandb = None

        self.global_step = 0
        self.best_loss = float("inf")

    # --------------------------------------------------------------- training
    def fit(self) -> None:
        epochs = int(self.cfg.epochs)
        for epoch in range(epochs):
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(epoch)
            avg = self._run_epoch(epoch)
            if is_main_process():
                self._save_checkpoint("last.pt", epoch, avg)
                if avg < self.best_loss:
                    self.best_loss = avg
                    self._save_checkpoint("best.pt", epoch, avg)
        # All ranks must reach this point before tearing down the process group.
        if self.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()  # ensure rank 0 finished writing checkpoints

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        n = 0
        pbar = tqdm(self.loader, disable=not is_main_process(), desc=f"epoch {epoch}")
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch.get("label")
            if y is not None and self.raw_model.num_classes:
                y = y.to(self.device, non_blocking=True)
            else:
                y = None

            with autocast(enabled=self.use_amp):
                loss = self.model(x, y) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                if self.cfg.get("grad_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.raw_model)
                self.global_step += 1

            loss_val = float(loss.detach()) * self.accum_steps
            total += loss_val
            n += 1
            if is_main_process():
                pbar.set_postfix(loss=f"{loss_val:.4f}")
                if self._wandb is not None and self.global_step % int(self.cfg.get("log_every", 50)) == 0:
                    self._wandb.log({"train/loss": loss_val, "step": self.global_step})

        return total / max(n, 1)

    # ---------------------------------------------------------- checkpointing
    def _save_checkpoint(self, name: str, epoch: int, loss: float) -> None:
        path = self.run_dir / name
        state = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "global_step": self.global_step,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, path)
