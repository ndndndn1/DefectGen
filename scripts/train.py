"""Training entry point.

Usage:
    python scripts/train.py --config configs/ddpm_wafer.yaml
    torchrun --nproc_per_node=4 scripts/train.py --config configs/flow_matching_pcb.yaml --ddp
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data import build_dataset
from src.models import build_model
from src.training.ddp import (
    DDPTrainer,
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from src.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ddp", action="store_true", help="(informational) launched via torchrun")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.model.name}_{cfg.dataset.name}_{ts}"
        cfg.training.run_name = run_name
        cfg.training.run_dir = str(Path("runs") / run_name)
        print(f"[defectgen] run: {run_name}  world_size={world_size}")

    model = build_model(cfg.model)
    dataset = build_dataset(cfg.dataset)
    if len(dataset) == 0 and is_main_process():
        print("[defectgen] WARNING: dataset is empty — check `dataset.root`. Continuing for smoke test.")

    trainer = DDPTrainer(model=model, dataset=dataset, cfg=cfg.training, device=device, local_rank=local_rank)
    trainer.fit()
    cleanup_distributed()


if __name__ == "__main__":
    main()
