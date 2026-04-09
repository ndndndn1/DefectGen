"""Evaluation entry point: load checkpoint, generate samples, compute metrics."""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data import build_dataset
from src.evaluation import DefectCoverage, FID, IntraClassDiversity
from src.models import build_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="fid,defect_coverage,intra_class_diversity")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["ema"] if "ema" in state else state["model"])

    dataset = build_dataset(cfg.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    metrics = {}
    if "fid" in args.metrics:
        metrics["fid"] = FID(device=device)
    if "defect_coverage" in args.metrics:
        metrics["defect_coverage"] = DefectCoverage()
    if "intra_class_diversity" in args.metrics:
        metrics["intra_class_diversity"] = IntraClassDiversity()

    collected = 0
    for batch in loader:
        if collected >= args.num_samples:
            break
        real = batch["image"].to(device)
        b = real.shape[0]
        with torch.no_grad():
            fake = model.sample(b, device=device, num_steps=int(cfg.get("eval_steps", 50)))
        for m in metrics.values():
            m.update(real, fake)
        collected += b

    for name, m in metrics.items():
        print(f"{name}: {m.compute():.4f}")


if __name__ == "__main__":
    main()
