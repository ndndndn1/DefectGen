"""Inference: generate virtual defect images and save to disk."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.evaluation.visualization import save_grid
from src.inference import GenerationPipeline, export_onnx
from src.models import build_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--defect_type", type=int, default=None)
    parser.add_argument("--output", type=str, default="generated/")
    parser.add_argument("--export_onnx", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["ema"] if "ema" in state else state["model"])

    pipe = GenerationPipeline(model, device=device)
    samples = pipe.generate(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        defect_type=args.defect_type,
        num_steps=args.num_steps,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(samples):
        save_image(((img.clamp(-1, 1) + 1) / 2), out_dir / f"sample_{i:05d}.png")
    save_grid(samples[: min(64, len(samples))], str(out_dir / "grid.png"))
    print(f"[defectgen] saved {len(samples)} samples → {out_dir}")

    if args.export_onnx:
        export_onnx(model, args.export_onnx)
        print(f"[defectgen] exported ONNX → {args.export_onnx}")


if __name__ == "__main__":
    main()
