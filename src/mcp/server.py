"""DefectGen MCP server.

Implements the Model Context Protocol so external clients can:

1. **list_models** — discover supported generators (DDPM, Flow Matching).
2. **list_datasets** — discover supported industrial datasets.
3. **load_checkpoint** — instantiate a model from a YAML config + .pt file.
4. **generate_defects** — sample N virtual defect images and write them to a
   client-specified output directory (returned in the tool result).
5. **evaluate** — compute FID / DefectCoverage / IntraClassDiversity against a
   real dataset.
6. **export_onnx** — export the per-step denoiser to a client path.

Run as stdio server::

    python -m src.mcp.server
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torchvision.utils import save_image

logger = logging.getLogger("defectgen.mcp")

# ----------------------------------------------------------- security policy
# Hard caps protect against client-driven OOM / disk exhaustion.
MAX_NUM_SAMPLES = 10_000
MAX_BATCH_SIZE = 256
MAX_NUM_STEPS = 1_000
MAX_LOADED_MODELS = 4
HANDLE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def _allowed_roots() -> list[Path]:
    """Filesystem roots the MCP server is permitted to read/write under.

    Configured via the ``DEFECTGEN_ALLOWED_ROOTS`` env var (``:``-separated).
    Defaults to ``$CWD/configs``, ``$CWD/runs``, ``$CWD/generated``,
    ``$CWD/export`` so a default install is sandboxed to the project tree.
    """
    raw = os.environ.get("DEFECTGEN_ALLOWED_ROOTS")
    if raw:
        roots = [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p]
    else:
        cwd = Path.cwd().resolve()
        roots = [cwd / "configs", cwd / "runs", cwd / "generated", cwd / "export"]
    return [r for r in roots if r]


def _safe_path(p: str, *, must_exist: bool = False, for_write: bool = False) -> Path:
    """Resolve ``p`` and verify it sits inside an allowed root.

    Raises:
        PermissionError: if the resolved path escapes every allowed root.
        FileNotFoundError: if ``must_exist`` and the file is missing.
    """
    if not isinstance(p, str) or not p:
        raise ValueError("path must be a non-empty string")
    resolved = Path(p).expanduser().resolve()
    roots = _allowed_roots()
    if not any(_is_relative_to(resolved, r) for r in roots):
        raise PermissionError(
            f"path '{resolved}' is outside allowed roots; "
            f"set DEFECTGEN_ALLOWED_ROOTS to permit it"
        )
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    if for_write:
        # Ensure parent directory exists *inside* an allowed root.
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_handle(handle: str) -> str:
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        raise ValueError("handle must match [A-Za-z0-9_.-]{1,64}")
    return handle


def _clamp(value: int, lo: int, hi: int, name: str) -> int:
    if not isinstance(value, int) or value < lo or value > hi:
        raise ValueError(f"{name} must be an int in [{lo}, {hi}]")
    return value

from ..data import build_dataset
from ..evaluation import DefectCoverage, FID, IntraClassDiversity
from ..evaluation.visualization import save_grid
from ..inference import GenerationPipeline, export_onnx as _export_onnx
from ..models import build_model
from ..models.base import BaseGenerator
from ..utils import load_config


# ----------------------------------------------------------- session state
@dataclass
class _Session:
    """In-process state shared across MCP tool calls."""

    models: dict[str, BaseGenerator] = field(default_factory=dict)
    pipelines: dict[str, GenerationPipeline] = field(default_factory=dict)
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def register(self, handle: str, model: BaseGenerator) -> None:
        self.models[handle] = model
        self.pipelines[handle] = GenerationPipeline(model, device=self.device)

    def get(self, handle: str) -> BaseGenerator:
        if handle not in self.models:
            raise KeyError(f"Unknown model handle '{handle}'. Call load_checkpoint first.")
        return self.models[handle]


SESSION = _Session()


# ------------------------------------------------------------- tool impls
def _tool_list_models() -> dict[str, Any]:
    return {
        "available": [
            {"name": "ddpm", "class": "DDPMGenerator", "paper": "Ho et al., NeurIPS 2020"},
            {"name": "flow_matching", "class": "FlowMatchingGenerator", "paper": "Lipman et al. / Tong et al., 2023"},
        ],
        "loaded": [
            {"handle": h, "num_classes": m.num_classes, "image_size": m.image_size, "in_channels": m.in_channels}
            for h, m in SESSION.models.items()
        ],
    }


def _tool_list_datasets() -> dict[str, Any]:
    return {
        "datasets": [
            {"name": "wm811k", "domain": "semiconductor wafer", "channels": 1},
            {"name": "deeppcb", "domain": "PCB", "channels": 3},
            {"name": "neudet", "domain": "steel surface", "channels": 3},
            {"name": "mvtec", "domain": "general anomaly", "channels": 3},
        ]
    }


def _tool_load_checkpoint(handle: str, config_path: str, checkpoint_path: Optional[str] = None) -> dict[str, Any]:
    handle = _validate_handle(handle)
    cfg_path = _safe_path(config_path, must_exist=True)
    cfg = load_config(cfg_path)
    if len(SESSION.models) >= MAX_LOADED_MODELS and handle not in SESSION.models:
        raise RuntimeError(
            f"max loaded models reached ({MAX_LOADED_MODELS}); call unload_model first"
        )
    model = build_model(cfg.model)
    if checkpoint_path:
        ckpt = _safe_path(checkpoint_path, must_exist=True)
        # weights_only=True blocks pickle RCE on untrusted .pt files (PyTorch ≥2.4).
        state = torch.load(ckpt, map_location=SESSION.device, weights_only=True)
        model.load_state_dict(state.get("ema") or state["model"])
        ckpt_status = f"loaded from {ckpt}"
    else:
        ckpt_status = "untrained (random init)"
    SESSION.register(handle, model)
    return {
        "handle": handle,
        "model": cfg.model.name,
        "checkpoint": ckpt_status,
        "num_parameters": model.num_parameters,
        "device": str(SESSION.device),
    }


def _tool_generate_defects(
    handle: str,
    output_dir: str,
    num_samples: int = 16,
    batch_size: int = 8,
    num_steps: int = 50,
    defect_type: Optional[int] = None,
    save_grid_image: bool = True,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    handle = _validate_handle(handle)
    num_samples = _clamp(num_samples, 1, MAX_NUM_SAMPLES, "num_samples")
    batch_size = _clamp(batch_size, 1, MAX_BATCH_SIZE, "batch_size")
    num_steps = _clamp(num_steps, 1, MAX_NUM_STEPS, "num_steps")
    if seed is not None:
        torch.manual_seed(int(seed))
    model = SESSION.get(handle)
    if defect_type is not None:
        if model.num_classes is None or not (0 <= int(defect_type) < model.num_classes):
            raise ValueError(
                f"defect_type out of range for model with num_classes={model.num_classes}"
            )
    pipe = SESSION.pipelines[handle]
    out_dir = _safe_path(output_dir, for_write=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pipe.generate(
        num_samples=num_samples,
        batch_size=batch_size,
        defect_type=defect_type,
        num_steps=num_steps,
    )

    paths: list[str] = []
    for i, img in enumerate(samples):
        p = out_dir / f"sample_{i:05d}.png"
        save_image(((img.clamp(-1, 1) + 1) / 2), p)
        paths.append(str(p))

    grid_path = None
    if save_grid_image:
        grid_path = str(out_dir / "grid.png")
        save_grid(samples[: min(64, len(samples))], grid_path)

    return {
        "handle": handle,
        "output_dir": str(out_dir),
        "num_samples": len(paths),
        "defect_type": defect_type,
        "files": paths,
        "grid": grid_path,
    }


def _tool_evaluate(
    handle: str,
    dataset_config_path: str,
    num_samples: int = 256,
    batch_size: int = 32,
    metrics: Optional[list[str]] = None,
    num_steps: int = 50,
) -> dict[str, Any]:
    handle = _validate_handle(handle)
    num_samples = _clamp(num_samples, 1, MAX_NUM_SAMPLES, "num_samples")
    batch_size = _clamp(batch_size, 1, MAX_BATCH_SIZE, "batch_size")
    num_steps = _clamp(num_steps, 1, MAX_NUM_STEPS, "num_steps")
    metrics = metrics or ["fid", "defect_coverage", "intra_class_diversity"]
    model = SESSION.get(handle).eval()
    cfg = load_config(_safe_path(dataset_config_path, must_exist=True))
    dataset = build_dataset(cfg.dataset)
    if len(dataset) == 0:
        return {"error": f"dataset {cfg.dataset.name} at {cfg.dataset.root} is empty"}
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    metric_objs: dict[str, Any] = {}
    if "fid" in metrics:
        metric_objs["fid"] = FID(device=SESSION.device)
    if "defect_coverage" in metrics:
        metric_objs["defect_coverage"] = DefectCoverage()
    if "intra_class_diversity" in metrics:
        metric_objs["intra_class_diversity"] = IntraClassDiversity()

    seen = 0
    for batch in loader:
        if seen >= num_samples:
            break
        real = batch["image"].to(SESSION.device)
        with torch.no_grad():
            fake = model.sample(real.shape[0], device=SESSION.device, num_steps=num_steps)
        for m in metric_objs.values():
            m.update(real, fake)
        seen += real.shape[0]

    return {name: float(m.compute()) for name, m in metric_objs.items()}


def _tool_export_onnx(handle: str, output_path: str, opset: int = 17) -> dict[str, Any]:
    handle = _validate_handle(handle)
    if not isinstance(opset, int) or not (7 <= opset <= 20):
        raise ValueError("opset must be int in [7, 20]")
    model = SESSION.get(handle)
    out = _safe_path(output_path, for_write=True)
    _export_onnx(model, str(out), opset=opset)
    return {"handle": handle, "onnx_path": str(out), "opset": opset}


def _tool_unload_model(handle: str) -> dict[str, Any]:
    handle = _validate_handle(handle)
    SESSION.models.pop(handle, None)
    SESSION.pipelines.pop(handle, None)
    return {"unloaded": handle, "remaining": list(SESSION.models)}


# ------------------------------------------------------------- dispatch
TOOLS: dict[str, dict[str, Any]] = {
    "list_models": {
        "description": "List supported and currently loaded DefectGen generative models.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": lambda **kw: _tool_list_models(),
    },
    "list_datasets": {
        "description": "List supported industrial defect datasets.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": lambda **kw: _tool_list_datasets(),
    },
    "load_checkpoint": {
        "description": "Build a generator from a YAML config and (optionally) load a .pt checkpoint. Registers it under `handle` for subsequent calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "Identifier to reference this model in later tool calls."},
                "config_path": {"type": "string", "description": "Path to a configs/*.yaml file."},
                "checkpoint_path": {"type": "string", "description": "Optional .pt checkpoint path."},
            },
            "required": ["handle", "config_path"],
        },
        "fn": lambda **kw: _tool_load_checkpoint(**kw),
    },
    "generate_defects": {
        "description": "Generate virtual defect images with a loaded model and write them to a client-specified `output_dir`. Returns the absolute file paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "output_dir": {"type": "string", "description": "Client-specified destination directory; created if missing."},
                "num_samples": {"type": "integer", "minimum": 1, "maximum": MAX_NUM_SAMPLES, "default": 16},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": MAX_BATCH_SIZE, "default": 8},
                "num_steps": {"type": "integer", "minimum": 1, "maximum": MAX_NUM_STEPS, "default": 50},
                "defect_type": {"type": "integer", "description": "Optional class index for class-conditional models."},
                "save_grid_image": {"type": "boolean", "default": True},
                "seed": {"type": "integer"},
            },
            "required": ["handle", "output_dir"],
        },
        "fn": lambda **kw: _tool_generate_defects(**kw),
    },
    "evaluate": {
        "description": "Compute FID / DefectCoverage / IntraClassDiversity for a loaded model against the dataset specified in a config.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "dataset_config_path": {"type": "string"},
                "num_samples": {"type": "integer", "minimum": 1, "maximum": MAX_NUM_SAMPLES, "default": 256},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": MAX_BATCH_SIZE, "default": 32},
                "num_steps": {"type": "integer", "minimum": 1, "maximum": MAX_NUM_STEPS, "default": 50},
                "metrics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["handle", "dataset_config_path"],
        },
        "fn": lambda **kw: _tool_evaluate(**kw),
    },
    "unload_model": {
        "description": "Drop a previously loaded model handle and free its memory.",
        "inputSchema": {
            "type": "object",
            "properties": {"handle": {"type": "string"}},
            "required": ["handle"],
        },
        "fn": lambda **kw: _tool_unload_model(**kw),
    },
    "export_onnx": {
        "description": "Export the per-step denoiser / vector-field network of a loaded model to ONNX at `output_path`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "output_path": {"type": "string"},
                "opset": {"type": "integer", "default": 17},
            },
            "required": ["handle", "output_path"],
        },
        "fn": lambda **kw: _tool_export_onnx(**kw),
    },
}


# --------------------------------------------- MCP server (official SDK)
def build_server():
    """Build a stdio MCP server using the official ``mcp`` SDK if available."""
    from mcp.server import Server  # type: ignore
    from mcp.types import TextContent, Tool  # type: ignore

    server = Server("defectgen")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(name=name, description=spec["description"], inputSchema=spec["inputSchema"])
            for name, spec in TOOLS.items()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name not in TOOLS:
            raise ValueError(f"Unknown tool: {name}")
        try:
            result = TOOLS[name]["fn"](**(arguments or {}))
        except (ValueError, PermissionError, FileNotFoundError, KeyError) as exc:
            # Validation / policy errors: safe to surface verbatim.
            logger.warning("tool %s rejected: %s", name, exc)
            result = {"error": type(exc).__name__, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            # Unexpected: log full detail server-side, return opaque message.
            logger.exception("tool %s failed", name)
            result = {"error": type(exc).__name__, "message": "internal error; see server logs"}
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

    return server


async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server  # type: ignore

    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Console entry point: start a stdio MCP server."""
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
