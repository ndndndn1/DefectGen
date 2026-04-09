"""Smoke tests for MCP tool implementations (no MCP transport required)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.mcp.server import TOOLS, SESSION


def test_list_tools_static():
    assert {"list_models", "list_datasets", "load_checkpoint", "generate_defects", "evaluate", "export_onnx"} <= set(TOOLS)


def test_load_and_generate(tmp_path: Path, monkeypatch):
    # Sandbox the MCP server to tmp_path for this test.
    monkeypatch.setenv("DEFECTGEN_ALLOWED_ROOTS", str(tmp_path))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model:
  name: ddpm
  params:
    image_size: 16
    in_channels: 1
    num_steps: 10
    base_channels: 16
    num_classes: 3
dataset:
  name: wm811k
  root: /nonexistent
  image_size: 16
""".strip()
    )
    out = TOOLS["load_checkpoint"]["fn"](handle="t", config_path=str(cfg))
    assert out["handle"] == "t" and "ddpm" in out["model"]

    res = TOOLS["generate_defects"]["fn"](
        handle="t", output_dir=str(tmp_path / "gen"), num_samples=4, batch_size=2, num_steps=2, defect_type=0, seed=0
    )
    assert res["num_samples"] == 4
    assert all(Path(p).exists() for p in res["files"])
    assert Path(res["grid"]).exists()

    SESSION.models.clear()
    SESSION.pipelines.clear()


def test_path_sandbox_blocks_escape(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEFECTGEN_ALLOWED_ROOTS", str(tmp_path))
    with pytest.raises(PermissionError):
        TOOLS["load_checkpoint"]["fn"](handle="x", config_path="/etc/passwd")


def test_input_validation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEFECTGEN_ALLOWED_ROOTS", str(tmp_path))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  name: ddpm\n  params:\n    image_size: 16\n    in_channels: 1\n    num_steps: 10\n    base_channels: 16\n    num_classes: 3\ndataset:\n  name: wm811k\n  root: /nonexistent\n  image_size: 16\n"
    )
    TOOLS["load_checkpoint"]["fn"](handle="v", config_path=str(cfg))
    with pytest.raises(ValueError):
        TOOLS["generate_defects"]["fn"](handle="v", output_dir=str(tmp_path / "g"), num_samples=10**9)
    with pytest.raises(ValueError):
        TOOLS["generate_defects"]["fn"](handle="v", output_dir=str(tmp_path / "g"), defect_type=999)
    with pytest.raises(ValueError):
        TOOLS["load_checkpoint"]["fn"](handle="../bad", config_path=str(cfg))
    SESSION.models.clear()
    SESSION.pipelines.clear()
