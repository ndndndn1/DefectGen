"""Distributed init / rank helpers. Single-GPU safe."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    """Initialize ``torch.distributed`` if launched via torchrun.

    Returns:
        (rank, world_size, local_rank).
    """
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1
