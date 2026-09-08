"""Atomic, exact training checkpoints for PyTorch experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 1


def capture_rng_state(generators: dict[str, torch.Generator] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "generators": {name: generator.get_state() for name, generator in (generators or {}).items()},
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any], generators: dict[str, torch.Generator] | None = None
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    available = generators or {}
    if set(state.get("generators", {})) != set(available):
        raise ValueError("checkpoint and runner generator names differ")
    for name, value in state.get("generators", {}).items():
        available[name].set_state(value.cpu())


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    step: int,
    micro_step: int,
    tokens_seen: int,
    config: dict[str, Any],
    generators: dict[str, torch.Generator] | None = None,
) -> None:
    """Write a checkpoint atomically; a completed rename is the commit point."""
    if step < 0 or micro_step < 0 or tokens_seen < 0:
        raise ValueError("checkpoint counters must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": {name: value.detach().cpu() for name, value in model.named_parameters()
                  if value.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "step": step,
        "micro_step": micro_step,
        "tokens_seen": tokens_seen,
        "config": config,
        "rng": capture_rng_state(generators),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    pointer = path.parent / "latest.json"
    pointer_tmp = pointer.with_name(f".{pointer.name}.{os.getpid()}.tmp")
    pointer_tmp.write_text(json.dumps({"checkpoint": path.name, "step": step}, indent=2) + "\n")
    os.replace(pointer_tmp, pointer)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    expected_config: dict[str, Any],
    generators: dict[str, torch.Generator] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, int]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if payload["config"] != expected_config:
        raise ValueError("resume config differs from checkpoint config")
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    if trainable.keys() != payload["model"].keys():
        raise ValueError("checkpoint trainable parameter set differs from model")
    with torch.no_grad():
        for name, parameter in trainable.items():
            saved = payload["model"][name]
            if parameter.shape != saved.shape:
                raise ValueError(f"checkpoint parameter shape differs: {name}")
            parameter.copy_(saved)
    optimizer.load_state_dict(payload["optimizer"])
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("checkpoint scheduler does not match runner")
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("checkpoint scaler does not match runner")
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"], generators)
    return {name: int(payload[name]) for name in ("step", "micro_step", "tokens_seen")}


def latest_checkpoint(run_dir: Path) -> Path:
    pointer = run_dir / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(f"no latest checkpoint pointer in {run_dir}")
    path = run_dir / json.loads(pointer.read_text())["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(f"latest checkpoint is missing: {path}")
    return path
