"""Device and dtype selection shared by local and CUDA runners."""
from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def training_dtype(device: torch.device) -> torch.dtype:
    # Qwen ships in BF16. MPS support varies across operators, so FP32 is the
    # conservative local correctness path; CUDA uses BF16 on supported GPUs.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32
