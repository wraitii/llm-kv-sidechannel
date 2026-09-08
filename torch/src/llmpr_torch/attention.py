"""Qwen attention-policy configuration and dense correctness references."""
from __future__ import annotations

from copy import deepcopy

import torch


def configure_fixed_swa(config, window: int):
    """Return a copied Qwen config with sliding attention in every layer."""
    if window < 1:
        raise ValueError("window must be positive")
    configured = deepcopy(config)
    configured.use_sliding_window = True
    configured.sliding_window = window
    configured.max_window_layers = 0
    configured.layer_types = ["sliding_attention"] * configured.num_hidden_layers
    return configured


def dense_causal_mask(
    length: int,
    *,
    windows: int | torch.Tensor | None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive `[batch, 1, query, key]` reference mask.

    This deliberately materializes quadratic storage and is restricted to
    short correctness tests. It is not the 8K/16K training implementation.
    The current query counts against the window budget.
    """
    if length < 1:
        raise ValueError("length must be positive")
    q = torch.arange(length, device=device)[None, :, None]
    k = torch.arange(length, device=device)[None, None, :]
    allowed = k <= q
    if windows is None:
        batch = 1
    else:
        values = torch.as_tensor(windows, device=device, dtype=torch.long)
        if values.ndim == 0:
            values = values[None]
        if values.ndim != 1 or torch.any(values < 1):
            raise ValueError("windows must be a positive scalar or rank-one tensor")
        batch = values.shape[0]
        allowed = allowed & (k > q - values[:, None, None])
    allowed = torch.broadcast_to(allowed, (batch, length, length))
    zero = torch.zeros((), device=device, dtype=dtype)
    blocked = torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.where(allowed[:, None], zero, blocked)


def qwen_mask_mapping(mask: torch.Tensor, *, layer_type: str = "full_attention") -> dict[str, torch.Tensor]:
    """Wrap an already causal additive mask for Transformers Qwen3."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Qwen mask must have shape [batch, 1, query, key]")
    if layer_type not in {"full_attention", "sliding_attention"}:
        raise ValueError(f"unsupported Qwen layer type: {layer_type}")
    return {layer_type: mask}
