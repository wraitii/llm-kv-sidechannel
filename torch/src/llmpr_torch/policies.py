"""Backend-independent policy specifications and reference visibility logic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FullAttention:
    kind: str = "full"


@dataclass(frozen=True)
class FixedSWA:
    window: int
    kind: str = "swa"

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be positive")


@dataclass(frozen=True)
class VariableSWA:
    minimum: int
    maximum: int
    kind: str = "variable_swa"

    def __post_init__(self) -> None:
        if self.minimum < 1 or self.maximum < self.minimum:
            raise ValueError("variable SWA requires 1 <= minimum <= maximum")

    def sample(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        return rng.integers(self.minimum, self.maximum + 1, size=batch_size,
                            dtype=np.int32)


@dataclass(frozen=True)
class StreamingLog:
    """Recent tokens plus irreversible log-age thinning of older tokens."""

    recent_tokens: int
    memory_tokens: int
    kind: str = "streaming_log"

    def __post_init__(self) -> None:
        if self.recent_tokens < 1 or self.memory_tokens < 1:
            raise ValueError("streaming-log budgets must be positive")

    @property
    def capacity(self) -> int:
        return self.recent_tokens + self.memory_tokens

    def survivors(self, length: int) -> tuple[int, ...]:
        """Reference schedule after admitting positions ``range(length)``.

        When full, preserve the oldest/newest older anchors and evict the
        interior older entry with the smallest log-age separation between its
        neighbours. Ties are resolved by the earlier storage index.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        memory: list[int] = []
        for query in range(length):
            newly_old = query - self.recent_tokens
            if newly_old < 0:
                continue
            memory.append(newly_old)
            if len(memory) <= self.memory_tokens:
                continue
            if self.memory_tokens == 1:
                remove = 0
            else:
                ages = newly_old - np.asarray(memory, dtype=np.float64) + 1
                gaps = np.log(ages[:-2]) - np.log(ages[2:])
                remove = int(np.argmin(gaps)) + 1
            memory.pop(remove)
        recent = range(max(0, length - self.recent_tokens), length)
        return tuple([*memory, *recent])


def causal_visibility(length: int, windows: int | np.ndarray | None = None) -> np.ndarray:
    """Dense boolean reference mask, shaped ``[batch, query, key]``."""
    q = np.arange(length)[None, :, None]
    k = np.arange(length)[None, None, :]
    allowed = k <= q
    if windows is None:
        return allowed
    values = np.asarray(windows, dtype=np.int32)
    if values.ndim == 0:
        values = values[None]
    if np.any(values < 1):
        raise ValueError("windows must be positive")
    return np.broadcast_to(allowed, (len(values), length, length)).copy() & (
        k > q - values[:, None, None])


def streaming_visibility(length: int, policy: StreamingLog) -> np.ndarray:
    """Dense reference visibility for the irreversible streaming schedule."""
    if length < 1:
        raise ValueError("length must be positive")
    mask = np.zeros((1, length, length), dtype=np.bool_)
    for query in range(length):
        mask[0, query, list(policy.survivors(query + 1))] = True
    return mask


def additive_from_visibility(
    visible: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert ``[batch, query, key]`` visibility into a Qwen additive mask."""
    if isinstance(visible, np.ndarray) and not visible.flags.writeable:
        visible = visible.copy()
    allowed = torch.as_tensor(visible, device=device, dtype=torch.bool)
    if allowed.ndim != 3:
        raise ValueError("visibility must have shape [batch, query, key]")
    return torch.where(
        allowed[:, None],
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype),
    )


def survivor_indices(
    length: int,
    policy: FullAttention | FixedSWA | StreamingLog,
) -> tuple[int, ...]:
    """Return the token IDs available when reconstructing before ``length``."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if isinstance(policy, FullAttention):
        return tuple(range(length))
    if isinstance(policy, FixedSWA):
        return tuple(range(max(0, length - policy.window), length))
    if isinstance(policy, StreamingLog):
        return policy.survivors(length)
    raise TypeError(f"unsupported reconstruction policy: {type(policy).__name__}")
