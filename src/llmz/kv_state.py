"""Runtime cache metadata, independent of physical cache length."""
from __future__ import annotations

import mlx.core as mx


class KVState(list):
    """Layer (K,V) pairs plus the information needed to replay an intervention.

    List compatibility keeps existing callers and mx.eval working. Padding and
    evicted entries may remain physically allocated, but cannot be attended to.
    """
    def __init__(self, layers, tokens, positions, valid, next_positions,
                 spans=None, alive=None, retention_scores=None):
        super().__init__(layers)
        self.tokens = tokens
        self.positions = positions
        self.valid = valid
        self.next_positions = next_positions
        self.spans = spans
        self.alive = alive
        self.retention_scores = retention_scores


def visibility(query_positions, key_positions, valid, spans=None, window=None):
    """Accept range spans [B,N,3] or expiry positions [B,horizon]."""
    q, k = query_positions[:, :, None], key_positions[:, None, :]
    allowed = (k <= q) & valid[:, None, :]
    if window is not None:
        if isinstance(window, mx.array):
            window = window[:, None, None]
        allowed = allowed & (k > q - window)
    if spans is not None:
        if spans.ndim == 2:
            expiry = mx.take_along_axis(spans, key_positions, axis=1)
            return allowed & (q <= expiry[:, None, :])
        for i in range(spans.shape[1]):
            s = spans[:, i]
            evicted = ((s[:, 0, None, None] >= 0)
                       & (k >= s[:, 0, None, None])
                       & (k < s[:, 1, None, None])
                       & (q > s[:, 2, None, None]))
            allowed = allowed & ~evicted
    return allowed


def additive_mask(allowed, dtype):
    return mx.where(allowed[:, None], mx.array(0, dtype=dtype),
                    mx.array(-1e9, dtype=dtype))
