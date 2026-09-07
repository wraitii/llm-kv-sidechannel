"""Learned, per-layer eviction with immutable retained KVs.

Dense storage is intentional: this is an experiment in information transport,
not a cache kernel benchmark. Every forward pass uses hard selection. Training
can use either the original sigmoid straight-through estimator or a
fixed-budget Soft-TopK straight-through gradient.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _laplace_cdf(value):
    """Laplace CDF used by LKV's budget-conserving Soft-TopK operator."""
    return 0.5 * (1.0 + mx.sign(value) * (1.0 - mx.exp(-mx.abs(value))))


def soft_topk(scores, eligible, count, temperature, iterations=32):
    """Return soft gates whose mass is ``min(count, eligible.sum())``.

    A detached bisection finds the forward threshold. The attached surrogate
    below gives that threshold its implicit derivative, so increasing one score
    reduces competing gates rather than acting like independent sigmoids.
    """
    if count < 1:
        return mx.zeros_like(scores, dtype=mx.float32)
    x = scores.astype(mx.float32)
    valid = eligible.astype(mx.bool_)
    available = mx.sum(valid, axis=-1, keepdims=True)
    target = mx.minimum(mx.array(float(count)), available.astype(mx.float32))
    # A 40-temperature margin saturates the bracketing endpoint in float32.
    lo = mx.min(mx.where(valid, x, 1e9), axis=-1, keepdims=True) - 40 * temperature
    hi = mx.max(mx.where(valid, x, -1e9), axis=-1, keepdims=True) + 40 * temperature
    for _ in range(iterations):
        midpoint = (lo + hi) * 0.5
        mass = mx.sum(mx.where(valid, _laplace_cdf((x - midpoint) / temperature), 0.0),
                      axis=-1, keepdims=True)
        lo = mx.where(mass > target, midpoint, lo)
        hi = mx.where(mass > target, hi, midpoint)
    threshold_value = (lo + hi) * 0.5
    provisional = mx.where(
        valid, _laplace_cdf((x - mx.stop_gradient(threshold_value)) / temperature), 0.0)
    slope = mx.stop_gradient(
        mx.where(valid, 0.5 * mx.exp(-mx.abs(
            (x - threshold_value) / temperature)) / temperature, 0.0))
    threshold_surrogate = (mx.sum(slope * x, axis=-1, keepdims=True)
                           / mx.maximum(mx.sum(slope, axis=-1, keepdims=True), 1e-12))
    threshold = (threshold_surrogate
                 + mx.stop_gradient(threshold_value - threshold_surrogate))
    gates = mx.where(valid, _laplace_cdf((x - threshold) / temperature), 0.0)
    # Exact endpoint semantics when the budget holds all (or none) of a row.
    return mx.where((target >= available) & valid, 1.0,
                    mx.where(target <= 0, 0.0, gates))


class RetentionScorer(nn.Module):
    def __init__(self, d_model, value_dim, config, dtype):
        super().__init__()
        self.window = int(config.get("recent_window", 32))
        self.memory = int(config.get("memory_tokens", 16))
        # Preserve the original eviction-boundary scoring unless configured
        # otherwise. A delay of zero scores an entry from its own hidden state.
        self.scoring_delay = int(config.get("scoring_delay", self.window))
        self.temperature = float(config.get("temperature", 1.0))
        self.training_method = config.get("training_method", "hard_st")
        dim = int(config.get("score_dim", 32))
        if (self.window < 1 or self.memory < 0 or dim < 1 or self.temperature <= 0
                or not 0 <= self.scoring_delay <= self.window
                or self.training_method not in {"hard_st", "soft_topk"}):
            raise ValueError("invalid scored_eviction budget, scoring delay, "
                             "dimension or temperature")
        self.query = nn.Linear(d_model, dim, bias=False)
        self.key = nn.Linear(value_dim, dim, bias=False)
        self.priority = nn.Linear(value_dim, 1, bias=False)
        for layer in (self.query, self.key, self.priority):
            layer.weight = (mx.random.normal(layer.weight.shape) * 0.02).astype(dtype)
        self.scale = dim ** -0.5

    def scores(self, h, values):
        """Compatibility helper for inspecting all query/key score pairs."""
        values = values.transpose(0, 2, 1, 3).reshape(values.shape[0], values.shape[2], -1)
        return ((self.query(h).astype(mx.float32)
                 @ self.key(values).astype(mx.float32).transpose(0, 2, 1)) * self.scale
                + self.priority(values).astype(mx.float32).transpose(0, 2, 1))

    def assign(self, h, values, candidates):
        """Assign a score to the entry whose configured delay just elapsed."""
        flat = values.transpose(0, 2, 1, 3).reshape(
            values.shape[0], values.shape[2], -1)
        selected = mx.sum(flat * candidates[:, :, None], axis=1)
        score = (mx.sum(self.query(h[:, 0]).astype(mx.float32)
                        * self.key(selected).astype(mx.float32), axis=-1)
                 * self.scale
                 + self.priority(selected).astype(mx.float32)[:, 0])
        return score[:, None]

    def sequence_scores(self, h, values):
        """Vectorize one-time scores assigned after ``scoring_delay`` tokens."""
        length = h.shape[1]
        if length <= self.scoring_delay:
            return mx.zeros((h.shape[0], length), dtype=mx.float32)
        flat = values.transpose(0, 2, 1, 3).reshape(
            values.shape[0], values.shape[2], -1)
        queries = self.query(h[:, self.scoring_delay:]).astype(mx.float32)
        candidates = (flat if self.scoring_delay == 0
                      else flat[:, :-self.scoring_delay])
        assigned = (mx.sum(queries * self.key(candidates).astype(mx.float32), axis=-1)
                    * self.scale
                    + self.priority(candidates).astype(mx.float32)[:, :, 0])
        return mx.concatenate(
            [assigned, mx.zeros(
                (h.shape[0], self.scoring_delay), dtype=mx.float32)],
            axis=1)

    def select(self, scores, eligible, positions, query_position):
        """Select from previous survivors + new entry; deleted entries never return."""
        recent = eligible & (positions > query_position[:, None] - self.window)
        older = eligible & ~recent
        count = min(self.memory, scores.shape[-1])
        keep = recent
        if count:
            ranked = mx.argpartition(
                mx.where(older, mx.stop_gradient(scores), -1e9),
                scores.shape[-1] - count, axis=-1)
            chosen = ranked[:, -count:]
            selected = mx.any(mx.arange(scores.shape[-1])[None, :, None]
                              == chosen[:, None, :], axis=-1)
            keep = keep | (older & selected)
        soft = mx.where(recent, 1.0,
                        mx.where(older, mx.sigmoid(scores / self.temperature), 0.0))
        gate = soft + mx.stop_gradient(keep.astype(mx.float32) - soft)
        return keep, gate

    def select_sequence(self, scores, eligible, positions, query_positions,
                        training_temperature=None):
        """Select every query in parallel for immutable entry scores."""
        causal = positions[:, None, :] <= query_positions[:, :, None]
        recent = (eligible[:, None, :] & causal
                  & (positions[:, None, :]
                     > query_positions[:, :, None] - self.window))
        older = eligible[:, None, :] & causal & ~recent
        keep = recent
        count = min(self.memory, scores.shape[-1])
        if count:
            ranked = mx.argpartition(
                mx.where(older, mx.stop_gradient(scores[:, None, :]), -1e9),
                scores.shape[-1] - count, axis=-1)
            chosen = ranked[:, :, -count:]
            selected = mx.any(mx.arange(scores.shape[-1])[None, None, :, None]
                              == chosen[:, :, None, :], axis=-1)
            keep = keep | (older & selected)
        if self.training_method == "soft_topk" and training_temperature is not None:
            soft = mx.where(recent, 1.0, soft_topk(
                mx.broadcast_to(scores[:, None, :], older.shape), older,
                count, training_temperature))
            gate = soft + mx.stop_gradient(keep.astype(mx.float32) - soft)
            return keep, gate
        soft = mx.where(recent, 1.0, mx.where(
            older, mx.sigmoid(scores[:, None, :] / self.temperature), 0.0))
        return keep, soft + mx.stop_gradient(keep.astype(mx.float32) - soft)

    def sequence(self, h, values, positions, valid, training_temperature=None):
        scores = self.sequence_scores(h, values)
        keep, gates = self.select_sequence(
            scores, valid, positions, positions, training_temperature)
        last = mx.max(mx.where(valid, mx.arange(valid.shape[1])[None], -1), axis=1)
        alive = keep[mx.arange(valid.shape[0]), last]
        return gates, alive, scores


def gated_attention(q, k, v, gate, scale, mask, capture=None):
    """Multiplicatively gate and renormalize attention probabilities."""
    group = q.shape[1] // k.shape[1]
    if group > 1:
        k, v = mx.repeat(k, group, axis=1), mx.repeat(v, group, axis=1)
    scores = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2) * scale
    scores = scores + mask.astype(mx.float32)
    # Center on entries with nonzero gate mass before bounding excluded logits.
    active = mx.stop_gradient(gate > 0)[:, None]
    center = mx.max(mx.where(active, scores, -1e9), axis=-1, keepdims=True)
    weights = mx.exp(mx.clip(scores - center, -80, 30)) * gate[:, None]
    weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
    if capture is not None:
        capture.append(weights)
    return weights.astype(v.dtype) @ v
