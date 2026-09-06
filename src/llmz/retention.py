"""Learned, per-layer eviction with immutable retained KVs.

Dense storage is intentional: this is an experiment in information transport,
not a cache kernel benchmark. Hard selection is used in every forward pass.
A sigmoid straight-through estimator supplies gradients to the scoring head.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RetentionScorer(nn.Module):
    def __init__(self, d_model, value_dim, config, dtype):
        super().__init__()
        self.window = int(config.get("recent_window", 32))
        self.memory = int(config.get("memory_tokens", 16))
        self.temperature = float(config.get("temperature", 1.0))
        dim = int(config.get("score_dim", 32))
        if self.window < 1 or self.memory < 0 or dim < 1 or self.temperature <= 0:
            raise ValueError("invalid scored_eviction budget, dimension or temperature")
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
        """Score only entries crossing from the recent window into memory."""
        flat = values.transpose(0, 2, 1, 3).reshape(
            values.shape[0], values.shape[2], -1)
        selected = mx.sum(flat * candidates[:, :, None], axis=1)
        score = (mx.sum(self.query(h[:, 0]).astype(mx.float32)
                        * self.key(selected).astype(mx.float32), axis=-1)
                 * self.scale
                 + self.priority(selected).astype(mx.float32)[:, 0])
        return score[:, None]

    def sequence_scores(self, h, values):
        """Vectorize the one-time score assigned at each eviction boundary."""
        length = h.shape[1]
        if length <= self.window:
            return mx.zeros((h.shape[0], length), dtype=mx.float32)
        flat = values.transpose(0, 2, 1, 3).reshape(
            values.shape[0], values.shape[2], -1)
        queries = self.query(h[:, self.window:]).astype(mx.float32)
        candidates = flat[:, :-self.window]
        assigned = (mx.sum(queries * self.key(candidates).astype(mx.float32), axis=-1)
                    * self.scale
                    + self.priority(candidates).astype(mx.float32)[:, :, 0])
        return mx.concatenate(
            [assigned, mx.zeros((h.shape[0], self.window), dtype=mx.float32)],
            axis=1)

    def continuation_scores(self, h, values, stored_scores, query_positions,
                            history_length):
        """Assign all scores that will cross the boundary in this continuation."""
        length = h.shape[1]
        flat = values.transpose(0, 2, 1, 3).reshape(
            values.shape[0], values.shape[2], -1)
        candidate_positions = query_positions - self.window
        start = query_positions[:, :1]
        indices = mx.where(candidate_positions < start, candidate_positions,
                           history_length + candidate_positions - start)
        usable = indices >= 0
        clipped = mx.maximum(indices, 0)
        batch = mx.arange(h.shape[0])[:, None]
        candidates = flat[batch, clipped]
        assigned = (mx.sum(self.query(h).astype(mx.float32)
                           * self.key(candidates).astype(mx.float32), axis=-1)
                    * self.scale
                    + self.priority(candidates).astype(mx.float32)[:, :, 0])
        slots = mx.arange(flat.shape[1])[None, None, :] == indices[:, :, None]
        slots = slots & usable[:, :, None]
        updates = mx.sum(assigned[:, :, None] * slots, axis=1)
        touched = mx.any(slots, axis=1)
        return mx.where(touched, updates, stored_scores)

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

    def select_sequence(self, scores, eligible, positions, query_positions):
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
        soft = mx.where(recent, 1.0, mx.where(
            older, mx.sigmoid(scores[:, None, :] / self.temperature), 0.0))
        return keep, soft + mx.stop_gradient(keep.astype(mx.float32) - soft)

    def sequence(self, h, values, positions, valid):
        scores = self.sequence_scores(h, values)
        keep, gates = self.select_sequence(scores, valid, positions, positions)
        last = mx.max(mx.where(valid, mx.arange(valid.shape[1])[None], -1), axis=1)
        alive = keep[mx.arange(valid.shape[0]), last]
        return gates, alive, scores


def gated_attention(q, k, v, gate, scale, mask):
    """Exactly hard-support softmax forward, with a soft gate gradient surrogate."""
    group = q.shape[1] // k.shape[1]
    if group > 1:
        k, v = mx.repeat(k, group, axis=1), mx.repeat(v, group, axis=1)
    scores = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2) * scale
    scores = scores + mask.astype(mx.float32)
    # Center on retained entries, then bound only the surrogate's excluded logits.
    hard = mx.stop_gradient(gate > 0.5)[:, None]
    center = mx.max(mx.where(hard, scores, -1e9), axis=-1, keepdims=True)
    weights = mx.exp(mx.clip(scores - center, -80, 30)) * gate[:, None]
    weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
    return weights.astype(v.dtype) @ v
