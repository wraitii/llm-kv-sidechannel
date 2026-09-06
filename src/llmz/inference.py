"""Shared prefix construction and reproducible KV interventions."""
from __future__ import annotations

import hashlib
import numpy as np
import mlx.core as mx

from .carriers import expand_batch
from .transport import RecursiveCarrierPolicy
from .move_alignment import sample_transport
from .tokenizer import BOS, SEP, PAD


def row_seed(row, seed=1337):
    identity = row.get("example_id", row["asm"])
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "little")


def prepare_prefix(tokenizer, rows, max_source_tokens, policy=None,
                   transport=False, seed=1337):
    """Build each row independently, then right-pad; layout is batch-invariant."""
    prefixes, boundaries, span_rows = [], [], []
    for row in rows:
        source = tokenizer.encode_source(row["asm"])
        if len(source) > max_source_tokens:
            raise ValueError("overlong source: filter it instead of silently changing its FEN task")
        ids = [BOS, *source, SEP, *tokenizer.pause_sequence()]
        boundary = len(source) + 1 if tokenizer.causal_pause else len(ids)
        spans = np.empty((0, 3), dtype=np.int32)
        rng = np.random.default_rng(row_seed(row, seed))
        if isinstance(policy, RecursiveCarrierPolicy):
            batch = {"x": np.array([ids], dtype=np.int32),
                     "valid": np.ones((1, len(ids)), dtype=bool),
                     "source_tokens": np.array([len(source)]),
                     "prefix_lengths": np.array([boundary]),
                     "output_positions": np.array([[len(ids)-1]])}
            batch, expanded_spans, _ = expand_batch(
                batch, sample_transport(policy, [source], tokenizer, rng), policy, tokenizer.carrier_ids, rng=rng)
            ids = batch["x"][0].tolist()
            boundary = int(batch["prefix_lengths"][0])
            if transport:
                spans = expanded_spans[0]
        elif policy is not None and transport:
            spans = sample_transport(policy, [source], tokenizer, rng)[0]
            spans = np.where(spans >= 0, spans + 1, spans)
        prefixes.append(ids)
        boundaries.append(boundary)
        span_rows.append(spans)
    width = max(map(len, prefixes))
    tokens = np.full((len(rows), width), PAD, dtype=np.int32)
    valid = np.zeros_like(tokens, dtype=bool)
    spans = np.full((len(rows), max(map(len, span_rows)), 3), -1, dtype=np.int32)
    for i, ids in enumerate(prefixes):
        tokens[i, :len(ids)] = ids
        valid[i, :len(ids)] = True
        spans[i, :len(span_rows[i])] = span_rows[i]
    return mx.array(tokens), mx.array(valid), mx.array(boundaries), mx.array(spans)


class DecodeSession:
    """Restart before the final prefix query, optionally again at each decode.

    A preserved shadow stream sees the same tokens and supplies scored selection
    decisions, so the intervention changes representations, never survivor IDs.
    """
    def __init__(self, model, tokens, valid, boundaries, spans=None,
                 window=None, mode="preserve"):
        if mode not in {"preserve", "restart", "restart-each"}:
            raise ValueError(f"unknown cache mode: {mode}")
        self.model, self.window, self.mode = model, window, mode
        self.shadow = None
        if mode == "preserve":
            self.logits, self.cache = model.prefill(tokens, valid, boundaries, window, spans)
        else:
            if model.attention_mode != "causal":
                raise ValueError("restart requires a causal model")
            last = mx.max(mx.where(valid, mx.arange(tokens.shape[1])[None], -1), axis=1)
            before = valid & (mx.arange(tokens.shape[1])[None] != last[:, None])
            _, original = model.prefill(tokens, before, boundaries, window, spans)
            query = tokens[mx.arange(tokens.shape[0]), last][:, None]
            _, self.shadow = model.decode(query, original, window)
            self.cache = model.restart(original, window,
                                       [live[:, :-1] for live in self.shadow.alive])
            self.logits, self.cache = model.decode(query, self.cache, window,
                                                   forced_alive=self.shadow.alive)

    def advance(self, tokens):
        if self.shadow is not None:
            _, self.shadow = self.model.decode(tokens, self.shadow, self.window)
        if self.mode == "restart-each":
            self.cache = self.model.restart(self.cache, self.window,
                                            [live[:, :-1] for live in self.shadow.alive])
        self.logits, self.cache = self.model.decode(
            tokens, self.cache, self.window,
            forced_alive=self.shadow.alive if self.shadow is not None else None)
        return self.logits
