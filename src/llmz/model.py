"""A compact MLX transformer with prefix-LM attention."""
from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .transport import max_carrier_tokens
from .kv_state import KVState, visibility, additive_mask
from .retention import RetentionScorer, gated_attention

_ROPE_CACHE = {}


def rope_tables(length: int, dim: int, theta: float, dtype):
    key = (length, dim, theta, dtype)
    if key not in _ROPE_CACHE:
        inv = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        angles = mx.arange(length)[:, None].astype(mx.float32) * inv[None, :]
        _ROPE_CACHE[key] = (mx.repeat(mx.cos(angles), 2, axis=-1).astype(dtype),
                            mx.repeat(mx.sin(angles), 2, axis=-1).astype(dtype))
    return _ROPE_CACHE[key]


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        scale = mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2,
                                axis=-1, keepdims=True) + self.eps)
        return x * scale.astype(x.dtype) * self.weight


def apply_rope(x, cos, sin, offset: int = 0, positions=None):
    if positions is None:
        end = offset + x.shape[-2]
        cos, sin = cos[None, None, offset:end, :], sin[None, None, offset:end, :]
    else:
        cos, sin = cos[positions][:, None], sin[positions][:, None]
    even, odd = x[..., ::2], x[..., 1::2]
    rotated = mx.stack((-odd, even), axis=-1).reshape(x.shape)
    return x * cos + rotated * sin


def _window_term(sliding_window, q, k):
    """Return the SWA band condition for a scalar or per-row window array."""
    window = mx.array(sliding_window) if isinstance(sliding_window, np.ndarray) else sliding_window
    if isinstance(window, mx.array):
        window = window.astype(mx.int32)[:, None, None]
    return k > q - window


def causal_mask(valid, dtype=mx.float32, sliding_window: int | None = None):
    """Full causal attention, optionally restricted to a local token window."""
    if (sliding_window is not None and not isinstance(sliding_window, mx.array)
            and not isinstance(sliding_window, np.ndarray)
            and sliding_window < 1):
        raise ValueError("sliding_window must be positive")
    length = valid.shape[1]
    q = mx.arange(length)[None, :, None]
    k = mx.arange(length)[None, None, :]
    allowed = (k <= q) & valid[:, None, :]
    if sliding_window is not None:
        allowed = allowed & _window_term(sliding_window, q, k)
    return mx.where(allowed[:, None, :, :], mx.array(0, dtype=dtype),
                    mx.array(-1e9, dtype=dtype))


def decode_sliding_mask(cache_length: int, sliding_window: int, dtype):
    """Mask for one cached decoding query, including its just-appended key."""
    if sliding_window < 1:
        raise ValueError("sliding_window must be positive")
    keys = mx.arange(cache_length)[None, None, None, :]
    allowed = keys >= cache_length - sliding_window
    return mx.where(allowed, mx.array(0, dtype=dtype), mx.array(-1e9, dtype=dtype))


def same_pass_transport_mask(valid, spans, dtype=mx.float32,
                             sliding_window: int | None = None):
    """Causal mask with evicted key ranges per ``[start, end, visible_until]``.

    ``spans`` is ``[batch, n, 3]`` in sequence coordinates. Keys in
    ``[start, end)`` are hidden from every query strictly after
    ``visible_until``; queries up to ``visible_until`` still read them.
    Padded ``(-1, -1, -1)`` spans are ignored.

    ``sliding_window`` optionally applies the ordinary causal SWA band as an
    additional restriction.  Thus sparse Memento-style eviction and SWA are
    independent, composable training conditions.
    """
    if (sliding_window is not None and not isinstance(sliding_window, mx.array)
            and not isinstance(sliding_window, np.ndarray)
            and sliding_window < 1):
        raise ValueError("sliding_window must be positive")
    length = valid.shape[1]
    q = mx.arange(length)[None, :, None]
    k = mx.arange(length)[None, None, :]
    allowed = (k <= q) & valid[:, None, :]
    if sliding_window is not None:
        allowed = allowed & _window_term(sliding_window, q, k)
    for index in range(spans.shape[1]):
        start = spans[:, index, 0][:, None, None]
        end = spans[:, index, 1][:, None, None]
        visible = spans[:, index, 2][:, None, None]
        active = start >= 0
        evicted = ((k >= start) & (k < end) & (q > visible) & active)
        allowed = allowed & ~evicted
    return mx.where(allowed[:, None, :, :], mx.array(0, dtype=dtype),
                    mx.array(-1e9, dtype=dtype))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, kv_heads: int, hidden: int,
                 layers: int, max_length: int, rope_theta: float, dtype, scored_eviction=None):
        super().__init__()
        if d % heads or heads % kv_heads:
            raise ValueError("d_model must be divisible by heads; heads by kv_heads")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, d // heads
        self.max_length, self.rope_theta, self.dtype = max_length, rope_theta, dtype
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, heads * self.head_dim, bias=False)
        self.k = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        for layer in (self.q, self.k, self.v, self.o, self.mlp_in, self.down):
            layer.weight = (mx.random.normal(layer.weight.shape) * 0.02).astype(dtype)
        scale = 0.02 / (2 * layers) ** 0.5
        self.o.weight = (mx.random.normal(self.o.weight.shape) * scale).astype(dtype)
        self.down.weight = (mx.random.normal(self.down.weight.shape) * scale).astype(dtype)

        self.retention = (RetentionScorer(d, kv_heads * self.head_dim, scored_eviction, dtype)
                          if scored_eviction else None)

    def _qkv(self, x, offset: int = 0, positions=None):
        h = self.n1(x)
        batch, length, _ = h.shape
        q = self.q(h).reshape(
            batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(
            batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(
            batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_tables(
            self.max_length, self.head_dim, self.rope_theta, self.dtype)
        return (apply_rope(q, cos, sin, offset, positions),
                apply_rope(k, cos, sin, offset, positions), v)

    def _finish(self, x, q, k, v, mask, precomputed=None):
        batch, length, _ = x.shape
        if precomputed is None:
            precomputed = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        attended = precomputed.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        x = x + self.o(attended)
        gate, up = mx.split(self.mlp_in(self.n2(x)), 2, axis=-1)
        return x + self.down(nn.silu(gate) * up)

    def __call__(self, x, mask, capture=None, valid=None, positions=None):
        q, k, v = self._qkv(x)
        precomputed = None
        if self.retention is not None:
            if capture is not None:
                raise ValueError("attention capture is not implemented for scored eviction")
            gates, _, _ = self.retention.sequence(self.n1(x), v, positions, valid)
            precomputed = gated_attention(q, k, v, gates, self.head_dim ** -0.5, mask)
        if capture is not None:
            # Inspection path: explicit softmax attention so the probability
            # map for every head can be captured. Grouped-query keys/values
            # are expanded to the full head count first.
            group = self.heads // self.kv_heads
            keys = mx.repeat(k, group, axis=1) if group > 1 else k
            values = mx.repeat(v, group, axis=1) if group > 1 else v
            scores = (q.astype(mx.float32)
                      @ keys.astype(mx.float32).transpose(0, 1, 3, 2)
                      * self.head_dim ** -0.5)
            probs = mx.softmax(scores + mask.astype(mx.float32), axis=-1)
            capture.append(probs.astype(v.dtype))
            precomputed = probs.astype(v.dtype) @ values
        return self._finish(x, q, k, v, mask, precomputed)



class PrefixLM(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int,
                 d_model: int, layers: int, heads: int,
                 kv_heads: int, max_length: int, dtype=mx.bfloat16,
                 rope_theta: float = 500_000.0, carrier_vocab: int = 0,
                 scored_eviction=None):
        super().__init__()
        if carrier_vocab < 0:
            raise ValueError("carrier_vocab must be non-negative")
        self.max_length = max_length
        self.attention_mode = "causal"
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        vocab_size = (source_vocab_size + target_vocab_size - 4
                      + carrier_vocab)
        hidden = 256 * ((int(8 * d_model / 3) + 255) // 256)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * 0.02).astype(dtype)
        self.blocks = [
            Block(d_model, heads, kv_heads, hidden, layers, max_length,
                  rope_theta, dtype, scored_eviction)
            for _ in range(layers)
        ]
        self.norm = RMSNorm(d_model)

    def _target_weight(self):
        target_end = self.source_vocab_size + self.target_vocab_size - 4
        return mx.concatenate(
            [self.embed.weight[:4],
             self.embed.weight[self.source_vocab_size:target_end]], axis=0)

    def _output(self, h):
        return h @ self._target_weight().T

    def hidden_from_embeddings(self, embeddings, valid, prefix_lengths,
                               transport_spans=None, sliding_window=None,
                               capture=None):
        """Run the transformer on caller-supplied input embeddings.

        Pass a list as ``capture`` to collect per-layer attention
        probabilities with shape ``[batch, heads, query, key]``.
        """
        if embeddings.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {embeddings.shape[1]} exceeds {self.max_length}")
        if transport_spans is not None and transport_spans.shape[1] > 0:
            mask = same_pass_transport_mask(
                valid, transport_spans, embeddings.dtype, sliding_window)
        else:
            mask = causal_mask(valid, embeddings.dtype, sliding_window)
        h = embeddings
        for block in self.blocks:
            h = block(h, mask, capture=capture, valid=valid,
                      positions=mx.broadcast_to(mx.arange(h.shape[1]), valid.shape))
        return self.norm(h)

    def hidden(self, tokens, valid, prefix_lengths, transport_spans=None,
               sliding_window=None):
        """Return normalized hidden states without applying the output head."""
        return self.hidden_from_embeddings(
            self.embed(tokens), valid, prefix_lengths, transport_spans,
            sliding_window)

    def attention_maps(self, tokens, valid, prefix_lengths,
                       transport_spans=None, sliding_window=None):
        """Return normalized hidden states and per-layer attention maps."""
        capture: list = []
        h = self.hidden_from_embeddings(
            self.embed(tokens), valid, prefix_lengths, transport_spans,
            sliding_window, capture=capture)
        return h, capture

    def from_embeddings(self, embeddings, valid, prefix_lengths,
                        output_positions=None, transport_spans=None,
                        sliding_window=None):
        """Compute target logits from externally constructed embeddings."""
        h = self.hidden_from_embeddings(embeddings, valid, prefix_lengths,
                                        transport_spans, sliding_window)
        if output_positions is not None:
            batch = mx.arange(h.shape[0])[:, None]
            h = h[batch, output_positions]
        return self._output(h)

    def __call__(self, tokens, valid, prefix_lengths, output_positions=None,
                 transport_spans=None, sliding_window=None):
        return self.from_embeddings(
            self.embed(tokens), valid, prefix_lengths, output_positions,
            transport_spans, sliding_window)

    def prefill(self, tokens, valid, prefix_lengths, sliding_window=None,
                transport_spans=None, positions=None):
        """Compute KVs while preserving per-row positions, padding, and eviction state."""
        if positions is None:
            positions = mx.broadcast_to(mx.arange(tokens.shape[1]), tokens.shape)
        if int(mx.max(positions).item()) >= self.max_length:
            raise ValueError("position exceeds model maximum length")
        h = self.embed(tokens)
        mask = additive_mask(visibility(positions, positions, valid,
                                        transport_spans, sliding_window), h.dtype)
        layers, alive, retention_scores = [], [], []
        for block in self.blocks:
            q, k, v = block._qkv(h, positions=positions)
            attended = None
            live = valid
            if block.retention is not None:
                gates, live, stored_scores = block.retention.sequence(
                    block.n1(h), v, positions, valid)
                attended = gated_attention(q, k, v, gates, block.head_dim ** -0.5, mask)
                retention_scores.append(stored_scores)
            else:
                retention_scores.append(None)
            h = block._finish(h, q, k, v, mask, attended)
            layers.append((k, v))
            alive.append(live)
        last = mx.max(mx.where(valid, mx.arange(tokens.shape[1])[None], -1), axis=1)
        if bool(mx.any(last < 0).item()):
            raise ValueError("every prefill row needs at least one valid token")
        next_positions = mx.max(mx.where(valid, positions, -1), axis=1) + 1
        logits = self._output(self.norm(h[mx.arange(tokens.shape[0]), last][:, None]))
        return logits, KVState(layers, tokens, positions, valid, next_positions,
                               transport_spans, alive, retention_scores)

    def decode(self, tokens, cache, sliding_window=None, forced_alive=None):
        """Advance one token, with absolute positions independent of stored slots."""
        if tokens.shape[1] != 1:
            raise ValueError("cached decoding expects exactly one token")
        if not isinstance(cache, KVState):
            raise TypeError("decode requires the KVState returned by prefill")
        position = cache.next_positions[:, None]
        if int(mx.max(position).item()) >= self.max_length:
            raise ValueError("position exceeds model maximum length")
        positions = mx.concatenate([cache.positions, position], axis=1)
        valid = mx.concatenate([cache.valid, mx.ones(tokens.shape, dtype=mx.bool_)], axis=1)
        h = self.embed(tokens)
        mask = additive_mask(visibility(position, positions, valid,
                                        cache.spans, sliding_window), h.dtype)
        layers, alive, retention_scores = [], [], []
        for index, (block, (old_k, old_v)) in enumerate(zip(self.blocks, cache, strict=True)):
            q, k, v = block._qkv(h, positions=position)
            k, v = mx.concatenate([old_k, k], axis=2), mx.concatenate([old_v, v], axis=2)
            attended = None
            live = valid
            if block.retention is not None:
                eligible = mx.concatenate([cache.alive[index], mx.ones(tokens.shape, dtype=mx.bool_)], axis=1)
                stored_scores = mx.concatenate(
                    [cache.retention_scores[index],
                     mx.zeros(tokens.shape, dtype=mx.float32)], axis=1)
                newly_old = eligible & (positions == position - block.retention.window)
                assigned = block.retention.assign(block.n1(h), v, newly_old)
                stored_scores = mx.where(newly_old, assigned, stored_scores)
                if forced_alive is None:
                    live, gate = block.retention.select(
                        stored_scores, eligible, positions, position[:, 0])
                else:
                    live = forced_alive[index]
                    gate = live.astype(mx.float32)
                attended = gated_attention(q, k, v, gate[:, None], block.head_dim ** -0.5, mask)
            h = block._finish(h, q, k, v, mask, attended)
            layers.append((k, v))
            alive.append(live)
            retention_scores.append(stored_scores if block.retention is not None else None)
        next_cache = KVState(layers, mx.concatenate([cache.tokens, tokens], axis=1),
                             positions, valid, position[:, 0] + 1, cache.spans,
                             alive, retention_scores=retention_scores)
        return self._output(self.norm(h)), next_cache

    def restart(self, cache, sliding_window=None, forced_keep=None):
        """Rebuild surviving KVs before the next query, without changing positions.

        Freeze each layer's surviving support. The scorer is not rerun during
        reconstruction. Historical causal/SWA/transport restrictions still apply;
        discarded tokens cannot contribute as keys at any reconstruction step.
        The original token identities stay in dense storage solely for replay.
        """
        survivors = visibility(cache.next_positions[:, None], cache.positions,
                               cache.valid, cache.spans, sliding_window)[:, 0]
        h = self.embed(cache.tokens)
        layers = []
        for i, block in enumerate(self.blocks):
            keep = survivors & (cache.alive[i] if forced_keep is None else forced_keep[i])
            mask = additive_mask(visibility(cache.positions, cache.positions, keep,
                                            cache.spans, sliding_window), h.dtype)
            # Empty historical rows are zero attention, never a uniform read of
            # masked keys. This matters when every early key has been evicted.
            q, k, v = block._qkv(h, positions=cache.positions)
            allowed = mask == 0
            group = block.heads // block.kv_heads
            keys = mx.repeat(k, group, axis=1)
            values = mx.repeat(v, group, axis=1)
            scores = q.astype(mx.float32) @ keys.astype(mx.float32).transpose(0, 1, 3, 2)
            probs = mx.softmax(scores * block.head_dim ** -0.5 + mask.astype(mx.float32), axis=-1)
            probs = mx.where(allowed, probs, 0)
            h = block._finish(h, q, k, v, mask, probs.astype(v.dtype) @ values)
            layers.append((k, v))
        return KVState(layers, cache.tokens, cache.positions, cache.valid,
                       cache.next_positions, cache.spans, cache.alive,
                       retention_scores=cache.retention_scores)


def model_from_config(source_vocab_size: int, target_vocab_size: int,
                      cfg: dict, dtype) -> PrefixLM:
    carrier_vocab = int(cfg.get("carrier_vocab", 0))
    max_length = (cfg["max_source_tokens"] + cfg["max_target_tokens"] + 2
                  + max_carrier_tokens(
                      cfg.get("transport_policy"), cfg["max_source_tokens"]))
    return PrefixLM(
        source_vocab_size, target_vocab_size, cfg["d_model"], cfg["layers"],
        cfg["heads"], cfg["kv_heads"], max_length,
        dtype=dtype, rope_theta=cfg.get("rope_theta", 500_000.0),
        carrier_vocab=carrier_vocab,
        scored_eviction=cfg.get("scored_eviction"))
