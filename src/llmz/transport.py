"""Pluggable same-pass context-eviction policies.

Policies return source-coordinate spans ``[start, end, visible_until]``.
Keys in ``[start, end)`` are evicted for every query strictly after
``visible_until``; queries up to and including ``visible_until`` still read
them.  The single-carrier scheme is the special case ``visible_until = end``: the last token of a span reads it, later tokens
see only the carrier's K/V.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def uniform_range(spec: int | list[int], name: str, minimum: int) -> tuple[int, int]:
    """Validate an int or ``[low, high]`` spec into a sampling range."""
    if isinstance(spec, list):
        if len(spec) != 2 or spec[0] < minimum or spec[1] < spec[0]:
            raise ValueError(
                f"{name} range must be [low, high] with low >= {minimum} "
                f"and high >= low, got {spec!r}")
        return int(spec[0]), int(spec[1])
    if spec < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {spec!r}")
    return int(spec), int(spec)


@dataclass(frozen=True)
class RecursiveBlockPolicy:
    """Deterministic block eviction with recursive survivor summarization.

    Level 1 tiles the source into periods of hidden tokens followed by
    survivor tokens (and an optional visible gap).  Each hidden block is
    readable only by its own tokens (causally) and by the survivor tokens
    that immediately follow it; every later query must rely on the
    survivors, which stay visible downstream.  ``hidden_tokens``,
    ``survivor_tokens``, and ``gap_tokens`` are either fixed integers or
    ``[low, high]`` ranges sampled uniformly per period, so carrier
    identity cannot be learned positionally.

    Level 2 applies the same rule one step up the survivor stream: each
    run of ``group_size`` consecutive survivors is hidden from every query
    after the next ``survivor_tokens`` survivors (the meta-survivors).
    This repeats for ``depth - 1`` further levels, each level operating on
    the previous level's survivor stream, so information must climb an
    explicit hierarchy of carriers.  A trailing incomplete group at any
    level stays fully visible.  Sources too short for even one minimal
    period are left untouched; longer ones always tile and deeper levels
    degrade gracefully.
    """

    hidden_tokens: int | list[int] = 4
    survivor_tokens: int | list[int] = 2
    gap_tokens: int | list[int] = 0
    group_size: int = 8
    depth: int = 2
    alignment: str = "move"

    @staticmethod
    def _range(spec: int | list[int], name: str, minimum: int) -> tuple[int, int]:
        return uniform_range(spec, name, minimum)

    def __post_init__(self) -> None:
        if self.alignment != "move":
            raise ValueError("transport policies require move alignment")
        self._range(self.hidden_tokens, "hidden_tokens", 1)
        self._range(self.survivor_tokens, "survivor_tokens", 1)
        self._range(self.gap_tokens, "gap_tokens", 0)
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.depth < 1:
            raise ValueError("depth must be at least 1")

    def sample(self, source_lengths: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """Return padded [batch, span, (start, end, visible_until)] coordinates."""
        rows = [self._row_spans(length, rng) for length in source_lengths.tolist()]
        width = max((len(spans) for spans in rows), default=0)
        result = np.full((len(source_lengths), width, 3), -1, dtype=np.int32)
        for row, spans in enumerate(rows):
            for index, span in enumerate(spans):
                result[row, index] = span
        return result

    def _row_spans(self, length: int,
                   rng: np.random.Generator) -> list[tuple[int, int, int]]:
        h_lo, h_hi = self._range(self.hidden_tokens, "hidden_tokens", 1)
        s_lo, s_hi = self._range(self.survivor_tokens, "survivor_tokens", 1)
        g_lo, g_hi = self._range(self.gap_tokens, "gap_tokens", 0)
        if length < h_lo + s_lo:
            return []
        spans: list[tuple[int, int, int]] = []
        # Level 1: tile randomized [hidden block | survivors | gap] periods.
        survivors: list[int] = []
        position = 0
        while length - position >= h_lo + s_lo:
            hidden = int(rng.integers(h_lo, h_hi + 1))
            count = int(rng.integers(s_lo, s_hi + 1))
            count = min(count, length - position - h_lo)
            hidden = min(hidden, length - position - count)
            start, end = position, position + hidden
            spans.append((start, end, end + count - 1))
            survivors.extend(range(end, end + count))
            gap = int(rng.integers(g_lo, g_hi + 1))
            position = end + count + gap
        spans.extend(recursive_survivor_spans(survivors, self.group_size,
                                               self.depth, (s_lo, s_hi), rng))
        return spans


def recursive_survivor_spans(survivors, group_size, depth, count_range, rng):
    """Shared hierarchy geometry: group_size always counts individual tokens."""
    lo, hi = count_range
    spans = []
    for _ in range(depth - 1):
        if len(survivors) < group_size + lo:
            break
        next_survivors = []
        for edge in range(group_size, len(survivors), group_size):
            meta_edge = edge + int(rng.integers(lo, hi + 1))
            if meta_edge > len(survivors):
                break
            spans.append((survivors[edge-group_size], survivors[edge-1] + 1,
                          survivors[meta_edge-1]))
            next_survivors.extend(survivors[edge:meta_edge])
        survivors = next_survivors
    return spans


def max_carrier_tokens(policy_config: dict | None,
                       max_source_tokens: int) -> int:
    """Upper bound on inserted carriers for any source up to the max length."""
    if not policy_config or policy_config.get("kind") != "recursive_carriers":
        return 0
    h_lo, _ = uniform_range(policy_config.get("hidden_moves", 4), "hidden_moves", 1)
    g_lo, _ = uniform_range(policy_config.get("gap_moves", 0), "gap_moves", 0)
    _, c_hi = uniform_range(policy_config.get("carrier_tokens", 2),
                            "carrier_tokens", 1)
    periods = max_source_tokens // (h_lo + g_lo) + 1
    return periods * c_hi


@dataclass(frozen=True)
class FixedSparsePolicy:
    """Keep a fixed recent tail plus content-independent older memory tokens.

    All source tokens retain ordinary causal access while their KVs are built.
    After the source boundary, only ``recent_tokens`` contiguous tokens and
    ``memory_tokens`` older tokens remain visible. Older memory is placed either
    uniformly across history or logarithmically, with logarithmic placement
    denser near the recent tail. This is an endpoint-memory policy rather than a
    streaming eviction policy.
    """

    recent_tokens: int = 16
    memory_tokens: int = 16
    strategy: str = "uniform"
    alignment: str = "token"

    def __post_init__(self) -> None:
        if self.alignment != "token":
            raise ValueError("fixed_sparse requires alignment='token'")
        if self.recent_tokens < 1:
            raise ValueError("recent_tokens must be positive")
        if self.memory_tokens < 1:
            raise ValueError("memory_tokens must be positive")
        if self.strategy not in {"uniform", "log"}:
            raise ValueError("fixed_sparse strategy must be 'uniform' or 'log'")

    def _memory_indices(self, old_count: int) -> np.ndarray:
        count = min(self.memory_tokens, old_count)
        if count == old_count:
            return np.arange(old_count, dtype=np.int32)
        if self.strategy == "uniform":
            # Select bin centers for equal coverage without endpoint bias.
            return np.floor(
                (np.arange(count) + 0.5) * old_count / count).astype(np.int32)
        # Logarithmic backward distances give broad historical coverage while
        # placing more slots near the recent tail. Constrained rounding keeps
        # the requested number of indices distinct even for short histories.
        targets = old_count - np.geomspace(old_count, 1, count)
        selected = np.empty(count, dtype=np.int32)
        previous = -1
        for index, target in enumerate(targets):
            lower = previous + 1
            upper = old_count - (count - index)
            selected[index] = int(np.clip(np.rint(target), lower, upper))
            previous = int(selected[index])
        return selected

    def sample(self, source_lengths: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """Return spans hidden after the final source token in each row."""
        del rng  # The policy is deliberately deterministic and content-free.
        rows = []
        for length in source_lengths.tolist():
            recent_start = max(0, length - self.recent_tokens)
            kept = set(self._memory_indices(recent_start).tolist())
            kept.update(range(recent_start, length))
            spans = []
            start = None
            for position in range(length + 1):
                hidden = position < length and position not in kept
                if hidden and start is None:
                    start = position
                elif not hidden and start is not None:
                    spans.append((start, position, length - 1))
                    start = None
            rows.append(spans)
        width = max((len(spans) for spans in rows), default=0)
        result = np.full((len(rows), width, 3), -1, dtype=np.int32)
        for row, spans in enumerate(rows):
            if spans:
                result[row, :len(spans)] = spans
        return result


def policy_from_config(config: dict | None) -> RecursiveBlockPolicy | FixedSparsePolicy | None:
    """Resolve one of the supported transport policies."""
    if not config or config.get("kind", "none") == "none":
        return None
    kind = config.get("kind")
    alignment = config.get("alignment")
    if kind == "fixed_sparse":
        return FixedSparsePolicy(
            recent_tokens=config.get("recent_tokens", 16),
            memory_tokens=config.get("memory_tokens", 16),
            strategy=config.get("strategy", "uniform"),
            alignment=alignment)
    if alignment != "move":
        raise ValueError("transport policies require alignment='move'")
    config = dict(config)
    for stem in ("hidden", "survivor", "gap"):
        move_key, token_key = f"{stem}_moves", f"{stem}_tokens"
        if token_key in config:
            raise ValueError(f"{token_key} is unsupported; use {move_key}")
        if move_key in config:
            config[token_key] = config[move_key]
    if kind == "recursive_carriers":
        return RecursiveCarrierPolicy(
            hidden_tokens=config.get("hidden_tokens", 4),
            carrier_tokens=config.get("carrier_tokens", 2),
            gap_tokens=config.get("gap_tokens", 0),
            group_size=config.get("group_size", 8),
            depth=config.get("depth", 2), alignment=alignment)
    if kind == "recursive_blocks":
        return RecursiveBlockPolicy(
            hidden_tokens=config.get("hidden_tokens", 4),
            survivor_tokens=config.get("survivor_tokens", 2),
            gap_tokens=config.get("gap_tokens", 0),
            group_size=config.get("group_size", 8),
            depth=config.get("depth", 2), alignment=alignment)
    raise ValueError(f"unknown transport policy: {kind!r}")


@dataclass(frozen=True)
class CarrierPlan:
    """Per-row carrier insertion plan in source coordinates.

    ``blocks[row]`` lists ``(hidden_start, hidden_end, carrier_count)``
    triples in 0-based source coordinates (``end`` exclusive, i.e. the
    last hidden source token is ``end - 1``).
    """

    blocks: list[list[tuple[int, int, int]]]


@dataclass(frozen=True)
class RecursiveCarrierPolicy:
    """Recursive block eviction with explicit memento carrier tokens.

    Identical hiding geometry to :class:`RecursiveBlockPolicy`, except the
    survivor role is played by dedicated carrier tokens inserted into the
    sequence at runtime: per period, ``carrier_tokens`` carriers follow the
    hidden block, read it, and stay visible downstream.  Deeper levels hide
    runs of ``group_size`` carrier tokens (plus everything between
    them) behind the next group's carriers.  This policy produces a
    :class:`CarrierPlan`; the actual sequence expansion happens in
    :mod:`llmz.carriers`.
    """

    hidden_tokens: int | list[int] = 4
    carrier_tokens: int | list[int] = 2
    gap_tokens: int | list[int] = 0
    group_size: int = 8
    depth: int = 2
    alignment: str = "move"

    def __post_init__(self) -> None:
        if self.alignment != "move":
            raise ValueError("transport policies require move alignment")
        uniform_range(self.hidden_tokens, "hidden_tokens", 1)
        uniform_range(self.carrier_tokens, "carrier_tokens", 1)
        uniform_range(self.gap_tokens, "gap_tokens", 0)
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.depth < 1:
            raise ValueError("depth must be at least 1")

    def plan(self, source_lengths: np.ndarray,
             rng: np.random.Generator) -> CarrierPlan:
        rows = [self._row_blocks(length, rng)
                for length in source_lengths.tolist()]
        return CarrierPlan(rows)

    def _row_blocks(self, length: int,
                    rng: np.random.Generator) -> list[tuple[int, int, int]]:
        h_lo, h_hi = uniform_range(self.hidden_tokens, "hidden_tokens", 1)
        c_lo, c_hi = uniform_range(self.carrier_tokens, "carrier_tokens", 1)
        g_lo, g_hi = uniform_range(self.gap_tokens, "gap_tokens", 0)
        blocks: list[tuple[int, int, int]] = []
        position = 0
        while length - position >= h_lo:
            hidden = min(int(rng.integers(h_lo, h_hi + 1)), length - position)
            count = int(rng.integers(c_lo, c_hi + 1))
            blocks.append((position, position + hidden, count))
            position += hidden + int(rng.integers(g_lo, g_hi + 1))
        return blocks
