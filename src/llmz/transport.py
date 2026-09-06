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
class RandomContiguousPolicy:
    """Select non-overlapping random ``k``-token spans from a source prefix.

    The final token of each span is deliberately retained: it can read the
    preceding span, while later tokens cannot.  This keeps the only
    differentiable route to the future through an ordinary token's hidden
    state and K/V entries.
    """

    span_tokens: int = 16
    spans_per_example: int = 3
    min_source_tokens: int = 32
    min_gap_tokens: int = 4
    alignment: str = "token"

    def __post_init__(self) -> None:
        if self.alignment not in {"token", "move"}:
            raise ValueError("alignment must be token or move")
        if self.span_tokens < 2:
            raise ValueError("span_tokens must include a carrier and be at least 2")
        if self.spans_per_example < 1 or self.min_source_tokens < 1:
            raise ValueError("span and minimum-source counts must be positive")
        if self.min_gap_tokens < 0:
            raise ValueError("min_gap_tokens must be non-negative")

    def sample(self, source_lengths: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """Return padded [batch, span, (start, end, visible_until)] coordinates."""
        result = np.full((len(source_lengths), self.spans_per_example, 3), -1,
                         dtype=np.int32)
        for row, source_length in enumerate(source_lengths.tolist()):
            if source_length < max(self.min_source_tokens, self.span_tokens):
                continue
            chosen: list[tuple[int, int]] = []
            # Rejection sampling avoids overlap and leaves a small unmasked
            # region between relays, while preserving random placement.
            for _ in range(self.spans_per_example * 32):
                if len(chosen) == self.spans_per_example:
                    break
                start = int(rng.integers(0, source_length - self.span_tokens + 1))
                carrier = start + self.span_tokens - 1
                if any(not (carrier + self.min_gap_tokens < old_start or
                            start > old_carrier + self.min_gap_tokens)
                       for old_start, old_carrier in chosen):
                    continue
                chosen.append((start, carrier))
            for index, (start, carrier) in enumerate(sorted(chosen)):
                result[row, index] = (start, carrier, carrier)
        return result


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
    alignment: str = "token"

    @staticmethod
    def _range(spec: int | list[int], name: str, minimum: int) -> tuple[int, int]:
        return uniform_range(spec, name, minimum)

    def __post_init__(self) -> None:
        if self.alignment not in {"token", "move"}:
            raise ValueError("alignment must be token or move")
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
    h_lo, _ = uniform_range(policy_config.get("hidden_moves", policy_config.get("hidden_tokens", 4)),
                            "hidden_tokens", 1)
    g_lo, _ = uniform_range(policy_config.get("gap_moves", policy_config.get("gap_tokens", 0)),
                            "gap_tokens", 0)
    _, c_hi = uniform_range(policy_config.get("carrier_tokens", 2),
                            "carrier_tokens", 1)
    periods = max_source_tokens // (h_lo + g_lo) + 1
    return periods * c_hi


def policy_from_config(config: dict | None) -> (
        RandomContiguousPolicy | RecursiveBlockPolicy | None):
    """Resolve a policy config; future policy kinds extend this single seam."""
    if not config or config.get("kind", "none") == "none":
        return None
    kind = config.get("kind")
    alignment = config.get("alignment", "token")
    if alignment not in {"token", "move"}:
        raise ValueError("alignment must be token or move")
    # Legacy configs keep token semantics. New chess configs explicitly opt in
    # and name move-count fields so changing units cannot go unnoticed.
    config = dict(config)
    for stem in ("hidden", "survivor", "gap"):
        move_key, token_key = f"{stem}_moves", f"{stem}_tokens"
        if move_key in config:
            if alignment != "move" or token_key in config:
                raise ValueError(f"{move_key} requires move alignment and no {token_key}")
            config[token_key] = config[move_key]
    if kind == "random_contiguous":
        return RandomContiguousPolicy(
            span_tokens=config.get("span_tokens", 16),
            spans_per_example=config.get("spans_per_example", 3),
            min_source_tokens=config.get("min_source_tokens", 32),
            min_gap_tokens=config.get("min_gap_tokens", 4), alignment=alignment)
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
    alignment: str = "token"

    def __post_init__(self) -> None:
        if self.alignment not in {"token", "move"}:
            raise ValueError("alignment must be token or move")
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
