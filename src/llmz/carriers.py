"""Runtime insertion of explicit memento carrier tokens into packed batches.

Cached batches are never materialized with carriers; :func:`expand_batch`
interleaves them at batch-construction time, remaps every position index,
and produces transport spans in the expanded coordinate system so the
existing ``same_pass_transport_mask`` machinery works unchanged.

Layout per period (source coordinates)::

    [hidden block | C0 C1 … | gap]

The hidden block is evicted for queries after its carrier group; carriers
stay visible downstream; deeper levels hide runs of carrier groups behind
the next group's carriers (the whole inter-group range, including gaps).
"""
from __future__ import annotations

import numpy as np

from .tokenizer import PAD
from .transport import (CarrierPlan, RecursiveCarrierPolicy, recursive_survivor_spans,
                        uniform_range)


def expand_batch(batch: dict[str, np.ndarray], plan: CarrierPlan,
                 policy: RecursiveCarrierPolicy,
                 carrier_ids: list[int], rng=None,
                 ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Insert carriers into a packed batch.

    Returns ``(batch', spans, remap)`` where ``batch'`` carries expanded
    ``x``/``valid`` and remapped ``prefix_lengths``/``output_positions``,
    ``spans`` is ``[B, n, 3]`` in expanded coordinates, and ``remap`` maps
    original positions to expanded ones (for callers such as generation).
    """
    x, valid = batch["x"], batch["valid"]
    rows, width = x.shape
    totals = [sum(count for _, _, count in row) for row in plan.blocks]
    expanded_width = width + max(totals, default=0)
    if not carrier_ids:
        raise ValueError("carrier policy requires tokenizer carrier IDs")

    out_x = np.full((rows, expanded_width), PAD, dtype=np.int32)
    out_valid = np.zeros((rows, expanded_width), dtype=np.bool_)
    remap = np.zeros((rows, width), dtype=np.int32)
    max_spans = 0
    span_rows: list[list[tuple[int, int, int]]] = []
    for row in range(rows):
        source_length = int(batch["source_tokens"][row])
        blocks = plan.blocks[row]
        carriers_after = {end: (start, end, count) for start, end, count in blocks}
        exp_pos = 0
        carrier_slot = 0
        carrier_groups: list[list[int]] = []
        level1: list[tuple[int, int, int]] = []
        for position in range(width):
            out_x[row, exp_pos] = x[row, position]
            out_valid[row, exp_pos] = valid[row, position]
            remap[row, position] = exp_pos
            exp_pos += 1
            # Carriers are inserted after the block's final source token,
            # whose cached position is ``end`` (source index end - 1 + BOS).
            if 1 <= position <= source_length and position in carriers_after:
                start, end, count = carriers_after[position]
                start_expanded = remap[row, start + 1]
                end_expanded = exp_pos
                group: list[int] = []
                for _ in range(count):
                    out_x[row, exp_pos] = carrier_ids[carrier_slot % len(carrier_ids)]
                    carrier_slot += 1
                    out_valid[row, exp_pos] = True
                    group.append(exp_pos)
                    exp_pos += 1
                carrier_groups.append(group)
                level1.append((start_expanded, end_expanded,
                               end_expanded + count - 1))
        spans_row = list(level1)
        # A deterministic default is useful for direct callers; training and
        # inference provide their own independently seeded layout RNG.
        local_rng = rng if rng is not None else np.random.default_rng(0)
        spans_row.extend(recursive_survivor_spans(
            [position for group in carrier_groups for position in group],
            policy.group_size, policy.depth,
            uniform_range(policy.carrier_tokens, "carrier_tokens", 1), local_rng))
        span_rows.append(spans_row)
        max_spans = max(max_spans, len(spans_row))

    spans = np.full((rows, max_spans, 3), -1, dtype=np.int32)
    for row, spans_row in enumerate(span_rows):
        for index, span in enumerate(spans_row):
            spans[row, index] = span

    out = dict(batch)
    out["x"] = out_x
    out["valid"] = out_valid
    # prefix_lengths are counts (one past the last prefix position), so the
    # expanded count is the remapped last position plus one.
    out["prefix_lengths"] = (remap[np.arange(rows),
                                   batch["prefix_lengths"] - 1] + 1)
    out["output_positions"] = remap[np.arange(rows)[:, None],
                                    batch["output_positions"]]
    return out, spans, remap
