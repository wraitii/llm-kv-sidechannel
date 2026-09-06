"""Translate complete-move policy units into source-token coordinates."""
from __future__ import annotations

import re
import numpy as np

from .transport import CarrierPlan, RecursiveCarrierPolicy


def move_boundaries(source, tokenizer):
    """Return token offsets [0, end(move 1), ...], including following spaces.

    Reject partial moves and tokenizers that merge across move boundaries rather
    than silently changing source tokens or rounding to an unsafe boundary.
    """
    text = tokenizer.decode_source(list(source))
    matches = list(re.finditer(r"\S+\s*", text))
    if text and (not matches or matches[0].start() != 0):
        raise ValueError("move alignment requires canonical UCI history")
    if any(not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", m.group().strip()) for m in matches):
        raise ValueError("move alignment requires complete UCI moves")
    if hasattr(tokenizer.source, "inner"):
        encoded = tokenizer.source.inner.encode(text, add_special_tokens=False)
        if encoded.ids != list(source):
            raise ValueError("source tokens do not round-trip for move alignment")
        ends = {end: i + 1 for i, (_, end) in enumerate(encoded.offsets)}
        # An end shared with a following token's interior is not a boundary.
        boundaries = [0]
        for match in matches:
            edge = match.end()
            if edge not in ends or any(start < edge < end for start, end in encoded.offsets):
                raise ValueError("source tokenizer merges across a move boundary")
            boundaries.append(ends[edge])
    else:
        if tokenizer.encode_source(text) != list(source):
            raise ValueError("source tokens do not round-trip for move alignment")
        # BytesTokenizer on ASCII UCI text; verify rather than assume a custom
        # tokenizer uses one token per character.
        if len(source) != len(text):
            raise ValueError("move alignment needs token offsets for this tokenizer")
        boundaries = [0, *(m.end() for m in matches)]
    if boundaries[-1] != len(source):
        raise ValueError("move boundaries do not cover source")
    return np.array(boundaries, dtype=np.int32)


def sample_transport(policy, sources, tokenizer, rng):
    """Shared policy entry point for training, validation and generation."""
    aligned = policy.alignment == "move"
    edges = ([move_boundaries(source, tokenizer) for source in sources] if aligned
             else [np.arange(len(source) + 1, dtype=np.int32) for source in sources])
    lengths = np.array([len(edge) - 1 for edge in edges])
    if isinstance(policy, RecursiveCarrierPolicy):
        plan = policy.plan(lengths, rng)
        return CarrierPlan([[(int(edge[start]), int(edge[end]), count)
                             for start, end, count in blocks]
                            for edge, blocks in zip(edges, plan.blocks, strict=True)])
    spans = policy.sample(lengths, rng)
    for row, edge in enumerate(edges):
        for i, (start, end, visible) in enumerate(spans[row]):
            if start >= 0:
                spans[row, i] = edge[start], edge[end], edge[visible + 1] - 1
    return spans


def batch_sources(batch):
    return [row[1:1+int(length)].tolist()
            for row, length in zip(batch["x"], batch["source_tokens"], strict=True)]
