"""Deterministic state probes embedded in natural-language backgrounds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

ENTITIES = ("brass compass", "silver locket", "green ledger", "ivory key")
OWNERS = ("Eliza", "Clara", "Martin", "Thomas")


@dataclass(frozen=True)
class StateEvent:
    entity: str
    value: str
    char_start: int
    char_end: int
    event_index: int


@dataclass(frozen=True)
class StateEpisode:
    example_id: str
    pair_id: str
    variant: str
    prompt: str
    answer: str
    events: tuple[StateEvent, ...]
    query_entity: str
    background_id: str
    task_type: str = "state"
    difficulty: str = "natural"
    context_length: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _event_text(entity: str, old: str | None, new: str, natural: bool) -> str:
    if natural:
        if old is None:
            return f"Before continuing, {new} took possession of the {entity}."
        return f"Before continuing, {old} handed the {entity} to {new}."
    if old is None:
        return f"Registry update: the {entity} is assigned to {new}."
    return f"Registry update: the {entity} is transferred from {old} to {new}."


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:20]


def make_counterfactual_pair(
    background: str,
    *,
    background_id: str,
    seed: int,
    natural: bool = True,
) -> tuple[StateEpisode, StateEpisode]:
    """Create variants with the same suffix and different early final state.

    The two counterfactual values are distinct owner names. Exact token-length
    matching is tokenizer-dependent and is checked after Qwen tokenization.
    """
    if len(background) < 600:
        raise ValueError("background must contain at least 600 characters")
    rng = np.random.Generator(np.random.PCG64(seed))
    entity = ENTITIES[int(rng.integers(len(ENTITIES)))]
    # Counterfactual variants must preserve downstream character positions.
    # Tokenization performs the stronger model-specific length check later.
    owner_groups = (("Eliza", "Clara"), ("Martin", "Thomas"))
    finals = list(owner_groups[int(rng.integers(len(owner_groups)))])
    if bool(rng.integers(2)):
        finals.reverse()
    initial_choices = [owner for owner in OWNERS if owner not in finals]
    initial = initial_choices[int(rng.integers(len(initial_choices)))]
    cut1, cut2 = len(background) // 5, (len(background) * 2) // 5
    prefix, middle, suffix = background[:cut1], background[cut1:cut2], background[cut2:]
    pair_id = _stable_id(background_id, str(seed), entity,
                         "natural" if natural else "structured")
    episodes = []
    for variant, final_owner in zip(("a", "b"), finals, strict=True):
        first = _event_text(entity, None, initial, natural)
        update = _event_text(entity, initial, final_owner, natural)
        chunks = [prefix.rstrip(), "\n\n", first, "\n\n", middle.strip(), "\n\n"]
        first_start = sum(map(len, chunks[:2]))
        first_end = first_start + len(first)
        update_start = sum(map(len, chunks))
        chunks.extend([update, "\n\n", suffix.lstrip()])
        update_end = update_start + len(update)
        question = f"\n\nQuestion: Who currently possesses the {entity}?\nAnswer:"
        prompt = "".join(chunks) + question
        events = (
            StateEvent(entity, initial, first_start, first_end, 0),
            StateEvent(entity, final_owner, update_start, update_end, 1),
        )
        episodes.append(StateEpisode(
            example_id=f"{pair_id}-{variant}", pair_id=pair_id, variant=variant,
            prompt=prompt, answer=final_owner, events=events,
            query_entity=entity, background_id=background_id,
            task_type="state", difficulty="natural" if natural else "structured",
        ))
    return episodes[0], episodes[1]


def make_passcode_pair(
    background: str,
    *,
    background_id: str,
    seed: int,
) -> tuple[StateEpisode, StateEpisode]:
    """Create a salient key-retrieval pair with a position-matched value."""
    if len(background) < 600:
        raise ValueError("background must contain at least 600 characters")
    rng = np.random.Generator(np.random.PCG64(seed))
    alphabet = np.asarray(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"))
    values: list[str] = []
    while len(values) < 2:
        parts = ["".join(rng.choice(alphabet, size=4)),
                 "".join(rng.choice(alphabet, size=4))]
        value = "-".join(parts)
        if value not in values:
            values.append(value)
    cut = len(background) // 4
    prefix, suffix = background[:cut], background[cut:]
    pair_id = _stable_id(background_id, str(seed), "passcode")
    episodes = []
    for variant, value in zip(("a", "b"), values, strict=True):
        event = f"IMPORTANT PASSCODE: {value}. Keep this passcode for the question later."
        chunks = [prefix.rstrip(), "\n\n", event, "\n\n", suffix.lstrip()]
        event_start = len(chunks[0]) + len(chunks[1])
        prompt = "".join(chunks) + "\n\nQuestion: What was the important passcode?\nAnswer:"
        episodes.append(StateEpisode(
            example_id=f"{pair_id}-{variant}", pair_id=pair_id, variant=variant,
            prompt=prompt, answer=value,
            events=(StateEvent("passcode", value, event_start,
                               event_start + len(event), 0),),
            query_entity="passcode", background_id=background_id,
            task_type="passcode", difficulty="salient",
        ))
    return episodes[0], episodes[1]


def generate_pairs(
    backgrounds: Iterable[tuple[str, str]], *, seed: int, natural: bool = True
) -> Iterable[StateEpisode]:
    for index, (background_id, text) in enumerate(backgrounds):
        yield from make_counterfactual_pair(
            text, background_id=background_id, seed=seed + index, natural=natural)
