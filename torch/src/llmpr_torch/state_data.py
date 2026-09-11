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
class MemorySpan:
    char_start: int
    char_end: int
    memory_index: int
    placement: str
    after_event_index: int | None = None


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
    support_to_answer_tokens: tuple[int, ...] = ()
    memory_spans: tuple[MemorySpan, ...] = ()
    memory_layout: str = "none"
    memory_tokens_per_span: int = 0
    memory_compression_ratio: int | None = None

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
    insertion_char_positions: tuple[int, ...] | None = None,
) -> tuple[StateEpisode, StateEpisode]:
    """Create a compositional ownership chain with an early counterfactual state.

    Later operations name only people, not objects or their owners, so the
    answer cannot be copied from the final event. Exact token-length matching
    is checked after Qwen tokenization.
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
    if insertion_char_positions is None:
        insertion_char_positions = tuple(range(
            len(background) // 8, len(background), max(1, len(background) // 8)))
    cuts = tuple(sorted(set(position for position in insertion_char_positions
                            if 0 < position < len(background))))
    if len(cuts) < 3:
        raise ValueError("state probes require at least three insertion positions")
    other_entity = next(item for item in ENTITIES if item != entity)
    remaining_entities = [item for item in ENTITIES if item not in {entity, other_entity}]
    remaining_owners = [owner for owner in OWNERS if owner not in finals]
    swaps = []
    previous = None
    for _ in cuts[1:]:
        choices = [(a, b) for i, a in enumerate(OWNERS) for b in OWNERS[i + 1:]
                   if (a, b) != previous]
        swap = choices[int(rng.integers(len(choices)))]
        swaps.append(swap)
        previous = swap
    pair_id = _stable_id(background_id, str(seed), entity,
                         "natural" if natural else "structured")
    episodes = []
    for variant, final_owner in zip(("a", "b"), finals, strict=True):
        alternate = finals[1] if final_owner == finals[0] else finals[0]
        ownership = {
            entity: final_owner, other_entity: alternate,
            remaining_entities[0]: remaining_owners[0],
            remaining_entities[1]: remaining_owners[1],
        }
        if natural:
            first = (f"Before continuing, {final_owner} carried the {entity}, {alternate} "
                     f"carried the {other_entity}, {remaining_owners[0]} carried the "
                     f"{remaining_entities[0]}, and {remaining_owners[1]} carried the "
                     f"{remaining_entities[1]}.")
        else:
            first = (f"Registry: {final_owner}={entity}; {alternate}={other_entity}; "
                     f"{remaining_owners[0]}={remaining_entities[0]}; "
                     f"{remaining_owners[1]}={remaining_entities[1]}.")
        inserted = [first]
        state_values = [ownership[entity]]
        for left, right in swaps:
            inserted.append(
                f"Before continuing, {left} and {right} exchanged the objects they were carrying."
                if natural else f"Registry operation: swap all holdings of {left} and {right}.")
            for carried, owner in list(ownership.items()):
                if owner == left:
                    ownership[carried] = right
                elif owner == right:
                    ownership[carried] = left
            state_values.append(ownership[entity])
        chunks: list[str] = []
        events = []
        cursor = 0
        for event_index, (cut, event_text, state_value) in enumerate(
                zip(cuts, inserted, state_values, strict=True)):
            chunks.extend([background[cursor:cut].strip(), "\n\n"])
            event_start = sum(map(len, chunks))
            chunks.extend([event_text, "\n\n"])
            events.append(StateEvent(
                entity if event_index == 0 else "possessions",
                state_value, event_start, event_start + len(event_text), event_index))
            cursor = cut
        chunks.append(background[cursor:].lstrip())
        question = f"\n\nQuestion: Who currently possesses the {entity}?\nAnswer:"
        prompt = "".join(chunks) + question
        episodes.append(StateEpisode(
            example_id=f"{pair_id}-{variant}", pair_id=pair_id, variant=variant,
            prompt=prompt, answer=ownership[entity], events=tuple(events),
            query_entity=entity, background_id=background_id,
            task_type="state", difficulty="natural" if natural else "structured",
        ))
    return episodes[0], episodes[1]


def make_passcode_pair(
    background: str,
    *,
    background_id: str,
    seed: int,
    difficulty: str = "hard",
    insertion_char_positions: tuple[int, ...] | None = None,
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
    if difficulty not in {"easy", "hard"}:
        raise ValueError("passcode difficulty must be 'easy' or 'hard'")
    cut = (insertion_char_positions or (len(background) // 4,))[0]
    prefix, suffix = background[:cut], background[cut:]
    pair_id = _stable_id(background_id, str(seed), "passcode", difficulty)
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
            task_type=f"passcode_{difficulty}", difficulty=difficulty,
        ))
    return episodes[0], episodes[1]


def generate_pairs(
    backgrounds: Iterable[tuple[str, str]], *, seed: int, natural: bool = True
) -> Iterable[StateEpisode]:
    for index, (background_id, text) in enumerate(backgrounds):
        yield from make_counterfactual_pair(
            text, background_id=background_id, seed=seed + index, natural=natural)
