"""Qwen serialization with explicit prompt/answer boundaries."""
from __future__ import annotations

from dataclasses import dataclass

from .state_data import StateEpisode


@dataclass(frozen=True)
class TokenizedEpisode:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    answer_length: int
    support_token_spans: tuple[tuple[int, int], ...]


def tokenize_episode(tokenizer, episode: StateEpisode, *, max_length: int) -> TokenizedEpisode:
    prompt = tokenizer(
        episode.prompt, add_special_tokens=False, return_offsets_mapping=True)
    answer_ids = tokenizer(episode.answer, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")
    prompt_ids = prompt["input_ids"]
    input_ids = [*prompt_ids, *answer_ids, eos_id]
    if len(input_ids) > max_length:
        raise ValueError(f"episode has {len(input_ids)} tokens, limit is {max_length}")
    labels = [-100] * len(prompt_ids) + [*answer_ids, eos_id]
    spans = []
    for event in episode.events:
        overlapping = [i for i, (start, end) in enumerate(prompt["offset_mapping"])
                       if end > event.char_start and start < event.char_end]
        if not overlapping:
            raise ValueError("event did not overlap any prompt token")
        spans.append((overlapping[0], overlapping[-1]))
    return TokenizedEpisode(tuple(input_ids), tuple(labels), len(prompt_ids),
                            len(answer_ids) + 1, tuple(spans))


def validate_counterfactual_pair(a: TokenizedEpisode, b: TokenizedEpisode) -> None:
    """Ensure the causal comparison preserves positions and a shared suffix."""
    if a.prompt_length != b.prompt_length:
        raise ValueError("counterfactual prompts have different token lengths")
    divergence = [i for i, (x, y) in enumerate(zip(a.input_ids[:a.prompt_length],
                                                   b.input_ids[:b.prompt_length], strict=True))
                  if x != y]
    if not divergence:
        raise ValueError("counterfactual prompts are token-identical")
    suffix_start = max(a.support_token_spans[-1][1], b.support_token_spans[-1][1]) + 1
    if a.input_ids[suffix_start:a.prompt_length] != b.input_ids[suffix_start:b.prompt_length]:
        raise ValueError("counterfactual variants do not share the post-update suffix")
