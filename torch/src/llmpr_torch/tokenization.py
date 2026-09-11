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
    memory_token_spans: tuple[tuple[int, int], ...] = ()


def tokenize_episode(tokenizer, episode: StateEpisode, *, max_length: int) -> TokenizedEpisode:
    prompt = tokenizer(
        episode.prompt, add_special_tokens=False, return_offsets_mapping=True)
    answer_ids = tokenizer(episode.answer, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")
    prompt_ids = prompt["input_ids"]
    def token_spans(char_spans):
        spans = []
        for char_start, char_end in char_spans:
            overlapping = [i for i, (start, end) in enumerate(prompt["offset_mapping"])
                           if end > char_start and start < char_end]
            if not overlapping:
                raise ValueError("annotated span did not overlap any prompt token")
            spans.append((overlapping[0], overlapping[-1]))
        return tuple(spans)

    spans = token_spans((event.char_start, event.char_end) for event in episode.events)
    memory_spans = token_spans(
        (span.char_start, span.char_end) for span in episode.memory_spans)
    for annotation, (start, end) in zip(
            episode.memory_spans, memory_spans, strict=True):
        if annotation.replacement_token_ids:
            if len(annotation.replacement_token_ids) != end - start + 1:
                raise ValueError("memory replacement width does not match its token span")
            prompt_ids[start:end + 1] = annotation.replacement_token_ids
    input_ids = [*prompt_ids, *answer_ids, eos_id]
    if len(input_ids) > max_length:
        raise ValueError(f"episode has {len(input_ids)} tokens, limit is {max_length}")
    labels = [-100] * len(prompt_ids) + [*answer_ids, eos_id]
    return TokenizedEpisode(tuple(input_ids), tuple(labels), len(prompt_ids),
                            len(answer_ids) + 1, spans, memory_spans)


def validate_counterfactual_pair(a: TokenizedEpisode, b: TokenizedEpisode) -> None:
    """Ensure the causal comparison preserves positions and a shared suffix."""
    if a.answer_length != b.answer_length:
        raise ValueError("counterfactual answers have different token lengths")
    if a.prompt_length != b.prompt_length:
        raise ValueError("counterfactual prompts have different token lengths")
    if a.memory_token_spans != b.memory_token_spans:
        raise ValueError("counterfactual memory spans are not token-aligned")
    divergence = [i for i, (x, y) in enumerate(zip(a.input_ids[:a.prompt_length],
                                                   b.input_ids[:b.prompt_length], strict=True))
                  if x != y]
    if not divergence:
        raise ValueError("counterfactual prompts are token-identical")
    suffix_start = max(a.support_token_spans[-1][1], b.support_token_spans[-1][1]) + 1
    memory_positions = {
        position for start, end in a.memory_token_spans
        for position in range(start, end + 1)
    }
    a_suffix = tuple(token for position, token in enumerate(
        a.input_ids[suffix_start:a.prompt_length], start=suffix_start)
        if position not in memory_positions)
    b_suffix = tuple(token for position, token in enumerate(
        b.input_ids[suffix_start:b.prompt_length], start=suffix_start)
        if position not in memory_positions)
    if a_suffix != b_suffix:
        raise ValueError("counterfactual variants do not share the post-update suffix")
