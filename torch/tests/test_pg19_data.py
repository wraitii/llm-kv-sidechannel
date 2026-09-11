import pytest

from llmpr_torch.pg19_data import (
    LEVELS, fit_pair, inject_memory, insertion_positions, parse_lengths,
    remove_memory, select_books, stable_book_id,
)
from llmpr_torch.state_data import make_counterfactual_pair, make_passcode_pair
from llmpr_torch.tokenization import tokenize_episode, validate_counterfactual_pair


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)


TEXT = "A long chapter continued with ordinary prose and quiet conversation. " * 80


@pytest.mark.parametrize("builder", [
    make_passcode_pair,
    lambda text, **kw: make_counterfactual_pair(text, natural=False, **kw),
    lambda text, **kw: make_counterfactual_pair(text, natural=True, **kw),
])
@pytest.mark.parametrize("context_length", [1024, 2048, 3072, 4096])
def test_fit_pair_respects_complete_context_budget(builder, context_length):
    tokenizer = CharacterTokenizer()
    pair = fit_pair(tokenizer, TEXT, background_id="book", seed=7,
                    context_length=context_length, builder=builder)
    encoded = [tokenize_episode(tokenizer, row, max_length=context_length) for row in pair]
    validate_counterfactual_pair(*encoded)
    assert all(row.context_length == context_length for row in pair)
    assert max(len(row.input_ids) for row in encoded) <= context_length
    assert max(len(row.input_ids) for row in encoded) > context_length * 0.9


def test_fit_pair_tokenizes_only_a_bounded_window_of_a_large_book():
    class TrackingTokenizer(CharacterTokenizer):
        def __init__(self):
            self.largest_input = 0

        def __call__(self, text, **kwargs):
            self.largest_input = max(self.largest_input, len(text))
            return super().__call__(text, **kwargs)

    tokenizer = TrackingTokenizer()
    fit_pair(tokenizer, TEXT * 100, background_id="large-book", seed=11,
             context_length=4096, builder=make_passcode_pair)

    assert tokenizer.largest_input <= 4096 * 8


def test_book_selection_and_ids_are_stable():
    rows = [{"text": "short", "url": "skip"},
            {"text": "x" * 2000, "url": "book-1"},
            {"text": "y" * 2000, "url": "book-2"}]
    assert select_books(rows, 1)[0]["url"] == "book-1"
    assert stable_book_id(rows[1]) == stable_book_id(dict(rows[1]))


def test_length_parser():
    assert parse_lengths("1024,4096") == (1024, 4096)
    with pytest.raises(Exception):
        parse_lengths("128")


def test_token_driven_probe_positions_cover_easy_hard_and_state():
    tokenizer = CharacterTokenizer()
    text = "x" * 3800
    easy = insertion_positions(tokenizer, text, level="passcode_easy", seed=7)[0]
    hard = insertion_positions(tokenizer, text, level="passcode_hard", seed=7)[0]
    state = insertion_positions(tokenizer, text, level="natural", seed=7)
    assert 3000 <= easy <= 3480
    assert 380 <= hard <= 950
    assert len(state) >= 6
    assert all(300 <= right - left <= 600 for left, right in zip(state, state[1:]))


@pytest.mark.parametrize("level", ["passcode_easy", "passcode_hard"])
def test_passcode_levels_are_labeled_separately(level):
    pair = fit_pair(CharacterTokenizer(), TEXT, background_id="book", seed=5,
                    context_length=2048, builder=LEVELS[level], level=level)
    assert {row.task_type for row in pair} == {level}
    assert all(row.support_to_answer_tokens for row in pair)


@pytest.mark.parametrize("level", list(LEVELS))
@pytest.mark.parametrize("layout", ["event", "fixed"])
def test_memory_layouts_are_aligned(level, layout):
    pair = fit_pair(
        CharacterTokenizer(), TEXT, background_id="book", seed=17,
        context_length=3072, builder=LEVELS[level], level=level,
        memory_layout=layout, memory_tokens_per_span=4, memory_token="¤",
        memory_compression_ratio=20,
    )
    encoded = [tokenize_episode(CharacterTokenizer(), row, max_length=3072)
               for row in pair]
    validate_counterfactual_pair(*encoded)
    for row, tokenized in zip(pair, encoded, strict=True):
        assert row.memory_layout == layout
        if layout == "event":
            assert len(row.memory_spans) == len(row.events)
        else:
            assert row.memory_compression_ratio == 20
            assert row.memory_spans[-1].char_start > row.events[-1].char_end
        assert len(tokenized.memory_token_spans) == len(row.memory_spans)
        assert all(end - start + 1 == 4 for start, end in tokenized.memory_token_spans)
        assert all(row.prompt[span.char_start:span.char_end] == "¤" * 4
                   for span in row.memory_spans)
        if layout == "event":
            assert all(span.after_event_index == span.memory_index
                       for span in row.memory_spans)
            assert all(event.char_end < span.char_start
                       for event, span in zip(row.events, row.memory_spans, strict=True))
        else:
            assert all(span.after_event_index is None for span in row.memory_spans)


def test_memory_layout_none_preserves_episode():
    episode = make_passcode_pair(TEXT, background_id="book", seed=3)[0]
    assert inject_memory(
        episode, layout="none", tokens_per_span=0, memory_token="¤", seed=9,
    ) == episode


def test_event_and_fixed_layouts_derive_from_one_byte_identical_base_episode():
    base = make_counterfactual_pair(TEXT, background_id="book", seed=23)[0]
    event = inject_memory(
        base, layout="event", tokens_per_span=4, memory_token="¤", seed=31,
    )
    recovered = remove_memory(event)
    assert recovered == base
    fixed = inject_memory(
        recovered, layout="fixed", tokens_per_span=4, memory_token="¤", seed=31,
        tokenizer=CharacterTokenizer(), compression_ratio=20,
    )
    assert remove_memory(fixed) == base
    assert fixed.memory_compression_ratio == 20


@pytest.mark.parametrize("level", list(LEVELS))
def test_paired_fit_requires_alignment_in_both_memory_layouts(level):
    pair = fit_pair(
        CharacterTokenizer(), TEXT, background_id="book", seed=41,
        context_length=3000, builder=LEVELS[level], level=level,
        memory_layout="fixed", memory_tokens_per_span=4, memory_token="¤",
        memory_compression_ratio=20,
        additional_alignment_layouts=("event",),
    )
    base = tuple(remove_memory(row) for row in pair)
    event = tuple(inject_memory(
        row, layout="event", tokens_per_span=4, memory_token="¤", seed=7960,
    ) for row in base)
    validate_counterfactual_pair(*[
        tokenize_episode(CharacterTokenizer(), row, max_length=3000)
        for row in event
    ])
