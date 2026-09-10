import pytest

from llmpr_torch.pg19_data import (
    LEVELS, fit_pair, insertion_positions, parse_lengths, select_books, stable_book_id,
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
