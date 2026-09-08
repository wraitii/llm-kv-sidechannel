import pytest

from llmpr_torch.pg19_data import fit_pair, parse_lengths, select_books, stable_book_id
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
def test_fit_pair_respects_complete_context_budget(builder):
    tokenizer = CharacterTokenizer()
    pair = fit_pair(tokenizer, TEXT, background_id="book", seed=7,
                    context_length=1024, builder=builder)
    encoded = [tokenize_episode(tokenizer, row, max_length=1024) for row in pair]
    validate_counterfactual_pair(*encoded)
    assert all(row.context_length == 1024 for row in pair)
    assert max(len(row.input_ids) for row in encoded) <= 1024
    assert max(len(row.input_ids) for row in encoded) > 900


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
