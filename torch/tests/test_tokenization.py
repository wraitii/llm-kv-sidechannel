from llmpr_torch.state_data import make_counterfactual_pair
import pytest

from llmpr_torch.tokenization import (
    TokenizedEpisode, tokenize_episode, validate_counterfactual_pair,
)


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def test_answer_only_labels_and_support_spans():
    background = "The guests continued their lengthy conversation by the fire. " * 40
    encoded = None
    for seed in range(64):
        pair = make_counterfactual_pair(background, background_id="book", seed=seed)
        candidate = [tokenize_episode(CharacterTokenizer(), row, max_length=10000)
                     for row in pair]
        try:
            validate_counterfactual_pair(*candidate)
        except ValueError:
            continue
        encoded = candidate
        break
    assert encoded is not None
    for row in encoded:
        assert all(label == -100 for label in row.labels[:row.prompt_length])
        assert row.labels[-1] == CharacterTokenizer.eos_token_id
        assert row.support_token_spans[-1][1] < row.prompt_length


def test_counterfactual_answers_must_have_matching_token_lengths():
    shared = {
        "input_ids": (1, 2, 3, 4),
        "labels": (-100, -100, 3, 4),
        "prompt_length": 2,
        "support_token_spans": ((0, 0),),
    }
    with pytest.raises(ValueError, match="answers have different token lengths"):
        validate_counterfactual_pair(
            TokenizedEpisode(**shared, answer_length=2),
            TokenizedEpisode(**shared, answer_length=3),
        )
