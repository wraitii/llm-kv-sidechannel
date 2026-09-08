from llmpr_torch.state_data import make_counterfactual_pair
from llmpr_torch.tokenization import tokenize_episode, validate_counterfactual_pair


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def test_answer_only_labels_and_support_spans():
    background = "The guests continued their lengthy conversation by the fire. " * 40
    pair = make_counterfactual_pair(background, background_id="book", seed=2)
    encoded = [tokenize_episode(CharacterTokenizer(), row, max_length=10000)
               for row in pair]
    validate_counterfactual_pair(*encoded)
    for row in encoded:
        assert all(label == -100 for label in row.labels[:row.prompt_length])
        assert row.labels[-1] == CharacterTokenizer.eos_token_id
        assert row.support_token_spans[-1][1] < row.prompt_length
