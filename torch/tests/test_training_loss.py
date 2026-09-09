import math
import json

import pytest
import torch

from llmpr_torch.training import mixed_causal_loss
from llmpr_torch.lm_evaluation import clean_windows


def test_mixed_causal_loss_averages_prompt_and_answer_separately():
    logits = torch.zeros((1, 5, 7), requires_grad=True)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, -100, 3, 4]])

    combined, answer, prompt = mixed_causal_loss(
        logits, input_ids, labels, [3], prompt_loss_weight=0.25)

    expected = math.log(7)
    assert answer.item() == pytest.approx(expected)
    assert prompt.item() == pytest.approx(expected)
    assert combined.item() == pytest.approx(1.25 * expected)
    combined.backward()
    assert logits.grad is not None


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(word) for word in text.split()]}


def test_clean_windows_are_deterministic(tmp_path):
    path = tmp_path / "raw.jsonl"
    rows = [{"id": str(i), "text": (f"book{i} word " * 1000)} for i in range(3)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = list(clean_windows(WordTokenizer(), path, 32, 2, 7))
    second = list(clean_windows(WordTokenizer(), path, 32, 2, 7))
    assert first == second
    assert len(first) == 2
    assert all(len(window) == 32 for window in first)
