import json
import math

import pytest
import torch

from llmpr_torch.training import load_config, mixed_causal_loss, prompt_causal_loss
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


def test_mixed_causal_loss_supports_normalized_fraction():
    logits = torch.zeros((1, 5, 7), requires_grad=True)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, -100, 3, 4]])

    combined, answer, prompt = mixed_causal_loss(
        logits, input_ids, labels, [3], prompt_loss_fraction=0.1)

    expected = math.log(7)
    assert answer.item() == pytest.approx(expected)
    assert prompt.item() == pytest.approx(expected)
    assert combined.item() == pytest.approx(expected)


def test_prompt_loss_fraction_config_validation(tmp_path):
    base = {
        "model": "model", "data": "data", "run_dir": "run", "steps": 1,
        "learning_rate": 1e-4, "batch_size": 1, "grad_accum": 1,
        "policy": "memento",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**base, "prompt_loss_fraction": 0.1}))
    assert load_config(path)["prompt_loss_fraction"] == 0.1

    path.write_text(json.dumps({**base, "prompt_loss_fraction": 1.0}))
    with pytest.raises(ValueError, match="less than one"):
        load_config(path)

    path.write_text(json.dumps({
        **base, "prompt_loss_fraction": 0.1, "prompt_loss_weight": 0.1}))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path)


def test_prompt_causal_loss_excludes_answer_tokens():
    logits = torch.zeros((1, 5, 7), requires_grad=True)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])

    loss = prompt_causal_loss(logits, input_ids, [3])

    assert loss.item() == pytest.approx(math.log(7))
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 2:]).item() == 0


def test_full_attention_lm_routing_config_validation(tmp_path):
    base = {
        "model": "model", "data": "data", "run_dir": "run", "steps": 1,
        "learning_rate": 1e-4, "batch_size": 1, "grad_accum": 1,
        "policy": "variable-swa:64-128",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**base, "full_attention_lm_probability": 1.1}))
    with pytest.raises(ValueError, match="between zero and one"):
        load_config(path)

    path.write_text(json.dumps({
        **base, "full_attention_lm_probability": 0.05, "prompt_loss_weight": 0.1}))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path)

    path.write_text(json.dumps({
        **base, "full_attention_lm_probability": 0.1, "task_probability": 0.95}))
    with pytest.raises(ValueError, match="sum to at most one"):
        load_config(path)


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
