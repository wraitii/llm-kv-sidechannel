import json
import numpy as np
import pytest

from llmz.experiment import record_run, training_batch
from llmz.tokenizer import PairTokenizer, BytesTokenizer
from llmz.transport import RecursiveCarrierPolicy


def test_manifest_is_immutable_on_eval_and_resume(tmp_path):
    config = {"init_checkpoint": "baseline.npz", "steps": 5}
    record_run(tmp_path, config)
    original = (tmp_path / "config.json").read_bytes()
    record_run(tmp_path, {"init_checkpoint": None}, eval_only=True)
    record_run(tmp_path, {"steps": 10}, resume=True)
    assert (tmp_path / "config.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        record_run(tmp_path, config)
    assert len((tmp_path / "invocations.jsonl").read_text().splitlines()) == 2


def test_carrier_full_attention_share_has_only_two_conditions():
    tok = PairTokenizer(BytesTokenizer(), BytesTokenizer(), carrier_vocab=2)
    policy = RecursiveCarrierPolicy(hidden_tokens=2, carrier_tokens=1)
    batch = {"x": np.array([[1, 10, 11, 2]] * 8),
             "valid": np.ones((8, 4), bool), "source_tokens": np.full(8, 2),
             "prefix_lengths": np.full(8, 4), "output_positions": np.full((8, 1), 3)}
    rng = np.random.default_rng(10)
    observed = set()
    for _ in range(20):
        expanded, full = training_batch(batch, policy, tok, rng, 0.5)
        assert np.all(full == full[0])
        observed.add(bool(full[0]))
        assert (expanded["x"].shape[1] == 4) == bool(full[0])
    assert observed == {True, False}
