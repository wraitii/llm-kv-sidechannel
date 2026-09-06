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
    source = tok.encode_source("e2e4 e7e5 g1f3 b8c6")
    row = [1, *source, 2]
    batch = {"x": np.array([row] * 8),
             "valid": np.ones((8, len(row)), bool),
             "source_tokens": np.full(8, len(source)),
             "prefix_lengths": np.full(8, len(row)),
             "output_positions": np.full((8, 1), len(row) - 1)}
    rng = np.random.default_rng(10)
    observed = set()
    for _ in range(20):
        expanded, full = training_batch(batch, policy, tok, rng, 0.5)
        assert np.all(full == full[0])
        observed.add(bool(full[0]))
        assert (expanded["x"].shape[1] == len(row)) == bool(full[0])
    assert observed == {True, False}
