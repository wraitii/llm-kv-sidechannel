import json
import numpy as np
import pytest

from llmz.experiment import record_run, training_batch
from llmz.tokenizer import PairTokenizer, BytesTokenizer
from llmz.transport import StreamingLogPolicy


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


def test_full_attention_batch_bypasses_transport():
    batch = {"x": np.ones((2, 4), dtype=np.int32)}
    policy = StreamingLogPolicy(recent_tokens=2, memory_tokens=2, horizon=4)
    result = training_batch(
        batch, policy, PairTokenizer(BytesTokenizer(), BytesTokenizer()),
        np.random.default_rng(0), full_attention=True)
    assert result["transport_spans"].shape == (2, 0, 3)
