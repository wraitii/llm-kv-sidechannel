import json
import pytest

from llmz.experiment import record_run


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
