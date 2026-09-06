import json

import numpy as np

from llmz.data import PairDataset
from llmz.tokenizer import EOS, PairTokenizer, BytesTokenizer


def test_loss_starts_at_separator(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"asm": "ab", "code": "xy"}) + "\n")
    dataset = PairDataset(path, PairTokenizer(BytesTokenizer(), BytesTokenizer()), 10, 10)
    batch = dataset.batch(np.array([0]))
    assert batch["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1]]
    assert batch["y"][0, -1] == EOS
    # Input target bytes live after the source vocabulary; labels remain in
    # the fixed local target namespace consumed by the C softmax.
    assert batch["x"][0, 4] >= dataset.tokenizer.source.vocab_size
    assert batch["y"][0, 3] < dataset.tokenizer.target.vocab_size


def test_pause_is_an_unlabelled_bidirectional_prefix_position(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"asm": "ab", "code": "xy"}) + "\n")
    tokenizer = PairTokenizer(BytesTokenizer(), BytesTokenizer(), pause_token=True)
    batch = PairDataset(path, tokenizer, 10, 10).batch(np.array([0]))

    # BOS ab SEP PAUSE x y; PAUSE is the final prefix position and the first
    # target label is predicted from it.
    assert batch["x"].shape == (1, 7)
    assert batch["x"][0, 4] == tokenizer.pause_id
    assert batch["prefix_lengths"].tolist() == [5]
    assert batch["loss_mask"].tolist() == [[0, 0, 0, 0, 1, 1, 1]]
    assert batch["output_positions"].tolist() == [[4, 5, 6]]
