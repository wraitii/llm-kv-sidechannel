import numpy as np

from llmz.data import MixedPairDataset
from llmz.tokenizer import BytesTokenizer, PairTokenizer


class TinyDataset:
    def __init__(self, text: str):
        self.tokenizer = PairTokenizer(BytesTokenizer(), BytesTokenizer())
        self.max_source_tokens = self.max_target_tokens = 32
        self.text = text

    def __len__(self):
        return 4

    def encode(self, index):
        source = self.tokenizer.encode_source(self.text)
        target = self.tokenizer.encode_target_local("x")
        sequence = [1, *source, 2, *self.tokenizer.target_local_to_model(target), 3]
        return sequence, target, len(source) + 1, 1, len(self.text), len(source), 2


def test_mixed_dataset_draws_a_valid_batch_from_both_sources():
    dataset = MixedPairDataset(TinyDataset("a"), TinyDataset("long"), 0.5)
    batch = dataset.sample(np.random.default_rng(7), 10)
    assert batch["x"].shape[0] == 10
    assert set(batch["source_tokens"].tolist()) == {1, 4}
