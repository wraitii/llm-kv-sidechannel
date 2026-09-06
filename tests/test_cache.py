import argparse
import json

import numpy as np

from llmz.cache import build_cache
from llmz.data import CachedPairDataset, PairDataset
from llmz.tokenizer import BytesTokenizer, PairTokenizer


def test_cached_batches_match_fresh_batches(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"asm": "mov %edi, %eax\nret", "code": "int f(int x) { return x; }"},
        {"asm": "xor %eax, %eax\nret", "code": "int z(void) { return 0; }"},
    ]
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
    cache_dir = tmp_path / "cache"
    build_cache(argparse.Namespace(
        data_dir=data_dir, out=cache_dir, source_tokenizer="bytes",
        target_tokenizer="bytes", max_source_tokens=64, max_target_tokens=64))

    tokenizer = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    fresh = PairDataset(data_dir / "train.jsonl", tokenizer, 64, 64)
    cached = CachedPairDataset(cache_dir, "train", tokenizer)
    indices = np.array([1, 0])
    fresh_batch = fresh.batch(indices)
    cached_batch = cached.batch(indices)
    assert fresh_batch.keys() == cached_batch.keys()
    for key in fresh_batch:
        np.testing.assert_array_equal(fresh_batch[key], cached_batch[key])

    bucketed = CachedPairDataset(cache_dir, "train", tokenizer, bucket_size=2)
    sampled = bucketed.sample(np.random.default_rng(7), batch_size=2)
    assert sampled["x"].shape[0] == 2


def test_overlong_examples_are_dropped_without_changing_targets(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    rows = [{'asm': 'ab', 'code': 'xy'}, {'asm': 'abcdef', 'code': 'z'},
            {'asm': 'a', 'code': 'abcdef'}]
    (data_dir / 'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    tokenizer = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    fresh = PairDataset(data_dir / 'train.jsonl', tokenizer, 3, 3)
    assert len(fresh) == 1
    assert fresh.skipped_overlong == 2
    build_cache(argparse.Namespace(data_dir=data_dir, out=tmp_path/'cache',
        source_tokenizer='bytes', target_tokenizer='bytes', max_source_tokens=3, max_target_tokens=3))
    cached = CachedPairDataset(tmp_path/'cache', 'train', tokenizer)
    assert len(cached) == 1
    assert cached.encode(0) == fresh.encode(0)
    metadata = json.loads((tmp_path/'cache'/'metadata.json').read_text())
    assert metadata['splits']['train']['skipped_overlong'] == 2
