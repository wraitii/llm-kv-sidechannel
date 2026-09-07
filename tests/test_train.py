import numpy as np
import mlx.core as mx

from llmz.train import sample_train_windows
from llmz.model import PrefixLM


def test_sample_train_windows_scalar_range_and_none():
    rng = np.random.default_rng(0)
    assert sample_train_windows(rng, None, 4) is None
    windows = sample_train_windows(rng, 32, 4)
    assert windows.tolist() == [32, 32, 32, 32]
    windows = sample_train_windows(rng, [24, 40], 64)
    assert windows.min() >= 24 and windows.max() <= 40


def test_full_attention_bypasses_scored_eviction():
    mx.random.seed(0)
    model = PrefixLM(16, 16, 16, 1, 2, 1, 12, dtype=mx.float32,
                     scored_eviction={"recent_window": 2, "memory_tokens": 1})
    tokens = mx.array([[1, 4, 5, 6, 2, 16, 17]])
    valid = mx.ones(tokens.shape, dtype=mx.bool_)
    positions = mx.array([[4, 5, 6]])
    full = model(tokens, valid, mx.array([5]), positions,
                 full_attention=True)
    retentions = [block.retention for block in model.blocks]
    for block in model.blocks:
        block.retention = None
    expected = model(tokens, valid, mx.array([5]), positions)
    for block, retention in zip(model.blocks, retentions, strict=True):
        block.retention = retention
    mx.eval(full, expected)
    assert mx.allclose(full, expected, atol=1e-5, rtol=1e-5).item()
