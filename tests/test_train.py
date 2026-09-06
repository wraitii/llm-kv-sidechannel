import numpy as np

from llmz.train import sample_train_windows


def test_sample_train_windows_scalar_range_and_none():
    rng = np.random.default_rng(0)
    assert sample_train_windows(rng, None, 4) is None
    windows = sample_train_windows(rng, 32, 4)
    assert windows.tolist() == [32, 32, 32, 32]
    windows = sample_train_windows(rng, [24, 40], 64)
    assert windows.min() >= 24 and windows.max() <= 40
