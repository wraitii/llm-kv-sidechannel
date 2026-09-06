import numpy as np

from llmz.train import sample_train_windows


def test_sample_train_windows_scalar_range_and_none():
    rng = np.random.default_rng(0)
    assert sample_train_windows(rng, None, 4, None, 999) is None
    windows = sample_train_windows(rng, 32, 4, None, 999)
    assert windows.tolist() == [32, 32, 32, 32]
    windows = sample_train_windows(rng, [24, 40], 64, None, 999)
    assert windows.min() >= 24 and windows.max() <= 40


def test_sample_train_windows_full_attention_escape_rows():
    rng = np.random.default_rng(0)
    full_rows = np.array([True, False, True, False])
    windows = sample_train_windows(rng, [24, 40], 4, full_rows, 999)
    assert windows[0] == 999 and windows[2] == 999
    assert 24 <= windows[1] <= 40 and 24 <= windows[3] <= 40
    # scalar windows also become per-row when an escape share is active
    windows = sample_train_windows(rng, 32, 4, full_rows, 999)
    assert windows[0] == 999 and windows[1] == 32
