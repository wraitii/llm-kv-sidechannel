import numpy as np

from llmz.generate import sample_token


def test_greedy_generation_masks_input_only_specials():
    logits = np.array([100.0, 99.0, 98.0, 1.0, 2.0])
    assert sample_token(logits, np.random.default_rng(1), 0, 1, 0) == 4


def test_sampling_is_seeded():
    logits = np.array([0.0, 0.0, 0.0, -1.0, 1.0, 0.5])
    left = sample_token(logits, np.random.default_rng(7), 0.8, 0.9, 0)
    right = sample_token(logits, np.random.default_rng(7), 0.8, 0.9, 0)
    assert left == right
