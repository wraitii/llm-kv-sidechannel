import mlx.core as mx
import numpy as np

from llmz.model import causal_mask, prefix_lm_mask, same_pass_transport_mask


def test_prefix_bidirectional_target_causal_and_padding_hidden():
    valid = mx.array([[True, True, True, True, True, False]])
    mask = np.array(prefix_lm_mask(valid, mx.array([3])))
    allowed = mask[0, 0] == 0
    assert allowed[0].tolist() == [True, True, True, False, False, False]
    assert allowed[2].tolist() == [True, True, True, False, False, False]
    assert allowed[3].tolist() == [True, True, True, True, False, False]
    assert allowed[4].tolist() == [True, True, True, True, True, False]


def test_causal_mask_never_exposes_a_future_or_padding_key():
    mask = np.array(causal_mask(mx.array([[True, True, True, False]])))
    allowed = mask[0, 0] == 0
    assert allowed[0].tolist() == [True, False, False, False]
    assert allowed[2].tolist() == [True, True, True, False]


def test_causal_mask_can_restrict_attention_to_a_sliding_window():
    mask = np.array(causal_mask(mx.array([[True] * 6]), sliding_window=3))
    allowed = mask[0, 0] == 0
    assert allowed[5].tolist() == [False, False, False, True, True, True]


def test_transport_mask_composes_eviction_with_a_sliding_window():
    valid = mx.array([[True] * 8])
    mask = np.array(same_pass_transport_mask(
        valid, mx.array([[[2, 4, 4]]]), sliding_window=3))
    allowed = mask[0, 0] == 0
    # The carrier sees its three-token local window including the full span.
    assert allowed[4].tolist() == [False, False, True, True, True, False, False, False]
    # The next query still sees carrier 4, but not evicted span interiors 2--3.
    assert allowed[5].tolist() == [False, False, False, False, True, True, False, False]


def test_causal_mask_supports_per_row_sliding_windows():
    valid = mx.array([[True] * 6, [True] * 6])
    mask = np.array(causal_mask(valid, sliding_window=mx.array([2, 4])))
    allowed = mask[:, 0] == 0
    # Row 0 has window 2: the last query sees only the two nearest keys.
    assert allowed[0, 5].tolist() == [False, False, False, False, True, True]
    # Row 1 has window 4: the last query sees four keys back.
    assert allowed[1, 5].tolist() == [False, False, True, True, True, True]
