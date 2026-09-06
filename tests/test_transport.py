import mlx.core as mx
import numpy as np

from llmz.model import same_pass_transport_mask
from llmz.transport import RandomContiguousPolicy, RecursiveBlockPolicy, policy_from_config


def test_random_spans_are_ordered_nonoverlapping_and_keep_a_carrier():
    policy = RandomContiguousPolicy(span_tokens=4, spans_per_example=3,
                                    min_source_tokens=4, min_gap_tokens=0)
    spans = policy.sample(np.array([32, 3]), np.random.default_rng(7))
    selected = spans[0][spans[0, :, 0] >= 0]
    assert len(selected) == 3
    assert np.all(selected[:, 1] - selected[:, 0] == 3)  # [start, carrier) hidden
    assert np.all(selected[:, 2] == selected[:, 1])      # carrier is visible_until
    assert np.all(selected[1:, 0] > selected[:-1, 1])
    assert spans[1].tolist() == [[-1, -1, -1]] * 3
    assert policy_from_config({"kind": "random_contiguous", "span_tokens": 4})


def test_transport_mask_keeps_carrier_but_hides_its_span_from_future():
    valid = mx.array([[True] * 8])
    # Sequence positions [2, 3, 4] are a span; 4 is its ordinary carrier.
    mask = np.array(same_pass_transport_mask(valid, mx.array([[[2, 4, 4]]])))
    allowed = mask[0, 0] == 0
    assert allowed[4, :5].tolist() == [True, True, True, True, True]
    assert allowed[5].tolist() == [True, True, False, False, True, True, False, False]


def test_recursive_blocks_tile_then_recurse_over_survivors():
    policy = RecursiveBlockPolicy(hidden_tokens=4, survivor_tokens=2,
                                  group_size=4, depth=2)
    spans = policy.sample(np.array([24]), np.random.default_rng(0))
    # Level 1: four [4 hidden | 2 survivor] periods.
    assert spans[0, :4].tolist() == [[0, 4, 5], [6, 10, 11],
                                     [12, 16, 17], [18, 22, 23]]
    # Level 2: the first survivor group of four is hidden behind its two
    # meta-survivors; the final group has no successors and stays visible.
    active = spans[0][spans[0, :, 0] >= 0]
    assert [span.tolist() for span in active[4:]] == [[4, 12, 17]]


def test_recursive_blocks_mask_hides_survivors_behind_meta_survivors():
    from llmz.model import same_pass_transport_mask
    policy = RecursiveBlockPolicy(hidden_tokens=4, survivor_tokens=2,
                                  group_size=4, depth=2)
    spans = policy.sample(np.array([24]), np.random.default_rng(0))[0]
    # Sequence coordinates: +1 for BOS.
    mask = np.array(same_pass_transport_mask(
        mx.array([[True] * 26]), mx.array(spans[None] + 1)))
    allowed = mask[0, 0] == 0
    # Survivor 5 (sequence 6) is readable by its meta-survivor (query 18)
    # but hidden from later queries, which must rely on meta-survivors.
    assert allowed[18, 6].item()
    assert not allowed[20, 6].item()
    # Meta-survivors themselves stay visible downstream.
    assert allowed[20, 18].item()
    assert allowed[25, 18].item()


def test_recursive_blocks_degrade_gracefully_on_short_sources():
    policy = RecursiveBlockPolicy(hidden_tokens=4, survivor_tokens=2,
                                  gap_tokens=0, group_size=8, depth=2)
    spans = policy.sample(np.array([5, 10, 40]), np.random.default_rng(0))
    assert spans[0, 0, 0] == -1                # too short for even one period
    active10 = spans[1][spans[1, :, 0] >= 0]
    assert active10.tolist() == [[0, 4, 5]]    # one period, visible tail 6..9
    # 40 tokens: 6 periods cover 36 tokens; the 4-token tail stays visible.
    level1 = spans[2][(spans[2, :, 0] >= 0) & (spans[2, :, 1] - spans[2, :, 0] == 4)]
    assert level1[-1].tolist() == [30, 34, 35]  # last hidden block + survivors
    assert level1[-1, 1].item() < 40 - 4


def test_recursive_blocks_sample_sizes_uniformly_per_period():
    policy = RecursiveBlockPolicy(hidden_tokens=[3, 5], survivor_tokens=[2, 3],
                                  gap_tokens=[0, 2], group_size=8, depth=1)
    a = policy.sample(np.array([200]), np.random.default_rng(1))[0]
    b = policy.sample(np.array([200]), np.random.default_rng(2))[0]
    assert not np.array_equal(a, b)               # actually random
    for spans in (a, b):
        active = spans[spans[:, 0] >= 0]
        assert active[0, 0] == 0
        assert np.all(active[1:, 0] > active[:-1, 1])  # ordered, non-overlapping
        assert active[-1, 1] <= 200
        hidden = active[:, 1] - active[:, 0]
        visible = active[1:, 0] - active[:-1, 2] - 1   # survivors + gap
        assert hidden.min() >= 3 and hidden.max() <= 5
        # visible stretch between spans is the gap alone (survivors sit
        # between each span's end and its visible_until)
        assert visible.min() >= 0 and visible.max() <= 2
        # visible_until is the last survivor of each period
        assert np.all(active[:, 2] > active[:, 1])
        assert np.all(active[:, 2] < np.append(active[1:, 0], 200))


def test_recursive_blocks_rejects_bad_ranges():
    import pytest
    with pytest.raises(ValueError):
        RecursiveBlockPolicy(hidden_tokens=[5, 3])
    with pytest.raises(ValueError):
        RecursiveBlockPolicy(survivor_tokens=[0, 2])
    with pytest.raises(ValueError):
        RecursiveBlockPolicy(gap_tokens=[1, 2, 3])
