import mlx.core as mx
import numpy as np
import pytest

from llmz.model import same_pass_transport_mask
from llmz.transport import FixedSparsePolicy, RecursiveBlockPolicy, policy_from_config


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
    with pytest.raises(ValueError):
        RecursiveBlockPolicy(hidden_tokens=[5, 3])
    with pytest.raises(ValueError):
        RecursiveBlockPolicy(survivor_tokens=[0, 2])
    with pytest.raises(ValueError):
        RecursiveBlockPolicy(gap_tokens=[1, 2, 3])


def _visible_after_source(policy, length):
    spans = policy.sample(np.array([length]), np.random.default_rng(0))[0]
    hidden = set()
    for start, end, visible_until in spans:
        if start >= 0:
            assert visible_until == length - 1
            hidden.update(range(start, end))
    return [index for index in range(length) if index not in hidden]


def test_fixed_sparse_uniform_keeps_exact_recent_and_memory_budgets():
    policy = FixedSparsePolicy(recent_tokens=16, memory_tokens=16,
                               strategy="uniform")
    visible = _visible_after_source(policy, 100)
    assert len(visible) == 32
    assert visible[-16:] == list(range(84, 100))
    assert visible[:16] == sorted(visible[:16])
    assert visible[0] < 5 and visible[15] > 79


def test_fixed_sparse_log_is_denser_near_recent_tail():
    uniform = _visible_after_source(
        FixedSparsePolicy(strategy="uniform"), 100)[:16]
    logarithmic = _visible_after_source(
        FixedSparsePolicy(strategy="log"), 100)[:16]
    assert logarithmic[0] == 0
    assert logarithmic[-1] == 83
    assert len(set(logarithmic)) == 16
    assert sum(index >= 64 for index in logarithmic) > sum(
        index >= 64 for index in uniform)


def test_fixed_sparse_short_sources_remain_fully_visible():
    policy = FixedSparsePolicy(recent_tokens=16, memory_tokens=16)
    assert _visible_after_source(policy, 32) == list(range(32))
    assert _visible_after_source(policy, 12) == list(range(12))


def test_fixed_sparse_config_validation():
    policy = policy_from_config({"kind": "fixed_sparse", "alignment": "token",
                                 "recent_tokens": 16, "memory_tokens": 16,
                                 "strategy": "log"})
    assert isinstance(policy, FixedSparsePolicy)
    with pytest.raises(ValueError, match="alignment='token'"):
        policy_from_config({"kind": "fixed_sparse", "alignment": "move"})
    with pytest.raises(ValueError, match="strategy"):
        policy_from_config({"kind": "fixed_sparse", "alignment": "token",
                            "strategy": "random"})
