import numpy as np
import pytest

from llmpr_torch.policies import (
    FixedSWA, StreamingLog, VariableSWA, causal_visibility,
    memento_visibility, retain_memory_positions,
)


def test_fixed_and_variable_swa_reference_masks():
    FixedSWA(4)
    mask = causal_visibility(8, np.array([2, 4], dtype=np.int32))
    assert mask.shape == (2, 8, 8)
    assert mask[0, -1].sum() == 2
    assert mask[1, -1].sum() == 4
    assert not np.triu(mask, k=1).any()
    samples = VariableSWA(2, 4).sample(np.random.default_rng(3), 100)
    assert samples.min() >= 2 and samples.max() <= 4
    with pytest.raises(ValueError):
        FixedSWA(0)


def test_streaming_log_capacity_and_irreversibility():
    policy = StreamingLog(4, 4)
    previous = set()
    deleted = set()
    for length in range(1, 100):
        current = set(policy.survivors(length))
        deleted |= previous - current
        assert not (current & deleted)
        assert len(current) == min(length, policy.capacity)
        assert set(range(max(0, length - 4), length)) <= current
        previous = current


def test_streaming_log_incremental_schedule_matches_known_reference():
    policy = StreamingLog(4, 4)
    schedule = list(policy.survivor_schedule(20))
    assert schedule[11] == (0, 3, 6, 7, 8, 9, 10, 11)
    assert schedule[19] == (0, 11, 14, 15, 16, 17, 18, 19)
    assert schedule[-1] == policy.survivors(20)


def test_memory_positions_compose_with_streaming_log_as_pinned_keys():
    base = np.zeros((1, 10, 10), dtype=np.bool_)
    for query, alive in enumerate(StreamingLog(2, 2).survivor_schedule(10)):
        base[0, query, list(alive)] = True
    visible = retain_memory_positions(base, ((2, 3),))
    assert visible[0, 9, 2]
    assert visible[0, 9, 3]
    assert not base[0, 9, 2]
    assert not visible[0, 1, 2]
    assert np.array_equal(visible[:, :, 4:], base[:, :, 4:])


def test_memory_position_validation():
    with pytest.raises(ValueError, match="invalid inclusive memory span"):
        retain_memory_positions(causal_visibility(4), ((3, 4),))


def test_memento_mask_hides_completed_blocks_but_keeps_memories():
    visible = memento_visibility(12, ((3, 4), (8, 9)))[0]
    # The first memory can read its complete source block.
    assert set(np.flatnonzero(visible[3])) == {0, 1, 2, 3}
    # After M1, its source is gone and the new current block is visible.
    assert set(np.flatnonzero(visible[7])) == {3, 4, 5, 6, 7}
    # M2 sees M1 plus its current block, including itself causally.
    assert set(np.flatnonzero(visible[8])) == {3, 4, 5, 6, 7, 8}
    # After M2, only both memories and the final current block survive.
    assert set(np.flatnonzero(visible[11])) == {3, 4, 8, 9, 10, 11}


def test_memento_span_validation():
    with pytest.raises(ValueError, match="invalid or overlapping"):
        memento_visibility(8, ((2, 4), (4, 5)))
