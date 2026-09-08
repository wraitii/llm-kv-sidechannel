import numpy as np
import pytest

from llmpr_torch.policies import (
    FixedSWA, StreamingLog, VariableSWA, causal_visibility,
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
