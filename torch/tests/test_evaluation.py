import numpy as np

from llmpr_torch.evaluation import (
    RestartMode, last_restart_before, parse_policy, parse_restart,
    policy_visibility, reconstruction_positions,
)
from llmpr_torch.policies import FixedSWA, FullAttention, StreamingLog


def test_restart_boundaries_and_reconstruction_support():
    assert last_restart_before(31, 20, RestartMode(every=8)) == 24
    assert last_restart_before(9, 10, RestartMode(at_answer=True)) is None
    assert last_restart_before(10, 10, RestartMode(at_answer=True)) == 10
    assert reconstruction_positions(12, 8, FixedSWA(3)) == (5, 6, 7, 8, 9, 10, 11)
    assert reconstruction_positions(5, 3, FullAttention()) == (0, 1, 2, 3, 4)


def test_sparse_policy_visibility_uses_absolute_positions():
    positions = (0, 4, 5, 9)
    visible = policy_visibility(positions, FixedSWA(3))[0]
    assert np.array_equal(np.flatnonzero(visible[-1]), np.array([3]))
    log = StreamingLog(2, 2)
    log_visible = policy_visibility(tuple(range(12)), log)[0]
    assert np.array_equal(np.flatnonzero(log_visible[-1]), np.array(log.survivors(12)))


def test_policy_and_restart_parsing():
    assert parse_policy("swa:64") == FixedSWA(64)
    assert parse_policy("log:16+8") == StreamingLog(16, 8)
    assert parse_restart("restart:1").every == 1
