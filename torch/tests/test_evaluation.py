import numpy as np

from llmpr_torch.evaluation import (
    RestartMode, last_restart_before, parse_policy, parse_restart,
    paired_task_aggregates, policy_visibility, reconstruction_positions, task_loss_aggregates,
)
from llmpr_torch.policies import FixedSWA, FullAttention, StreamingLog, VariableSWA
from llmpr_torch.state_data import StateEpisode


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
    assert parse_policy("variable-swa:64-1024") == VariableSWA(64, 1024)
    assert parse_policy("log:16+8") == StreamingLog(16, 8)
    assert parse_restart("restart:1").every == 1


def test_task_loss_aggregates_can_split_task_types():
    def episode(task_type: str) -> StateEpisode:
        return StateEpisode(
            example_id=task_type, pair_id="pair", variant="a", prompt="prompt",
            answer="answer", events=(), query_entity="entity", background_id="book",
            task_type=task_type,
        )

    episodes = [episode("state"), episode("passcode"), episode("state")]
    split = task_loss_aggregates(episodes, [[1.0], [4.0, 6.0], [3.0]], split_task_type=True)
    assert split == [
        {"examples": 1, "answer_tokens": 2, "target_nll_per_token": 5.0,
         "task_type": "passcode"},
        {"examples": 2, "answer_tokens": 2, "target_nll_per_token": 2.0,
         "task_type": "state"},
    ]


def test_paired_aggregates_report_margin_and_accuracy_without_eos():
    episodes = [
        StateEpisode("a", "p", "a", "", "x", (), "e", "b", task_type="state"),
        StateEpisode("b", "p", "b", "", "y", (), "e", "b", task_type="state"),
    ]
    assert paired_task_aggregates(episodes, [(1.0, 3.0, 1), (4.0, 2.0, 1)])[0] == {
        "examples": 2, "answer_tokens": 2, "task_type": "state",
        "correct_answer_nll": 2.5, "counterfactual_answer_nll": 2.5,
        "mean_nll_margin": 0.0, "pairwise_accuracy": 0.5,
    }


def test_passcodes_report_token_weighted_answer_nll_only():
    episodes = [
        StateEpisode("a", "p1", "a", "", "x", (), "e", "b",
                     task_type="passcode_easy"),
        StateEpisode("b", "p2", "a", "", "y", (), "e", "b",
                     task_type="passcode_easy"),
    ]
    result = paired_task_aggregates(episodes, [(2.0, 9.0, 1), (6.0, 9.0, 3)])[0]
    assert result == {
        "examples": 2, "answer_tokens": 4, "task_type": "passcode_easy",
        "answer_nll_per_token": 2.0,
    }
