from argparse import Namespace

from llmz.prepare_chess import sampled_prefix_plies, state_records


def test_state_records_target_the_board_at_each_requested_prefix():
    moves = ["e2e4", "e7e5", "g1f3"]
    records = state_records(moves, "game", {"winner": "white"}, [1, 3])
    assert [record["history"] for record in records] == ["e2e4", "e2e4 e7e5 g1f3"]
    assert [record["plies"] for record in records] == [1, 3]
    assert records[0]["code"].startswith("rnbqkbnr/pppppppp/8/8/4P3")
    assert records[1]["example_id"] == "game#ply-3"


def test_prefix_sampling_is_stable_and_within_requested_bounds():
    args = Namespace(seed=7, min_prefix_plies=8, state_prefixes_per_game=4)
    first = sampled_prefix_plies("game", 20, args)
    assert first == sampled_prefix_plies("game", 20, args)
    assert len(first) == 4
    assert first == sorted(first)
    assert all(8 <= ply <= 20 for ply in first)
