import json

import pytest

from llmpr_torch.state_data import generate_pairs, make_counterfactual_pair


BACKGROUND = "A long passage about a house and everyone inside it. " * 40


def test_pair_is_deterministic_and_has_shared_suffix():
    first = make_counterfactual_pair(BACKGROUND, background_id="book-1", seed=7)
    second = make_counterfactual_pair(BACKGROUND, background_id="book-1", seed=7)
    assert first == second
    a, b = first
    assert a.pair_id == b.pair_id
    assert a.answer != b.answer
    assert a.prompt[a.events[0].char_end:] == b.prompt[b.events[0].char_end:]
    for episode in first:
        last_event = episode.events[-1]
        assert episode.answer not in episode.prompt[last_event.char_start:last_event.char_end]
        assert len(episode.events) >= 3
        assert json.loads(episode.to_json())["example_id"] == episode.example_id


def test_generation_changes_seed_by_background():
    rows = list(generate_pairs((("one", BACKGROUND), ("two", BACKGROUND)), seed=9))
    assert len(rows) == 4
    assert len({row.example_id for row in rows}) == 4


def test_short_background_is_rejected():
    with pytest.raises(ValueError, match="600"):
        make_counterfactual_pair("short", background_id="x", seed=0)
