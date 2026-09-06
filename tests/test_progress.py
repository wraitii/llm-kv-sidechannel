from llmz.prepare_random_chess import progress_bar


def test_progress_bar_writes_completion_to_stderr(capsys):
    progress_bar(10, 10, 0.0)
    assert "10/10" in capsys.readouterr().err
