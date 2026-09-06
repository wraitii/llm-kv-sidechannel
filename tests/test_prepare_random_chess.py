import random

import chess

from llmz.prepare_random_chess import random_legal_trajectory


def test_random_legal_trajectory_is_replayable_and_in_requested_range():
    moves = random_legal_trajectory(random.Random(11), 20, 20)
    board = chess.Board()
    for text in moves:
        move = chess.Move.from_uci(text)
        assert move in board.legal_moves
        board.push(move)
    assert len(moves) == 20
