"""Shared semantic scoring for generated or teacher-forced board text."""
from __future__ import annotations

from collections.abc import Iterable


def score_board_outputs(outputs: Iterable[str], references: Iterable[str]) -> tuple[int, int, int, int, int]:
    """Return parseable, valid, exact, square-error, and metadata-error counts."""
    import chess

    parseable = valid = exact = square_errors = metadata_errors = 0
    for output, reference_text in zip(outputs, references, strict=True):
        output = output.strip()
        reference_text = reference_text.strip()
        try:
            if len(output.split()) != 6:
                continue
            board = chess.Board(output)
            reference = chess.Board(reference_text)
        except ValueError:
            continue
        parseable += 1
        valid += int(board.is_valid())
        exact += int(output == reference_text)
        square_errors += sum(
            board.piece_at(square) != reference.piece_at(square)
            for square in chess.SQUARES)
        metadata_errors += sum((board.turn != reference.turn,
                                board.castling_rights != reference.castling_rights,
                                board.ep_square != reference.ep_square))
    return parseable, valid, exact, square_errors, metadata_errors
