"""Inspect retention after completed quiet piece reversals and board cycles.

Run from the repository root after evaluate.py. These are positional redundancy
probes, not claims that move history or FEN clocks can be discarded losslessly.
"""
import json
from pathlib import Path

import chess
import mlx.core as mx
import numpy as np

from llmz.inspect_attention import build_sequence, source_pieces, char_move_map
from llmz.model import causal_mask
from llmz.runtime import load_config, model_and_tokenizer
from llmz.train import load_model_checkpoint

out = Path('runs/controlled-scored/inspection')
rows = [e['row'] for e in json.loads((out / 'results.json').read_text())[0]['examples_detail']]
cfg = load_config('configs/controlled/scored.json')
model, tok = model_and_tokenizer(cfg)
load_model_checkpoint(Path('runs/controlled-scored/checkpoint-0001000.npz'), model)
records = []
comparisons = []
for row in rows:
    moves = row['asm'].split()
    board = chess.Board()
    identities = {s: chess.square_name(s) for s in board.piece_map()}
    previous = {}
    episodes = []
    states = {tuple(board.fen().split()[:4]): 0}
    for i, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        identity = identities.pop(move.from_square)
        quiet = (not board.is_capture(move) and not board.is_castling(move)
                 and board.piece_type_at(move.from_square) != chess.PAWN)
        rights = board.castling_rights
        if board.is_en_passant(move):
            identities.pop(move.to_square + (-8 if board.turn else 8))
        identities.pop(move.to_square, None)
        if board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            kingside = move.to_square > move.from_square
            rook = identities.pop(chess.square(7 if kingside else 0, rank))
            identities[chess.square(5 if kingside else 3, rank)] = rook
            previous.pop(rook, None)
        identities[move.to_square] = identity
        board.push(move)
        quiet = quiet and board.castling_rights == rights
        old = previous.get(identity)
        if quiet and old and old['quiet'] and old['uci'][:2] == uci[2:] and old['uci'][2:] == uci[:2]:
            episodes.append(dict(kind='quiet_piece_return', moves=[old['index'], i], completed_move=i))
        previous[identity] = dict(index=i, uci=uci, quiet=quiet)
        key = tuple(board.fen().split()[:4])
        if key in states and i + 1 - states[key] <= 8:
            episodes.append(dict(kind='board_cycle', moves=list(range(states[key], i + 1)), completed_move=i))
        states[key] = i + 1
    if not episodes:
        continue
    assert board.fen() == row['code']
    cmap = char_move_map(row['asm'])
    move_positions = {i: [] for i in range(len(moves))}
    for pos, (_, start, end) in enumerate(source_pieces(tok, row['asm']), 1):
        for i in {cmap[c] for c in range(start, end) if c in cmap}:
            move_positions[i].append(pos)
    seq, valid, boundary, _, labels, _ = build_sequence(tok, row, cfg)
    sep = int(boundary[0]) - 1
    x, valid = mx.array(seq), mx.array(valid)
    h = model.embed(x)
    mask = causal_mask(valid, h.dtype)
    positions = mx.broadcast_to(mx.arange(h.shape[1]), valid.shape)
    layer_data = []
    for block in model.blocks:
        _, _, values = block._qkv(h)
        raw = block.retention.sequence_scores(block.n1(h), values)
        keep, _ = block.retention.select_sequence(raw, valid, positions, positions)
        layer_data.append((np.asarray(raw)[0], np.asarray(keep)[0]))
        h = block(h, mask, valid=valid, positions=positions)
    known_redundant = set()
    for episode in episodes:
        completed = max(move_positions[episode['completed_move']])
        for i in episode['moves']:
            known_redundant.update(p for p in move_positions[i] if completed <= p + 24)
    for p in range(1, sep):
        query = p + 24
        # Exclude the period before there is competition for eight memory slots.
        if p < 8 or query >= len(labels) - 1:
            continue
        comparisons.append(dict(file_index=row['file_index'], position=p,
            known_return=p in known_redundant, scored_during_source=query < sep,
            rejected=[not bool(keep[query, p]) for _, keep in layer_data]))
    for episode in episodes:
        completed = max(move_positions[episode['completed_move']])
        tokens = sorted({p for i in episode['moves'] for p in move_positions[i]})
        probes = []
        for pos in tokens:
            assigned_at = pos + 24
            # Last useful input query predicts EOS; exclude the appended EOS query.
            if assigned_at >= len(labels) - 1:
                continue
            probes.append(dict(position=pos, label=labels[pos], assigned_at=assigned_at,
                return_known_when_scored=completed <= assigned_at,
                scored_during_source=assigned_at < sep,
                score=[float(scores[pos]) for scores, keep in layer_data],
                kept_when_scored=[bool(keep[assigned_at, pos]) for scores, keep in layer_data],
                kept_at_sep=[bool(keep[sep, pos]) for scores, keep in layer_data],
                kept_at_last_prediction=[bool(keep[-2, pos]) for scores, keep in layer_data]))
        records.append(dict(file_index=row['file_index'], kind=episode['kind'], move_indices=episode['moves'],
            moves=[moves[i] for i in episode['moves']], completion_position=completed,
            separator_position=sep, tokens=probes))
(out / 'redundancy.json').write_text(json.dumps(records, indent=2))
(out / 'redundancy-comparison.json').write_text(json.dumps(comparisons, indent=2))
summary = []
for source in (True, False):
    pairs = []
    for token in comparisons:
        if not token['known_return'] or token['scored_during_source'] != source:
            continue
        controls = [other for other in comparisons
                    if other['file_index'] == token['file_index']
                    and not other['known_return']
                    and other['scored_during_source'] == source
                    and abs(other['position'] - token['position']) <= 8]
        if controls:
            control = min(controls, key=lambda other:
                          (abs(other['position'] - token['position']), other['position']))
            pairs.append((token, control))
    summary.append(dict(scored_during_source=source, matched_tokens=len(pairs),
        return_rejection_rate=sum(sum(a['rejected']) for a, _ in pairs) / (6 * len(pairs)) if pairs else None,
        control_rejection_rate=sum(sum(b['rejected']) for _, b in pairs) / (6 * len(pairs)) if pairs else None))
(out / 'redundancy-summary.json').write_text(json.dumps(summary, indent=2))
for record in records:
    known = [t for t in record['tokens'] if t['return_known_when_scored']]
    if not known:
        continue
    print(record['file_index'], record['kind'], record['moves'])
    for t in known:
        print(' ', t['position'], t['label'], 'source' if t['scored_during_source'] else 'FEN',
              'kept at scoring', [i for i,k in enumerate(t['kept_when_scored']) if k],
              'at final prediction', [i for i,k in enumerate(t['kept_at_last_prediction']) if k])
