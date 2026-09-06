"""Independent FEN readouts attached to one shared causal move history."""
from __future__ import annotations

import chess
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .experiment import training_batch
from .kv_state import KVState
from .move_alignment import batch_sources, move_boundaries
from .tokenizer import BOS, SEP, EOS, PAD


def prepare_readouts(raw, policy, tokenizer, policy_rng, readout_rng,
                     points=3, min_plies=8, max_target_tokens=96,
                     negative_probability=0.0, layers=1,
                     full_attention_share=0.0):
    """Sample up to points complete-move readouts, always including the endpoint.

    Histories are the source prefixes sampled by the existing dataset. Earlier
    targets are reconstructed from legal UCI moves, never inferred from a FEN
    string. The policy is sampled once for the whole shared history.
    """
    if points < 1 or min_plies < 1:
        raise ValueError("readout points and minimum plies must be positive")
    if tokenizer.pause_tokens:
        raise ValueError("shared readouts currently require pause_tokens=0")
    sources = batch_sources(raw)
    width = max(map(len, sources)) + 2
    x = np.full((len(sources), width), PAD, dtype=np.int32)
    valid = np.zeros_like(x, dtype=bool)
    positions = np.ones((len(sources), points), dtype=np.int32)
    targets = [[[] for _ in range(points)] for _ in sources]
    readout_valid = np.zeros((len(sources), points), dtype=bool)
    for row, source in enumerate(sources):
        edges = move_boundaries(source, tokenizer)
        moves = tokenizer.decode_source(source).split()
        count = len(moves)
        if not count:
            raise ValueError("readouts require a nonempty legal move history")
        eligible = np.arange(min(min_plies, count), count)
        chosen = sorted([*readout_rng.choice(eligible, min(points-1, len(eligible)), replace=False).tolist(), count])
        lookup = {ply: index for index, ply in enumerate(chosen)}
        board = chess.Board()
        for ply, move in enumerate(moves, 1):
            board.push_uci(move)
            if ply in lookup:
                index = lookup[ply]
                target = tokenizer.encode_target_local(board.fen())
                if len(target) > max_target_tokens:
                    raise ValueError("readout FEN exceeds max_target_tokens")
                positions[row, index] = int(edges[ply]) + 1
                targets[row][index] = target
                readout_valid[row, index] = True
        ids = [BOS, *source, SEP]
        x[row, :len(ids)] = ids
        valid[row, :len(ids)] = True
    batch = dict(x=x, valid=valid, source_tokens=np.array(list(map(len, sources))),
                 prefix_lengths=np.array([len(s)+2 for s in sources]),
                 output_positions=positions)
    batch, full_rows = training_batch(
        batch, policy, tokenizer, policy_rng, full_attention_share)
    endpoints = batch['prefix_lengths'] - 1
    width = int(endpoints.max())
    source_valid = batch['valid'][:, :width] & (np.arange(width)[None] < endpoints[:, None])
    max_answer = max(len(t)+1 for row in targets for t in row)
    branch_x = np.full((len(sources), points, max_answer), PAD, dtype=np.int32)
    branch_y = np.full_like(branch_x, PAD)
    branch_valid = np.zeros_like(branch_x, dtype=bool)
    for row in range(len(sources)):
        for point in range(points):
            if readout_valid[row, point]:
                target = targets[row][point]
                branch_x[row, point, :len(target)+1] = [SEP, *tokenizer.target_local_to_model(target)]
                branch_y[row, point, :len(target)+1] = [*target, EOS]
                branch_valid[row, point, :len(target)+1] = True
    negative = np.full(4, -1, dtype=np.int32)
    scored_rows = np.flatnonzero(~full_rows)
    if (negative_probability and len(scored_rows)
            and readout_rng.random() < negative_probability):
        row = int(readout_rng.choice(scored_rows))
        point = int(readout_rng.choice(np.flatnonzero(readout_valid[row])))
        negative[:] = (row, point, int(readout_rng.integers(layers)),
                       int(readout_rng.integers(2**30)))
    return dict(source_x=batch['x'][:, :width], source_valid=source_valid,
                spans=batch['transport_spans'], points=batch['output_positions'],
                branch_x=branch_x, branch_y=branch_y, branch_valid=branch_valid,
                negative=negative, full_rows=full_rows)


def _slice_cache(cache, index):
    """Take one row without detaching its differentiable history."""
    spans = cache.spans[index:index + 1] if cache.spans is not None else None
    alive = [value[index:index + 1] for value in cache.alive]
    return KVState([(k[index:index + 1], v[index:index + 1]) for k, v in cache],
                   cache.tokens[index:index + 1], cache.positions[index:index + 1],
                   cache.valid[index:index + 1], cache.next_positions[index:index + 1],
                   spans, alive, retention_scores=[
                       value[index:index + 1] if value is not None else None
                       for value in cache.retention_scores])


def _repeat_cache(cache, repeats):
    """Repeat a one-row cache for batched support counterfactuals."""
    spans = mx.repeat(cache.spans, repeats, axis=0) if cache.spans is not None else None
    alive = [mx.repeat(value, repeats, axis=0) for value in cache.alive]
    return KVState([(mx.repeat(k, repeats, axis=0), mx.repeat(v, repeats, axis=0))
                    for k, v in cache],
                   mx.repeat(cache.tokens, repeats, axis=0),
                   mx.repeat(cache.positions, repeats, axis=0),
                   mx.repeat(cache.valid, repeats, axis=0),
                   mx.repeat(cache.next_positions, repeats, axis=0), spans, alive,
                   retention_scores=[
                       mx.repeat(value, repeats, axis=0) if value is not None else None
                       for value in cache.retention_scores])


def shared_readout_loss(model, source_x, source_valid, spans, points,
                        branch_x, branch_y, branch_valid, sliding_window=None,
                        negative=None, negative_weight=0.1, negative_swaps=2,
                        negative_alternatives=4, return_metrics=False,
                        full_rows=None):
    """Compute independent FEN branches after one differentiable history pass."""
    rows, readouts, answer_length = branch_x.shape
    _, history = model.prefill(source_x, source_valid,
                               mx.sum(source_valid, axis=1),
                               sliding_window, spans, full_rows=full_rows)
    row_index = mx.repeat(mx.arange(rows), readouts)
    branch_window = (mx.repeat(sliding_window, readouts)
                     if isinstance(sliding_window, mx.array) else sliding_window)
    branch_full_rows = (mx.repeat(full_rows, readouts)
                        if full_rows is not None else None)
    point = points.reshape(-1)
    positions = history.positions[row_index]
    valid = history.valid[row_index] & (positions < point[:, None])
    layers = [(k[row_index], v[row_index]) for k, v in history]
    alive = []
    retention_scores = []
    for layer, snapshots in enumerate(history.alive_history):
        if snapshots is None:
            alive.append(valid)
            retention_scores.append(None)
        else:
            snapshot = snapshots[row_index, mx.maximum(point - 1, 0)]
            alive.append(snapshot & valid)
            assigned = positions <= point[:, None] - model.blocks[layer].retention.window
            retention_scores.append(history.retention_scores[layer][row_index]
                                    * assigned)
    repeated_spans = history.spans[row_index] if history.spans is not None else None
    cache = KVState(layers, history.tokens[row_index], positions, valid, point,
                    repeated_spans, alive, retention_scores=retention_scores)
    flat_x = branch_x.reshape(rows * readouts, answer_length)
    flat_y = branch_y.reshape(rows * readouts, answer_length)
    flat_valid = branch_valid.reshape(rows * readouts, answer_length)
    first_cache = cache
    first_trace = []
    logits = model.continue_sequence(flat_x, flat_valid, cache, branch_window,
                                     selection_trace=first_trace,
                                     full_rows=branch_full_rows)
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), flat_y)
    weighted = losses * flat_valid
    total = mx.sum(weighted)
    per_branch = mx.sum(weighted, axis=1)
    count = mx.maximum(mx.sum(flat_valid), 1)
    base_loss = total / count
    empty_metrics = (mx.array(0.0), mx.array(0.0), mx.array(0.0))
    if (negative is None or not first_trace or int(negative[0].item()) < 0
            or negative_weight <= 0 or negative_swaps < 1):
        return (base_loss, *empty_metrics) if return_metrics else base_loss

    row, readout, layer, seed = [int(value.item()) for value in negative]
    branch = row * readouts + readout
    if layer >= len(first_trace):
        raise ValueError("negative scorer layer is out of range")
    scores, eligible, normal_live = first_trace[layer]
    scores = scores[branch:branch + 1]
    eligible = eligible[branch:branch + 1]
    normal_live_one = normal_live[branch:branch + 1]
    query_position = point[branch]
    cache_positions = first_cache.positions[branch:branch + 1]
    history_length = cache_positions.shape[1]
    history_live = normal_live_one[:, :history_length]
    older = (eligible[:, :history_length]
             & (cache_positions <= query_position - model.blocks[layer].retention.window))
    retained = older & history_live
    rejected = first_cache.valid[branch:branch + 1] & ~history_live
    rejected = rejected & (cache_positions <= query_position - model.blocks[layer].retention.window)
    alternatives = int(negative_alternatives)
    if alternatives < 1:
        raise ValueError("negative_alternatives must be positive")
    noise = mx.array(np.random.default_rng(seed).random(
        (alternatives, cache_positions.shape[1])), dtype=mx.float32)
    retained = mx.repeat(retained, alternatives, axis=0)
    rejected = mx.repeat(rejected, alternatives, axis=0)

    def random_pick(candidates):
        count = min(negative_swaps, candidates.shape[-1])
        ranked = mx.argpartition(mx.where(candidates, noise, -1.0),
                                 candidates.shape[-1] - count, axis=-1)
        chosen = ranked[:, -count:]
        picked = mx.any(mx.arange(candidates.shape[-1])[None, :, None]
                        == chosen[:, None, :], axis=-1)
        return picked & candidates

    removed, inserted = random_pick(retained), random_pick(rejected)
    changed = (mx.any(removed, axis=1) & mx.any(inserted, axis=1)).astype(mx.float32)
    repeated_live = mx.repeat(history_live, alternatives, axis=0)
    replacement = (repeated_live & ~removed) | inserted
    replacement = mx.concatenate(
        [replacement, mx.repeat(normal_live_one[:, history_length:],
                                alternatives, axis=0)], axis=1)
    forced = [mx.repeat(live[branch:branch + 1], alternatives, axis=0)
              for _, _, live in first_trace]
    forced[layer] = replacement
    alt_cache = _repeat_cache(_slice_cache(first_cache, branch), alternatives)
    alt_window = (branch_window[branch:branch + 1]
                  if isinstance(branch_window, mx.array) else branch_window)
    if isinstance(alt_window, mx.array):
        alt_window = mx.repeat(alt_window, alternatives)
    alt_count = mx.maximum(mx.sum(flat_valid[branch]), 1)
    alt_logits = model.continue_sequence(
        mx.repeat(flat_x[branch:branch + 1], alternatives, axis=0),
        mx.repeat(flat_valid[branch:branch + 1], alternatives, axis=0), alt_cache,
        alt_window, first_forced_alive=forced)
    alt_losses = nn.losses.cross_entropy(
        alt_logits.astype(mx.float32),
        mx.repeat(flat_y[branch:branch + 1], alternatives, axis=0))
    alt_total = mx.sum(
        alt_losses * mx.repeat(flat_valid[branch:branch + 1], alternatives, axis=0),
        axis=1)
    normal_loss = per_branch[branch] / alt_count
    alt_loss = alt_total / alt_count
    repeated_scores = mx.repeat(scores[:, :history_length], alternatives, axis=0)
    removed_score = (mx.sum(repeated_scores * removed, axis=1)
                     / mx.maximum(mx.sum(removed, axis=1), 1))
    inserted_score = (mx.sum(repeated_scores * inserted, axis=1)
                      / mx.maximum(mx.sum(inserted, axis=1), 1))
    gap = removed_score - inserted_score
    relative_advantage = mx.stop_gradient(mx.clip(
        (alt_loss - normal_loss) / mx.maximum(mx.abs(normal_loss), 1e-3),
        -0.25, 0.25))
    preference = mx.sign(relative_advantage)
    weights = mx.abs(relative_advantage) * changed
    ranking = (mx.sum(nn.softplus(-preference * gap) * weights)
               / mx.maximum(mx.sum(changed), 1))
    result = base_loss + negative_weight * ranking
    metrics = (mx.sum((alt_loss - normal_loss) * changed),
               mx.sum((alt_loss < normal_loss) * changed), mx.sum(changed))
    return (result, *metrics) if return_metrics else result
