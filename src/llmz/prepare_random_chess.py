"""Generate uniformly random legal chess trajectories for state tracking."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .prepare_chess import sampled_prefix_plies, split_for, state_records


def random_legal_trajectory(rng: random.Random, min_plies: int,
                            max_plies: int) -> list[str]:
    """Return a uniformly legal UCI trajectory of random requested length."""
    import chess

    target_plies = rng.randint(min_plies, max_plies)
    for _ in range(10_000):
        board = chess.Board()
        moves: list[str] = []
        while len(moves) < target_plies:
            legal_moves = list(board.legal_moves)
            # Do not call Board.is_game_over(claim_draw=True) here: checking
            # every possible repetition is far costlier than legal-move
            # generation, and a claimable draw does not make a continuation
            # illegal. Only positions with no legal move need a restart.
            if not legal_moves:
                break
            move = legal_moves[rng.randrange(len(legal_moves))]
            moves.append(move.uci())
            board.push(move)
        if len(moves) == target_plies:
            return moves
    raise RuntimeError("could not generate a legal random trajectory")


def progress_bar(completed: int, total: int, started: float) -> None:
    """Render a dependency-free progress bar suitable for a long local run."""
    width = 30
    fraction = completed / total
    filled = int(width * fraction)
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed / elapsed
    remaining = (total - completed) / rate if rate else 0.0
    print(f"\r[{'#' * filled}{'.' * (width - filled)}] "
          f"{completed:,}/{total:,} ({fraction:6.2%}) "
          f"{rate:,.1f} trajectories/s ETA {remaining:,.0f}s",
          end="", file=sys.stderr, flush=True)


def prepare(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    handles = {name: (args.out / f"{name}.jsonl").open("w")
               for name in ("train", "val", "test")}
    rows = {name: 0 for name in handles}
    games = {name: 0 for name in handles}
    try:
        for index in range(args.trajectories):
            moves = random_legal_trajectory(rng, args.min_plies, args.max_plies)
            game_id = f"random-{args.seed}-{index:08d}"
            split = split_for(game_id, args.seed, args.val_fraction, args.test_fraction)
            records = state_records(
                moves, game_id,
                {"trajectory_type": "uniform_random_legal", "result": "*",
                 "winner": None},
                sampled_prefix_plies(game_id, len(moves), args))
            games[split] += 1
            for record in records:
                handles[split].write(json.dumps(record) + "\n")
                rows[split] += 1
            if args.log_every and (index + 1) % args.log_every == 0:
                print(json.dumps({"event": "progress", "trajectories": index + 1,
                                  "rows": rows}), flush=True)
            if args.progress and ((index + 1) % args.progress_every == 0 or
                                  index + 1 == args.trajectories):
                progress_bar(index + 1, args.trajectories, started)
    finally:
        for handle in handles.values():
            handle.close()
    if args.progress:
        print(file=sys.stderr)
    manifest = {
        "dataset": "generated:uniform-random-legal-chess:v1",
        "seed": args.seed,
        "trajectories": args.trajectories,
        "min_plies": args.min_plies,
        "max_plies": args.max_plies,
        "min_prefix_plies": args.min_prefix_plies,
        "state_prefixes_per_game": args.state_prefixes_per_game,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "games": games,
        "rows": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="generate uniformly random legal chess trajectories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=50_000)
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--min-prefix-plies", type=int, default=8)
    parser.add_argument("--state-prefixes-per-game", type=int, default=8)
    parser.add_argument("--seed", type=int, default=714_727)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction,
                        default=True, help="show a live terminal progress bar")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="refresh progress every N trajectories")
    args = parser.parse_args()
    if args.trajectories < 1:
        parser.error("--trajectories must be positive")
    if args.min_plies < 1 or args.max_plies < args.min_plies:
        parser.error("invalid ply range")
    if not 1 <= args.min_prefix_plies <= args.max_plies:
        parser.error("--min-prefix-plies must be within the game ply range")
    if args.state_prefixes_per_game < 1:
        parser.error("--state-prefixes-per-game must be positive")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("validation + test fractions must be below 1")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
