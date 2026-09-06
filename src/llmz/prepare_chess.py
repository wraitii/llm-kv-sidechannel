"""Stream Lichess games and build compact history -> terminal-state examples."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def split_for(group: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def parse_game(movetext: str):
    import chess
    import chess.pgn

    class MainlineVisitor(chess.pgn.BaseVisitor):
        def __init__(self):
            self.headers = {}
            self.board = chess.Board()
            self.moves = []

        def visit_header(self, tagname, tagvalue):
            self.headers[tagname] = tagvalue

        def begin_variation(self):
            # Comments and side variations are irrelevant to the state target.
            return chess.pgn.SKIP

        def visit_move(self, board, move):
            self.moves.append(move.uci())
            self.board.push(move)

        def result(self):
            return self

    visitor = chess.pgn.read_game(io.StringIO(movetext), Visitor=MainlineVisitor)
    if visitor is None:
        return None
    return visitor.headers, visitor.board, visitor.moves


def state_records(moves: list[str], game_id: str, metadata: dict,
                  prefix_plies: list[int]) -> list[dict]:
    """Build history -> board-state examples at requested legal prefixes."""
    import chess

    requested = set(prefix_plies)
    board = chess.Board()
    records = []
    for ply, move_text in enumerate(moves, start=1):
        board.push_uci(move_text)
        if ply not in requested:
            continue
        history = " ".join(moves[:ply])
        record = {
            "history": history,
            "final_fen": board.fen(),
            "last_move": move_text,
            "plies": ply,
            "game_id": game_id,
            "example_id": f"{game_id}#ply-{ply}",
            **metadata,
        }
        # Generic trainer compatibility: <BOS> history <fen> target <EOS>.
        record["asm"] = history
        record["code"] = record["final_fen"]
        records.append(record)
    return records


def sampled_prefix_plies(game_id: str, total_plies: int, args: argparse.Namespace) -> list[int]:
    """Deterministically select state probes without crossing game splits."""
    available = list(range(args.min_prefix_plies, total_plies + 1))
    count = min(args.state_prefixes_per_game, len(available))
    if count == 0:
        return []
    # Game-local RNG makes the examples stable if upstream scan order changes.
    seed = int.from_bytes(hashlib.sha256(
        f"{args.seed}\0{game_id}".encode()).digest()[:8], "big")
    return sorted(random.Random(seed).sample(available, count))


def prepare(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    # Reservoir sampling lets us obtain a varied subset without materializing
    # or downloading the billions of rows in the upstream split.
    selected: list[dict] = []
    scanned = eligible = 0
    if args.parquet is not None:
        import pyarrow.parquet as pq

        def rows():
            for path in args.parquet:
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(
                        batch_size=args.parquet_batch_size,
                        columns=["movetext", "Result", "Site", "WhiteElo", "BlackElo"]):
                    for row in batch.to_pylist():
                        yield row
        dataset = rows()
    elif args.source == "api":
        def rows():
            for offset in range(0, args.scan_limit, args.api_batch_size):
                query = urllib.parse.urlencode({
                    "dataset": args.dataset, "config": "default",
                    "split": args.upstream_split, "offset": offset,
                    "length": min(args.api_batch_size, args.scan_limit - offset),
                })
                url = "https://datasets-server.huggingface.co/rows?" + query
                for attempt in range(7):
                    try:
                        with urllib.request.urlopen(url, timeout=120) as response:
                            payload = json.load(response)
                        break
                    except urllib.error.HTTPError as error:
                        # The rows service occasionally returns transient
                        # gateway failures on large scans. Treat them like a
                        # rate limit rather than discarding an hours-long run.
                        if error.code not in {429, 500, 502, 503, 504} or attempt == 6:
                            raise
                        time.sleep(min(60, 2 ** attempt))
                    except urllib.error.URLError:
                        if attempt == 6:
                            raise
                        time.sleep(min(60, 2 ** attempt))
                yield from (item["row"] for item in payload.get("rows", []))
        dataset = rows()
    else:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset, split=args.upstream_split,
                               streaming=True, revision=args.revision)

    for row in dataset:
        if args.scan_limit and scanned >= args.scan_limit:
            break
        scanned += 1
        if args.log_every and scanned % args.log_every == 0:
            print(json.dumps({"event": "progress", "scanned": scanned,
                              "eligible": eligible, "retained": len(selected)}),
                  flush=True)
        movetext = str(row.get("movetext", ""))
        if not movetext:
            continue
        try:
            white_elo = int(row.get("WhiteElo", 0) or 0)
            black_elo = int(row.get("BlackElo", 0) or 0)
        except (TypeError, ValueError):
            continue
        if (white_elo < args.min_elo or black_elo < args.min_elo or
                white_elo > args.max_elo or black_elo > args.max_elo):
            continue
        # Avoid invoking SAN parsing for obviously long games. Move numbers
        # remain present even when the PGN contains comments/variations.
        move_numbers = [int(value) for value in
                        re.findall(r"(?<![\w.])(\d+)\.(?!\.)", movetext)]
        if move_numbers and max(move_numbers) * 2 > args.max_plies + 4:
            continue
        try:
            parsed = parse_game(movetext)
        except (ValueError, RuntimeError):
            continue
        if parsed is None:
            continue
        headers, board, moves = parsed
        plies = len(moves)
        result = headers.get("Result", str(row.get("Result", "*")))
        if result not in {"1-0", "0-1"} or not args.min_plies <= plies <= args.max_plies:
            continue
        eligible += 1
        site = str(row.get("Site", headers.get("Site", f"row-{scanned}")))
        record = {
            "moves": moves,
            "game_id": site,
            "result": result,
            "winner": "white" if result == "1-0" else "black",
            "white_elo": white_elo,
            "black_elo": black_elo,
        }
        if len(selected) < args.limit:
            selected.append(record)
        else:
            if not args.reservoir:
                break
            index = rng.randrange(eligible)
            if index < args.limit:
                selected[index] = record

    args.out.mkdir(parents=True, exist_ok=True)
    handles = {name: (args.out / f"{name}.jsonl").open("w")
               for name in ("train", "val", "test")}
    counts = {name: 0 for name in handles}
    try:
        for game in selected:
            record_meta = {key: value for key, value in game.items()
                           if key not in {"moves", "game_id"}}
            records = state_records(
                game["moves"], game["game_id"], record_meta,
                sampled_prefix_plies(game["game_id"], len(game["moves"]), args))
            split = split_for(game["game_id"], args.seed,
                              args.val_fraction, args.test_fraction)
            for record in records:
                handles[split].write(json.dumps(record) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "dataset": args.dataset, "revision": args.revision,
        "upstream_split": args.upstream_split, "seed": args.seed,
        "scan_limit": args.scan_limit, "scanned": scanned,
        "eligible": eligible, "selected_games": len(selected),
        "min_plies": args.min_plies, "max_plies": args.max_plies,
        "min_prefix_plies": args.min_prefix_plies,
        "state_prefixes_per_game": args.state_prefixes_per_game,
        "min_elo": args.min_elo, "max_elo": args.max_elo,
        "reservoir": args.reservoir,
        "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
        "rows": counts,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Lichess/standard-chess-games")
    parser.add_argument("--parquet", type=Path, nargs="+",
                        help="one or more local Parquet shards; bypasses the remote dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--upstream-split", default="train")
    parser.add_argument("--source", choices=["api", "stream"], default="api",
                        help="bounded rows API (default) or datasets streaming")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20_000,
                        help="number of eligible games to retain")
    parser.add_argument("--reservoir", action="store_true",
                        help="continue scanning and reservoir-sample instead of stopping at limit")
    parser.add_argument("--scan-limit", type=int, default=500_000,
                        help="maximum upstream rows to inspect")
    parser.add_argument("--api-batch-size", type=int, default=100)
    parser.add_argument("--parquet-batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--min-prefix-plies", type=int, default=8,
                        help="earliest history length eligible for a <fen> query")
    parser.add_argument("--state-prefixes-per-game", type=int, default=1,
                        help="deterministic random board-state probes per retained game")
    parser.add_argument("--min-elo", type=int, default=1600)
    parser.add_argument("--max-elo", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.limit <= 0 or args.scan_limit <= 0:
        parser.error("--limit and --scan-limit must be positive")
    if not 1 <= args.api_batch_size <= 100:
        parser.error("--api-batch-size must be between 1 and 100")
    if args.parquet_batch_size < 1:
        parser.error("--parquet-batch-size must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if args.min_plies < 1 or args.max_plies < args.min_plies:
        parser.error("invalid ply range")
    if not 1 <= args.min_prefix_plies <= args.max_plies:
        parser.error("--min-prefix-plies must be within the game ply range")
    if args.state_prefixes_per_game < 1:
        parser.error("--state-prefixes-per-game must be positive")
    if args.min_elo < 0 or args.max_elo < args.min_elo:
        parser.error("invalid Elo range")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("validation + test fractions must be below 1")
    prepare(args)


if __name__ == "__main__":
    main()
