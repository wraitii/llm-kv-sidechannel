"""Create state-probe JSONL from a simple background-text JSONL file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .state_data import generate_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL rows containing id and text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--structured", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    backgrounds = []
    with args.input.open() as handle:
        for line in handle:
            row = json.loads(line)
            backgrounds.append((str(row["id"]), str(row["text"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for episode in generate_pairs(backgrounds, seed=args.seed,
                                      natural=not args.structured):
            handle.write(episode.to_json() + "\n")
