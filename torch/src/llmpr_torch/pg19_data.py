"""Stream PG-19 from Hugging Face and construct small, reproducible probes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from transformers import AutoTokenizer
from huggingface_hub import HfApi

from .state_data import StateEpisode, make_counterfactual_pair, make_passcode_pair
from .tokenization import tokenize_episode, validate_counterfactual_pair


LEVELS: dict[str, Callable[..., tuple[StateEpisode, StateEpisode]]] = {
    "passcode": make_passcode_pair,
    "structured": lambda text, **kw: make_counterfactual_pair(text, natural=False, **kw),
    "natural": lambda text, **kw: make_counterfactual_pair(text, natural=True, **kw),
}


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item)
    if not lengths or min(lengths) < 256:
        raise argparse.ArgumentTypeError("context lengths must be comma-separated integers >= 256")
    return lengths


def stable_book_id(row: dict) -> str:
    source = str(row.get("url") or row.get("short_book_title") or
                 row.get("id") or row["text"][:200])
    return hashlib.sha256(source.encode()).hexdigest()[:20]


def fit_pair(
    tokenizer,
    text: str,
    *,
    background_id: str,
    seed: int,
    context_length: int,
    builder: Callable[..., tuple[StateEpisode, StateEpisode]],
) -> tuple[StateEpisode, StateEpisode]:
    """Find the largest prefix whose complete prompt and answer fit the budget."""
    # Do not tokenize a multi-million-token book to construct one short row.
    # Select a stable, generously oversized character window first.
    window_chars = max(4000, context_length * 8)
    available = max(0, len(text) - window_chars)
    selector = hashlib.sha256(f"{background_id}\0{seed}\0{context_length}".encode()).digest()
    start = int.from_bytes(selector[:8], "big") % (available + 1)
    if start:
        next_space = text.find(" ", start, min(len(text), start + 200))
        if next_space >= 0:
            start = next_space + 1
    window = text[start:start + window_chars]
    token_ids = tokenizer(window, add_special_tokens=False)["input_ids"]
    low, high = 1, min(len(token_ids), context_length)
    best: tuple[StateEpisode, StateEpisode] | None = None
    while low <= high:
        count = (low + high) // 2
        background = tokenizer.decode(token_ids[:count], skip_special_tokens=True)
        if len(background) < 600:
            low = count + 1
            continue
        pair = None
        encoded = None
        # Random alphanumeric strings do not always have matching BPE lengths.
        # Deterministically seek a token-aligned counterfactual rather than
        # weakening the position-matched comparison.
        for attempt in range(64):
            candidate = builder(background, background_id=background_id,
                                seed=seed + attempt)
            candidate_encoded = [
                tokenize_episode(tokenizer, row, max_length=10**12)
                for row in candidate
            ]
            try:
                validate_counterfactual_pair(*candidate_encoded)
            except ValueError:
                continue
            pair, encoded = candidate, candidate_encoded
            break
        if pair is None or encoded is None:
            raise ValueError("could not generate a token-aligned counterfactual pair")
        if max(len(row.input_ids) for row in encoded) > context_length:
            high = count - 1
        else:
            best = pair
            low = count + 1
    if best is None:
        raise ValueError(f"could not fit a probe in {context_length} tokens")
    return tuple(replace(row, context_length=context_length) for row in best)  # type: ignore[return-value]


def select_books(rows: Iterable[dict], count: int, *, minimum_chars: int = 2000) -> list[dict]:
    selected = []
    for row in rows:
        if len(str(row.get("text", ""))) >= minimum_chars:
            selected.append(dict(row))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} usable books, found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF model ID or local tokenizer path")
    parser.add_argument("--dataset", default="emozilla/pg19")
    parser.add_argument("--revision", default="main",
                        help="HF dataset revision; pin a commit for archival runs")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--context-lengths", type=parse_lengths, default=(1024,))
    parser.add_argument("--levels", default="passcode,structured,natural")
    parser.add_argument("--train-books", type=int, default=8)
    parser.add_argument("--validation-books", type=int, default=2)
    parser.add_argument("--test-books", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()
    levels = tuple(item for item in args.levels.split(",") if item)
    unknown = set(levels) - LEVELS.keys()
    if unknown:
        parser.error(f"unknown levels: {sorted(unknown)}")
    counts = {"train": args.train_books, "validation": args.validation_books,
              "test": args.test_books}
    if min(counts.values()) < 0 or not any(counts.values()):
        parser.error("book counts must be nonnegative and at least one must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {args.output_dir}")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install data support with: uv sync --extra data") from error

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    resolved_revision = HfApi().dataset_info(
        args.dataset, revision=args.revision).sha
    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {}
    for split, count in counts.items():
        if count == 0:
            continue
        source = load_dataset(
            args.dataset, split=split, revision=resolved_revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            streaming=not args.no_streaming,
        )
        books = select_books(source, count)
        raw_path = args.output_dir / f"pg19-{split}.jsonl"
        episode_path = args.output_dir / f"{split}.jsonl"
        episode_count = 0
        with raw_path.open("x") as raw, episode_path.open("x") as episodes:
            for book_index, book in enumerate(books):
                book_id = stable_book_id(book)
                raw.write(json.dumps({
                    "id": book_id, "title": book.get("short_book_title"),
                    "publication_date": book.get("publication_date"),
                    "url": book.get("url"), "text": book["text"],
                }, ensure_ascii=False) + "\n")
                for length in args.context_lengths:
                    for level_index, level in enumerate(levels):
                        pair = fit_pair(
                            tokenizer, book["text"], background_id=book_id,
                            seed=args.seed + book_index * 1009 + level_index * 97 + length,
                            context_length=length, builder=LEVELS[level],
                        )
                        for row in pair:
                            episodes.write(row.to_json() + "\n")
                            episode_count += 1
        totals[split] = {"books": len(books), "episodes": episode_count,
                         "raw": str(raw_path), "data": str(episode_path)}

    manifest = {
        "format_version": 1, "dataset": args.dataset,
        "requested_revision": args.revision, "resolved_revision": resolved_revision,
        "streaming": not args.no_streaming, "model": args.model,
        "context_lengths": args.context_lengths, "levels": levels,
        "seed": args.seed, "splits": totals,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
