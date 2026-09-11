"""Stream PG-19 from Hugging Face and construct small, reproducible probes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from .state_data import (
    MemorySpan, StateEpisode, StateEvent, make_counterfactual_pair, make_passcode_pair,
)
from .tokenization import tokenize_episode, validate_counterfactual_pair


def _easy_passcode(text: str, **kwargs) -> tuple[StateEpisode, StateEpisode]:
    return make_passcode_pair(text, difficulty="easy", **kwargs)


LEVELS: dict[str, Callable[..., tuple[StateEpisode, StateEpisode]]] = {
    "passcode_easy": _easy_passcode,
    "passcode_hard": make_passcode_pair,
    "structured": lambda text, **kw: make_counterfactual_pair(text, natural=False, **kw),
    "natural": lambda text, **kw: make_counterfactual_pair(text, natural=True, **kw),
}

MEMORY_LAYOUTS = {"none", "event", "fixed", "fixed-copy"}


def _shift_event(event: StateEvent, insertions: list[tuple[int, str, int | None]],
                 insertion_lengths: list[int]) -> StateEvent:
    start_shift = sum(length for (position, _, _), length in zip(
        insertions, insertion_lengths, strict=True) if position <= event.char_start)
    end_shift = sum(length for (position, _, _), length in zip(
        insertions, insertion_lengths, strict=True) if position < event.char_end)
    return replace(event, char_start=event.char_start + start_shift,
                   char_end=event.char_end + end_shift)


def _fixed_ratio_memory_positions(
    episode: StateEpisode,
    *,
    tokenizer,
    tokens_per_span: int,
    compression_ratio: int,
) -> list[int]:
    """Choose task-agnostic boundaries at a fixed ordinary-token interval."""
    question = episode.prompt.rfind("\n\nQuestion:")
    if question < 1:
        raise ValueError("episode has no question boundary")
    if compression_ratio < 1:
        raise ValueError("memory compression ratio must be positive")
    blocked = [(event.char_start, event.char_end) for event in episode.events]
    encoded = tokenizer(
        episode.prompt[:question], add_special_tokens=False,
        return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    interval = tokens_per_span * compression_ratio
    block_count = max(1, round(len(offsets) / interval))
    positions = []
    for block_index in range(1, block_count):
        target = round(block_index * len(offsets) / block_count)
        position = offsets[target][0]
        containing = next(((start, end) for start, end in blocked
                           if start < position < end), None)
        if containing is not None:
            position = containing[1]
        positions.append(position)
    # Compress the final, possibly short block before exposing the question.
    positions.append(question)
    return sorted(set(positions))


def _fixed_stride_memory_positions(
    episode: StateEpisode,
    *,
    tokenizer,
    tokens_per_span: int,
    compression_ratio: int,
) -> list[int]:
    """Place memories after complete fixed-size blocks, leaving a live tail."""
    question = episode.prompt.rfind("\n\nQuestion:")
    if question < 1:
        raise ValueError("episode has no question boundary")
    blocked = [(event.char_start, event.char_end) for event in episode.events]
    offsets = tokenizer(
        episode.prompt[:question], add_special_tokens=False,
        return_offsets_mapping=True)["offset_mapping"]
    interval = tokens_per_span * compression_ratio
    positions = []
    for target in range(interval, len(offsets) + 1, interval):
        position = offsets[target - 1][1]
        containing = next(((start, end) for start, end in blocked
                           if start < position < end), None)
        if containing is not None:
            # Do not split a task event or shift the global stride phase.
            # Skipping this boundary makes the next block an integer multiple
            # of the nominal interval and preserves its fixed-phase samples.
            continue
        positions.append(position)
    return sorted(set(positions))


def inject_memory(
    episode: StateEpisode,
    *,
    layout: str,
    tokens_per_span: int,
    memory_token: str,
    seed: int,
    tokenizer=None,
    compression_ratio: int = 20,
) -> StateEpisode:
    """Insert meaningless memory-token spans and preserve character metadata."""
    if layout not in MEMORY_LAYOUTS:
        raise ValueError(f"unknown memory layout: {layout}")
    if layout == "none":
        return episode
    if tokens_per_span < 1:
        raise ValueError("memory tokens per span must be positive")
    if not memory_token:
        raise ValueError("memory token must be nonempty")
    if layout == "event":
        insertions = [(event.char_end, "event", event.event_index)
                      for event in episode.events]
    else:
        if tokenizer is None:
            raise ValueError("fixed-ratio memory placement requires a tokenizer")
        position_fn = (_fixed_stride_memory_positions
                       if layout == "fixed-copy" else _fixed_ratio_memory_positions)
        insertions = [(position, layout, None) for position in position_fn(
            episode, tokenizer=tokenizer, tokens_per_span=tokens_per_span,
            compression_ratio=compression_ratio)]
    insertions.sort()
    token_text = memory_token * tokens_per_span
    rendered = f"\n\n{token_text}\n\n"
    insertion_lengths = [len(rendered)] * len(insertions)
    chunks = []
    spans = []
    cursor = 0
    source_encoding = None
    if layout == "fixed-copy":
        if tokenizer is None:
            raise ValueError("fixed-copy memory placement requires a tokenizer")
        if tokens_per_span < 2:
            raise ValueError("fixed-copy requires a sentinel and at least one copied token")
        source_encoding = tokenizer(
            episode.prompt, add_special_tokens=False, return_offsets_mapping=True)
        sentinel = tokenizer(memory_token, add_special_tokens=False)["input_ids"]
        if len(sentinel) != 1:
            raise ValueError("memory sentinel must encode to exactly one token")
    previous_position = 0
    phase_rng = np.random.Generator(np.random.PCG64(seed))
    for memory_index, (position, placement, after_event) in enumerate(insertions):
        chunks.append(episode.prompt[cursor:position])
        base = sum(map(len, chunks))
        chunks.append(rendered)
        replacements: tuple[int, ...] = ()
        source_positions: tuple[int, ...] = ()
        span_phase = None
        if source_encoding is not None:
            eligible = [index for index, (start, end) in enumerate(
                        source_encoding["offset_mapping"])
                        if start >= previous_position and end <= position]
            span_phase = int(phase_rng.integers(compression_ratio))
            selected = eligible[span_phase::compression_ratio][:tokens_per_span - 1]
            if len(selected) != tokens_per_span - 1:
                raise ValueError("ordinary block is too short for fixed-copy memory")
            source_positions = tuple(selected)
            replacements = (sentinel[0], *(source_encoding["input_ids"][i]
                                             for i in selected))
        spans.append(MemorySpan(
            char_start=base + 2, char_end=base + 2 + len(token_text),
            memory_index=memory_index, placement=placement,
            after_event_index=after_event,
            replacement_token_ids=replacements,
            source_token_positions=source_positions,
            copy_phase=span_phase,
        ))
        cursor = position
        previous_position = position
    chunks.append(episode.prompt[cursor:])
    suffix = f"-memory-{layout}-{tokens_per_span}"
    return replace(
        episode,
        example_id=episode.example_id + suffix,
        pair_id=episode.pair_id + suffix,
        prompt="".join(chunks),
        events=tuple(_shift_event(event, insertions, insertion_lengths)
                     for event in episode.events),
        memory_spans=tuple(spans),
        memory_layout=layout,
        memory_tokens_per_span=tokens_per_span,
        memory_compression_ratio=(compression_ratio
                                  if layout in {"fixed", "fixed-copy"} else None),
    )


def remove_memory(episode: StateEpisode) -> StateEpisode:
    """Recover the byte-identical pre-memory episode from an annotated row."""
    if not episode.memory_spans:
        return episode
    removals = [(span.char_start - 2, span.char_end + 2)
                for span in episode.memory_spans]
    chunks = []
    cursor = 0
    for start, end in removals:
        chunks.append(episode.prompt[cursor:start])
        cursor = end
    chunks.append(episode.prompt[cursor:])
    prompt = "".join(chunks)
    events = []
    search_start = 0
    for event in episode.events:
        text = episode.prompt[event.char_start:event.char_end]
        char_start = prompt.find(text, search_start)
        if char_start < 0:
            raise ValueError("could not recover event after removing memory spans")
        events.append(replace(event, char_start=char_start,
                              char_end=char_start + len(text)))
        search_start = char_start + len(text)
    suffix = f"-memory-{episode.memory_layout}-{episode.memory_tokens_per_span}"
    example_id = episode.example_id.removesuffix(suffix)
    pair_id = episode.pair_id.removesuffix(suffix)
    return replace(
        episode, example_id=example_id, pair_id=pair_id, prompt=prompt,
        events=tuple(events), memory_spans=(), memory_layout="none",
        memory_tokens_per_span=0, memory_compression_ratio=None,
    )


def insertion_positions(tokenizer, background: str, *, level: str, seed: int) -> tuple[int, ...]:
    """Choose token-driven injection points and return their character offsets."""
    offsets = tokenizer(
        background, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    length = len(offsets)
    rng = np.random.Generator(np.random.PCG64(seed))
    if level == "passcode_hard":
        token_positions = [int(rng.integers(max(1, length // 10), max(2, length // 4 + 1)))]
    elif level == "passcode_easy":
        token_positions = [max(1, length - int(rng.integers(320, 801)))]
    else:
        token_positions = []
        cursor = int(rng.integers(150, 301))
        while cursor < length - 150:
            token_positions.append(cursor)
            cursor += int(rng.integers(300, 601))
        while len(token_positions) < 3:
            token_positions.append(max(1, (len(token_positions) + 1) * length // 4))
    return tuple(offsets[min(position, length - 1)][0] for position in token_positions)


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
    level: str | None = None,
    memory_layout: str = "none",
    memory_tokens_per_span: int = 0,
    memory_token: str = "<|fim_pad|>",
    memory_compression_ratio: int = 20,
    additional_alignment_layouts: tuple[str, ...] = (),
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
            inferred_level = level or ("passcode_hard" if builder is make_passcode_pair else "state")
            positions = insertion_positions(
                tokenizer, background, level=inferred_level, seed=seed + attempt)
            base_candidate = builder(
                background, background_id=background_id, seed=seed + attempt,
                insertion_char_positions=positions)
            candidate = base_candidate
            if memory_layout != "none":
                candidate = tuple(inject_memory(
                    row, layout=memory_layout,
                    tokens_per_span=memory_tokens_per_span,
                    memory_token=memory_token, seed=seed + attempt + 7919,
                    tokenizer=tokenizer,
                    compression_ratio=memory_compression_ratio,
                ) for row in candidate)
            candidate_encoded = [
                tokenize_episode(tokenizer, row, max_length=10**12)
                for row in candidate
            ]
            try:
                validate_counterfactual_pair(*candidate_encoded)
                for alignment_layout in additional_alignment_layouts:
                    aligned = tuple(inject_memory(
                        row, layout=alignment_layout,
                        tokens_per_span=memory_tokens_per_span,
                        memory_token=memory_token, seed=seed + attempt + 7919,
                        tokenizer=tokenizer,
                        compression_ratio=memory_compression_ratio,
                    ) for row in base_candidate)
                    validate_counterfactual_pair(*[
                        tokenize_episode(tokenizer, row, max_length=10**12)
                        for row in aligned
                    ])
            except ValueError:
                continue
            pair, encoded = candidate, candidate_encoded
            break
        if pair is None or encoded is None:
            raise ValueError("could not generate a token-aligned counterfactual pair")
        if max(len(row.input_ids) for row in encoded) > context_length:
            high = count - 1
        else:
            distances = [tuple(row.prompt_length - end - 1 for _, end in row.support_token_spans)
                         for row in encoded]
            best = tuple(replace(row, support_to_answer_tokens=distances[index])
                         for index, row in enumerate(pair))
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
    parser.add_argument(
        "--levels", default="passcode_easy,passcode_hard,structured,natural")
    parser.add_argument("--train-books", type=int, default=8)
    parser.add_argument("--validation-books", type=int, default=2)
    parser.add_argument("--test-books", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--memory-layout", choices=sorted(MEMORY_LAYOUTS | {"both"}), default="none",
                        help="insert memory spans after events or at a fixed compression ratio")
    parser.add_argument("--memory-tokens-per-span", type=int, default=0,
                        help="width of every inserted memory span; zero disables memory")
    parser.add_argument("--memory-token", default="<|fim_pad|>",
                        help="single-token text repeated within each memory span")
    parser.add_argument("--memory-compression-ratio", type=int, default=20,
                        help="ordinary tokens per memory token in fixed/both layouts")
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()
    levels = tuple(item for item in args.levels.split(",") if item)
    unknown = set(levels) - LEVELS.keys()
    if unknown:
        parser.error(f"unknown levels: {sorted(unknown)}")
    if (args.memory_layout == "none") != (args.memory_tokens_per_span == 0):
        parser.error("use zero memory tokens with layout=none, or a positive count otherwise")
    if args.memory_compression_ratio < 1:
        parser.error("--memory-compression-ratio must be positive")
    memory_token_ids = None
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
    if args.memory_layout != "none":
        memory_token_ids = tokenizer(
            args.memory_token, add_special_tokens=False)["input_ids"]
        if len(memory_token_ids) != 1:
            parser.error("--memory-token must encode to exactly one token")
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
        layouts = (("event", "fixed") if args.memory_layout == "both"
                   else (args.memory_layout,))
        episode_paths = {
            layout: args.output_dir / (
                f"{split}-{layout}.jsonl" if args.memory_layout == "both"
                else f"{split}.jsonl")
            for layout in layouts
        }
        episode_count = 0
        with ExitStack() as stack:
            raw = stack.enter_context(raw_path.open("x"))
            outputs = {layout: stack.enter_context(path.open("x"))
                       for layout, path in episode_paths.items()}
            for book_index, book in enumerate(books):
                book_id = stable_book_id(book)
                raw.write(json.dumps({
                    "id": book_id, "title": book.get("short_book_title"),
                    "publication_date": book.get("publication_date"),
                    "url": book.get("url"), "text": book["text"],
                }, ensure_ascii=False) + "\n")
                for length in args.context_lengths:
                    for level_index, level in enumerate(levels):
                        fit_layout = "fixed" if args.memory_layout == "both" else args.memory_layout
                        fit_length = length - 16 if args.memory_layout == "both" else length
                        pair = fit_pair(
                            tokenizer, book["text"], background_id=book_id,
                            seed=args.seed + book_index * 1009 + level_index * 97 + length,
                            context_length=fit_length, builder=LEVELS[level], level=level,
                            memory_layout=fit_layout,
                            memory_tokens_per_span=args.memory_tokens_per_span,
                            memory_token=args.memory_token,
                            memory_compression_ratio=args.memory_compression_ratio,
                            additional_alignment_layouts=(
                                ("event",) if args.memory_layout == "both" else ()),
                        )
                        pairs = {fit_layout: pair}
                        if args.memory_layout == "both":
                            base_pair = tuple(remove_memory(row) for row in pair)
                            event_pair = tuple(inject_memory(
                                row, layout="event",
                                tokens_per_span=args.memory_tokens_per_span,
                                memory_token=args.memory_token,
                                seed=args.seed + book_index * 1009 + level_index * 97
                                + length + 7919,
                                tokenizer=tokenizer,
                                compression_ratio=args.memory_compression_ratio,
                            ) for row in base_pair)
                            encoded = [tokenize_episode(tokenizer, row, max_length=length)
                                       for row in event_pair]
                            validate_counterfactual_pair(*encoded)
                            pairs["event"] = tuple(replace(
                                row,
                                support_to_answer_tokens=tuple(
                                    item.prompt_length - end - 1
                                    for _, end in item.support_token_spans),
                            ) for row, item in zip(event_pair, encoded, strict=True))
                            pairs = {layout: tuple(replace(row, context_length=length)
                                                  for row in layout_pair)
                                     for layout, layout_pair in pairs.items()}
                        for layout, layout_pair in pairs.items():
                            for row in layout_pair:
                                outputs[layout].write(row.to_json() + "\n")
                                episode_count += 1
        totals[split] = {"books": len(books), "episodes": episode_count,
                         "raw": str(raw_path),
                         "data": ({layout: str(path) for layout, path in episode_paths.items()}
                                  if args.memory_layout == "both"
                                  else str(next(iter(episode_paths.values()))))}

    manifest = {
        "format_version": 1, "dataset": args.dataset,
        "requested_revision": args.revision, "resolved_revision": resolved_revision,
        "streaming": not args.no_streaming, "model": args.model,
        "context_lengths": args.context_lengths, "levels": levels,
        "memory_layout": args.memory_layout,
        "memory_tokens_per_span": args.memory_tokens_per_span,
        "memory_token": args.memory_token,
        "memory_token_id": None if memory_token_ids is None else memory_token_ids[0],
        "memory_compression_ratio": args.memory_compression_ratio,
        "memory_copy_phase": ("seeded-per-block"
                              if args.memory_layout == "fixed-copy" else None),
        "paired_layout_fit_slack": 16 if args.memory_layout == "both" else 0,
        "seed": args.seed, "splits": totals,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
