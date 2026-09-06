"""Inspect where a trained model's target tokens attend in the source.

Dumps per-layer, per-head attention probabilities for one teacher-forced
chess example and prints a move-level report: which source moves each FEN
output token reads from, how mass is distributed over the move history,
and how much lands on structural tokens (BOS/SEP).

Usage:
    uv run llmpr-attention --config configs/chess-move-bpe512-fen512-causal-20m.json --index 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .tokenizer import EOS, PairTokenizer
from .runtime import load_config, model_and_tokenizer
from .train import latest_checkpoint, load_model_checkpoint
from .transport import policy_from_config
from .inference import prepare_prefix


def pretty(token: str) -> str:
    """Render byte-level BPE spaces (Ġ) as a visible middle dot."""
    return token.replace("Ġ", "\u00b7")


def source_pieces(tokenizer: PairTokenizer, text: str) -> list[tuple[str, int, int]]:
    """Return (piece, char_start, char_end) for each source token."""
    if hasattr(tokenizer.source, "inner"):
        encoding = tokenizer.source.inner.encode(text, add_special_tokens=False)
        return list(zip(encoding.tokens, [o[0] for o in encoding.offsets],
                        [o[1] for o in encoding.offsets]))
    return [(chr(b), i, i + 1) for i, b in enumerate(
        tokenizer.source.encode(text))]


def target_pieces(tokenizer: PairTokenizer, text: str) -> list[tuple[str, int, int]]:
    if hasattr(tokenizer.target, "inner"):
        encoding = tokenizer.target.inner.encode(text, add_special_tokens=False)
        return list(zip(encoding.tokens, [o[0] for o in encoding.offsets],
                        [o[1] for o in encoding.offsets]))
    return [(chr(b), i, i + 1) for i, b in enumerate(
        tokenizer.target.encode(text))]


def char_move_map(text: str) -> dict[int, int]:
    """Map character offsets to 0-based move indices (spaces excluded)."""
    mapping: dict[int, int] = {}
    move = -1
    previous = " "
    for index, char in enumerate(text):
        if char != " " and previous == " ":
            move += 1
        if char != " ":
            mapping[index] = move
        previous = char
    return mapping


def fen_square_labels(code: str) -> list[str]:
    """Square name (a8..h1) for each character of the FEN board field."""
    board = code.split(" ")[0]
    labels = [""] * len(code)
    for index, char in enumerate(board):
        if char in "/":
            continue
        if char.isdigit():
            continue
        labels[index] = f"{chr(ord('a') + index % 8)}{8 - index // 8}"
    return labels


def target_token_labels(tokenizer: PairTokenizer, code: str) -> list[str]:
    """Human label per target BPE token: covered square(s) or raw piece."""
    squares = fen_square_labels(code)
    labels = []
    for piece, start, end in target_pieces(tokenizer, code):
        piece = pretty(piece)
        covered = [s for s in squares[start:end] if s]
        if covered:
            labels.append(f"{piece}@{covered[0]}" if len(covered) == 1
                          else f"{piece}@{covered[0]}-{covered[-1]}")
        else:
            labels.append(piece)
    return labels


def build_sequence(tokenizer: PairTokenizer, row: dict, cfg: dict) -> tuple:
    """Teacher-forced single-example inputs plus token/group label tables."""
    source = tokenizer.encode_source(row["asm"])
    target = tokenizer.encode_target(row["code"])
    if (len(source) > cfg["max_source_tokens"]
            or len(target) > cfg["max_target_tokens"]):
        raise ValueError("example exceeds configured token limits")
    pieces = source_pieces(tokenizer, row["asm"])[:len(source)]
    move_of_char = char_move_map(row["asm"])
    policy = policy_from_config(cfg.get("transport_policy"))
    prefix, _, prefix_lengths, spans = prepare_prefix(
        tokenizer, [row], cfg["max_source_tokens"], policy,
        transport=bool(policy and cfg.get("transport_eval", False)))
    prefix_ids = np.asarray(prefix)[0].tolist()
    source_labels, source_groups = ["<bos>"], [-1]
    piece_index = 0
    for token_id in prefix_ids[1:-1]:
        if token_id in tokenizer.carrier_ids:
            source_labels.append(f"<carrier:{tokenizer.carrier_ids.index(token_id)}>")
            source_groups.append(None)
        else:
            piece, start, _ = pieces[piece_index]
            source_labels.append(pretty(piece))
            source_groups.append(move_of_char.get(start, -1))
            piece_index += 1
    source_labels.append("<fen>")
    source_groups.append(-1)
    tokens = [*prefix_ids, *target, EOS]
    labels = source_labels + target_token_labels(tokenizer, row["code"]) + ["<eos>"]
    groups = source_groups + [None] * (len(target) + 1)
    sequence = np.array([tokens], dtype=np.int32)
    valid = np.ones_like(sequence, dtype=np.bool_)
    return sequence, valid, np.asarray(prefix_lengths), np.asarray(spans), labels, groups


def attention_tensor(model, tokenizer, row, cfg, layers, heads):
    sequence, valid, prefix_lengths, spans, labels, groups = build_sequence(
        tokenizer, row, cfg)
    tokens = mx.array(sequence)
    sliding = cfg.get("eval_sliding_window")
    _, maps = model.attention_maps(
        tokens, mx.array(valid), mx.array(prefix_lengths),
        transport_spans=mx.array(spans), sliding_window=sliding)
    # capture entries are [batch=1, heads, query, key]; drop the batch axis.
    probs = np.stack([np.asarray(p, dtype=np.float32) for p in maps])[:, 0]
    if layers == "last":
        layer_index = [probs.shape[0] - 1]
    else:
        layer_index = [int(v) for v in layers.split(",")]
    if heads:
        head_index = [int(v) for v in heads.split(",")]
    else:
        head_index = list(range(probs.shape[1]))
    n_source = int(prefix_lengths[0]) - 2
    return probs[layer_index][:, head_index], labels, groups, n_source


def bar(value: float, scale: float, width: int = 24) -> str:
    filled = int(round(value / scale * width)) if scale > 0 else 0
    return "█" * min(filled, width)


def report(probs: np.ndarray, labels: list, groups: list, n_source: int,
           row: dict, top_groups: int, top_keys: int, max_rows: int,
           include_target_keys: bool) -> None:
    # [T, T] mean over selected layers/heads; target-token queries only.
    # Sequence is [BOS, *source, SEP, *target, EOS]: target queries start at
    # n_source + 2 and keys 0..n_source + 1 cover BOS, source, SEP.
    mass = probs[:, :, n_source + 2:, :].mean(axis=(0, 1))
    total = mass[:, :n_source + 2].sum()
    if total <= 0:
        raise RuntimeError("no attention mass in source region; check indices")

    move_labels = {}
    for key, group in enumerate(groups):
        if group is None:
            continue
        if group >= 0:
            move_labels.setdefault(group, []).append(key)

    print(f"\nexample: {row.get('example_id', '?')}  "
          f"type={row.get('trajectory_type', '?')}  plies={row.get('plies', '?')}")
    print(f"fen: {row['code']}")
    print(f"source tokens: {n_source}  target tokens: {mass.shape[0] - 1} "
          f"(incl. <eos>)\n")

    # Attention mass per source move.
    scores = []
    for group, keys in sorted(move_labels.items()):
        scores.append((group, float(mass[:, keys].sum())))
    bos = float(mass[:, 0].sum())
    sep = float(mass[:, n_source + 1].sum())
    scale = max(v for _, v in scores) if scores else 1.0
    print(f"attention from target tokens to source moves "
          f"(BOS {bos / total:.1%}, SEP {sep / total:.1%}):")
    for group, value in sorted(scores, key=lambda item: -item[1])[:top_groups]:
        print(f"  m{group:03d} {bar(value, scale)} {value / total:6.1%}")
    rest = sum(v for _, v in scores[top_groups:])
    if rest > 0:
        print(f"  ... {rest / total:6.1%} over remaining moves")

    # Recency profile over raw source positions.
    recent = [(w, float(mass[:, max(0, n_source + 1 - w):n_source + 1].sum()))
              for w in (4, 8, 16, 32)]
    print("\nmass by recency window (last w source tokens, excl. SEP):")
    print("  " + "  ".join(f"w={w}:{v / total:.1%}" for w, v in recent))

    # Per-target-token top sources.
    print(f"\ntop source positions per target token (grouped by move):")
    shown = 0
    offset = n_source + 2  # sequence position of the first target query
    for query in range(mass.shape[0]):
        if max_rows and shown >= max_rows:
            print(f"  ... {mass.shape[0] - shown} more target tokens")
            break
        keys = np.argsort(-mass[query])
        parts, seen_moves = [], set()
        for key in keys:
            if not include_target_keys and key >= offset:
                continue  # skip previously emitted target tokens
            group = groups[key] if key < len(groups) else None
            name = labels[key] if key < len(labels) else "?"
            if group is not None and group >= 0:
                if group in seen_moves:
                    continue
                seen_moves.add(group)
                parts.append(f"m{group:03d}:{mass[query, key]:.2f}")
            else:
                parts.append(f"{name}:{mass[query, key]:.2f}")
            if len(parts) >= top_keys:
                break
        print(f"  {labels[offset + query]:<12} "
              f"-> {' '.join(parts)}")
        shown += 1


def save_outputs(path: Path, probs: np.ndarray, labels: list, groups: list,
                 row: dict, cfg: dict, png: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, attention=probs.astype(np.float16),
        labels=np.array(labels), groups=np.array(
            [-2 if g is None else g for g in groups], dtype=np.int32),
        config=json.dumps(cfg), example=json.dumps(row))
    print(f"\nwrote {path}")
    if not png:
        return
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed; skipping PNG "
              "(uv pip install matplotlib)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = probs.shape[0]
    figure, axes = plt.subplots(1, layers, figsize=(4 * layers, 4),
                                squeeze=False)
    for index in range(layers):
        image = axes[0][index].imshow(probs[index].mean(axis=0), origin="upper")
        axes[0][index].set_title(f"layer {index}")
        axes[0][index].set_xlabel("key")
    axes[0][0].set_ylabel("query")
    figure.colorbar(image, ax=axes[0][-1], fraction=0.046)
    figure.suptitle(str(path.parent.name))
    figure_path = path.with_suffix(".png")
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"wrote {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump and summarize attention for one chess example")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--layers", default="last",
                        help="'last', or comma-separated layer indices")
    parser.add_argument("--heads", default="",
                        help="comma-separated head indices (default: all)")
    parser.add_argument("--top-groups", type=int, default=16)
    parser.add_argument("--top-keys", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=48)
    parser.add_argument("--include-target-keys", action="store_true",
                        help="show previously emitted target tokens in the "
                             "per-token top-key listing")
    parser.add_argument("--out", type=Path, default=None,
                        help="output npz path (default under the run dir)")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = model_and_tokenizer(cfg)
    run_dir = Path(cfg["run_dir"])
    checkpoint = (latest_checkpoint(run_dir) if args.checkpoint == "auto"
                  else Path(args.checkpoint))
    state = load_model_checkpoint(checkpoint, model)

    path = Path(cfg["data_dir"]) / f"{args.split}.jsonl"
    with path.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index == args.index:
                row = json.loads(line)
                break
        else:
            parser.error(f"index {args.index} not found in {path}")

    probs, labels, groups, n_source = attention_tensor(
        model, tokenizer, row, cfg, args.layers, args.heads)
    print(f"checkpoint: {checkpoint.name} (step {state.get('step')}), "
          f"layers {probs.shape[0]}x heads {probs.shape[1]} selected, "
          f"attention_mode={model.attention_mode}")
    report(probs, labels, groups, n_source, row,
           args.top_groups, args.top_keys, args.max_rows,
           args.include_target_keys)
    out = args.out or run_dir / "attention" / f"{args.split}-{args.index}.npz"
    save_outputs(out, probs, labels, groups, row, cfg, args.png)


if __name__ == "__main__":
    main()
