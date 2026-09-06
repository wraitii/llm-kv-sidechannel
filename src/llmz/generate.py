"""Generate a FEN from move history or a prepared dataset example."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

from .model import PrefixLM
from .tokenizer import EOS, PairTokenizer
from .runtime import load_config, model_and_tokenizer
from .train import latest_checkpoint, load_model_checkpoint


def sample_token(logits: np.ndarray, rng: np.random.Generator,
                 temperature: float, top_p: float, top_k: int) -> int:
    logits = logits.astype(np.float64, copy=True)
    # PAD, BOS, and <fen> are structural input tokens, never generated output.
    logits[:3] = -np.inf
    return sample_class(logits, rng, temperature, top_p, top_k)


def sample_class(logits: np.ndarray, rng: np.random.Generator,
                 temperature: float, top_p: float, top_k: int) -> int:
    logits = logits.astype(np.float64, copy=True)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature
    if top_k > 0 and top_k < logits.size:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        cumulative = np.cumsum(probabilities[order])
        keep = cumulative <= top_p
        keep[0] = True
        # Include the token that crosses the threshold.
        crossing = int(np.sum(keep))
        if crossing < len(keep):
            keep[crossing] = True
        filtered = np.zeros_like(probabilities)
        filtered[order[keep]] = probabilities[order[keep]]
        probabilities = filtered / filtered.sum()
    return int(rng.choice(logits.size, p=probabilities))


def decode_generated(tokenizer: PairTokenizer, ids: list[int]) -> str:
    return tokenizer.target.decode(ids)


def read_example(data_dir: Path, split: str, index: int) -> tuple[str, str]:
    path = data_dir / f"{split}.jsonl"
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                row = json.loads(line)
                return row["asm"], row["code"]
    raise IndexError(f"example {index} not found in {path}")


def board_ascii(text: str) -> str | None:
    """Return an ASCII board for a FEN-shaped output, even if position is illegal."""
    try:
        import chess
        fields = text.strip().split()
        if len(fields) != 6:
            return None
        board = chess.Board(None)
        board.set_board_fen(fields[0])
        return str(board)
    except (ImportError, ValueError):
        return None


def print_board(label: str, text: str) -> None:
    rendered = board_ascii(text)
    if rendered is not None:
        print(f"=== {label} board ===")
        print(rendered)


def generate(model: PrefixLM, tokenizer: PairTokenizer, asm: str,
             max_source_tokens: int, max_new_tokens: int,
             temperature: float, top_p: float, top_k: int,
             seed: int, policy=None, transport=False, sliding_window=None,
             cache_mode="preserve") -> tuple[str, int]:
    from .eval_chess import generate_batch
    output = generate_batch(model, tokenizer, [{"asm": asm}], max_source_tokens,
                            max_new_tokens, temperature, top_p, sliding_window,
                            policy, transport, cache_mode, seed, top_k)[0]
    return output, len(tokenizer.encode_source(asm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="auto",
                        help="checkpoint .npz, or 'auto' for run_dir/latest.json")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--asm", help="literal UCI move history")
    inputs.add_argument("--file", type=Path, help="move history input file")
    inputs.add_argument("--example", type=int, help="zero-based prepared dataset row")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--show-reference", action="store_true")
    parser.add_argument("--show-board", action="store_true",
                        help="render generated output as ASCII when it is a legal FEN")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 is greedy decoding")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--metal-cache-mib", type=int, default=512,
                        help="cap MLX's cache of unused Metal buffers (default: 512 MiB)")
    parser.add_argument("--transport", action="store_true")
    parser.add_argument("--sliding-window", type=int)
    parser.add_argument("--cache-mode", choices=["preserve", "restart", "restart-each"], default="preserve")
    parser.add_argument("--no-carriers", action="store_true")
    args = parser.parse_args()
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.metal_cache_mib < 0:
        parser.error("--metal-cache-mib must be non-negative")

    # MLX otherwise permits its pool of reusable, currently-unused Metal
    # buffers to grow as large as its overall memory limit.
    mx.set_cache_limit(args.metal_cache_mib * 1024**2)

    cfg = load_config(args.config)
    model, tokenizer = model_and_tokenizer(cfg)
    run_dir = Path(cfg["run_dir"])
    checkpoint = (latest_checkpoint(run_dir) if args.checkpoint == "auto"
                  else Path(args.checkpoint))
    state = load_model_checkpoint(checkpoint, model)

    reference = None
    if args.asm is not None:
        asm = args.asm
    elif args.file is not None:
        asm = args.file.read_text()
    elif args.example is not None:
        asm, reference = read_example(args.data_dir or Path(cfg["data_dir"]),
                                      args.split, args.example)
    else:
        asm = sys.stdin.read()
    if not asm:
        parser.error("assembly input is empty")

    from .transport import policy_from_config, RecursiveCarrierPolicy
    policy = policy_from_config(cfg.get("transport_policy"))
    if args.no_carriers and isinstance(policy, RecursiveCarrierPolicy):
        if args.transport:
            parser.error("--no-carriers cannot apply a carrier transport policy")
        policy = None
    if args.sliding_window is not None and args.sliding_window < 1:
        parser.error("--sliding-window must be positive")
    maximum = min(args.max_new_tokens or cfg["max_target_tokens"],
                  cfg["max_target_tokens"])
    output, source_tokens = generate(
        model, tokenizer, asm, cfg["max_source_tokens"], maximum,
        args.temperature, args.top_p, args.top_k, args.seed, policy, args.transport,
        args.sliding_window, args.cache_mode)
    print(f"checkpoint={checkpoint} step={state['step']} source_tokens={source_tokens}",
          file=sys.stderr)
    if reference is not None and args.show_reference:
        print("=== generated ===")
        print(output)
        if args.show_board:
            print_board("generated", output)
        print("=== reference ===")
        print(reference)
        if args.show_board:
            print_board("reference", reference)
    else:
        print(output)
        if args.show_board:
            print_board("generated", output)


if __name__ == "__main__":
    main()
