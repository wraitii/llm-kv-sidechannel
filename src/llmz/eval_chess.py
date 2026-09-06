"""Small sampled-decoding evaluation for chess FEN outputs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .generate import decode_generated, sample_token
from .board_eval import score_board_outputs
from .tokenizer import EOS, PairTokenizer
from .runtime import load_config, model_and_tokenizer
from .train import latest_checkpoint, load_model_checkpoint
from .model import PrefixLM
from .transport import RecursiveCarrierPolicy, policy_from_config


def generate_batch(model: PrefixLM, tokenizer: PairTokenizer, rows: list[dict],
                   max_source_tokens: int, max_new_tokens: int,
                   temperature: float, top_p: float,
                   sliding_window: int | None = None,
                   carrier_policy=None, transport=False, cache_mode="preserve",
                   seed=1337, top_k=0) -> list[str]:
    from .inference import prepare_prefix, DecodeSession, row_seed
    if not rows:
        return []
    prefix = prepare_prefix(tokenizer, rows, max_source_tokens, carrier_policy, transport, seed)
    session = DecodeSession(model, *prefix, window=sliding_window, mode=cache_mode)
    generated = [[] for _ in rows]
    finished = np.zeros(len(rows), dtype=bool)
    rngs = [np.random.default_rng(row_seed(row, seed)) for row in rows]
    for step in range(max_new_tokens):
        mx.eval(session.logits, list(session.cache))
        logits_np = np.asarray(session.logits)
        next_ids = np.full(len(rows), EOS, dtype=np.int32)
        for index in np.flatnonzero(~finished):
            next_ids[index] = sample_token(logits_np[index, 0], rngs[index], temperature, top_p, top_k)
            if next_ids[index] == EOS:
                finished[index] = True
            else:
                generated[index].append(int(next_ids[index]))
        if finished.all() or step + 1 == max_new_tokens:
            break
        session.advance(mx.array(tokenizer.target_local_to_model(next_ids.tolist()))[:, None])
    return [decode_generated(tokenizer, ids) for ids in generated]


def teacher_forced_nll(model, tokenizer, rows, max_source_tokens, sliding_window=None,
                       policy=None, transport=False, cache_mode="preserve", seed=1337):
    """Per-example NLL on identical reference continuations for paired interventions."""
    from .inference import prepare_prefix, DecodeSession
    session = DecodeSession(model, *prepare_prefix(tokenizer, rows, max_source_tokens,
                                                   policy, transport, seed),
                            window=sliding_window, mode=cache_mode)
    targets = [tokenizer.encode_target_local(row["code"]) + [EOS] for row in rows]
    sums = np.zeros(len(rows))
    for step in range(max(map(len, targets))):
        ids = [target[step] if step < len(target) else EOS for target in targets]
        log_probs = session.logits[:, 0].astype(mx.float32)
        log_probs = log_probs - mx.logsumexp(log_probs, axis=-1, keepdims=True)
        losses = -np.asarray(log_probs[mx.arange(len(rows)), mx.array(ids)])
        sums += losses * np.array([step < len(target) for target in targets])
        if step + 1 < max(map(len, targets)):
            session.advance(mx.array(tokenizer.target_local_to_model(ids))[:, None])
    return sums, np.array(list(map(len, targets)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperatures", default="0.3,0.7,1.0")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--windows", default=None,
                        help="comma-separated causal windows; use full for no limit")
    parser.add_argument("--no-carriers", action="store_true",
                        help="evaluate without memento carrier tokens even if "
                             "the config trains with them (the plain-causal "
                             "condition from the escape share)")
    parser.add_argument("--transport", action="store_true", help="apply configured eviction spans during prefill and decode")
    parser.add_argument("--cache-modes", default="preserve", help="comma-separated preserve,restart,restart-each")
    parser.add_argument("--teacher-forced", action="store_true", help="also measure paired reference-continuation NLL")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--disable-scoring", action="store_true", help="evaluate a scored checkpoint with ordinary full/SWA attention")
    parser.add_argument("--scored-recent-window", type=int,
                        help="override the scored model's recent window for a budget sweep")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.scored_recent_window is not None:
        if args.scored_recent_window < 1 or not cfg.get("scored_eviction"):
            parser.error("--scored-recent-window requires scored_eviction and a positive value")
        cfg["scored_eviction"] = {
            **cfg["scored_eviction"], "recent_window": args.scored_recent_window}
    model, tokenizer = model_and_tokenizer(cfg)
    carrier_policy = policy_from_config(cfg.get("transport_policy"))
    if args.no_carriers and isinstance(carrier_policy, RecursiveCarrierPolicy):
        if args.transport:
            parser.error("--no-carriers cannot apply an inserted-carrier transport policy")
        carrier_policy = None
    run_dir = Path(cfg["run_dir"])
    checkpoint = latest_checkpoint(run_dir) if args.checkpoint == "auto" else Path(args.checkpoint)
    state = load_model_checkpoint(checkpoint, model)
    if args.disable_scoring:
        for block in model.blocks:
            block.retention = None
    rows = []
    skipped = seen = 0
    sample_rng = np.random.default_rng(args.seed)
    if args.examples < 1:
        parser.error("--examples must be positive")
    with ((args.data_dir or Path(cfg["data_dir"])) / f"{args.split}.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if (len(tokenizer.encode_source(row["asm"])) > cfg["max_source_tokens"]
                    or len(tokenizer.encode_target_local(row["code"])) > cfg["max_target_tokens"]):
                skipped += 1
                continue
            seen += 1
            if len(rows) < args.examples:
                rows.append(row)
            else:
                index = int(sample_rng.integers(seen))
                if index < args.examples:
                    rows[index] = row
    temperatures = [float(value) for value in args.temperatures.split(",")]
    window_spec = args.windows or ("full" if cfg.get("scored_eviction") and not args.disable_scoring else "full,128,64,32,16")
    windows = [None if value == "full" else int(value)
               for value in window_spec.split(",")]
    if cfg.get("scored_eviction") and not args.disable_scoring and any(w is not None for w in windows):
        parser.error("scored eviction already defines its recent+memory budget; use --windows full, or --disable-scoring for SWA controls")
    if any(window is not None and window < 1 for window in windows):
        parser.error("window sizes must be positive")
    if any(window is not None for window in windows) and model.attention_mode != "causal":
        parser.error("sliding-window evaluation requires attention_mode=causal")
    modes = args.cache_modes.split(",")
    if not rows or args.batch_size < 1 or not 0 < args.top_p <= 1:
        parser.error("need eligible examples, positive batch size and 0 < top_p <= 1")
    if any(mode not in {"preserve", "restart", "restart-each"} for mode in modes):
        parser.error("unknown cache mode")
    if args.transport and carrier_policy is None:
        parser.error("--transport requires a transport policy")
    if model.attention_mode != "causal" and (args.transport or modes != ["preserve"]):
        parser.error("eviction/restart requires causal attention")
    nll_results = {}
    for window in windows:
        for mode in modes:
            if args.teacher_forced:
                sums, counts = [], []
                for start in range(0, len(rows), args.batch_size):
                    nll, count = teacher_forced_nll(
                        model, tokenizer, rows[start:start+args.batch_size], cfg["max_source_tokens"],
                        window, carrier_policy, args.transport, mode, args.seed)
                    sums.extend(nll.tolist())
                    counts.extend(count.tolist())
                nll_results[(window, mode)] = (np.array(sums), np.array(counts))
            for temperature in temperatures:
                outputs = []
                started = time.time()
                for start in range(0, len(rows), args.batch_size):
                    outputs.extend(generate_batch(
                        model, tokenizer, rows[start:start+args.batch_size],
                        cfg["max_source_tokens"], cfg["max_target_tokens"], temperature,
                        args.top_p, window, carrier_policy, args.transport, mode, args.seed))
                parsed, valid, exact, squares, metadata = score_board_outputs(
                    outputs, [row["code"] for row in rows])
                record = {"checkpoint": str(checkpoint), "step": state["step"],
                          "temperature": temperature, "sliding_window": window,
                          "cache_mode": mode, "transport": args.transport,
                          "scored_eviction": bool(cfg.get("scored_eviction")) and not args.disable_scoring,
                          "scored_recent_window": (cfg.get("scored_eviction") or {}).get("recent_window")
                          if not args.disable_scoring else None,
                          "scored_memory_tokens": (cfg.get("scored_eviction") or {}).get("memory_tokens")
                          if not args.disable_scoring else None,
                          "carriers": isinstance(carrier_policy, RecursiveCarrierPolicy),
                          "seed": args.seed, "split": args.split, "examples": len(rows), "skipped_overlong": skipped,
                          "parseable": parsed, "valid": valid, "exact": exact,
                          "mean_board_square_error": squares / parsed if parsed else None,
                          "mean_metadata_errors": metadata / parsed if parsed else None,
                          "elapsed_s": time.time() - started}
                if args.teacher_forced:
                    nll, count = nll_results[(window, mode)]
                    record["target_nll_per_token"] = float(nll.sum() / count.sum())
                print(json.dumps(record), flush=True)
        if args.teacher_forced and "preserve" in modes:
            base, counts = nll_results[(window, "preserve")]
            for mode in modes:
                if mode == "preserve":
                    continue
                other, _ = nll_results[(window, mode)]
                delta = (other - base) / counts
                print(json.dumps({"event": "paired_transport", "sliding_window": window,
                                  "cache_mode": mode, "examples": len(rows),
                                  "nll_increase_per_token": float((other-base).sum()/counts.sum()),
                                  "mean_example_nll_increase": float(delta.mean()),
                                  "example_delta_std": float(delta.std())}), flush=True)


if __name__ == "__main__":
    main()
