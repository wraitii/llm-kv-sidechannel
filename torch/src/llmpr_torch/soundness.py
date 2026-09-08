"""Model/backend semantic checks, independent of capacity benchmarking."""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .attention import configure_fixed_swa, dense_causal_mask, qwen_mask_mapping
from .devices import select_device, training_dtype


@torch.no_grad()
def cached_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    cache = None
    rows = []
    for position in range(tokens.shape[1]):
        output = model(
            input_ids=tokens[:, position:position + 1], past_key_values=cache,
            use_cache=True, cache_position=torch.tensor([position], device=tokens.device))
        cache = output.past_key_values
        rows.append(output.logits[:, -1:])
    return torch.cat(rows, dim=1)


@torch.no_grad()
def run_checks(model, tokens: torch.Tensor, window: int, swa_model=None) -> dict[str, float | bool]:
    model.eval()
    full = model(input_ids=tokens, use_cache=False).logits
    incremental = cached_logits(model, tokens)
    cache_error = float((full - incremental).abs().max().cpu())
    dense_mask = qwen_mask_mapping(dense_causal_mask(
        tokens.shape[1], windows=window, device=tokens.device,
        dtype=next(model.parameters()).dtype))
    dense = model(input_ids=tokens, attention_mask=dense_mask, use_cache=False).logits
    # Token-by-token with a bounded raw replay is an independent SWA reference.
    replay = []
    for end in range(1, tokens.shape[1] + 1):
        start = max(0, end - window)
        positions = torch.arange(start, end, device=tokens.device)[None]
        output = model(input_ids=tokens[:, start:end], position_ids=positions,
                       use_cache=False, logits_to_keep=1)
        replay.append(output.logits[:, -1:])
    replay_logits = torch.cat(replay, dim=1)
    # Only positions before eviction should be identical: after eviction replay
    # intentionally removes contextual information carried by surviving KVs.
    no_eviction_error = float((dense[:, :window] - replay_logits[:, :window]).abs().max().cpu())
    report = {
        "finite": bool(torch.isfinite(full).all()),
        "cached_full_max_abs_error": cache_error,
        "no_eviction_restart_max_abs_error": no_eviction_error,
    }
    if swa_model is not None:
        swa_model.eval()
        native = swa_model(input_ids=tokens, use_cache=False).logits
        report["native_swa_dense_max_abs_error"] = float(
            (native - dense).abs().max().cpu())
        report["native_swa_evicted_tokens"] = max(0, tokens.shape[1] - window)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    device = select_device(args.device)
    if args.length <= args.window:
        parser.error("--length must exceed --window so the native SWA check exercises eviction")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=training_dtype(device), attn_implementation="eager").to(device)
    swa_config = configure_fixed_swa(AutoConfig.from_pretrained(args.model), args.window)
    swa_model = AutoModelForCausalLM.from_pretrained(
        args.model, config=swa_config, dtype=training_dtype(device),
        attn_implementation="sdpa").to(device)
    vocab = model.config.vocab_size
    tokens = torch.randint(vocab, (1, args.length), generator=torch.Generator().manual_seed(7)).to(device)
    report = run_checks(model, tokens, args.window, swa_model=swa_model)
    report.update({"model": args.model, "device": str(device), "length": args.length,
                   "window": args.window, "atol": args.atol})
    report["passed"] = (report["finite"]
                        and report["cached_full_max_abs_error"] <= args.atol
                        and report["no_eviction_restart_max_abs_error"] <= args.atol
                        and report["native_swa_dense_max_abs_error"] <= args.atol
                        and report["native_swa_evicted_tokens"] > 0)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
