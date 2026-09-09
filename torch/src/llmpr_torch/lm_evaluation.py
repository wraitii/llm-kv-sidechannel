"""Evaluate clean PG-19 next-token loss on deterministic held-out windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .devices import select_device, training_dtype
from .evaluation import parse_policy, policy_visibility
from .lora import attach_lora
from .policies import VariableSWA, additive_from_visibility


def clean_windows(tokenizer, path: Path, length: int, count: int, seed: int):
    """Yield deterministic interior token windows from distinct raw books."""
    with path.open() as handle:
        for index, line in enumerate(handle):
            if index >= count:
                break
            row = json.loads(line)
            text = row["text"]
            span = min(len(text), max(length * 8, 4000))
            available = max(0, len(text) - span)
            digest = hashlib.sha256(f"{row.get('id', index)}\0{seed}".encode()).digest()
            start = int.from_bytes(digest[:8], "big") % (available + 1)
            if start:
                space = text.find(" ", start, min(len(text), start + 200))
                if space >= 0:
                    start = space + 1
            ids = tokenizer(text[start:start + span], add_special_tokens=False)["input_ids"]
            ids = ids[:length]
            if len(ids) >= 2:
                yield tuple(ids)


def load_model(model_path: str, checkpoint: Path | None, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=training_dtype(device), attn_implementation="eager").to(device).eval()
    if payload:
        config = payload["config"]
        model = attach_lora(model, rank=int(config.get("lora_rank", 16)),
                            alpha=int(config.get("lora_alpha", 32)))
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in payload["model"].items():
                parameters[name].copy_(value.to(device))
    return model


@torch.no_grad()
def window_nll(model, ids: tuple[int, ...], policy, device: torch.device) -> tuple[float, int]:
    tokens = torch.tensor([ids], device=device)
    positions = tuple(range(len(ids)))
    mask = additive_from_visibility(
        policy_visibility(positions, policy), device=device,
        dtype=next(model.parameters()).dtype)
    logits = model(input_ids=tokens, attention_mask={"full_attention": mask},
                   use_cache=False).logits[:, :-1].float()
    loss = F.cross_entropy(logits.transpose(1, 2), tokens[:, 1:], reduction="sum")
    return float(loss.cpu()), len(ids) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--books", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--policies", default="full")
    args = parser.parse_args()
    if args.length < 2 or args.books < 1:
        parser.error("length must be at least 2 and books must be positive")
    policies = [parse_policy(value) for value in args.policies.split(",")]
    if any(isinstance(policy, VariableSWA) for policy in policies):
        parser.error("evaluate variable SWA checkpoints at fixed windows")
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    windows = list(clean_windows(tokenizer, args.data, args.length, args.books, args.seed))
    model = load_model(args.model, args.checkpoint, device)
    for policy in policies:
        total_loss = 0.0
        total_tokens = 0
        for ids in windows:
            loss, tokens = window_nll(model, ids, policy, device)
            total_loss += loss
            total_tokens += tokens
        nll = total_loss / total_tokens
        print(json.dumps({"policy": policy.kind, "policy_config": policy.__dict__,
                          "books": len(windows), "tokens": total_tokens,
                          "nll_per_token": nll, "perplexity": math.exp(nll)}))


if __name__ == "__main__":
    main()
