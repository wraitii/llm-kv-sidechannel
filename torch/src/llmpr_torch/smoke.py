"""Load Qwen, attach LoRA, and perform one answer-only optimizer step."""
from __future__ import annotations

import argparse
import json
import platform

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .attention import configure_fixed_swa
from .devices import select_device, training_dtype
from .lora import attach_lora
from .state_data import make_counterfactual_pair
from .tokenization import tokenize_episode, validate_counterfactual_pair


BACKGROUND = ("The rain passed over the old house while the guests spoke quietly. "
              "No one paid much attention to the objects arranged in the library. ") * 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--window", type=int,
                        help="enable fixed sliding attention in every layer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    dtype = training_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pair = make_counterfactual_pair(BACKGROUND, background_id="smoke", seed=1337)
    encoded = [tokenize_episode(tokenizer, row, max_length=args.max_length) for row in pair]
    validate_counterfactual_pair(*encoded)

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if args.window is not None:
        config = configure_fixed_swa(config, args.window)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, config=config, dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = attach_lora(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "dtype": str(dtype), "model": args.model,
        "revision": args.revision, "prompt_tokens": encoded[0].prompt_length,
        "window": args.window,
        "trainable_parameters": trainable, "total_parameters": total,
        "lora_modules": len(model.targeted_module_names),
    }
    if not args.dry_run:
        row = encoded[0]
        input_ids = torch.tensor([row.input_ids], device=device)
        labels = torch.tensor([row.labels], device=device)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=1e-4)
        model.train()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        report["loss"] = float(loss.detach().cpu())
        if device.type == "cuda":
            report["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(report, indent=2, sort_keys=True))
