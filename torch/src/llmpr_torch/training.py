"""Correctness-first answer-only Qwen/LoRA training runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from .devices import select_device, training_dtype
from .evaluation import load_episodes, parse_policy, Policy
from .lora import attach_lora
from .policies import (
    FullAttention, FixedSWA, StreamingLog, VariableSWA,
    additive_from_visibility, causal_visibility, streaming_visibility,
)
from .tokenization import TokenizedEpisode, tokenize_episode
from .scored import ScoredRetention, enable_scored_retention


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    required = {"model", "data", "run_dir", "steps", "learning_rate", "batch_size", "grad_accum", "policy"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"training config is missing: {sorted(missing)}")
    if min(config["steps"], config["batch_size"], config["grad_accum"]) < 1:
        raise ValueError("steps, batch_size, and grad_accum must be positive")
    return config


def make_mask(rows: list[TokenizedEpisode], policy: Policy, device, dtype,
              windows: list[int] | None = None) -> torch.Tensor:
    length = max(len(row.input_ids) for row in rows)
    masks = []
    if isinstance(policy, VariableSWA) and (windows is None or len(windows) != len(rows)):
        raise ValueError("variable SWA requires one sampled window per row")
    for index, row in enumerate(rows):
        active = len(row.input_ids)
        if isinstance(policy, FullAttention):
            visible = causal_visibility(active)
        elif isinstance(policy, FixedSWA):
            visible = causal_visibility(active, policy.window)
        elif isinstance(policy, VariableSWA):
            visible = causal_visibility(active, windows[index])
        elif isinstance(policy, StreamingLog):
            visible = streaming_visibility(active, policy)
        elif isinstance(policy, ScoredRetention):
            visible = causal_visibility(active)
        else:
            raise TypeError(type(policy).__name__)
        padded = np.zeros((length, length), dtype=np.bool_)
        padded[:active, :active] = visible[0]
        # A padded query must have one finite key to avoid an all-masked SDPA row.
        padded[active:, 0] = True
        masks.append(padded)
    return additive_from_visibility(np.stack(masks), device=device, dtype=dtype)


def collate(rows: list[TokenizedEpisode], policy: Policy, device, dtype,
            windows: list[int] | None = None):
    length = max(len(row.input_ids) for row in rows)
    input_ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(rows):
        input_ids[index, :len(row.input_ids)] = torch.tensor(row.input_ids, device=device)
        labels[index, :len(row.labels)] = torch.tensor(row.labels, device=device)
    return input_ids, labels, {
        "full_attention": make_mask(rows, policy, device, dtype, windows=windows)}


def cosine_scheduler(optimizer, warmup: int, total: int):
    def ratio(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-after", type=int,
                        help="stop after this absolute step without changing the saved config")
    args = parser.parse_args()
    config = load_config(args.config)
    policy = parse_policy(config["policy"])
    device = select_device(args.device)
    dtype = training_dtype(device)
    seed = int(config.get("seed", 1337))
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = torch.Generator(device="cpu").manual_seed(seed + 1)
    generators = {"sampler": sampler}

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision", "main"))
    episodes = list(load_episodes(Path(config["data"])))
    encoded = [tokenize_episode(tokenizer, episode, max_length=int(config.get("max_length", 8192)))
               for episode in episodes]
    if not encoded:
        raise ValueError("training data is empty")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config.get("revision", "main"), dtype=dtype,
        attn_implementation="eager").to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if isinstance(policy, ScoredRetention):
        model = enable_scored_retention(model, policy)
    model = attach_lora(model, rank=int(config.get("lora_rank", 16)),
                        alpha=int(config.get("lora_alpha", 32)))
    if isinstance(policy, ScoredRetention):
        for name, parameter in model.named_parameters():
            if ".retention_scorer." in name:
                parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]), betas=(0.9, 0.95),
        weight_decay=float(config.get("weight_decay", 0.0)))
    scheduler = cosine_scheduler(optimizer, int(config.get("warmup_steps", 0)), int(config["steps"]))
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    start_step = micro_step = tokens_seen = 0
    if args.resume:
        checkpoint = latest_checkpoint(run_dir) if args.resume == "auto" else Path(args.resume)
        counters = load_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
            expected_config=config, generators=generators, map_location=device)
        start_step, micro_step, tokens_seen = (counters[name] for name in ("step", "micro_step", "tokens_seen"))
        print(json.dumps({"event": "resume", "checkpoint": str(checkpoint), **counters}), flush=True)
    metrics = run_dir / "metrics.jsonl"
    started = time.time()
    model.train()
    final_step = min(int(config["steps"]), args.stop_after or int(config["steps"]))
    if final_step < start_step:
        parser.error("--stop-after precedes the resumed checkpoint")
    for step in range(start_step + 1, final_step + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(int(config["grad_accum"])):
            indices = torch.randint(len(encoded), (int(config["batch_size"]),), generator=sampler).tolist()
            batch = [encoded[index] for index in indices]
            windows = None
            if isinstance(policy, VariableSWA):
                windows = torch.randint(
                    policy.minimum, policy.maximum + 1, (len(batch),),
                    generator=sampler).tolist()
            input_ids, labels, mask = collate(
                batch, policy, device, dtype, windows=windows)
            loss = model(input_ids=input_ids, labels=labels, attention_mask=mask, use_cache=False).loss
            (loss / int(config["grad_accum"])).backward()
            accumulated += float(loss.detach().cpu()) / int(config["grad_accum"])
            tokens_seen += int((labels != -100).sum())
            micro_step += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
        optimizer.step()
        scheduler.step()
        record = {"event": "train", "step": step, "micro_step": micro_step,
                  "loss": accumulated, "grad_norm": float(grad_norm),
                  "learning_rate": scheduler.get_last_lr()[0], "tokens_seen": tokens_seen,
                  "elapsed_s": time.time() - started}
        with metrics.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        save_every = int(config.get("save_every", 100))
        if step % save_every == 0 or step == final_step:
            save_checkpoint(
                run_dir / f"checkpoint-{step:07d}.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=None, step=step, micro_step=micro_step,
                tokens_seen=tokens_seen, config=config, generators=generators)


if __name__ == "__main__":
    main()
