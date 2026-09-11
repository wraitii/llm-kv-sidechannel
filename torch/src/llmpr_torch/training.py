"""Correctness-first answer-only Qwen/LoRA training runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from .devices import select_device, training_dtype
from .evaluation import load_episodes, parse_policy, Policy
from .lora import attach_lora
from .monitoring import SystemMonitor
from .policies import (
    FullAttention, FixedSWA, MementoOnly, StreamingLog, VariableSWA,
    additive_from_visibility, causal_visibility, memento_visibility,
    retain_memory_positions, streaming_visibility,
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
    if float(config.get("prompt_loss_weight", 0.0)) < 0:
        raise ValueError("prompt_loss_weight must be nonnegative")
    prompt_loss_fraction = float(config.get("prompt_loss_fraction", 0.0))
    memory_loss_fraction = float(config.get("memory_loss_fraction", 0.0))
    if not 0.0 <= prompt_loss_fraction < 1.0:
        raise ValueError("prompt_loss_fraction must be at least zero and less than one")
    if not 0.0 <= memory_loss_fraction < 1.0:
        raise ValueError("memory_loss_fraction must be at least zero and less than one")
    if prompt_loss_fraction + memory_loss_fraction >= 1.0:
        raise ValueError("prompt and memory loss fractions must sum to less than one")
    if prompt_loss_fraction and config.get("prompt_loss_weight", 0.0):
        raise ValueError("prompt_loss_fraction and prompt_loss_weight are mutually exclusive")
    if memory_loss_fraction and config.get("prompt_loss_weight", 0.0):
        raise ValueError("memory_loss_fraction and prompt_loss_weight are mutually exclusive")
    full_lm_probability = float(config.get("full_attention_lm_probability", 0.0))
    task_probability = float(config.get("task_probability", 1.0 - full_lm_probability))
    if not 0.0 <= full_lm_probability <= 1.0:
        raise ValueError("full_attention_lm_probability must be between zero and one")
    if not 0.0 <= task_probability <= 1.0 or task_probability + full_lm_probability > 1.0:
        raise ValueError("task_probability and full_attention_lm_probability must be valid and sum to at most one")
    if float(config.get("full_attention_lm_weight", 1.0)) < 0:
        raise ValueError("full_attention_lm_weight must be nonnegative")
    if full_lm_probability and (config.get("prompt_loss_weight", 0.0)
                                or prompt_loss_fraction or memory_loss_fraction):
        raise ValueError(
            "mixed prompt loss and full_attention_lm_probability are mutually exclusive")
    return config


def mixed_causal_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_lengths: list[int],
    prompt_loss_weight: float = 0.0,
    prompt_loss_fraction: float = 0.0,
    memory_token_spans: list[tuple[tuple[int, int], ...]] | None = None,
    memory_loss_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine independently averaged answer, ordinary-prompt, and memory losses."""
    if prompt_loss_weight and (prompt_loss_fraction or memory_loss_fraction):
        raise ValueError("choose either prompt_loss_weight or normalized loss fractions")
    shift_logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    answer_mask = labels[:, 1:] != -100
    prompt_mask = torch.zeros_like(answer_mask)
    for row, prompt_length in enumerate(prompt_lengths):
        # targets[:, i] is the original input token at position i + 1.
        prompt_mask[row, :max(0, prompt_length - 1)] = True
    memory_mask = torch.zeros_like(answer_mask)
    if memory_loss_fraction:
        if memory_token_spans is None:
            raise ValueError("memory loss requires memory token spans")
        for row, spans in enumerate(memory_token_spans):
            if not spans:
                raise ValueError("memory loss requires at least one memory span per row")
            for start, end in spans:
                # Loss index i predicts the input token at i + 1. Exclude the
                # whole inserted span from ordinary LM, but supervise only the
                # copied tokens after its sentinel as memory targets.
                prompt_mask[row, max(0, start - 1):end] = False
                memory_mask[row, start:end] = True
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2), targets, reduction="none")
    answer_loss = token_losses[answer_mask].mean()
    prompt_loss = token_losses[prompt_mask].mean()
    memory_loss = (token_losses[memory_mask].mean() if memory_loss_fraction
                   else answer_loss.detach().new_zeros(()))
    if prompt_loss_fraction or memory_loss_fraction:
        answer_fraction = 1.0 - prompt_loss_fraction - memory_loss_fraction
        combined = (answer_fraction * answer_loss
                    + float(prompt_loss_fraction) * prompt_loss
                    + float(memory_loss_fraction) * memory_loss)
    else:
        combined = answer_loss + float(prompt_loss_weight) * prompt_loss
    return combined, answer_loss, prompt_loss, memory_loss


def prompt_causal_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: list[int],
) -> torch.Tensor:
    """Average next-token loss only over tokens belonging to the prompt."""
    shift_logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    prompt_mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, prompt_length in enumerate(prompt_lengths):
        prompt_mask[row, :max(0, prompt_length - 1)] = True
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2), targets, reduction="none")
    return token_losses[prompt_mask].mean()


def make_mask(rows: list[TokenizedEpisode], policy: Policy, device, dtype,
              windows: list[int] | None = None,
              retain_memory_tokens: bool = False) -> torch.Tensor:
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
        elif isinstance(policy, MementoOnly):
            if not row.memory_token_spans:
                raise ValueError("Memento policy requires annotated memory spans")
            visible = memento_visibility(active, row.memory_token_spans)
        elif isinstance(policy, ScoredRetention):
            visible = causal_visibility(active)
        else:
            raise TypeError(type(policy).__name__)
        if retain_memory_tokens:
            if not row.memory_token_spans:
                raise ValueError("memory retention requires annotated memory spans")
            visible = retain_memory_positions(visible, row.memory_token_spans)
        padded = np.zeros((length, length), dtype=np.bool_)
        padded[:active, :active] = visible[0]
        # A padded query must have one finite key to avoid an all-masked SDPA row.
        padded[active:, 0] = True
        masks.append(padded)
    return additive_from_visibility(np.stack(masks), device=device, dtype=dtype)


def collate(rows: list[TokenizedEpisode], policy: Policy, device, dtype,
            windows: list[int] | None = None,
            retain_memory_tokens: bool = False):
    length = max(len(row.input_ids) for row in rows)
    input_ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(rows):
        input_ids[index, :len(row.input_ids)] = torch.tensor(row.input_ids, device=device)
        labels[index, :len(row.labels)] = torch.tensor(row.labels, device=device)
    return input_ids, labels, {
        "full_attention": make_mask(
            rows, policy, device, dtype, windows=windows,
            retain_memory_tokens=retain_memory_tokens)}


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
    retain_memory_tokens = bool(config.get("retain_memory_tokens", False))
    if retain_memory_tokens and isinstance(policy, ScoredRetention):
        raise ValueError("memory retention is not yet composable with learned scored retention")
    if retain_memory_tokens and isinstance(policy, MementoOnly):
        raise ValueError("Memento policy retains memory intrinsically; remove retain_memory_tokens")
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
    monitor_interval = float(config.get("monitor_interval_s", 10))
    if monitor_interval < 0:
        raise ValueError("monitor_interval_s must be nonnegative")
    monitor = SystemMonitor(run_dir, monitor_interval) if monitor_interval else None
    if monitor:
        monitor.start()
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
        step_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        accumulated = answer_accumulated = prompt_accumulated = memory_accumulated = 0.0
        task_micro_steps = prompt_lm_micro_steps = full_lm_micro_steps = constrained_lm_micro_steps = 0
        context_tokens = answer_tokens = lm_tokens = 0
        for _ in range(int(config["grad_accum"])):
            indices = torch.randint(len(encoded), (int(config["batch_size"]),), generator=sampler).tolist()
            batch = [encoded[index] for index in indices]
            full_lm_probability = float(config.get("full_attention_lm_probability", 0.0))
            task_probability = float(config.get("task_probability", 1.0 - full_lm_probability))
            route = torch.rand((), generator=sampler).item()
            full_lm_update = route < full_lm_probability
            task_update = full_lm_probability <= route < full_lm_probability + task_probability
            active_policy = FullAttention() if full_lm_update else policy
            windows = None
            if isinstance(active_policy, VariableSWA):
                windows = torch.randint(
                    active_policy.minimum, active_policy.maximum + 1, (len(batch),),
                    generator=sampler).tolist()
            input_ids, labels, mask = collate(
                batch, active_policy, device, dtype, windows=windows,
                retain_memory_tokens=retain_memory_tokens and not full_lm_update)
            prompt_loss_weight = float(config.get("prompt_loss_weight", 0.0))
            prompt_loss_fraction = float(config.get("prompt_loss_fraction", 0.0))
            memory_loss_fraction = float(config.get("memory_loss_fraction", 0.0))
            if not task_update:
                output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
                prompt_loss = prompt_causal_loss(
                    output.logits, input_ids, [row.prompt_length for row in batch])
                loss = ((float(config.get("full_attention_lm_weight", 1.0))
                         if full_lm_update else 1.0) * prompt_loss)
                answer_loss = loss.detach().new_zeros(())
                memory_loss = loss.detach().new_zeros(())
                if full_lm_update:
                    full_lm_micro_steps += 1
                else:
                    constrained_lm_micro_steps += 1
                prompt_lm_micro_steps += 1
            elif prompt_loss_weight or prompt_loss_fraction or memory_loss_fraction:
                output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
                loss, answer_loss, prompt_loss, memory_loss = mixed_causal_loss(
                    output.logits, input_ids, labels,
                    [row.prompt_length for row in batch], prompt_loss_weight,
                    prompt_loss_fraction,
                    [row.memory_token_spans for row in batch], memory_loss_fraction)
                prompt_lm_micro_steps += 1
            else:
                loss = answer_loss = model(
                    input_ids=input_ids, labels=labels,
                    attention_mask=mask, use_cache=False).loss
                prompt_loss = loss.detach().new_zeros(())
                memory_loss = loss.detach().new_zeros(())
            if task_update:
                task_micro_steps += 1
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / int(config["grad_accum"])).backward()
            accumulated += float(loss.detach().cpu()) / int(config["grad_accum"])
            answer_accumulated += float(answer_loss.detach().cpu()) / int(config["grad_accum"])
            prompt_accumulated += float(prompt_loss.detach().cpu()) / int(config["grad_accum"])
            memory_accumulated += float(memory_loss.detach().cpu()) / int(config["grad_accum"])
            batch_answer_tokens = int((labels != -100).sum())
            if task_update:
                answer_tokens += batch_answer_tokens
                tokens_seen += batch_answer_tokens
                if prompt_loss_weight or prompt_loss_fraction or memory_loss_fraction:
                    batch_lm_tokens = sum(max(0, row.prompt_length - 1) for row in batch)
                    lm_tokens += batch_lm_tokens
                    tokens_seen += batch_lm_tokens
            else:
                batch_lm_tokens = sum(max(0, row.prompt_length - 1) for row in batch)
                lm_tokens += batch_lm_tokens
                tokens_seen += batch_lm_tokens
            context_tokens += sum(len(row.input_ids) for row in batch)
            micro_step += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_s = time.perf_counter() - step_started
        eta_s = step_s * (final_step - step)
        accumulation_steps = int(config["grad_accum"])
        reported_answer_loss = (
            answer_accumulated * accumulation_steps / task_micro_steps
            if task_micro_steps else 0.0)
        reported_prompt_loss = (
            prompt_accumulated * accumulation_steps / prompt_lm_micro_steps
            if prompt_lm_micro_steps else 0.0)
        reported_memory_loss = memory_accumulated
        record = {"event": "train", "step": step, "micro_step": micro_step,
                  "loss": accumulated, "answer_loss": reported_answer_loss,
                  "prompt_lm_loss": reported_prompt_loss,
                  "memory_loss": reported_memory_loss,
                  "prompt_loss_weight": float(config.get("prompt_loss_weight", 0.0)),
                  "prompt_loss_fraction": float(config.get("prompt_loss_fraction", 0.0)),
                  "memory_loss_fraction": float(config.get("memory_loss_fraction", 0.0)),
                  "full_attention_lm_probability": float(
                      config.get("full_attention_lm_probability", 0.0)),
                  "task_probability": float(config.get(
                      "task_probability", 1.0 - config.get("full_attention_lm_probability", 0.0))),
                  "full_attention_lm_weight": float(config.get("full_attention_lm_weight", 1.0)),
                  "task_micro_steps": task_micro_steps,
                  "prompt_lm_micro_steps": prompt_lm_micro_steps,
                  "full_attention_lm_micro_steps": full_lm_micro_steps,
                  "constrained_lm_micro_steps": constrained_lm_micro_steps,
                  "grad_norm": float(grad_norm),
                  "learning_rate": scheduler.get_last_lr()[0], "tokens_seen": tokens_seen,
                  "context_tokens": context_tokens, "answer_tokens": answer_tokens,
                  "lm_tokens": lm_tokens,
                  "step_s": step_s,
                  "context_tokens_per_s": context_tokens / step_s,
                  "answer_tokens_per_s": answer_tokens / step_s,
                  "eta_s": eta_s, "elapsed_s": time.time() - started}
        if device.type == "cuda":
            record.update({
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            })
        with metrics.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        save_every = int(config.get("save_every", 100))
        if step % save_every == 0 or step == final_step:
            save_checkpoint(
                run_dir / f"checkpoint-{step:07d}.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=None, step=step, micro_step=micro_step,
                tokens_seen=tokens_seen, config=config, generators=generators)
    if monitor:
        monitor.close()


if __name__ == "__main__":
    main()
