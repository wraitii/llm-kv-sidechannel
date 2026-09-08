"""Destructive-in-memory capacity probe; never reads or writes checkpoints."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .devices import select_device, training_dtype
from .lora import attach_lora
from .evaluation import parse_policy
from .policies import (
    FullAttention, FixedSWA, StreamingLog, VariableSWA,
    additive_from_visibility, causal_visibility, streaming_visibility,
)
from .scored import ScoredRetention, enable_scored_retention


def hardware_report(device: torch.device) -> dict:
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "device": str(device),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        report.update({"gpu": torch.cuda.get_device_name(device),
                       "vram_bytes": torch.cuda.get_device_properties(device).total_memory})
        try:
            report["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,name,memory.total,power.limit",
                 "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            report["nvidia_smi_error"] = str(error)
    elif device.type == "mps":
        report["mps_recommended_working_set_bytes"] = int(
            torch.mps.recommended_max_memory())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--microbatches", default="1,2,4")
    parser.add_argument("--effective-batch", type=int, required=True,
                        help="fixed effective batch; must be divisible by every microbatch")
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--vocab-limit", type=int,
                        help="sample IDs below this value when testing a custom tokenizer")
    parser.add_argument("--policy", default="full",
                        help="full, swa:N, variable-swa:MIN-MAX, log:R+M, or scored:R+M")
    args = parser.parse_args()
    if args.updates < 1 or args.warmup < 0 or args.effective_batch < 1:
        parser.error("updates/effective-batch must be positive and warmup non-negative")
    microbatches = [int(value) for value in args.microbatches.split(",")]
    if any(value < 1 or args.effective_batch % value for value in microbatches):
        parser.error("every microbatch must be positive and exactly divide --effective-batch")
    device = select_device(args.device)
    dtype = training_dtype(device)
    policy = parse_policy(args.policy)
    print(json.dumps({"event": "machine", **hardware_report(device)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if isinstance(policy, ScoredRetention):
        model = enable_scored_retention(model, policy)
    model = attach_lora(model)
    if isinstance(policy, ScoredRetention):
        for name, parameter in model.named_parameters():
            if ".retention_scorer." in name:
                parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    vocab = min(model.config.vocab_size, args.vocab_limit or model.config.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    for length in map(int, args.lengths.split(",")):
        for batch_size in microbatches:
            grad_accum = args.effective_batch // batch_size
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            elif device.type == "mps":
                torch.mps.empty_cache()
            durations = []
            status = "ok"
            error = None
            try:
                for update in range(args.warmup + args.updates):
                    optimizer.zero_grad(set_to_none=True)
                    started = time.perf_counter()
                    for _ in range(grad_accum):
                        ids = torch.randint(
                            vocab, (batch_size, length), generator=generator).to(device)
                        labels = ids.clone()
                        if isinstance(policy, FullAttention) or isinstance(policy, ScoredRetention):
                            visible = causal_visibility(length)
                        elif isinstance(policy, FixedSWA):
                            visible = causal_visibility(length, policy.window)
                        elif isinstance(policy, VariableSWA):
                            windows = torch.randint(
                                policy.minimum, policy.maximum + 1, (batch_size,),
                                generator=generator).numpy()
                            visible = causal_visibility(length, windows)
                        else:
                            visible = streaming_visibility(length, policy)
                        mask = additive_from_visibility(
                            np.broadcast_to(visible, (batch_size, length, length)),
                            device=device, dtype=dtype)
                        loss = model(input_ids=ids, labels=labels,
                                     attention_mask={"full_attention": mask}, use_cache=False).loss
                        (loss / grad_accum).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    duration = time.perf_counter() - started
                    if update >= args.warmup:
                        durations.append(duration)
            except torch.OutOfMemoryError as caught:
                status, error = "oom", str(caught)
            record = {"event": "capacity", "status": status, "length": length,
                      "policy": args.policy,
                      "microbatch": batch_size, "grad_accum": grad_accum,
                      "effective_batch": args.effective_batch,
                      "complete_updates": len(durations),
                      "mean_step_s": sum(durations) / len(durations) if durations else None,
                      "tokens_per_s": (length * args.effective_batch * len(durations) / sum(durations)) if durations else None,
                      "error": error}
            if device.type == "cuda":
                record.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            elif device.type == "mps":
                record.update({"current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                               "driver_allocated_bytes": int(torch.mps.driver_allocated_memory())})
            print(json.dumps(record), flush=True)
            if status == "oom" and device.type == "cuda":
                # Do not trust later allocator measurements in this process.
                return


if __name__ == "__main__":
    main()
