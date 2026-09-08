# PyTorch/Qwen runner

This is the CUDA-oriented implementation of the Qwen experiments. The existing
`src/llmz` package remains the MLX reference implementation.

The initial vertical slice provides deterministic state-probe construction and
a full-attention Qwen/LoRA smoke step. The correctness-first runner now supports
full attention, fixed SWA, streaming-log retention, and learned per-layer scored
retention. Memento masks and carrier tokens remain out of scope.

Sliding-window attention and KV restart are intentionally not approximated by
cache cropping. Evaluation reconstructs from surviving raw token IDs at their
original positions. This dense replay is deliberately slow and provides the
semantic reference for a later efficient cache backend.

## Training and exact resume

Training consumes state-episode JSONL produced by `llmpr-prepare-state` and uses
answer-only loss. A checkpoint contains every trainable parameter (LoRA and,
when enabled, retention scorers), optimizer/scheduler state, all process RNGs,
the named sampling generator, and counters. Resume rejects any config change.

```bash
uv run --locked llmpr-train --config configs/full.json
uv run --locked llmpr-train --config configs/full.json --resume
```

Policy strings are `full`, `swa:N`, `log:R+M`, and `scored:R+M`. Scored
retention assigns an immutable priority in each layer and uses an exact hard
selection in the forward pass with a straight-through scorer gradient.

## Evaluation sweep

```bash
uv run --locked llmpr-evaluate \
  --model models/Qwen3-1.7B-Base --checkpoint outputs/run/checkpoint-0001000.pt \
  --data data/state.jsonl \
  --policies full,swa:1024,swa:512,swa:256,swa:128,swa:64,log:128+128 \
  --restart-modes preserve,restart:answer,restart:32,restart:8,restart:1
```

For a scored checkpoint, add its native `scored:R+M` policy. Its preserved pass
freezes per-layer priorities; restart replay reuses them and never reranks from
reconstructed hidden states. `restart:N` rebuilds before positions divisible by
N, while `restart:answer` rebuilds once at the answer boundary.

## Machine soundness and capacity

These are separate commands. Soundness checks cache/full logits and the
no-eviction replay identity. Capacity performs complete LoRA updates and never
loads, writes, or resumes experiment checkpoints.

```bash
uv run --locked llmpr-soundness --model models/Qwen3-1.7B-Base --device cuda
uv run --locked llmpr-capacity --model models/Qwen3-1.7B-Base --device cuda \
  --policy full --lengths 2048,4096,8192,12288,16384 --microbatches 1,2,4
```

Run capacity once per policy/backend. The current streaming-log and scored
training paths use explicit quadratic reference storage; their measurements are
correctness baselines, not claims of bounded physical KV memory.

## Local smoke test

Use Python 3.12-3.13. The project is separately locked so the MLX environment is
not modified.

```bash
cd torch
uv sync --extra dev
uv run pytest -q
uv run llmpr-qwen-smoke --model Qwen/Qwen3-1.7B-Base --device auto
uv run llmpr-qwen-smoke --model Qwen/Qwen3-1.7B-Base --device auto --window 128
```

To keep model weights out of the global Hugging Face cache, download a snapshot
into `models/Qwen3-1.7B-Base` and pass that path as `--model`. The `models/`,
`data/`, and `outputs/` directories are ignored by Git. The tested upstream
commit and weight checksum are recorded in `model-snapshots.json`.

The last command downloads the model and performs one BF16/FP32 LoRA optimizer
step. On Apple Silicon, `auto` selects MPS; otherwise it selects CUDA or CPU.
Use `--dry-run` to validate tokenization and adapter attachment without a
backward pass.

## Vast.ai baseline

Use an RTX 50-series-compatible image with a driver capable of running the CUDA
13 libraries selected by the Linux portion of `uv.lock`. PyTorch 2.14 is pinned;
do not use an older template-provided Torch in preference to the locked wheel.
Record the image digest, NVIDIA driver, and smoke-command output in every run.
Do not rely on instance-local storage for checkpoints or result files.
Follow [`VAST_CHECKLIST.md`](VAST_CHECKLIST.md) on every new instance.
