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

## PG-19 dataset preparation

The preparation command streams real books from the `emozilla/pg19` Hugging
Face mirror. Streaming fetches only the Parquet shards needed to reach the
requested number of books; it does not materialize the full corpus. Install the
optional data dependency before the first run:

```bash
cd torch
uv sync --extra dev --extra data
```

### Small local dataset

The following produces 48 training episodes and 12 episodes in each evaluation
split: `books x context lengths x task levels x 2 counterfactual variants`.

```bash
uv run llmpr-prepare-pg19 \
  --model models/Qwen3-1.7B-Base \
  --output-dir data/pg19-local-1k \
  --context-lengths 1024 \
  --train-books 8 --validation-books 2 --test-books 2
```

The default task levels are `passcode_easy`, `passcode_hard`, `structured`, and
`natural`. Restrict
them with, for example, `--levels passcode,structured`. Every generated row is
a complete prompt, answer, and EOS that fits within its declared token budget.
Multiple budgets can be generated together:

```bash
uv run llmpr-prepare-pg19 \
  --model models/Qwen3-1.7B-Base \
  --output-dir data/pg19-2k-8k \
  --context-lengths 2048,4096,8192 \
  --train-books 256 --validation-books 50 --test-books 100
```

This larger example creates 4,608 training and 2,700 evaluation episodes. It is
a dataset construction example, not a claim that every policy and length fits
on a particular GPU. Run `llmpr-capacity` first, then omit lengths that do not
fit. When training on mixed lengths, set `max_length` to the largest generated
budget. Batches are padded to their longest row, so `batch_size: 1` or separate
runs per length avoid wasted padding during initial capacity measurements.

### Output and reproducibility

The output directory contains:

- `train.jsonl`, `validation.jsonl`, and `test.jsonl`: answer-supervised probe
  episodes consumed by `llmpr-train` and `llmpr-evaluate`;
- `pg19-{split}.jsonl`: the selected unmodified books, retained for future
  clean-text language-model evaluation;
- `manifest.json`: requested settings, task counts, and the immutable HF commit
  resolved from `--revision`.

PG-19's original book-level splits are preserved, so a book cannot cross from
training into evaluation. `--revision main` is convenient for exploration and
is resolved to an exact commit before data is read. For archival runs, pass a
known commit explicitly. Public access works without credentials; setting the
standard `HF_TOKEN` environment variable raises Hugging Face rate limits. Use
`--cache-dir PATH` to select a cache and `--no-streaming` only when intentionally
materializing a split.

Generation is deterministic for a given tokenizer, resolved dataset revision,
and seed. The command refuses to write into a non-empty output directory, so
choose a new directory or deliberately remove the old derived data before
rerunning it. The repository ignores `torch/data/` and `torch/outputs/`.

### Train and evaluate

The checked-in configs are the five current 3.8K experiment arms. They expect
the generated dataset at `data/pg19-3800-v2` and write to distinct directories
under `outputs/`:

```bash
uv run --locked llmpr-train --config configs/variable-swa-lm-task.json
uv run --locked llmpr-evaluate \
  --model models/Qwen3-1.7B-Base \
  --checkpoint outputs/variable-swa-lm-task/checkpoint-0000200.pt \
  --data data/pg19-3800-v2/validation.jsonl \
  --max-length 3800 \
  --policies full,swa:512,swa:256 \
  --restart-modes preserve,restart:answer,restart:512,restart:256
```

Set `task_probability` and `full_attention_lm_probability` to route updates
between answer-only probe loss, full-attention prompt LM, and constrained-policy
prompt LM (the remaining probability). Values `0.15` and `0.05` produce the
15% task / 5% full LM / 80% constrained LM protocol.

Set `prompt_loss_weight` to add an independently token-averaged next-token
loss over the prompt: `answer_loss + prompt_loss_weight * prompt_lm_loss`.
The prompt is overwhelmingly untouched PG-19 text, and both components are
logged separately. Measure clean-text regression on the raw held-out books:

For a variable-SWA generalization run, set `full_attention_lm_probability` to
route that fraction of microbatches to prompt-only LM training under full
attention. Explicit `task_probability` routes task updates; the remainder uses
prompt-only LM under the configured policy. Optional
`full_attention_lm_weight` scales those LM-only updates (default `1.0`). Do not
combine this routing mode with `prompt_loss_weight`.

```bash
uv run --locked llmpr-evaluate-lm \
  --model models/Qwen3-1.7B-Base \
  --data data/pg19-3800-640/pg19-test.jsonl \
  --checkpoint outputs/RUN/checkpoint-0000050.pt \
  --device cuda --length 3800 --books 32 --policies full
```

Omit `--checkpoint` for the frozen base-model baseline. The evaluator selects
the same deterministic interior window from each book for every checkpoint.

During training, `metrics.jsonl` records loss, gradient norm, learning rate,
step time, context- and answer-token throughput, peak allocated/reserved VRAM,
and ETA. A background sampler writes GPU utilization, VRAM, temperature, power,
CPU/RAM, process RSS, and disk space to `system.jsonl` every 10 seconds. Set
`monitor_interval_s` in the run config to change the interval, or to `0` to
disable system sampling. Non-finite loss or gradient norm stops training before
another checkpoint can be written.

For the rented-machine lifecycle, follow [`VAST_CHECKLIST.md`](VAST_CHECKLIST.md)
and run capacity calibration before launching a checked-in config.

### End-to-end one-step smoke test

This exercises HF streaming, Qwen tokenization, generated JSONL loading, LoRA,
backpropagation, and checkpoint writing on Apple Silicon or CUDA:

```bash
uv run llmpr-prepare-pg19 \
  --model models/Qwen3-1.7B-Base \
  --output-dir data/pg19-smoke \
  --context-lengths 512 \
  --levels passcode_easy,passcode_hard,structured,natural \
  --train-books 1 --validation-books 1 --test-books 1
```

For a local optimizer smoke test, make a temporary copy of the closest current
config and change only its data path, run directory, length, and step count.

## Training and exact resume

Training consumes state-episode JSONL produced by `llmpr-prepare-state` and uses
answer-only loss. A checkpoint contains every trainable parameter (LoRA and,
when enabled, retention scorers), optimizer/scheduler state, all process RNGs,
the named sampling generator, and counters. Resume rejects any config change.

```bash
uv run --locked llmpr-train --config configs/full-lm.json
uv run --locked llmpr-train --config configs/full-lm.json --resume
```

Policy strings are `full`, `swa:N`, `variable-swa:MIN-MAX`, `log:R+M`, and
`scored:R+M`. Variable SWA samples one window independently for every training
row and is evaluated using an explicit sweep of fixed `swa:N` policies. Scored
retention assigns an immutable priority in each layer and uses an exact hard
selection in the forward pass with a straight-through scorer gradient.

## Evaluation sweep

```bash
uv run --locked llmpr-evaluate \
  --model models/Qwen3-1.7B-Base --checkpoint outputs/run/checkpoint-0001000.pt \
  --data data/state.jsonl \
  --policies full,swa:1024,swa:512,swa:256,swa:128,swa:64,log:128+128 \
  --restart-modes preserve,restart:answer,restart:512,restart:256,restart:128
```

For a scored checkpoint, add its native `scored:R+M` policy. Its preserved pass
freezes per-layer priorities; restart replay reuses them and never reranks from
reconstructed hidden states. `restart:N` rebuilds before positions divisible by
N, while `restart:answer` rebuilds once at the answer boundary.

## Machine soundness and capacity

These are separate commands. Soundness checks cache/full logits, the
no-eviction replay identity, and native SDPA SWA against a dense reference with
actual eviction. Capacity performs complete LoRA updates at a fixed effective
batch and never loads, writes, or resumes experiment checkpoints.

```bash
uv run --locked llmpr-soundness --model models/Qwen3-1.7B-Base --device cuda
uv run --locked llmpr-capacity --model models/Qwen3-1.7B-Base --device cuda \
  --policy full --lengths 2048,4096,8192,12288,16384 --microbatches 1,2,4 \
  --effective-batch 16
```

Run capacity once per policy/backend. The current streaming-log and scored
training paths use explicit quadratic reference storage. Fixed-SWA training
also materializes a dense quadratic mask; it changes visibility semantics but
does not currently provide a physical VRAM reduction. These measurements are
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
