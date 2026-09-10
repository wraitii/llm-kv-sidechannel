# Short 3.8K experiment run card

This run compares three LoRA fine-tunes on the same `pg19-3800-v2` training
set and evaluates all checkpoints on its validation split:

| Arm | Steps | Training policy | Loss routing |
| --- | ---: | --- | --- |
| Full task | 100 | full | 100% answer-only task |
| Fixed SWA | 500 | SWA-512 | 80% constrained LM / 15% task / 5% full LM |
| Variable SWA | 500 | SWA-128--800 | 80% constrained LM / 15% task / 5% full LM |

The two constrained arms have the same configured step count, effective batch,
optimizer, seed, dataset, and routing mixture. They checkpoint every 25 steps,
so `--stop-after N` can end either run early at a checkpoint and `--resume` can
continue it later without changing the config or learning-rate schedule. The
full-task arm is intentionally shorter because it is expected to saturate
quickly. Its checkpoints every 10 steps make 50 through 100 available without
rerunning training.

## Before launch

Follow `VAST_CHECKLIST.md`, including the locked install, model checksum,
tests, soundness checks, capacity confirmation, save/resume test, and artifact
recovery test. Confirm these exact inputs before spending GPU time:

```bash
test -f models/Qwen3-1.7B-Base/model.safetensors
test -f data/pg19-3800-v2/manifest.json
test -f data/pg19-3800-v2/train.jsonl
test -f data/pg19-3800-v2/validation.jsonl
jq . data/pg19-3800-v2/manifest.json
sha256sum data/pg19-3800-v2/{manifest,train,validation,test}.json*
```

Use new output directories if any of `outputs/full-task`,
`outputs/fixed-swa-lm-task`, or `outputs/variable-swa-lm-task` already exists;
checkpoint resume requires an exactly matching config.

## Train

```bash
HF_HUB_OFFLINE=1 uv run --locked llmpr-train \
  --config configs/full-task.json --device cuda
HF_HUB_OFFLINE=1 uv run --locked llmpr-train \
  --config configs/fixed-swa-lm-task.json --device cuda
HF_HUB_OFFLINE=1 uv run --locked llmpr-train \
  --config configs/variable-swa-lm-task.json --device cuda
```

To take an early checkpoint-aligned exit, add (for example) `--stop-after 200`.
Continue later with the same config and `--resume`; do not reduce `steps` in the
config, because resume deliberately rejects config changes.

## Probe evaluation

Run the following for each arm, replacing `RUN` and `STEP`. The intended final
values are `full-task`/`0000100`, `fixed-swa-lm-task`/`0000500`, and
`variable-swa-lm-task`/`0000500`, or the matching checkpoint for an intentional
early stop. For the full arm, also evaluate step 50 if the metrics show early
saturation.

```bash
HF_HUB_OFFLINE=1 uv run --locked llmpr-evaluate \
  --model models/Qwen3-1.7B-Base \
  --checkpoint outputs/RUN/checkpoint-STEP.pt \
  --data data/pg19-3800-v2/validation.jsonl \
  --max-length 3800 \
  --policies swa:512,swa:128 \
  --restart-modes preserve,restart:answer,restart:512,restart:128 \
  --split-task-type \
  | tee outputs/RUN/validation-STEP.jsonl
```

Here, "512/128 only" applies to attention policies and periodic restart
intervals. Preserve and restart-at-answer remain mandatory comparison modes.

## Clean-text evaluation

Probe evaluation does not measure language regression. Run this separately for
each final checkpoint on the raw held-out validation books:

```bash
HF_HUB_OFFLINE=1 uv run --locked llmpr-evaluate-lm \
  --model models/Qwen3-1.7B-Base \
  --checkpoint outputs/RUN/checkpoint-STEP.pt \
  --data data/pg19-3800-v2/pg19-validation.jsonl \
  --device cuda --length 3800 --books 32 \
  --policies swa:512,swa:128 \
  | tee outputs/RUN/lm-validation-STEP.jsonl
```

Also run the same command without `--checkpoint` once for the frozen-base
reference. Archive configs, manifest and hashes, metrics, system telemetry,
evaluation JSONL, checkpoints, soundness output, and the machine description
before releasing the instance.
