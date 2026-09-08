# Vast.ai instance checklist

Run this checklist whenever switching to a new instance or physical host.

## Before renting

- [ ] RTX 5090 with 32 GB VRAM; verified host with high reliability.
- [ ] Enough rental duration, disk space, RAM, CPU, and network bandwidth.
- [ ] NVIDIA driver supports the CUDA 13 libraries selected by `uv.lock`.
- [ ] Prefer on-demand until checkpoint/resume is proven for the run.

## On every new instance

- [ ] Record offer/instance ID, host ID, image digest, hourly price, and date.
- [ ] Save `nvidia-smi` output, including driver, VRAM, power limit, and GPU name.
- [ ] Clone the exact code commit and confirm `git status --short` is clean.
- [ ] Run `uv sync --locked --extra dev` with Python 3.12 or 3.13.
- [ ] Record Python, Torch, Transformers, PEFT, CUDA, and cuDNN versions.
- [ ] Copy the pinned model snapshot and verify the SHA-256 in
      `model-snapshots.json`.
- [ ] Copy or regenerate data; verify dataset manifest/hash and example counts.
- [ ] Confirm output/checkpoint storage is synced off-host.
- [ ] Run `uv run --locked pytest -q`.
- [ ] Run `llmpr-soundness` and archive its JSON report.
- [ ] Run one full-attention LoRA optimizer step.
- [ ] Run one fixed-SWA LoRA optimizer step with actual eviction.
- [ ] Run the short dense-reference versus efficient-attention equivalence test.
- [ ] Test save/resume by comparing the next loss and update counter.
- [ ] Confirm the previously selected context/microbatch setting with one warmup
      and two complete training updates; compare peak VRAM and throughput.
- [ ] Run one preserve and restart evaluation example at the selected context.
- [ ] If confirmation fails or differs materially, run the full calibration.
- [ ] Only then start or resume the experiment.

## Full capacity calibration — only when the execution configuration changes

Run this before the first experiment, and repeat it after changing the physical
host/GPU, image, driver, CUDA, PyTorch, attention kernel, model, LoRA setup,
precision, optimizer, activation checkpointing, attention policy, or sequence
packing strategy. Also repeat it when the selected setting is close to OOM.

Measure complete training updates, not inference or forward-only passes. Include
LoRA backward, gradient clipping, and the optimizer step. Test full attention
and each efficient retention backend separately; their limits may differ.

- [ ] Start with microbatch 1 and sweep context lengths 2K, 4K, 8K, then 12K
      and 16K if useful.
- [ ] At the intended context length, sweep microbatch 1, 2, 4 until OOM or
      throughput stops improving.
- [ ] Benchmark preserve and restart evaluation separately at batch 1; replay
      may have a lower maximum context than ordinary training or inference.
- [ ] Run at least five consecutive updates for each candidate after one warmup
      update; a single successful allocation is insufficient.
- [ ] Record policy/backend, context length, microbatch, gradient accumulation,
      effective batch, peak allocated/reserved VRAM, tokens/second, and step time.
- [ ] Confirm actual non-padding token counts; do not infer context from a config
      value when examples are shorter.
- [ ] Select a setting with at least 10% VRAM headroom for evaluation,
      checkpointing, allocator variation, and unusually shaped batches.
- [ ] Preserve the protocol's effective batch by setting
      `grad_accum = effective_batch / microbatch`; require exact divisibility.
- [ ] Restart the benchmark process after an OOM before trusting later memory
      results; allocator state can make results misleading.
- [ ] Save the chosen capacity record with the run manifest. A new instance may
      reuse it only after passing the shorter confirmation above.

## Commands

```bash
nvidia-smi
git rev-parse HEAD
git status --short
uv sync --locked --extra dev
uv run --locked pytest -q

sha256sum models/Qwen3-1.7B-Base/model.safetensors

HF_HUB_OFFLINE=1 uv run --locked llmpr-qwen-smoke \
  --model models/Qwen3-1.7B-Base --device cuda

HF_HUB_OFFLINE=1 uv run --locked llmpr-qwen-smoke \
  --model models/Qwen3-1.7B-Base --device cuda --window 128
```

The expected model hash is recorded in `model-snapshots.json`. Use
`llmpr-capacity` in a fresh process for each policy. Restart the process after
any OOM; the command stops at the first OOM for that reason.

## Before stopping or destroying

- [ ] Upload adapters, optimizer/scheduler state, RNG state, configs, manifests,
      logs, per-example results, and environment/hardware report.
- [ ] Verify the uploaded files and hashes from outside the instance.
- [ ] Confirm the latest checkpoint resumes before destroying local storage.
- [ ] Destroy unused instances/volumes after verifying the backup; stopped
      instances still incur storage charges.
