# Vast.ai instance checklist

Run this checklist whenever switching to a new instance or physical host.

## Before renting

- [ ] RTX 5090 with 32 GB VRAM; verified host with high reliability.
- [ ] Use SSH-key authentication and confirm the offer provides direct SSH.
- [ ] Enough rental duration, disk space, RAM, CPU, and network bandwidth.
- [ ] NVIDIA driver supports the CUDA 13 libraries selected by `uv.lock`.
- [ ] Prefer on-demand until checkpoint/resume is proven for the run.
- [ ] Understand the billing shown by the offer: GPU/compute and attached
      storage are separate. Stopping compute does not stop storage charges.
- [ ] Choose a durable destination on the local workstation before starting
      and confirm it has enough free disk. Vast instance disks and persistent
      volumes are host-local convenience storage, not the sole backup.

## On every new instance

- [ ] Record offer/instance ID, host ID, image digest, hourly price, and date.
- [ ] Save `nvidia-smi` output, including driver, VRAM, power limit, and GPU name.
- [ ] Record `df -h` and `free -h`; confirm enough disk for the environment,
      model, HF cache, selected raw books, datasets, and multiple checkpoints.
- [ ] Confirm PyTorch sees CUDA and reports the expected device and VRAM.
- [ ] Clone the exact code commit and confirm `git status --short` is clean.
- [ ] Run `uv sync --locked --extra dev` with Python 3.12 or 3.13.
- [ ] Record Python, Torch, Transformers, PEFT, CUDA, and cuDNN versions.
- [ ] Copy the pinned model snapshot and verify the SHA-256 in
      `model-snapshots.json`.
- [ ] Copy or regenerate data; verify dataset manifest/hash and example counts.
- [ ] Keep credentials in environment variables or a secrets mechanism, never
      in Git, configs, shell history, logs, or dataset manifests.
- [ ] Create explicit locations for dataset/cache, outputs/checkpoints, logs,
      and final artifacts; verify the training config points to them.
- [ ] Test copying a small file from Vast to the local workstation before
      training. Confirm it can be listed and read locally.
- [ ] Run `uv run --locked pytest -q`.
- [ ] Run `llmpr-soundness` and archive its JSON report.
- [ ] Run one full-attention LoRA optimizer step.
- [ ] Run one fixed-SWA LoRA optimizer step and separately verify actual
      eviction through the soundness checks.
- [ ] Run the short dense-reference versus efficient-attention equivalence test.
- [ ] Test save/resume by comparing the next loss and update counter.
- [ ] Confirm the previously selected context/microbatch setting with one warmup
      and two complete training updates; compare peak VRAM and throughput.
- [ ] Run one preserve and restart evaluation example at the selected context.
- [ ] If confirmation fails or differs materially, run the full calibration.
- [ ] Only then start or resume the experiment.

## End-to-end lifecycle smoke test

After machine soundness and capacity calibration, exercise the exact real-run
config for 100--500 steps, or the longest affordable interval that crosses at
least two checkpoint boundaries. Keep metrics logging enabled and run the
intended evaluation command at a checkpoint.

- [ ] Train through a periodic checkpoint and record throughput and peak VRAM.
- [ ] Stop the process cleanly after a checkpoint.
- [ ] Resume from that checkpoint with `--resume` and cross the next checkpoint.
- [ ] Verify step, micro-step, token, optimizer, scheduler, and RNG counters
      continue rather than restart.
- [ ] Run validation using the resumed checkpoint.
- [ ] Copy a checkpoint, config, manifest, metrics, and validation output to
      the local workstation, then verify them locally.
- [ ] Do not commit to the long run until this complete lifecycle succeeds.

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
- [ ] Choose by stable tokens/second, not maximum allocation alone. A larger
      microbatch that technically fits may have worse throughput or stability.
- [ ] Preserve the protocol's effective batch by setting
      `grad_accum = effective_batch / microbatch`; require exact divisibility.
      In general, effective batch is per-device microbatch times gradient
      accumulation times GPU count; this runner currently targets one GPU.
- [ ] Restart the benchmark process after an OOM before trusting later memory
      results; allocator state can make results misleading.
- [ ] Save the chosen capacity record with the run manifest. A new instance may
      reuse it only after passing the shorter confirmation above.

## Commands

```bash
nvidia-smi
df -h
free -h

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    print("gpu:", device.name)
    print("vram_gb:", device.total_memory / 1e9)
PY

git rev-parse HEAD
git status --short
uv sync --locked --extra dev
uv run --locked pytest -q

sha256sum models/Qwen3-1.7B-Base/model.safetensors

HF_HUB_OFFLINE=1 uv run --locked llmpr-qwen-smoke \
  --model models/Qwen3-1.7B-Base --device cuda

HF_HUB_OFFLINE=1 uv run --locked llmpr-qwen-smoke \
  --model models/Qwen3-1.7B-Base --device cuda --window 128

HF_HUB_OFFLINE=1 uv run --locked llmpr-soundness \
  --model models/Qwen3-1.7B-Base --device cuda --length 32 --window 8

HF_HUB_OFFLINE=1 uv run --locked llmpr-capacity \
  --model models/Qwen3-1.7B-Base --device cuda --policy full \
  --lengths 2048,4096,8192,12288,16384 --microbatches 1,2,4 \
  --effective-batch 16
```

The expected model hash is recorded in `model-snapshots.json`. Use
`llmpr-capacity` in a fresh process for each policy. Restart the process after
any OOM; the command stops at the first OOM for that reason.

During calibration and the real run, monitor more than allocated VRAM:

```bash
watch -n 1 nvidia-smi
watch -n 5 df -h
```

Record GPU utilization, tokens/second, step time, CPU utilization, free disk,
and signs of network or data-loading stalls. Use `ps`, `top`, `htop`, `iostat`,
or equivalent host tools when available. Do not install unpinned Python packages
into the locked project environment merely to obtain monitoring utilities.

## Storage and backup

A typical instance layout is:

```text
/workspace/
  repo/
  dataset/
  cache/
  checkpoints/
  logs/
  final/
```

Local dataset/cache storage may be disposable. Checkpoints, configs, manifests,
logs, and final results are not. Write checkpoints to the Vast disk for speed,
then periodically pull important artifacts to this workstation. Run these
commands on the workstation, replacing the placeholders with the SSH details
shown by Vast:

```bash
mkdir -p artifacts/vast/RUN_ID/outputs
rsync -avP -e "ssh -p SSH_PORT" \
  root@VAST_HOST:/workspace/repo/torch/outputs/RUN_ID/ \
  artifacts/vast/RUN_ID/outputs/
rsync -avP -e "ssh -p SSH_PORT" \
  root@VAST_HOST:/workspace/repo/torch/configs/RUN_CONFIG.json \
  artifacts/vast/RUN_ID/
rsync -avP -e "ssh -p SSH_PORT" \
  root@VAST_HOST:/workspace/repo/torch/data/DATASET/manifest.json \
  artifacts/vast/RUN_ID/
```

Rerun the same `rsync` commands whenever a checkpoint should become durable;
`--partial` (included by `-P`) allows interrupted large transfers to continue.
After the final copy, rerun with `-avnc` instead of `-avP`: no listed file
changes means the local tree matches by checksum. Also open the copied manifest
and metrics, compare checkpoint sizes, and perform the documented resume check
from the local copy before destroying the instance.

Do not copy these commands blindly: replace `RUN_ID`, `RUN_CONFIG.json`,
`DATASET`, `SSH_PORT`, and `VAST_HOST`, and verify both source and destination.
The local workstation is the durable source of truth for the initial runs.
Object storage such as S3, R2, B2, or GCS remains an optional second backup.
Never make a Vast persistent volume the only backup; it remains tied to a
particular Vast host.

## Before stopping or destroying

- [ ] Stop training cleanly and wait for the final checkpoint write to finish.
- [ ] Copy adapters, optimizer/scheduler state, RNG state, configs, manifests,
      logs, per-example results, and environment/hardware report locally.
- [ ] Verify the local copy by checksum and read key files on this workstation.
- [ ] Confirm the latest checkpoint resumes before destroying local storage.
- [ ] If returning soon, stop the instance: compute billing stops, but attached
      storage billing continues and files remain on the host.
- [ ] After verified backup, destroy instances no longer needed. Destruction
      deletes their local disks and stops billing for that instance/storage.
- [ ] Separately delete unused persistent volumes; destroying an instance does
      not imply that every separately billed volume is gone.

Core rule: Vast is disposable compute plus temporary host storage. For the
initial runs, this workstation's artifact directory is the durable source of
truth.
