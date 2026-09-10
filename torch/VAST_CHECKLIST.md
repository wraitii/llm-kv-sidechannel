# Vast.ai instance checklist

Run this checklist whenever switching to a new instance or physical host.

The staged helper scripts live in `torch/scripts/vast/`. They never chain offer
search into rental, training into shutdown, or shutdown into destruction. Run
`00-preflight.sh`, then `10-search-offers.sh`; choose an offer yourself before
calling `20-create-instance.sh`.

```text
00-preflight.sh                          local/auth/SSH/Git checks (read-only)
10-search-offers.sh                      worldwide guarded 5090 search (read-only)
20-create-instance.sh OFFER [LABEL] [GB] show offer, confirm, then rent
30-show-instance.sh INSTANCE             status and SSH endpoint (read-only)
40-bootstrap-instance.sh INSTANCE [SHA]  clone exact commit and run uv sync
80-recover-files.sh INSTANCE RUN CONFIG DATASET
                                         pull outputs/config/manifest locally
90-stop-instance.sh INSTANCE             show status and require confirmation
99-destroy-instance.sh INSTANCE          require recovery marker + confirmation
```

The default image is `vastai/base-image:cuda-12.8.1-auto`, disk allocation
is 100 GB, maximum all-in hourly price is $0.80, and minimum reliability is
0.99. Override these locally with `VAST_IMAGE`, `VAST_STORAGE_GB`,
`VAST_MAX_HOURLY`, or `VAST_MIN_RELIABILITY`. Do not put overrides containing
secrets into committed files.

## Before renting

- [ ] Push the intended commit and rerun preflight until
      `remote_has_local_commit=true`; bootstrap clones the public remote.
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
- [ ] Run `uv sync --locked --extra dev --extra data` with Python 3.12 or 3.13.
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

## Next-run card: 1K, 500-step pilot

This is the next planned gate before a 4K run. Use a unique UTC-stamped `RUN_ID`
in both the config filename and `run_dir`; never reuse an output directory.

1. Run preflight, rent a current offer, wait for `running`, and bootstrap the
   exact pushed commit. Repeat the short machine checks above.
2. Download the pinned model directly on the instance and verify its checksum:

   ```bash
   cd /workspace/llm-kv-sidechannel/torch
   uv run --locked python - <<'PY'
   import json
   from huggingface_hub import snapshot_download

   spec = json.load(open("model-snapshots.json"))["Qwen3-1.7B-Base"]
   snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                     local_dir=spec["local_dir"])
   PY
   expected=$(jq -r '."Qwen3-1.7B-Base".model_safetensors_sha256' model-snapshots.json)
   printf '%s  %s\n' "$expected" models/Qwen3-1.7B-Base/model.safetensors \
     | sha256sum -c -
   ```

3. Generate only the 1K pilot dataset. The output directory must be new:

   ```bash
   uv run --locked llmpr-prepare-pg19 \
     --model models/Qwen3-1.7B-Base \
     --output-dir data/pg19-pilot-1k \
     --context-lengths 1024 \
     --train-books 32 --validation-books 8 --test-books 8 \
     --cache-dir /workspace/cache/pg19
   sha256sum data/pg19-pilot-1k/{manifest,train,validation,test}.json*
   ```

4. Make a temporary pilot copy of `configs/full-lm.json` and change its data,
   run directory, length, and schedule for this qualification only. Confirm `steps=500`, `batch_size=1`,
   `grad_accum=16`, `save_every=100`, and `monitor_interval_s=10`.
5. Train to step 200, inspect and evaluate that checkpoint, then resume to 500:

   ```bash
   HF_HUB_OFFLINE=1 uv run --locked llmpr-train \
     --config configs/RUN_CONFIG.json --device cuda --stop-after 200
   tail -n 5 outputs/RUN_ID/{metrics,system}.jsonl
   HF_HUB_OFFLINE=1 uv run --locked llmpr-evaluate \
     --model models/Qwen3-1.7B-Base \
     --checkpoint outputs/RUN_ID/checkpoint-0000200.pt \
     --data data/pg19-pilot-1k/validation.jsonl --max-length 1024 \
     --policies full,swa:512,swa:256 \
     --restart-modes preserve,restart:answer --examples 12 \
     | tee outputs/RUN_ID/validation-step-0000200.jsonl
   HF_HUB_OFFLINE=1 uv run --locked llmpr-train \
     --config configs/RUN_CONFIG.json --device cuda --resume
   HF_HUB_OFFLINE=1 uv run --locked llmpr-evaluate \
     --model models/Qwen3-1.7B-Base \
     --checkpoint outputs/RUN_ID/checkpoint-0000500.pt \
     --data data/pg19-pilot-1k/validation.jsonl --max-length 1024 \
     --policies full,swa:512,swa:256 \
     --restart-modes preserve,restart:answer --examples 12 \
     | tee outputs/RUN_ID/validation-step-0000500.jsonl
   ```

6. Inspect final metrics, system telemetry, disk space, checkpoints, and
   validation before recovery. Expected training time is roughly 15 minutes;
   allow 20--30 minutes including evaluation and transfers.
7. From this workstation, run `80-recover-files.sh INSTANCE RUN_ID
   configs/RUN_CONFIG.json data/pg19-pilot-1k`. Open the recovered JSONL and
   checkpoint locally, confirm `RECOVERY_COMPLETE`, then stop or destroy.
8. After destruction, verify both `vast show instances --raw` and
   `vast show volumes --raw` contain no unintended billable resources.

## Full capacity calibration — only when the execution configuration changes

Run this before the first experiment, and repeat it after changing the physical
host/GPU, image, driver, CUDA, PyTorch, attention kernel, model, LoRA setup,
precision, optimizer, activation checkpointing, attention policy, or sequence
packing strategy. Also repeat it when the selected setting is close to OOM.

Measure complete training updates, not inference or forward-only passes. Include
LoRA backward, gradient clipping, and the optimizer step. Test full attention
and each efficient retention backend separately; their limits may differ.

- [ ] On a 32 GB GPU, start with microbatch 1 and sweep context lengths 1K, 2K,
      3K, and 4K. Test 6K/8K or higher only when the lower points leave useful
      headroom or the machine has more VRAM.
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
uv sync --locked --extra dev --extra data
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
tail -f outputs/RUN_ID/metrics.jsonl
tail -f outputs/RUN_ID/system.jsonl
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

- [ ] Recover files from the instance with `80-recover-files.sh` while SSH is
      still available; do this before stopping or destroying it.
- [ ] Confirm the local `RECOVERY_COMPLETE` marker exists for the correct
      instance and run. This marker is required by `99-destroy-instance.sh`.
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

## Test-flight record: 2026-09-08

Instance `50283926` used offer `46691993` on machine `140868` in South Korea:
an RTX 5090 with 32,607 MiB VRAM, driver 595.71.05, CUDA 13.2 capability,
61 GiB system RAM, and a 100 GB instance disk. The working image was
`vastai/base-image:cuda-12.8.1-auto`; the automatic base-image tag had no image
compatible with compute capability 12.0 and CUDA 13.2. The environment used
commit `72655190c8c8a72dec886784a3eadcff3dca9445`, Python 3.12.3, Torch
2.14.0+cu130, Transformers 5.16.1, PEFT 0.20.0, and cuDNN 9.24.

The locked suite passed (24 tests on the original commit; 25 locally after the
CUDA resume regression test was added). The pinned Qwen3-1.7B-Base snapshot at
revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` matched SHA-256
`6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841`.
Full-attention and fixed-SWA BF16 LoRA optimizer steps passed. Soundness passed
in FP32 at `atol=2e-4`, with cache/full and no-eviction restart maximum errors
of `1.4495849609375e-4`, native-SWA/dense error `7.796287536621094e-5`, and 24
tokens actually evicted. BF16 is appropriate for training but produced large
false-positive semantic comparison errors across differing kernel shapes.

Full-attention capacity at microbatch 1 and effective batch 16 was:

| Context | Mean optimizer step | Tokens/s | Peak reserved VRAM |
| ---: | ---: | ---: | ---: |
| 1,024 | 1.61 s | 10,194 | 11.17 GB |
| 2,048 | 3.47 s | 9,441 | 17.50 GB |
| 3,072 | 5.80 s | 8,478 | 24.05 GB |
| 4,096 | 9.43 s | 6,947 | 30.71 GB |

At 2K, microbatch 2 passed but reserved 30.74 GB and microbatch 4 OOMed. At
4K, microbatch 1 passed five measured updates after warmup but retained only
about 8.8% reserved-memory headroom. Both 6K and 8K OOMed. Native causal,
forced Flash SDPA, answer-tail logits, and native SWA-1024 probes did not make
8K fit with this locked stack. Use 4K only as an upper-edge setting and prefer
3K when the checklist's 10% headroom is required.

The checkpoint test found that CUDA RNG tensors loaded with `map_location=cuda`
must be moved back to CPU before `torch.cuda.set_rng_state_all`. After that fix,
the resumed step, micro-step, tokens, loss, and gradient norm matched the
uninterrupted run exactly. Preserve/restart evaluation completed for full
attention and SWA-256. Reports, configs, manifest, metrics, and checkpoints were
checksum-recovered under `artifacts/vast/instances/50283926/qualification-50283926/`.
Stopping changed Vast status to `exited`, retained the 100 GB disk, and made SSH
refuse connections as expected.
