"""Train byte or domain-BPE prefix-LM baselines on MLX."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .data import CachedPairDataset, MixedPairDataset, PairDataset
from .board_eval import score_board_outputs
from .model import PrefixLM, model_from_config
from .tokenizer import EOS, PairTokenizer, load_tokenizer
from .transport import RecursiveCarrierPolicy, policy_from_config
from .carriers import expand_batch
from .experiment import training_batch, record_run
from .move_alignment import sample_transport, batch_sources
from .readouts import prepare_readouts, shared_readout_loss


def sample_train_windows(rng: np.random.Generator, spec, batch_size: int,
                         full_rows: np.ndarray, full_window: int):
    """Per-row training windows: fixed scalar, [low, high] range, or none.

    Rows in ``full_rows`` get ``full_window`` (a window wide enough to cover
    the whole sequence), i.e. ordinary full attention for that share.
    """
    if spec is None:
        return None
    if isinstance(spec, list):
        lo, hi = spec
        windows = rng.integers(lo, hi + 1, size=batch_size).astype(np.int32)
    else:
        windows = np.full(batch_size, spec, dtype=np.int32)
    if full_rows is not None and full_rows.any():
        windows = windows.copy()
        windows[full_rows] = full_window
    return mx.array(windows)


def dtype_for(name: str):
    return {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[name]


def as_mx(batch: dict[str, np.ndarray]) -> dict[str, mx.array]:
    return {key: mx.array(value) for key, value in batch.items()}


def source_spans_to_sequence(spans: np.ndarray) -> np.ndarray:
    """Offset real source coordinates for BOS without reviving padded spans."""
    return np.where(spans >= 0, spans + 1, spans).astype(np.int32)


def save_checkpoint(path: Path, model, optimizer, metadata: dict) -> None:
    mx.eval(model.state, optimizer.state)
    values = {}
    for key, value in tree_flatten(model.state):
        values[f"model::{key}"] = value
    for key, value in tree_flatten(optimizer.state):
        values[f"optimizer::{key}"] = value
    mx.savez(str(path), **values)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_checkpoint(path: Path, model, optimizer) -> dict:
    values = mx.load(str(path))
    model_values = []
    optimizer_values = []
    for key, value in values.items():
        if key.startswith("model::"):
            model_values.append((key[7:], value))
        elif key.startswith("optimizer::"):
            optimizer_values.append((key[11:], value))
    expected = dict(tree_flatten(model.parameters()))
    loaded = dict(model_values)
    if expected.keys() != loaded.keys() or any(
            expected[name].shape != loaded[name].shape for name in expected):
        raise ValueError("resume requires identical model parameters; use --init-checkpoint for new arms")
    model.update(tree_unflatten(model_values))
    optimizer.state = tree_unflatten(optimizer_values)
    return json.loads(path.with_suffix(".json").read_text())


def load_model_checkpoint(path: Path, model) -> dict:
    """Load model weights, tolerating added carrier-vocab embedding rows.

    New rows (rows beyond the checkpoint's ``embed.weight``) get the same
    small-random init as fresh embeddings.  Copying a trained special token
    instead (e.g. ``<fen>``) catastrophically perturbs the pretrained
    function, since multiple copies of a control token appear mid-source.
    """
    values = mx.load(str(path))
    target_shapes = dict(tree_flatten(model.parameters()))
    model_values = []
    for key, value in values.items():
        if not key.startswith("model::"):
            continue
        name, weight = key[7:], value
        shape = target_shapes.get(name)
        if (name == "embed.weight" and shape is not None
                and tuple(shape.shape) != tuple(weight.shape)
                and weight.ndim == 2 and shape.shape[1] == weight.shape[1]
                and weight.shape[0] < shape.shape[0]):
            extra = shape.shape[0] - weight.shape[0]
            fresh = (mx.random.normal((extra, weight.shape[1])) * 0.02
                     ).astype(weight.dtype)
            weight = mx.concatenate([weight, fresh], axis=0)
        if shape is None or tuple(shape.shape) != tuple(weight.shape):
            raise ValueError(f"incompatible checkpoint parameter {name}")
        model_values.append((name, weight))
    missing = set(target_shapes) - {name for name, _ in model_values}
    if any(".retention." not in name for name in missing):
        raise ValueError(f"checkpoint is missing model parameters: {sorted(missing)}")
    model.update(tree_unflatten(model_values))
    mx.eval(model.parameters())
    return json.loads(path.with_suffix(".json").read_text())


def latest_checkpoint(run_dir: Path) -> Path:
    pointer = run_dir / "latest.json"
    if pointer.exists():
        return run_dir / json.loads(pointer.read_text())["checkpoint"]
    checkpoints = sorted(run_dir.glob("checkpoint-*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found in {run_dir}")
    return checkpoints[-1]


def rotate_checkpoints(run_dir: Path, keep: int, milestone_every: int = 0) -> None:
    checkpoints = sorted(run_dir.glob("checkpoint-*.npz"))
    recent = set(checkpoints[-max(1, keep):])
    for path in checkpoints:
        step = int(path.stem.rsplit("-", 1)[1])
        milestone = milestone_every > 0 and step % milestone_every == 0
        if path in recent or milestone:
            continue
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)


def board_state_metrics(predicted: np.ndarray, output_y: np.ndarray,
                        output_mask: np.ndarray,
                        tokenizer: PairTokenizer) -> tuple[int, int, int, int]:
    """Measure board validity and semantic distance in teacher-forced predictions."""
    outputs = []
    references = []
    for row, reference, mask in zip(predicted, output_y, output_mask, strict=True):
        active = mask.astype(bool)
        ids = row[active].tolist()
        reference_ids = reference[active].tolist()
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
        if EOS in reference_ids:
            reference_ids = reference_ids[:reference_ids.index(EOS)]
        outputs.append(tokenizer.target.decode(ids))
        references.append(tokenizer.target.decode(reference_ids))
    parseable, valid, _, square_errors, metadata_errors = score_board_outputs(
        outputs, references)
    return parseable, valid, square_errors, metadata_errors


def student_board_metrics(model: PrefixLM, tokenizer: PairTokenizer,
                          dataset, indices: np.ndarray,
                          batch_size: int,
                          carrier_policy=None, transport=False, sliding_window=None) -> tuple[int, int, int, int, int]:
    """Measure board quality from free-running greedy generation."""
    from .eval_chess import generate_batch

    groups: dict[int, list[dict]] = {}
    for index in indices:
        sequence, target_local, sep_index, *_ = dataset.encode(int(index))
        source = sequence[1:sep_index]
        row = {"asm": tokenizer.decode_source(source),
               "code": tokenizer.target.decode(target_local)}
        groups.setdefault(len(source), []).append(row)
    outputs = []
    references = []
    for rows in groups.values():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            generated = generate_batch(model, tokenizer, batch_rows,
                                       dataset.max_source_tokens,
                                       dataset.max_target_tokens, 0.0, 1.0,
                                       carrier_policy=carrier_policy, transport=transport,
                                       sliding_window=sliding_window)
            outputs.extend(generated)
            references.extend(row["code"] for row in batch_rows)
    return score_board_outputs(outputs, references)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="override config steps")
    parser.add_argument("--resume", nargs="?", const="auto",
                        help="resume PATH, or latest in run_dir when passed without PATH")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="initialize model weights only; optimizer starts fresh")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluate the selected checkpoint without training")
    parser.add_argument("--fixed-eval-seed", type=int,
                        help="use one deterministic validation subset in eval-only mode")
    parser.add_argument("--metal-cache-mib", type=int, default=2048,
                        help="cap MLX's cache of unused Metal buffers (default: 2048 MiB)")
    args = parser.parse_args()
    if args.metal_cache_mib < 0:
        parser.error("--metal-cache-mib must be non-negative")
    mx.set_cache_limit(args.metal_cache_mib * 1024**2)
    mx.reset_peak_memory()
    cfg = json.loads(args.config.read_text())
    if args.steps is not None:
        cfg["steps"] = args.steps

    rng = np.random.default_rng(cfg["seed"])
    policy_rng = np.random.default_rng(cfg["seed"] + 100_000)
    readout_rng = np.random.default_rng(cfg["seed"] + 300_000)
    mx.random.seed(cfg["seed"])
    tokenizer = PairTokenizer(load_tokenizer(cfg["source_tokenizer"]),
                              load_tokenizer(cfg["target_tokenizer"]),
                              cfg.get("pause_token", False), cfg.get("pause_tokens"),
                              cfg.get("distinct_pause_tokens", False),
                              cfg.get("causal_pause", False),
                              cfg.get("carrier_vocab", 0))
    if cfg.get("cache_dir"):
        train = CachedPairDataset(cfg["cache_dir"], "train", tokenizer,
                                  bucket_size=cfg.get("length_bucket_size", 0))
        # Keep validation sampling unbucketed so metrics retain the original
        # example distribution without correlated batches.
        val = CachedPairDataset(cfg["cache_dir"], "val", tokenizer)
        if cfg.get("random_cache_dir"):
            random_train = CachedPairDataset(
                cfg["random_cache_dir"], "train", tokenizer,
                bucket_size=cfg.get("length_bucket_size", 0))
            train = MixedPairDataset(
                train, random_train, cfg.get("random_train_weight", 0.25))
    else:
        train = PairDataset(Path(cfg["data_dir"]) / "train.jsonl", tokenizer,
                            cfg["max_source_tokens"], cfg["max_target_tokens"])
        val = PairDataset(Path(cfg["data_dir"]) / "val.jsonl", tokenizer,
                          cfg["max_source_tokens"], cfg["max_target_tokens"])
    if (train.max_source_tokens != cfg["max_source_tokens"]
            or train.max_target_tokens != cfg["max_target_tokens"]):
        raise ValueError("configured token limits differ from the token cache")
    model = model_from_config(tokenizer.source.vocab_size, tokenizer.target.vocab_size,
                              cfg, dtype_for(cfg["dtype"]))
    # --resume wins: a config's init_checkpoint only applies to a fresh run.
    init_checkpoint = (Path(args.init_checkpoint) if args.init_checkpoint
                       else (Path(cfg["init_checkpoint"])
                             if cfg.get("init_checkpoint") and not args.resume
                             else None))
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if init_checkpoint and not args.eval_only:
        init_state = load_model_checkpoint(init_checkpoint, model)
        print(f"initialized model weights from {init_checkpoint} "
              f"(source step {init_state.get('step', 'unknown')})", flush=True)
    transport_policy = policy_from_config(cfg.get("transport_policy"))
    if transport_policy is not None and model.attention_mode != "causal":
        raise ValueError("transport policies require attention_mode=causal")
    train_sliding_window = cfg.get("train_sliding_window")
    if train_sliding_window is not None:
        if (isinstance(train_sliding_window, list)
                and (len(train_sliding_window) != 2
                     or not all(isinstance(v, int) for v in train_sliding_window)
                     or train_sliding_window[0] < 1
                     or train_sliding_window[1] < train_sliding_window[0])):
            raise ValueError("train_sliding_window range must be [low, high] "
                             "with 1 <= low <= high")
        if not isinstance(train_sliding_window, (int, list)) or (
                isinstance(train_sliding_window, int) and train_sliding_window < 1):
            raise ValueError("train_sliding_window must be a positive integer "
                             "or a [low, high] range")
        if model.attention_mode != "causal":
            raise ValueError("train_sliding_window requires attention_mode='causal'")
    full_attention_share = float(cfg.get("full_attention_share", 0.0))
    if not 0.0 <= full_attention_share < 1.0:
        raise ValueError("full_attention_share must be in [0, 1)")
    training_readouts = int(cfg.get("training_readouts", 3))
    if training_readouts < 1:
        raise ValueError("training_readouts must be positive")
    negative_probability = float(cfg.get("negative_score_probability", 0.0))
    if not 0.0 <= negative_probability <= 1.0:
        raise ValueError("negative_score_probability must be in [0, 1]")
    if negative_probability and not cfg.get("scored_eviction"):
        raise ValueError("negative scorer examples require scored_eviction")
    if float(cfg.get("negative_score_weight", 0.1)) < 0:
        raise ValueError("negative_score_weight must be non-negative")
    if int(cfg.get("negative_score_swaps", 2)) < 1:
        raise ValueError("negative_score_swaps must be positive")
    if int(cfg.get("negative_score_alternatives", 4)) < 1:
        raise ValueError("negative_score_alternatives must be positive")
    if cfg.get("scored_eviction") and (transport_policy is not None
            or train_sliding_window is not None):
        raise ValueError("scored_eviction is a separate arm; omit transport_policy and train_sliding_window")
    if cfg.get("scored_eviction") and full_attention_share and training_readouts == 1:
        raise ValueError("scored full-attention escape requires shared training readouts")
    # A window at least this wide covers every legal query/key pair.
    full_window = model.max_length + 1
    parameter_count = sum(value.size for _, value in
                          tree_flatten(model.trainable_parameters()))
    run_dir = Path(cfg["run_dir"])
    run_config = {**cfg, "config_path": str(args.config),
                  "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else cfg.get("init_checkpoint"),
                  "input_vocab_size": tokenizer.vocab_size,
                  "source_vocab_size": tokenizer.source.vocab_size,
                  "target_vocab_size": tokenizer.target.vocab_size,
                  "parameter_count": parameter_count}
    record_run(run_dir, run_config, resume=bool(args.resume), eval_only=args.eval_only)
    metrics_path = run_dir / "metrics.jsonl"

    warmup = max(1, cfg["warmup_steps"])
    decay = max(1, cfg["steps"] - warmup)
    schedule = optim.join_schedules([
        optim.linear_schedule(cfg["learning_rate"] / warmup,
                              cfg["learning_rate"], warmup),
        optim.cosine_decay(cfg["learning_rate"], decay, cfg["min_learning_rate"]),
    ], [warmup])
    optimizer = optim.AdamW(schedule, betas=[0.9, 0.95],
                            weight_decay=cfg["weight_decay"])
    start_step = target_tokens_seen = model_tokens_seen = 0
    prior_elapsed = 0.0
    if args.resume:
        checkpoint = (latest_checkpoint(run_dir) if args.resume == "auto"
                      else Path(args.resume))
        state = load_checkpoint(checkpoint, model, optimizer)
        start_step = state["step"]
        target_tokens_seen = state["target_tokens_seen"]
        model_tokens_seen = state["model_tokens_seen"]
        prior_elapsed = state.get("elapsed_s", 0.0)
        rng.bit_generator.state = state["numpy_rng_state"]
        if "policy_rng_state" in state:
            policy_rng.bit_generator.state = state["policy_rng_state"]
        if "readout_rng_state" in state:
            readout_rng.bit_generator.state = state["readout_rng_state"]
        print(f"resumed {checkpoint} at step {start_step}", flush=True)

    def loss_fn(active_model, x, output_y, valid, prefix_lengths,
                output_positions, output_mask, transport_spans, sliding_windows):
        logits = active_model(x, valid, prefix_lengths, output_positions,
                              transport_spans, sliding_windows).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, output_y)
        denominator = mx.maximum(mx.sum(output_mask), 1)
        return mx.sum(losses * output_mask) / denominator

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def readout_loss_fn(active_model, source_x, source_valid, spans, points,
                        branch_x, branch_y, branch_valid, sliding_windows,
                        negative, full_rows):
        return shared_readout_loss(
            active_model, source_x, source_valid, spans, points,
            branch_x, branch_y, branch_valid, sliding_windows, negative,
            cfg.get("negative_score_weight", 0.1),
            cfg.get("negative_score_swaps", 2),
            cfg.get("negative_score_alternatives", 4), True,
            full_rows)

    readout_value_and_grad = nn.value_and_grad(model, readout_loss_fn)
    fixed_eval_indices = None
    if args.fixed_eval_seed is not None:
        count = min(cfg["eval_batches"] * cfg["batch_size"], len(val))
        fixed_eval_indices = np.random.default_rng(args.fixed_eval_seed).choice(
            len(val), count, replace=False)

    def evaluate() -> dict[str, float]:
        eval_rng = np.random.default_rng(args.fixed_eval_seed if args.fixed_eval_seed is not None else cfg["seed"] + 1_000_000)
        layout_rng = np.random.default_rng(cfg["seed"] + 2_000_000)
        nll = mx.array(0.0)
        correct = mx.array(0.0)
        tokens = mx.array(0.0)
        perfect_examples = mx.array(0.0)
        byte_count = mx.array(0.0)
        source_bytes = mx.array(0.0)
        source_tokens = mx.array(0.0)
        examples = mx.array(0.0)
        parseable_boards = 0
        valid_boards = 0
        square_errors = 0
        metadata_errors = 0
        if fixed_eval_indices is None:
            eval_batches = [val.sample(eval_rng, cfg["batch_size"])
                            for _ in range(cfg["eval_batches"])]
        else:
            eval_batches = [val.batch(fixed_eval_indices[start:start + cfg["batch_size"]])
                            for start in range(0, len(fixed_eval_indices), cfg["batch_size"])]
        for batch_np in eval_batches:
            if isinstance(transport_policy, RecursiveCarrierPolicy):
                plan = sample_transport(transport_policy, batch_sources(batch_np), tokenizer, layout_rng)
                batch_np, spans, _ = expand_batch(
                    batch_np, plan, transport_policy, tokenizer.carrier_ids, rng=layout_rng)
                if not cfg.get("transport_eval", False):
                    spans = np.full_like(spans, -1)
                batch_np["transport_spans"] = spans
            elif transport_policy is not None and cfg.get("transport_eval", False):
                batch_np["transport_spans"] = source_spans_to_sequence(
                    sample_transport(transport_policy, batch_sources(batch_np), tokenizer, layout_rng))
            else:
                batch_np["transport_spans"] = np.full(
                    (len(batch_np["x"]), 0, 3), -1, dtype=np.int32)
            batch = as_mx(batch_np)
            logits = model(batch["x"], batch["valid"], batch["prefix_lengths"],
                           batch["output_positions"],
                           batch["transport_spans"], cfg.get("eval_sliding_window")).astype(mx.float32)
            if cfg.get("board_eval", False):
                mx.eval(logits)
                predicted = np.asarray(mx.argmax(logits, axis=-1))
                parsed, valid, squares, metadata = board_state_metrics(
                    predicted,
                    np.asarray(batch["output_y"]),
                    np.asarray(batch["output_mask"]), tokenizer)
                parseable_boards += parsed
                valid_boards += valid
                square_errors += squares
                metadata_errors += metadata
            losses = nn.losses.cross_entropy(logits, batch["output_y"])
            nll += mx.sum(losses * batch["output_mask"])
            token_correct = mx.argmax(logits, axis=-1) == batch["output_y"]
            correct += mx.sum(token_correct * batch["output_mask"])
            tokens += mx.sum(batch["output_mask"])
            example_correct = mx.sum(token_correct * batch["output_mask"], axis=-1)
            example_tokens = mx.sum(batch["output_mask"], axis=-1)
            perfect_examples += mx.sum(example_correct == example_tokens)
            byte_count += mx.sum(batch["target_bytes"])
            source_bytes += mx.sum(batch["source_bytes"])
            source_tokens += mx.sum(batch["source_tokens"])
            examples += batch["x"].shape[0]
        student_metrics = None
        if cfg.get("student_board_eval", cfg.get("board_eval", False)):
            count = (len(fixed_eval_indices) if fixed_eval_indices is not None else
                     min(cfg["eval_batches"] * cfg["batch_size"], len(val)))
            student_indices = (fixed_eval_indices if fixed_eval_indices is not None else
                               eval_rng.choice(len(val), count, replace=False))
            student_metrics = student_board_metrics(
                model, tokenizer, val, student_indices, cfg["batch_size"],
                carrier_policy=transport_policy, transport=cfg.get("transport_eval", False),
                sliding_window=cfg.get("eval_sliding_window"))
        mx.eval(nll, correct, tokens, perfect_examples, byte_count,
                source_bytes, source_tokens, examples)
        result = {"eval_sliding_window": cfg.get("eval_sliding_window"),
                  "eval_transport": cfg.get("transport_eval", False),
                  "eval_scored_eviction": bool(cfg.get("scored_eviction")),
                  "target_nll_per_token": float(nll.item() / tokens.item()),
                "target_nll_per_byte": float(nll.item() / byte_count.item()),
                "target_token_accuracy": float(correct.item() / tokens.item()),
                "target_exact_match": float(perfect_examples.item() / examples.item()),
                "source_tokens_per_example": float(source_tokens.item() / examples.item()),
                "source_bytes_per_token": float(source_bytes.item() / source_tokens.item())}
        if cfg.get("board_eval", False):
            result["target_fen_parseable_rate"] = parseable_boards / examples.item()
            result["target_board_valid_rate"] = valid_boards / examples.item()
            result["target_invalid_board_rate"] = 1.0 - valid_boards / examples.item()
            result["target_mean_board_square_error"] = (
                square_errors / parseable_boards if parseable_boards else 64.0)
            result["target_mean_metadata_errors"] = (
                metadata_errors / parseable_boards if parseable_boards else 3.0)
        if student_metrics is not None:
            parsed, valid, exact, squares, metadata = student_metrics
            result["student_target_fen_parseable_rate"] = parsed / count
            result["student_target_board_valid_rate"] = valid / count
            result["student_target_exact_match"] = exact / count
            result["student_target_mean_board_square_error"] = (
                squares / parsed if parsed else 64.0)
            result["student_target_mean_metadata_errors"] = (
                metadata / parsed if parsed else 3.0)
        return result

    def log(record: dict) -> None:
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    output_shape = f"{tokenizer.target.vocab_size:,}-way softmax"
    print(f"params={parameter_count / 1e6:.2f}M input_vocab={tokenizer.vocab_size:,} "
          f"output={output_shape} "
          f"train={len(train):,} val={len(val):,}", flush=True)
    if cfg.get("random_cache_dir"):
        print(f"train mixture: {1 - train.auxiliary_weight:.0%} primary / "
              f"{train.auxiliary_weight:.0%} random legal", flush=True)
    if args.eval_only:
        if not args.resume:
            checkpoint = latest_checkpoint(run_dir)
            load_model_checkpoint(checkpoint, model)
        else:
            checkpoint = (latest_checkpoint(run_dir) if args.resume == "auto"
                          else Path(args.resume))
            load_model_checkpoint(checkpoint, model)
        print(json.dumps({"event": "eval", "checkpoint": str(checkpoint),
                          "step": json.loads(checkpoint.with_suffix(".json").read_text())["step"],
                          **evaluate()}), flush=True)
        return
    started = time.time()
    def elapsed() -> float:
        return prior_elapsed + time.time() - started

    logged_cf_delta = logged_cf_better = logged_cf_count = 0.0
    for step in range(start_step + 1, cfg["steps"] + 1):
        gradients = None
        train_loss = 0.0
        counterfactual_delta = counterfactual_better = counterfactual_count = 0.0
        for _ in range(cfg["grad_accum"]):
            batch_np = train.sample(rng, cfg["batch_size"])
            if training_readouts > 1:
                prepared = prepare_readouts(
                    batch_np, transport_policy, tokenizer, policy_rng,
                    readout_rng, training_readouts,
                    cfg.get("min_readout_plies", 8), cfg["max_target_tokens"],
                    negative_probability, cfg["layers"], full_attention_share)
                sliding_windows = sample_train_windows(
                    policy_rng, train_sliding_window, len(prepared["source_x"]),
                    prepared["full_rows"], full_window)
                batch = as_mx({key: value for key, value in prepared.items()
                               if isinstance(value, np.ndarray)})
                (loss, cf_delta, cf_better, cf_count), grad = readout_value_and_grad(
                    model, batch["source_x"], batch["source_valid"],
                    batch["spans"], batch["points"], batch["branch_x"],
                    batch["branch_y"], batch["branch_valid"], sliding_windows,
                    batch["negative"], batch["full_rows"])
                counterfactual_delta += float(cf_delta.item())
                counterfactual_better += float(cf_better.item())
                counterfactual_count += float(cf_count.item())
                target_count = int(batch["branch_valid"].sum().item())
                model_count = (int(batch["source_valid"].sum().item())
                               + target_count)
            else:
                batch_np, full_rows = training_batch(
                    batch_np, transport_policy, tokenizer, policy_rng,
                    full_attention_share)
                sliding_windows = sample_train_windows(
                    policy_rng, train_sliding_window, len(batch_np["x"]),
                    full_rows, full_window)
                batch = as_mx(batch_np)
                loss, grad = value_and_grad(
                    model, batch["x"], batch["output_y"], batch["valid"],
                    batch["prefix_lengths"], batch["output_positions"],
                    batch["output_mask"], batch["transport_spans"],
                    sliding_windows)
                target_count = int(batch["output_mask"].sum().item())
                model_count = int(batch["valid"].sum().item())
            grad = tree_map(lambda value: value / cfg["grad_accum"], grad)
            gradients = grad if gradients is None else tree_map(
                lambda left, right: left + right, gradients, grad)
            train_loss += float(loss.item()) / cfg["grad_accum"]
            target_tokens_seen += target_count
            model_tokens_seen += model_count
        logged_cf_delta += counterfactual_delta
        logged_cf_better += counterfactual_better
        logged_cf_count += counterfactual_count
        gradients, grad_norm = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        if step == 1 or step % 20 == 0:
            record = {"event": "train", "step": step, "loss": train_loss,
                      "grad_norm": float(grad_norm.item()),
                      "target_tokens_seen": target_tokens_seen,
                      "model_tokens_seen": model_tokens_seen,
                      "elapsed_s": elapsed(),
                      "active_memory_mib": mx.get_active_memory() / 1024**2,
                      "cache_memory_mib": mx.get_cache_memory() / 1024**2,
                      "peak_memory_mib": mx.get_peak_memory() / 1024**2}
            if logged_cf_count:
                record["counterfactual_mean_loss_delta"] = (
                    logged_cf_delta / logged_cf_count)
                record["counterfactual_better_rate"] = (
                    logged_cf_better / logged_cf_count)
                record["counterfactual_count"] = int(logged_cf_count)
            log(record)
            print(json.dumps(record), flush=True)
            logged_cf_delta = logged_cf_better = logged_cf_count = 0.0
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            record = {"event": "eval", "step": step,
                      "target_tokens_seen": target_tokens_seen,
                      "model_tokens_seen": model_tokens_seen,
                      **evaluate()}
            log(record)
            print(json.dumps(record), flush=True)
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            checkpoint = run_dir / f"checkpoint-{step:07d}.npz"
            save_checkpoint(checkpoint, model, optimizer, {
                "step": step,
                "target_tokens_seen": target_tokens_seen,
                "model_tokens_seen": model_tokens_seen,
                "elapsed_s": elapsed(),
                "numpy_rng_state": rng.bit_generator.state,
                "policy_rng_state": policy_rng.bit_generator.state,
                "readout_rng_state": readout_rng.bit_generator.state,
                "training_config": run_config,
                "parameter_count": parameter_count,
            })
            (run_dir / "latest.json").write_text(json.dumps({
                "checkpoint": checkpoint.name, "step": step}, indent=2) + "\n")
            rotate_checkpoints(run_dir, cfg.get("keep_checkpoints", 2),
                               cfg.get("checkpoint_milestone_every", 0))


if __name__ == "__main__":
    main()
