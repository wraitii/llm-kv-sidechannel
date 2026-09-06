"""Experiment construction and immutable run provenance."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

from .carriers import expand_batch
from .transport import RecursiveCarrierPolicy
from .move_alignment import sample_transport, batch_sources


def training_batch(batch, policy, tokenizer, rng, full_attention_share=0.0):
    """Return batch and full-attention rows; carrier mixtures use whole batches."""
    count = len(batch["x"])
    if isinstance(policy, RecursiveCarrierPolicy):
        plain = rng.random() < full_attention_share if full_attention_share else False
        full_rows = np.full(count, plain)
        if plain:
            batch = dict(batch)
            batch["transport_spans"] = np.full((count, 0, 3), -1, dtype=np.int32)
        else:
            batch, spans, _ = expand_batch(batch, sample_transport(policy, batch_sources(batch), tokenizer, rng),
                                           policy, tokenizer.carrier_ids, rng=rng)
            batch["transport_spans"] = spans
    else:
        full_rows = rng.random(count) < full_attention_share if full_attention_share else np.zeros(count, bool)
        batch = dict(batch)
        if policy is None:
            spans = np.full((count, 0, 3), -1, dtype=np.int32)
        else:
            spans = sample_transport(policy, batch_sources(batch), tokenizer, rng)
            spans = np.where(spans >= 0, spans + 1, spans).astype(np.int32)
            spans[full_rows] = -1
        batch["transport_spans"] = spans
    return batch, full_rows


def record_run(run_dir: Path, config: dict, resume=False, eval_only=False):
    """Never replace a training manifest, including during evaluation or resume."""
    if eval_only:
        return
    config = {**config, "source_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))}}
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "config.json"
    if not resume:
        if manifest.exists() or any(run_dir.glob("checkpoint-*.npz")):
            raise FileExistsError(f"run already exists: {run_dir}; resume it or choose a new run_dir")
        with manifest.open("x") as handle:
            handle.write(json.dumps(config, indent=2) + "\n")
    with (run_dir / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps({"resume": resume, "config": config}) + "\n")
