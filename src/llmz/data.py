"""Paired ASM/C data loading and prefix-LM batch construction."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .tokenizer import BOS, EOS, PAD, SEP, PairTokenizer


def batch_examples(examples: list[tuple], causal_pause: bool = False) -> dict[str, np.ndarray]:
    """Pack variable-length cached or freshly-tokenized pairs."""
    width = max(len(example[0]) - 1 for example in examples)
    x = np.full((len(examples), width), PAD, dtype=np.int32)
    y = np.full_like(x, PAD)
    loss_mask = np.zeros_like(x, dtype=np.float32)
    valid = np.zeros_like(x, dtype=np.bool_)
    prefix_lengths = np.zeros((len(examples),), dtype=np.int32)
    target_bytes = np.zeros((len(examples),), dtype=np.int32)
    source_bytes = np.zeros((len(examples),), dtype=np.int32)
    source_tokens = np.zeros((len(examples),), dtype=np.int32)
    target_tokens = np.zeros((len(examples),), dtype=np.int32)
    max_outputs = max(example[6] for example in examples)
    output_positions = np.zeros((len(examples), max_outputs), dtype=np.int32)
    output_y = np.full((len(examples), max_outputs), PAD, dtype=np.int32)
    output_mask = np.zeros((len(examples), max_outputs), dtype=np.float32)
    for row, (sequence, target_local, sep_index, byte_count, source_byte_count,
              source_token_count, target_token_count) in enumerate(examples):
        length = len(sequence) - 1
        target_start = length - target_token_count
        x[row, :length] = sequence[:-1]
        y[row, target_start:length] = np.array([*target_local, EOS], dtype=np.int32)
        valid[row, :length] = True
        loss_mask[row, target_start:length] = 1.0
        # The pause is a model-only input token. Treat it as the final
        # bidirectional prefix position while its position remains excluded
        # from the loss (its hidden state predicts the first target token).
        prefix_lengths[row] = (sep_index if causal_pause and target_start > sep_index
                               else target_start + int(target_start > sep_index))
        target_bytes[row] = byte_count
        source_bytes[row] = source_byte_count
        source_tokens[row] = source_token_count
        target_tokens[row] = target_token_count
        output_positions[row, :target_token_count] = np.arange(
            target_start, target_start + target_token_count, dtype=np.int32)
        output_y[row, :target_token_count] = np.array([*target_local, EOS], dtype=np.int32)
        output_mask[row, :target_token_count] = 1.0
    return {"x": x, "y": y, "loss_mask": loss_mask, "valid": valid,
            "prefix_lengths": prefix_lengths, "target_bytes": target_bytes,
            "source_bytes": source_bytes, "source_tokens": source_tokens,
            "target_tokens": target_tokens, "output_positions": output_positions,
            "output_y": output_y, "output_mask": output_mask}


class PairDataset:
    def __init__(self, path: str | Path, tokenizer: PairTokenizer,
                 max_source_tokens: int, max_target_tokens: int):
        self.path = Path(path)
        with self.path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = [row for row in rows
                     if len(tokenizer.encode_source(row["asm"])) <= max_source_tokens
                     and len(tokenizer.encode_target_local(row["code"])) <= max_target_tokens]
        self.skipped_overlong = len(rows) - len(self.rows)
        if not self.rows:
            raise ValueError(f"no examples in {self.path}")
        self.tokenizer = tokenizer
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def encode(self, index: int) -> tuple[list[int], int, int, int, int, int]:
        row = self.rows[index]
        source = self.tokenizer.encode_source(row["asm"])
        target_local = self.tokenizer.encode_target_local(row["code"])
        target = self.tokenizer.target_local_to_model(target_local)
        sequence = [BOS, *source, SEP]
        if self.tokenizer.pause_tokens:
            sequence.extend(self.tokenizer.pause_sequence())
        sequence.extend([*target, EOS])
        sep_index = len(source) + 1
        retained_source = self.tokenizer.decode_source(source)
        retained_target = self.tokenizer.decode_target(target)
        return (sequence, target_local, sep_index, len(retained_target.encode("utf-8")),
                len(retained_source.encode("utf-8")), len(source), len(target) + 1)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return batch_examples([self.encode(int(i)) for i in indices],
                              self.tokenizer.causal_pause)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        return self.batch(rng.integers(0, len(self), size=batch_size))


class CachedPairDataset:
    """Memory-mapped token pairs created by ``llmz-cache``."""

    def __init__(self, cache_dir: str | Path, split: str, tokenizer: PairTokenizer,
                 bucket_size: int = 0):
        self.cache_dir = Path(cache_dir)
        self.split_dir = self.cache_dir / split
        metadata = json.loads((self.cache_dir / "metadata.json").read_text())
        split_meta = metadata["splits"][split]
        self.max_source_tokens = metadata["max_source_tokens"]
        self.max_target_tokens = metadata["max_target_tokens"]
        if metadata["source_vocab_size"] != tokenizer.source.vocab_size:
            raise ValueError("cache source vocabulary does not match configured tokenizer")
        if metadata["target_vocab_size"] != tokenizer.target.vocab_size:
            raise ValueError("cache target vocabulary does not match configured tokenizer")
        from .cache import fingerprint
        for side in ("source", "target"):
            expected = metadata.get(f"{side}_tokenizer_fingerprint")
            tok = getattr(tokenizer, side)
            actual = fingerprint(str(tok.path) if hasattr(tok, "path") else "bytes")
            if expected is not None and expected != actual:
                raise ValueError(f"cache {side} tokenizer fingerprint mismatch")
        self.tokenizer = tokenizer
        self.rows = split_meta["rows"]
        dtype = np.dtype(metadata["token_dtype"])
        self.source = np.memmap(self.split_dir / "source.bin", mode="r", dtype=dtype)
        self.target = np.memmap(self.split_dir / "target.bin", mode="r", dtype=dtype)
        self.source_offsets = np.load(self.split_dir / "source-offsets.npy", mmap_mode="r")
        self.target_offsets = np.load(self.split_dir / "target-offsets.npy", mmap_mode="r")
        self.source_bytes = np.load(self.split_dir / "source-bytes.npy", mmap_mode="r")
        self.target_bytes = np.load(self.split_dir / "target-bytes.npy", mmap_mode="r")
        # Legacy caches truncated silently. Exact-limit rows cannot be told
        # apart from truncated rows, so conservatively exclude both in v1.
        source_lengths = np.diff(self.source_offsets)
        target_lengths = np.diff(self.target_offsets)
        eligible = np.ones(self.rows, dtype=bool)
        if metadata.get("format_version", 1) < 2:
            eligible = ((source_lengths < self.max_source_tokens)
                        & (target_lengths < self.max_target_tokens))
        self.indices = np.flatnonzero(eligible)
        self.skipped_overlong = self.rows - len(self.indices)
        if self.skipped_overlong:
            warnings.warn(f"{self.split_dir}: excluded {self.skipped_overlong} legacy "
                          "cache rows at truncation limits; rebuild a v2 cache to retain exact-limit rows",
                          stacklevel=2)
        self.rows = len(self.indices)
        if not self.rows:
            raise ValueError(f"no eligible examples in {self.split_dir}")
        self.bucket_size = bucket_size
        self.length_order = None
        if bucket_size:
            source_lengths = np.diff(self.source_offsets)
            target_lengths = np.diff(self.target_offsets)
            self.length_order = np.argsort((source_lengths + target_lengths)[self.indices], kind="stable")

    def __len__(self) -> int:
        return self.rows

    def encode(self, index: int) -> tuple:
        index = int(self.indices[index])
        source = self.source[self.source_offsets[index]:self.source_offsets[index + 1]].tolist()
        target_local = self.target[self.target_offsets[index]:self.target_offsets[index + 1]].tolist()
        target = self.tokenizer.target_local_to_model(target_local)
        sequence = [BOS, *source, SEP]
        if self.tokenizer.pause_tokens:
            sequence.extend(self.tokenizer.pause_sequence())
        sequence.extend([*target, EOS])
        sep_index = len(source) + 1
        return (sequence, target_local, sep_index, int(self.target_bytes[index]),
                int(self.source_bytes[index]), len(source), len(target) + 1)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return batch_examples([self.encode(int(i)) for i in indices],
                              self.tokenizer.causal_pause)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        if not self.bucket_size:
            indices = rng.integers(0, len(self), size=batch_size)
        else:
            # A uniform anchor in the sorted population selects buckets in
            # proportion to their size; sampling inside it keeps examples
            # approximately uniform while sharply reducing padding.
            anchor = int(rng.integers(0, len(self)))
            start = (anchor // self.bucket_size) * self.bucket_size
            stop = min(start + self.bucket_size, len(self))
            positions = rng.integers(start, stop, size=batch_size)
            indices = self.length_order[positions]
        return self.batch(indices)


class MixedPairDataset:
    """Sample paired datasets at a fixed per-example mixture weight."""

    def __init__(self, primary, auxiliary, auxiliary_weight: float):
        if not 0.0 <= auxiliary_weight <= 1.0:
            raise ValueError("auxiliary_weight must be between zero and one")
        if (primary.max_source_tokens != auxiliary.max_source_tokens or
                primary.max_target_tokens != auxiliary.max_target_tokens):
            raise ValueError("mixed datasets must share source and target limits")
        self.primary = primary
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.tokenizer = primary.tokenizer
        self.max_source_tokens = primary.max_source_tokens
        self.max_target_tokens = primary.max_target_tokens

    def __len__(self) -> int:
        return len(self.primary) + len(self.auxiliary)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        auxiliary_count = int(rng.binomial(batch_size, self.auxiliary_weight))
        primary_count = batch_size - auxiliary_count
        examples = [self.primary.encode(int(index)) for index in
                    rng.integers(0, len(self.primary), size=primary_count)]
        examples.extend(self.auxiliary.encode(int(index)) for index in
                        rng.integers(0, len(self.auxiliary), size=auxiliary_count))
        # Avoid tying source type to a fixed batch position.
        rng.shuffle(examples)
        return batch_examples(examples, self.tokenizer.causal_pause)
