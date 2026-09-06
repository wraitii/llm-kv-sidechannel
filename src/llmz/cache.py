"""Build memory-mapped token caches for repeatable baseline training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .tokenizer import PairTokenizer, load_tokenizer


def fingerprint(spec: str) -> str:
    if spec == "bytes":
        return "builtin:bytes:v1"
    digest = hashlib.sha256()
    with Path(spec).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def cache_split(path: Path, out: Path, tokenizer: PairTokenizer,
                max_source: int, max_target: int, dtype: np.dtype) -> dict:
    out.mkdir(parents=True)
    source_offsets = [0]
    target_offsets = [0]
    source_bytes: list[int] = []
    target_bytes: list[int] = []
    started = time.time()
    skipped = 0
    with path.open() as source_jsonl, (out / "source.bin").open("wb") as source_file, \
            (out / "target.bin").open("wb") as target_file:
        for index, line in enumerate(source_jsonl):
            if not line.strip():
                continue
            row = json.loads(line)
            source = tokenizer.encode_source(row["asm"])
            target = tokenizer.encode_target_local(row["code"])
            if len(source) > max_source or len(target) > max_target:
                skipped += 1
                continue
            np.asarray(source, dtype=dtype).tofile(source_file)
            np.asarray(target, dtype=dtype).tofile(target_file)
            source_offsets.append(source_offsets[-1] + len(source))
            target_offsets.append(target_offsets[-1] + len(target))
            source_bytes.append(len(tokenizer.decode_source(source).encode("utf-8")))
            target_bytes.append(len(tokenizer.target.decode(target).encode("utf-8")))
            if (index + 1) % 10_000 == 0:
                print(f"{path.stem}: {index + 1:,} rows ({time.time() - started:.0f}s)",
                      flush=True)
    np.save(out / "source-offsets.npy", np.asarray(source_offsets, dtype=np.uint64))
    np.save(out / "target-offsets.npy", np.asarray(target_offsets, dtype=np.uint64))
    np.save(out / "source-bytes.npy", np.asarray(source_bytes, dtype=np.uint32))
    np.save(out / "target-bytes.npy", np.asarray(target_bytes, dtype=np.uint32))
    return {"rows": len(source_bytes), "skipped_overlong": skipped, "source_tokens": source_offsets[-1],
            "target_tokens": target_offsets[-1]}


def build_cache(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"cache already exists: {args.out}")
    source = load_tokenizer(args.source_tokenizer)
    target = load_tokenizer(args.target_tokenizer)
    tokenizer = PairTokenizer(source, target)
    dtype = np.dtype("uint16" if max(source.vocab_size, target.vocab_size) <= 65535
                     else "uint32")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{args.out.name}-", dir=args.out.parent))
    try:
        splits = {}
        for split in ("train", "val", "test"):
            path = args.data_dir / f"{split}.jsonl"
            if path.exists():
                splits[split] = cache_split(path, temp / split, tokenizer,
                                            args.max_source_tokens,
                                            args.max_target_tokens, dtype)
        metadata = {
            "format_version": 2,
            "overlong_policy": "drop",
            "data_dir": str(args.data_dir),
            "source_tokenizer": args.source_tokenizer,
            "source_tokenizer_fingerprint": fingerprint(args.source_tokenizer),
            "target_tokenizer": args.target_tokenizer,
            "target_tokenizer_fingerprint": fingerprint(args.target_tokenizer),
            "source_vocab_size": source.vocab_size,
            "target_vocab_size": target.vocab_size,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "token_dtype": dtype.name,
            "splits": splits,
        }
        (temp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temp.rename(args.out)
        print(json.dumps(metadata, indent=2), flush=True)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--max-source-tokens", type=int, default=768)
    parser.add_argument("--max-target-tokens", type=int, default=384)
    args = parser.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
