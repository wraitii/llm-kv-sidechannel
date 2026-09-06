"""Source/target tokenizers with disjoint lexical ID namespaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Protocol

from tokenizers import Tokenizer
from tokenizers import Regex
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<fen>", "<eos>"]
PAD, BOS, FEN_QUERY, EOS = range(4)
# Backward-compatible internal name.  ID 2 is now explicitly the query
# "represent the board as FEN now", not a generic source/target separator.
SEP = FEN_QUERY
BYTE_OFFSET = len(SPECIAL_TOKENS)


class TextTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class BytesTokenizer:
    """Lossless UTF-8 tokenizer with four reserved special IDs."""

    vocab_size = BYTE_OFFSET + 256

    def encode(self, text: str) -> list[int]:
        return [value + BYTE_OFFSET for value in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        values = bytes(i - BYTE_OFFSET for i in ids if i >= BYTE_OFFSET)
        return values.decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.inner = Tokenizer.from_file(str(self.path))

    @property
    def vocab_size(self) -> int:
        return self.inner.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=True)


def load_tokenizer(spec: str | Path) -> TextTokenizer:
    return BytesTokenizer() if str(spec) == "bytes" else BPETokenizer(spec)


class PairTokenizer:
    """Map source and target lexical tokens into disjoint model namespaces."""

    def __init__(self, source: TextTokenizer, target: TextTokenizer,
                 pause_token: bool = False, pause_tokens: int | None = None,
                 distinct_pause_tokens: bool = False,
                 causal_pause: bool = False,
                 carrier_vocab: int = 0):
        self.source = source
        self.target = target
        self.pause_tokens = int(pause_token) if pause_tokens is None else pause_tokens
        if self.pause_tokens < 0:
            raise ValueError("pause_tokens must be non-negative")
        if carrier_vocab < 0:
            raise ValueError("carrier_vocab must be non-negative")
        self.pause_token = self.pause_tokens > 0
        self.distinct_pause_tokens = distinct_pause_tokens
        self.causal_pause = causal_pause
        self.target_offset = source.vocab_size - len(SPECIAL_TOKENS)
        self.vocab_size = source.vocab_size + target.vocab_size - len(SPECIAL_TOKENS)
        # Keep this model-only token outside both tokenizer namespaces. It is
        # an input/embedding token, never a target-softmax class.
        self.pause_ids = (list(range(self.vocab_size,
                                     self.vocab_size + self.pause_tokens))
                          if distinct_pause_tokens and self.pause_tokens
                          else ([self.vocab_size] if self.pause_tokens else []))
        self.pause_id = self.pause_ids[0] if self.pause_ids else None
        self.vocab_size += len(self.pause_ids)
        # Memento carrier tokens: model-only scratch positions, cycled by
        # slot index within a carrier group.
        self.carrier_ids = list(range(self.vocab_size,
                                      self.vocab_size + carrier_vocab))
        self.vocab_size += len(self.carrier_ids)

    def carrier_id(self, slot: int) -> int:
        if not self.carrier_ids:
            raise ValueError("no carrier tokens configured")
        return self.carrier_ids[slot % len(self.carrier_ids)]

    def pause_sequence(self) -> list[int]:
        return (self.pause_ids if self.distinct_pause_tokens
                else self.pause_ids * self.pause_tokens)

    def encode_source(self, text: str) -> list[int]:
        return self.source.encode(text)

    def decode_source(self, ids: list[int]) -> str:
        return self.source.decode(ids)

    def encode_target(self, text: str) -> list[int]:
        return self.target_local_to_model(self.encode_target_local(text))

    def encode_target_local(self, text: str) -> list[int]:
        """Target IDs in the fixed target-softmax namespace."""
        return self.target.encode(text)

    def target_local_to_model(self, ids: list[int]) -> list[int]:
        """Map target IDs to their disjoint input-embedding namespace."""
        return [i if i < BYTE_OFFSET else i + self.target_offset for i in ids]

    def decode_target(self, ids: list[int]) -> str:
        local = [i if i < BYTE_OFFSET else i - self.target_offset for i in ids]
        return self.target.decode(local)


def iter_field(path: Path, field: str, max_examples: int = 0) -> Iterable[str]:
    with path.open() as handle:
        for index, line in enumerate(handle):
            if max_examples and index >= max_examples:
                break
            if line.strip():
                yield json.loads(line)[field]


def train_bpe(data: Path, field: str, out: Path, vocab_size: int,
              pretokenizer: str = "standard", max_examples: int = 0) -> None:
    tokenizer = Tokenizer(BPE(unk_token=None))
    if pretokenizer == "standard":
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    elif pretokenizer == "line":
        # Permit merges across spaces and punctuation inside an instruction,
        # but never create the whole-function pair explosion of an unsplit
        # byte stream.
        tokenizer.pre_tokenizer = Sequence([
            Split(Regex(r"\n"), behavior="merged_with_next"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    elif pretokenizer.startswith("block"):
        lines = int(pretokenizer.removeprefix("block"))
        # Match consecutive bounded chunks including their internal newlines.
        # The final non-newline-terminated instruction remains its own chunk.
        pattern = Regex(rf"(?:[^\n]*\n){{1,{lines}}}|[^\n]+$")
        tokenizer.pre_tokenizer = Sequence([
            Split(pattern, behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    elif pretokenizer == "function":
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    elif pretokenizer == "chess":
        # Keep each UCI move (and its following separator) as the atomic
        # pre-tokenization unit. This prevents BPE from memorizing long game
        # prefixes while still allowing frequent move chunks to merge.
        tokenizer.pre_tokenizer = Sequence([
            Split(Regex(r"\S+ ?"), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        raise ValueError(f"unknown pretokenizer: {pretokenizer}")
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                         special_tokens=SPECIAL_TOKENS,
                         initial_alphabet=ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator(iter_field(data, field, max_examples), trainer=trainer)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"wrote {out} ({tokenizer.get_vocab_size():,} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=["source", "target", "chess"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--pretokenizer",
                        choices=["standard", "line", "block2", "block4", "block8", "function", "chess"],
                        default="standard")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="train on at most this many rows (0 means all)")
    args = parser.parse_args()
    field = {"source": "asm", "target": "code", "chess": "history"}[args.side]
    train_bpe(args.data, field,
              args.out, args.vocab_size, args.pretokenizer, args.max_examples)


if __name__ == "__main__":
    main()
