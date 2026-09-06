"""Shared construction of configured tokenizers and models."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .model import model_from_config
from .tokenizer import PairTokenizer, load_tokenizer


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("attention_mode", "causal") != "causal":
        raise ValueError("only causal attention is supported")
    return config


def dtype_for(name: str):
    try:
        return {"bf16": mx.bfloat16, "fp16": mx.float16,
                "fp32": mx.float32}[name]
    except KeyError as error:
        raise ValueError(f"unknown dtype: {name}") from error


def tokenizer_from_config(config: dict) -> PairTokenizer:
    return PairTokenizer(load_tokenizer(config["source_tokenizer"]),
                         load_tokenizer(config["target_tokenizer"]),
                         carrier_vocab=int(config.get("carrier_vocab", 0)))


def model_and_tokenizer(config: dict):
    tokenizer = tokenizer_from_config(config)
    model = model_from_config(tokenizer.source.vocab_size,
                              tokenizer.target.vocab_size, config,
                              dtype_for(config["dtype"]))
    return model, tokenizer
