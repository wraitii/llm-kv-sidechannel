"""Learned immutable-priority retention for Qwen3 eager attention."""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


@dataclass(frozen=True)
class ScoredRetention:
    recent_tokens: int
    memory_tokens: int
    scoring_delay: int | None = None
    score_dim: int = 32
    temperature: float = 1.0
    kind: str = "scored"

    def __post_init__(self):
        delay = self.recent_tokens if self.scoring_delay is None else self.scoring_delay
        if (self.recent_tokens < 1 or self.memory_tokens < 1 or self.score_dim < 1
                or self.temperature <= 0 or not 0 <= delay <= self.recent_tokens):
            raise ValueError("invalid scored-retention configuration")
        object.__setattr__(self, "scoring_delay", delay)


class RetentionScorer(nn.Module):
    def __init__(self, hidden_size: int, value_size: int, config: ScoredRetention):
        super().__init__()
        self.config = config
        self.query = nn.Linear(hidden_size, config.score_dim, bias=False)
        self.key = nn.Linear(value_size, config.score_dim, bias=False)
        self.priority = nn.Linear(value_size, 1, bias=False)
        for module in (self.query, self.key, self.priority):
            nn.init.normal_(module.weight, std=0.02)

    def token_scores(self, hidden: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        flat = values.transpose(1, 2).flatten(2)
        delay = int(self.config.scoring_delay)
        query = self.query(hidden[:, delay:]).float()
        candidates = flat if delay == 0 else flat[:, :-delay]
        assigned = ((query * self.key(candidates).float()).sum(-1)
                    * self.config.score_dim ** -0.5
                    + self.priority(candidates).float().squeeze(-1))
        tail = torch.zeros(hidden.shape[0], delay, device=hidden.device, dtype=assigned.dtype)
        return torch.cat((assigned, tail), dim=1)

    def gates(self, hidden: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.token_scores(hidden, values)
        positions = torch.arange(scores.shape[1], device=scores.device)
        return self.gates_from_scores(scores, positions), scores

    def gates_from_scores(self, scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        length = scores.shape[1]
        q = positions[:, None]
        k = positions[None, :]
        causal = k <= q
        recent = causal & (k > q - self.config.recent_tokens)
        older = causal & ~recent
        count = min(self.config.memory_tokens, length)
        ranked = torch.topk(
            scores[:, None, :].expand(-1, length, -1).masked_fill(~older, -torch.inf),
            count, dim=-1).indices
        chosen = torch.zeros_like(older[None].expand(scores.shape[0], -1, -1))
        chosen.scatter_(-1, ranked, True)
        hard = recent[None] | (older[None] & chosen)
        soft = torch.where(recent[None], torch.ones_like(scores[:, None, :]),
                           torch.where(older[None], torch.sigmoid(
                               scores[:, None, :] / self.config.temperature),
                               torch.zeros_like(scores[:, None, :])))
        return soft + (hard.to(soft.dtype) - soft).detach()


def _scored_forward(self, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, **kwargs):
    if past_key_values is not None:
        raise ValueError("scored retention currently requires a same-pass forward")
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    keys, values = repeat_kv(key, self.num_key_value_groups), repeat_kv(value, self.num_key_value_groups)
    weights = torch.matmul(query, keys.transpose(2, 3)).float() * self.scaling
    if attention_mask is not None:
        weights = weights + attention_mask.float()
    if getattr(self, "retention_disabled", False):
        scores = torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
        gates = torch.ones(hidden_states.shape[0], hidden_states.shape[1],
                           hidden_states.shape[1], device=hidden_states.device)
    elif getattr(self, "frozen_retention_scores", None) is not None:
        positions = self.llmpr_positions.to(hidden_states.device)
        scores = self.frozen_retention_scores[:, positions]
        gates = self.retention_scorer.gates_from_scores(scores, positions)
    else:
        gates, scores = self.retention_scorer.gates(hidden_states, value)
    # Exact hard forward with a straight-through multiplicative gate.
    allowed = torch.isfinite(weights) & (weights > torch.finfo(weights.dtype).min / 2)
    center = weights.masked_fill(~allowed, -torch.inf).amax(-1, keepdim=True)
    probabilities = torch.exp((weights - center).clamp(-80, 30)) * gates[:, None]
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-20)
    probabilities = F.dropout(probabilities.to(query.dtype), p=self.attention_dropout,
                              training=self.training)
    output = torch.matmul(probabilities, values).transpose(1, 2).contiguous()
    self.last_retention_scores = scores.detach()
    return self.o_proj(output.reshape(*hidden_states.shape[:-1], -1)), probabilities


def enable_scored_retention(model, config: ScoredRetention):
    """Install per-layer scorers without changing pretrained Qwen parameters."""
    for layer in model.model.layers:
        attention = layer.self_attn
        value_size = model.config.num_key_value_heads * attention.head_dim
        attention.retention_scorer = RetentionScorer(
            model.config.hidden_size, value_size, config).to(
                device=attention.v_proj.weight.device, dtype=attention.v_proj.weight.dtype)
        attention.forward = MethodType(_scored_forward, attention)
        attention.frozen_retention_scores = None
        attention.retention_disabled = False
    model.config.llmpr_scored_retention = config.__dict__
    return model
