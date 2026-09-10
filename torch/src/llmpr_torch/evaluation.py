"""Teacher-forced policy and periodic cache-restart sweeps."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .devices import select_device, training_dtype
from .lora import attach_lora
from .policies import FullAttention, FixedSWA, StreamingLog, VariableSWA, additive_from_visibility
from .tokenization import tokenize_episode
from .state_data import StateEpisode, StateEvent
from .scored import ScoredRetention, enable_scored_retention


Policy = FullAttention | FixedSWA | VariableSWA | StreamingLog | ScoredRetention


def task_loss_aggregates(
    episodes: Sequence[StateEpisode],
    losses_by_episode: Sequence[Sequence[float]],
    *,
    split_task_type: bool = False,
) -> list[dict]:
    """Aggregate teacher-forced answer losses overall or by task type."""
    if len(episodes) != len(losses_by_episode):
        raise ValueError("episodes and losses must have the same length")
    groups = sorted({episode.task_type for episode in episodes}) if split_task_type else [None]
    results = []
    for task_type in groups:
        selected = [
            losses for episode, losses in zip(episodes, losses_by_episode)
            if task_type is None or episode.task_type == task_type
        ]
        losses = [loss for episode_losses in selected for loss in episode_losses]
        result = {
            "examples": len(selected),
            "answer_tokens": len(losses),
            "target_nll_per_token": sum(losses) / max(1, len(losses)),
        }
        if task_type is not None:
            result["task_type"] = task_type
        results.append(result)
    return results


def paired_task_aggregates(episodes: Sequence[StateEpisode],
                           scores: Sequence[tuple[float, float, int]],
                           *, split_task_type: bool = False,
                           distance_buckets: tuple[int, ...] = ()) -> list[dict]:
    """Aggregate correct-versus-counterfactual answer NLL, excluding EOS."""
    def bucket(episode: StateEpisode) -> str | None:
        if not distance_buckets or not episode.support_to_answer_tokens:
            return None
        distance = episode.support_to_answer_tokens[-1]
        lower = 0
        for upper in distance_buckets:
            if distance <= upper:
                return f"{lower}-{upper}"
            lower = upper + 1
        return f"{lower}+"
    task_types = {episode.task_type for episode in episodes}
    separate_tasks = split_task_type or len(task_types) > 1
    groups = sorted({(episode.task_type if separate_tasks else next(iter(task_types)), bucket(episode))
                     for episode in episodes}, key=lambda item: str(item))
    results = []
    for task_type, distance_bucket in groups:
        selected = [score for episode, score in zip(episodes, scores)
                    if (task_type is None or episode.task_type == task_type)
                    and bucket(episode) == distance_bucket]
        tokens = sum(item[2] for item in selected)
        result = {"examples": len(selected), "answer_tokens": tokens, "task_type": task_type}
        if task_type.startswith("passcode"):
            result["answer_nll_per_token"] = (
                sum(item[0] for item in selected) / max(1, tokens))
        else:
            margins = [(alternate - correct) / max(1, count)
                       for correct, alternate, count in selected]
            result.update({
                "correct_answer_nll": sum(item[0] for item in selected) / max(1, tokens),
                "counterfactual_answer_nll": (
                    sum(item[1] for item in selected) / max(1, tokens)),
                "mean_nll_margin": sum(margins) / max(1, len(margins)),
                "pairwise_accuracy": (
                    sum(margin > 0 for margin in margins) / max(1, len(margins))),
            })
        if distance_bucket is not None:
            result["support_distance_bucket"] = distance_bucket
        results.append(result)
    return results


def qwen_layers(model):
    current = model
    for _ in range(5):
        if hasattr(current, "layers"):
            return current.layers
        if hasattr(current, "model"):
            current = current.model
        elif hasattr(current, "base_model"):
            current = current.base_model
        else:
            break
    raise TypeError("could not locate Qwen decoder layers")


@dataclass(frozen=True)
class RestartMode:
    every: int | None = None
    at_answer: bool = False

    @property
    def name(self) -> str:
        if self.at_answer:
            return "restart:answer"
        return "preserve" if self.every is None else f"restart:{self.every}"


def parse_policy(value: str) -> Policy:
    if value == "full":
        return FullAttention()
    if value.startswith("swa:"):
        return FixedSWA(int(value.split(":", 1)[1]))
    if value.startswith("variable-swa:"):
        minimum, maximum = value.split(":", 1)[1].split("-")
        return VariableSWA(int(minimum), int(maximum))
    if value.startswith("log:"):
        recent, memory = value.split(":", 1)[1].split("+")
        return StreamingLog(int(recent), int(memory))
    if value.startswith("scored:"):
        recent, memory = value.split(":", 1)[1].split("+")
        return ScoredRetention(int(recent), int(memory))
    raise ValueError(
        f"unknown policy {value!r}; use full, swa:N, variable-swa:MIN-MAX, "
        "log:R+M, or scored:R+M")


def parse_restart(value: str) -> RestartMode:
    if value == "preserve":
        return RestartMode()
    if value == "restart:answer":
        return RestartMode(at_answer=True)
    if value.startswith("restart:") and int(value.split(":", 1)[1]) > 0:
        return RestartMode(every=int(value.split(":", 1)[1]))
    raise ValueError(f"invalid restart mode: {value}")


def policy_visibility(positions: tuple[int, ...], policy: Policy) -> np.ndarray:
    """Visibility on a possibly sparse list of original absolute positions."""
    pos = np.asarray(positions, dtype=np.int64)
    q, k = pos[:, None], pos[None, :]
    visible = k <= q
    if isinstance(policy, FixedSWA):
        visible &= k > q - policy.window
    elif isinstance(policy, StreamingLog):
        rows_by_query = {int(query): row for row, query in enumerate(pos)}
        visible[:] = False
        for query, alive in enumerate(policy.survivor_schedule(int(pos[-1]) + 1)):
            row = rows_by_query.get(query)
            if row is not None:
                alive_positions = np.intersect1d(
                    pos, np.asarray(alive, dtype=np.int64), assume_unique=True)
                visible[row, np.searchsorted(pos, alive_positions)] = True
    return visible[None]


def last_restart_before(target: int, prompt_length: int, mode: RestartMode) -> int | None:
    """Boundary before the context token that predicts ``target``."""
    if mode.at_answer:
        return prompt_length if target >= prompt_length else None
    if mode.every is None or target < mode.every:
        return None
    return (target // mode.every) * mode.every


def reconstruction_positions(end: int, restart: int, policy: Policy) -> tuple[int, ...]:
    """Raw tokens replayed after discarding contextual KVs at ``restart``."""
    if isinstance(policy, FullAttention):
        support = range(restart)
    elif isinstance(policy, FixedSWA):
        support = range(max(0, restart - policy.window), restart)
    elif isinstance(policy, StreamingLog):
        support = policy.survivors(restart)
    else:
        raise ValueError("scored reconstruction needs the frozen per-layer schedules")
    return tuple(dict.fromkeys([*support, *range(restart, end)]))


def scored_reconstruction_positions(end: int, restart: int, policy: ScoredRetention,
                                    attentions) -> tuple[int, ...]:
    """Use the union of frozen layer-specific survivors as replay input."""
    support = set(range(max(0, restart - policy.recent_tokens), restart))
    older_end = max(0, restart - policy.recent_tokens)
    for attention in attentions:
        scores = attention.frozen_retention_scores[0, :older_end]
        count = min(policy.memory_tokens, len(scores))
        if count:
            support.update(torch.topk(scores, count).indices.tolist())
    return tuple([*sorted(support), *range(restart, end)])


@torch.no_grad()
def token_nlls(
    model,
    input_ids: tuple[int, ...],
    labels: tuple[int, ...],
    prompt_length: int,
    policy: Policy,
    restart_mode: RestartMode,
    device: torch.device,
) -> list[float]:
    """Evaluate answer tokens, rebuilding from raw IDs at requested boundaries.

    This deliberately recomputes dense prefixes. It is a semantic reference,
    not the capacity-efficient CUDA evaluator.
    """
    attentions = [layer.self_attn for layer in qwen_layers(model)]
    scored = isinstance(policy, ScoredRetention)
    if attentions and hasattr(attentions[0], "retention_scorer"):
        for attention in attentions:
            attention.retention_disabled = not scored
            attention.frozen_retention_scores = None
    if scored:
        full_positions = tuple(range(len(input_ids)))
        for attention in attentions:
            attention.llmpr_positions = torch.tensor(full_positions, device=device)
        full_tokens = torch.tensor([input_ids], device=device)
        full_mask = additive_from_visibility(
            policy_visibility(full_positions, FullAttention()), device=device,
            dtype=next(model.parameters()).dtype)
        model(input_ids=full_tokens, position_ids=torch.tensor([full_positions], device=device),
              attention_mask={"full_attention": full_mask}, use_cache=False)
        for attention in attentions:
            attention.frozen_retention_scores = attention.last_retention_scores.clone()
    losses: list[float] = []
    for target in range(1, len(input_ids)):
        if labels[target] == -100:
            continue
        restart = last_restart_before(target, prompt_length, restart_mode)
        positions = (tuple(range(target)) if restart is None else
                     scored_reconstruction_positions(target, restart, policy, attentions)
                     if scored else reconstruction_positions(target, restart, policy))
        tokens = torch.tensor([[input_ids[index] for index in positions]], device=device)
        position_ids = torch.tensor([positions], device=device)
        for attention in attentions:
            if hasattr(attention, "retention_scorer"):
                attention.llmpr_positions = position_ids[0]
        mask = additive_from_visibility(
            policy_visibility(positions, policy), device=device,
            dtype=next(model.parameters()).dtype,
        )
        logits = model(
            input_ids=tokens, position_ids=position_ids,
            attention_mask={"full_attention": mask}, use_cache=False,
            logits_to_keep=1,
        ).logits[0, -1].float()
        loss = F.cross_entropy(logits[None], torch.tensor([input_ids[target]], device=device))
        losses.append(float(loss.cpu()))
    return losses


def load_episodes(path: Path) -> Iterable[StateEpisode]:
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            yield StateEpisode(
                example_id=row["example_id"], pair_id=row["pair_id"], variant=row["variant"],
                prompt=row["prompt"], answer=row["answer"],
                events=tuple(StateEvent(**event) for event in row["events"]),
                query_entity=row["query_entity"], background_id=row["background_id"],
                task_type=row.get("task_type", "state"),
                difficulty=row.get("difficulty", "natural"),
                context_length=row.get("context_length"),
                support_to_answer_tokens=tuple(row.get("support_to_answer_tokens", ())),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--policies", default="full,swa:1024,swa:512,swa:256,swa:128,swa:64")
    parser.add_argument(
        "--restart-modes",
        default="preserve,restart:answer,restart:512,restart:256,restart:128",
    )
    parser.add_argument("--examples", type=int)
    parser.add_argument("--split-task-type", action="store_true",
                        help="emit separate aggregates for each task_type")
    parser.add_argument("--distance-buckets", default="512,1024,2048",
                        help="comma-separated inclusive support-distance upper bounds; empty disables")
    args = parser.parse_args()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    checkpoint_payload = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                          if args.checkpoint else None)
    trained_policy = (parse_policy(checkpoint_payload["config"]["policy"])
                      if checkpoint_payload and "config" in checkpoint_payload else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=training_dtype(device), attn_implementation="eager").to(device).eval()
    if isinstance(trained_policy, ScoredRetention):
        model = enable_scored_retention(model, trained_policy)
    if checkpoint_payload:
        training_config = checkpoint_payload.get("config", {})
        model = attach_lora(model, rank=int(training_config.get("lora_rank", 16)),
                            alpha=int(training_config.get("lora_alpha", 32)))
        for name, parameter in model.named_parameters():
            if ".retention_scorer." in name:
                parameter.requires_grad_(True)
        trainable = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in checkpoint_payload.get("model", checkpoint_payload).items():
                if name not in trainable:
                    raise ValueError(f"checkpoint parameter is absent from model: {name}")
                trainable[name].copy_(value.to(device))
    policies = [parse_policy(value) for value in args.policies.split(",")]
    if any(isinstance(policy, VariableSWA) for policy in policies):
        raise ValueError(
            "variable SWA is a train-only distribution; evaluate its checkpoint at fixed swa:N values")
    if any(isinstance(policy, ScoredRetention) for policy in policies) and not isinstance(trained_policy, ScoredRetention):
        raise ValueError("scored evaluation requires a scored training checkpoint")
    modes = [parse_restart(value) for value in args.restart_modes.split(",")]
    episodes = list(load_episodes(args.data))
    if args.examples is not None:
        episodes = episodes[:args.examples]
    for policy in policies:
        for mode in modes:
            paired_scores: list[tuple[float, float, int]] = []
            alternatives = {episode.pair_id: [] for episode in episodes}
            for episode in episodes:
                alternatives[episode.pair_id].append(episode.answer)
            for episode in episodes:
                encoded = tokenize_episode(tokenizer, episode, max_length=args.max_length)
                choices = [answer for answer in alternatives[episode.pair_id]
                           if answer != episode.answer]
                if len(choices) != 1:
                    raise ValueError(f"pair {episode.pair_id!r} must contain two distinct answers")
                alternate_ids = tokenizer(choices[0], add_special_tokens=False)["input_ids"]
                prompt_ids = encoded.input_ids[:encoded.prompt_length]
                correct_ids = encoded.input_ids[encoded.prompt_length:-1]
                def score(candidate_ids) -> tuple[float, int]:
                    ids = (*prompt_ids, *candidate_ids, tokenizer.eos_token_id)
                    labels = (*([-100] * encoded.prompt_length), *candidate_ids, -100)
                    losses = token_nlls(
                        model, ids, labels, encoded.prompt_length, policy, mode, device)
                    return sum(losses), len(losses)
                correct_loss, correct_count = score(correct_ids)
                alternate_loss, alternate_count = score(alternate_ids)
                if correct_count != alternate_count:
                    raise ValueError("counterfactual answers must have matching token lengths")
                paired_scores.append((correct_loss, alternate_loss, correct_count))
            for aggregate in paired_task_aggregates(
                    episodes, paired_scores, split_task_type=args.split_task_type,
                    distance_buckets=tuple(int(value) for value in args.distance_buckets.split(",")
                                           if value)):
                print(json.dumps({
                    "policy": policy.kind, "policy_config": policy.__dict__,
                    "restart_mode": mode.name, **aggregate,
                }), flush=True)


if __name__ == "__main__":
    main()
