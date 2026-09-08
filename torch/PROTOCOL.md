# Qwen3-1.7B experiment scope

## Phase 1

- Qwen3-1.7B-Base with BF16 LoRA.
- PG-19 background text with position-matched counterfactual state updates.
- Context length 8,192; measure 16,384 only after the 8K path is stable.
- Full-attention common adaptation and continued-full control.
- Fixed SWA and one window sampled per example for variable SWA.
- Answer-only loss and preserve/restart teacher-forced evaluation.

## Phase 2

- Streaming-log retention with matched recent and older-memory budgets.
- A Qwen-scale analogue of the MLX 16+16 condition will choose budgets in Qwen
  tokens after inspecting event coverage; `16+16` is a policy name from the
  small chess tokenizer, not a default Qwen capacity.
- Learned scored retention is a separate optional arm. It must freeze survivor
  schedules during paired restart evaluation.

Memento masks and explicit carrier tokens are not included.

## Required equivalence checks

1. Efficient full/SWA attention matches a dense float32 reference on short
   sequences.
2. Cached token-by-token logits match the corresponding masked full pass.
3. No-eviction preserve and restart logits agree within a frozen tolerance.
4. Restart removes counterfactual divergence once all differing evidence is
   evicted and both variants have identical surviving IDs and positions.
5. Streaming retention respects its capacity and never revives deleted keys.
