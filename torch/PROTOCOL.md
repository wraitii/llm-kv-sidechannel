# Qwen3-1.7B experiment scope

## Phase 1

- Qwen3-1.7B-Base with BF16 LoRA.
- PG-19 background text with position-matched counterfactual state updates.
- Validate the end-to-end lifecycle with a 1,024-token, 500-step pilot first.
- Use 4,096 as the first full-attention experiment target on a 32 GB RTX 5090;
  it is an upper-edge setting and must be reconfirmed on every host. Attempt
  8,192 or 16,384 only on a machine that passes capacity with 10% headroom.
- Full-attention common adaptation and continued-full control.
- Fixed SWA and one window sampled per training row for variable SWA; evaluate
  the latter as a sweep of fixed windows.
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
