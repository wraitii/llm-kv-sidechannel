# Qwen3-1.7B experiment protocol

## Phase 1

- Qwen3-1.7B-Base with BF16 LoRA.
- PG-19 text with position-matched counterfactual probes. State probes use
  indirect possession swaps every 300--600 tokens. Passcodes have separate
  easy (320--800 tokens from the answer) and hard (10--25% into the document)
  levels; record exact token distances.
- Validate the end-to-end lifecycle with a 1,024-token, 500-step pilot first.
- Use 4,096 as the first full-attention experiment target on a 32 GB RTX 5090;
  it is an upper-edge setting and must be reconfirmed on every host. Attempt
  8,192 or 16,384 only on a machine that passes capacity with 10% headroom.
- Full-attention common adaptation and continued-full control.
- Fixed SWA and one window sampled per training row for variable SWA; evaluate
  the latter as a sweep of fixed windows.
- For constrained runs, use 85% ordinary LM loss under the trained policy, 10%
  answer-only probe loss under that policy, and 5% ordinary full-attention LM
  loss. Match trained tokens and optimizer steps across controls.
- Evaluate clean LM under full and constrained attention. Probe evaluation must
  exclude EOS, compare each correct answer with its paired counterfactual, and
  report margin/accuracy separately by task and distance.
- Always evaluate preserve, restart-at-answer, and restart-every-N (including
  intervals near and below the trained attention window).

## Phase 2

- Streaming-log retention with matched recent and older-memory budgets.
- A Qwen-scale analogue of the MLX 16+16 condition will choose budgets in Qwen
  tokens after inspecting event coverage; `16+16` is a policy name from the
  small chess tokenizer, not a default Qwen capacity.
- Learned scored retention is a separate optional arm. It must freeze survivor
  schedules during paired restart evaluation.

Memento masks and explicit carrier tokens are not included.

## Core ablations

1. Full-attention LM control.
2. Constrained LM without probe loss.
3. Constrained LM plus the 10% probe mixture.

## Required equivalence checks

1. Efficient full/SWA attention matches a dense float32 reference on short
   sequences.
2. Cached token-by-token logits match the corresponding masked full pass.
3. No-eviction preserve and restart logits agree within a frozen tolerance.
4. Restart removes counterfactual divergence once all differing evidence is
   evicted and both variants have identical surviving IDs and positions.
5. Streaming retention respects its capacity and never revives deleted keys.
