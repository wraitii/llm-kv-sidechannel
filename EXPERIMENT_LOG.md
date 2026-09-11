# Experiment log

This is a concise record of the experimental sequence and current rationale.
Commands and configuration details remain in `torch/README.md` and
`torch/configs/`.

## Earlier KV-leakage runs

- Built fixed-, full-, and variable-attention controls for the synthetic
  passcode and state-management tasks.
- Tested multiple context sizes. The runs found essentially no recoverable KV
  leakage at any tested size, and the full-attention controls did not materially
  change that conclusion.
- These results motivated explicitly training memory behavior rather than
  relying on it to emerge from the base model.

## First Memento-style run

- Added paired `event` and `fixed` memory layouts derived from identical base
  episodes. `event` inserts memory after task updates; `fixed` inserts it at an
  approximately 20x compression ratio.
- Used 16 repeated Qwen `<|fim_pad|>` tokens per memory span, without block
  markers. The attention policy treats everything since the previous completed
  memory as the current block.
- Generated the 3K-context paired dataset and trained Qwen3-1.7B-Base for 500
  steps on the fixed layout with normalized 90% task / 10% prompt-LM loss.
- A 32-example evaluation showed lower passcode NLL when source-conditioned
  memory KVs were preserved than when raw memory token IDs were replayed after
  restart. State-task pairwise accuracy remained weak, and greedy answers were
  incorrect. This is evidence that the memory states carried some contextual or
  task-format information, but not convincing target-specific recovery.
- The restart branch is an ablation: it replays memory token IDs without their
  source-conditioned KVs. In sampled passcode outputs it collapsed to a generic
  `" 1. 1. "` response.

## Causally supervised memory copies

- Repeated placeholder tokens do not force the model to read the preceding
  block in order to predict the memory span. Added the `fixed-copy` layout to
  provide that causal training signal.
- A complete 320-token ordinary block is followed by a 16-token memory span:
  one literal `<|fim_pad|>` sentinel and 15 exact token IDs copied at 20-token
  intervals from the preceding block.
- Each block independently receives one deterministic seeded phase `p_b`; its
  copied offsets are `p_b, p_b+20, ..., p_b+280`. Counterfactual variants share
  the same phase sequence. There is no per-copy jitter or episode-wide phase.
- Serialized prompts retain sentinel placeholders for reliable character-span
  accounting. Tokenization substitutes the recorded source IDs directly, so
  BPE decode/re-encode cannot alter them. Incomplete final blocks remain live
  tails and receive no memory span.
- One known tradeoff is that the randomly selected phase is not announced, so
  the first copied target has phase uncertainty. Later copied targets can use
  the already emitted span. A future variant could expose or deterministically
  derive the phase if fully predictable memory targets are desired.

## Current state

- `fixed-copy` implementation, tests, documentation, and the matching 3K
  90/10 training configuration were pushed to `main` in commit `532904b`.
- The Torch suite passes 76 tests.
- Regeneration of `data/pg19-3000-memory-fixed-copy` is in progress on the
  current Vast instance. Validate its manifest, split counts, paired phase
  alignment, and actual token substitutions before training.
