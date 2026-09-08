# llm-pseudorecurrent

Experiments on information carried from evicted tokens into surviving Transformer
KVs. The controlled task is UCI move history → terminal FEN, using a small MLX
causal Transformer and separate source/target BPE vocabularies.

The central comparison is whether ordinary-token representations can serve as
useful memory when training and inference restrict direct access to old tokens.
Dedicated carrier tokens are an optional experimental condition.

## Qwen/PyTorch port

The CUDA-oriented Qwen3-1.7B work lives in [`torch/`](torch/README.md) with its
own Python environment and lockfile. It targets full attention, fixed and
variable SWA, streaming-log retention, and optionally learned scored retention;
Memento masks and carrier tokens are not part of that port. The original MLX
implementation remains the reference for intervention semantics.

## Implemented conditions

- **Full causal attention**, including a continued-finetuning control.
- **Fixed SWA** and **variable SWA**. Variable SWA samples one window per example;
  it does not fluctuate within a sequence.
- **Memento-style masks on ordinary tokens**: contiguous hidden blocks followed
  by surviving ordinary tokens, optionally with recursive survivor eviction.
- **Inserted carriers**: model-only scratch tokens after hidden blocks. These
  add positions and computation; they are not textual summaries.
- **Learned scored eviction**: each layer protects its recent `W` positions and
  retains up to `M` older entries. When a token exits the recent window, it
  competes with the previous memory entries. Scores depend on the cached value
  representation and the current layer input. Existing memory is reconsidered
  at every step; previously deleted entries cannot return. No retained KV is
  rewritten during normal inference.

Memento policies are **move-aligned**: `alignment: "move"`,
`hidden_moves: [2,3]`, `gap_moves: [0,1]`, and (for ordinary survivors)
`survivor_moves: 1`. Source block sizes are measured in complete UCI moves,
including promotion suffixes and following whitespace. Carrier counts remain
scratch-token counts. Boundaries are recovered from source tokens.

Ordinary survivor moves remain whole through every recursion level; `group_size`
counts survivor moves in that condition. In the explicit-carrier condition it
counts scratch tokens. Alignment prevents partial moves, but does not guarantee
that a block is a semantically complete reasoning episode. SWA itself is still
measured in tokens. Both policies share the recursion code.
Intervening ordinary tokens can also relay information: these masks impose
bottlenecks, not exclusive routing through the designated survivor positions.

For immutable-KV Transformers, each relay consumes model depth. Retaining an old
entry can bridge arbitrary temporal distance within the supported positional
range, but repeated transfers are not an unlimited recurrent state update.

## Controlled retraining

The expensive baseline remains at:

```
runs/baseline/checkpoint-0006000.npz
```

New configs in `configs/controlled/` initialize from that checkpoint and write to
new `runs/controlled-*` directories. They share training data, optimization,
seed, 2,000 finetuning steps, and endpoint-only FEN supervision:

Set `full_attention_share` to mix ordinary full-causal-attention training into a
constrained arm. The choice is made once per microbatch; selected microbatches
bypass transport masks, sliding windows, and scored eviction. For example,
`"full_attention_share": 0.05` gives a 5% full-attention escape mixture.

| Config | Training constraint | Native training-time validation |
| --- | --- | --- |
| `full.json` | Full attention | Full attention |
| `swa32.json` | Window 32 | Window 32 |
| `swa-variable.json` | Window uniformly sampled from 16–48 | Window 32 |
| `streaming-log16x16.json` | 16 recent + 16 older KVs, irreversible log-age thinning over source and FEN | Same streaming policy |
| `memento.json` | Ordinary survivors, recursive masks | Same mask family |
| `memento-swa32.json` | Ordinary survivors + SWA-32 | Both constraints |
| `memento-carriers.json` | Inserted carriers, recursive masks | Same mask family + carriers |
| `scored.json` | 24 recent + 8 older entries per layer | Same scored budget |
| `scored-soft-topk.json` | Soft-TopK with eviction-boundary scoring (delay 24) | Same hard scored budget |
| `scored-soft-topk-delay0.json` | Soft-TopK with immediate entry scoring | Same hard scored budget |

SWA-32, streaming log 16+16, and scored 24+8 each limit visible entries to 32
throughout source and target processing.
Memento masks have a variable visible-token budget. Inserted carriers also alter
sequence length and the number of original moves covered by a fixed window;
those comparisons should not be described as matched-compute experiments.

Streaming log counts all tokens (including BOS and the FEN separator) in its
32-entry budget. On overflow it discards the interior older entry with the
smallest log-age separation between its neighbors, preserving the oldest and
newest older anchors. Discarded KVs never return. Training, validation, and
decoding use the same expiry schedule; attention and cache storage remain dense.
The fresh run initializes from baseline step 6000:

```bash
uv run --locked llmpr-train --config configs/controlled/streaming-log16x16.json
```

For standalone evaluation use `--transport --windows full` for the native
configured policy, or `--streaming-log-budget 64` for a 32+32 capacity override.
Omit both flags to evaluate true full attention with `--windows full`.

```bash
uv sync --extra dev
uv run --locked llmpr-train --config configs/controlled/swa32.json
uv run --locked llmpr-train --config configs/controlled/memento.json
uv run --locked llmpr-train --config configs/controlled/scored.json
uv run --locked llmpr-train --config configs/controlled/scored-soft-topk.json
uv run --locked llmpr-train --config configs/controlled/scored-soft-topk-delay0.json
```

Run whichever arms are relevant. These commands are not required in a particular
order. Existing nonempty run directories require `--resume`; starting a new arm
requires a new `run_dir`. The baseline weights load unchanged; scored arms add
small scoring heads and carrier arms may add embedding rows. Initialization
starts a fresh optimizer; resume requires matching model parameters.

The 6k baseline checkpoint and its original metadata remain under `runs/baseline/`.
`configs/baseline.json` is its evaluation config. Run manifests are immutable,
contain source hashes, and evaluation never overwrites them. Resume invocations
are logged separately. Training data, training masks, and validation use separate
RNG streams; validation uses a fixed seed.

## Paired KV-restart evaluation

`llmpr-eval-chess` uses seeded reservoir sampling from eligible validation or test
rows. Carrier layouts and sampling RNGs are derived from example identity, so
changing batch size does not change them. Generation and teacher-forced evaluation
support transport masks during both prefill and decode.

```bash
# Test ordinary SWA transport on the original baseline.
uv run --locked llmpr-eval-chess \
  --config configs/baseline.json \
  --checkpoint runs/baseline/checkpoint-0006000.npz \
  --examples 512 --temperatures 0 --windows 32 \
  --cache-modes preserve,restart,restart-each --teacher-forced

# Test an ordinary-token Memento checkpoint under its actual eviction masks.
uv run --locked llmpr-eval-chess \
  --config configs/controlled/memento.json \
  --examples 512 --temperatures 0 --windows full --transport \
  --cache-modes preserve,restart --teacher-forced

# Test scored eviction at its own 24+8 budget.
uv run --locked llmpr-eval-chess \
  --config configs/controlled/scored.json \
  --examples 512 --temperatures 0 --windows full \
  --cache-modes preserve,restart --teacher-forced
```

Here `--windows full` means **no additional positional SWA band**. Transport
masks and learned scored retention still operate when enabled. Use
`--disable-scoring --windows full,32` to evaluate a scored checkpoint under
ordinary full/SWA attention. For static policies, `--transport` explicitly opts
into the eviction schedule; omitting it measures transfer to full attention/SWA.
`--no-carriers` disables inserted tokens when evaluating a carrier checkpoint,
but cannot be combined with its carrier-dependent transport schedule.

Cache modes:

- `preserve`: retain the original contextual KVs.
- `restart`: immediately before the final prefix query (normally `<fen>`),
  reconstruct surviving KVs using only surviving key support at each layer.
  Keep original token identities, absolute RoPE positions, and historical mask
  restrictions. Then process the query normally.
- `restart-each`: perform the boundary intervention and repeat reconstruction
  before each subsequently fed output token. This probes transport during FEN
  decoding too; it is not a restart at every source-side eviction event.

For scored models, a preserved shadow computation on the same supplied tokens
fixes all retention decisions. Reconstruction never reranks the candidates. Thus
paired teacher-forced comparisons change KV content while holding token text,
positions, and selection schedules fixed.

`--teacher-forced` additionally measures NLL on identical reference continuations
and emits `paired_transport` records. A **positive** `nll_increase_per_token`
means restarting hurts: preserved contextual KVs were useful. Free-running FEN
metrics complement this measurement but can diverge through generated token text.
Reconstruction is an intervention and may introduce distribution shift; the
paired gap is evidence of reliance on contextual representations, not by itself
proof of a particular learned compression algorithm.

`--split test` selects the test split. `--data-dir data/chess-random-50k` evaluates
random legal games with the same tokenizers. Board square error is conditional
on a parseable six-field FEN; inspect parseability and exact-match alongside it.

## Scorer training details

`scored.json` uses hard top-M during both training and inference. A sigmoid
straight-through gate supplies an experimental, biased gradient approximation
to train the scorer from future target loss. Its attention forward pass uses the
hard selected support, not a larger soft cache.

`scored-soft-topk.json` changes only the training gradient. Its differentiable
Laplace-CDF surrogate shares exactly M units of mass across eligible older
entries, so candidates compete under the actual memory budget. A
straight-through correction makes the forward pass exactly hard top-M during
training as well as inference. The surrogate temperature decays exponentially
from 1.0 to 0.001 over training. No Gumbel noise, budget curriculum, per-head
budgeting, or distillation is included, keeping this an isolated test of the
selection gradient. Retained KVs remain differentiable through ordinary
attention paths under either method, so the model can learn to place useful
information into future survivors.

Each KV entry is scored once. `scoring_delay` controls how many subsequent
tokens are visible to its scoring query: zero scores from the entry's own hidden
state, while the default `recent_window` value scores when it crosses into
memory, preserving the original behavior. Intermediate delays such as 4 or 8
are supported; values must lie between zero and `recent_window`. The resulting
priority is stored with the cache entry and reused rather than rescored at every
query. Training computes the top-M support for all sequence positions in one
batched operation, and cached inference assigns newly due scores once per token.

The implementation uses per-layer policies shared across attention heads and a
partial top-k selection rather than sorting the whole cache. Cache
storage remains dense and evicted entries are masked. This intentionally favors
clear experimental semantics over memory savings. There is no claim that the
scorer has learned useful transport until the trained checkpoints are evaluated.
Scored training is a separate arm; combining it with static transport or SWA
training constraints is currently rejected. The attention inspectors capture the
actual hard-support probabilities for scored models as well.

## Data and cache integrity

Game-level splits keep state probes from one game together. Training uses a
75% Lichess / 25% random-legal mixture in the supplied configs.

Token caches use format v2: overlong source **or target** examples are dropped,
not truncated while retaining an incompatible FEN. The ordinary data loader and
standalone evaluator follow the same no-truncation rule. Tokenizer fingerprints
are checked when loading caches.

```bash
uv run --locked llmpr-cache \
  --data-dir data/chess-2shards \
  --out data/cache/chess-v2 \
  --source-tokenizer artifacts/tokenizers/chess-move-bpe-512-full.json \
  --target-tokenizer artifacts/tokenizers/chess-fen-bpe-512-full.json \
  --max-source-tokens 256 --max-target-tokens 96
```

Update `cache_dir` in a new run config to use the rebuilt cache. The baseline
checkpoint does not need retraining to use these data fixes.

## Checks and inspection

```bash
uv run --locked pytest -q
uv run --locked llmpr-attention-html \
  --config configs/baseline.json --index 3
```

Regression tests cover cached/same-pass equivalence, variable carrier padding,
batch-invariant generation, removal of the old-context channel by restarting,
no-eviction restart equivalence, scorer gradients, hard budgets and irreversible
selection, frozen scored schedules, checkpoint initialization, and run metadata.
