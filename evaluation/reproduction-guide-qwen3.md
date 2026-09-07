# Testing contextual KV memory: a standalone reproduction guide

Prepared 7 September 2026. This document explains the experiment, summarizes the
existing chess evidence, and specifies a first reproduction on **Qwen/Qwen3-4B
with a text state-tracking task**. It can be read independently of the repository.

The chess experiments already exist. The Qwen experiment below is a proposed
port and protocol, not a completed run or an existing command-line feature.
The current repository uses a custom 13.1M-parameter MLX Transformer; its training
and evaluation commands do not load Hugging Face Qwen checkpoints. A GPU runner
must implement the cache intervention described here in a Qwen-compatible backend.

## 1. What are we testing?

A Transformer caches a key and value vector (KV) for each processed token at
each layer. Those vectors are contextual: a surviving token's representation
can depend on earlier tokens that are subsequently removed from attention.
Consequently, deleting an old token's own KV does not necessarily delete all
information about it.

Our hypothesis is that training under restricted attention encourages the model
to carry useful information into representations that will survive eviction.
Ordinary input tokens can act as memory; special memory tokens are optional.

The central test holds the surviving token text, original positions, and eviction
decisions fixed, but rebuilds the surviving KVs without access to evicted keys.
If correct answers become less likely, the original contextual representations
were useful beyond the surviving text alone.

This is called “pseudorecurrent” because a later token can inherit information
from an earlier token. In a standard Transformer with immutable cached KVs,
each successive transfer consumes model depth. It is not an indefinitely
updatable recurrent state. Keeping an old KV can bridge temporal distance within
the positional range, but does not update that KV.

For fixed SWA in every layer, an L-layer model's logits at position q can
depend on input positions no earlier than `q - L*(W-1)`. This follows by
expanding the causal window once per layer. With Qwen's 36 layers, W=64 gives
a maximum reach of 2,268 positions before q. Evidence beyond that bound cannot
influence those logits through this architecture. Check reach at the query
that predicts each scored answer token; generating more tokens does not create
an unlimited source-memory channel. This bound does not apply to full attention
or policies that retain arbitrarily old entries.

## 2. The three evaluation modes

Let the tokenized prompt be `p[0], ..., p[q]`, and the reference answer be
`y[0], ..., y[T-1]`. The boundary query `p[q]` is the final prompt token whose
logits predict `y[0]`.

| Mode | Operation |
| --- | --- |
| Preserve (P) | Process the entire sequence under the selected attention/retention policy, keeping original contextual KVs. |
| Restart (R) | Process through `p[q-1]`; rebuild the historical KVs that may survive for query `q`; process `p[q]` normally; continue normally. |
| Restart-each (RE) | Apply the same boundary restart, then rebuild again immediately before feeding each subsequent answer token. |

RE does **not** mean restarting at every eviction during the source prompt.
Both P and R have identical source processing before the boundary intervention.

For teacher forcing, all modes receive the identical reference answer tokens.
Use summed negative log likelihood (natural logs) divided by the number of
scored answer tokens:

```text
NLL(mode) = -sum_i sum_t log p_mode(y_i[t] | prompt_i, y_i[:t]) / sum_i T_i
delta_R  = NLL(R)  - NLL(P)
delta_RE = NLL(RE) - NLL(P)
```

A positive delta means reconstruction hurts. Report absolute NLL too: a large
gap is not useful if both conditions fail the task. Free generation complements
this measurement, but generated tokens can diverge, so it does not isolate KV
content as cleanly as teacher forcing.

Reconstruction changes the distribution of hidden states. A positive gap is
evidence of reliance on contextual KVs, not proof of a specific compression
algorithm, exact state storage, or unlimited recurrence.

## 3. Rebuilding KVs correctly

This is the essential implementation contract for any model port.

1. Keep original token IDs, absolute positions, padding validity, historical
   attention restrictions, and per-layer survivor IDs. Cache storage indices
   and original token positions are separate concepts.
2. Determine which old entries are available **for the next query**, including
   any eviction that adding that query causes. A recent-window budget includes
   the query itself: window `W` permits keys `q-W < k <= q`.
3. Start reconstruction from token embeddings, not preserved hidden states.
   At each layer, allow only that layer's frozen surviving historical keys.
   Also intersect with causality and the original historical SWA/transport
   restrictions for every replayed query position.
4. Apply the model's original normalization, Q/K/V projections, positional
   transforms, attention, residuals, and MLP. Store the rebuilt KVs. Keep
   absolute RoPE positions, including gaps; never renumber survivors from zero.
5. An attention row with no valid keys must contribute zero attention, not a
   uniform average over masked values or NaNs. Preserve normal residual/MLP
   behavior. Mask padding independently.
6. Process the boundary query normally using this rebuilt cache. The boundary
   query's own K/V comes from this new forward pass.

For ordinary SWA, surviving positions agree across layers, so replaying the
surviving suffix with the proper positions and historical masks can implement
this. For layer-dependent selection, replay a common set of token positions
with separate key-support masks per layer. Taking a single intersection of all
layer survivor sets changes the intervention. Dense replay is a useful reference:
discarded entries may remain allocated, but cannot contribute as keys where
they are not selected. Replayed query rows are not an additional key permission.

For learned selection, run a **preserved shadow stream** on the same supplied
tokens. It determines retention for each upcoming query, including selection
after admitting the new token. Force those same choices in the intervention
stream and never rerank reconstructed KVs. With teacher forcing, one shared
reference-token schedule pairs all modes. For free generation, each branch's
shadow follows that branch's generated text; later schedules need not agree
across branches.

Frozen learned survivor identities are themselves a possible information
channel about the discarded history. The scored comparison isolates KV content
conditional on that schedule; it does not remove information in the selection
pattern. With static SWA, this particular confound is absent.

Do not reconstruct by processing the full original history and merely dropping
its old KVs afterward: that restores the information channel being tested.
Likewise, cropping an already constructed cache while prefilling with full
attention does not reproduce SWA throughout the prompt.

## 4. Existing experiment and evidence

The original task maps a chess game's UCI move history to its terminal six-field
FEN (board placement, side to move, castling rights, en-passant field, halfmove
clock, fullmove number). Training supervises only the final FEN, without
intermediate boards or textual summaries.

The model has six layers, width 416, eight query heads and four KV heads, with
separate 512-entry source/target BPE vocabularies. Source and target limits are
256 and 96 tokens. Overlong examples are dropped, not truncated. Game-level
splits prevent positions from one game appearing across splits. The supplied
training mixture is 75% Lichess and 25% random legal games.

Controlled finetunes start from the same baseline step-6000 checkpoint with a
fresh optimizer. Configurations specify 2,000 additional steps, effective batch
64 (16 × 4 accumulation), learning rate 0.0003, 100 warmup steps, minimum rate
0.00003, weight decay 0.1, and seed 1337. These are small-model settings, not
prescribed Qwen hyperparameters. Most reported sweeps use step 1000.

| Training arm | Constraint |
| --- | --- |
| Full control | Continued full causal attention. |
| Fixed SWA | Recent window 32. |
| Variable SWA | Uniform integer window 16–48, sampled once per example. |
| Variable SWA low-full | Window 8–32 plus 5% full-attention microbatches; changes two factors. |
| Streaming log | 16 recent + 16 older entries; irreversible log-age thinning. |
| Memento ordinary | Each period is 2–3 hidden moves, then one survivor move, then 0–1 gap moves. Hidden moves remain readable through their survivor, then expire. Recursively group three survivor moves, depth two. |
| Memento carriers | Insert scratch tokens as survivors, with recursive eviction. These add positions and computation. |
| Scored | Per-layer 24 recent + up to eight older KVs; policy shared across attention heads. |

Streaming log evicts the interior older entry with the smallest log-age separation
between its neighbors, preserving the oldest and newest older anchors. Deleted
entries never return. BOS, separators, and target tokens count toward budgets.

Scored retention assigns each entry one immutable priority. At each step the
previous memory competes with newly eligible entries for top-M retention;
deleted entries cannot return. Default scoring occurs as the entry leaves the
recent window; delay zero scores from its own hidden state. “Reconsidered” means
selection is repeated, not that stored entries receive fresh scores each step.
The original scorer uses a sigmoid straight-through gradient. The soft-top-k
variant uses a budget-coupled Laplace-CDF surrogate with min(M, eligible count) units of mass and
temperature decaying from 1 to 0.001; both have hard top-M forward passes in
training and inference. No ordinary inference step rewrites retained KVs.

Selected recorded teacher-forced results (nats per target token):

| Checkpoint / evaluation | P | R | RE | R − P |
| --- | ---: | ---: | ---: | ---: |
| Baseline 6K, full | 0.100 | 0.100 | 0.100 | 0.000 |
| Baseline 6K, SWA-32 | 3.525 | 3.612 | 3.779 | 0.087 |
| Fixed SWA finetune 1K, SWA-32 | 0.429 | 0.958 | 1.457 | 0.529 |
| Variable SWA finetune 1K, SWA-32 | 0.449 | 0.760 | 1.315 | 0.311 |
| Streaming log finetune 1K, 16+16 | 0.440 | 1.386 | 2.506 | 0.946 |
| Scored finetune 1K, 24+8 | 0.478 | 0.804 | 1.300 | 0.326 |
| Soft-top-k finetune 1K, 24+8 | 0.511 | 0.750 | 1.247 | 0.239 |
| Memento ordinary finetune 1K, native mask | 0.763 | 1.182 | 1.182 | 0.419 |

All rows use the same 32 eligible validation examples, seed 1337, with greedy
generation in the companion evaluation. These are exploratory observations,
not a statistically established ranking. At SWA-32, the fixed-SWA model's
greedy exact FEN count is only 4/32 under preserve. Strong likelihood improvement
does not imply reliable exact state tracking. The baseline and finetunes also
have different total training exposure; use a continued-full control in a new
study. Memento budgets vary and carrier insertion adds compute, so those arms
are not matched-compute comparisons.

The table's NLL includes answer EOS and is weighted by target token count.
Its deltas are calculated from rounded displayed values. Streaming and
soft-top-k rows can also be checked against their raw JSONL files in the
repository; the remaining selected rows are transcribed from the sweep report.
NLL magnitudes across different tasks or tokenizers are not directly comparable.

## 5. A concrete Qwen text task

Start with synthetic object-location tracking. It provides exact labels, needs
information from prior events, and avoids chess notation competence becoming
the main bottleneck. This is a new task testing the same hypothesis, not a
numerical reproduction of the chess table.

Each episode has eight uniquely named objects and eight locations. Initialize
every object, then present chronologically ordered moves. After all events,
request every object's final location in a fixed order. Generate labels with a
plain dictionary simulator; never ask an LLM to label the data.

```text
Track the location of each object. A move replaces its previous location.
Return only object=location pairs, one per line, in the requested order.

Initially, the amber key is in the attic.
Initially, the blue cup is in the kitchen.
[initialize the other six objects]
The amber key moves to the cellar.
The blue cup moves to the garden.
[more events]

Give the final locations in this order: amber key, blue cup, ...
```

The brackets above explain the example; actual prompts contain complete events
and all eight requested names. Targets contain eight real `object=location`
lines, with no commentary or intermediate state.

Suggested initial dataset specification:

- Use this ordered object list: `amber key, blue cup, copper coin, green book,
  red ball, silver ring, white bowl, yellow box`. Use this ordered location
  list: `attic, cellar, garden, hallway, kitchen, office, pantry, bedroom`.
  Use the initialization and move wording above, with one sentence per line,
  initialization and answer fields in object-list order, and the complete
  object list in the final request. Publish the exact prompt renderer.
  Use NumPy `Generator(PCG64(seed))` and record the NumPy version. Sample initial
  locations uniformly. For each event sample an object uniformly and a new
  location uniformly from the other seven locations.
- Use 50,000 training, 2,000 validation, and 2,000 test episodes with generator
  seeds 1337, 1338, and 1339. Deduplicate whole event sequences across splits;
  group any multiple probes from the same episode in a single split.
- Sample event counts uniformly from 32–128 for the primary distribution.
  Generate a separate 129–256-event test set to examine length transfer.
- Tokenize with the exact Qwen tokenizer and rendered chat template. Drop
  complete episodes exceeding an initial 4,096-token prompt-plus-answer cap;
  publish counts and final length distributions. Do not silently truncate.
- Label each answer field with the inclusive token span `[a,b]` of its final
  supporting event (or initialization), mapped into the rendered chat prompt.
  Include any token overlapping the event text. At boundary q, an event is
  entirely outside SWA-W if `b <= q-W`; record `q-b` and evaluate visibility
  again at each answer field's prediction positions.

The uniform task is a smoke test, not a sufficient long-memory benchmark:
after n updates, the probability a particular object has not been updated is
`(7/8)^n`. Its final update will usually be recent even in a long episode.
Do not rely on rejection sampling to obtain extremely rare long gaps.

Add a separate **controlled-gap evaluation** with 512 episodes for each nominal
gap G in `128, 256, 512, 1024`, using seeds `1400+G`. For each episode:

1. Initialize all objects and generate 32 uniform updates as above. Select
   one probe object uniformly and move it to a uniformly chosen different
   location. This is its last update.
2. Append updates sampled uniformly over the other seven objects, excluding
   their current locations when choosing destinations. Never update the probe
   again. After each event, render/tokenize the complete prompt with the final
   question; stop at the first event for which `q-b >= G`, where b is the end
   of the probe's last-update token span.
3. If prompt plus answer exceeds 4,096 tokens, reject the episode and log that
   rejection; otherwise save the episode, probe identity, actual gap, spans,
   and full answer. Continue the same RNG stream until 512 accepted episodes.
4. Evaluate each frozen episode under every budget and mode, and report probe
   accuracy separately from all-object accuracy. A nominal gap G does not
   imply eviction under budgets greater than G. Also flag evidence beyond the
   architectural reach for each window.

Keep this constructed distribution separate from the uniform test distribution.
It tests long-gap transfer after uniform-task training. If you later mix
controlled-gap episodes into training, use disjoint generation seeds, specify
the mixture, and rerun all training arms with that same mixture. Report that as
a separate experiment. In either task, absolute-location updates chiefly test
last-event retrieval and retention; they do not reproduce chess's full state
transition complexity.

An answer obtainable entirely from the surviving suffix is a weak transport
probe. Evaluate such fields as controls, and separately report fields whose
last update has been evicted. For noncontiguous policies, use actual retained
token IDs to characterize direct availability; a distance threshold alone does
not establish that the evidence was deleted.

## 6. Qwen setup and scope of the port

Use the exact checkpoint **`Qwen/Qwen3-4B`**, its native tokenizer, and a pinned
model revision. The model card documents 36 layers and grouped-query attention
with 32 query heads and eight KV heads. It supports a non-thinking chat template
via `enable_thinking=False`; use that setting consistently in this experiment.
These model details and template behavior are documented in the
[official model card](https://huggingface.co/Qwen/Qwen3-4B).

Render one user message containing the task, with `add_generation_prompt=True`
and `enable_thinking=False`. Save the resulting prompt token IDs. Define `q` as
the final token of that rendered prefix, not an assumed textual delimiter.
Use a fixed answer serialization and the tokenizer's actual assistant end token;
record its ID and whether it is included in NLL (recommended: include it).
Do not supervise the prompt. When tokenizing rendered text separately, avoid
adding special tokens a second time. Verify answer boundaries by decoding IDs.

For an unambiguous first implementation, tokenize the rendered generation
prefix and canonical answer separately with `add_special_tokens=False`, then
concatenate their IDs and append exactly one assistant EOS ID. Use precisely
these IDs for training and every evaluation mode; do not retokenize the joined
string in another path, since subword merges may change the boundary. Serialize
the answer as eight LF-separated `object=location` lines with no final newline.
The current official tokenizer uses `<|im_end|>` (151645) as EOS, and its
non-thinking generation prefix supplies the empty thinking block; retain that
whole prefix as prompt. Verify these details against the pinned
[tokenizer configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json).

Build a separate Python environment with a CUDA-compatible PyTorch and a pinned
Transformers release supporting Qwen3; the model card identifies 4.51.0 as the
minimum supporting release. Record the versions actually tested rather than
treating that minimum as a tested environment. Use the
[official Transformers Qwen3 API documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3)
for the selected version, and inspect its installed attention/cache implementation
before patching it.

Begin with explicit dense causal masks and an inspectable attention path.
Preserve Qwen's Q/K normalization, grouped KV heads, RoPE, residuals, and MLPs.
Implement a recent-window mask for both training and every prefill/decode query,
plus reconstruction from section 3. Do not assume a configuration flag or a
generic cache cropping method supplies those semantics. Verify any attention
backend against the dense reference before using it for speed.

During training, the answer-only loss must backpropagate through contextual
source representations across all layers. Do not build the source cache under
`no_grad`, detach its KVs, or substitute independently trained sequence chunks.
Those changes remove or shorten the learning path being tested. A single
masked forward pass with activation checkpointing is the reference approach.
For evaluation, disable gradients.

Required runner capabilities, to be implemented:

```text
render_and_tokenize(episode) -> prompt_ids, answer_ids, original_positions
forward_with_policy(tokens, policy, cache) -> logits, cache, survivor_metadata
restart(cache, next_query_position, frozen_support) -> rebuilt_cache
evaluate_pair(example, policy, modes=[P, R, RE]) -> token_losses, generations
train(episodes, initial_checkpoint, policy, answer_only_loss) -> checkpoint
```

These are interface sketches, not functions currently shipped for Qwen.
Implement fixed/variable SWA first. Streaming and learned retention require
additional scheduling and layer-specific cache machinery. Port Memento only
after the core intervention works; align blocks to whole text events instead
of chess moves, and publish the new block/survivor schedule.

Four billion BF16 weights alone occupy approximately 8 GB (decimal). Training
needs additional gradients, optimizer states, activations, and attention
workspace; dense replay and a shadow stream add evaluation overhead. Measure
peak memory with one short example on the actual GPU before scaling. A bounded
visible KV count in a dense reference implementation is not a claim of bounded
physical memory or faster inference.

## 7. Training and evaluation plan

First evaluate the untouched model under full attention and SWA. This tests
existing contextual reliance. It cannot establish that constraint training
teaches transport.

Then establish a competent full-attention task checkpoint, and branch every
controlled finetune from that same checkpoint. If the full model cannot perform
the task, simplify event lengths or train the common task checkpoint before
interpreting eviction results.

Suggested first controlled run, explicitly new settings rather than inherited
chess settings:

| Arm | Training policy | Evaluation policies |
| --- | --- | --- |
| Full continuation | Full attention | Full and SWA sweep |
| Fixed SWA | W=256 | Full and SWA sweep |
| Variable SWA | Integer W sampled uniformly from 128–384 once per example | Full and SWA sweep |

Use budgets `full, 1024, 512, 256, 128, 64` in **Qwen tokens**. The chess window
32 does not translate into the same semantic coverage with another tokenizer.
Apply the policy to system/template tokens, source, and answers alike; do not
silently exempt instruction tokens or introduce attention sinks.

For a resource-conscious initial experiment, use the same LoRA configuration
in the common task adaptation and all branches: rank 16, alpha 32, dropout 0,
targeting Q/K/V/O projections and the gate/up/down MLP projections. Freeze base
weights, train adapters, and record exact matched module names. This is a
parameter-efficient variant of the original experiment. Keep a fixed adapter
strategy across arms; full-weight finetuning is a separate experiment.
Clone the trained common adapter weights into each branch and initialize a
fresh optimizer; do not restart each branch with an untrained adapter. Pin the
adapter library as well as Transformers; the
[PEFT LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora)
documents adapter configuration and target-module selection.

Suggested pilot optimizer settings are AdamW, learning rate 1e-4, weight decay
0.01, effective batch eight episodes, 50 warmup steps, 1,000 branch updates,
and cosine decay to 1e-5. Use BF16 if supported, gradient accumulation and
checkpointing as needed, and answer-only cross entropy. Save at updates 0,
250, 500, and 1000. Choose the common checkpoint using full-attention validation
performance; record its training schedule separately. Tune pilot settings on
validation before freezing the controlled protocol, not separately on test
results for each arm. These settings have not been validated on this task.

Use AdamW betas (0.9, 0.999), epsilon 1e-8, and gradient norm clipping at 1.0
for this pilot. Normalize accumulated loss by the total number of non-padding
answer tokens in the effective batch, including EOS, so different microbatch
partitions implement the same objective. Cycle through a seeded shuffled
training set as needed, disable example packing initially, and publish the
actual microbatch/accumulation settings. For the common full-attention task
adaptation, start with this same 1,000-step schedule and choose the saved
checkpoint with lowest validation NLL, breaking ties by earlier step. If task
accuracy is inadequate, revise that common pilot before branching; record the
final schedule and accuracy rather than silently extending selected arms.

Keep example order, effective batch, optimizer settings, update count, and
initialization identical across branches. Use independent RNG streams for
data ordering, masks, and evaluation. Different attention patterns need not
have identical wall time: report updates, tokens, trainable parameters, GPU
hours, and peak memory instead of asserting equal compute.

Evaluate 32 fixed validation episodes first to debug, then at least 512 fixed
held-out episodes after freezing choices. Use P/R/RE teacher forcing and greedy
generation with a fixed answer-token limit large enough for complete targets
(verify 128 tokens against the dataset). Mark capped generations explicitly.
Use argmax decoding (`do_sample=False`, one beam) without repetition penalties
or other logits processors, and stop at the single designated assistant EOS.
Report this as a controlled greedy diagnostic, not optimized Qwen sampling.
Use three training seeds if resources allow; keep the same test examples across
seeds. Add the streaming, scorer, and carrier arms only after the first study
passes its correctness checks.

## 8. Checks required before believing a gap

- **No-eviction identity:** full attention, or a budget exceeding the entire
  sequence, gives equal P/R/RE teacher-forced logits within measured numerical
  tolerance. Fix any systematic mismatch first.
- **Prefill/decode agreement:** token-by-token cached execution agrees with a
  single masked causal pass, including across the exact eviction boundary.
- **Positions and masks:** survivors keep original RoPE positions; no future,
  padded, or evicted keys receive attention; empty rows remain finite.
- **Restart removes old context:** construct two prompts with identical
  surviving token IDs at identical positions and identical static masks, but
  different evicted histories. After restart, their logits agree within
  tolerance. Preserve logits may differ. Freeze identical schedules when
  extending this check to scored policies.
- **Budget and irreversibility:** count visible keys per query and layer,
  including the current token. Once evicted, an entry never returns.
- **Pairing:** P/R/RE score identical reference token IDs; learned schedules are
  frozen; evaluation uses model evaluation mode with dropout disabled.
- **Batch invariance:** single-example and padded batched results agree within
  tolerance; example selection and mask layouts depend on stable example IDs.
- **No label leakage:** prompt generation and masks do not use final labels;
  full-attention training microbatches, if later added, are explicitly logged.
  Constructing controlled gaps from event positions is allowed; masks must not
  use the gold locations or adapt to whether an answer is correct.

Calibrate tolerances on tiny deterministic examples by comparing a float32
dense reference with the chosen precision/backend and with cached/replayed
execution. Repeating the exact same deterministic path alone does not measure
roundoff differences between paths. Record maximum logit and NLL errors and
freeze tolerances before the experiment; do not widen them to accommodate a
systematic restart mismatch. Rounded aggregate equality alone is insufficient.

## 9. What to report and send back

For each checkpoint, budget, policy, and mode, report answer NLL, exact full
answer accuracy, per-object location accuracy, formatting success, and capped
generation count. Score missing/malformed fields as incorrect in the primary
accuracy metric; any accuracy conditional on valid formatting is supplementary.
Include direct-evidence availability and event-distance strata.

Define exact match as equality to the canonical answer after removing only the
terminal EOS token; do not strip prose or thinking text. For formatting and
field accuracy, require eight lines in the prescribed order with the exact
object names, one `=` per line, and a location from the fixed list. Score a
missing, duplicated, misplaced, or malformed field as incorrect; count extra
lines as a formatting failure and an exact-match failure. Publish the parser.
Report location-token NLL separately from whole-answer NLL if possible, since
predictable object names and punctuation can dilute the aggregate memory gap.

Save one record per example and condition, including:

```text
example_id, dataset_hash, model_revision, checkpoint_hash, training_seed,
policy, budget, mode, prompt_token_count, target_token_count, dataset_stratum,
target_nll_sum, generated_text, exact_match, correct_fields, total_fields,
format_valid, generation_capped, survivor_schedule_hash,
probe_object, support_spans, per_field_correct, per_token_target_losses
```

Compute paired confidence intervals by bootstrapping **episodes**, keeping
all modes for an episode together. For the token-weighted NLL gap, resample
episodes and recompute total loss difference divided by total target tokens
in every bootstrap sample. Do not treat answer tokens as independent samples.
Report variation across training seeds separately.

The most persuasive pattern is: constrained training improves preserve accuracy
and NLL at the same budget relative to the continued-full control, restarting
removes some of that improvement on fields whose evidence was evicted, and
the no-eviction controls remain equal. A larger restart gap alone is not a win.
A null gap can mean no useful transport, redundant surviving evidence, a weak
task, or an implementation issue; inspect those possibilities before concluding.

The handoff should include the frozen dataset and generator, environment lock,
model/tokenizer revisions, adapter/checkpoint files, complete configs, code
snapshot, raw per-example results, aggregate tables with paired intervals, and
hardware/memory/runtime measurements. With these files, another person should
be able to rerun the comparison without reconstructing choices from chat.

## 10. Optional: rerun the existing chess implementation

These commands are for the original MLX project, not the Qwen port. They require
a compatible MLX environment and the repository's data, tokenizers, configs,
and checkpoint artifacts; those artifacts are not embedded in this document.

From the repository root:

```bash
uv sync --extra dev
uv run --locked llmpr-eval-chess \
  --config configs/controlled/swa32.json \
  --checkpoint runs/controlled-swa32/checkpoint-0001000.npz \
  --examples 32 --batch-size 8 --seed 1337 --split val --temperatures 0 \
  --windows full,128,64,48,32,24,16 \
  --cache-modes preserve,restart,restart-each --teacher-forced
```

For baseline evaluation, replace config with `configs/baseline.json` and
checkpoint with `runs/baseline/checkpoint-0006000.npz`. Use `--split test` and
`--examples 512` for a larger held-out evaluation.

For static Memento/streaming policies, `--transport --windows full` activates
the configured policy without an additional SWA band. Omitting transport
disables those static policies. For scored checkpoints, `--windows full` does
not disable scoring: use `--disable-scoring --windows full` for the true full
control, and `--scored-budget 32 --windows full` for the native 24+8 budget.
Streaming's numeric capacity override is `--streaming-log-budget 32` for 16+16.
Carrier-on/off transport comparisons keep inserted carriers in both arms;
`--no-carriers` is a different intervention and cannot use carrier-dependent
transport simultaneously.

Repository implementation references, if porting: `src/llmz/model.py` contains
reconstruction; `inference.py` defines boundary timing and the shadow stream;
`kv_state.py` defines visibility; `retention.py` and `transport.py` define
retention policies; `eval_chess.py` performs the paired evaluation.
The source experiment is documented in `README.md` and
`evaluation/window-restart-sweep.md`. Existing results are copied above for
context; all Qwen task specifications and hyperparameters are proposed here.
