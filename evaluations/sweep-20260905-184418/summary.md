# Causal baseline and step-500 finetuning sweep

32 identical seeded validation examples; greedy generation; teacher-forced NLL on identical reference continuations. Baseline: step 6000. Memento and variable-SWA: 500 finetuning steps from that baseline. No Memento inference masks. Lower NLL is better.

Each cell lists **preserved / boundary restart / restart every output step**.

| Window | Causal baseline          | Memento                  | Variable SWA             | Scored (query-dependent)      | Scored* (eviction-time)       |
| ------ | -----------------------: | -----------------------: | -----------------------: | ----------------------------: | ----------------------------: |
| Full   | 0.1003 / 0.1003 / 0.1003 | 0.7002 / 0.7002 / 0.7002 | 0.5602 / 0.5602 / 0.5602 | 0.7632 / 0.7632 / 0.7632      | 0.2349 / 0.2349 / 0.2349      |
| 192    | 0.1029 / 0.1031 / 0.1084 | 0.7000 / 0.6996 / 0.6991 | 0.5591 / 0.5609 / 0.5549 | 0.7594 / 0.7602 / 0.7588      | 0.2360 / 0.2366 / 0.2381      |
| 128    | 0.2474 / 0.2518 / 0.2892 | 0.7053 / 0.7047 / 0.7023 | 0.5025 / 0.4975 / 0.4964 | 0.7298 / 0.7263 / 0.7315      | 0.2442 / 0.2524 / 0.2975      |
| 96     | 0.6455 / 0.6557 / 0.7520 | 0.6582 / 0.6600 / 0.6783 | 0.3731 / 0.3971 / 0.4453 | 0.5845 / 0.5956 / 0.6273      | 0.2803 / 0.3169 / 0.3833      |
| 64     | 1.4621 / 1.5376 / 1.7587 | 0.6596 / 0.6717 / 0.7602 | 0.3559 / 0.4057 / 0.6987 | 0.4118 / 0.4538 / 0.6538      | 0.3650 / 0.4303 / 0.5875      |
| 32     | 3.5248 / 3.6117 / 3.7794 | 0.9780 / 1.0491 / 1.2777 | 0.5919 / 0.7475 / 1.2216 | 0.5197 / 0.7764 / 1.1911      | 0.5318 / 0.7297 / 1.2375      |
| 16     | 4.7573 / 4.8428 / 4.9793 | 1.9210 / 1.9479 / 2.1796 | 1.0090 / 1.0724 / 1.7246 | 1.4036 / 1.6735 / 2.1449      | 1.7005 / 1.9365 / 2.5503      |

Scored retention uses **8 older-memory slots** at every finite budget, with the
remaining slots assigned to recency: budgets 192, 128, 96, 64, 32, and 16 use
recent windows 184, 120, 88, 56, 24, and 8 respectively. “Full” disables scored
eviction entirely. The same step-500 checkpoint is used throughout, without
retraining; only budget 32 matches its training budget. Rows compare maximum
visible-entry counts, not identical positional windows or computational cost.

\*The eviction-time scored checkpoint used three shared readouts,
counterfactual scorer training, and a 5% full-context escape mixture. Its full
row therefore includes directly trained full-context behavior and is not a clean
measure of transfer from scored retention alone.

Boundary restart reconstructs surviving KVs before the FEN query. Repeated restart also reconstructs before every subsequently fed output token. All three modes are included for the full-attention-trained baseline.

The causal baseline also benefits from preserved contextual KVs under SWA. Variable-SWA finetuning substantially improves small-window performance, while all three finetuned models lose full-attention performance compared with the baseline. This is a small pilot; restart penalties alone do not isolate whether training improved information encoding, selection, or use.

## Exact FEN counts with preserved KVs

| Window | Causal baseline | Memento | Variable SWA | Scored (query-dependent) | Scored* (eviction-time) |
| ------ | --------------: | ------: | -----------: | -----------------------: | ----------------------: |
| Full   | 12/32           | 0/32    | 4/32         | 3/32                     | 6/32                    |
| 192    | 12/32           | 0/32    | 4/32         | 3/32                     | 6/32                    |
| 128    | 11/32           | 0/32    | 4/32         | 3/32                     | 6/32                    |
| 96     | 9/32            | 0/32    | 4/32         | 3/32                     | 6/32                    |
| 64     | 4/32            | 0/32    | 5/32         | 3/32                     | 4/32                    |
| 32     | 0/32            | 0/32    | 2/32         | 3/32                     | 3/32                    |
| 16     | 0/32            | 0/32    | 0/32         | 1/32                     | 0/32                    |

## Scored eviction at step 500

Same 32 examples and evaluation settings. Scored retention uses **24 recent + 8 older entries per layer**, compared below with the other checkpoints under SWA-32. This matches the maximum visible-entry count, not compute or which historical positions can be read. No additional SWA band is applied to scored retention.

| Model / inference             | Preserved NLL | Restart NLL | Repeated NLL | Boundary penalty | Exact preserved |
| ----------------------------- | ------------: | ----------: | -----------: | ---------------: | --------------: |
| Causal baseline / SWA-32      | 3.5248        | 3.6117      | 3.7794       | +0.0870          | 0/32            |
| Memento / SWA-32              | 0.9780        | 1.0491      | 1.2777       | +0.0711          | 0/32            |
| Variable SWA / SWA-32         | 0.5919        | 0.7475      | 1.2216       | +0.1556          | 2/32            |
| Scored / recent 24 + memory 8 | 0.5197        | 0.7764      | 1.1911       | +0.2567          | 3/32            |

Scored retention has slightly lower preserved NLL than variable-SWA at the same maximum entry count, and a larger boundary-restart penalty. The scorer’s retention decisions are frozen from a preserved shadow stream during restart, so the intervention changes representations rather than selecting different tokens. This pilot supports useful contextual KVs in the scored system, but does not establish statistical superiority or separate adaptive retention from improved information transport.

### Interpretation and next control

Comparable performance after 500 finetuning steps is encouraging: the scored
model must jointly learn useful representations and retention decisions through
a biased straight-through gradient approximation, whereas SWA supplies a fixed,
predictable retention rule. This makes scored retention plausibly harder to
optimize; the present experiment does not measure optimization difficulty directly.

Scoring also has a more flexible inference policy. Its eight older-memory slots
can preserve evidence that SWA-32 must discard, despite the same maximum number
of visible entries. Comparable performance therefore does not establish that
scoring solves an equally constrained task with a harder training procedure.

The restart penalty shows that the scored system benefits substantially from the
contextual content of retained KVs, beyond their visible token identities. It does
not yet establish that the scorer specifically learned to favor tokens carrying
information from evicted context. The next control is random retention with the
same **24 recent + 8 older slots**, trained and evaluated under that policy, to
isolate the contribution of learned selection from the memory layout itself.

## Eviction-time scored retention with shared readouts at step 500

Same 32 seeded examples and evaluation settings. This newer checkpoint assigns
each entry one immutable score when it leaves the 24-token recent window and
retains eight older entries per layer. It was trained with three FEN readouts per
history, paired swapped-support counterfactuals, and a 5% full-context escape
mixture. No additional positional SWA mask was applied during this evaluation.

| Model / inference                          | Preserved NLL | Restart NLL | Repeated NLL | Boundary penalty | Repeated penalty | Exact preserved |
| ------------------------------------------ | ------------: | ----------: | -----------: | ---------------: | ---------------: | --------------: |
| Earlier scored / query-dependent scores    | 0.5197        | 0.7764      | 1.1911       | +0.2567          | +0.6714          | 3/32            |
| New scored / immutable eviction-time score | 0.5318        | 0.7297      | 1.2375       | +0.1979          | +0.7057          | 3/32            |

The new checkpoint is broadly similar on this small sample: preserved NLL is
slightly worse, boundary-restart NLL is better, repeated-restart NLL is slightly
worse, and exact FEN count is unchanged. Its smaller boundary penalty suggests
less dependence on contextualized retained KVs at the final query, while the
large repeated-restart penalty remains. This is not an isolated comparison of
scoring rules: shared readouts and counterfactual training changed, and 5% of
training histories kept full context. In particular, preservation of
full-context behavior may partly reflect direct full-context rehearsal rather
than transfer from the scored-retention objective.
