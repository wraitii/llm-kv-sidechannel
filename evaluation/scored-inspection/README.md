# Scored finetune inspection

Evaluated `runs/controlled-scored/checkpoint-0001000.npz` on 32 eligible validation examples, reservoir seed 1337, greedy decoding, batch size 8. This reuses the window-sweep sample, so it is exploratory validation, not an independent test. Source lengths in the manually inspected examples: 9, 55, and 211 tokens (validation file indices 62640, 23289, 21935).

## Scores are saturated, but finite

Native 24+8 retention, temperature 1. Each layer contributes 2,888 actual boundary-assigned scores across the 32 teacher-forced sequences; the unassigned trailing window is excluded. These include the inspection sequence's final EOS query. Layers are zero-indexed.

| Layer | Median | Minimum | Maximum | Fraction with absolute score > 10 | MLX float32 sigmoid exactly 1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.57 | -58.59 | 25.37 | 20.0% | 0.2% |
| 1 | 17.68 | -35.74 | 46.47 | 83.7% | 52.0% |
| 2 | 15.98 | -3.06 | 31.22 | 82.4% | 40.4% |
| 3 | 13.89 | -2.62 | 36.90 | 74.8% | 21.3% |
| 4 | 15.12 | -7.50 | 40.64 | 86.9% | 30.2% |
| 5 | 16.26 | -17.22 | 52.33 | 81.8% | 43.6% |

All sampled scores are finite. Step 900 is already similarly saturated (layer medians 7.22, 18.16, 16.54, 15.92, 12.86, 14.35), so these two checkpoints do not show a sudden numerical explosion. They cannot establish when saturation began.

The surrogate uses sigmoid(raw score / temperature), while hard retention uses top-k rankings. Large positive shifts therefore saturate the surrogate without invalidating rankings. At scores with sigmoid exactly 1, its local derivative is zero; elsewhere large magnitudes make it tiny. This is evidence of a weak/saturated scorer training signal, not a measurement of total parameter gradients or proof that the policy stopped learning. Retained KVs also receive gradients through ordinary attention.

A useful next training experiment would compare the existing surrogate against one centered on a detached selection cutoff, and log score quantiles, saturation, scorer gradient norms, and memory turnover. Merely changing temperature during evaluation does not repair training: forward selection is hard.

## Paired behavior

| Condition | NLL/token | Parseable /32 | Valid /32 | Exact /32 | Mean square errors among parseable |
|---|---:|---:|---:|---:|---:|
| Scored 24+8 | 0.478 | 32 | 29 | 3 | 7.50 |
| Scored 36+12 | 0.418 | 27 | 23 | 3 | 6.07 |
| Scoring off, full | 0.791 | 8 | 7 | 3 | 1.75 |
| Scoring off, SWA-32 | 0.729 | 32 | 27 | 0 | 10.50 |
| Scored 24+8, restart | 0.804 | 31 | 26 | 3 | 11.90 |

Native scoring improves over equal-budget SWA on this checkpoint. Larger memory improves teacher-forced NLL but hurts parseability. Full attention's low square-error average covers only eight parseable outputs and is misleading in isolation. Restart reuses frozen scored schedules while rebuilding surviving KVs; its degradation supports reliance on contextual KVs, without proving a particular compression algorithm. These controls do not isolate the benefit of learned scores from other sparse retention policies or the effect of finetuning itself.

## Manual attention and prediction checks

The existing attention capture path now supports scored attention using the exact probabilities from `gated_attention`. Capturing maps leaves scored hidden states exactly unchanged in the six inspected example/budget combinations. All six layers obey the 32/48 entry limits. A regression test checks causal support, protected recent entries, no resurrection, probability normalization, and unchanged hidden states.

- **9 source tokens:** native scored decoding reproduces the reference exactly. No older-memory entries exist at the FEN separator because the source fits inside the recent window. Retention can still matter later as FEN tokens consume the budget; SWA-32 misses the g1 knight.
- **55 source tokens:** native output is almost correct, leaving a white pawn at h2 instead of h3. Different layers keep different older moves. Layer 0 keeps queenside castling `e1c1`; layer 4 retains BOS, while most layers discard it. Attention at the first FEN prediction includes later black castling and rook moves. This is qualitatively plausible, but attention alone does not establish causal relevance. Native NLL is 0.186 versus 0.499 for SWA-32 on this row.
- **211 source tokens:** the native output remains a sparse endgame but misplaces pieces and misses the black rook. SWA-32 produces a crowded, opening-like board instead. Scored memory contains a mixture of early and late positions across layers: layer 2 keeps seven entries from positions 7–47 and one at 142, while layer 1 keeps entries from 77–185. Early anchors exist, but this is not universal BOS retention or identical memory across layers. Some old retained entries receive substantial attention: layer 0 gives position 88 (`a8c8`) about 16% of head-averaged attention at the separator. This does not prove that this particular retained move is useful.

Raw outputs and per-example NLLs are in [results.json](results.json). Score distributions, parameter norms, and per-example retained positions are in [scores.json](scores.json). HTML maps and additional summaries are generated under the ignored `runs/controlled-scored/inspection/` directory.

Reproduce from the repository root with the local checkpoint, data and tokenizers present:

```sh
.venv/bin/python evaluation/scored-inspection/evaluate.py
.venv/bin/python evaluation/scored-inspection/scores.py
```

These scripts write fresh artifacts under `runs/`; the JSON files alongside this report are snapshots of this inspection.

## External relevance: replaying actual piece histories

`relevance.py` replays both games with python-chess, tracks piece identities through captures and castling, verifies the final FEN, and maps retained BPE token spans back to full moves. Run it after the two scripts above. The following checks concern older memory at the FEN separator; the recent window supplies additional moves, and retention continues during FEN decoding.

For the 55-token game, several older survivors directly explain final piece locations:

| Move | Final-board relevance | Layers retaining a token from it |
|---|---|---|
| `b1c3` | White knight still on c3 | 0, 2 |
| `e1c1` | White king still on c1 and castled rook on d1 | 0, 1, 2, 5 |
| `d6d5` | Black pawn still on d5 | 1, 2, 3, 4, 5 |
| `c7c6` | Black pawn still on c6 | 0, 1, 5 |
| `e4e5` | White pawn still on e5 | None |
| `h2h3` | White pawn still on h3 | None |

The missing `h2h3` is particularly suggestive: the native prediction leaves that pawn on h2. However, it correctly places the e5 pawn despite dropping `e4e5`, demonstrating that literal move survival is not necessary when contextual KVs carry information. This is an association, not an intervention establishing the cause of the h-pawn error. Several retained moves also involve pieces later captured; some are themselves captures and can explain absent pieces.

For the 211-token endgame, all four surviving pieces' last moves fall inside the recent window. Older memory is much less interpretable as a final-board summary. For example, the heavily attended `a8c8` moves a rook that is later captured, and layers 2–3 retain many opening moves involving pieces that no longer exist. Other layers preserve histories of the surviving king and rook, and layer 5 emphasizes old captures. Thus the policy retains externally meaningful moves in the medium example, but relevance is mixed in the long example. Differentiated scores alone do not establish selection quality. A matched-budget alternative retention policy or forced keep/drop experiment would be needed for a causal comparison.

## Does it drop redundant moves?

A follow-up checks redundancy visible **when the one-time score is assigned**, rather than blaming immutable priorities for later events. `redundancy.py` finds 40 quiet piece-return episodes and two overlapping four-ply board cycles in 15 of the 32 games. Quiet returns exclude pawn moves, captures, castling, and moves that change castling rights. They restore that piece's position, but intervening moves may change the board. Full board cycles restore piece placement, side to move, castling rights, and en-passant state. Neither category restores FEN clocks or repetition history, so “redundant” here means positional redundancy, not lossless removal of history. The contextual value of the corresponding KV may also exceed the literal move's value.

Concrete decisions:

- In game 21935, the rook's `g2h2 → h2g2` round trip is complete before either move's tokens are scored. All six layers reject all four BPE tokens immediately on leaving the recent window. Likewise all four tokens of `c4c3 → c3c4` are rejected by every layer.
- In game 52188, `e2f3 e5d5 f3e2 d5e5` restores the entire position apart from counters. Six of its seven tokens are rejected by every layer; one fragment, `d5`, survives in layer 5. The cycle completes during the source, but its tokens leave the recent window during teacher-forced FEN processing.
- In game 23289, `d8h4 → h4d8` returns the queen before its tokens are scored. Each of its four tokens is nevertheless admitted by three layers, and all four remain through the final prediction in at least one layer. Thus there is no consistent rule rejecting completed excursions.

For a rough comparison, exclude memory warmup and pair each already-returned token with the nearest unflagged token within eight positions in the same game and scoring phase (source/FEN). Ties prefer the earlier position; controls may be reused. These are descriptive, correlated comparisons, not independent samples or a causal test.

| Scoring phase | Matched returned tokens | Returned-token rejection | Nearby control rejection |
|---|---:|---:|---:|
| Source | 82 | 80.3% | 83.3% |
| Teacher-forced FEN | 44 | 89.0% | 86.0% |

Many obvious returns are dropped, but these matched comparisons show no clear preferential rejection of redundant moves. The raw source comparison looks more favorable (80.3% versus 71.6% for all unflagged tokens), but returned tokens occur later on average, when admission is harder. Matching nearby positions removes that apparent advantage. Better-controlled synthetic trajectories would be needed to establish learned redundancy detection.

Reproduce with `.venv/bin/python evaluation/scored-inspection/redundancy.py` after `evaluate.py`. [redundancy.json](redundancy.json) records episodes, token positions, whether completion was known at scoring, scores, and per-layer retention decisions. [redundancy-summary.json](redundancy-summary.json) contains the matched rates. Full control candidates are generated under the ignored run directory.
