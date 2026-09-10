# Qwen3-1.7B experiment protocol

- Qwen3-1.7B-Base with BF16 LoRA.
- PG-19 text with position-matched counterfactual probes. State probes use
  indirect possession swaps every 300--600 tokens. Passcodes have separate
  easy (320--800 tokens from the answer) and hard (10--25% into the document)
  levels; record exact token distances.
- Validate the end-to-end lifecycle with a 1,024-token, 500-step pilot first.
- Use 3.8K as the first sequence-length target on a 32 GB RTX 5090;
  it is an upper-edge setting and must be reconfirmed on every host.
- For constrained runs, use 80% ordinary LM loss under the trained policy, 15%
  answer-only probe loss under that policy, and 5% ordinary full-attention LM
  loss. Match trained tokens and optimizer steps across controls.
- Evaluate clean LM under full and constrained attention. Probe evaluation must
  exclude EOS, compare each correct answer with its paired counterfactual, and
  report margin/accuracy separately by task and distance.
- Always evaluate preserve, restart-at-answer, and restart-every-N (including
  intervals near and below the trained attention window).

Compare the frozen base model and full-attention LM control against fixed-SWA
and variable-SWA training, each with and without the 15% task-loss mixture.

## Reporting

Clean-language cells report perplexity, with NLL and change from the frozen
base retained in the raw results.

```text
Checkpoint               Full             SWA-512          SWA-256          SWA-128
-----------------------  ---------------  ---------------  ---------------  ---------------
Frozen base              A                B                C                D
Full-attention LM        A                B                C                D
Fixed-SWA LM only        A                B                C                D
Fixed-SWA LM + task      A                B                C                D
Variable-SWA LM only     A                B                C                D
Variable-SWA LM + task   A                B                C                D
```

In both task tables, each cell is `preserve / restart-at-answer /
restart-every-N`.

### Passcodes

Each component reports answer-token NLL, excluding EOS and prompt formatting.

```text
Checkpoint               Difficulty  Full       SWA-512    SWA-256    SWA-128
-----------------------  ----------  ---------  ---------  ---------  ---------
Frozen base              easy        A / B / C  A / B / C  A / B / C  A / B / C
                         hard        A / B / C  A / B / C  A / B / C  A / B / C
Full-attention LM        easy        A / B / C  A / B / C  A / B / C  A / B / C
                         hard        A / B / C  A / B / C  A / B / C  A / B / C
Fixed-SWA LM only        easy        A / B / C  A / B / C  A / B / C  A / B / C
                         hard        A / B / C  A / B / C  A / B / C  A / B / C
Fixed-SWA LM + task      easy        A / B / C  A / B / C  A / B / C  A / B / C
                         hard        A / B / C  A / B / C  A / B / C  A / B / C
Variable-SWA LM only     easy        A / B / C  A / B / C  A / B / C  A / B / C
                         hard        A / B / C  A / B / C  A / B / C  A / B / C
Variable-SWA LM + task   easy        A / B / C  A / B / C  A / B / C  A / B / C
                         hard        A / B / C  A / B / C  A / B / C  A / B / C
```

### State

Each component reports `counterfactual NLL margin (pairwise accuracy)`,
excluding EOS. Retain correct-answer NLL in the raw results.

```text
Checkpoint               Full       SWA-512    SWA-256    SWA-128
-----------------------  ---------  ---------  ---------  ---------
Frozen base              A / B / C  A / B / C  A / B / C  A / B / C
Full-attention LM        A / B / C  A / B / C  A / B / C  A / B / C
Fixed-SWA LM only        A / B / C  A / B / C  A / B / C  A / B / C
Fixed-SWA LM + task      A / B / C  A / B / C  A / B / C  A / B / C
Variable-SWA LM only     A / B / C  A / B / C  A / B / C  A / B / C
Variable-SWA LM + task   A / B / C  A / B / C  A / B / C  A / B / C
```

For restart-every-N, run intervals near and below the trained window and make a
separate table for each N when more than one interval is evaluated.

Before interpreting a run, verify dense/full/SWA equivalence, cached versus
masked logits, no-eviction restart identity, counterfactual-divergence removal,
and irreversible streaming-retention capacity.
