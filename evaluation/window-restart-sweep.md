# Window and restart sweep

Validation split, 32 eligible examples, seed 1337, greedy decoding, and
teacher-forced scoring. Within each cell, results are ordered as
`preserve / restart / restart-each`.

Finetunes use their step-1000 checkpoints. The baseline reference uses its
step-6000 checkpoint and is not a matched-training-compute comparison.

Memento is evaluated both with and without its configured transport. For
`Memento on`, the `full` row is the native transport condition and numeric
windows add the stated SWA constraint on top of transport. `Memento off`
measures transfer to ordinary full/SWA attention.

For `Baseline 6K SWA`, numeric rows are ordinary contiguous SWA windows;
`full` is true full attention.

For Scored, `full` is true full attention with scored eviction disabled. Numeric
rows use learned scored retention and preserve its trained 3:1 recent-to-memory
ratio: 128 is 96+32, 64 is 48+16, 48 is 36+12, 32 is the native 24+8
condition, 24 is 18+6, and 16 is 12+4.

Streaming log-16+16 is trained and evaluated with irreversible log-age thinning
across source and FEN tokens. Numeric budgets split equally between recent and
older entries: 128 is 64+64, 64 is 32+32, 48 is 24+24, 32 is native 16+16,
24 is 12+12, and 16 is 8+8. BOS, the separator, and target tokens count toward
the total budget; discarded entries never return. `full` disables eviction.
This sweep uses the step-1000 checkpoint, batch size 8, and the same 32 examples.
Raw results: [streaming-log16x16-step-1000.jsonl](streaming-log16x16-step-1000.jsonl).

Memento carriers uses `runs/controlled-memento-carriers/checkpoint-0001000.npz`
with inserted carrier tokens enabled in both conditions. `Memento carriers on`
applies the configured recursive carrier transport; `off` uses ordinary full/SWA
attention over the same carrier-augmented sequences. As above, `full` means no
additional SWA constraint. Both sweeps ran sequentially with batch size 8, using
the same 32 eligible examples, seed, windows, and cache modes as the other arms.
Raw results: [transport on](memento-carriers-on-step-1000.jsonl) and
[transport off](memento-carriers-off-step-1000.jsonl).

Variable SWA low-full uses
`runs/controlled-swa-variable-lowfull/checkpoint-0001000.npz`. Unlike the
earlier variable-SWA arm, 5% of its training microbatches used full attention,
and its sampled SWA range was shorter (8–32 rather than 16–48), making the
constrained training condition harder. The sweep uses the same 32 examples,
seed, windows, cache modes, and batch size as the other finetune arms. Raw
results: [swa-variable-lowfull-step-1000.jsonl](swa-variable-lowfull-step-1000.jsonl).

Scored soft-top-k uses
`runs/controlled-scored-soft-topk/checkpoint-0001000.npz`. It has the same
scored-retention architecture and 3:1 recent-to-memory budget split as Scored,
but uses the budget-coupled soft-top-k surrogate for its scorer gradient. A
straight-through correction keeps the training forward pass identical to hard
top-k inference. The sweep otherwise matches the same examples, seed, budgets,
cache modes, and batch size. Raw results:
[scored-soft-topk-step-1000.jsonl](scored-soft-topk-step-1000.jsonl).

Scored soft-top-k delay 0 uses
`runs/controlled-scored-soft-topk-delay0/checkpoint-0001000.npz` and differs
only by assigning each entry's immutable score immediately from its own hidden
state, rather than when it crosses the 24-token recent window. Raw results:
[scored-soft-topk-delay0-step-1000.jsonl](scored-soft-topk-delay0-step-1000.jsonl).
The same arm is also shown at step 2000 as a training-progress comparison, not
a matched-training-compute result. Raw results:
[scored-soft-topk-delay0-step-2000.jsonl](scored-soft-topk-delay0-step-2000.jsonl).

To reproduce, run this command to completion, then repeat it without `--transport`
and change the output filename from `on` to `off`:

```bash
.venv/bin/llmpr-eval-chess \
  --config configs/controlled/memento-carriers.json \
  --checkpoint runs/controlled-memento-carriers/checkpoint-0001000.npz \
  --examples 32 --batch-size 8 --seed 1337 --split val --temperatures 0 \
  --windows full,128,64,48,32,24,16 \
  --cache-modes preserve,restart,restart-each --teacher-forced --transport \
  > evaluation/memento-carriers-on-step-1000.jsonl
```

The low-full variable-SWA arm was reproduced with:

```bash
.venv/bin/llmpr-eval-chess \
  --config configs/controlled/swa-variable-lowfull.json \
  --checkpoint runs/controlled-swa-variable-lowfull/checkpoint-0001000.npz \
  --examples 32 --batch-size 8 --seed 1337 --split val --temperatures 0 \
  --windows full,128,64,48,32,24,16 \
  --cache-modes preserve,restart,restart-each --teacher-forced \
  > evaluation/swa-variable-lowfull-step-1000.jsonl
```

The soft-top-k scored arm used true full attention for its control, followed by
one invocation per scored budget:

```bash
.venv/bin/llmpr-eval-chess \
  --config configs/controlled/scored-soft-topk.json \
  --checkpoint runs/controlled-scored-soft-topk/checkpoint-0001000.npz \
  --examples 32 --batch-size 8 --seed 1337 --split val --temperatures 0 \
  --windows full --cache-modes preserve,restart,restart-each --teacher-forced \
  --disable-scoring \
  > evaluation/scored-soft-topk-step-1000.jsonl

for budget in 128 64 48 32 24 16; do
  .venv/bin/llmpr-eval-chess \
    --config configs/controlled/scored-soft-topk.json \
    --checkpoint runs/controlled-scored-soft-topk/checkpoint-0001000.npz \
    --examples 32 --batch-size 8 --seed 1337 --split val --temperatures 0 \
    --windows full --cache-modes preserve,restart,restart-each --teacher-forced \
    --scored-budget "$budget" \
    >> evaluation/scored-soft-topk-step-1000.jsonl
done
```

The delay-0 arm was reproduced with the same commands after replacing
`scored-soft-topk` with `scored-soft-topk-delay0` in the config, checkpoint,
and output paths. Its step-2000 sweep additionally replaces `0001000`/`1000`
with `0002000`/`2000` in the checkpoint and output paths.

## Teacher-forced NLL per token

Lower is better.

| Window | Fixed SWA-32              | Variable SWA              | Memento on                | Memento off               |
|-------:|--------------------------:|--------------------------:|--------------------------:|--------------------------:|
|   full | 1.207 / 1.207 / 1.207     | 0.641 / 0.641 / 0.641     | 0.763 / 1.182 / 1.182     | 1.080 / 1.080 / 1.080     |
|    128 | 1.160 / 1.132 / 1.136     | 0.574 / 0.576 / 0.547     | 0.820 / 1.222 / 1.226     | 1.056 / 1.056 / 1.061     |
|     64 | 0.725 / 0.799 / 0.986     | 0.287 / 0.395 / 0.727     | 0.937 / 1.229 / 1.211     | 0.832 / 0.851 / 0.915     |
|     48 | 0.583 / 0.759 / 1.043     | 0.313 / 0.539 / 0.884     | 0.985 / 1.209 / 1.222     | 0.864 / 0.916 / 1.032     |
|     32 | 0.429 / 0.958 / 1.457     | 0.449 / 0.760 / 1.315     | 1.138 / 1.238 / 1.460     | 1.035 / 1.115 / 1.344     |
|     24 | 0.933 / 1.179 / 1.823     | 0.606 / 0.845 / 1.531     | 1.395 / 1.447 / 1.781     | 1.328 / 1.359 / 1.726     |
|     16 | 1.864 / 2.000 / 2.412     | 0.874 / 1.024 / 1.819     | 1.987 / 2.010 / 2.289     | 1.938 / 1.972 / 2.249     |

| Budget | Baseline 6K SWA       | Scored                | Streaming log-16+16   |
|-------:|----------------------:|----------------------:|----------------------:|
|   full | 0.100 / 0.100 / 0.100 | 0.791 / 0.791 / 0.791 | 1.174 / 1.174 / 1.174 |
|    128 | 0.247 / 0.252 / 0.289 | 0.774 / 0.791 / 0.825 | 1.141 / 1.138 / 1.070 |
|     64 | 1.462 / 1.538 / 1.759 | 0.517 / 0.625 / 0.773 | 0.600 / 0.694 / 1.228 |
|     48 | 2.349 / 2.420 / 2.705 | 0.418 / 0.646 / 0.906 | 0.424 / 0.777 / 1.776 |
|     32 | 3.525 / 3.612 / 3.779 | 0.478 / 0.804 / 1.300 | 0.440 / 1.386 / 2.506 |
|     24 | 4.254 / 4.326 / 4.387 | 1.008 / 1.248 / 1.714 | 0.754 / 1.974 / 3.089 |
|     16 | 4.757 / 4.843 / 4.979 | 1.835 / 1.997 / 2.127 | 1.918 / 3.190 / 3.665 |

| Window | Memento carriers on   | Memento carriers off  |
|-------:|----------------------:|----------------------:|
|   full | 0.777 / 1.167 / 1.167 | 1.224 / 1.224 / 1.224 |
|    128 | 0.854 / 1.225 / 1.205 | 1.055 / 1.050 / 1.033 |
|     64 | 0.967 / 1.230 / 1.234 | 0.866 / 0.900 / 0.912 |
|     48 | 1.024 / 1.248 / 1.265 | 0.915 / 0.958 / 0.978 |
|     32 | 1.162 / 1.323 / 1.496 | 1.060 / 1.101 / 1.339 |
|     24 | 1.389 / 1.511 / 1.812 | 1.313 / 1.360 / 1.714 |
|     16 | 1.892 / 1.966 / 2.253 | 1.852 / 1.898 / 2.217 |

| Window | Variable SWA low-full  |
|-------:|-----------------------:|
|   full | 0.285 / 0.285 / 0.285  |
|    128 | 0.315 / 0.316 / 0.324  |
|     64 | 0.412 / 0.462 / 0.658  |
|     48 | 0.463 / 0.580 / 0.838  |
|     32 | 0.538 / 0.746 / 1.218  |
|     24 | 0.623 / 0.825 / 1.400  |
|     16 | 0.810 / 0.976 / 1.613  |

| Budget | Scored soft-top-k       | Delay 0, 1K               | Delay 0, 2K               |
|-------:|------------------------:|--------------------------:|--------------------------:|
|   full | 0.772 / 0.772 / 0.772   | 0.741 / 0.741 / 0.741     | 0.724 / 0.724 / 0.724     |
|    128 | 0.756 / 0.764 / 0.762   | 0.719 / 0.720 / 0.712     | 0.682 / 0.688 / 0.696     |
|     64 | 0.499 / 0.547 / 0.660   | 0.565 / 0.596 / 0.691     | 0.482 / 0.539 / 0.682     |
|     48 | 0.455 / 0.540 / 0.776   | 0.553 / 0.645 / 0.806     | 0.438 / 0.577 / 0.871     |
|     32 | 0.511 / 0.750 / 1.247   | 0.609 / 0.790 / 1.170     | 0.478 / 0.818 / 1.245     |
|     24 | 0.840 / 1.185 / 1.795   | 0.963 / 1.188 / 1.674     | 0.787 / 1.170 / 1.804     |
|     16 | 1.925 / 2.238 / 2.179   | 1.884 / 2.035 / 2.110     | 1.931 / 2.103 / 2.280     |

## Greedy generation

Each condition is `parseable / exact / mean square error`; counts are out of 32
and mean square error is conditional on parseability. Conditions within each
cell are separated by semicolons and ordered `P; R; RE`.

| Window | Fixed SWA-32                                  | Variable SWA                                 | Memento on                                   | Memento off                                  |
|-------:|:----------------------------------------------|:---------------------------------------------|:---------------------------------------------|:---------------------------------------------|
|   full | ` 6/2/ 1.50;  6/2/ 1.50;  6/2/ 1.50`          | `19/3/ 2.26; 19/3/ 2.26; 19/3/ 2.26`         | `32/0/14.38; 32/0/21.78; 32/0/21.78`         | `10/0/ 8.50; 10/0/ 8.50; 10/0/ 8.50`         |
|    128 | ` 6/2/ 1.50;  6/2/ 1.50;  6/2/ 1.50`          | `19/3/ 2.26; 19/3/ 2.26; 20/3/ 2.85`         | `32/0/14.69; 32/0/21.97; 32/0/22.25`         | ` 8/0/ 6.75; 11/0/ 7.45; 11/0/ 8.27`         |
|     64 | `17/2/ 4.76; 15/2/ 4.60; 12/2/ 6.75`          | `32/4/ 3.69; 32/4/ 4.69; 29/2/ 7.69`         | `32/0/15.47; 32/0/21.84; 31/0/21.48`         | `28/0/12.46; 28/0/12.75; 27/0/13.78`         |
|     48 | `32/1/ 5.12; 27/1/ 6.44; 26/1/11.96`          | `32/4/ 4.28; 30/4/ 6.57; 31/1/12.55`         | `32/0/16.69; 32/0/20.69; 30/0/20.67`         | `27/0/13.70; 28/0/12.29; 27/0/14.63`         |
|     32 | `31/4/ 7.13; 30/4/11.97; 20/0/16.60`          | `32/4/ 6.69; 30/4/11.83; 25/0/17.84`         | `28/0/18.46; 22/0/19.91; 10/0/20.00`         | `27/0/15.96; 29/0/17.31; 15/0/19.73`         |
|     24 | `25/0/12.24; 25/0/16.08;  6/0/21.50`          | `32/3/10.19; 31/3/14.97; 12/0/19.92`         | ` 6/0/19.00; 10/0/18.60;  1/0/18.00`         | `12/0/17.00; 13/0/19.00;  2/0/22.00`         |
|     16 | ` 2/0/28.00;  1/0/31.00;  1/0/32.00`          | `26/0/16.77; 28/0/19.89;  8/0/26.50`         | ` 3/0/26.00;  1/0/32.00;  0/0/  n/a`         | ` 2/0/22.00;  1/0/26.00;  0/0/  n/a`         |

| Budget | Baseline 6K SWA                         | Scored                                  | Streaming log-16+16                     |
|-------:|:----------------------------------------|:----------------------------------------|:----------------------------------------|
|   full | `32/12/ 1.19; 32/12/ 1.19; 32/12/ 1.19` | ` 8/ 3/ 1.75;  8/ 3/ 1.75;  8/ 3/ 1.75` | `13/ 1/ 5.85; 13/ 1/ 5.85; 13/ 1/ 5.85` |
|    128 | `30/11/ 1.73; 31/11/ 1.90; 30/11/ 2.07` | ` 8/ 3/ 1.75;  8/ 3/ 1.75;  8/ 3/ 1.75` | `13/ 1/ 5.85; 13/ 1/ 5.85; 13/ 1/ 5.85` |
|     64 | `26/ 4/ 8.31; 25/ 4/ 8.04; 26/ 3/11.77` | `18/ 3/ 5.00; 16/ 3/ 5.75; 20/ 2/ 8.80` | `21/ 1/ 6.95; 23/ 1/ 8.43; 27/ 2/13.11` |
|     48 | `15/ 2/10.93; 23/ 2/14.00; 17/ 1/17.53` | `27/ 3/ 6.07; 26/ 3/ 7.65; 23/ 1/11.78` | `31/ 5/ 6.19; 31/ 5/10.77; 31/ 1/16.87` |
|     32 | `14/ 0/21.07; 10/ 0/19.70;  2/ 0/22.50` | `32/ 3/ 7.50; 31/ 3/11.90;  9/ 0/12.67` | `32/ 5/ 6.84; 32/ 4/16.56; 21/ 0/21.33` |
|     24 | ` 0/ 0/  n/a;  0/ 0/  n/a;  0/ 0/  n/a` | ` 9/ 1/ 6.22;  9/ 1/ 8.33;  7/ 0/22.43` | `27/ 2/10.30; 27/ 2/19.93;  6/ 0/25.83` |
|     16 | ` 0/ 0/  n/a;  0/ 0/  n/a;  0/ 0/  n/a` | ` 0/ 0/  n/a;  2/ 0/23.50;  1/ 0/36.00` | ` 2/ 0/21.00;  1/ 0/27.00;  0/ 0/  n/a` |

| Window | Memento carriers on                  | Memento carriers off                 |
|-------:|:-------------------------------------|:-------------------------------------|
|   full | `32/0/15.69; 32/0/20.88; 32/0/20.88` | `11/1/ 7.73; 11/1/ 7.73; 11/1/ 7.73` |
|    128 | `32/0/16.53; 32/0/21.81; 32/0/21.78` | `15/1/11.47; 14/1/11.00; 15/1/10.93` |
|     64 | `32/0/18.28; 32/0/21.88; 31/0/22.32` | `32/1/13.03; 31/1/13.97; 32/1/14.69` |
|     48 | `32/0/18.84; 32/0/23.47; 29/0/24.00` | `32/1/14.47; 32/1/15.22; 31/1/17.32` |
|     32 | `30/0/21.00; 25/0/24.04;  9/0/21.67` | `29/0/16.97; 30/0/18.53; 17/0/19.00` |
|     24 | ` 8/0/20.00;  5/0/18.60;  1/0/15.00` | `11/0/14.82;  9/0/17.44;  3/0/24.33` |
|     16 | ` 0/0/  n/a;  1/0/20.00;  2/0/28.00` | ` 1/0/16.00;  0/0/  n/a;  1/0/28.00` |

| Window | Variable SWA low-full                 |
|-------:|:--------------------------------------|
|   full | `30/4/ 3.73; 30/4/ 3.73; 30/4/ 3.73` |
|    128 | `30/4/ 3.70; 30/4/ 3.67; 30/4/ 4.07` |
|     64 | `30/3/ 5.53; 31/3/ 6.48; 29/2/ 8.45` |
|     48 | `32/2/ 6.28; 32/2/ 7.84; 32/1/12.16` |
|     32 | `32/3/ 8.03; 31/3/12.06; 26/0/16.19` |
|     24 | `30/2/10.87; 32/2/15.84; 13/0/20.00` |
|     16 | `31/0/14.26; 31/0/19.84; 13/0/25.69` |

| Budget | Scored soft-top-k                      | Delay 0, 1K                             | Delay 0, 2K                             |
|-------:|:---------------------------------------|:----------------------------------------|:----------------------------------------|
|   full | `14/2/ 6.71; 14/2/ 6.71; 14/2/ 6.71` | `16/1/ 6.94; 16/1/ 6.94; 16/1/ 6.94`  | `14/1/ 6.29; 14/1/ 6.29; 14/1/ 6.29`  |
|    128 | `14/2/ 6.71; 14/2/ 6.71; 14/2/ 6.71` | `19/1/ 7.84; 18/1/ 7.78; 19/1/ 7.74`  | `14/1/ 6.29; 14/1/ 6.29; 14/1/ 6.29`  |
|     64 | `23/2/ 6.87; 25/2/ 7.04; 26/2/ 8.81` | `28/1/ 8.61; 30/1/ 8.90; 30/1/ 9.87`  | `29/1/ 7.17; 29/1/ 7.59; 30/1/ 9.53`  |
|     48 | `30/3/ 6.70; 32/3/ 8.38; 32/3/10.62` | `31/2/ 8.10; 32/2/ 9.50; 31/1/11.35`  | `32/2/ 6.88; 32/2/ 8.59; 28/1/10.93`  |
|     32 | `32/3/ 7.84; 32/3/11.62; 16/0/15.06` | `32/1/ 9.09; 32/1/11.69; 10/0/13.40`  | `32/3/ 7.38; 31/3/11.19; 10/0/11.80`  |
|     24 | `11/0/10.91;  9/0/19.89;  0/0/  n/a` | `14/0/13.29;  7/0/18.14;  1/0/21.00`  | `12/0/11.50; 10/0/15.10;  3/0/23.00`  |
|     16 | ` 0/0/  n/a;  0/0/  n/a;  0/0/  n/a` | ` 0/0/  n/a;  0/0/  n/a;  1/0/22.00`  | ` 1/0/25.00;  2/0/21.00;  3/0/28.67`  |

Results are exploratory because the sweep contains only 32 examples.
