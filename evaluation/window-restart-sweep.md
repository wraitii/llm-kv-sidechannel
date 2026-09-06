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

Results are exploratory because the sweep contains only 32 examples.
