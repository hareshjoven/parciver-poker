# LEAF-EV (SC5): result

Run 15 Oct 2026, 07:17-07:48 SGT, from `parciver-sc5` at commit `7ef6474` (clean), as PREREG.md and its
Amendment 2 fix it. Record: `gate-a.json` (copied verbatim from the run). The first run, not VOID: it ran to its
registered stop with the GPU otherwise idle.

## Verdict: STOP at step 3. SC5 closes.
S's selection loss, 0.013497820025475825 (epoch 187), is not below P's, 0.013437177158544897 (epoch 183): it is
0.45% higher. Per the registration nothing else runs, so step 4 (the gate half) never ran. SC5 closes for the
factorization under today's loss. Nothing changes live. The branch is kept as the tag `archive/scout-sc5`.
DEVN's own EV-space loss stays untested (IDEAS).

- **The reproduction held to the bit.** P's recipe was retrained with the scout's code, matchup off. It got best
  epoch 183 and selection loss 0.013437177158544897, and the net's sha256 (`0c196d46...d45489`) equals P's.
  - All 200 epochs' train loss, validation loss and validation MAE equal P's record. Only the wall-clock
    seconds differ: 805.9 s to the best epoch against P's 822.3.
  - So the scout's default path, the driver and cuBLAS left P's training unchanged.
- **S** (`runs/turn-net/leafev-c0-wd0.1/`, sha256 `9be57caf...151f53`): P's recipe with `matchup=True`.

## The numbers (from gate-a.json and the two nets' record.json)
| | P (reproduced) | S (matchup) | S vs P |
|---|---|---|---|
| best epoch | 183 | 187 | |
| selection loss (Huber, validation split) | 0.013437177 | 0.013497820 | +0.45% |
| validation MAE, range-weighted, % of the pot | 4.719 | 4.656 | -1.34% |
| train loss at the best epoch | 0.0025085 | 0.0023688 | -5.6% |
| seconds per epoch (200 epochs) | 4.40 | 4.60 | +4.6% |

Means over the last 20 epochs give the same split: validation loss +0.46%, validation MAE -1.32%, train
loss -5.11%. S's validation loss was below P's in 82 of 200 epochs, its validation MAE in 171 and its train loss
in 194.

## Descriptive, not gating
- **The two validation metrics disagree.** S's range-weighted MAE is 1.3% lower through the late epochs, but its
  Huber loss is 0.45% higher, and it fits the training set 5% better.
  - The Huber loss weighs every live holding equally and, below one pot of error, squares the error. The MAE
    weighs holdings by their range mass, linearly.
  - So S's errors are smaller where the ranges are, on average, but larger somewhere the squared loss counts.
    Which holdings carry the difference was not measured.
  - The selection rule (the loss) was fixed before any data, so this changes nothing. It is one seed: how either
    difference compares with seed-to-seed noise is unknown.
- **SC5 scored no net on the gate half.** Its 12,514 samples were loaded but never evaluated here; LEAF-NET's
  Gate A scored them on P and N0. Selecting by the gate's own metric, or DEVN's EV-space loss, would be a new
  arm with its own registration, made after seeing this result.
- **GPU state:** start 8%, 939 MiB; end 0%, 8,719 MiB (the scout's own cached memory at its last step). The
  `python` lists hold a desktop app's two interpreters at both ends, which are not compute jobs. No
  other job ran.
