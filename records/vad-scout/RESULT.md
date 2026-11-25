# vad-scout: RESULT (25 Nov 2026) - **NO. No arm meets the pass rule; do not pre-register**

Run as registered (`PREREG.md`, `scripts/vad_scout.py`, committed before data; the turn gate was amended once, before any result was read). Raw numbers: `raw.json`. Gate: river `live` arm bit-identical to `GpuSolver` on all 6 river spots; turn `live` arm within 3% of a stock run on all 6 (assert passed).

Exploitability ratio to the live DCFR (below 1 = better). Median over spots, [min, max]; `<=0.9` = spots at or below 0.90 (pass rule: at least 9 of 12 at both budgets, and no turn spot above 1.00).


## T = 100 iterations

| arm | river median [min,max] | turn median [min,max] | spots <=0.90 | turn spots >1.00 |
|---|---|---|---|---|
| vneg | 0.969 [0.91, 1.14] | 1.036 [0.74, 1.08] | 1/12 | 5 |
| pow15 | 0.826 [0.62, 1.17] | 0.966 [0.64, 1.01] | 5/12 | 1 |
| opt | 1.399 [1.24, 1.68] | 1.879 [1.19, 2.20] | 0/12 | 6 |
| boost | 1.108 [0.85, 1.32] | 1.054 [0.67, 1.07] | 2/12 | 4 |
| warm25 | 0.976 [0.93, 1.00] | 0.932 [0.91, 0.94] | 0/12 | 0 |
| wop | 1.065 [0.94, 1.25] | 1.659 [1.10, 1.78] | 0/12 | 6 |
| vad | 1.141 [0.97, 1.21] | 1.666 [1.11, 1.83] | 0/12 | 6 |
| vad_nowarm | 1.143 [0.96, 1.32] | 1.775 [1.19, 1.97] | 0/12 | 6 |
| hs30 | 0.944 [0.89, 1.39] | 0.880 [0.66, 1.04] | 5/12 | 1 |
| hs15 | 0.878 [0.81, 1.17] | 0.901 [0.58, 1.00] | 8/12 | 1 |

## T = 400 iterations

| arm | river median [min,max] | turn median [min,max] | spots <=0.90 | turn spots >1.00 |
|---|---|---|---|---|
| vneg | 0.964 [0.89, 1.09] | 1.048 [0.87, 2.90] | 2/12 | 5 |
| pow15 | 0.897 [0.66, 1.12] | 1.017 [0.77, 1.06] | 4/12 | 4 |
| opt | 1.553 [1.27, 1.70] | 2.061 [1.41, 2.13] | 0/12 | 6 |
| boost | 1.139 [0.87, 1.22] | 1.104 [0.83, 1.16] | 2/12 | 5 |
| warm25 | 0.983 [0.97, 1.01] | 0.956 [0.94, 0.99] | 0/12 | 0 |
| wop | 1.277 [1.00, 1.37] | 1.775 [1.36, 1.89] | 0/12 | 6 |
| vad | 1.386 [1.08, 1.51] | 1.831 [1.43, 2.02] | 0/12 | 6 |
| vad_nowarm | 1.389 [1.07, 1.52] | 1.897 [1.47, 2.06] | 0/12 | 6 |
| hs30 | 1.366 [1.13, 1.70] | 1.393 [0.90, 1.57] | 1/12 | 5 |
| hs15 | 1.166 [1.04, 1.56] | 1.244 [0.83, 1.29] | 1/12 | 5 |

## Reading

- **VAD-CFR and WOP-CFR do not transfer.** `wop`, `vad` and `vad_nowarm` are worse than live on every turn spot at both budgets
  (median 1.7-1.9x the live exploitability) and on most river spots (1.07-1.39x). The loss comes from the optimistic term
  (`opt` alone: 1.4-2.1x); the sign-dependent negative discount (`vneg`) is a wash; the x1.1 boost (`boost`) is slightly worse;
  the scaled warm start (`warm25`) is a small steady gain (0.93-0.98 median; never worse than live on a turn spot, at most 1% worse on a river spot) that is far below the bar.
- **Policy power 1.5 (`pow15`)** helps on the river at T=100 (median 0.83; 4/6 river spots at or below 0.90) and is neutral on the turn (1/6);
  no pass.
- **HS-DCFR `hs15` is the only arm with a signal, and it is budget-specific.** At T=100 it reaches median 0.88 (river) and
  0.90 (turn), 8/12 spots at or below 0.90 - one spot short of the count bar, and one turn spot (1.003) also fails the 1.00 clause. At T=400 it is worse than live (1.17 river, 1.24 turn).
  Its schedule is tuned to a known budget n; the live river search is already registered at 500 iterations (`records/river-iters`),
  so the T=100 gain sits outside the budgets we run.
- Caveats: 12 spots with synthetic uniform and peaky ranges, not blueprint ranges; float32; turn runs carry about 0.5% run-to-run
  noise from GPU atomics; one outlier (`vneg` at T=400, one turn spot 2.9x) was not investigated; the two ambiguous readings of the
  VAD text named in `PREREG.md` were each tried one way only; the v2 volatility extras were not tested.

## Decision

No arm passes, so there is no real pre-registration and no change to the live solver. The "VAD/HS transfer to real-time HUNL
search" question is answered *no on this testbed*, and the one signal (`hs15` at 100 iterations) matters only if a search budget
below ~200 iterations ever becomes the operating point. Not a novelty claim; a negative replication result for the repo's record.
