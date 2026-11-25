# RIVER-ITERS scout: result (18 Oct 2026) — **PROCEED**

Run as registered (`PREREG.md`, `scripts/river_iters.py` at 319008e, audited read-only before any data).
Record: `scout.json` (every job's curve, the bootstraps, provenance, the GPU checks; not VOID).

## Data
- `collect` 20:41-21:37 SGT: B1 against B0 on the 200bb baseline deals 0-499 (seed 410001, 4 threads, AIVAT on),
  5 chunks of 100, every chunk's arms proven. **332 river jobs** (0.664 a deal; 65/79/64/67/57 by chunk, as P2's),
  all distinct, on 162 boards. The deals' results are identical to P2's, deal for deal: the same play, so the same
  river jobs the live config solves.
- `solve` 22:01-22:03 SGT: 116 s; the GPU had no other job of this Python at the start or the end.

## Exploitability in the river subgame, % of the pot (mean [95% board-bootstrap], median)

| iterations | dcfr_g2 (live) | shift1_g5 | solve seconds, median (dcfr_g2) |
|---|---|---|---|
| 100 | 3.431 [2.952, 3.960], 2.221 | 4.002 [3.397, 4.656], 2.289 | 0.015 |
| **200 (live)** | **1.285 [1.095, 1.495]**, 0.803 | 1.355 [1.155, 1.569], 0.758 | 0.023 |
| 300 | 0.719 [0.615, 0.831], 0.451 | 0.762 [0.649, 0.884], 0.418 | 0.030 |
| 500 | 0.358 [0.304, 0.419], 0.217 | 0.367 [0.314, 0.426], 0.216 | 0.045 |
| 1000 | 0.135 [0.113, 0.158], 0.080 | 0.142 [0.120, 0.166], 0.078 | 0.083 |
| 2000 | 0.050 [0.042, 0.059], 0.028 | 0.054 [0.045, 0.064], 0.027 | 0.158 |

The seconds include building the solver and are descriptive only (not the live in-lock times; p95 within 10% of
the median at every checkpoint).

## Reading (as registered)
1. E(dcfr_g2, 200) = 1.285% >= 0.50%: the gap is worth a registration.
2. Halved with the intervals apart (E <= 0.643%, upper end < 1.095%): dcfr_g2 and shift1_g5 at 500, 1000 and 2000.
   300 iterations is not enough (0.719%).
3. **PROCEED; candidate dcfr_g2 at 500 iterations** (the smallest n; dcfr_g2 first at equal n). The registration
   picks the final arm and n, with the river bar (p95 <= 2 s) checked by a mock.

## What it says
- The live river stops 1.29% of the pot from equilibrium, about the live turn's gap after B1 (0.96-1.31%), and
  the river is where the pots are largest.
- Iterations are almost free here: 2000 iterations take ~0.16 s, against the 2 s bar, and cut the gap 26x
  (to 0.05%). shift1_g5 is no better than the default at any checkpoint.
