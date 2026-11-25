# Verification: re-checking every quoted result against known defects

A research log that quotes dozens of numbers accumulates defects underneath them: a play bug, a solver precision issue, silent failures, a statistics error. On 3 Oct 2026 the author froze a plan before any re-check data existed: list every known defect, mark every previously quoted result that a defect could touch, and queue a small run for each exposed result. Outcome: 20 results were reviewed. Most stood. Two kinds of number were replaced (the "min_raiser" opponent rows, which a play bug had inflated, and the old search results, which were re-measured). The v1 bot's LBR exploitability turned out higher than first quoted, though every qualitative conclusion held. One item stayed open: a re-check of the flop-search calibration after a solver fix was never recorded. Status words used below: STANDS (no exposure, or checked), SUPERSEDED (replaced by a re-measurement), SUSPECT (exposed, not yet checked).

## Defects

- **K1, a play bug (fixed 30 Sep).** Calling an all-in *reading* of a bet (see `docs/translation-and-exploitability.md`) was turned into a shove or a fold even when the real bet was not all-in.
  - Measured once, it moved a baseline row from +18.50 to +19.95 bb/100 (487 of 50,000 deals differ).
  - Every match, LBR (local best response exploitability estimate) and scorecard run on an older engine carried it.
- **K2, the fused GPU solver at deep stack-to-pot ratio.**
  - In the last rows of the training-data run, stored values sat about 12% of the pot from a float64 reference, against 0.8-1.9% for the reference solver.
  - Traced partly to zero-tie amplification on the CUDA side.
- **K3, silent failed searches.** Before a fix, a failed search fell back to the blueprint without being counted, so a result could be a blueprint/search mix.
- **K4, live turn-search precision.** 100 iterations sit 8.7% of the pot from equilibrium on real turn starts (400 iterations: 0.86%). Not a bug, but a bound on every turn-search result.
- **K5, statistics.** Scorecard intervals used z = 1.96 on 50 blocks, whereas t with 49 degrees of freedom is 2.010 (about 2.5% too narrow). Fixed on 4 Oct.
  - All 563 stored intervals in receipts were rechecked: 2 verdicts changed, both on one row, going from just above zero to containing it.
  - They were +6.30 [+0.14, +12.45], now [-0.02, +12.61], and +6.11 [+0.01, ...], now [-0.14, ...]. No recorded decision cited them.

## Checks and results

**V1 and V2 (K1 against all scorecards, CPU).** Every bot was re-evaluated on the fixed engine and compared per row with its old receipt using paired 50-block intervals. Nearly every row stayed within its interval (v2-2B baseline +1.45 [-2.21, +5.11], LBR +2.78 [-3.36, +8.92]). The exceptions:
- The min_raiser rows (an opponent that always makes the minimum raise) were K1's signature in every bot: at 100bb v1 +344.7 -> +61.1, v2-1B +256.5 -> +140.3, v2-2B +232.6 -> +115.5; at 200bb v1 +904.1 -> +287.3, v2-1B +805.3 -> +499.9, v2-2B +762.4 -> +457.5 (v2-2B change -304.9 [-327.7, -282.1]). All old min_raiser numbers were superseded.
- LBR rose for v1 (K1 had made it look less exploitable): 100bb +153.5 -> +171.0 (+17.5 [+8.5, +26.5]); 200bb +107.0 -> +131.1 (+24.1 [+8.5, +39.6]). At 100bb v2-1B went +255.4 -> +263.2 and v2-2B +270.9 -> +276.0.
- Conclusions re-derived on the new receipts and kept: v2-1B strength +8.63 [+3.26, +14.00]; "2B vs 1B convergence ruled out" (strength +0.86 [-3.53, +5.25] at 100bb, -0.23 [-6.68, +6.21] at 200bb; LBR +12.8 [-25.9, +51.5] and +67.2 [+8.5, +125.9]); the 200bb LBR gap between v2-1B and v1 is +111.1 [+26.5, +195.7], of which the postflop part is only +9.8 [-56.1, +75.6], that is, it is essentially a preflop effect.
- The bet-translation experiment was re-decided with its own unchanged rule on the fixed engine and reached the same outcome: M1 -52.2 [-61.6, -42.8] passed, the x1.5 row -12.65 [-19.00, -6.30] and realistic row -3.93 [-7.32, -0.53] still failed, so the alternative translation stayed unadopted. Its standard-LBR gain shrank to -3.0 [-51.8, +45.8] (it had been -62.3), so part of that earlier gain was K1; the peek-free gain did not move. The postflop exploitability decomposition was re-run and stood (+0.55 [-0.02, +1.11]).

**V3 (K3 against river-search LBR, CPU).** 2,000 deals with failure counts.
- 0 failed searches.
- River search cut LBR for v1 by -118.4 [-181.5, -55.3] and for v2-2B by -210.3 [-288.8, -131.8]; same direction as before, numbers superseded.

**V4 (K2 against live turn search, GPU).** 40 recorded real turn starts (31 distinct games). The production fused solver against the generic float64 solver, scored by a float64 reference walk, exploitability in % of pot:
- At 100 iterations: 11.29 vs 11.22 (ratio 1.006, worst spot +3.26). At 400 iterations: 1.476 vs 1.448 (ratio 1.019, worst +1.09).
- The pre-registered rule flagged the deviation as REACHING live search because one spot exceeded 2% of the pot at 100 iterations.
- The means were far inside the 25% limit, and the outlier was a single game (spots 31 and 33: 18.96 vs 16.21 and 15.70).
- The generic float64 solver is itself non-deterministic on CUDA (twin runs differ by up to 0.75 at 100 iterations, median 0.07).

**V5 (search results after the fixes, GPU).** Arm D was the bot as it plays; arm F added float64 terminal sums, on the same deals.
- Re-quoted from arm D, turn-search LBR (2,500 deals): v1 -20.0 [-121.9, +81.9], v2-2B +40.8 [-30.9, +112.6]; against no search -165.8 [-230.6, -101.0] and -270.7 [-362.1, -179.4].
- Head to head, v2-2B vs v1: with turn search +7.4 [-14.7, +29.5]; with river search +13.2 [-5.0, +31.5]. These replaced the earlier search numbers.
- The K2 effect F - D (Bonferroni intervals for three numbers): LBR v1 +23.9 [+2.3, +45.5]; LBR v2-2B -5.3 [-38.9, +28.4]; head to head +4.5 [-3.7, +12.6].
- By rule that read as K2 MATTERS, on v1 alone, in the direction that the exact float64 sums made v1 *more* exploitable. Arm F cost 3.2-3.3x the time per deal.

## What did not work, and limits

- The re-check of R14 (flop search: 500 iterations, gamma 5, calibrated with the fused solver) "after a K2 fix" was never recorded. It stands as measured, certified by solver exploitability bounds. The flop-search M1 loss numbers were judged self-certified by the exact oracle's brackets (`docs/flop-search.md`).
- The question of whether live search needs a faster exact-terminal fix was answered by the later float-float kernel (`docs/gpu-kernels.md`), not by this log.
- The 2B-vs-1B LBR difference at 200bb (+67.2 [+8.5, +125.9] on 5,000 deals) was later not reproduced on 100,000 deals (+9.7 [-3.6, +22.9], `docs/translation-and-exploitability.md`); the strength result (2B not stronger) is unaffected.
- K1's size on the min_raiser rows was never traced to a mechanism; only its magnitude is known.
- Checks V4 and V5 are limited by the 40 starts and 2,500-deal LBR runs; V5's single positive Δ has a lower bound of +2.3, barely above zero.
- Receipts and scripts: `records/verify/v4.jsonl`, `records/verify/v4_turn_search.py`, `records/verify/v5.json`.
