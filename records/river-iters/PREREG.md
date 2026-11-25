# RIVER-ITERS scout: how far is the live river solve from equilibrium?

Registered 18 Oct 2026 (SGT), before any data. A scout: it only decides whether a river change is pre-registered
(as LIVESEARCH B1 was for the turn). Nothing here changes the live bot.

## Why
- The live bot solves the river at 200 iterations (DCFR gamma 2, `slumbot_run.py` `SEARCH`, Q4 terminals). Its
  distance from equilibrium on live spots has never been measured.
- The river is cheap: in P2, B1's river solves took median 0.028 s, p95 0.072 s inside the GPU lock
  (`records/livesearch/p2.json`, n 6,606); the bar is p95 <= 2 s. If 200 iterations leave a real gap, more
  iterations cost little time: the turn's B1 step (11.04% -> 0.96% of the pot on P1's jobs) at a tenth of the
  latency.
- For scale: the flop search converges to 0.20% of the pot at 500 iterations (FLOPSEARCH D3); the live turn is
  at 0.96% (P1, B1).

## Data
- **Spots:** B1's river jobs (5-card boards) as the engine sends them, from P2's protocol: B1 (the live config,
  `scripts/livesearch.py` `build_arms`, 4 threads) against B0 on the 200bb scorecard's baseline deals 0-499
  (seed and deals as P2's first 500, mirrored), in 5 chunks of 100, each chunk's arms proven as P2's (`prove`).
  Every B1 river job is kept (copied before the solve), about 340 expected (0.68 a deal in P2).
  -> `runs/river-iters/jobs-<first>.pkl` with their sha256 in the record.
- **Solves:** each job from scratch, as P1 solves (`FusedSolver`, float32, Q4's terminals, its frozen slots
  held), with two schedules: **dcfr_g2** (the live river's) and **shift1_g5** (the live turn's, Stage 1).
  Exploitability in its own subgame, % of the pot (`pot_of`), at **100, 200, 300, 500, 1000, 2000**
  iterations; each run's wall seconds to each checkpoint (synchronised) are reported.
- GPU alone for this Python (P1's `busy` check at start and end; a peer at the end marks the record VOID and the
  solve step reruns). Other apps on the PC are allowed: this measures precision, not latency (the times are
  descriptive only).

## Reading (decided now)
- **E(arm, n)** = the mean exploitability over the jobs, with a 95% board-clustered bootstrap interval
  (10,000 resamples, seed 0; P1's `board_bootstrap`). Medians reported.
- **PROCEED** (pre-register a river change, its own P1/P2/mock like B1) iff
  1. E(dcfr_g2, 200) >= **0.50%** of the pot (the gap is worth a registration), and
  2. some (arm, n) with n <= 2000 has E <= **half** of E(dcfr_g2, 200), with its interval's upper end below
     E(dcfr_g2, 200)'s lower end.
  The candidate it names: the smallest such n, then dcfr_g2 before shift1_g5 at equal n. The registration
  itself picks the final arm and n, with the latency bar (river p95 <= 2 s) checked by a mock.
- **CLOSE** otherwise: the river stays at 200 and the record says by how much it is from equilibrium.
- **VOID:** a crash, a failed proof, or another GPU job of this Python beside the solves.

## Cost (estimates)
- Collection: 500 deals at ~7 s/deal (P2's pace), ~1 h GPU. Solves: ~340 jobs x 2 arms x 2,000 iterations at
  ~0.15 ms an iteration plus 6 exploitability evaluations, minutes.

## Code
`scripts/river_iters.py` (`collect`, `solve`): reuses `scripts/livesearch.py`'s arms, proofs, tap, solve and
bootstrap; changes no file under src/, engine/ or the live runner. Read-only audit before any data.
