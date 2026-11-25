# Flop search: a value net at the start of the turn

Heads-up No-Limit Hold'em at 200bb is too deep to solve exactly while a hand is being played. Earlier work in this project already re-solved the river and the turn in real time on the GPU. This document covers the next step back: solving the **flop** by searching only to the start of the turn and asking a learned value network what the turn is worth. The outcome: against an exact full-tree oracle on 40 flop spots, the search removed 63% of the blueprint's loss (primary check passed, M1). Head to head against the blueprint it was indistinguishable from zero (M2, 10,000 mirrored deals, +/- 18 bb/100), which met the pre-registered adopt rule. Three descriptive exploitability checks (M3) were all inconclusive. It has been live since 11 Oct.

## Design

- **Leaf values.** A DeepStack-style value net is evaluated at the start of the turn, in place of the Pluribus approach of continuation strategies. The net maps (pot, board, both ranges) to per-hand counterfactual values. Code: `src/parciver/value_net.py`, `src/parciver/turn_values.py`, `scripts/turn_net.py`.
- **Chance averaging.** Leaves are averaged over the 49 possible turn cards with exact card removal. An all-in at a leaf is valued by equity over the 990 turn-river card pairs. Code: `src/parciver/flop_search.py`, `src/parciver/flop_leaves.py`.
- **Solver.** The flop subgame (57 nodes, 9 leaves at 200bb) is solved by the fused GPU DCFR solver (alpha 1.5, negative regrets halved, alternating updates, averaging exponent gamma) with CUDA graph replay, which took an iteration from 23.4 to 5.0 ms. A leaf-plumbing test matches the Rust full solve to 1e-12.
- **Settings.** 500 iterations with gamma 5. On a calibration set, mean exploitability (percent of pot) fell with iteration count as 7.24 / 2.46 / 0.74 / 0.22 / 0.07 at 100 / 200 / 400 / 800 / 1600. Gamma 5 at 400 / 500 / 600 / 800 iterations gave exploitability 0.60 / 0.39 / 0.29 / 0.17 and value MAE 0.35 / 0.23 / 0.17 / 0.10; 500 was the compromise (exploitability 0.20% at 500 in the final measurement).
- **Training data.** 500,004 solved samples (about 119 GPU hours). Sweep best was width 2,048, depth 5, MAE 5.37%; the final net reached 5.21%. Learning curve at 25 / 50 / 100% of the data: 7.31 / 6.08 / 5.37%. A float32 output layer cost +11.72% step time and was kept.
- **Gate G2** (before any M-check): net MAE 5.111% against 18.646% for a check-down baseline (bar 9.323%), on 12,514 gate samples. Real-play spots 2.876 vs 17.477, random spots 7.345 vs 19.814. PASS.

## The oracle and the checks

- The 40 spots are flop decisions drawn (seeded) from the agent's own hands at 200bb.
- The oracle solves the **full** game (flop, turn and river chance nodes) with the candidate's strategy frozen. Loss(X) = V* - V(X), where V* is the oracle value and X a candidate method.
- Oracle error was e = 0.1% of the pot (measured e_1000 = 0.092%).
- X_net is not suit-symmetric, so the full game was used rather than suit isomorphism. Producing the net-based solves took 23.4 h on CPU.
- Pre-registered rule: adopt if M1's interval is entirely below zero (a) and M2 is not DOWN (b); M3 is descriptive.

**M1 (primary, 40 spots), loss in % of pot**

| method | loss [95% CI] |
|---|---|
| X_bp (blueprint with bet translation, what the agent did before) | +3.404 [+2.409, +4.398] |
| X_net (flop search with the net) | +1.275 [+1.020, +1.530] |
| X_eq (check-down leaves instead of the net; ablation) | +11.943 [+10.934, +12.952] |
| X_net - X_bp (primary) | -2.129 [-3.062, -1.196], upper + e = -1.096 |

- The primary contrast passed, including under the conservative readings.
- The search was better on 38 of 40 spots; it was worse on spot 2 (6bb, +0.36 +/- 0.17) and spot 38 (2bb, +0.95 +/- 0.34).
- By seat: OOP 3.11 -> 1.30, IP 3.70 -> 1.25.
- Flop-search latency: median 2.27 s, p95 2.44 s (rechecked 2.29 / 2.43).

**M2 (head to head, 10,000 mirrored deals).** The agent with flop + turn search (arm A) against the same agent with turn search only (arm B): -0.85 bb/100 [-18.87, +17.18].

- The adopt rule asked only for "not worse" (interval not entirely below zero), so the verdict was ADOPT.
- Search decided 59.1% of A's decisions and 32.0% of B's; 1.45 flop solves per deal.
- 3.81 s per deal, 10.6 h total, 0 failed searches, flop latency median 2.34 s, p95 2.60 s.

**M3 (descriptive, all three readings UNCLEAR).** The same two arms against two exploiting opponents and against LBR.

- The opponents are bots playing the baseline blueprint with every bet size scaled x1.5, and with realistic sizes (5,000 deals each).
- LBR is the local best response exploitability estimate: a heuristic opponent that plays a best response to the agent's observed ranges one decision at a time (2,500 deals).
- Against the two bots the values are each arm's winnings in bb/100 (higher is better); under LBR they are what LBR wins against the arm (lower is better). 95% CIs over 50 blocks.

| run | A | B | A - B |
|---|---|---|---|
| bet sizes x1.5 | +18.9 [-13.0, +50.7] | +37.0 [+0.8, +73.1] | -18.1 [-54.8, +18.6] |
| realistic sizes | +10.8 [-18.3, +40.0] | +28.7 [+1.5, +55.9] | -17.8 [-47.2, +11.6] |
| LBR (2,500 deals) | +8.2 [-45.9, +62.3] | +40.8 [-30.9, +112.6] | -32.6 [-99.9, +34.7] |

The M3 run took about 29.3 h against 21 expected. The LBR part was cut from 5,000 to 2,500 deals because the projected ~63 GPU hours exceeded the 48 h cap.

## What did not work, and limits

- M3 resolved nothing: every A - B interval includes zero. Against the two scaled-size bots the point estimates lean toward the turn-search-only arm (about -18 each), under LBR toward the flop-search arm (LBR won 32.6 bb/100 less, a point estimate). The adoption rested on M1 (exact-oracle loss) and M2 (no harm head to head), not on exploitability evidence.
- M2 is wide: +/- 18 bb/100 cannot detect a modest change in either direction.
- The first M2 run hit a deadlock (a rayon re-entry on a table lock). It was fixed in the engine; no deal from that run counted and M2 restarted from zero.
- The oracle is itself approximate (e = 0.1% of pot), and the 40 spots are a sample, not a population.
- Turn search ran only 100 iterations at the time of M1-M3, which sits 8.7% of the pot from equilibrium (400 iterations gives 0.86%); the live bot now uses 400.
- Live since 11 Oct, with symmetric leaves since 12 Oct; the more precise net tested next (N1, D2b) was later adopted. Pre-registrations and checks are in `records/live-flop/`, `records/flop-sym/` and `records/leaf-net/`. Scripts: `scripts/flop_m1.py`, `scripts/flop_eval.py`. Kernel work behind the solver speed is in `docs/gpu-kernels.md`.
