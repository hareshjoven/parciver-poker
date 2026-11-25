# Results

Every number below is read from a write-up or a machine-readable receipt in [`records/`](records); the few that
exist only in a write-up's prose, or whose receipt was not exported, are marked (see [`records/README.md`](records/README.md)).
Where a result was negative, inconclusive or not adopted, it is listed with the rest.
Intervals are 95% unless stated. "bb/100" is big blinds per 100 hands; the rating scale's "point" is 1 mbb/hand
(0.1 bb/100).

## 1. The outside benchmark: Slumbot

[Slumbot](https://www.slumbot.com) (winner of the 2018 Annual Computer Poker Competition's heads-up no-limit
event) is run by a third party over HTTP, 50/100 blinds, 200 bb stacks. Both matches were pre-registered
(`PREREG.md`), played for exactly 50,000 counted hands, and read by the registered rule: the result is only
"won" or "lost" if the interval excludes zero.

| run | config | bb/100 | 95% interval | verdict |
|---|---|---|---|---|
| SLUMBOT (5-7 Oct 2026) | v2-2B blueprint, turn/river search, no flop search | **-1.54** | [-15.00, +11.91] | unresolved |
| SLUMBOT-2 (25-28 Oct 2026) | v2-8B blueprint, full live search (flop value net, turn 400, river 2,000 iterations) | **-12.54** | [-26.41, +1.33] | unresolved |
| difference, run 2 less run 1 | | -10.99 | [-30.32, +8.33] | within noise |

Breakdown of SLUMBOT-2 (`records/slumbot-2/RESULT.md`): as big blind -24.42 [-43.12, -5.73] over 25,000 hands;
as small blind -0.65 [-21.15, +19.85]; all-in-adjusted (59 pre-river all-ins settled at equity) -7.59
[-20.89, +5.71]. By 10,000-hand block: -4.2, +4.8, -40.2, -18.7, -4.4; each block's own interval is about +/-31.

What to take from it:
- The bot is **not shown to beat or to lose to** the benchmark. A 50,000-hand match at this variance resolves
  about +/-14 bb/100, so only a much larger edge, either way, would have been visible.
- The big-blind seat was negative in *both* matches (-20.4 and -24.4 bb/100, intervals excluding zero); the small
  blind seat was not. The seat split mostly measures the value of position, which no strategy removes (the small
  blind acts last after the flop), so the pre-registered reading is the two seats' mean. A leak in the big-blind
  defence is plausible, not shown (`records/slumbot/RESULT.md`, `records/slumbot-2/RESULT.md`).
- The second run's point estimate is lower than the first despite the stronger configuration. The difference
  is within noise, and the evidence does not support "the upgrade made it worse" or "better".

Integrity checks (all in the result files): hand numbers contiguous; every one of 50,100 payoffs recomputed
locally and equal to the server's; Slumbot's own running totals equal ours to the chip; 13 hands lost to network
failures were excluded under a pre-set cap (250), with the reason for each recorded. Per-hand logs:
`records/slumbot/hands.jsonl.xz`, `records/slumbot-2/hands.jsonl.xz`; analysis: `scripts/slumbot_report.py`.

## 2. The internal rating (PR)

Design in [docs/rating.md](docs/rating.md). One number, chip-linear, with an exact standard error: the
difference in mean winnings against a frozen pool of anchor bots (Nash-averaged, so near-copies cannot inflate a
rating), relative to the v1 100 bb blueprint, which defines 1500. Paired deals, AIVAT variance reduction.

| configuration | PR | 95% |
|---|---|---|
| v1 100 bb blueprint (reference) | 1500 | by definition |
| 200 bb blueprint, 2B iterations (`A2`) | 1665.6 | re-verified |
| live bot with search (rated 13 Oct 2026, blueprint A2) | **1860.4** | [1771.4, 1949.4] (quoted from the write-up; that receipt is not in this tree) |

The live bot has changed since (an 8B blueprint, turn and river upgrades, a new flop net); those changes were
each judged by their own pre-registered test (below) and the whole has not been re-rated.

## 3. What was adopted, what was not

Each row is a pre-registered experiment with its own write-up in `records/`.

| experiment | question | result | decision |
|---|---|---|---|
| FLOPSEARCH | search the flop with a learned value net at the leaves | M1 benchmark: **-2.1% of the pot** vs the blueprint alone (lower is better) ([docs/flop-search.md](docs/flop-search.md)) | adopted |
| KERNELS Q4 | faster GPU terminal evaluation | precise, and 2.9x faster than Q3 (the float64-precise terminals) ([docs/gpu-kernels.md](docs/gpu-kernels.md)) | adopted; later kernel variants did not pass |
| RIVER R1 | river search at 2,000 iterations instead of 200 | +11.28 bb/100 [+2.85, +19.72] over 10,000 deals (AIVAT +7.39 [+1.13, +13.65]); river 1.29% of the pot from equilibrium at 200 iterations, 0.05% at 2,000 | adopted (`records/river-live`) |
| BP-8B | train the blueprint to 8 billion iterations | v2-8B vs the 2B blueprint: **+27.6 mbb/hand** [24.3, 30.9] on the rating scale; live match -0.61 bb/100 [-19.4, +18.2] (not down) | adopted (`records/bp-8b`) |
| D2b | a better flop net (N1) from more turn-start data | 160 benchmark spots: **-0.585% of the pot** [-0.757, -0.412]; 10,000-deal match -7.24 bb/100 [-23.80, +9.32], not down | adopted on its pre-registered gates; the match is inconclusive (`records/d2b`) |
| LEAF-NET | a weight-decay leaf net | 9% lower leaf error on average, but on one benchmark spot the loss was 8.1 points of the pot worse than the live net (`m1-report-leafnet.json`) | not adopted (`records/leaf-net`) |
| FLOP-SCHED | alternative flop update schedules | none beat the live one at equal precision | not adopted (`records/flop-sched`) |
| LEAF-EV | a matchup-factor leaf net | stopped at selection (loss +0.45%) | not adopted (`records/leaf-ev`) |
| TRANSLATE | a different off-tree bet translation | the stand-in reading failed its pre-registered adoption rule | not adopted ([docs/translation-and-exploitability.md](docs/translation-and-exploitability.md)) |
| preflop-open F1/F2 | handle small preflop opens better (F2: patch the strategy; F1: retrain with them in the grid) | F2 was worse in all four LBR arms; F1 was *more* exploitable against 2 bb and 2.5 bb opens (+34.5 and +21.2 bb/100) and lost head to head (PR -4.8 mbb/hand) | neither adopted (`records/preflop-open`) |
| D3 | a better flop net from another 1,000,008 turn-start rows | labelling paused at about 500 shards (501 at the second look) by choice; the look read CONTINUE (-7.10% vs a -4% bar) | in progress (`records/d3`, `records/d3-sweep`) |

## 4. Validation (the ladder under all of it)

A separate re-check of every quoted result against known defects (a play bug, a solver precision issue, a statistics fix) is in [docs/verification.md](docs/verification.md): most results stood, a few numbers were replaced.

| check | result |
|---|---|
| Kuhn poker, CFR | exact game value -1/18 |
| Leduc hold'em | exactly 288 information sets; simultaneous-update DCFR reaches ~1.6e-2 chips exact exploitability at 2,000 iterations, with the expected ordering DCFR <= CFR+ < vanilla |
| Rust vs Python solvers | exploitability of the Rust and Python solvers' strategies agrees to 1e-6 at equal iterations (a test in `tests/`) |
| 7-card hand evaluator | per-category counts match the known combinatorial totals over all 2,598,960 five-card and all 133,784,560 seven-card hands |
| Suit isomorphism | 1,326 holdings -> 169 classes; 22,100 flops -> 1,755 classes |
| Live client | every payoff of 50,100 Slumbot hands recomputed locally; a 1,000-hand mock and a 10,000-history parser check gate every live run |

## 5. Closed research track

`rmapss*` (multiplayer opponent exploitation) was a separate hypothesis-driven track. One narrow claim was
confirmed; the later stages failed their locked gates and two hypotheses were killed. It is kept in the tree
because its negative results are recorded in tests and receipts; the bot does not use it.
