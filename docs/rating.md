# PR: one reproducible number for the bot's strength

Poker results are noisy, and a single head-to-head number or a quick Elo-style score is easy to fool oneself with. This document describes **PR (parciver rating)**, a strength number pre-registered on 2026-10-03 before any rating data existed, built so that it is computed only by code from stored per-deal results, can be replayed bit for bit, and carries an exact confidence interval. One PR point is 1 mbb/hand (0.1 bb/100). As of 13 Oct 2026 the live search bot rated PR 1860.4 (95% 1771.4..1949.4, VERIFIED) and the blueprint alone (anchor A2) 1665.6, re-verified on 12 Oct. The live bot's interval is wide (about +/-89 points) because its rating uses only 20,000 deals per anchor; see the limits below.

## Definition

For a bot configuration x (blueprint file + search settings + bet translation):

    PR(x) = 1500 + sum_j p_j * [u(x, j) - u(A0, j)]

- u(x, j) is x's mean win against anchor j in mbb/hand, heads-up NLHE at 200bb (blinds 50/100, stacks 20,000 chips), on mirrored seeded deals: every deal is played twice with seats swapped, and deal i depends only on (seed, i).
- A0 is the fixed reference (the v1 100bb blueprint trained for 1e9 iterations) and defines 1500. x and A0 face each anchor on the same seeds, so the difference is estimated deal by deal and card luck cancels.
- p are **Nash-averaging weights** (Balduzzi et al., 2018): the maximum-entropy Nash equilibrium of the anchor pool's own zero-sum meta-game, frozen per pool version. A near-duplicate anchor splits one weight and cannot inflate a rating. In a transitive pool the weight sits on the strongest anchors, so PR measures play against the best frozen opponents relative to A0.
- Chips, not Elo. Elo models win probabilities; poker outcomes are chip amounts with heavy variance, and reducing them to wins discards information. PR is linear in measured chip results with frozen weights, so its standard error is exact.

**Pool v1 anchors:** A0 (v1 blueprint, plays 200bb through bet translation), A1 (200bb blueprint), A2 (200bb blueprint at 2B iterations), and Ln, Ls, Lm (A0 with a deliberate leak at strength 0.20: nit, station, maniac). Built-in engine opponents are not anchors.

## Calibration and acceptance

Each pool version is calibrated once per engine: every anchor pair plays 1,000,000 deals (Amendment 1; originally 200,000), the payoff matrix A is formed, and p is the maximum-entropy Nash equilibrium found by exact support enumeration (6 anchors, 63 supports). A pool is used only if all gating checks pass:

- **C1** determinism: replayed chunks are bit-identical.
- **C2** seed consistency: A0's results on rating seeds and calibration seeds agree within 3 combined standard errors.
- **C3** the Nash conditions hold (best-response gap at most 1e-9 of the largest payoff, p non-negative, sums to 1).
- **C4** role symmetry: swapped-role pairs are exact negations deal by deal (reported in Amendment 1, made exact and gating in Amendment 2).
- **C6** (gating) cross-pool determinism: pool-v2's calibration arrays equal pool-v1's bit for bit.
- **C7** (reported) continuity: pool-v2's PR for A1 and A2 should fall inside pool-v1's intervals.

## Why the number cannot be typed in

1. PR is recomputed from receipts' per-deal arrays on every view (`scripts/rating.py`, `src/parciver/rating.py`); each array's sha256 is checked first.
2. A receipt is **VERIFIED** only after a replay under the current engine and code that reproduces the per-deal results bit for bit. It is **STALE** when the engine or rating code has changed since (shown grey), and **INVALID** on any replay mismatch, hash mismatch or missing failed-search count. Only VERIFIED ratings are headlines.
3. The headline is the newest receipt of a configuration, identified by a fingerprint of its blueprint hash, search and translation (not a free-text label), with no fallback to an older receipt.
4. Tests include known-answer meta-games (rock-paper-scissors gives uniform weights, a transitive pool puts weight on the strongest, a duplicated anchor changes nothing), PR(A0) = 1500 exactly, replay determinism and tamper detection. `scripts/phase_audit.py` recomputes the headline from receipts and fails on any mismatch.

## Amendments (dated, appended after data)

- **Amendment 1 (3 Oct, before rating data):** sample sizes raised to 1,000,000 deals per pair, an LBR guard (local best response exploitability estimate, shown beside PR, never folded into it), role-symmetry check C4.
- **Amendment 2 (3 Oct, after pool-v1):** pool-v1 gave A2 PR 1679.5 [1660.7, 1698.2]. A code audit found paths where an under-verified number could reach the headline.
  - "VERIFIED" was redefined to mean replayed; the calibration is re-derived on every view; receipts are checked against the pool protocol.
  - A new pool-v2 (same anchors and seeds) plays 20,000,000 deals against every anchor with non-zero weight. The A2 half-width was expected to fall from +/-18.7 to about +/-5 points.
  - Search configurations were refused until the rating could play them with the solver actually used in play.
- **Amendment 3 (12 Oct, before any search-configuration data):** AIVAT variance reduction was adopted for the live rating.
  - AIVAT is a control-variate correction that uses known chance and strategy structure to cut per-deal noise without bias.
  - On a pre-registered test it was unbiased (z +0.05) and reduced variance to r = 0.491 of raw (95% CI 0.352..0.673). Receipts: `records/rating-live/`; replay determinism: `records/rating-live/determinism.json`.
  - The live configuration is rated only against A2 (the only anchor with non-zero weight) at D = 20,000 deals, with an expected 95% half-width of about +/-82 points (per-deal sd 11.86 bb).
  - Search receipts must record zero failed searches, and replays require both the AIVAT and the raw arrays to match bit for bit.

## What did not work, and limits

- The pool-v1 interval was too wide to be useful: A0's reference row was nearly uncorrelated with candidates' rows (per-deal correlation about 0.06), so pairing helped little. That motivated the 20,000,000-deal protocol of Amendment 2.
- The live bot's rating rests on 20,000 deals, so its interval (about +/-89 points) cannot separate small improvements. GPU search is slow, which set that budget.
- No LBR guard exists for search configurations: LBR would run the GPU search at every decision, taking days per receipt, so the row records that LBR was not measured.
- PR is only as informative as the pool. With one dominant anchor (A2) it measures play against that one opponent family, relative to A0; it does not measure exploitability, which is why LBR is kept as a separate guard where it can be run.
- Ratings from different pool versions are never mixed on one line.
