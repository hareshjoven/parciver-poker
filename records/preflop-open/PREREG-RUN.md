# LBR-open and F2: the run, pre-registered before any LBR-open data

Written 11 Oct 2026, ~14:45, before any LBR-open or F2 number exists. It makes FINDING.md's plan
concrete for this code. The plan's measurement, statistics and decision rule are unchanged; where this
file is more exact, it governs. F1 runs only if F2 fails the rule, or if the user chooses it.

## What is measured
- **v2** = v2-2B (`runs/blueprint-20260927-141645/blueprint.bin`, sha256 `1976A53C…`, A2 in the pool)
  as it plays now: `SavedBlueprint(...).play_agent()`, translation `snap`, no search.
- **F2** = the same agent with `.with_open_reading("limp")` (engine/src/play.rs `OpenBelow::Limp`).
  The small blind's first preflop raise is the hand's first action, made into the blinds' pot. When it is
  below the smallest grid size (x < 1 pot), it is read as `Between{limp, pot-size open}`, with
  P(limp) = (1 − x) / (1 + x): 1/3 for 2bb, 1/7 for 2.5bb. Under a limp reading the big blind's real
  call is its abstract check. Nothing else is read differently (tests:
  `an_open_below_the_grid_is_the_grid_open_or_with_f2_between_the_limp_and_it`,
  `f2_changes_no_reading_but_the_small_blinds_open`).
- **LBR-open(s, opener)** (engine/src/lbr.rs `Open`). In the small blind, LBR's first action is replaced;
  everything after it is standard LBR. LBR in the big blind is unchanged.
  - `all`: raise to s with every hand.
  - `bp`: v2-2B's blueprint root decision for LBR's hand (`decide_inner`; a non-blueprint source is an
    error). It is sampled with u = splitmix(hand seed ^ `OPEN_STREAM`), LBR's own stream, so it is the
    same draw against v2 and against F2. Where the decision is the pot-size open (Add 250), LBR raises
    to s instead; fold, limp and all-in are kept. The opener is always v2-2B (snap), for both bots.
  - The engine counts `open_counts = [LBR raised to s, the agent folded to it, the agent raised it]`.
- **v2-2B's small-blind root** (uniform over the 1,326 hands, 200bb stakes 50/100/20000; read from the
  blueprint on 11 Oct, before any measurement): **open 61.252%, limp 30.591%, fold 8.157%, all-in
  0.000%.** So `bp` opens 61% of hands, far from `all`'s 100%, and (a′) is a real guard.
  - A first count read A1's file (`blueprint-20260927-060745`: 58.46 / 34.06 / 7.47%). It was caught
    and redone before this file existed; A1 plays no part here.

## Decision arms (no search; CPU)
- Stakes 50/100/20000, the scorecard's LBR seed 410101, deals 0..99,999, mirrored (two hands per deal:
  LBR in each seat), `lbr_peek = False`, `lbr_sizes = fixed`, chunks of 10,000 deals (results do not
  depend on chunking).
- The runner is `records/preflop-open/lbr_open.py decide`. It refuses a dirty tree (engine/src,
  engine/Cargo.toml, engine/pyproject.toml, src, scripts, records/preflop-open/*.py and this file), an
  engine without `with_open_reading`, and an engine the rating is not re-verified on (every pool-v2
  receipt VERIFIED under the installed engine's identity). It records the commit, the code's git trees,
  the engine build's sha256 and identity, the abstraction's sha256 and the blueprint's sha256. Each run writes `runs/preflop-open/decide-<UTC stamp>/` (per-deal arrays, npz,
  never overwritten) and a JSON summary next to this file. The first complete run that is not VOID
  counts, and every run is kept.
- Arms, in this order:
  1. standard peek-free LBR (no open) on v2, then on F2;
  2. s = 300 (3bb, on the grid: the control), opener `all` then `bp`, each on v2 and then F2;
  3. s = 200 (2bb), the same four;
  4. s = 250 (2.5bb), the same four.
- **Gates.** If any gate fails, the run is VOID (a bug, not a result): nothing is reported as a
  measurement, and the failure goes to the user.
  - G1: arm 1's per-deal results are bit-identical between v2 and F2 (standard LBR never raises
    preflop).
  - G2: s = 300 is bit-identical between v2 and F2, for both openers (F2 never fires on the grid's open).
  - G3: `all` arms have open_counts[0] = 100,000 (one small-blind hand per deal); `bp` arms have
    open_counts[0] equal between v2 and F2 (the same draws).
  - Descriptive, decides nothing: arm 1 on v2 against `translate-postfix/v2-2B-nopeek.npz` (an older
    engine, 3 Oct) on the same deals. It is reported as "reproduces" or "differs"; if it differs, that
    is shown to the user with the result.
- **Statistics.**
  - Per-deal net in bb. Δ(s, opener) = LBR-open_F2 − LBR-open_v2, paired per deal: mean ± 1.96 SE
    (ddof = 1). It is reported in bb/100 hands (2 hands per deal), as V2LEAK and TRANSLATE report.
    Only the small-blind hand can differ, so per 100 small-blind hands it is twice that (descriptive).
  - Status: DOWN if the CI is entirely below 0, UP if entirely above, else FLAT.
  - Descriptive: each arm's LBR; the price LBR-open(s) − LBR-open(300) per bot; open_counts (the agent's
    fold and raise rates against the open); the postflop and preflop-ended parts.

## (b), (c), (d) for F2: identity checks, not LBR arms
- **A departure from FINDING.md, recorded here.** FINDING says (b) and (c) hold by construction for F2
  and are "checked by a test … not by a run". The engine tests do check it on small samples. This file
  goes further: (b) and (c) are also checked at full size, as bit-identity against v2 on the same deals
  (the scorecard's LBR stream and the whole rating pool). That is stricter than the rule, never looser,
  and it costs only CPU time. A difference is not a measured effect: by construction there can be none.
  So a difference in either is a bug, not a measurement: that check is VOID, it goes to the user, and
  F2 is not adopted while it stands.
- These run on the new engine after it is installed, and after the rating is re-verified on it. Both
  runners refuse an engine under which any pool-v2 receipt is not VERIFIED
  (`pool_identity.rated_engine`), and a tree with uncommitted changes to the code under test (as for
  the decision arms, plus engine/Cargo.toml and engine/pyproject.toml). Each records the commit and
  the engine's identity.
- **(b) standard LBR not UP.** `scorecard.py evaluate` on v2 (label `v2-2B-preopen`), then on F2
  (`--open-reading limp`, label `v2-2B-F2`, `--previous` the former). Then
  `records/preflop-open/lbr_open.py same-lbr` on the two receipts compares their `raw.lbr`
  `per_deal_bb` (and `per_deal_preflop_bb`) bit for bit (5,000 deals, the peek, seed 410101) and writes
  `same-lbr-<stamp>.json`. IDENTICAL makes (b) FLAT by identity. G1 is the same on the peek-free stream.
- **(c) PR not DOWN.** `records/preflop-open/pool_identity.py` replays A2's VERIFIED receipt
  (`runs/rating/pool-v2/receipts/20261003T151853368534Z-A2`) with F2. That means every anchor's
  per-deal array at the pool's seeds and protocol counts (25,000,000 deals) and its LBR (20,000 deals),
  each compared with the receipt's arrays by `numpy.array_equal`. If all are identical, PR(F2) = PR(A2)
  exactly. If any array differs, the check is VOID, as above.
- **(d) no scorecard row DOWN.** The scorecard rows of `v2-2B-F2` against `v2-2B-preopen`, paired, with
  the scorecard's own status. `min_raiser` (2bb opens), `baseline_sizes_x0.6` (2.2bb) and
  `baseline_realistic_sizes` (a third at 2.75bb) are the rows F2 can change.
- scorecard.py gains `--open-reading {snap,limp}`. It applies to the candidate only, is recorded in the
  receipt's candidate as `open_reading` (only when not snap, so older receipts read as snap), and is
  re-applied by `lbr-split`. rating.py does not change now; if F2 is adopted, its configuration identity
  gains `open_reading` before any F2 receipt is made.

## Confirmation arm (descriptive; vetoes nothing; GPU)
- Live config: v2-2B with `with_search(from_turn=True, turn_iters=100, river_iters=200, rescue=False,
  solver=Watched(SubgameGpu(terminals64=True)))` (Q4, as `slumbot_run.py` plays since 11 Oct). This
  departs from FINDING.md's "terminals64", which meant Q3 when it was written; Q4 is what plays now.
  F2 is the same with `.with_open_reading("limp")` applied before `with_search`.
- LBR-open(200, `all`), with the peek (peek-free LBR refuses search), deals 0..4,999, seed 410101. v2
  and F2 are interleaved in chunks of 50 deals, so that a stop leaves paired deals. A failed search
  stops the chunk, and nothing of that chunk is kept. Resumable only on the same engine, code trees and
  configuration (a commit that changes nothing under test may differ).
- Reported: Δ with its CI and status. The peek sees how F2 read each open, so this arm is biased against
  F2. If Δ is entirely above 0, it is reported to the user with the decision.

## Decision (FINDING.md's rule, unchanged)
ADOPT F2 if all of these hold: (a) Δ(200, all) and Δ(250, all) are both DOWN; (a′) Δ(200, bp) and
Δ(250, bp) are not UP; (b) standard LBR is not UP; (c) PR is not DOWN; (d) no scorecard row is DOWN.
Otherwise do not adopt, and name what failed. If only (c) or (d) fails, the trade-off goes to the user.
Adopting means making F2 the default reading of the live agent (`slumbot_run.py`, the scorecard and the
rating). That is a later change, made with the user's go, after this result.

## Order
1. One read-only audit of the engine change, this file and the runners; fix the findings; commit.
2. Build the engine and install it into the venv. Re-verify the rating (the receipts are STALE on a new
   engine): pytest, crosscheck, verify, current, audit.
3. The decision arms; (b) and (d) (scorecard); (c) (pool identity).
4. The confirmation arm (GPU; it needs the GPU to be free).
5. RESULT.md; one read-only audit; commit; report to the user.
