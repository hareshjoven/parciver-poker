# FLOP-SCHED: a cheaper flop solve schedule (IDEAS SC2, scout)

Registered 2026-10-14 (the commit that adds this file is the registration), before any SC2 code, run or
data. The user's go: "both if it's fine" (14 Oct, SC2 and SC5 after LEAF-NET's Gate B). A scout: it changes
nothing live. A pass only recommends a live gate, which is registered separately with the user's go. Audited
(read-only) before commit; the audit's findings are folded in.

## Why
- The live flop search runs 500 iterations of `FlopSolver`: alternating updates, DCFR with alpha 1.5, negative
  regrets halved, average weights gamma 5.
  - It converges: 0.20% of the pot on the check-down game (FLOPSEARCH, D3 calibration).
  - Its cost is the net. Each iteration makes two leaf calls, one per player (`flop_search.py`:497-509), at
    ~4.4 ms per iteration (KERNELS Step 3).
- **Flop latency is close to the bar.**
  - The registered C2 mock's flop p95 is 2.791 s against the 3.0 s bar (`records/flop-sym/mock_check.json`).
  - With dense N0 leaves, a hand's first flop decisions facing a bet ran 2.863 s median and 2.889 s p95
    (`records/live-flop/PREREG.md`:105-106).
- **The idea comes from Supremus** (Zarick et al. 2020, section 4.1).
  - They found simultaneous updates converge faster than alternating when a value net sits at the depth
    limit. Their setting was a preflop subgame, measured per iteration.
  - Without a net, they note, the opposite has historically held.
  - A simultaneous iteration here needs one leaf call, because every leaf path returns both players' values.
    It has never been tried on this solver.

## The changes (`FlopSolver(..., simultaneous=False, avg_delay=None)`; defaults unchanged)
- **`simultaneous=True`:** one iteration does the following, after the same `begin` (the regret discount).
  - Regret matching once (`current`), and the reach once.
  - The strategy sums of every unfrozen slot, each with its owner's reach, as today.
  - The leaf values once, for both players: `leaves(...)` as today, or `players=(0, 1)` for a one-player
    evaluator. They go into their own buffer, not `self.val`.
  - Then, per player, terminals and `_up(p, sig, "cfr")` with that same `sig` (`self.val` is one buffer).
  - `step` and the CUDA-graph capture in `solve` (:536-539) both take this path.
- **`avg_delay=d`:** iteration t's weight in the average strategy is max(0, t - d). This is Supremus's "DCFR+",
  not Xu et al.'s DCFR+, which LIVESEARCH lists for the turn.
  - Supremus's DCFR+ is DCFR(1.5, 0, 2) with only the averaging changed. Our regret discount is already
    DCFR(1.5, 0) (alpha 1.5, negatives halved), so the d100 arms are its DCFR+ exactly.
  - **The `_w` table:** index t - 1 (read at iteration t) holds (t - 1 - d) / (t - d) for t > d + 1, and 0 for
    t <= d + 1. The sums restart at t = d + 1. Nothing else changes.
  - d is set before the first `_grow`. Gamma is ignored when `avg_delay` is set.
  - **Through iteration d + 1, the "average" is the last iterate.** Every weight up to t = d is 0. So the d100
    arms' checkpoint 100 is the current strategy at iteration 100.
  - It changes no kernel, so its cost per iteration is its update scheme's.
- **Tests before any data** (`tests/test_flop_search.py`):
  - **The default path is unchanged.** The existing tests pass, and the default `_w` table is the same tensor.
  - **Graph replay** is bit-identical to eager steps for each new option and for both together.
  - **One leaf call:** `simultaneous` makes exactly one leaf call per iteration (a counting evaluator).
  - **A float64 reference for `simultaneous`, over 5 iterations on a small spot.** It is written out
    directly: one `current`, the sums weighted by the owner's reach, both players' regrets from the same
    `sig` and the same leaf values. The regrets and sums must match to float64 rounding.
  - **Convergence:** `simultaneous` converges on the check-down game to the full game's value.
  - **`avg_delay` (n > d, float64, a small spot), under both update schemes:** `average()` equals the direct
    weighted average, over iterations, of the strategy each slot's sum update used at that iteration. The
    weights are max(0, t - d) times the owner's reach. (Under alternating updates, player 0's slots use
    the `sig` computed after player 0's own regret update.)

## The scout (`scripts/flop_schedule.py`)
- **Spots:** the 40 D3 calibration spots, seed 540,000.
  - File: `runs/flop-eval/spots-540000.json`, untracked, sha256 92ccb7e2...5117c1a, checked at the start.
  - Each round is solved from its first action with no frozen nodes, as `flop_eval.py calibrate` does
    (`root = dict(job, frozen=[])`).
  - Their pots are 2bb (12), 6bb (24) and 18bb (4).
- **Arms (4):**

  | arm | updates | averaging |
  |---|---|---|
  | `alt-g5` (today, the baseline) | alternating | gamma 5 |
  | `sim-g5` | simultaneous | gamma 5 |
  | `alt-d100` | alternating | max(0, t - 100) |
  | `sim-d100` | simultaneous | max(0, t - 100) |

  - Regret discounting is the same in all four. d = 100 is Supremus's value; no other d is tried.
  - Expected, not a gate: Supremus's finding is about net leaves, so `sim` may lose on Game E, which has none.
- **Checkpoints:** 100, 200, 300, 400, 500, 600, 700, 800, 900 and 1,000 iterations of one run per arm and spot. No
  arm's average at n depends on later iterations.
- **Game E (exact), set up as `calibrate` does:**
  - `NetLeaves(board, CheckdownValues(board, dtype=torch.float32))`, `runout_equity(board)`, and `FlopSolver`
    in float32 with the graph, with gamma passed explicitly (5.0).
  - E = `100.0 * s.exploitability() / pot`, with pot = `flop_eval.pot_of(job)`: the sum of both best
    responses, in % of the round-start pot.
- **Game N (the live leaves):**
  - The live net when the scout runs (see Order), identified by sha256 and recorded:
    - P, `runs/turn-net/leafnet-c0-wd0.1/net.pt`, 0c196d46...d45489;
    - or N0, `runs/turn-net/final-w2048-d5-e200/net.pt`, a8bf00be...38ef65 (both in
      `records/leaf-net/gate-a.json`).
    - Any other sha256 stops the scout.
  - The leaf path is the one `FlopSearch(leaves="sym")` would take: `SymLeaves` (C2) when the flop has a
    stabilizer beyond the identity and the ranges are symmetric, else dense `NetLeaves`.
  - N = `FlopSolver.exploitability()`, in % of the pot. With a net, this is a proxy: each leaf's ranges are
    held at the average profile's (D3: no exact meaning).
- **Time:**
  - ms per iteration of the replayed graph on Game N's leaves, for `alt-g5` and `sim-g5`.
  - Per spot, in the order A B B A (A = `alt-g5`): a fresh solver, the capture, 20 warm-up replays, then 200
    timed replays with `torch.cuda.synchronize` around them.
  - ms_X = the total time over the total iterations, over all spots. The d100 arms cost the same as their
    update scheme's g5 arm, by construction.
  - This is solve time only. Table builds and the turn's searches are not in it.
- **Reproduction check:** `alt-g5`'s Game E at 100, 200, 300, 500 and 1,000 iterations must equal
  `runs/flop-eval/calibrate.json`'s gamma 5 rows, spot by spot, bit for bit.
  - The file is untracked; sha256 ead93dd4...4ddfca7, checked at the start.
  - If they differ, nothing else runs until the cause is found and an amendment records it. Unexplained:
    the scout is VOID.
  - Cause candidates: driver or cuBLAS changes. A code change on this path is a bug.

## The decision
- **The baseline's bars:** `alt-g5`'s values at 500 iterations.
  - E_b and N_b are its means over the 40 spots.
  - E_b2 is its Game E mean over the 12 2bb spots, the hardest class (0.47% at 500 in D3).
- **Each other arm X:**
  - n_X = the smallest checkpoint **from which** all three bars hold at it and at every later checkpoint, with
    no tolerance: X's mean E <= E_b, its mean N <= N_b, and its 2bb mean E <= E_b2.
  - If there is no such checkpoint, X fails.
  - Cost_X = n_X * ms_X, against Cost_b = 500 * ms_alt.
- **X passes if Cost_X <= 0.85 * Cost_b:** at least 15% less flop solve time at no loss on any bar.
  A smaller gain is not worth an M1 gate (~18 CPU-hours) plus a new live code path.
- **Pick:** among the passing arms, those with Cost_X <= 1.03 x the lowest passing Cost_X are tied. A tie
  goes in the order `alt-d100` (a table only), `sim-g5`, `sim-d100`: the least new code first.
- **A pass recommends a live gate for the pick at n_X**, registered with the user's go:
  - the M1 non-inferiority test on the live leaves;
  - the latency mock against the 3.0 s flop bar.

  Nothing changes live before that gate passes.
- **No pass:** SC2 closes, and the result is reported in IDEAS.md.
- **Reported, not gating:**
  - per-iteration curves of E and N for all arms;
  - the tables by pot class (2bb, 6bb, 18bb);
  - per-spot n at which each arm reaches the baseline;
  - the time ratio ms_sim / ms_alt and the GPU.
- **Output:** `records/flop-sched/scout.json`, holding:
  - rows per arm, spot and checkpoint;
  - the net's sha256 and both input files' sha256s;
  - the commit and the torch version.
- **Cost:** about 40 spots x 4 arms x 1,000 iterations on each game, plus timing. Under an hour of GPU (an
  estimate).

## Order
- **The code lives on its own branch** (`scout-sc2`, in the worktree `-sc2`; SC5
  has its own), and the scout runs from there. `rating.py` hashes every `src/parciver/**/*.py` into the play
  identity (`scripts/rating.py`:213-249), so main and the live receipts stay untouched.
  - Its input and net files are read from the main checkout's `runs/` (untracked), by the sha256s above.
  - **The result goes to main whatever the verdict:** `records/flop-sched/scout.json` and the report, records
    only, no code.
  - Only a passed live gate merges code into main, followed by `rating.py verify`, as that gate's
    registration will say.
  - A failed scout's branch is kept as an `archive/*` tag.
- **The scout runs after LEAF-NET's Gate B and its runbook step 5** (re-verify, the NET switch, the mock, the
  rating), with the GPU otherwise idle.
  - The live net is then settled: P if the switch happened, N0 otherwise.
  - It never runs beside an M1 oracle or a blueprint-loading job (RAM).

## VOID and bugs
- **VOID:** a crash, kill or reboot, or another compute job on the GPU. The run is VOID, its cause recorded, and
  it is repeated.
- **A bug found after any data:** that run is VOID and still reported. The fix and the rerun need an amendment
  here first, and both results stay in the record.

## Not allowed after this commit
- Changing the arms, d, the checkpoints, the spots, the bars, the 15% margin or the pick order after any SC2
  data.
- Adding an arm after seeing results. A new schedule needs a new registration.

## Amendment 1 (15 Oct 06:23, after the code audit, before any SC2 data)
- **The live net is N0.** LEAF-NET's Gate B failed (main ae3213c), so its runbook step 5 (re-verify, NET switch,
  mock, rating) never ran. "After step 5" is now; the code accepts only N0's sha256 (a8bf00be...38ef65) for Game
  N, and refuses P.
- **The reproduction check runs first, on all 40 spots:** alt-g5's Game E alone, compared with `calibrate.json`
  before any other arm, game or timing runs. A mismatch then leaves no arm data, as "nothing else runs" intends.
  Those rows are alt-g5's Game E rows in the scout (the same computation, not repeated).
- **A record is never overwritten:** the scout refuses if `scout.json` exists. A VOID run's record is moved to
  `scout-void-<n>.json` and reported.
- **Recorded:** the GPU's utilization and memory at the start and the end, and any other Python process holding
  the GPU. Under Windows' WDDM, nvidia-smi lists every process with a graphics context, including a
  desktop app's own tool processes; those are not compute jobs. The spots' pots are checked as 12 x 2bb, 24 x 6bb
  and 4 x 18bb.
- **Tests added:** an evaluator that takes `players` (FactoredLeaves' calling convention) gives a bit-identical
  solve, alternating and simultaneous; the 15% margin and the 3% tie are inclusive at their boundaries; the
  refusals above.
