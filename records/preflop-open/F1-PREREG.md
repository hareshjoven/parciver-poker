# F1, the grid fix: pre-registered before any F1 training or measurement

Written 20 Oct 2026, ~23:55 SGT, on the user's yes (20 Oct, "yes to f1", after `records/rm2/SCOUT.md`). No F1
blueprint exists yet and no F1 number has been measured. FINDING.md's plan and rule govern; this file makes them
concrete for F1, as PREREG-RUN.md did for F2. Where this file is more exact, it governs. Slumbot is not a
criterion anywhere.

## The grid: `200bb-f1`
- `NlheConfig::deep_200bb_f1` (engine/src/nlhe.rs, grid name `200bb-f1`): the 200bb grid, except that the small
  blind's open (the first raise preflop) has the pot fractions {0.5, 0.75, 1.0}, i.e. opens to 2, 2.5 and 3bb.
  Every other node is the 200bb grid's. The new config field `preflop_open_fracs` is empty in every older grid,
  which then plays exactly as before.
- Engine tests (engine/src/nlhe.rs, engine/src/play.rs):
  - the root offers fold, limp, 2bb, 2.5bb, 3bb and all-in; each open has exactly one sized 3-bet, pot-size (to
    6, 7.5 and 9bb);
  - after a limp, the big blind's options are the 200bb grid's;
  - the subtrees under fold, limp, 3bb and all-in are node for node the 200bb grid's in count, and the F1 tree is
    the 200bb tree plus the two new subtrees;
  - the F1 agent reads the 2, 2.5 and 3bb opens each as its own grid open, in every translation mode, and F2
    never fires on them;
  - every older test unchanged (158 passed).
- Older grids unchanged, at full size: G0 below, and the rating's re-verification (every pool-v2 receipt without
  search replayed bit for bit on the F1 engine).

## The engine (built beside the installed one, never in place of it)
- Built from the registering commit by `records/preflop-open/f1_build.py`: it refuses an engine source that
  differs from HEAD, runs `maturin build --release -m engine/Cargo.toml`, unpacks the wheel into a fresh
  `runs/f1-engine/site/` and writes `runs/f1-engine/BUILD.json` (commit, engine trees, .pyd sha256).
  `f1_eval.py` refuses a loaded engine that is not that .pyd, or whose engine trees are not HEAD's. Every F1
  command runs the venv python **without `-I`**, with
  `PYTHONPATH=runs/f1-engine/site`, so it loads this engine.
- The installed engine (E58C7292) is not rebuilt or replaced: R1, BP-8B's match and the runner's GATES pin it.
  The F1 .pyd's sha256 is recorded in RESULT before any arm.
- Installing it in the venv is part of a live adoption only (below).

## Training (CPU)
- **Stage A (decides):** `scripts/train_blueprint.py --grid 200bb-f1 --target-iters 2000000000 --threads 12
  --run-dir runs/f1-stageA --no-browser`. Variant linear, seed 0 and `data/abstraction` are the defaults.
  - This is v2-2B's recipe: linear external-sampling MCCFR, seed 0, the same abstraction, 2,000,023,000
    iterations.
  - v2-2B ran 16 threads and was resumed at 1B. Parallel MCCFR is not bit-reproducible, so neither the thread
    count nor a resume matters beyond that noise.
  - The parent shell is BelowNormal and the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`). Expected ~7 h at ~77.5k it/s
    (870532e, the 200bb table); longer, since F1's table is larger and other work runs beside it.
- **Beside D1b.** D1b's 4 PC workers (GPU-bound, ~0.9 core each) run at Normal priority, the trainer at
  BelowNormal. Before the trainer the GPU ran at 100% (20 Oct 23:52 SGT) and each worker at 0.18-0.19 solves/s
  (a shard takes ~1.4 h). The assistant samples the GPU's utilization in the trainer's first minutes and reads each
  worker's next shard line. If the GPU falls below 90%, or a worker's solves/s more than 10% below its
  pre-trainer rate, the trainer is stopped and resumed after D1b.
- **Stage B (adoption only):** `--resume runs/f1-stageA --target-iters 8000000000 --run-dir runs/f1-stageB`
  (the same flags; from stage A's last folder if stage A was resumed, see VOID), so stage A stays on disk. It may train once stage A is complete. It is evaluated only as
  "Adoption" below says.
- **VOID:** a refused resume, or a stage ending other than `completed`. An interruption is not a VOID: the
  stage resumes from its latest folder's checkpoint into a new folder with the same flags,
  `--resume runs/f1-stageA --target-iters 2000000000 --run-dir runs/f1-stageA-r1` (then `-r2` from `-r1`, and so
  on; likewise for stage B). The stage's blueprint is the last folder's `blueprint.bin`, and every command below
  that names `runs/f1-stageA/blueprint.bin` means that file. Each blueprint is pinned by its sha256 once
  written.

## Before any arm: the rating re-verified on the F1 engine
- Under the F1 engine, from the main folder: `pytest tests/test_rating.py tests/test_rating_receipts.py`, then
  `rating.py crosscheck --dir runs/rating/pool-v2 --other runs/rating/pool-v1`, `rating.py verify --dir
  runs/rating/pool-v2` and `rating.py audit --dir runs/rating/pool-v2`.
- Every pool-v2 receipt without search must be VERIFIED. The verification lines are appended; those under
  E58C7292 stay.
- `rating.py audit` will report FAIL on the F1 engine, and that is expected: the live receipt is STALE and the
  current headline is `live-flopsym`. The audit passes for F1 when its only failures are that receipt and that
  headline; any other failure stops F1 and goes to the user.
- **A departure from lbr_open.py, recorded here.** lbr_open.py's `rated_engine` requires every receipt. The
  live receipt (`live-flopsym`, 13 Oct) replays only with the live runner's search as it was then; it is STALE
  under every engine since B1. So `f1_eval.py` requires every receipt without search and lists the live ones.

## Decision arms (no search; CPU): `records/preflop-open/f1_eval.py decide`
- Bots: **base** = v2-2B (`SavedBlueprint(...).play_agent()`, sha256 `1976A53C…`); **fix** = F1-2B (stage A's
  `blueprint.bin`, grid checked). Both use translation `snap` and snap opens, with no search.
- As PREREG-RUN: stakes 50/100/20000, the scorecard's LBR seed 410101, deals 0..99,999, mirrored, peek-free,
  `lbr_sizes = fixed`, chunks of 10,000.
- Arms: standard LBR, then s = 300, 200, 250, each with the openers `all` and `bp`, each on base and then fix.
  - `bp` is always v2-2B's root (`decide_inner`, the `OPEN_STREAM` draw), for both bots, so both meet the same
    opens.
- The runner refuses:
  - a dirty tree (engine/src, engine/Cargo.toml, engine/pyproject.toml, src, scripts,
    records/preflop-open/*.py, this file);
  - an engine the rating is not re-verified on;
  - a base whose sha256 is not the pinned one;
  - a fix not trained on `200bb-f1`.

  It records the commit, the code's git trees, the engine (sha256, identity, path), the abstraction and both
  blueprints. Each run writes `runs/preflop-open/f1-decide-<stamp>/` and `f1-decide-<stamp>.json` beside this
  file. The first complete run that is not VOID counts, and every run is kept.
- **Gates.** Any failure makes the run VOID (a bug, not a result), and it goes to the user.
  - **G0:** each v2-2B arm equals the 11 Oct run's v2 arm (`runs/preflop-open/decide-20261011T075534Z/`, engine
    E93347A5) per deal, bit for bit, in total and in the preflop part. So the old play is unchanged on this
    engine. A dry run (20 Oct, the F1 engine as built before the commit, a 420k-iteration toy F1 blueprint)
    matched on deals 0..199 in all 7 arms.
  - **G3:** `all` arms open 100,000 times. `bp` arms open 61,271 times, 11 Oct's count, since the draw is v2-2B's
    whoever plays.
  - **G5:** the fix's standard LBR is not the base's deal for deal, which would mean the same blueprint.
  - PREREG-RUN's G1 and G2 do not apply: F1 is another blueprint, so no arm is equal by construction.
- **Statistics** as PREREG-RUN. Δ = fix − base, paired per deal, mean ± 1.96 SE (ddof 1), in bb/100 hands (2
  per deal); DOWN, UP or FLAT. Also reported: the postflop and preflop-ended parts, each arm's LBR, the price
  LBR-open(s) − LBR-open(300) per bot (decides nothing), and `open_counts`.

## The rule (FINDING.md's, for F1 against v2-2B)
ADOPT if all of these hold:
- **(a)** Δ(200, all) and Δ(250, all) are both DOWN.
- **(a′)** Δ(200, bp) and Δ(250, bp) are not UP.
- **(b)** standard LBR is not UP: Δ(std) of the decide run, 100,000 deals, peek-free. It is chosen because it
  matches the arms and is stricter (20 times the deals). Snap still randomizes some readings (a size between two
  grid sizes; a bet committing half the stack or more, read between the largest size and all-in), so this can
  differ from the scorecard's LBR (5,000 deals, the peek), which is reported beside it.
- **(c)** PR is not DOWN.
  - `rating.py rate --dir runs/rating/pool-v2 --candidate runs/f1-stageA/blueprint.bin --label F1-2B
    --no-browser`: no search, snap, the pool's protocol. Its receipt must be VERIFIED by `rating.py verify`.
  - Then `f1_eval.py pr --fix-label F1-2B --base-label A2`: the paired change `parciver.rating.change` (as
    BP-8B), DOWN iff the 95% interval is entirely below 0.
  - **A departure from FINDING, recorded here.** FINDING's (c) compares against the current headline. That is
    `live-flopsym`, which plays with search and cannot replay on the F1 engine. F1 replaces v2-2B, so (c)
    compares against v2-2B's anchor receipt A2, as BP-8B compared two blueprints.
  - Under E58C7292 the F1 receipt cannot replay (that engine has no `200bb-f1`), so it stays STALE there, like
    the live receipt. Nothing pinned reads pool-v2; `scripts/phase_audit.py` reads it, and its rating gate,
    already failing on the live receipt, will also list the F1 receipt as not replayable under E58C7292.
- **(d)** no scorecard row is DOWN.
  - `scorecard.py evaluate --benchmark runs/scorecard/scorecard-v2-200bb/benchmark.json` on v2-2B (`--candidate` its
    blueprint, `runs/blueprint-20260927-141645/blueprint.bin`; `--label v2-2B-f1base`), then on F1-2B
    (`--candidate runs/f1-stageA/blueprint.bin --label F1-2B --previous` the former's receipt), both on the F1 engine,
    `--no-browser`.
  - (d) reads the F1 receipt's paired row statuses. Every row counts, including `min_raiser`,
    `baseline_sizes_x0.6` and `baseline_realistic_sizes`, the rows F1 aims at.

If any of (a)-(d) fails, F1 is not adopted, and RESULT names what failed. If only (c) or (d) fails, the trade-off
goes to the user, as FINDING says. F1 is another blueprint, so (b)-(d) also carry two solves' training noise;
they guard against F1 being worse overall, which is their job.

FINDING's confirmation arm (live config, the peek, GPU; descriptive, it vetoes nothing) is not part of this
decision. It runs only with the user's go, when the GPU is free.

## Adoption (not decided here)
ADOPT makes F1 the candidate for the live config. Making it live is a later change, under its own registration
and with the user's go, as BP-8B did it: a live L1-vs-L0 match and a registered mock, both on the GPU. The turn
net N0 was trained on situations from v2's ranges and is untested on F1's; the live match tests it.

Which F1:
- If BP-8B adopts v2-8B, the live comparator is v2-8B. Stage B (F1-8B) is then judged first by the same rule
  against v2-8B (an extension of FINDING, whose rule names v2-2B):
  - `f1_eval.py decide --base v2-8B --fix runs/f1-stageB/blueprint.bin`. G0 does not apply; G3 and G5 do;
    `bp` stays v2-2B's root.
  - `f1_eval.py pr --fix-label F1-8B --base-label v2-8B`.
  - The scorecard on v2-8B and on F1-8B.
- Otherwise F1-2B is the candidate.

F1 and the Slumbot rematch do not wait for each other unless the user decides so (`records/slumbot-2/
PREREG-DRAFT.md`, B2).

## Order
1. One read-only audit of the engine change, the trainer change, `f1_eval.py` and this file; fix the findings;
   commit.
2. Build the engine at that commit into `runs/f1-engine/`, and record its sha256.
3. Stage A training. Stage B may follow.
4. Re-verify the rating on the F1 engine, then the decision arms, (c) and (d). All CPU.
5. RESULT-F1.md; one read-only audit; commit; report to the user.
