# RIVER R1: the live river at 2,000 iterations

Registered 18 Oct 2026, before any R1 data. It follows from the RIVER-ITERS scout (`records/river-iters/
RESULT.md`, PROCEED). The user chose 2,000 iterations (18 Oct ~22:10: "Yes, river 2000"). This is LIVESEARCH
B1's protocol (LIVESEARCH.md Amendment 3, Stage 2) with the river in place of the turn.

## Known
- On B1's own 332 river jobs (P2's deals 0-499), the live river (200 iterations, DCFR gamma 2, Q4) stops 1.285%
  of the pot [1.095, 1.495] from equilibrium. At 2,000 iterations it is 0.050% [0.042, 0.059]. A solve takes
  ~0.16 s against ~0.023 s (solver construction included).
- The live mock (18 Oct 20:38, `records/livesearch/mock_check.json`) had river decisions at p95 0.035 s and
  0.35 river solves a hand. Mocks play one hand at a time, so a river solve never waits for another decision's
  solve; at 2,000 iterations its p95 is expected near 0.2-0.3 s, against the 2 s bar. In LIVESEARCH P2 (4
  threads, solves queued on the GPU lock) the river ran p95 0.072 s, max 0.43 s at 200; scaled, ~0.5 s and
  multi-second maxima at 2,000 (descriptive there).

## The arms
- **R0:** the live bot as `scripts/slumbot_run.py` builds it at this commit: v2-2B, `snap`; flop 500 iterations
  at gamma 5 on N0 with sym leaves; turn 400 at `TURN_SCHEDULE` (gamma 5, shift 1); river 200 at the default
  (DCFR gamma 2); Q4 terminals; `rescue=False`.
- **R1:** the same with `river_iters` 2,000. Nothing else changes. Each arm has its own solver objects, built as
  LIVESEARCH's `Arm`. The GPU lock serialises solves.

## P2 (primary, strength): R1 against R0
- **The deals:** M2's protocol, as LIVESEARCH P2: the 200bb scorecard's `baseline` row (seed 410,001), its first
  10,000 deals, 2 hands each with the seats swapped. bb/100 from R1's side. 4 threads. Chunks of 100 deals,
  saved atomically. 50 blocks of 200 deals, mean and 95% t interval (`scorecard.summarize`).
- **P2 fails (DOWN) iff the interval's upper end is < 0.** A positive interval is reported as a gain.
  AIVAT is on and reported, not gating.
- **Each chunk proves its arms before it is saved** (`livesearch.prove`). Every job as the engine sent it is at
  the arm's iterations on its street: flop 500, turn 400 and river 2,000 (R1) or 200 (R0). Every turn solve ran
  `TURN_SCHEDULE` and every river solve the default, as the solver objects report it. Each arm made at least one
  flop, one turn and (added here: R1 and R0 differ only on the river) one river solve, and no solve failed. A
  failed chunk stops the run before it is saved; the failure is written into the state, and a rerun refuses to
  go on (reported, not patched or retried). A crash, kill or reboot is not a failed chunk: P2 resumes.
- **Resume:** only on the same configuration. That means the play code (git objects of `src`, `engine`,
  `scripts/livesearch.py`, `scripts/slumbot_run.py`, `scripts/river_iters.py` and `scripts/river_live.py`, none changed and uncommitted),
  the engine build, the blueprint's and the net's sha256, the benchmark's manifest hash, both arms' settings,
  the threads, torch and triton. The commit is recorded per session but not compared.
- **Before P2, a smoke run:** 50 deals on seed 777,004 (unused), at 1 and at 4 threads, each run's arms proven,
  fresh arms for each, each arm with at least one river solve. The per-deal results, raw and AIVAT, must be
  identical. If they differ, P2 runs at 1 thread. P2 refuses to start without a smoke run on the same play code
  and engine, and at 4 threads without an identical one.
- **Descriptive:** each arm's river, turn and flop in-lock solve seconds (median, p95), solves a deal and
  s/deal. Chunks with the GPU shared (another process of this Python, or unknown) and the chunks that copy P1's
  jobs are left out of the latency figures.
- `runs/river-live/p2-state.json` holds the state. At the end, `records/river-live/p2.json` is written once.

## P1 (primary, precision), on fresh live decisions
- **The jobs:** every river job (5-card board) R0's solver receives in P2's deals **500-799** (chunks 5-7). These
  are not the scout's deals (0-499). Each is copied before its solve, as the engine sent it, and saved with its
  chunk (`runs/river-live/p1-jobs-<first>.pkl`, sha256 in the chunk's record).
- **The solves:** each job is solved twice from scratch, as R0 (200 iterations) and as R1 (2,000), both DCFR
  gamma 2, float32, Q4 terminals, frozen slots held, every slot summed. The measure is exploitability in its own
  subgame, % of the pot (`livesearch.pot_of`), as LIVESEARCH P1 and the scout measure it.
- **The statistic:** D_j = expl_R1(j) - expl_R0(j). Mean, with a 95% percentile bootstrap: 10,000 resamples of
  river boards (the flop as a set, the turn, the river) with their jobs, `default_rng(0)`. Rows are sorted by
  board and content hash.
- **P1 holds iff the interval's upper end is < 0.** The scout makes this near-certain; P1 is the registered
  check on fresh decisions. The GPU must hold no other job of this Python at P1's start and end, or P1 is VOID
  and reruns. P1 runs on P2's play code (all of it), torch and triton.

## Adoption (only if P1 holds and P2 is not DOWN)
1. `scripts/slumbot_run.py`: `SEARCH` gets `river_iters` 2,000, committed with the user's yes. The bars are
   unchanged (flop p95 <= 3.0 s, turn and river p95 <= 2 s each).
2. The registered mock (`slumbot_run.py --mock`, 1,000 hands) uses LIVE-FLOP's fail and VOID rules. The PC is
   in normal use, and the GPU-client log is reported. A pass becomes GATES (`records/livesearch/mock_check.json`)
   with the user's yes.
3. A fail reverts the commit. Stop and report: the live config stays. A fail on the flop bar alone (R1 does not
   touch the flop) goes to the user, as in Amendment 4.
- **ADOPT R1** iff P1 holds, P2 is not DOWN and the mock meets its bars. Anything else: stop and report.
- **The rating:** the live receipt (PR 1860.4) is B0's and already STALE. Rating R1 is the user's call.

## Code (`scripts/river_live.py`: `smoke`, `p2`, `p1`; reuses `scripts/livesearch.py` and `scripts/river_iters.py`)
- It changes no file under `src/`, `engine/` or the live runner until adoption.
- Tests before any data (`tests/test_river_live.py`): R1's arm differs from R0's only in `river_iters`. The
  proof fails a river job at the wrong iterations. The tap keeps R0's river jobs only, and only in chunks 5-7;
  both arms get the river tap and the live one is restored. A chunk without a river solve per arm fails. P2
  needs a smoke run on its code. The resume refuses a changed configuration. The P1 statistic on toy data.
- A read-only audit before any data, its findings verified and folded in (18 Oct: no bug; folded in: the river
  solve per arm, the failed chunk recorded, the smoke run before P2 at 1 thread too, P1 on all of P2's play
  code, the cost and latency figures corrected).

## Cost (estimates) and VOID
- Smoke: minutes. P2: ~8 s/deal (LIVESEARCH P2 7.3, whose B0 ran the turn at 100 where R0 runs 400; plus ~0.7
  river solves per arm a deal x ~0.14 s for R1), ~22 h GPU. P1: ~205 jobs (0.69 a deal x 300) x ~0.2 s, minutes. The mock: about 40 min.
- During P2 the GPU holds no other job of this Python, and no M1 oracle runs. A CPU blueprint job may run beside
  it at BelowNormal (RAM: 64 GB, ~37 GB free). bb/100 is deterministic per deal; only the descriptive latency
  can be affected, and the chunks it touches are reported.
- **VOID:** a crash, kill or reboot of the smoke run, P1 or the mock, or another GPU job of this Python beside
  them, voids that step. Its record is moved aside as `*-void-<n>` and the step reruns. P2 resumes by chunks.

## Amendment 1 (19 Oct 2026 ~04:10, the user's yes; P2 at 2,300 of 10,000 deals, no P1 or mock data)
- **What happened:** chunk 2300 (deals 2300-2399) failed its proof at 03:50, and P2 stopped as registered.
  Two of R0's 333 search solves (a flop and a river job; R0 is the live arm, not the changed one) raised
  `AcceleratorError: ... operation failed due to a previous error during capture`
  (`cudaErrorStreamCaptureInvalidated`), and 3 of R0's decisions fell back. The System log shows no driver
  reset (nvlddmkm) near it. It is the first such error in LIVESEARCH's and this harness's matches (~13,000
  deals). It is read as a rare race between match threads around CUDA-graph capture; the live runner and its
  mock play one hand at a time. Not diagnosed further here.
- **The rule from now:** a chunk whose proof fails only through solves that raised a CUDA error (the solve and
  job counts differ by exactly the failed solves) is rerun once, on the same code. Play is deterministic per
  deal (the smoke run), so a clean rerun gives those deals' results as a clean first run would. Each failure is
  kept in the state (`cuda_failures`) and in `p2.json`. A second failure of the same chunk, a third CUDA failure
  in P2 (this one counts), or any other proof failure stops P2 for good: reported, not patched.
- **How:** no code changes (the resume pins hold). The state's `failed` entry is moved by hand into
  `cuda_failures` with a note naming this amendment, then `river_live.py p2` resumes at deal 2,300.
- **Seen before this amendment (descriptive, not a verdict):** R1 - R0 +22.08 bb/100, standard error 9.63, over
  deals 0-2,299. Nothing above depends on it.

## Amendment 2 (19 Oct 2026 ~07:30, the user's yes; written at 3,800 of 10,000 P2 deals, no P1 or mock data)
- **What happened:** a second CUDA failure (chunk 3700, 07:14: one river solve of R1 and one flop solve of
  R0), rerun under Amendment 1 (passed, 07:34). Both failures were preceded by one torch warning,
  `CUDAGraph.cpp:404 ... operation not permitted when stream is capturing (function reset)`: a CUDA graph was
  reset while another thread captured one. Each time one reset and two failed solves: the second is read as a
  cascade of the first in the same process.
- **The cause:** `FlopSearch.__call__` (`src/parciver/flop_search.py`) left its `FlopSolver`, which holds the
  solve's CUDA graph, to be freed when the call returned, outside `GPU_LOCK`. Freeing it resets the graph (torch
  captures in global mode), and a reset while another match thread captures inside the lock invalidates that
  capture. `SubgameGpu` (turn and river) already freed its solver inside the lock. Multi-threaded matches only:
  the live runner and its mock play one hand at a time.
- **The fix:** the flop solver (and the leaf tables it used) are freed inside the lock. On a failed solve, in
  both `FlopSearch` and `SubgameGpu`, the solver is dropped and the traceback's frames are cleared
  (`gpu_fused.drop_frames`) inside the lock, so an error cannot carry a graph out of it. It changes when memory
  is freed, not any value computed. Tests (GPU): `test_the_flop_solver_is_freed_inside_the_gpu_lock` and
  `test_the_subgame_solver_is_freed_inside_the_gpu_lock`, each for a clean and a failing solve: the solver and
  its graph are freed before the outer lock is released (the flop one fails on the old code). Audited
  read-only before any data on it (no bug; its findings folded in). Not changed: torch leaves its capture
  stream current after a failed `capture_end` (it lasts only until that P2 process exits on the failure).
- **Shown to change no play:** with P2 stopped (a stop mid-chunk loses only that unsaved chunk, replayed
  deterministically) and the GPU otherwise ours, the GPU tests above pass, then the smoke run on the new code
  (seed 777,004, 50 deals, 1 and 4 threads, fresh arms). A script checks that both of its runs' per-deal results,
  raw and AIVAT, equal the smoke run at 59a3b7a (kept as `runs/river-live/smoke-59a3b7a.json`; its sha256 is
  P2's `smoke_sha256`), that it is itself identical across threads, and reports the solve counts. If anything
  differs, P2 does not continue: stop and report.
- **Then P2 continues** from its last saved chunk on the new code. The state's pinned play code is updated by
  hand to the new commit's. The old pins go into the state's `code_history` with the deals played under them
  and the new smoke run's sha256. `p2.json` now also carries `cuda_failures` and `code_history`
  (`river_live.py`'s `p2_finish`, in this commit; Amendment 1's claim holds through it). Every other pin is
  unchanged. The CUDA failures stand; the limit on the new code is Amendment 3's. A
  reappearance of the reset warning in `p2.log` is reported.
- **Adoption:** unchanged. The mock it needs runs on the new play code. LIVESEARCH's records are not touched.

## Amendment 3 (19 Oct 2026 ~11:40, the user's yes; P2 stopped at 5,500 of 10,000 deals, no P1 or mock data)
- **What happened:** Amendment 2 was written but not yet committed, and P2 kept running on 59a3b7a's code.
  At 11:20 chunk 5500 failed the same way: one reset warning (`CUDAGraph.cpp:404 ... (function reset)`),
  then one river solve of each arm raised `cudaErrorStreamCaptureInvalidated`. That is P2's third CUDA failure,
  so P2 stopped for good under Amendment 1, as registered.
- **Before this amendment:** the GPU tests of Amendment 2 passed on the new code, with the GPU otherwise
  ours (57 passed in `tests/test_flop_search.py` and `tests/test_gpu_fused.py`). The new flop test was also run
  against 59a3b7a's `flop_search.py`, and both of its cases failed there.
- **The rule from now:** Amendment 1's stop is lifted for this cause only, which is diagnosed and fixed. P2
  continues at deal 5,500 under Amendment 2's procedure: the smoke run on the new code must equal the old one
  deal for deal, then the pins move by hand. The third failure is kept in `cuda_failures` and in p2.json, with a
  note naming this amendment. **Any CUDA failure on the new code stops P2 for good** (no rerun), as does any
  other proof failure. Nothing else changes: the deals, the decision rule (DOWN iff hi < 0), P1 and adoption.
- **Why the results cannot be selected by this:** play is deterministic per deal (the smoke run). A failed
  chunk saves nothing, and its rerun gives the results a clean first run would have given. Amendment 2 changes
  only when memory is freed, and its smoke check must show that.
- **Seen before this amendment (descriptive, not a verdict):** R1 - R0 +18.62 bb/100 (standard error 6.11) over
  deals 0-5,499. Nothing above depends on it.

