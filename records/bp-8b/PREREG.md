# BP-8B: a longer blueprint (v2-2B resumed to 4B, then 8B)

Registered 18 Oct 2026, before any training of it. The user chose it after RIVER-ITERS (18 Oct ~22:10:
"Longer blueprint"). Nothing here changes the live config until "Adoption" passes.

## Known
- v2-2B (`runs/blueprint-20260927-141645/blueprint.bin`) is the 200bb v2 grid, linear MCCFR, resumed from 1B:
  2,000,023,000 iterations, 39,839,607 infosets, final mean |regret| 9.8e-4, 18,848 s at ~118k it/s on 16
  threads, ~29 GB of RAM while training.
- From 1B to 2B, blueprint against blueprint: A2 PR 1665.6 against A1 1631.9 (pool-v2; the Nash weight is all on
  A2, so this is A2 beating A1 by ~34 mbb/hand). The scorecards saw no change (strength -0.1 [-4.2, +4.1] at
  100bb, -2.7 [-9.4, +3.9] at 200bb) and LBR at 200bb rose (+65 [+8, +121]). So doubling bought head-to-head
  edge, not robustness, and it is not known to survive search. The live match below is the test that counts.
- The turn net N0 was trained on solved turn values (FLOPSEARCH D1), not on the blueprint's strategy, so it
  carries over to a new blueprint unchanged.

## Training (CPU)
- **Stage 1:** `scripts/train_blueprint.py --resume runs/blueprint-20260927-141645 --target-iters 4000000000
  --threads 12`, everything else at its default (blueprint save 15 min, checkpoint 30 min, dashboard on).
  The variant and grid come from the checkpoint. **Stage 2:** the same, resumed from Stage 1's run dir, to
  8,000,000,000. Each stage has its own run dir, so 4B stays on disk.
- 12 threads, so that a GPU match beside it (4 threads) keeps its cores. The parent shell is BelowNormal
  before the spawn. Expected ~5-6 h (stage 1) and ~10-12 h (stage 2) at 12 threads on an idle machine
  (16 threads ran ~118k it/s); more beside other work.
- Parallel MCCFR is not bit-reproducible; each blueprint is pinned by its sha256 once written.
- **VOID:** a refused resume, or a stage ending other than `completed`. An interruption resumes from the run's
  own `checkpoint.bin` (its iteration count continues); that is not a VOID.

## Rating (primary for the blueprint), pool-v2 under RATING.md
- `scripts/rating.py rate --dir runs/rating/pool-v2 --candidate <blueprint.bin> --label v2-4B` (then `v2-8B`),
  no search, snap translation, the pool's protocol (20M deals against each anchor with p_j > 0, 1M otherwise).
  Each receipt is replayed by `verify` and must be VERIFIED.
- **The statistic:** the paired change against A2, `parciver.rating.change(x = the candidate's anchor rows,
  y = A2's receipt rows, weights = the calibration's p)`: the same deals, so the reference cancels.
- **GAIN iff the 95% interval's lower end is > 0** for v2-8B. v2-4B is reported (the dose: is 8B's gain more
  than 4B's?), not gating. LBR is reported beside it, not gating; a rise is noted as V2LEAK's pattern.
- If v2-8B is not a GAIN: stop and report. The live config stays.

## Live match (only on a GAIN): L1 against L0
- **L0:** the live bot as `scripts/slumbot_run.py` builds it at that time (river at 200 or 2,000, whichever is
  live then). **L1:** the same with v2-8B in place of v2-2B. Nothing else changes: the same net, search and
  schedules, both arms `snap`.
- LIVESEARCH P2's protocol: the 200bb baseline row (seed 410,001), its first 10,000 deals, seats swapped, 4
  threads, chunks of 100 proven by `livesearch.prove`, 50 blocks of 200, AIVAT reported. The resume pins as
  RIVER R1's plus both blueprints' sha256. A smoke run first (50 deals on seed 777,005, 1 against 4 threads,
  identical per-deal results).
- **DOWN iff the interval's upper end is < 0.** Positive: reported as a gain.
- Its script is written, tested and audited (read-only) before any of its data, with an amendment here if it
  needs one.

## Adoption (only on a GAIN and a live match not DOWN)
1. `scripts/slumbot_run.py` points at v2-8B, committed with the user's yes. The bars are unchanged.
2. The registered mock (1,000 hands, LIVE-FLOP's fail and VOID rules, normal PC use, GPU-client log). A pass
   becomes GATES (`records/livesearch/mock_check.json`) with the user's yes. A fail reverts the commit.
- **ADOPT v2-8B** iff the rating is a GAIN, the live match is not DOWN and the mock passes. Anything else:
  stop and report.

## Cost and side effects
- Training ~15-18 h CPU (more beside other work); ~3.8 GB per run dir (blueprint + checkpoint), 232 GB free.
- RAM: ~29 GB while training, beside RIVER R1's P2 (2 blueprints) within 64 GB. No M1 oracle runs beside it.
- The ratings are minutes each; the live match ~20 h GPU; the mock ~40 min.

## Stage 3 (addendum, registered 19 Oct 2026 21:13, the user's yes; before any stage-3 training)
The user's yes (19 Oct): "register stage 3 and run it on cloud cores 10-18 after stage 2 ends, with the 2% D1b stop
rule". Why: the cloud droplet's CPU sits ~59% idle beside D1b (records/d1b), whose GPU work cannot use it; training
there frees the PC to rate v2-8B at the same time. Nothing above changes: v2-8B's rating, GAIN rule, live match and
adoption stand as registered.
- **Where:** the `rocm` container on the user's MI300X droplet (Xeon 8568Y+, 20 vCPU; ~216 GB RAM free), its engine
  built at 7839aa6 (no Rust change since b6a3c7e, the PC engine's source). Abstraction: `data/abstraction` copied
  file by file, sha256-checked (`stage3/push.sh`).
- **Training:** from stage 2's final `checkpoint.bin` (8,000,000,000 iterations; copied sha256-checked) with
  `train_blueprint.py --resume <its dir> --run-dir /root/bp-s3/run --abstraction-dir /root/bp-s3/abstraction
  --target-iters 16000000000 --threads 9 --max-seconds S --no-dashboard`, under `taskset -c 10-18` and nice 10;
  S ends it ~20:30 Tue 20 Oct. Just before launch, D1b's running processes are re-pinned to cores 0-9
  (`taskset -a -p`; no restart).
- **The end is the time box.** The trainer prints "interrupted" for any stop short of the target; here that is the
  registered end, not a VOID. **VOID:** a refused resume; a crash with no checkpoint after it (one with a checkpoint
  resumes once from its own, as stages 1-2); the blueprint not on the PC with a matching sha256 by 21:30 Tue.
- **The result:** `runs/bp-8b-stage3/` (blueprint.bin + checkpoint.bin, sha256 both sides), label **v2-S3**, its
  iteration count recorded. Parallel MCCFR is not bit-reproducible; it is pinned by its sha256.
- **D1b first (the stop rule).** Baseline: D1b's mean shard time before the re-pin, per worker between consecutive
  shards (19 Oct 21:10: 1,129.9 s, sd 3.6, n = 56; 3.54 solves/s). A watcher in the container takes each block of
  8 shard times whose shards started after the launch; if a block's mean exceeds baseline / 0.98 (D1b's rate down
  more than 2%), it stops the training (SIGTERM; the trainer saves and exits) and logs it. D1b is never touched;
  stage 3 is not restarted after such a stop (its last save is the result).
- **Rating:** v2-S3 is rated as v2-4B was (pool-v2, the paired change vs A2), after v2-8B's rating, and reported.
  It enters the live match or adoption only by a later amendment with the user's yes.

## Amendment 1 (20 Oct 2026, before any of the live match's data): its script
The user's yes (20 Oct, after v2-8B's GAIN): "write, test and audit the BP-8B L1-vs-L0 live match script ... Run
nothing on its data yet." Nothing above changes; this fills in what "Live match" left to the script.
- **Script:** `scripts/bp8b_live.py` (`smoke`, `match`), tests `tests/test_bp8b_live.py`. Each arm is LIVESEARCH's
  `Arm` with the live runner's search, turn schedule, net, leaves and flop gamma; L0 loads the runner's blueprint,
  L1 v2-8B (refused unless its sha256 is fd27fe50...c7ae071, and refused if the runner already plays it). L1 is the
  engine's A: the statistic is L1 - L0 in bb/100 over 50 blocks of 200 deals.
- **Pins:** RIVER R1's (the benchmark row, deals, chunk, threads, blocks, net, leaves, gamma, terminals64, turn
  schedule, search, engine build, play code = livesearch's PLAY_CODE + this script, torch, triton) plus both
  blueprints' path and sha256, hashed again once loaded in every session (a replaced file is a failed chunk); both
  must have the same grid (the engine builds the match's tree from L1's). A smoke run (`runs/bp-8b-live/smoke.json`) on the same play
  code, engine and blueprints comes first; 4 threads need it identical to 1 thread deal for deal, else the match
  runs at 1 thread.
- **A failed chunk** (a proof, a failed search, a CUDA or any other error) is recorded in `runs/bp-8b-live/state.json`
  and stops the match; a rerun refuses it, as RIVER R1's P2. Going on needs an amendment with the user's yes. A
  Ctrl-C or a reboot is not a failure: the match resumes from its last saved chunk.
- **Record:** `records/bp-8b/live.json`, written once, with any `failures` and `code_history` an amendment adds to
  the state by hand (as RIVER R1's `cuda_failures`). Latency in the GPU lock is reported (chunks with another GPU
  job of this Python, or nvidia-smi unknown, are left out), not gating: the latency bars are the mock's (Adoption 2).
