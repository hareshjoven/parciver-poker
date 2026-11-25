# D1b: more turn-start data, made on the cloud MI300X

Drafted 19 Oct 2026 12:20 SGT, before any D1b data; revised 12:45 SGT with Step 0's results (no D1b data
yet). **Status: REGISTERED 19 Oct 12:51 SGT** (the user's yes, "okay go ahead", to this file and to Amendment 1
with Q4, after the two options were laid out; before any D1b data). Recording may start now; generation only
after the speed test is a go. This step makes data only: how D1b is
used (a D2-style retrain on D1 + D1b, and its gates) is registered separately, before any net is trained on it.

## Known
- D1 (FLOPSEARCH): 250,002 turn starts recorded from v2-2B's own play (18 cells: 200bb and 100bb x 9
  opponents, 13,889 each; match seeds 520000 + 1000 x cell + chunk) plus a DeepStack random twin each =
  500,004 solves, 500 iterations, gamma 5, float32 terminals, ~119 h on the PC at 1.17 solves/s. Its tail
  (K2: float32 O(n) terminals, the over-10% rows) was patched by PREREG15 (float64-precise terminals;
  `records/d1-tail/`).
- D2's learning curve: selection loss 0.02109 / 0.01711 / 0.01464 at 25 / 50 / 100% of D1; a fit gives
  ~-10% at 2x data (an estimate). Labels are not the bottleneck (500-iteration labels are 0.23% of the pot
  from 3,000-iteration references in value MAE).
- The cloud: the user's AMD MI300X (ROCm 7.2.3, torch 2.11 HIP, Triton 3.6), until **22:50 SGT Tue 20 Oct**
  (hard stop). For the speed test and the run, the user stops every local model server.

## Step 0, the probe (rule fixed in `scripts/d1b_probe.py` sha256 710f586f... before any cloud data)
Done 19 Oct 11:55-12:40 SGT, with vLLM still on the GPU (24 GiB free):
- **Setup:** code 4eacf33 as a git bundle plus the uncommitted generator; a separate venv on the system torch;
  rustup + maturin; the engine built. PASS.
- **Tests:** `test_gpu_fused` 44, `test_gpu_search` 9, `test_turn_values` 7, `test_turn_data` 3,
  `test_kernel_designs` 10, `test_flop_search` 13: **86 passed, 0 skipped**. Rust: **157 passed, 0 failed**
  (16 ignored, as on the PC). PASS.
- **Parity, D1's mode** (D1's first 150 situations, both sources, 300 jobs; G1-G4): **PASS.** G1-G4 all
  true; 286 ordinary jobs, median e_c/e_s 0.9987; value move d median 0.027% / p95 0.153% of the pot;
  mean e 0.349% (cloud) vs 0.360% (stored); repeats 20/20 bit-identical.
- **Parity, Q4** (`--terminals64`): **FAIL as written**, on G4 only (G1-G3 pass): 18 ordinary jobs have
  e_c > e_s + 0.05%; median e_c/e_s 1.0007; mean e 0.351%.
  - Diagnosis (run after the failure): **the PC's own Q4 run fails the same bar** (12 jobs; 5 shared with the
    cloud's 18; mean e 0.371%). Cloud Q4 vs PC Q4: median |dE| 0.0062%, p95 0.073%, median ratio 1.0026.
    500-iteration float32 solves are chaotic job by job (job 17: stored 0.167%, PC Q4 0.995%, cloud Q4
    0.313%), so a per-job bar against D1's float32 labels tests Q4 against float32, not the cloud against
    the PC. The bar was wrong; the MI300X is not at fault.
- **End to end** (6 stored situations): 2 workers' shards = 1 worker's bit for bit; `--terminals64` shards =
  the probe's Q4 solves bit for bit; generate.json carries `terminals: q4` / `workers: 2`. PASS (rerun on the
  audited code at 12:39 SGT: PASS).

**Amendment 1 (Q4's G4; APPROVED 12:51 SGT by the user; written after seeing both Q4 runs, so it is a
post-hoc rule and is marked as such).** Q4's G4 compares the cloud's Q4 with the PC's Q4 on the same 286
ordinary jobs (`records/d1b/step0/parity-pc-q4.json`, 12:35 SGT), as distributions: the median e_cloud/e_pc
lies in [0.8, 1.25] and mean e_cloud <= mean e_pc + 0.02% of the pot. On the data in hand: 1.0026 and
0.351 vs 0.371, a pass. A4 below re-checks the same way on 200 fresh D1b jobs. *The alternative* (declined): generate in D1's float32 mode, which passed every bar as registered, and patch D1b's
over-10% tail on the PC afterwards with PREREG15's recipe (~4x a row's cost; D1's 1,920 rows took ~2 h).

**Speed** (still to do, with the GPU free): `d1b_probe.py speed` at 1, 2, 4, 8 processes, 2 min each, in
D1's mode then in Q4. Every process solves once (Triton compile), then all start counting together at a
barrier; the rate is the sum over processes. (A 15 s smoke at 2 processes with vLLM still loaded: 3.96
solves/s; not a measurement.) **Go iff** the best D1-mode rate is >= 3.51 solves/s (3x the PC's 1.17).

## Record (on the PC, CPU)
- `scripts/turn_data.py --out runs/turn-data-d1b record --situations 250002 --seed 600000 --threads 8
  --exclude runs/turn-data/situations.npz`, BelowNormal (beside BP-8B stage 2 and RIVER R1 P2). D1's
  recipe unchanged: the same 18 cells, the same agent (v2-2B with the recording solver), the same per-cell
  count; only the seed differs. Match seeds 600000 + 1000 x cell + chunk lie in [600000, 618000), clear of
  D1's [520000, 538000) and the benchmarks' 410xxx; `record` refuses an `--exclude` run whose range touches
  D1's.
- Situations already in D1 (same board, half, stack, first and float32 ranges) are dropped and counted per
  cell (`excluded_hits`). A cell records more chunks until it has its full count of fresh situations
  (chunk 999 at most), so no cell comes out short.
- Upload: situations.npz (~2.3 GB, as D1's) to the cloud by scp, sha256-checked (~9 min at ~4.2 MB/s).
- Known limit, as D1: the ranges are the blueprint's flop play, not the live flop search's.

## Generate (on the cloud)
- `turn_data.py --out <dir> generate --iters 500 --gamma 5 --samples 500004 --seed 600007 --terminals64
  --workers W --worker i` for i = 0..W-1, one tmux window each, **worker 0 first** (it writes generate.json;
  the others refuse to start without it). Worker i takes shards k with k mod W = i. D1's settings except
  the terminals: Q4 (float32 solves, fold and showdown sums in pairs of float32; the live solver's
  terminals since 11 Oct), so D1's K2 tail is not made again.
- **W** = the smallest process count whose Q4 rate is >= 95% of the best Q4 rate measured in Step 0.
- `workers` is in generate.json and clashes on resume: a stopped run resumes with the same W, each worker
  with its own command (complete shards are kept by the resume rule). Only worker 0's resumes are listed
  in `resumed`.
- Shards are copied to the PC (`runs/turn-data-d1b/`) at least hourly, sha256-checked on both sides.
- **Stop:** when all 501 shards exist, or at **21:30 SGT 20 Oct**, whichever is first; then the last copy
  and its check by 22:30. Each shard mixes every cell, so any set of complete shards is a representative
  sample; partial shards are not used.

**Amendment 2 (the PC finishes the run; the user's decision, 20 Oct ~21:46 SGT, "let's complete d1B now";
written 21:53 SGT after the cloud's stop and before any PC shard).** The cloud stopped at the 21:30 rule with
372 complete shards (shard-00000 to -00371), all copied and sha256-checked (last pass 21:37:32 SGT,
`runs/turn-data-d1b/copy.log`). The other 129 (372-500: 33 / 32 / 32 / 32 for workers 0-3) are made on the
PC's RTX 4070 instead of being dropped:
- Exactly the registered command (`turn_data.py --out runs/turn-data-d1b generate --iters 500 --gamma 5
  --samples 500004 --seed 600007 --terminals64 --workers 4 --worker i`, i = 0..3, worker 0 first; `samples`
  is not in the clash check, so it is copied, not retyped) into the same directory, so the resume rule keeps
  the cloud's shards and a settings clash refuses to start; `code` stays 7839aa6 and the PC start is listed
  in `resumed` (an added list, not a setting, for A1).
  The generator's code (`scripts/turn_data.py`, `parciver.turn_values`, the GPU kernels, the installed engine
  .pyd E58C7292) is unchanged since 7839aa6; later commits change only the Rust trainer sources (not in the
  .pyd), `bp8b_live.py`, `slumbot_run.py`, tests and docs.
- A shard's jobs depend only on its number and the seed, not on the worker or the machine; the PC's Q4
  solves differ from the cloud's job by job only as Amendment 1 measured (median ratio 1.0026).
- **Stop:** all 501 shards (no clock). Nothing else uses the PC GPU meanwhile; CPU jobs may run beside it at
  BelowNormal with the GPU hidden.
- Acceptance as registered, over all 501 shards; A1-A3 are also reported separately for the cloud part
  (0-371) and the PC part (372-500). A4 draws from the copied (cloud) shards, as written: its 200 jobs are
  uniform over the 372,000 rows of shards 0-371 (numpy seed 2026); re-solving PC shards on the PC would
  check nothing.

## Acceptance (on the PC, before any use; all must hold, else D1b is not used)
- **A1 layout:** each shard has 1,000 rows (the last may be short by the rule), indexes and sources as the
  generator writes them, `generate.json` = the settings above.
- **A2 finite and zero-sum:** every value finite; with each range scaled to mass 1,
  |r0 . v0 + r1 . v1| <= 1e-5 pots per solve.
- **A3 quality:** mean exploitability <= 0.5% of the pot (D1's bar); the share over 3% reported beside D1's.
- **A4 PC re-verification:** 200 jobs drawn from the copied shards (numpy seed 2026, uniform over rows) are
  re-solved on the PC in the same mode (once the PC GPU is free of P2/P1): G2 and G3 on every job, and the
  median of e_pc / e_cloud in [0.8, 1.25].

## Reported, not gating
Rates per worker; GPU memory; the excluded-hit counts; D1b's exploitability distribution beside D1's; any
worker restarts. Results go in `records/d1b/RESULT.md` with true times (`date`), and the probe's JSON files
(`records/d1b/step0/`).

## Code (to be committed before the run, with the user's yes; read-only audit before the run)
- `scripts/turn_data.py`: `generate --terminals64 --worker --workers` (`terminals` and `workers` in
  generate.json, both clash on resume; only worker 0 writes the file; others need it first), `record
  --exclude` (with the top-up, the seed guard, and a refusal if a cell still ends short at the chunk cap).
  generate.json is written whole and renamed; the prep pool is `spawn` on every platform (D1's, on Windows). `turn_values.solve(..., terminals64=)`. Defaults
  unchanged, so D1's path is the same code. `tests/test_turn_data.py`: 3 new tests (6 pass).
- `scripts/d1b_probe.py`: extract / parity / speed. Since the parity rule was fixed (710f586f), only
  `speed` changed (the common-start barrier; timeouts so a dead process ends the test instead of hanging it).
  Audited read-only twice (12:30, 12:55 SGT); every real finding fixed.
- The cloud runs the committed code: a fresh bundle of that commit, its hash checked on both sides.
  (The working tree is CRLF and the repo LF, so file hashes are compared on the same checkout.)
- `records/d1b/step0/e2e.sh`: the end-to-end check (a script beside the probe, copied into the record).
