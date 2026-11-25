# D1b — result

Registered: `PREREG.md` (19 Oct 2026 12:51 with Amendment 1, Q4; Amendment 2, the PC finishes the run,
20 Oct 21:53, commit 3e272db). Data: `runs/turn-data-d1b/` (situations.npz sha256 f5dc4be3...; 501 shards). Acceptance: `scripts/d1b_accept.py` (`check` = A1-A3, `verify` = A4).

## Generation
- **Cloud (MI300X, 4 Q4 workers, code 7839aa6):** 19 Oct 16:35:51 (first shard) to 20 Oct 21:27:56
  (last), stopped by the 21:30 rule: **372 complete shards** (0-371; 93 per worker, no restarts), ~0.89
  solves/s per worker (~3.55 in all), GPU memory 0.42-0.44 GiB per worker. All 372 copied and sha256-checked
  (last pass 21:37:32, `copy.log`); the cloud's logs are `w0-w3.out` and `generate.log` (cloud times in UTC).
- **PC (RTX 4070, Amendment 2):** the registered command via `records/d1b/pc_gen.ps1`, started 20 Oct 21:53:55
  (worker 0) and 21:54:03 (workers 1-3) at 3e272db (`resumed` in generate.json); logs `pc-w0..3.out`.
  Shards 372-500 (129). First shards 23:22-23:24: 0.18-0.19 solves/s per worker (~0.73 in all; the cloud
  ran ~3.55), GPU 100%, ETA 45.6-48.4 h, so the end is about Thu 22 Oct ~21:00-23:30. **End: all 501 shards, last line 22 Oct
  21:58:13** (workers' last lines 21:15:05-21:58:13); one `PC start` per worker, no restarts, no errors.
- Then `records/d1b/gpu_queue.ps1` (log `runs/gpu-queue.log`): when no generator is left (relaunching
  pc_gen.ps1 at most twice if shards are missing), A1-A3, A4, then BP-8B's smoke run again (the match pins the
  play code's git objects, and TRAINER-PERF changed the engine's source tree after the 20 Oct smoke; the built
  .pyd is the same E58C7292) and, if it passes, the match (L1 = v2-8B). Audited read-only (3 real findings
  fixed: the smoke pin, a dirty-tree refusal on relaunch, acceptance starting beside a live worker).

## Acceptance
- **Dry run of A1-A3 on the cloud's 372 shards (20 Oct 21:56, not of record):** layout, settings and
  source-0 ranges (= situations.npz bit for bit) all as registered; every value finite; zero-sum error max
  9.4e-8 pots (bar 1e-5); mean exploitability **0.326%** of the pot (bar 0.5%), median 0.184%, p99 2.42%;
  over 3%: **0.28%** of rows (D1 as stored in its shards: 0.30%, mean 0.332%); over 10%: 0.003% (D1 0.002%).
- **A1-A3 over all 501 shards: PASS** (`accept-a1a3.json`, 22 Oct 21:58:44-21:59:23). 501 of 501 shards,
  500,004 rows, layout and settings as registered (no clash); every value finite; zero-sum max 9.4e-8 pots (bar
  1e-5, 0 rows over); **mean exploitability 0.327%** of the pot (bar 0.5%; D1 0.332%), median 0.184%, p99 2.43%;
  over 3%: 0.286% of rows (D1 0.304%); over 10%: 0.0038% (D1 0.0018%).
  - Parts (reported): cloud 372,000 rows, mean 0.326%, over 3% 0.284%; PC 128,004 rows, mean 0.329%, over 3%
    0.293%, zero-sum max 5.6e-8.
- **A4 (200 cloud rows, numpy seed 2026, re-solved on the PC): PASS** (`accept-a4.json`, 22 Oct 21:59:23-22:04:16
 , RTX 4070, torch 2.14.0+cu132, 285 s). G2 and G3 on all 200 jobs, every rebuilt input equal to the stored
  one; **median e_pc / e_cloud 1.001** (bar [0.8, 1.25]). Reported: mean e_cloud 0.358%, e_pc 0.343%. The draw and the job
  rebuild were checked on the CPU (6 of 6 rows rebuild the stored ranges bit for bit, both sources). The
  script also fails A4 if a rebuilt job's ranges differ from the stored ones (G1 of the probe; stricter than
  A4 as written, since a different input would make G3 meaningless).
- **Verdict: D1b ACCEPTED** (A1-A4 all hold); its use (a net trained on it) needs its own gate.
