# K10's pre-gate under Q4: the run, fixed before its data

Written 2026-10-11 14:17 SGT, before any of the run's numbers. The gate itself (its inputs, the ceiling `C`,
the rule `C >= 1.05`) is KERNELS.md K10 "Pre-gate, before any code", registered 6 Oct 03:50 SGT. This note fixes
only how it is run now that Q4 is the live config (11 Oct). Nothing here changes the bot.

## What runs (`pregate.py`, this checkout's `src`, one process)
- **Code under test:** `main` at the commit that adds this note (Q4 live: float32 solvers with `terminals64`).
- **GPU guard:** rule 3's 4 Oct amendment (`--max-busy 60`; no other process of this Python on the GPU),
  checked before and after; shared at the end = VOID, rerun.
- **P1 (descriptive):** a live solve's stages under Q4 on the 53 recorded jobs, at their own iterations and at
  400 (`kernel_bench.profile_stages`, now built with `T64`).
- **P2 (the gate's inputs):** leave-one-out on replayed graphs (KERNELS Step 1's method amendment) under Q4 on
  P2's four picks (smallest, median, largest turn job, median river job); the median turn job again with
  float32 terminals.
- **Code pinned:** `pregate.py` refuses uncommitted changes to `src`, `scripts` or its own files and records
  the commit in its output.
- **`g`:** `kernel_bench.launch_cost`: replayed graphs of 200 and of 20 launches of a one-program Triton kernel
  with an empty body, 200 replays x 7 alternating rounds, medians; `g = (t200 - t20) / 180`. Descriptive beside
  it, deciding nothing: the slope from 200 to 1,000 launches (were the 20-launch graph bound by the host's
  replay submission, `g` would read low and this slope higher), and the intercept `t20 - 20 g` beside P2's
  prep-only graph (both per-replay overhead). A low `g` makes `C` more generous, as the gate intends; the probe
  is the real test.
- **The job:** the median turn job, 9,609 rows (as on 4 Oct). K10w's launches per traversal for it are counted
  from its tree by `waves.py`: **105 against 21 today** (the 33 LBR turn jobs give 101 / 105 / 125, min /
  median / max, today 21 on every job: the registered table, reproduced). So `D = 2 x 84 x g`.
- **The gate's terms** (each a sum over both traversals of P2's leave-one-out classes):
  - `T` = the Q4 iteration's replayed graph;
  - down + up = the `down` and `up` classes;
  - terminals = `fold_ff` + `showdown_ff`;
  - float32 terminals = `matmul` + `fold` + `showdown` of the float32-terminals profile;
  - `r`, `k`, `S`, `C` exactly as KERNELS K10.
- **Reported beside `C` (descriptive):** `C` at 1 and 2 µs a launch, and the `g` below which `C` would reach
  1.05. KERNELS' Q3 figure "1.04 at 1 µs" is 1.035 unrounded (T 6.2516, down + up 0.7561, terminals 5.4714,
  float32 terminals 0.6479); its no-go reading is unchanged.

## Readings, decided now
- **Which run counts:** every run writes its own `pregate-<UTC stamp>.json`, never overwritten, and all are
  kept. The first run that is not VOID counts. VOID: the GPU shared at the end (rule 3), or `g <= 0`; a VOID
  run is rerun and reported with the result.
- `C` is taken at the measured `g`.
- `C >= 1.05`: the pre-gate passes. The next registered step is the probe (a timing-only prototype in a scratch
  copy). It is a build: it starts only on the user's go.
- `C < 1.05`: K10 stays closed on the live config; recorded in KERNELS K10 with these numbers.
- Either way the P1 and P2 numbers are recorded as the post-Q4 profile (KERNELS Step 1), descriptive.

## Disclosed before the run
- A smoke test of `launch_cost` (20 replays x 3 rounds, CPU busy with `rating.py verify`) printed
  g = 0.84 µs while the harness was being written. It does not count and changes nothing above: the method
  and the rule were fixed on 6 Oct.
- Harness changes (`scripts/kernel_bench.py`, audited before this run): `Tap` wraps `_fold64`,
  `_showdown64`, `_fold_ff`, `_showdown_ff` (where the package has them; the selftest covers a Q4 step and a
  package without them); P1 builds its solvers with `T64`; `profile` accepts `--terminals64`; P2 gains `g`
  (`launch_cost`). One read-only audit (no blocker; its three should-fixes, which run counts, the code pin
  and `g <= 0`, are fixed above).
