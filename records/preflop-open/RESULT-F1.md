# F1: result

Pre-registration: `F1-PREREG.md` (commit d188bd0). Measured 21 Oct 2026 on the CPU, all runs valid (no VOID).

**Verdict: NOT ADOPTED.** FINDING's rule fails on (a), (a′), (c) and (d); only (b) passes. Against LBR's 2bb and
2.5bb opens, F1-2B is *more* exploitable than v2-2B, not less (+34.5 and +21.2 bb/100), it loses to v2-2B head to
head (PR -4.8 mbb/hand, DOWN), and it does worse against `min_raiser` (-62 bb/100, DOWN). v2 stays the grid; stage
B is not trained.

## The engine
- Built by `f1_build.py` at d188bd0 (21 Oct 2026, 00:04): `runs/f1-engine/BUILD.json`.
  - .pyd sha256 `906227B41D658E880EC7F4CBEF595F8E9661599AE2C10E802965EBF2D33089C7` (identity
    `906227B41D658E88+5AD0B211E9C7D830`).
  - Wheel sha256 `A8A69C4C8317A6C0B0A778B5E1B35510FAEEB4A12946AD3909A0A8F272E1B974`; Cargo.lock sha256
    `9A8F0F5A0B9E2A20FC94F0207EB056BBB15237D750B2E8CF7345A80C3E21254D`.
- The installed engine is unchanged (E58C7292…, checked before and after the build).

## Stage A
- Started 21 Oct 2026, 00:04:37, by `launch_f1.ps1`: interpreter PID 19312 at BelowNormal, GPU hidden, log
  `runs/f1-stageA.log`. First line: 83,538 it/s solving at 1.9M iterations (30.4M infosets).
- Beside D1b: GPU at 100% in all 20 samples, 3 s apart (00:04:44-00:05:44; read by hand, not logged).
  `f1_watch_d1b.py` (log `runs/f1-watch/watch.log`, from 00:10) applies the stop rule until the trainer ends:
  each worker's shard time (consecutive shard lines; the printed solves/s is an average since 21:54) against its
  first PC shard, stop above baseline / 0.9, the shard in progress included; and the GPU's 5-minute mean below 90%.
  Baselines 5,453 / 5,422 / 5,441 / 5,305 s (w0-w3).
- Recorded before any stop: a stop after the first checkpoint resumes per F1-PREREG's VOID rule once D1b is done. A
  stop before it (`checkpoint.bin` is first written ~30 min in, and `taskkill /F` skips the final save) leaves
  nothing to resume: stage A then starts afresh once D1b is done, with the same command into `runs/f1-stageA-r1`
  (no `--resume`); `runs/f1-stageA` is kept.
- End: `completed` at 2,000,175,000 iterations, 21 Oct 2026 13:32 (wall 48,456 s, 13.5 h; 61,200,445
  infosets; final mean |regret| 8.01e-4; ~43.6k it/s solving at the end). Not VOID; never stopped or resumed.
- F1-2B = `runs/f1-stageA/blueprint.bin`, sha256
  `DDB06447E14E139D577DC9F1F606C3A322BA4EED3DA7703DBC6ACA79376854EC`.
- The watch: no stop. 36 D1b shards judged, the slowest -0.9% in rate against its worker's baseline (limit -10%);
  every 5-minute GPU mean 100.0%. It ended with the trainer (13:33:18).

## The rating re-verified on the F1 engine
- `f1_verify.ps1` (logs in `runs/f1-eval/`), 21 Oct 13:58-14:33, identity `906227B41D658E88+5AD0B211E9C7D830`:
  - pytest (test_rating, test_rating_receipts): exit 0, 2 skipped.
  - crosscheck vs pool-v1: C6 PASS; C7 A1 and A2 inside pool-v1's 95% intervals.
  - verify: all 9 receipts without search replay ok, bit for bit (A0-reference, A1, A2, Ln, Ls, Lm, v2-4B, v2-8B,
    v2-S3); `live-flopsym` REFUSED (the live runner's search), exit 1.
  - audit: FAIL on exactly the live receipt (STALE) and the headline it makes (none VERIFIED); everything else
    passes. By F1-PREREG this is a pass.
- Then `f1_chain.ps1` (decide, rate, verify of the F1 receipt, pr, scorecard on v2-2B then F1-2B), 14:34-15:48
 ; every step exit 0 (logs in `runs/f1-eval/`).

## Decision arms (`f1-decide-20261021T063406Z.json`, 14:34-15:31)
- Pins: commit 2f1e305, engine 906227B4 (built at d188bd0, engine trees = HEAD's), base v2-2B `1976A53C…`, fix
  F1-2B `DDB06447…`. 100,000 deals per arm, peek-free, snap, no search.
- Gates: G0 PASS (all 7 v2-2B arms equal 11 Oct's per deal, bit for bit), G3 PASS (100,000 / 61,271 opens), G5 PASS.
- LBR per arm, bb/100 (95%), and how the big blind met the open (folds / raises of the opens):

| arm | v2-2B | F1-2B | v2-2B BB fold / raise | F1-2B BB fold / raise |
|---|---|---|---|---|
| std | +269.1 [+257.0, +281.2] | +276.8 [+264.8, +288.9] | | |
| 300-all | +272.4 [+259.3, +285.5] | +265.9 [+253.0, +278.8] | 39.2% / 16.5% | 38.9% / 15.5% |
| 300-bp | +263.0 [+250.4, +275.6] | +259.0 [+246.5, +271.4] | 39.5% / 16.2% | 39.3% / 15.3% |
| 200-all | +241.0 [+229.2, +252.9] | +275.5 [+262.8, +288.2] | 39.2% / 16.5% | 14.9% / 19.9% |
| 200-bp | +241.5 [+229.6, +253.3] | +265.4 [+253.1, +277.7] | 39.5% / 16.2% | 15.2% / 19.5% |
| 250-all | +254.5 [+242.1, +267.0] | +275.7 [+262.8, +288.5] | 39.2% / 16.5% | 28.7% / 17.6% |
| 250-bp | +250.4 [+238.2, +262.6] | +270.8 [+258.3, +283.3] | 39.5% / 16.2% | 29.1% / 17.3% |

(v2-2B reads every open as its 3bb open, so its responses are the same in all its arms.)

- Δ = F1-2B − v2-2B, paired, bb/100 (95%):

| arm | total | postflop | preflop-ended |
|---|---|---|---|
| std | +7.75 [-6.58, +22.09] FLAT | +7.52 FLAT | +0.23 [+0.04, +0.43] UP |
| 300-all | -6.52 [-21.30, +8.27] FLAT | -6.41 FLAT | -0.11 FLAT |
| 300-bp | -4.02 [-18.49, +10.46] FLAT | -3.96 FLAT | -0.05 FLAT |
| 200-all | **+34.47 [+20.03, +48.91] UP** | +46.47 [+32.02, +60.91] UP | -11.99 [-12.13, -11.86] DOWN |
| 200-bp | **+23.96 [+9.74, +38.19] UP** | +31.29 [+17.06, +45.51] UP | -7.32 DOWN |
| 250-all | **+21.15 [+6.56, +35.74] UP** | +26.25 [+11.65, +40.84] UP | -5.10 DOWN |
| 250-bp | **+20.36 [+5.95, +34.77] UP** | +23.43 [+9.02, +37.84] UP | -3.07 DOWN |

- Price LBR-open(s) − LBR-open(300), per bot (decides nothing): v2-2B 200-all -31.4 [-36.3, -26.5], 200-bp -21.5,
  250-all -17.9, 250-bp -12.6 (all DOWN: small opens earn LBR *less* than 3bb against v2-2B); F1-2B 200-all +9.6
  [-1.1, +20.3] FLAT, 200-bp +6.5 FLAT, 250-all +9.8 FLAT, 250-bp +11.8 [+3.4, +20.2] UP.
- Rule: **(a) FAIL** (both UP, needed DOWN); **(a′) FAIL** (both UP, needed not UP); **(b) PASS** (std FLAT).

## (c) PR (`f1-pr-20261021T074609Z.json`)
- `rating.py rate … --label F1-2B`: receipt `20261021T073651388613Z-F1-2B.json`, VERIFIED by `rating.py verify`
  (replay bit for bit). PR 1660.9 [1655.6, 1666.2]; LBR +2672 ± 270 mbb/hand. Beside it: A2 (v2-2B) 1665.6, v2-8B
  1693.3.
- F1-2B vs A2, 20,000,000 deals: -4.76 mbb/hand. Paired change F1-2B − A2: **-4.76 [-8.28, -1.23] mbb/hand, DOWN:
  (c) FAIL.**

## (d) Scorecard (`scorecard-v2-200bb`, both on the F1 engine)
- v2-2B (`receipt-20261021T074634Z-v2-2B-f1base.json`) then F1-2B (`receipt-20261021T074730Z-F1-2B.json`,
  `--previous` the former). Paired change per row, bb/100 (95%):
  - **`min_raiser` -62.1 [-102.0, -22.3] DOWN ON THIS BENCHMARK: (d) FAIL.**
  - Every other row UNCLEAR: baseline -2.4, nit +7.3, station +2.4, maniac +4.1, `baseline_sizes_x0.6` -2.0,
    `baseline_sizes_x1.5` -7.4, `baseline_realistic_sizes` -8.3; LBR (5,000 deals, the peek) -8.7 [-71.6, +54.3].

## Reading (the assistant's; decides nothing)
- F1 changed the big blind's play as the grid change should: against 2bb it folds 14.9% instead of 39.2%, so it
  gives up its blind less often (LBR's preflop-ended part -12.0 bb/100). But LBR's postflop winnings rise by 46
  bb/100, including on the hands F1-2B now defends, so on balance F1-2B is worse there, and worse against
  `min_raiser` (it min-raises at every decision, so it opens 2bb).
- Against v2-2B, LBR's 2bb open earned 31 bb/100 less than its 3bb open; against F1-2B the 2bb price is not
  distinguishable from zero (+9.6 [-1.1, +20.3], FLAT), all four F1-2B prices are positive and 250-bp is UP. That
  price mixes the misreading with the smaller pot (`RESULT.md`), so it does not say what LBR exploits.
- Two readings fit and this run cannot tell them apart: the wider defence is played worse postflop because the new
  subtrees are thinner at 2e9 iterations (a longer run under a new registration could test it; F1-PREREG trains
  stage B only for adoption), or a wider defence is simply more exploitable by LBR at this abstraction. Neither
  changes the verdict, which was registered against v2-2B at this size.
- Scope: this measured LBR, PR and the scorecard, not Slumbot. RM2 (the fold rate against Slumbot's 2bb open) stays
  plausible and unshown; F1 at 2e9 iterations is not the fix on the registered measures.
- The F1-2B rating receipt and both scorecard receipts record commit 2f1e305 with `git.dirty: true` (no file list).
  Against 2f1e305 the tree differs only in docs and `records/` (this file, the two launchers, the outputs,
  `records/slumbot-2/PREREG-DRAFT.md`); no file under `engine/`, `src/`, `scripts/` or `tests/` (checked 21 Oct
  15:54). F1-PREREG's clean-tree rule is the runner's: `f1_eval.py` refuses uncommitted changes to the code
  under test, and both its runs passed it.
