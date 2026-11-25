# BP-8B — result

Registered: `PREREG.md` (18 Oct 2026; Stage 3 addendum 19 Oct 21:13 SGT).

**Verdict: ADOPTED** (25 Oct 13:00 SGT): the rating a GAIN, the live match NOT DOWN, Adoption 1 and 2 done; the live
runner plays v2-8B and its registered mock is GATES.

## Training
- **Stage 1:** v2-4B `runs/bp-8b-stage1/blueprint.bin` (sha256 8b1a0db7...3500).
- **Stage 2:** completed 20 Oct 2026 01:29:30 SGT (exit 0): **v2-8B** `runs/bp-8b-stage2/blueprint.bin`,
  8,000,247,000 iterations, final mean |regret| 9.060e-4, sha256
  fd27fe50324eba29f0454ab5c68604c2920f2bad18dd437544255d754c7ae071 (checkpoint 4f5ab12a...67b8b1).
- **Stage 3:** resumed on the cloud 20 Oct 01:45 SGT from stage 2's checkpoint (8,000,247,000); ended by its time
  box 20:30 SGT at **12,348,045,000** iterations (39,883,752 infosets, mean |regret| 8.912e-4, ~64,390 it/s):
  **v2-S3** `runs/bp-8b-stage3/blueprint.bin`, sha256 0e462bbd04b269d598c39e3a553ef68b7191d509c3b0b5daa0d5ff6faa0ac888
  (checkpoint 5f0ccc5b...adb915d), pulled sha-checked 20:37 / 20:48 SGT (PULL-DONE 20:48:37; `chain.log`).

## Rating (pool-v2; the registered statistic, GAIN iff lo > 0)
`scripts/rating.py rate --dir runs/rating/pool-v2 --candidate runs/bp-8b-stage2/blueprint.bin --label v2-8B`
(no search, snap, 8 threads, BelowNormal; 20 Oct 10:53-10:57 SGT; log `runs/bp-8b-stage2/rate-v2-8B.log`).

| | PR [95%] | paired change vs A2, mbb/hand [95%] | LBR |
|---|---|---|---|
| A2 (v2-2B) | 1665.6 | 0 | +2776 |
| v2-4B (reported) | 1685.5 [1680.4, 1690.6] | +19.9 [16.8, 23.0] | +2711 +/-277 |
| **v2-8B** | **1693.3 [1688.0, 1698.5]** | **+27.6 [24.3, 30.9]** (se 1.69) | +2815 +/-272 |
| v2-S3 (reported) | 1695.7 [1690.5, 1701.0] | +30.1 [26.7, 33.5] (se 1.72) | +2760 +/-272 |

- Receipt `20261020T025654251378Z-v2-8B.json`, VERIFIED. Paired change: `parciver.rating.change` on the
  candidate's anchor rows vs A2's receipt rows, weights the calibration's p (A2 = 1.0, 20M deals); the same
  script reproduces v2-4B's +19.88 [16.77, 23.00].
- **GAIN** (lo 24.3 > 0). The dose: 2B->4B +19.9, 2B->8B +27.6 (4B->8B ~+7.7). LBR within noise of A2 and v2-4B
  (reported, not gating).
- **v2-S3 (Stage 3, reported; 20 Oct 21:54-21:59 SGT, the same command with `--label v2-S3 --no-browser`, CPU
  only beside D1b's PC generation; log `runs/bp-8b-stage3/rate-v2-S3.log`):** receipt
  `20261020T135846365217Z-v2-S3.json`, VERIFIED; vs A2 +30.1 [26.7, 33.5]; paired vs v2-8B on the same deals
  **+2.5 [-0.4, +5.4]** (not significant: 8B->12.3B adds at most ~5 mbb/hand). Per the addendum it enters the
  live match or adoption only by a later amendment with the user's yes; the match stays L1 = v2-8B.
- Then per PREREG (done): the live match L1 (v2-8B) against L0 (the live bot, river 2,000), its script written,
  tested and audited before any data.
- **Live match (`scripts/bp8b_live.py match`, L1 = v2-8B, 10,000 deals, 4 threads):** smoke re-run PASS 22 Oct
  22:04-22:23 SGT (1 and 4 threads identical deal for deal, L1 -0.801 bb/deal on the smoke deals); the match started
  22:23. It stopped 23 Oct 07:59 SGT (exit 0x40010004: the session was signed out ~07:59 for Windows Update (7002
  logged 07:59:37) and the PC restarted 08:00-08:02, again 19:37-19:41; System log events 7002, 1074, 6006) after
  4,500 deals (last chunk saved 07:52). Not a failure by PREREG ("a reboot ... resumes from its last saved chunk"):
  resumed 23 Oct 22:05 SGT (log line in `runs/gpu-queue.log`). ~7.6 s/deal, so ~11.6 h left. Windows
  Update restarted the PC again 24 Oct (restart started 08:22:36 by MoUsoCoreWorker, then 08:24:58 by
  TrustedInstaller; System log 1074) after 9,300 deals (last chunk saved 08:19); resumed 24 Oct 11:46 SGT.
  Three sessions in `live.json` (commits e134929, e134929, c2958df; the play code pins equal throughout), no failed
  chunk, no code history.
- **Live match result (`records/bp-8b/live.json`, ended 24 Oct 13:21 SGT): L1 - L0 = -0.61 bb/100, 95% [-19.39,
  +18.16] over 10,000 deals (50 blocks of 200): NOT DOWN** (DOWN iff the upper end < 0). AIVAT (reported) +1.89
  [-10.44, +14.22]. 7.65 s/deal; solves per deal L0 / L1: flop 1.42 / 1.42, turn 0.97 / 0.98, river 0.71 / 0.71;
  flop latency in the GPU lock median 2.07 s, p95 2.37 s, max 3.52 s (L0) / 3.25 s (L1) (reported; the bars are the
  mock's).
- **Adoption, per PREREG:** the rating is a GAIN and the live match is not DOWN, so the next step is Adoption 1
  (`scripts/slumbot_run.py` points at v2-8B, committed with the user's yes), then Adoption 2 (the registered
  1,000-hand mock; its pass becomes GATES with the user's yes, a fail reverts). The user's yes, 25 Oct: "yes go
  ahead".
- **Adoption 1 done:** `scripts/slumbot_run.py` plays v2-8B, commit 6032470 (25 Oct 12:12 SGT); the bars, search,
  net, schedules, seed and N unchanged. With it, from the read-only audit: the resume branch refuses a run recorded
  on another blueprint, and `run.json`'s prereg list names this PREREG.
- **Adoption 2, the registered mock: PASSED** (`runs/slumbot/mock-20261025T041221Z`, 25 Oct 12:12:43-12:50:01 SGT,
  at 6032470, clean; copy `records/bp-8b/mock_check.json`, sha256 0B10426B...). 1,000 hands, 0 lost, chips
  conserved, every history legal, none refused, 1,000 winnings checked; parser check 214,221 prefixes agree; 2,755
  decisions; 2.21 s/hand.
  - flop p95 **2.753 s** (bar 3.0; median 2.196, max 3.231, n 746; 70% of flop solves symmetric); turn p95 **1.110
    s** (bar 2.0, n 501; max 3.328); river p95 **0.184 s** (bar 2.0, n 334). RIVER R1's mock (v2-2B, 19-20 Oct): flop
    2.821 / turn 1.124 / river 0.180 s, 2.34 s/hand.
  - Not VOID: no crash, kill or reboot (one session; the last boot 07:59, before the run); no other compute job on
    the GPU. The PC in normal use: the GPU-client log (801 reads) saw other clients on 36% of reads, at most
    3.7% (a desktop app, 3D), reported only, as in R1's mock.
- **GATES written** with the user's yes (25 Oct 12:58 SGT: "yes, write the BP-8B mock as GATES"):
  `records/livesearch/mock_check.json` is this mock (sha256 0B10426B..., replacing R1's mock on v2-2B). `gates_refuse`
  checked before the commit: only "not committed" left (engine E58C7292, blueprint, net, leaves and play code match).
  **BP-8B: ADOPTED.**
