# RIVER R1 — result (live river at 2,000 iterations vs 200)

Registered: `PREREG.md` (18 Oct 2026, with Amendments 1-3). Records: `p2.json`, `p1.json` (full rows inside).

## P2 (not DOWN) — finished 19 Oct 2026 21:49
- 10,000 deals, R1 - R0 = **+11.28 bb/100, 95% [+2.85, +19.72]** (50 blocks): not DOWN (DOWN iff hi < 0).
  AIVAT (reported): +7.39 [+1.13, +13.65].
- Sessions: 59a3b7a (18 Oct), a66eb5c (x2), f5e5d4f (19 Oct 11:48, Amendment 3, the graph-capture race fixed).
  The CUDA failures before Amendment 3 are listed in `p2.json` (`cuda_failures`); none after it.
- Latency in the lock (descriptive), p95: river R0 0.079 s, R1 0.257 s (bar 2 s); flop and turn the same in both
  arms (flop 2.47 s, turn 0.90 s). ~8.0 s/deal.

## P1 (HOLDS) — 19 Oct 2026 21:53:57-21:54:47
- 212 river jobs from P2's deals 500-799 on 93 boards: R1 - R0 = **-1.3386% of the pot, 95% [-1.6145, -1.0912]**
  (R0 1.398%, R1 0.060%): holds (upper end < 0). Not VOID: no other GPU job of this Python at start or end.
- **How it ran on P2's play code:** P2 ran at f5e5d4f; at HEAD (0f0e37b) `src/` differs only by 7839aa6 (D1b's
  `src/parciver/turn_values.py`), so P1 at HEAD refused itself (`runs/river-live/p1-refused-at-HEAD.log`). P1 was
  run unchanged from a clean detached worktree at f5e5d4f (`C:/tmp/p1-at-f5e5d4f`, its `runs/river-live` a
  junction to the main one, the venv's engine), so its own check passed; `p1.json` records commit f5e5d4f and was
  copied back byte for byte (sha256 2A7B79AE...). The worktree was then removed.

## Adoption (PREREG "Adoption"): ADOPT R1 (P1 holds, P2 not DOWN, the mock met its bars)
1. **The change:** `river_iters` 2,000 in `scripts/slumbot_run.py`, commit af9f06b (19 Oct 23:33:24, after
   the user's yes; its message's "~22:00" is wrong). Bars unchanged. With it, from the read-only audit: the
   resume guard refuses a run of another river, `mock_check.json` records `river_iters`.
2. **The registered mock: PASSED** (`runs/slumbot/mock-20261019T153337Z`, 19 Oct 23:33:37 - 20 Oct 00:13:32,
   at af9f06b, clean; copy `mock_check.json` here, sha256 DE7E7342...). 1,000 hands, 0 lost, chips conserved,
   every history legal, none refused, parser check 214,221 prefixes agree; 2,819 decisions; 2.34 s/hand.
   - flop p95 **2.821 s** (bar 3.0; median 2.207, max 2.937, n 776); turn p95 **1.124 s** (bar 2.0, n 511);
     river p95 **0.180 s** (bar 2.0, n 353; median 0.173, max 3.130). B1's mock at 177972f, river 200: flop
     2.759 / turn 1.060 / river 0.035 s, 2.17 s/hand.
   - Not VOID: no crash, kill or reboot; no other GPU job of this Python. The PC in normal use (a browser open):
     the GPU-client log saw other clients on every read, at most 5.9% (a browser 4.5%), reported only.
3. **GATES written** (the user's yes after 00:15 20 Oct; committed 4f5d2cc at 00:41): `records/livesearch/mock_check.json` is this mock's
   check (sha256 DE7E7342...); 177972f's (B1, 18 Oct) is in git at 319008e. A live run is allowed at af9f06b's
   play code.
- The rating: the live receipt (PR 1860.4) is B0's and STALE; rating R1 is the user's call.
