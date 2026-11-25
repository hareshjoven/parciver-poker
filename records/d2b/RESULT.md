# D2b — result

Registered: `PREREG.md` (30 Oct 2026; Amendments 1 and 2, 1 Nov, before any net).

**Verdict: ADOPTED, better only** (9 Nov 2026): Gate A PROCEED, Gate B PASS, the 10,000-deal match NOT DOWN, the user's yes
(9 Nov), and the registered 1,000-hand mock PASSED. The live bot's flop search now takes its leaves from **N1**
(`scripts/slumbot_run.py` NET, commit 6608183); the mock at that commit is GATES (3df72b1).

## Nets
- **N0** `runs/turn-net/final-w2048-d5-e200/net.pt`, sha256 a8bf00be...2ef65 (D2's; reproduced bit-for-bit by `d2b-check`).
- **N1** `runs/turn-net/d2b-w2048-wd0.1/net.pt`, sha256 54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72:
  w2048-d5, weight decay 0.1, trained on D1 + D1b (union fingerprint 7d5a77c30aaad5d6, 951,466 rows). N0's ops, so N0's speed.

## Gate A (1 Nov 2026, `gate-a.json`, `selection.json`): PROCEED
Selection loss N1 0.011799 vs N0 0.014849 (**-20.5%**), the D1b gate half passed (paired MAE -1.05 [-1.09, -1.01] % of the pot).
Other arms on the same half: w2048 wd 0 0.012736, seed 1 0.011937, w1024 wd 0.1 0.012189, w1024 wd 0 0.012895, D1b-only 0.014643.
7 nets trained, no compute job on the GPU in any of 154 looks.

## Gate B (7 Nov 2026, `m1-report.json`): PASS
160 M1 spots, 280 solves, no interim look. Loss(X_N1) - Loss(X_N0) = **-0.585% of the pot**, CI (z 1.96) [-0.757, -0.412];
upper + e = -0.312% <= 0.25%: non-inferior, and the upper end is below 0. The 40 carried spots -0.216% [-0.437, +0.006], the
120 new -0.708% [-0.922, -0.494], reported apart.

## Whole-bot safety
- **Smoke** (8 Nov 13:17-13:30, 50 deals, seed 777,006): 1 and 4 threads identical deal for deal (L1 +2.812 bb/deal).
- **Match** (8 Nov 13:31 to 9 Nov 10:27; `live.json`): L1 (N1) - L0 (N0) over 10,000 deals, baseline row seed 410,001,
  seats swapped, 4 threads, 100 chunks, 0 voided, 0 failures, no compute job on the GPU in any chunk:
  **-7.24 bb/100, raw 95% CI [-23.80, +9.32]** (se 8.24, 50 blocks) — the upper end is >= 0, so **NOT DOWN**.
  AIVAT (reported, not gating) -0.57 [-13.59, +12.45]. Latency in the lock: flop p95 2.376 s (N1) vs 2.380 s (N0), turn 0.844
  vs 0.845 s, river 0.269 vs 0.270 s.
- **Reading, as registered:** the match can catch a gross loss only; its interval (about +/-16.6 bb/100 raw) is inside the
  registered +/-9 to +/-19 range, and it can neither show a gain nor a decline. The point estimate is negative and the
  interval contains zero, so this is not a finding that N1 plays worse, and not one that it plays better. The strength
  evidence for N1 is Gate A and Gate B (flop decisions); the match is the veto, and it did not fire.

## Adoption
1. NET -> N1 in `scripts/slumbot_run.py` with the user's yes (9 Nov 2026): one line, commit 6608183; read-only audit first,
   no other play code names N0's path or sha.
2. **The registered 1,000-hand mock PASSED** (`runs/slumbot/mock-20261109T031133Z`, 9 Nov 11:11-11:52, at 6608183, seed
   20261024, not VOID; `mock_check.json` here and as `records/livesearch/mock_check.json`): 1000 hands, 0 lost, chips conserved,
   every history legal, none refused, 1000 winnings checked, parser check 214,221 prefixes agree; **flop p95 2.976 s (bar 3.0;
   median 2.342, max 3.235, n 765), turn p95 1.175 s (2.0), river p95 0.193 s (2.0), 2.416 s/hand**; 530/765 flop solves
   symmetric. GPU clients: other share at most 3.5%, no compute job.
   **The flop margin is thin** (0.024 s): N0's registered mock was 2.888 s, the live SLUMBOT-2 run's flop p95 2.663 s, and the
   match's lock-in 2.376 s for both nets. p95 over 765 solves moves with the machine; the bar was met as registered, and
   nothing in these runs says N1 is slower than N0 (the match's latencies are equal).
3. Live runs are allowed again at this play code (GATES checked with the committed file: no refusal).

## Reported speed of the cheaper net (not a gate)
`flop_sym_speed.py --cand d2b` (`speed.json`; 9 Nov 11:54-12:53, GPU otherwise idle, 479 jobs of
`runs/flop-eval/m1-jobs-sym.pkl`, A B B A, the same sym leaves, 500 iterations, gamma 5): A = N1, B = `d2b-w1024-wd0.1`
(the better w1024 on the selection half: loss 0.012189 vs N1's 0.011799, +3.3%). **Median ratio B/A 0.540, flop-clustered
95% CI [0.529, 0.544]** (by flop class 0.500 to 0.577). Solve time: N1 median 2.384 s, p95 2.995 s; w1024 median 1.274 s,
p95 1.516 s. PREREG expected ~0.45x if forward time followed multiply-adds (0.349x); it is 0.54x, so forward time does not
scale fully with the multiply-adds. Reported only: nothing here adopts the compact net, which costs +3.3% selection loss. Note that N1's own flop p95 on these
jobs is 2.995 s, close to the 3.0 s bar like the mock's 2.976 s: if that bar ever binds, a faster net is one lever, and it
would need its own registration.

## What it does not decide
Roadmap C (more turn-start data, `records/d3/PREREG-DRAFT.md`, a draft written by another session) needs its own
registration and the user's yes; RM2 and the rating of the new live config likewise. The Slumbot final exam was run on N0
(SLUMBOT-2, UNRESOLVED) and is not rerun here.
