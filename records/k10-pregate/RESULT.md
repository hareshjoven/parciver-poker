# K10's pre-gate under Q4: result

Run 11 Oct 2026, 14:43 (`pregate-20261011T064342Z.json`), as PREREG.md fixes it.
- **Code:** commit `985c2d5`. It differs from the commit that added PREREG.md (`54a5f6e`) only in
  HANDOFF.md. The tree was clean for the code under test. `engine/src` held uncommitted preflop-open edits,
  but the installed engine was built on 7 Oct, before them, and is the one the run loaded.
- **GPU:** RTX 4070, busy 36% before and 11% after (limit 60).
- **Status:** not VOID, and the first run, so it counts.

## Verdict: GO to the probe (C = 1.154 >= 1.05)
The median turn job (9,609 rows; 105 wave launches against 21 today, from `waves.py`), per replayed
iteration:

| term | ms |
|---|---|
| T (Q4 iteration) | 2.1528 |
| down + up | 0.7910 |
| terminals, Q4 (`fold_ff` + `showdown_ff`) | 1.3366 |
| terminals, float32 (`matmul` + `fold` + `showdown`) | 0.6813 |
| r = 1.962, so k = 0.038 | |
| S = 0.5 x 0.7910 + 0.8 x 0.038 x 0.6813 | 0.4164 |
| g = 0.767 µs, D = 2 x 84 x g | 0.1288 |
| **C = T / (T - S + D)** | **1.154** |

- C is 1.130 at 1 µs and 1.039 at 2 µs, and it reaches 1.05 at any g below 1.87 µs.
- g is 0.767 µs. Its 200 → 1,000-launch slope is 0.803 µs (descriptive). That slope is 5% above g, the
  direction PREREG named as the sign of a host bound, but small; so the 20-launch graph shows little sign
  of being bound by the host.
- g's intercept is 11.1 µs per replay, beside P2's prep-only graph at 15.7 µs (both per-replay overhead;
  descriptive).
- **Against KERNELS' registered expectation** (Q4 terminals at 2-3.5x float32's: C 1.08-1.11 at 1 µs,
  passing below g 1.4-1.7 µs): Q4's terminals landed at 1.96x, just below that range, so k > 0. The run's C
  is 1.130 at 1 µs and passes below 1.87 µs, slightly above the band.

## The post-Q4 profile (descriptive; KERNELS Step 1's P1 and P2; P3 and P4 not run)
- **The iteration** (P2, Q4, leave-one-out):
  - **Turn jobs** of 8,457, 9,609 and 11,733 rows: 1.90, 2.15 and 2.63 ms. Q3's median was 6.25 ms,
    2.9x as long (S1's live x2.89).
  - **The median job's shares:** `showdown_ff` 46% (23.2 + 22.8), `fold_ff` 16%, `up` 21%, `down` 16%.
    The terminals are now 62% of the iteration and down + up 37%.
  - **The median river job** (117 rows): 0.084 ms.
- **A live solve's stages** (P1, Q4, the 53 recorded jobs):
  - **Turn, own iterations:** 33 solves. 229 ms in the GPU lock, of which replays are 95%. `hands` (5.6 ms)
    and `plan` (85.7 ms) run outside the lock, so replays are 68% of the whole solve (218.6 of 320.6 ms).
  - **River, own iterations:** 20 solves in 113 ms. This row includes the river kernels' compiles:
    `profile_stages` warms up on `jobs[:3]`, and all three are turn jobs, although its comment says "both
    streets' shapes". So its 77% "first step + capture" is mostly compiling (3.7 ms in the 400-iteration
    pass, which runs after it). Read it as contaminated.
  - **At 400 iterations:** replays are 99% of the turn's locked time and 80% of the river's.
- **Reading.** C is a ceiling on the replayed iteration, and generous by design. It assumes no L2 reuse
  today, but float32 `_down` already runs near peak bandwidth.
  - 95% of S is down + up. The terminals add almost nothing (k = 0.038): at r ~ 2 the gate counts Q4's
    terminals as bound by arithmetic, which cache reuse does not shorten.
  - **At most:** the median turn job's replayed iteration takes 13% less time (1 - 1/1.154). A turn solve
    takes about 9% less (ceiling 1.10), since replays are 68% of the whole solve. Over the locked time
    alone (replays at 95%), the ceiling is 1.146.

## What it decides
By the registered rule, K10 goes on to the probe: a timing-only prototype in a scratch copy, then 12
interleaved rounds against `waves=0`, with K10w built only if the median paired ratio is >= 1.05. The
probe is a build and starts only on the user's go.
