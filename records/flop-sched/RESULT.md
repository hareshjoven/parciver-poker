# FLOP-SCHED (SC2): result

Run 15 Oct 2026, 06:59-07:16 SGT, from `parciver-sc2` at commit `ec8d330` (clean), as PREREG.md and its
Amendment 1 fix it. Record: `scout.json` (copied verbatim from the run). The first run, not VOID: it ran to the
end with the GPU otherwise idle, and the reproduction held.

## Verdict: no arm passes. SC2 closes.
Nothing changes live. Per the registration, the result goes to IDEAS.md and the branch is kept as the tag
`archive/scout-sc2`.

- **Simultaneous updates (sim-g5, sim-d100) never reach the bars by 1,000 iterations.** One leaf call per
  iteration instead of two makes an iteration 0.553x as long. But at equal iterations sim-g5's Game E is
  2.7-4.3x alt-g5's (sim-d100's 4.1-10.2x): 0.843% of the pot against 0.2015% at 500.
- **At equal Game E, sim-g5 costs about 1.3x more** (descriptive, linear interpolation): it needs about 2.4x the
  iterations. sim-g5 at 1,000 iterations (0.277%, 2,206 ms) matches alt-g5 at about 422 iterations (1,681 ms).
- **Delayed averaging (alt-d100) is worse on mean Game E at every checkpoint** (1.02-1.33x alt-g5's). Both
  Game E bars hold from 600 iterations. The Game N proxy bar (0.579%) holds only from 1,000 (N 0.590-0.613% at
  600-900), so n = 1,000: Cost 2.0x the baseline. That bar is noisy: alt-g5 itself is above it at 600 and 800.
  On Game E, dropping the first 100 iterations outright cost more than it saved; DCFR's gamma-5 average already
  discounts them.

## The numbers (all read from scout.json by `report.py scout.json`)

- **Commit** ec8d330dc898783c35cf283ad99bb1076926573b, torch 2.14.0+cu132, NVIDIA GeForce RTX 4070; net N0 (final-w2048-d5-e200) (sha256 a8bf00be).
- **Reproduction:** alt-g5's Game E equals calibrate.json's gamma 5 rows on all 40 spots at 100, 200, 300, 500, 1000 iterations, bit for bit (134 s).
- **Time per iteration** (Game N's leaves, A B B A, 200 timed replays per spot): alt 3.988 ms, sim 2.206 ms, ratio 0.553.
- **Bars (alt-g5 at 500):** E 0.2015%, N 0.5793%, E on the 2bb spots 0.4744% of the pot; Cost_b = 1994.1 ms.

| arm | n_X | Cost_X (ms) | of Cost_b | passes (<= 0.85) |
| --- | --- | --- | --- | --- |
| sim-g5 | none | - | - | fail |
| alt-d100 | 1000 | 3988.1 | 2.000 | fail |
| sim-d100 | none | - | - | fail |

**No arm passes: SC2 closes.**

**Game E (exact check-down leaves), mean over 40 spots, % of the pot:**

| arm | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alt-g5 | 3.0901 | 1.0236 | 0.5170 | 0.2975 | 0.2015 | 0.1478 | 0.1181 | 0.0960 | 0.0803 | 0.0684 |
| sim-g5 | 8.4408 | 3.7937 | 1.7845 | 1.1742 | 0.8432 | 0.6283 | 0.4903 | 0.3921 | 0.3229 | 0.2768 |
| alt-d100 | 3.1418 | 1.0685 | 0.5984 | 0.3679 | 0.2588 | 0.1942 | 0.1537 | 0.1255 | 0.1057 | 0.0913 |
| sim-d100 | 31.6078 | 4.2362 | 2.1210 | 1.3939 | 1.0293 | 0.7946 | 0.6387 | 0.5252 | 0.4395 | 0.3733 |

**Game N (the live net's leaves, a proxy), mean over 40 spots, % of the pot:**

| arm | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alt-g5 | 2.9239 | 1.1087 | 0.7121 | 0.5734 | 0.5793 | 0.6080 | 0.5738 | 0.5989 | 0.5364 | 0.5201 |
| sim-g5 | 8.3844 | 3.2531 | 2.3562 | 1.2773 | 1.8751 | 1.2373 | 0.9113 | 0.7463 | 0.7081 | 0.5854 |
| alt-d100 | 10.2566 | 1.1225 | 0.7687 | 0.6156 | 0.5756 | 0.6049 | 0.6079 | 0.6133 | 0.5898 | 0.5669 |
| sim-d100 | 33.2005 | 3.6339 | 2.4634 | 1.5954 | 1.5227 | 1.3182 | 1.1174 | 0.9882 | 0.9243 | 0.8228 |

**Game E on the 12 2bb spots, % of the pot:**

| arm | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alt-g5 | 7.3548 | 2.5261 | 1.2712 | 0.7117 | 0.4744 | 0.3457 | 0.2749 | 0.2225 | 0.1858 | 0.1571 |
| sim-g5 | 16.6960 | 8.8253 | 3.8406 | 2.5307 | 1.8345 | 1.3561 | 1.0715 | 0.8645 | 0.7119 | 0.6111 |
| alt-d100 | 7.0085 | 2.6470 | 1.4885 | 0.8997 | 0.6279 | 0.4691 | 0.3705 | 0.3017 | 0.2541 | 0.2197 |
| sim-d100 | 74.9732 | 10.2328 | 4.8282 | 3.1404 | 2.3250 | 1.7899 | 1.4363 | 1.1796 | 0.9842 | 0.8316 |

**By pot class, at 500 and 1,000 iterations (% of the pot):**

| pot | spots | arm | E@500 | E@1000 | N@500 | N@1000 |
| --- | --- | --- | --- | --- | --- | --- |
| 2bb | 12 | alt-g5 | 0.4744 | 0.1571 | 0.7924 | 0.8553 |
| 2bb | 12 | sim-g5 | 1.8345 | 0.6111 | 4.8974 | 0.9155 |
| 2bb | 12 | alt-d100 | 0.6279 | 0.2197 | 0.8037 | 0.9150 |
| 2bb | 12 | sim-d100 | 2.3250 | 0.8316 | 3.6179 | 1.5423 |
| 6bb | 24 | alt-g5 | 0.0907 | 0.0326 | 0.3818 | 0.2422 |
| 6bb | 24 | sim-g5 | 0.4050 | 0.1330 | 0.4422 | 0.3747 |
| 6bb | 24 | alt-d100 | 0.1096 | 0.0395 | 0.4020 | 0.2850 |
| 6bb | 24 | sim-d100 | 0.4568 | 0.1753 | 0.4986 | 0.4144 |
| 18bb | 4 | alt-g5 | 0.0478 | 0.0166 | 1.1252 | 1.1822 |
| 18bb | 4 | sim-g5 | 0.4988 | 0.1365 | 1.4051 | 0.8598 |
| 18bb | 4 | alt-d100 | 0.0465 | 0.0167 | 0.9333 | 1.2142 |
| 18bb | 4 | sim-d100 | 0.5773 | 0.1867 | 1.3816 | 1.1146 |

**Per spot: the first checkpoint from which an arm stays at or under alt-g5's value at 500 on that spot** (descriptive):

| arm | game | spots reaching | never | median n | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| sim-g5 | E | 9 | 31 | 1000 | 800 | 1000 |
| sim-g5 | N | 24 | 16 | 800 | 200 | 1000 |
| alt-d100 | E | 40 | 0 | 600 | 500 | 700 |
| alt-d100 | N | 30 | 10 | 600 | 200 | 1000 |
| sim-d100 | E | 2 | 38 | 1000 | 1000 | 1000 |
| sim-d100 | N | 15 | 25 | 600 | 200 | 1000 |

- **GPU state:** start ['8 %, 940 MiB'], end ['100 %, 1876 MiB']; other python processes on the GPU list at the start 2, at the end 2 (under WDDM the desktop's apps are listed; a desktop app's interpreters are not compute jobs).
- **Timing rows:** 160 (40 spots x A B B A).

## Descriptive, not gating
- **Game N stops improving after about 400 iterations.** alt-g5's mean N is 0.573% of the pot at 400 and
  0.579% at 500, then 0.52-0.61% through 1,000, while Game E keeps falling (0.298% to 0.068%). That fits the
  leaf net's error, not the solver, setting the floor (an interpretation, not measured). N is only a proxy,
  with no exact meaning (D3), so this changes
  nothing. Whether the live flop could stop at 400 is a question for M1, not for this record.
- **The 18bb spots' N is the highest (1.13% at 500) though their E is the lowest (0.048%).** N is not monotone
  in the pot: 2bb 0.79%, 6bb 0.38%, 18bb 1.13% at 500. There are only 4 18bb spots.
- **The end GPU reading (100%, 1,876 MiB) is the scout's own load at its last step.** The `python` lists hold
  a desktop app's two interpreters, which are not compute jobs; no other job ran.
