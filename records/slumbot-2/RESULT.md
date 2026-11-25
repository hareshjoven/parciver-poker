# SLUMBOT-2: the reading (30 Oct 2026 17:53)

**UNRESOLVED.** v2-8B with the full live search played Slumbot for the registered 50,000 counted hands. It
finished **-12.54 bb/100, 95% [-26.41, +1.33]**. By PREREG's rule the bot is not shown to beat or to lose to the
2018 ACPC winner. The analysis is `scripts/slumbot_report.py`; the pre-registration is `PREREG.md`.

| reading | bb/100 | 95% interval | sd, bb/hand | status |
|---|---|---|---|---|
| **win rate, 50,000 counted hands** | **-12.54** | **[-26.41, +1.33]** | 15.83 | registered, primary |
| as big blind, 25,000 hands | -24.42 | [-43.12, -5.73] | 15.08 | registered |
| as small blind, 25,000 hands | -0.65 | [-21.15, +19.85] | 16.54 | registered |
| all-in adjusted (59 pre-river all-ins at equity) | -7.59 | [-20.89, +5.71] | 15.17 | registered, descriptive |
| Slumbot's "Baseline" (ours less Slumbot's own play in our seat) | -8.57 | [-23.63, +6.49] | 17.18 | not registered |
| **this run less the 5 Oct run** | **-10.99** | **[-30.32, +8.33]** | | new, descriptive |

By 10,000-hand block (descriptive): -4.2, +4.8, -40.2, -18.7, -4.4. Each block's own interval is about +/- 31, so
these swings are what noise looks like at this sample size.

## The run, checked from its own files
- `runs/slumbot/20261025T074051Z`, at 8d16d87 (play code as GATES' 1169e6e); log `runs/slumbot/live-slumbot2.log`.
  - Session 1, launched 25 Oct 15:40:51, 15:41:14-15:49:27: the 100-hand pilot, discarded. It projected 68.1 h, over `LIMIT_HOURS`
    60, so the runner paused and the assistant resumed with `--go` under P2 (at most 100 h).
  - Session 2, 25 Oct 15:50:14 to 28 Oct 10:22:40: the counted hands. No restart, so Amendment 1's resume was
    never used; its report checks were.
- 50,000 counted hands, seats 25,000 BB / 25,000 SB, hand numbers contiguous, every one of the 50,100 payoff checks
  "ok". 75 request retries in 22 bursts: 9 bursts (14 retries) resumed; 13 bursts (61 retries) failed, and those are
  the 13 lost hands.
- **13 hands lost, excluded (the limit is 250).** Slumbot deals the cards and the losses do not depend on them:
  - 25647-25657, 11 in a row: no answer to `new_hand` after 6 tries, 27 Oct from 01:10 to about 01:57. None
    was dealt to us (no decision, no winnings), and Slumbot's totals pass through them unchanged.
  - 37146 and 46993: a retried `act` was refused (the first try may have reached the server). Slumbot did count
    these two, -100 and +100 chips, and +100 and +1,300 on its baseline total.
- Slumbot's own totals agree to the chip. Its session total is -626,868 chips over 50,002 hands; our counted hands
  sum to -626,868 (the two cut hands net to 0). Its baseline total is -427,118; ours is -428,518, and the 1,400
  apart is exactly those two hands. The report chains the totals hand by hand; the 3 hands right after a lost
  one have only Slumbot's hand count checked, and the arithmetic here covers those 3.
- The bot, as PREREG fixed it: blueprint v2-8B (sha256 `FD27FE50...C7AE071`), translation snap, net N0, seed
  20261024; flop 500 iterations (gamma 5, sym leaves), turn 400 (shift1_g5), river 2,000, Q4 terminals.
- Speed: 4.79 s per hand elapsed (4.71 within hands, network included), of which 1.95 s thinking. The counted hands
  took 66.5 h, against the pilot's 68.1 h. Searched decisions (71,853, mean 1.36 s, longest 3.16 s):

| street | n | median, s | p95, s | max, s | the mock's p95 and bar |
|---|---|---:|---:|---:|---|
| flop | 33,039 | 2.302 | 2.663 | 3.157 | 2.888, bar 3.0 |
| turn | 22,378 | 0.885 | 0.926 | 3.070 | 1.114, bar 2.0 |
| river | 16,436 | 0.164 | 0.174 | 2.295 | 0.180, bar 2.0 |

  The bars gate the mock, not the run; 5 of the flop decisions passed 3.0 s.
- GPU clients, session 2 (65,803 reads, 48 errors): other clients on 30.4% of reads, together at most 16.4% (25 Oct
  19:16:56: a browser 13.0). The named ones are all desktop apps and Windows, no other python (one read names no
  client). The log cannot tell compute from 3D (our own solver reads as 3D), so this is by name. Reported only. The
  pilot's: 52% of reads (83 of 161), at most 10.0% (a desktop app 9.3%, a desktop app, no compute job).

## How to read it
- **Why unresolved.** The interval is +/- 13.9, and it covers zero by 1.33. PREREG's rule has no "nearly", and N
  was fixed, so the reading is UNRESOLVED and no more hands are played on it. The pilot's sd of 6.04 was too few
  hands to see the tail; PREREG's 15.35 was close to the realised 15.83.
- **Against the 5 Oct run: no gain shown, no decline shown.** -10.99 [-30.32, +8.33], the interval of a difference
  of independent means (PREREG said about +/- 19; it came out +/- 19.33). A gain above about +8 bb/100 is outside
  the interval. What we had measured of those changes, all against our own bots: PR, v2-8B against A2 +27.6 mbb/hand
  [24.3, 30.9], which is +2.76 bb/100 [2.43, 3.09]. Head to head, in bb/100: the BP-8B live match -0.61 [-19.39,
  +18.16], the flop search -0.85 [-18.87, +17.18] (8 Oct, before sym leaves and Q4, which have no head-to-head figure),
  turn 400 +2.55 [-6.56, +11.65], river 2,000 +11.28 [+2.85, +19.72]. Of the four, only the river's excludes zero, and
  the low ends of PR and the river are +2.43 and +2.85: a small gain is supported, a large one is not. If the gains
  simply added (river plus blueprint, about +14) that would be above +8.33, but adding them across different opponents
  is an assumption, so this run neither confirms nor refutes them. The runs differ in the blueprint, the flop search and its
  net N0, the turn schedule and iterations, the river iterations, the terminals (Q3, now Q4) and the seed, so this
  attributes nothing to any one part.
- **All-in luck.** Scoring the 59 all-ins called before the river at our exact equity moves the rate to -7.59: we
  ran about 5 bb/100 below equity in them. Against 5 Oct on that score: -3.96 [-22.31, +14.40].
- **The seats** mostly measure the value of position, which no strategy removes (the small blind acts last after
  the flop), so they are not a separate finding; the registered reading is their mean. The big blind's interval
  excludes zero here and in the 5 Oct run. Against 5 Oct: BB -4.00 [-29.18, +21.18], SB -17.99 [-47.32, +11.34].
- **Slumbot's Baseline** (not registered) estimates the same rate and again is noisier, not quieter: its variance is
  118% of raw (115% on 5 Oct). See `records/slumbot/RESULT.md` for what the field is.
- **What it is not.** Not a claim of unexploitability: Slumbot is one opponent and a result measures us against it.
  It changes no PR or gate. The live rating (PR 1860.4) is still B0's, stale for this config.

## The big blind against Slumbot's 2bb open (descriptive)
`scripts/rm2_scout.py` on the new hands (`scout.json`), against the 5 Oct figures (`records/rm2/scout.json`,
`records/rm2/SCOUT.md`):

| | 5 Oct, v2-2B | this run, v2-8B |
|---|---:|---:|
| big-blind hands facing the 2bb open | 22,472 | 22,531 |
| actions: fold / call / raise | 8,797 / 10,065 / 3,610 | 8,581 / 10,108 / 3,842 |
| mean fold probability | 39.2% | 38.1% |
| folded hands' equity against Slumbot's real hand | 40.5% [40.2, 40.8] | 40.1% [39.9, 40.4] |
| folded hands with equity >= 33% / >= 40% | 73.6% / 36.6% | 71.5% / 35.2% |
| check-down gain per fold, bb | +0.620 [0.610, 0.631] | +0.605 [0.594, 0.616] |
| the same gain as bb/100 of the BB seat | 21.8 | 20.8 |
| called hands: real less check-down, bb | -0.388 [-0.628, -0.148] | -0.436 [-0.686, -0.186] |

So four times more blueprint iterations left this spot about where it was. The check-down gain is a ceiling, not
what calling would earn (Slumbot bets and folds postflop). The hands we do call earn about 0.44 bb less than their
check-down; less that gap, a fold's gain is +0.17 bb, about [-0.08, +0.42], not shown above zero (called hands are
not folded hands, so this is arithmetic, not a measurement). SCOUT's reading stands: a leak is plausible, not shown.

## Flop decisions facing an off-grid bet (descriptive)
`scripts/slumbot_offgrid.py` (`offgrid.json`). Off the grid, as in `scripts/v2leak.py`: not an all-in, and not the
amount the 200bb blueprint grid gives at the real betting state. Its replay of the betting reproduces the recorded
bet options of all 33,039 decisions (the flop search's menu at that state), which checks the replayed state and the
formula; the off-grid test itself has its own unit tests. The real state is the point: after a plain `b200c`
preflop (15,888 of the decisions, 48.1%) the real pot is 4bb, so a half-pot bet is about a third of the 6bb pot
the translated 3bb open would make. Such a bet counts as on the grid here, because the flop search works in real chips.

| searched flop decisions | n | share | median, s | p95, s | max, s |
|---|---:|---:|---:|---:|---:|
| no bet to answer | 23,947 | 72.5% | 2.389 | 2.682 | 3.157 |
| facing a bet on the grid, the hand's first flop decision | 2,279 | 6.9% | 2.429 | 2.711 | 3.101 |
| facing a bet on the grid, a later flop decision | 6,313 | 19.1% | 1.903 | 2.121 | 2.667 |
| **facing a bet off the grid** (all later decisions) | **492** | **1.5%** | **1.967** | **2.395** | **2.566** |
| facing an all-in | 8 | 0.0% | 1.823 | 2.066 | 2.161 |

- Off-grid bets are 5.4% of the 9,092 decisions facing a bet, and every one is the same thing: Slumbot raising
  our bet by half the pot. They are not what makes the flop slow: next to the on-grid later decisions they cost
  0.06 s at the median and 0.27 s at p95, and the slowest decisions are a hand's first flop decisions.
- **A correction.** `records/live-flop/PREREG.md` and an old HANDOFF line said Slumbot's flop bets are "often off the
  grid", expecting it from the mock (which bets every legal size) and without measuring it. Measured: all 8,421
  of Slumbot's first flop bets were exactly half pot (8,044) or pot (377). Off-grid decisions were 35.6% of the
  mock's flop decisions (p95 2.966 s; `offgrid_mock.json`), 1.5% here, which helps the live flop p95 (2.663 s) sit
  below the mock's (2.888 s). It does not explain all of the latency: with no bet to answer the live decisions are the
  slower ones (median 2.389 s against the mock's 2.183, p95 2.682 against 2.426). Seen, not explained. This is about
  the flop only: preflop, its 2bb open is off the grid.

## Files
| file | what |
|---|---|
| `PREREG.md` | the pre-registration (25 Oct 14:36) and Amendment 1 |
| `result.json` | the reading, as `scripts/slumbot_report.py --json` writes it |
| `scout.json`, `offgrid.json`, `offgrid_mock.json` | the descriptive readings above; the last is the same script on the mock's hands |
| `hands.jsonl.xz` | every hand (3.6 MB, sha256 `99654dee...540c77`; xz of the run's 48.8 MB `hands.jsonl`, `bba22f86...a46a8e8`) |
| `gpu_clients.jsonl.xz` | the GPU-client log (0.4 MB; xz of 5.0 MB) |
| `run.json`, `events.jsonl`, `status.json` | the run's configuration, events and final state, as the runner wrote them |
| `mock_check.json` | the registered mock's check |

The full run folder stays local (gitignored). It also holds `requests.jsonl`, every request and response (58.8 MB,
sha256 `09050c61...b6c748`). To re-derive the reading from this folder alone (tested 30 Oct: the same as
`result.json` but its `run`, the folder's name; `rm2_scout.py` and `slumbot_offgrid.py` on the xz gave the same files):

    xz -dk records/slumbot-2/hands.jsonl.xz
    .venv/Scripts/python.exe scripts/slumbot_report.py records/slumbot-2

## Before the run
Registered: `PREREG.md` (25 Oct 2026 14:36, commit 1169e6e; the user's yes: "yes, register SLUMBOT-2 with N0
and run the mock"). The bot: v2-8B (fd27fe50...c7ae071), N0, seed 20261024, N 50,000; flop 500 / turn 400 /
river 2,000, Q4, sym leaves.

### The registered mock: PASSED
- `runs/slumbot/mock-20261025T063547Z` (25 Oct 14:36:11-15:15:16, at 1169e6e, clean; log
  `runs/slumbot/mock-slumbot2.log`; copy `mock_check.json`, sha256 B5C6C67A...).
- 1,000 hands, 0 lost, chips conserved, every history legal, none refused, 1,000 winnings checked (0 unchecked);
  parser check 214,221 prefixes agree; 2,789 decisions; 2.32 s/hand.
- flop p95 **2.888 s** (bar 3.0; median 2.211, max 3.107, n 781; 71% of flop solves symmetric); turn p95 **1.114
  s** (bar 2.0, n 509; max 1.300); river p95 **0.180 s** (bar 2.0, n 329). BP-8B's mock (same bot but the seed,
  so other deals; 25 Oct): flop 2.753 / turn 1.110 / river 0.184 s, 2.21 s/hand.
- Not VOID: one session, no crash, kill or reboot (the last boot 07:59, before the run), no stall, no other compute
  job on the GPU. The PC in normal use: the GPU-client log (827 reads) saw other clients on 45% of reads, at most
  1.8% (a browser, 3D), reported only.
- **GATES written** (PREREG "Registered mock first": committed before the first live hand):
  `records/livesearch/mock_check.json` is this check (replacing BP-8B's mock at 6032470).

### Amendment 1: a restart (25 Oct 15:25, before the first live hand)
The user's yes: a restart is resumed with `--dir <run>`, cut and lost hands are excluded (the 250-hand stop
unchanged), uneven seats are allowed (PREREG "Amendment 1"). `scripts/slumbot_report.py` now chains Slumbot's
running totals hand by hand per session and sums each session's playing time. Run on the 5 Oct run, every number
is the same. `tests/test_slumbot_report.py` covers lost hands, a cut session, uneven seats, a broken chain, the
limit, totals restarting from zero, a mismatch on a lost hand and a torn line.
The report is not play code: GATES stands.
