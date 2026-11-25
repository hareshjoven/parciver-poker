# SLUMBOT: the reading (7 Oct 2026 19:00 SGT)

**UNRESOLVED.** v2-2B with live search played Slumbot for the registered 50,000 counted hands. It finished
**-1.54 bb/100, 95% [-15.00, +11.91]**. Read by PREREG's rule, the bot is not shown to beat or to lose to
the 2018 ACPC winner. Every figure below was recomputed independently by a read-only audit from the raw
hands. The pre-registration is `PREREG.md`; the analysis is `scripts/slumbot_report.py`.

| reading | bb/100 | 95% interval | sd, bb/hand | status |
|---|---|---|---|---|
| **win rate, 50,000 counted hands** | **-1.54** | **[-15.00, +11.91]** | 15.35 | registered, primary |
| as big blind, 25,000 hands | -20.42 | [-37.29, -3.56] | 13.61 | registered |
| as small blind, 25,000 hands | +17.34 | [-3.64, +38.31] | 16.92 | registered |
| all-in adjusted (63 pre-river all-ins at equity) | -3.63 | [-16.28, +9.01] | 14.43 | registered, descriptive |
| Slumbot's "Baseline" (ours less Slumbot's own play in our seat) | +0.66 | [-13.75, +15.08] | 16.44 | not registered |

By 10,000-hand block (descriptive): -24.0, +7.3, +11.0, -15.7, +13.6. Each block's own interval is about
+/- 30, so these swings are what noise looks like at this sample size.

## The run, checked from its own files
- 50,000 counted hands after the 100-hand pilot (discarded), in one session. 0 lost hands and 0 failed
  searches. 9 network retries, all of which resumed the same request with the same token.
- Every one of the 50,100 payoff checks is "ok". Hand numbers are contiguous, the pilot is exactly the first
  100 hands, and seats are 25,000 BB / 25,000 SB, alternating from hand 1.
- Our summed winnings equal Slumbot's own running `session_total` to the chip: -82,346 over 50,100 hands,
  pilot included. The counted hands sum to -77,204. The baseline sum matches Slumbot's
  `session_baseline_total` the same way.
- The bot, as PREREG fixed it:
  - commit `2f471fa`, blueprint v2-2B (sha256 `1976A53C...`), translation snap;
  - search from the turn at 100 iterations and on the river at 200;
  - `SubgameGpu(terminals64=True)`, i.e. Q3 at 2f471fa (today that is Q4; Q3 is `q3=True`), as V5's K2
    MATTERS reading required.
- Speed:
  - 3.49 s per hand elapsed, of which 0.50 s was thinking; the rest is network.
  - 37,704 searched decisions averaged 0.66 s each; the longest took 10.6 s.
  - The counted hands took 48.5 h, against the pilot's projection of 54.7 h.

## How to read it
- **Why unresolved.** Hands swung more than PREREG assumed: 15.35 bb per hand against ~10, because 200 bb
  stacks make big pots. So 50,000 hands give +/- 13.5 bb/100, not +/- 9. N was fixed and does not change.
  The pilot's 100 hands showed 10.25, which is too few to see the tail.
- **The seats.** The seat split mostly measures the value of position, which no strategy removes. The small
  blind acts last after the flop, so even an equilibrium strategy wins there and loses in the big blind. The
  per-seat figures are therefore not a separate finding about the bot. The registered reading is their mean.
- **All-in luck.** Scoring the 63 all-ins called before the river at our exact equity moves the rate to
  -3.63. We ran about 2 bb/100 above equity in them, which is luck in our favour, not skill. The other 117
  all-ins were river all-ins, already scored at their result. The interval barely narrows (sd 14.43 against
  15.35).
- **Slumbot's Baseline** (not registered).
  - Slumbot returns `baseline_winnings` with every hand. slumbot.com describes it as the player's
    "relative performance to Slumbot playing in the same seat with the same cards".
  - The field is already that difference, not Slumbot's result. It is 0 in all 2,528 hands where Slumbot's
    small blind folded at once, which an identical replay requires.
  - Over an even seat split it estimates the same win rate: +0.66. It is noisier, not quieter: its variance
    is 115% of raw. Slumbot's replay is its own sampled play, so subtracting it adds Slumbot's noise, and
    its correlation with our winnings (0.45) is too weak to pay for that.
  - The audit caught this field's meaning (the first analysis read it as Slumbot's own result). It was
    corrected before this record.
- **What it is not.** One opponent: a result against Slumbot measures Slumbot, not our exploitability. It
  is reported next to the rating's PR and changes no PR or gate.

## Files
| file | what |
|---|---|
| `PREREG.md` | the pre-registration (5 Oct 03:08 SGT) |
| `result.json` | the reading, as `scripts/slumbot_report.py --json` writes it |
| `hands.jsonl.xz` | every hand (3.9 MB; xz of the run's 46.5 MB `hands.jsonl`, sha256 `0f0b087c...e26037`) |
| `run.json`, `events.jsonl`, `status.json` | the run's configuration, events and final state, as the runner wrote them |
| `mock_check.json` | the registered mock run (1,000 hands) that cleared the client |

The full run folder, `runs/slumbot/20261005T100610Z/`, stays local (gitignored). It also holds
`requests.jsonl`, every request and response (58.5 MB, sha256 `ccc116a3...231898`). To re-derive the
reading from this folder alone, unpack the hands next to the other files and run the script on it. This
was tested on 7 Oct, and the reading came out identical to `result.json`:

    xz -dk records/slumbot/hands.jsonl.xz
    .venv/Scripts/python.exe scripts/slumbot_report.py records/slumbot
