# SLUMBOT-2: the final exam (REGISTERED 2026-10-25 14:36; drafted 2026-10-20 22:58)

**Status: REGISTERED** on the user's yes, 25 Oct 2026 ~14:29: "yes, register SLUMBOT-2 with N0 and run the
mock". Drafted on the user's yes (20 Oct, "draft the Slumbot rematch pre-registration, leaving the config open
until the match result"). B1 and B2 were closed by BP-8B and F1; B4 and P1-P3 on the user's okay (25 Oct) to
The assistant's recommendations; B3 by the yes to register. The draft's conditions, all met at registration:
- the BP-8B live match has read;
- the user has decided F1 (`records/rm2/SCOUT.md`): done, F1 ran and was NOT ADOPTED (21 Oct);
- every **OPEN** item below is closed;
- the user says yes;
- it is renamed PREREG.md, stamped and committed.

No live hand is played before the registered mock passes and its check is committed as GATES. This is HANDOFF's
Roadmap 4: "a new Slumbot run (registered mock first) with the full config".

## Why
The 5 Oct run (`records/slumbot/`) measured v2-2B with turn and river search only. It read **−1.54 bb/100,
95% [−15.00, +11.91], UNRESOLVED**.

Since then the live bot gained:
- flop search (LIVE-FLOP, FLOP-SYM C2);
- turn search at 400 iterations (LIVESEARCH B1);
- river search at 2,000 iterations (RIVER R1);
- Q4 terminal sums;
- the v2-8B blueprint (BP-8B ADOPTED 25 Oct).

The exam measures that full config against the same outside anchor. As before, it is reported next to PR and
changes no gate.

## Known
As in `records/slumbot/PREREG.md` "Known". The API, blinds 50/100, 200bb stacks, the action grammar and
anonymous play are unchanged as of the 5 Oct run. The registered mock and the pilot recheck them.

## Bot (fixed at registration: the runner's constants at the registering commit)
The bot is `scripts/slumbot_run.py` exactly as committed at registration. Every constant it records in
`run.json` (commit, engine and blueprint sha256, search, seed, N) is the bot.

At registration (the registering commit; `run.json` records it):
- blueprint v2-8B (`runs/bp-8b-stage2/blueprint.bin`, sha256 fd27fe50...c7ae071; B1), translation `snap`;
- net N0 (B3); seed 20261024 (B4); N 50,000 (P3);
- flop search at 500 iterations (final net, gamma 5, sym leaves);
- turn search at 400 iterations (shift1_g5);
- river search at 2,000 iterations;
- `SubgameGpu(terminals64=True)` (Q4), `rescue=False`.

- **B1, the blueprint: closed 25 Oct, v2-8B.** BP-8B ADOPTED it (`records/bp-8b/RESULT.md`; the runner at
  6032470, its mock GATES). Before that it read: it follows BP-8B's registered Adoption (`records/bp-8b/PREREG.md`):
  - v2-8B iff BP-8B adopts it, else v2-2B;
  - v2-S3 only by its own amendment (it is not in the match).
  
  Adoption is a runner change made under BP-8B's rules, not here.
- **B2, the preflop grid: closed 21 Oct.** The current grid. F1 ran on the user's yes and failed FINDING's rule
  (`records/preflop-open/RESULT-F1.md`), so nothing waits for it. Before that it read: the current grid unless
  the user starts F1 and F1 passes FINDING's rule before registration (`records/preflop-open/FINDING.md`,
  `records/rm2/SCOUT.md`).
  - Slumbot is not F1's criterion, and its result cannot confirm F1: the scout's point estimates (~1–4
    bb/100, intervals that include zero) are far below the exam's precision.
- **B3, the net: closed 25 Oct, N0** (D2's final net, `runs/turn-net/final-w2048-d5-e200`; the user's yes to register).
  No D1b-trained net exists or is gated (D1b's data was ACCEPTED 22 Oct; a net trained on it needs its own gate).
  The assistant's view: do not wait for one.
- **B4, the seed: closed 25 Oct, 20261024** (the user's okay). The runner's SEED was 20261005, the 5 Oct run's; the
  one-constant runner change is committed before registration. It seeds our action sampling, the mapping of Slumbot's
  off-grid bets onto our grid (`engine/src/play.rs`), and the mock's dealer; Slumbot deals the cards.
- **Guards.** Unchanged from the runner:
  - a failed search stops the run;
  - uncommitted play code is refused;
  - a change of play code between sessions is refused (unless `--accept-change`, recorded);
  - lost hands above 0.5% of N stop the run.

## Protocol
As in `records/slumbot/PREREG.md` "Protocol":
- anonymous; one client and one request in flight;
- append-only JSONL;
- the parser check against `sample_api.py`.

The changes:
- **Registered mock first.** A 1,000-hand mock run (`slumbot_run.py --mock`) at the registering commit. It
  must pass every bar `mock_check.json` checks:
  - chips conserved, every action legal, nothing lost;
  - flop p95 ≤ 3.0 s, turn and river p95 ≤ 2.0 s each, each street on its own.

  Its check is committed as the runner's GATES file before the first live hand. A failed bar means no live
  run, and the failure goes to the user. BP-8B adopted v2-8B, and its Adoption's own registered mock became
  GATES (25 Oct); this mock, at the registering commit, then replaces it.
- **GPU quiet.** Nothing else runs on the PC's GPU from the mock to the last counted hand. That includes the
  BP-8B match, D1b and any rating. `gpu_clients.jsonl` is reported.
  - **P1, closed 25 Oct (the user's okay):** CPU-only jobs at BelowNormal (GPU hidden) may run beside it, at most
    4 threads; the flop bar is close (BP-8B's mock: 2.753 s against 3.0 s).
- **Pilot.** The first 100 live hands are discarded. They project the run's hours.
- **Hours and `--go`.**
  - Expected ~5.6 s per hand, ~78 h for 50,000 hands. That is an estimate: HANDOFF Roadmap 4's 5.46 s/hand
    was built on B1's mock (2.17 s/hand); R1's mock ran 2.34; BP-8B's mock (v2-8B), today's GATES, ran 2.21.
  - The runner pauses any projection over `LIMIT_HOURS` = 60 for the user's go.
  - **P2, closed 25 Oct (the user's okay): a ceiling of 100 h.** The assistant resumes with `--go` iff the pilot projects
    at most 100 h. Above it, the run waits for the user.
- **N. P3, closed 25 Oct (the user's okay): N = 50,000** (the runner's N; the 5 Oct run's).
  - The 5 Oct sd was 15.35 bb per hand, so 50,000 hands give about ±13.5 bb/100.
  - 100,000 hands give ±9.5, at ~156 h.
  - Whatever is chosen is fixed: no stop or extension depends on the running result.
  - The assistant's view, accepted: 50,000. The exam is an anchor, not a test of any one change, and ±13.5 against ±9.5 does
    not justify another ~3 days of GPU.
- **Lost hands** are counted and excluded; above 0.5% the run stops (unchanged). A restart: Amendment 1.

## Reading
**Primary**
- Our win rate in bb/100 over the N counted hands, from `winnings`, with a 95% normal interval on the
  per-hand values.
- WIN if the lower bound is above 0, LOSE if the upper bound is below 0, else UNRESOLVED.
  (`scripts/slumbot_report.py`; its run checks amended by Amendment 1, the reading unchanged.)

**Also reported, as in the 5 Oct run**
- the win rate by seat;
- the all-in-adjusted win rate (descriptive);
- seconds per hand, and the searched decisions' latency by street;
- Slumbot's "Baseline" (not registered).

**New, descriptive only** (none decides anything)
- **Against the 5 Oct run.** The difference of the two win rates, with the interval of a difference of
  independent means (about ±19 bb/100 at N = 50,000 each).
  - The two runs differ in more than one thing, so this does not attribute a change to any one part.
- **The big blind against Slumbot's 2bb open.**
  - `scripts/rm2_scout.py` on the new hands: fold rate, the folded hands' equity against Slumbot's real
    hands, and the check-down gain.
  - It is compared with the 5 Oct figures in `records/rm2/scout.json`.
- **Flop decisions facing an off-grid bet:** their share and their latency.

**Not a claim of unexploitability.** Slumbot is one opponent; a result measures us against it.

## Before registration (the checklist)
1. The BP-8B match has read; B1 follows its Adoption. Done 25 Oct: ADOPTED, B1 = v2-8B.
2. B2 (F1): closed 21 Oct. B4 (seed 20261024), P1 (CPU beside, BelowNormal, at most 4 threads), P2 (100 h) and P3
   (50,000): closed 25 Oct on the user's okay. B3 (N0): closed by the user's yes to register.
3. The runner at the registering commit is the bot. Any runner change is committed and audited first:
   B1's blueprint (6032470, 25 Oct) and B4's seed (25 Oct) are.
4. This file is renamed PREREG.md and stamped with the user's yes. A read-only audit is done before the commit.
   Done 25 Oct (the registering commit; the runner's `run.json` prereg list now names this file).
5. The registered mock passes and its check is committed as GATES. Only then the first live hand.

## Amendment 1: a restart (2026-10-25 15:25, before the first live hand)
The user's yes, 25 Oct: "yes, add the reboot amendment (resume after a restart, exclude cut hands, allow uneven
seats), update the report script, then start the live run". It answers the registering audit's open point: the
live run takes ~78-100 h, so a reboot (a Windows Update restart, a power cut), a crash or a kill is likely. Before
this, the runner already recorded a cut hand as lost and resumed, but `scripts/slumbot_report.py` refused any lost
hand and uneven seats, so one reboot would have voided the run.
- **A restart is not a failure.** A restart here is a session that ended without an `end` event (a reboot, a
  power cut, a kill, a crash outside a hand), or whose `end` message says "interrupted". The assistant resumes with
  `slumbot_run.py --dir <run>` at the same commit, never with `--accept-change`. A fault the runner records in an
  `end` event (a failed search, a protocol error, a payoff mismatch, an error inside a hand, lost hands over the
  limit) is not a restart. A resumed session is a new Slumbot session, which starts in the big blind. A go given
  after the pilot holds for every later session.
- **Cut and lost hands are excluded.** A hand cut by a restart, or lost to the network, is recorded as lost and
  excluded from every number. The run plays on until N = 50,000 counted hands have finished. The 0.5% stop
  (250 lost counted hands) is unchanged. A lost pilot hand counts toward the 100-hand pilot (the runner's rule).
- **Uneven seats are allowed** and reported. Each new session starts in the big blind, so a restart can leave
  one seat a few hands ahead. The by-seat rates are read as before.
- **The report's checks, amended** (`scripts/slumbot_report.py`; `tests/test_slumbot_report.py`):
  - N finished counted hands; lost counted hands at most the run's `lost_limit`;
  - hand numbers contiguous, the pilot first, every finished hand's payoff check "ok", the run done (unchanged);
  - Slumbot's running totals are chained hand by hand within each session: each finished hand's
    `session_total`, `session_baseline_total` and `session_num_hands` advance by its own winnings, baseline
    and one. This replaces the per-session sums, which a lost hand breaks.
  - After a lost hand the chain restarts, because Slumbot may have counted that hand. The next hand is reported
    as unchained: only Slumbot's hand count is checked (one more than before, plus at most the lost hands
    between). Within a session, a hand whose totals restart from zero (a new token, which
    Slumbot's `sample_api.py` says can come after a long pause) is checked from zero and reported.
  - A lost hand whose winnings disagreed with the cards (killed after the mismatch was recorded) still fails.
  - A line cut by a power loss in the run's JSONL is skipped, as the runner does (its hand is logged again), and
    reported.
  - Seconds per hand sum each session's own playing time: from its first counted hand's start to its `end`
    event, or, for a session cut without one, to its last hand's start plus its seconds (a cut hand has none).
    The time between sessions is not counted.
  - On the 5 Oct run (no lost hand, one session) every number is the same as before.
- **GATES stands.** The report is not play code (the runner's `PLAY_CODE` is `src` (less
  `src/parciver/dashboard`), `engine` and `scripts/slumbot_run.py`), so the registered mock at 1169e6e still
  covers the live run.
- **Unchanged:** any other stop still goes to the user. That means a failed search, a protocol error, a payoff
  mismatch, more than 250 lost hands, or a play-code change. The 100 h ceiling is checked once, by the pilot's
  projection; restarts do not reopen it.
