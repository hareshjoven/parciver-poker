# SLUMBOT: an external anchor (PRE-REGISTERED 2026-10-05 03:08; drafted 2026-10-04 20:02)

Approved by the user on 5 Oct 03:05 ("okay for all", after the assistant proposed tying the search setting to V5's
reading). No hand has been played against Slumbot. V5 was running (arm D, deal ~1,500 of 40,000) and no V5
reading existed when the search rule below was fixed. Any amendment must be stamped and made before the first
live hand. The user said yes to Slumbot (4 Oct) as the next step after HANDOFF step 2.

## Why
The rating's PR is relative to our own pool: the Nash weight is all on A2, so PR is "vs A2". Slumbot won the
2018 Annual Computer Poker Competition and plays anyone through a public API. A win rate against it is an
anchor outside our pool. It is reported next to PR and does not change PR or any gate.

## Known (read from slumbot.com's API page and its `sample_api.py`, 4 Oct)
- Heads-up NLHE, blinds 50/100, 200 bb stacks (20,000 chips), reset every hand.
- HTTPS POST with JSON bodies to `slumbot.com/slumbot/api/{new_hand, act, login}`.
- Each response carries a token, and the next request uses the newest one.
- `client_pos` 0 means we are the big blind; a session starts in the big blind. Slumbot advises an even
  number of hands so each seat is played equally often.
- Actions: `k` check, `c` call, `f` fold, `b<n>`. The `n` counts chips put in on that street only. `/` ends a
  street; an all-in can leave later streets empty. `winnings` appears when the hand ends.
- Login is optional and only for the leaderboards. The leaderboards report "Raw" and "Baseline" bb/100. No
  rate limit is published.

## Bot (fixed when this is approved)
- v2-2B (`runs/blueprint-20260927-141645`, 200 bb), translation `snap`.
- Production live search, as the default `PlayAgent` at the commit the run records: today GPU turn search at
  100 iterations and river search at 200. If LIVESEARCH adopts B1 before the run, the run uses B1, and the
  record says which.
- The GPU solver's terminal sums follow VERIFY V5's reading (`records/verify/v5.json`, fixed here before it
  exists):
  - K2 NEGLIGIBLE: the default kernels, `SubgameGpu()` (V5's arm D).
  - K2 MATTERS: `SubgameGpu(terminals64=True)` (V5's arm F), whatever the default is then. V5 would have shown
    that the default kernels change real results, and this anchor measures the bot without that defect. It is
    ~4x slower per turn solve; the pilot's go/no-go rule below covers the time.
  - No V5 reading (V5 failed or was stopped): no counted hand until V5 has a reading.
- Decisions come from the engine's play layer (`PlayAgent.options`, the same layer the rating uses), in chips
  as Slumbot counts them (sb 50, bb 100, stacks 20,000). Actions are sampled with a seed the run records.
  One of our sizes rounds to the nearest chip.
- Prerequisite: Finding 16's engine half. `options` must report a failed search; today it drops the
  failure index. Every failed search is counted, and any count above 0 stops the run, as `search_h2h` does.
- The run refuses uncommitted `src/` or `engine/`. It records the commit, the engine's and the blueprint's
  sha256, and the search settings.

## Protocol
- Anonymous: no login and no account. No credential is entered by the assistant. If the user wants the leaderboard
  "Baseline" figure, the user logs in themselves; that is a separate run.
- One client and one request in flight, with no parallel sessions.
- Every request and response goes, as it happens, to an append-only JSONL file under `runs/slumbot/<UTC>/`.
  Each hand's record holds the action string, both seats' cards where shown, the board, `client_pos`,
  `winnings`, and our options and sampled action at each decision.
- Before any counted hand:
  - The client's action-string parser must agree with `sample_api.py`'s `ParseAction` on every prefix of
    10,000 random legal sequences from an offline mock server built from the rules above.
  - A mock-server run of 1,000 hands must end with chips conserved and every action legal.
- Pilot: the first 100 live hands are discarded whatever they show. They test the client and measure
  seconds per hand.
  - If the pilot projects more than 60 h for N hands, the user decides go or no-go. N does not change.
- Counted hands: N = 50,000 (even), played straight after the pilot in sessions of even length.
- N is fixed: no stop or extension depends on the running result. The dashboard may show it.
- Lost hands (a network error that a retry of the same request with the same token cannot resume):
  - They are counted and excluded.
  - Above 0.5% of hands, the run stops and is reported.
- Pause rule: the GPU work of HANDOFF step 2 (PREREG15, done 5 Oct 02:20; V5) comes first. The run uses the GPU
  only when that work is done, and after the engine rebuild that carries Finding 16's engine half.

## Reading
- Primary: our win rate in bb/100 over the N counted hands, from `winnings`. The CI is a 95% normal interval on
  the per-hand values.
  - WIN if the lower bound is above 0, LOSE if the upper bound is below 0, else UNRESOLVED.
- Also reported:
  - the win rate by seat;
  - the all-in-adjusted win rate (an all-in with both hands shown is scored at its equity over the remaining
    board), descriptive;
  - seconds per hand.
- Precision: with sigma about 10 bb per hand, N = 50,000 gives about +/- 9 bb/100. The pilot's sigma is
  reported, but N stays as fixed.
- Not a claim of unexploitability: Slumbot is one opponent, and a win against it measures it, not us.
