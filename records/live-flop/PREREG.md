# LIVE-FLOP: flop search in the live config (PRE-REGISTERED 2026-10-11 20:19, before any data of it)

The user's decisions, 11 Oct (HANDOFF Next 1):
- ~20:12: option A, "raise the flop bar" (no kernel work first).
- 20:19: the bar's value, "yes sure" to the assistant's proposal of **p95 <= 3.0 s for flop decisions**.

At registration (20:19) no hand had been played with this configuration. A 20-hand development mock followed
(disclosed under "Tests"). The registered mock run is the gate's first data. This file, the LIVESEARCH amendment
and the code were written 20:19-20:40 around that development mock and its audit; nothing in "The bar" or "The
gate" moved after its numbers.

## Why
- D4 ADOPTED flop search (FLOPSEARCH "M2 result": `from_flop=True`, 500 iterations, gamma 5, the final net
  `runs/turn-net/final-w2048-d5-e200/net.pt`). M1: it removes 63% of the blueprint's flop loss.
- It was not live because its latency is over LIVESEARCH's p95 <= 2 s.
- That 2 s bar was set on 30 Sep for **turn** decisions, the choice left to us. Its reason was "well inside
  online action clocks, fast-fold included, and leaves room for the flop and river searches in the same hand".
- Measured latency per flop solve (`FlopSearch.seconds`, the solve alone inside the GPU lock, agent against
  agent on the grid's sizes):
  - 11 Oct, E93347A5 (`runs/flop-eval/latency-final-e93347a5.json`; FlopSearch's default float32 turn solver,
    so Q4 is not exercised): median 2.29 s, p95 2.43 s over 53 solves (max 3.00 s).
  - M2's run, 4 threads, C879EB24 (14,510 solves): median 2.34 s, p95 2.60 s.
- No precision-neutral speed work is near: KERNELS N1 + N1b did not land (1.90-1.97x slower). Triton scouting
  gives ~1.08x with moderate fusion, and 2 s would need >= 1.22x.

## The bar (amends LIVESEARCH's latency budget for flop decisions only)
- **Flop decisions: p95 <= 3.0 s**, measured as the gate below measures it (the whole decision).
  - It sits above M2's 2.60 s. A bar set just above today's 2.43 s would be post hoc and could fail a
    re-measure.
  - The margin is smaller than those solve-only numbers suggest. The gate times the whole `options_failure`
    call, though the second development mock shows that adds only a few ms over the solve. The bigger
    difference is the spots: the mock opponent bets every legal size, so our flop decisions often face
    off-grid bets. The first development mock (n = 17) put on-grid flop decisions at ~2.2-2.3 s and those
    facing off-grid bets at ~2.6-2.7 s; its p95 was 2.77 s.
  - Online action clocks are typically 15 s or more. The 5-7 Oct Slumbot run had a 10.6 s maximum searched
    decision (turn and river search only) without a problem.
- Turn and river decisions: unchanged (p95 <= 2 s).
- The cost of the 3.0 s bar: a hand that searches on the flop takes ~2.3 s longer. M2 ran 3.81 s/deal against
  0.49 s/deal for turn search alone (`search-h2h-turn-gpu.json`).

## The change (`scripts/slumbot_run.py`, the live runner; nothing else changes)
- `SEARCH`: adds `from_flop=True, flop_iters=500`. Turn 100 and river 200 iterations and `rescue=False` stay.
- Solver: `Watched(FlopSearch(net, gamma=5, turn=SubgameGpu(terminals64=True), leaves="dense"))`.
  - Flop jobs: FlopSearch with M2's leaves (dense).
  - Turn and river jobs: the live solver as before (Q4, the user's call 11 Oct).
  - The combination is new; each part is adopted on its own (M2 ran float32 terminals at the turn).
  - Other differences from M2, none of which changes a value: one hand at a time (M2: 4 threads), FlopSearch's
    cache of 4 flops (M2: 8), and the opponent (the mock's random legal sizes, Slumbot live; M2: agent against
    agent).
- `run.json` records the net's path and sha256. A resume refuses another net or a run without flop search.
- The registered gate moves to `records/live-flop/mock_check.json`. `records/slumbot/` stays the 5-7 Oct
  run's record, untouched.
- `mock_check` adds:
  - the net's sha256;
  - the flop decisions' latency: decisions on street 1 with at least one solve, each one's `seconds` (the
    whole `options_failure` call);
  - a check that fails when there is no searched flop decision, or when their p95 (`numpy.percentile`,
    linear) is over 3.0 s;
  - the turn and river searched decisions' p95, median and max, reported and not gated.
- Each decision's record adds `solve_seconds`, the solver's calls alone (`Watched`'s timing), so the whole
  call and the solve can be told apart. Descriptive only.

## The gate (before any live hand, as PREREG's checks)
- The registered mock run, 1,000 hands (`slumbot_run.py --mock`), at the committed play code.
- All of the 5 Oct PREREG's checks: parser check, chips conserved, every action legal, none refused, no
  winnings mismatch, no hand lost. A failed search stops the run, as before.
- **Plus: the flop decisions' p95 <= 3.0 s.** The median, mean, max and count are reported.
- Run conditions: nothing else heavy on the GPU (no other compute job); the desktop's own apps stay as they
  are.
- **Pass:** `mock_check.json` says passed. Copied to and committed as `records/live-flop/mock_check.json`, it
  lets a live run start (`gates_refuse`: the same checks as before plus the net's sha256). This registration
  starts no live run: that is HANDOFF Next 4, with its own registration.
- **Fail:** any failed check of `mock_check`, or the run stopping on its own before 1,000 hands (a failed
  search, a lost hand, a protocol error or a winnings mismatch). Stop and report to the user; the bar is not
  moved after the data. Any re-run after a fail needs the user's go and a new entry here.
- **VOID** (not a fail; its cause recorded, fixed, and the mock run repeated): only a process crash (an
  exception outside the runner's own stops), a kill, a reboot, or another compute job found on the GPU during
  the run. A stall is no new hand for 10 minutes; it is stopped and is VOID.

## Tests before the mock run
- Flop tests (`tests/test_flop_search.py`) and the selftests pass.
- A short development mock (`--mock --hands 20 --allow-dirty`, not the registered run) exercises the path
  before the commit. Its numbers are not the gate.
  - Disclosed: it ran 11 Oct 20:21-20:22 (`runs/slumbot/mock-20261011T122111Z`). 20 hands, 0 lost, chips
    conserved, every history legal; 17 searched flop decisions, median 2.33 s, p95 2.77 s, max 2.98 s. Nothing
    above was changed after it.
  - Disclosed: a second development mock after the audit's fixes (`solve_seconds`, the turn/river report), 11
    Oct 20:29 (`runs/slumbot/mock-20261011T122900Z`), 6 hands, to check the new fields. 3 flop decisions
    (2.83, 2.66, 2.29 s); each decision's whole call exceeded its `solve_seconds` by 1-13 ms.

## Result (11 Oct 21:06; commit 409586d, clean play code (run.json `dirty: null`); run `runs/slumbot/mock-20261011T123050Z`) — **PASSED**
- The registered mock run: 20:31:10 (run.json `started`; folder stamp 20:30:50) - 21:06:37, 1,000 hands, 0 lost, chips conserved (net +585,307 =
  -bot_net), every history legal, none refused, 1,000 winnings checked, 0 mismatches; parser check passed
  (214,221 prefixes agree). Engine E93347A5, blueprint 1976A53C…, net A8BF00BE…. Just before the start
  (nvidia-smi at 20:30:46, seen in the session, not stored): GPU 8% busy, 587 MiB in use, no other compute job.
  Nothing logged the GPU during the run; no other job of ours ran on it.
- **Flop decisions (whole call): n 772, median 2.359 s, p95 2.872 s <= 3.0 s, mean 2.435, max 2.933.**
- Turn and river searched decisions (reported, not gated): n 868, median 0.259 s, p95 0.367 s, max 3.743 s
  (a turn decision after calling the mock's off-grid 14,624 flop bet); 3 turn/river decisions over 2 s (3.743,
  2.621, 2.403 s).
- 2.10 s/hand over the mock's mix (the 5 Oct mock without flop search: 0.445 s/hand).
- `records/live-flop/mock_check.json` is the run's `mock_check.json`, copied and committed: the live config's
  gate for HANDOFF Next 4 (`gates_refuse` checks it with the engine, blueprint and net shas and the play code).
- Descriptive (not registered): each flop call exceeded its `solve_seconds` by a median 2.2 ms (max 7.8 ms);
  turn/river calls by 12.0 ms (max 65.1 ms). First flop decisions of a hand (which build the flop's tables):
  not facing a bet n 437, median 2.354 s / p95 2.370 s; facing a bet n 153, 2.863 / 2.889 s. So the tail is
  decisions facing a bet (a bigger subgame), ~0.11 s under the bar; not the table build or the call. Slumbot's
  bets are often off the grid: Next 4's registration should expect that.
- The interim count and flop latency were looked at once (hand 870, 21:01; p95 2.87 s then); nothing was changed
  and the run was not stopped.
