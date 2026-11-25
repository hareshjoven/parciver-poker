# RATING-LIVE V: AIVAT-style variance reduction for rating the live bot (Next 2)

DRAFT written 2026-10-11 23:09. It becomes pre-registered at its commit, before any data from the
estimator below and before any rating match of a search configuration. The user chose "variance reduction
first" for Next 2 (11 Oct): build and validate the estimator on cheap CPU matches, then rate the live bot with
fewer GPU hours. Nothing here runs beside the FLOP-SYM M1 oracle except the build and the Rust tests (2 cores,
BelowNormal); the CPU measurements wait until the oracle ends.

## Known (before this step)
- The live bot's rating needs only one GPU match: PR(live) = 1500 + u(live, A2) - u(A0, A2), because pool-v2's
  Nash weights are all on A2 (RATING.md Amendment 2). u(A0, A2) is A0's 20,000,000-deal reference row (sd 9,547
  mbb/hand, its noise ~2 points); the GPU match's noise is the whole interval.
- The closest measured pair: FLOPSEARCH M2 (flop + turn search vs turn search on v2-2B, 200bb, 10,000 mirrored
  deals, `runs/flop-eval/m2.json`): per-deal sd 17.0 bb, **66.9% of deals exactly 0** (both seatings played
  alike), 3.81 s/deal. A rating at that sd: about +/-53 points at 100,000 deals, ~106 GPU hours.
- VARRED (2 Oct): the chance terms alone (board luck after the arms diverge) removed ~1/3 of the variance on
  blueprint pairs (r per deal 0.680, 95% [0.646, 0.714]) and nothing on a river-search pair (the arms diverge
  after the last card). For flop search only the turn and the river can be removed. VARRED2 (2 Oct, a draft,
  never built) proposed AIVAT's action terms as the remaining lever.
- Checked in the code now (engine/src/play.rs, subgame.rs): every agent action, blueprint or search, is drawn
  from `Decision::options` with its own uniform `unit(seed, stream, decision index)` (the blueprint on
  STREAM_SAMPLE, the search on STREAM_SEARCH; the translation of the opponent's bets on STREAM_TRANSLATE). So
  given the history and the translation draws, the action's distribution is `options`, and the draws of
  different decisions are independent. A fallback (missing key) is one action with probability 1.
- Exception (the audit, 11 Oct): a search's `options` is the solver's row as it came. The GPU solvers work in
  float32, so a row's sum misses 1 by ~1e-7. The draw then gives the missing mass to the last action, or cuts
  the tail; a last action with p == 0 can so be drawn (chance ~1e-7 per such decision). `options` is not
  changed: it also feeds range tracking, and changing it would change the GPU path's play.

## The estimator (AIVAT: Burch, Schmid, Moravcik, Morrill, Bowling 2018)
For bot A in one hand, with V(s) = A's value of state s for the actual hole pair:
- not terminal: sign * (2 eq0(board(s)) - 1) * m(s), eq0 seat 0's exact all-in equity against seat 1's hole
  (ties half), m(s) = min(committed_0, committed_1), sign +1 if A sits in seat 0 else -1 (VARRED's v);
- terminal: A's exact net (folds included). At a showdown both formulas agree (tested).

  u_aivat = u_A - sum over board deals [V(after) - V(before)] - sum over decisions [V(s + a_chosen) - sum_a pi(a) V(s + a)]

pi = the acting bot's `options` at that decision, divided by its sum (both bots are ours; both seats' terms
are used).
- Unbiased for any policies, search included: each chance term has mean 0 given the past (equity is a
  martingale; VARRED), and each action term has mean 0 given the past (a ~ pi, independent draw, above), for
  ANY fixed V. So a poor V costs variance, never bias.
- Bias bound for unnormalized rows: pi differs from the real draw by at most 2|1 - sum| in total, so a term's
  mean is off by at most 2|1 - sum| * max|V| (max|V| <= the stack, 20,000 chips). The match reports the
  largest |1 - sum| met (`aivat_mass_error`) and errors above 1e-4. At ~1e-7: <= 0.004 chips per decision,
  ~0.04 mbb per searched decision, against a rating SE of tens of points.
- V depends only on fold vs continue here: a check or a bet leaves m where it was; a call or a raise lifts it
  to the bet faced. So the action terms remove the luck of mixed fold/continue choices, not of bet sizes or
  bluffs. Fixed now: no other V in this step (a solver-valued V is a later step if this one pays).
- Coefficient 1, no fitted beta (a fitted beta would be biased).
- Self-play (one agent both sides): the mirrored seatings take the same terms with opposite signs, so a deal
  stays exactly 0. Two different bots that draw the same actions do not cancel where their policies differ (a
  seat's term uses the policy of the bot sitting there): deals that are exactly 0 in raw u (66.9% in M2) can be
  non-zero in u_aivat. Not a bias; P reports both zero shares.
- Action terms only for `Bot::Agent` on both sides (a perturbed or "realistic" bot changes its action after the
  draw): any other bot with `aivat=True` is an error.

## Implementation
- `match_against(..., aivat=False)`: with `aivat=True` the result also has `per_deal_bb_aivat` (chance + action
  terms), `per_deal_bb_cv` (chance terms only, as `luck=True`) and `aivat_mass_error`. Off by default: every
  existing call gives the same bytes and does the same work. Real 52-card games only, as `luck`.
- Cost: the preflop equity (1.7M boards, ~22 ms/deal on 4 threads, VARRED) for every deal; nothing next to a
  GPU deal (~3.8 s).
- The equities are VARRED's per-deal cache (one enumeration per board length, shared by the two seatings).

## Tests (all must pass before any measurement)
1. `per_deal_bb` and `per_deal_bb_cv` are bit-identical with `aivat` on and off.
2. Mirrored self-play (one agent both sides): every deal's `per_deal_bb_aivat` is 0.
3. At every decision with `aivat`, the policy's sum is within 1e-4 of 1 and the action played is one of its
   actions (p == 0 allowed: the case above, inside the bias bound); otherwise the match is an error (in the
   code, so in every measurement too). An external solver whose rows are scaled by 1 + 1e-7 or 1 - 1e-7 runs
   and reports that miss; by 1 + 1e-3, an error.
4. On 300 random no-fold lines to a showdown: V(terminal) = the payoff, the formula (sign * edge * m) gives the
   payoff too, and so does V of the state before a river check that ends the hand (not terminal); all-in
   run-outs included.
5. A deterministic policy (one-hot options) has zero action terms: `per_deal_bb_aivat` = `per_deal_bb_cv`.
6. Statistical: (a) for fixed cards, over many hand seeds (the draws), the action terms average 0 within 4 SE,
   for a uniform agent and a CPU river-searching agent (whose draws must include searched decisions), each
   against a maniac-leaked uniform agent (Rust); (b) Python, a uniform agent against an empty strategy (the
   fallback), blinds 50/100, stacks 10,000, seed 722000, 20,000 deals, 4 threads:
   mean(u_aivat - u) within 3 SE of 0 (`test6.py`; a 300-deal version is in the suite).
7. An error for a non-agent bot with `aivat=True` (checked before any deal); `aivat=False` returns no
   `per_deal_bb_aivat`.

## Measurements
- V1 and V2 (`measure.py`): blinds 50/100, stacks 20,000 (the rating's stakes), 4 threads, `aivat=True`, in
  chunks via `first_deal` (identical to one call); a resume is refused unless the configuration and the loaded
  engine binary's sha256 match.
- **V1 (CPU, after the oracle; gating on bias):** A2 against A0 (the reference pairing), seed 720000, deals
  0..199,999 (chunks of 20,000): z = mean(u_aivat - u) / SE (deal-level) and r = Var(u_aivat) / Var(u)
  per deal with a deal-bootstrap 95% CI (seed 0, 2,000 resamples). Also r for the chance terms alone.
- **V2 (CPU; gating on bias, r descriptive):** v2-2B with CPU turn search (turn 100, river 200 iterations)
  against v2-2B, seed 721000, 2,000 deals (chunks of 250): |z| < 3 (the search's draws), and r (the
  same-blueprint shape: the arms diverge on the turn).
- **P (GPU pilot, after FLOP-SYM's speed check and mock):** the live configuration at the time against A2
  (v2-2B blueprint), seed 730000, 2,000 deals (~2.1 h): r with its bootstrap CI, z, `aivat_mass_error`, the
  zero-deal shares of u and u_aivat, s/deal. Pilot deals are never part of a rating.

## Decision rule
- ADOPT `per_deal_bb_aivat` as the live rating's primary per-deal value iff tests 1-7 pass, |z| < 3 on V1, V2
  and P (P is the only bias check on the GPU path), P's `aivat_mass_error` <= 1e-5, and P's r has a 95% CI
  upper end <= 0.9. Raw u is always reported beside it.
- Otherwise the rating uses raw u, and the result is reported.
- Either way the rating run itself is fixed in a RATING.md Amendment 3 before its data: the configuration
  (fingerprint), D sized from P's sd for a stated half-width, Amendment 2 (8)'s refusal lifted only for the GPU
  solver path, the estimator mean(x[:D]) - mean(ref[:R]) with x the adopted per-deal value and ref A0's raw
  reference row (unbiased: E[u_aivat] = E[u]), and the replay rule for a search receipt. The aivat array moves
  continuously with the solver's probabilities, so GPU float nondeterminism could break a bit-identical
  replay of it even where raw u replays exactly: Amendment 3 must either show the GPU path is deterministic
  (two runs of the same chunk, bit-identical) or fix a tolerance for the aivat array's replay.

## Result (12 Oct)
- Tests 1-7 pass (arena tests 12/12 and the suite at b6a3c7e; test 6(b) z -0.22, `test6.json`).
- V1 (4582d94): z -1.04; r 0.507 [0.492, 0.522]; mass error 3e-16.
- V2 (0931077): z -0.29; r 0.378 [0.268, 0.550]; mass error 4e-16.
- P (bbc8c2d; 15:11-17:07, `p.json`, `p.log`): the live configuration (sym flop leaves, Q4) against A2,
  2,000 deals, no chunk refused, 2,937 flop solves. z +0.05 (mean(u_aivat - u) +0.012 bb, SE 0.255);
  r 0.491, 95% CI [0.352, 0.673] (board terms alone 0.872 [0.745, 1.014]); `aivat_mass_error` 1.7e-7;
  zero deals 59.8% raw, 40.4% after AIVAT; sd 16.9 bb raw, 11.9 after; 3.47 s/deal.
- **ADOPTED:** `per_deal_bb_aivat` is the live rating's primary per-deal value (raw u reported beside it).
  At r ~ 0.49 a rating needs about half the deals of raw u for the same half-width. Amendment 3 next.
