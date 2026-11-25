# Preflop opens below our grid: finding, and a pre-registered plan for the next version

Written 6 Oct 2026, 00:09. The user asked for it on 5 Oct, adding that we must not only get better
against Slumbot. Nothing here changes the Slumbot live run: its bot is fixed until N = 50,000. Every
measurement below is general, and none of them uses Slumbot.

## The finding
- **Grid.** The 200bb grid (`NlheConfig::deep_200bb`, engine/src/nlhe.rs) has one first raise preflop:
  pot-size, so the small blind opens to **3bb**. After that come pot-size 3-, 4- and 5-bets (9 / 27 / 81bb),
  then all-in. The 100bb grid opens 3bb too.
- **Translation.** A real raise smaller than the smallest grid size is read as that size, under every
  translation mode (`Agent::translate_dist` in engine/src/play.rs: `x <= cands[0]` gives `Exact`;
  TRANSLATE.md: "Sizes below the smallest grid size snap up"). An open to 2bb (a 0.5-pot raise) or
  2.5bb (0.75 pot) is therefore read as 3bb.
- **Cost.** The big blind defends as if the open were 3bb, so it folds hands that the real price makes
  profitable to play.

  | open | equity the BB needs to call | minimum defence 0.75/(1 + x) |
  |---|---|---|
  | 2bb (x = 0.5) | 25% (call 1bb into a 4bb final pot) | 50% |
  | 2.5bb (x = 0.75) | 30% | 42.9% |
  | 3bb (x = 1) | 33% | 37.5% |

- **Why this is general, not Slumbot's.** 2–2.5bb opens are common heads-up sizes, so many strong
  opponents open below our grid.
  - Postflop the problem is smaller. Wherever search runs, the opponent's real size is added to the
    subgame's menu (engine/src/subgame.rs, "its real size added when off the menu").
  - The flop has no search until D3. Preflop never has search.
- **Our metrics do not measure it.**
  - LBR plays preflop only by checking or calling (engine/src/lbr.rs), so no exploitability number we
    have ever tests preflop sizing.
  - The rating pool (RATING.md: A0, A1, A2, Ln, Ls, Lm) has only bots that open 3bb.
  - Three scorecard rows do open below the grid: `min_raiser` (a builtin that min-raises, so it opens
    2bb with every hand), `baseline_sizes_x0.6` (opens 2.2bb with our ranges) and
    `baseline_realistic_sizes` (opens 2.75, 3 or 3.5bb, a third each; only the 2.75bb third is
    below the grid). None of those opponents exploits
    anything, so the rows report results against them, not what the misreading costs against an
    opponent that exploits it. A fix changes these rows, so they serve as guards (rule (d)).
- **What surfaced it (interim, descriptive; it decides nothing).** Slumbot run, counted hands up to hand
  5,772:
  - Slumbot raised to 2bb in 2,560 of its 2,836 small-blind hands and folded the other 276. It never
    limped and never opened to another size.
  - Our big blind folded 39.3%, called 45.9% and raised 14.7%, so it defended 60.7% against 2bb, which
    it read as 3bb.
  - Hands that ended preflop: −10.9 bb/100 (95% −17.2 to −4.6). This slice is selected on the outcome
    (a hand ends preflop because of the actions taken), so it is not an estimate of the leak's cost.
    The whole run was −13.3 bb/100 (95% −52.8 to +26.1).

## Pre-registered plan (fixed before any measurement; Slumbot is not a criterion anywhere)
Revised 6 Oct ~00:25 after a read-only audit of the first version (af88e9b), before any
measurement. The changes: the defence arithmetic, peek-free LBR, a paired fix-versus-v2 statistic, a
range opener as a guard against over-defence, (b) and (c) holding by construction for F2, and a
deal count we can afford. Revised again ~00:30 after a second read-only audit: peek-free LBR
refuses search, the live-config cost, `bp` defined exactly, and the adoption sentence.

### Measurement: an LBR that opens off the grid
- **LBR-open(s, opener).** LBR in the small blind replaces its first preflop action. Later preflop
  decisions (facing a 3-bet) are standard LBR (call), and from the flop on it plays standard LBR. LBR in
  the big blind is unchanged. A fixed preflop policy followed by LBR is a legal strategy, so it gives a
  valid lower bound on exploitability, like LBR itself.
  - opener `all`: raise to s with every hand;
  - opener `bp`: sample v2-2B's blueprint root action for the hand (on LBR's own seed stream); where it
    is the 3bb open (`Bet(1.0)`), raise to s instead; fold, limp and all-in are kept. Its root
    frequencies (fold / limp / open / all-in) are read from the blueprint and recorded before any
    measurement. If it opens nearly every hand, `bp` is close to `all` and (a′) guards little; that is
    reported with the result.
- **Sizes:** s ∈ {2bb, 2.5bb, 3bb}. 3bb is on the grid and serves as the control.
- **Peek-free in every no-search arm** (`peek = false`, engine/src/lbr.rs). F2 randomizes the
  reading, and a peeking LBR would see which way it fell, which no real opponent can. With `Exact`
  readings the peek-free LBR must equal the peeking one (test). Peek-free LBR refuses search
  (`options_many_forced` in engine/src/play.rs: "forced translations are not supported where the
  agent searches"; V2LEAK.md), so the live-config confirmation uses the peek.
- **Bots, deals, cost:**
  - decision: v2-2B without search (translation `snap`), and the same with each fix; 100,000 deals per
    arm (V2LEAK Stage 2b ran peek-free no-search LBR at that count), paired per deal on the scorecard's
    LBR stream (seed 410101), 95% CI as in TRANSLATE.md;
  - confirmation (descriptive; it vetoes nothing): the live config (turn search, `terminals64`), with
    the peek. LBR on v2-2B with `terminals64` costs 3.85 s/deal (VERIFY.md, arm F), so it runs s = 2bb,
    opener `all`, 5,000 deals, v2-2B against the fix (~11 h of GPU, after the Slumbot run). The peek
    sees how F2 read each open, so this arm is biased against F2. The preflop reading does not depend
    on search (search starts on the turn).
- **Statistics:**
  - the fix's effect (decides): Δ(s, opener) = LBR-open_fix(s, opener) − LBR-open_v2(s, opener), in
    bb/100, paired per deal;
  - the reading's price (descriptive): LBR-open(s) − LBR-open(3bb) for one bot. This mixes the
    misreading with the smaller pot, so it decides nothing.
- **Engine.** The engine work is built and tested in a scratch build. It is installed into the venv
  only after the Slumbot run ends, since the run's engine must not change (its resume refuses another
  engine).
  - Tests:
    - the default LBR (check/call preflop) stays bit-identical, and its receipts reproduce;
    - after a 2bb or 2.5bb open the agent's reading is `Exact(3bb open)` without the fix (every
      translation mode), and the F2 mixture with it;
    - LBR-open(3bb) gives `Exact` and is identical with and without F2;
    - peek-free LBR never reads F2's translation seed (the counterpart of lbr.rs's existing
      never-reads-the-seed test);
    - the opener, s and peek are recorded in the receipt.

### Candidate fixes, in order
- **F2, translation (cheap; tested first).** A preflop first raise below the smallest grid size is read
  pseudo-harmonically between the limp (x = 0) and the 3bb open (x = 1).
  - P(read as a limp) = (1 − x) / (1 + x): 1/3 for a 2bb open, 1/7 for a 2.5bb open.
  - Under a limp reading the big blind's abstract check is played as a real call; it never folds there.
    The mixture therefore defends P(limp) + (1 − P(limp)) · d₃, where d₃ is the defence at the 3bb
    reading. With d₃ = 60.7% (above) that is 73.8% against 2bb and 66.3% against 2.5bb. Both are above
    the minimum defence (50%, 42.9%). The mapping's 1/(1 + x) property is about postflop bets into a
    pot and does not carry over here, so nothing guarantees these are the right frequencies. Whether F2
    helps is for the measurement to decide.
  - Known side effect: a limp reading puts the hand in the limped-pot tree with a larger real pot.
    Search (turn and river now, the flop after D3) re-solves with the real pot, so the cost falls
    mostly on the flop without D3.
  - Postflop sizes are unchanged.
- **F1, grid (next blueprint).** A new 200bb grid adds preflop first-raise sizes of 0.5 and 0.75 pot (2bb
  and 2.5bb opens), preflop only.
  - It needs per-street raise sizes in `NlheConfig`, a new blueprint (days of CPU) and a new rating.
  - It goes ahead only if F2 fails the rule below, or the user chooses it anyway.

### Decision rule (per fix, against v2-2B as it plays now; paired; no-search arms)
ADOPT if all of these hold:
- (a) Δ(2bb, all) and Δ(2.5bb, all) both fall: the CI is entirely below 0. This is close to automatic
  for any fix that widens defence (v2 folds ~39% to an every-hand opener), and LBR-open calls every
  preflop 3-bet, so Δ also credits value-raising against a preflop caller. (a′) is the real guard;
- (a′) Δ(2bb, bp) and Δ(2.5bb, bp) are not UP (the CI is not entirely above 0). An every-hand opener
  rewards any wider defence, so (a) alone would pass a fix that defends too wide against a real range;
- (b) standard LBR, total, is not UP;
- (c) PR is not DOWN (RATING.md, against the current headline; rating.py refuses search candidates, so
  PR is the no-search config);
- (d) no scorecard row is DOWN.

For F2, (b) and (c) hold by construction: standard LBR never raises preflop, and every pool bot opens on
the grid, so F2 never fires. They are checked by a test (identical actions with and without F2 on the
LBR stream and on a pool sample), not by a run. For F1 they are run.

If any of (a)–(d) fails, do not adopt, and report which condition failed. If only (c) or (d) fails,
the trade-off goes to the user (as in TRANSLATE.md). The live-config confirmation decides nothing; if
its Δ is entirely above 0, that is reported to the user with the decision.
