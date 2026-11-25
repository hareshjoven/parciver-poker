# FLOP-SYM: suit-symmetric flop leaves (PRE-REGISTERED 2026-10-11 21:19, df6df79, before any data of them)

The user's request (11 Oct ~21:10): improve performance and all the requirements (speed, precision,
strength). At registration no candidate had been implemented or run; the only numbers used were counts over
the 22,100 flops (CPU, no solver) and the existing M1 files.

**Revised ~21:35 after its read-only audit, before any M1 or speed data** (all 20 findings taken; the
changes: C1's multiplicity (z = 2.241 for both candidates), the oracle-hour comparison, left cosets, the
copy rule's estimand and provenance, the speed rule's ratio, table builds and cluster bootstrap, a VOID / bug
rule, and wording). Disclosed: between the two commits the code and tests were written and a development
strategy run of C2 on 3 spots of another seed (560000, `C:/tmp/flopsym-dev`, not M1's spots) checked the path:
24 of 36 flop solves took the symmetric path (the rest on the rainbow spot), and the engine accepted both
symmetric spots' X for the reduced game. No oracle value was computed.

## Why
- D4's flop search runs the final net (`runs/turn-net/final-w2048-d5-e200/net.pt`) at every leaf for each of
  the 49 turn cards: 441 rows per leaf call on the 9-leaf 200bb tree, two calls per iteration. The forward is
  84% of an iteration (KERNELS.md, Step 3).
- The net is not suit-symmetric. FLOPSEARCH M1: X_net's suit-isomorphic holdings differ by up to 1.0 in
  probability, so the CPU oracle could not use the reduced game: X_net's 27 symmetric spots took 15.4 h in
  the full game against bp's 10.1 h in the reduced one (23.4 h over all 40). The true leaf values are
  symmetric; the net's asymmetry is error.
- On 69% of flops (all but the 6,864 unpaired rainbow ones) the flop's own suit swaps (its stabilizer G) map turn cards onto each other. With ranges
  that G leaves unchanged, those turns' net inputs are relabellings of each other: one net row per turn
  orbit is enough, and the rest follow by relabelling. That is fewer rows (faster) and symmetric values.

## The candidates (`FlopSearch(leaves=...)`; "dense" stays the default and unchanged)
Notation: G = the suit permutations that map the flop onto itself; for a turn t, b = flop + t and H_b = the
suit permutations that map b onto itself; G_t = the members of G that also fix t (= G within H_b).
- **C2** (`leaves="sym"`): on a flop with G larger than the identity, and a job whose ranges and frozen
  strategies are exactly unchanged by G (checked per job):
  - one canonical turn t per G-orbit of turn cards: its lowest card id;
  - at t, the net is run on the leaf range x and on h.x for one member h of each other **left coset** hG_t of
    G_t in H_b (h.x = h'.x exactly when h' is in hG_t, since G_t leaves x unchanged); each output is relabelled
    back by h's inverse and the results averaged (equal weights). With the G-average below this is the
    average over all of H_b;
  - the canonical turns' contributions, each weighted by its orbit's size, are summed and averaged over G
    (exactly: the G-average is taken once per holding orbit and copied to its members, so equal holdings
    get bit-equal values);
  - the search's returned strategies are averaged over G the same way. In exact arithmetic the solve is
    G-symmetric; in float32 the fold and runout sums are not bit-symmetric and regret matching can amplify
    that (D4's drift study: up to ~1e-3), so the average removes float-level asymmetry, reported, and makes X
    exactly symmetric for the oracle.
  - Otherwise (G is the identity: the 6,864 unpaired rainbow flops, 31%; or a job not G-symmetric) the job
    takes the dense path unchanged, so its result is bit-identical to today's. Paired rainbow flops (1,872)
    have G larger than the identity and take C2.
  - The leaf values are then equivariant under each turn board's H_b (A(h.x) = h.A(x), the true values'
    symmetry; to float rounding) and bit-exactly symmetric under G.
- **C1** (`leaves="orbit"`): C2 without the H_b cosets (one row per orbit; averaged over G).
- Net rows per leaf (a call runs this times the leaf count), mean over the 22,100 flops (counted on the CPU by orbit
  enumeration; the test file recomputes it): today 49; **C2 40.32** (unpaired rainbow 49; two-tone 37, paired
  two-tone 38, paired rainbow 40; monotone 23; trips 28); **C1 39.42**. No flop needs more rows than today. On
  the flops that change (G not the identity) C2 has 36.41 and C1 35.11 (3.6% fewer; 7.5% on paired rainbow,
  11% on trips).
  - Correction to HANDOFF (11 Oct 21:08): "C2 59.8 rows" counted |H_b| rows per orbit. Inputs that a
    member of G_t relabels are the same input, so the count is the number of cosets [H_b : G_t]: 41.25 over
    all flops, 52 on unpaired rainbow flops. That is why C2 skips them (52 > 49 would be slower there).

## The order
1. C2: M1, then speed. If C2 passes both, it is the candidate for the live config and C1 is not run.
2. C1 only if C2 fails M1 (not if C2 fails only on speed: C1 has 3.6% fewer rows on the flops that change,
   the same structure, so it would not rescue a speed fail).
3. Neither passes: stop and report; dense stays.
- C1 runs after C2 fails, which gives two chances to pass: so both M1 tests use a Bonferroni CI, z = 2.241
  (one-sided 1.25% each) in place of 1.96.

## Tests before any data (CPU and GPU unit tests; not M1 or speed data)
- With a random evaluator that has no symmetry (float64, CPU), C2 equals dense run on the brute-force
  symmetrization by definition (each turn's input mapped to its orbit's canonical turn, the evaluator averaged
  over all of H_b, mapped back; C1: over G_t) to 1e-10, on two-tone, paired rainbow (whose aces make trips),
  paired two-tone, monotone and trips flops. Paired rainbow and trips are the cases where G_t is not normal in
  H_b: they are what catch a wrong coset side or permutation direction (the code audit's mutations confirmed
  this; the other classes cannot tell cosets apart). Outputs are bit-equal across each holding orbit of G.
- With the exact (symmetric) check-down evaluator, through a turn-indexed adapter (`CheckdownValues` itself
  answers only its flop's 49 turns in order), C1 and C2 match dense to 1e-9 (float64).
- On a rainbow flop, and on a job made asymmetric by hand, `FlopSearch(leaves="sym")` returns bit-identical
  results to `"dense"`.
- CUDA graph replay equals eager execution.
- The flop tests (`tests/test_flop_search.py`) and the selftests pass; dense is unchanged (its tests pass).
- The net-row counts above are recomputed in a test.

## M1: precision (the primary test)
- The registered M1 harness (`scripts/flop_m1.py`): the same 40 spots (`runs/flop-eval/m1-spots.json`), the
  same menus, flop_iters 500, gamma 5, the same net, the same oracle and rule (gamma 5, stop at 0.2% or 1,000
  iterations, f32 store, 12 threads), ISOMORPHISM on (the engine checks exactly and falls back to the full
  game). New candidate names `sym` / `orbit`; the existing files are not touched.
- The 13 rainbow spots (G = identity), per spot: if X_C's strategies at that spot are bit-identical to
  `m1-x-net.npz`'s, the game solved is the same (the oracle is deterministic) and its row is copied from
  `m1-v-net.json`, marked `copied`; otherwise that spot is solved (still valid; ~8 h more if none copies:
  X_net dates from 6 Oct and the engine was rebuilt on 11 Oct, numerics-neutral by its diff). The m1-v file's header records the sha256
  of `m1-x-net.npz` and `m1-v-net.json`; the report re-checks each copied spot's bit identity.
- Statistic: per spot, D = Loss(X_C) - Loss(X_net) = V(X_net) - V(X_C) (% of the pot), paired over the 40
  spots; mean and CI as M1's `paired` (normal) with z = 2.241. e = ORACLE_ERROR = 0.1% of the pot.
- **Pass (non-inferior):** the CI's upper end + e <= **0.25% of the pot**. (X_net's loss is 1.275%; its
  oracle brackets' mean half-width is 0.132%. Fixed before any C data.)
  - The estimand is the 40-spot mean, play's flop mix (31% unpaired rainbow, D = 0 there by construction):
    it allows about 0.37% (0.25 / 0.675) of harm on the spots that change.
  - Power: after e only 0.15% is left for the mean plus 2.24 standard errors, so C must be about as good as
    X_net or better to pass. Accepted: the user's rule is nothing less precise.
- Reported: superior if the upper end + e < 0; Loss(X_C) against star with its CI; the strategies' largest
  change from the G-averaging (rounding size expected); the oracle's isomorphism use per spot; CPU hours.
- X_net's oracle took 23.4 h over the 40 spots (15.4 h on its 27 symmetric spots, full game; 8.0 h on the 13
  rainbow); C2's 27 symmetric spots are expected near bp's 10.1 h (reduced game), plus ~8 h for rainbow spots
  whose rows cannot be copied. Each run is reported when it ends; nothing is decided at an interim look.

## Speed (after M1 passes; the GPU otherwise idle)
- Jobs: every flop job that X_C's `strategy` run sends to the search (`m1-jobs-<cand>.pkl`, recorded by that
  run, before any timing), each solved by dense (A) and by C (B) in the order A B B A, 500 iterations, gamma 5,
  the same net, in one process (`scripts/flop_sym_speed.py`). Both modes' tables for a flop are built, untimed,
  before its first job (again if evicted from the cache); each mode runs untimed warm-up solves on the first
  job of each flop class (|G| and C's rows), repeated on the class's next job while C's solve fell back to dense
  (lazy CUDA and cuBLAS setup and the class's row count; the leaf count still varies by job); no job is
  dropped. The speed run refuses a net other than the one X_C's jobs were made with (sha256).
- Per job: ratio = (tB1 + tB2) / (tA1 + tA2).
- **Pass:** on the jobs whose flop has G larger than the identity (those that fell back to dense counted, and
  their number reported), the median ratio is below 1 and its 95% percentile-bootstrap CI (10,000 resamples
  of spots, i.e. flops, with their jobs; seed 0) is entirely below 1; on the unpaired rainbow jobs the results
  are bit-identical (the same code path; timing reported).
- Reported: the ratio by flop class (|G| and C's rows); the median and p95 solve times of each; the net's sha256.
- Expected (not a gate; from the rows and the 84% share): ~1.2x faster on two-tone and paired flops, ~1.8x
  on monotone; unpaired rainbow equal. The mean flop solve should fall by at most ~15% (14.9% from the rows,
  before the gathers' cost). Forecast for p95: LIVE-FLOP's tail is decisions facing a bet (n 153 of 772, ~2.86-
  2.89 s); ~31% of those, ~47 or 6% of all flop decisions, are on unpaired rainbow flops, which do not speed
  up, so flop p95 will likely stay ~2.86-2.89 s.

## Adoption (if C passes M1 and speed)
- `scripts/slumbot_run.py` switches `leaves` to C; run.json records it; a resume refuses another `leaves`.
- The gate is LIVE-FLOP's, repeated with C: a new registered 1,000-hand mock run (`--mock`, committed play
  code), all of its checks, flop decisions' p95 <= 3.0 s; its `mock_check.json` committed as
  `records/flop-sym/mock_check.json` and `gates_refuse` pointed at it. A fail there leaves the live config
  on dense. No live hand is played by this registration (that is HANDOFF Next 4).
- The mock run also reports the share of flop jobs that took the symmetric path (the others had a
  rainbow flop or ranges G changes).
- Strength is not tested head to head here: M1 measures the leaves' effect on the flop strategy against an
  oracle directly; HANDOFF Next 2's rating will cover the live config.

- The FLOP-SYM result goes to `m1-report-flopsym.json` (`m1-report.json` keeps FLOPSEARCH's keys only). The
  report gives a verdict only when all 40 spots are in; before that it prints "interim" and no verdict. It
  refuses an X_C made with another net path than X_net's, or whose recorded net sha256 differs from that file
  now (X_net's meta predates the sha256).
- **Amended 11 Oct (after the code audit of 125dc85, before any M1 or speed data):** the warm-up solves, the
  report file and the interim rule above; drift and the net's sha256 recorded.

## VOID and bugs
- A crash, kill, reboot or another compute job on the GPU (speed) / the CPU oracle's machine overloaded by
  another heavy job (M1): that run is VOID, its cause recorded, and it is repeated.
- A bug found in C after any M1 or speed data: that run is VOID and still reported; the fix and the rerun
  need an amendment committed here before the rerun, and both results stay in the record.

## Not allowed after this commit
- Moving the margin, the order, the spots, the oracle rule or the speed rule after any C data.
- Choosing between C1 and C2 on their numbers (the order above decides).

## M1 result (12 Oct 2026; the oracle solve ran 11 Oct 22:11 - 12 Oct 06:56)
- **C2 (sym) PASSES M1**: Loss(X_sym) - Loss(X_net) = **+0.009% of the pot, CI (z 2.241) [-0.008, +0.027]** on
  40 spots; upper + e = +0.127% <= 0.25%. Loss(X_sym) +1.284% pot [+1.029, +1.539] vs Loss(X_net) +1.275%.
  323 of 479 flop solves symmetric; largest G-average change 0.0028. Spots 39-40 (unpaired rainbow) copied
  X_net's rows, bit-identical. FLOPSEARCH's M1 is unchanged (-2.129% [-3.062, -1.196]).
- Correction (12 Oct, LEAF-NET's audit): the line above should say all 13 rainbow spots (0-based: 1, 9, 13, 14, 18, 21, 23,
  24, 26, 27, 33, 38, 39) copied X_net's rows, bit-identical, as `m1-report-flopsym.json` lists, not spots 39-40.
- Data: `runs/flop-eval/m1-{x,v}-sym.*`, `m1-solve-sym.log`; report `records/flop-sym/m1-report-flopsym.json`
  (`flop_m1.py report`, 12 Oct 09:56). C1 (orbit) is not run (the order: only if C2 failed).
- The speed script had never run; its first launch failed on `import parciver` (no `src` path, unlike
  flop_eval.py): one line added, the rule unchanged.

## Speed result (12 Oct 2026, ~10:16-11:26; `records/flop-sym/sym-speed-sym.json`)
- **PASSES**: on the 323 jobs of the 27 flops with a stabilizer beyond the identity (none fell back to dense),
  the median ratio sym/dense is **0.900, 95% CI [0.899, 0.903]** (flop-clustered bootstrap, seed 0). The 156
  unpaired-rainbow jobs: bit-identical, median ratio 1.000. All jobs: solve median 2.226 -> 2.043 s, p95 2.536
  -> 2.487 s. Net sha256 a8bf00be.
- Next (Adoption): `slumbot_run.py` to `leaves="sym"`, then the registered 1,000-hand mock.

## Adoption result (12 Oct 2026; the registered mock run 11:31-12:03, `runs/slumbot/mock-20261012T033126Z`)
- **PASSED** at 4f71d66 (`slumbot_run.py` LEAVES = "sym"): 1,000 hands, 0 lost, chips conserved, every history
  legal, none refused, 1,000 winnings checked; parser check 214,221 prefixes agree. **Flop decisions p95 2.791 s**
  (<= 3.0; median 2.160, max 2.908, n 750); turn/river p95 0.368 s; **1.86 s/hand**. 526/750 flop solves (70.1%)
  took the symmetric path; largest G-average change 6.5e-5. LIVE-FLOP's dense mock for comparison: p95 2.87 s,
  median 2.36 s, 2.10 s/hand (another seed's spots are not paired; descriptive).
- Gate: `records/flop-sym/mock_check.json` (GATES). **The live config is now flop search with sym leaves.** No
  live hand is played by this registration (HANDOFF Next 4).
