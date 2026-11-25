# D3: a better leaf net N2 from +1,000,008 turn-start rows (HANDOFF Roadmap C)

**Status: REGISTERED 9 Nov 2026 13:59 SGT** (`date`: 13:59 at registration), after the user's three clickable rounds of answers
the same day (section 12). It replaces `PREREG-DRAFT.md` (a side session's draft of 8 Nov, kept beside this file as `PREREG-DRAFT.md`, superseded, committed with it).
**This is a registration only: nothing has run, no data was made, no net was trained, no GPU or CPU job was started.**
Data generation, training and every compute job need the user's separate go after this commit (section 13). It was written
from the repo's records only; no label and no net output was read. "D3" is this study's name, not FLOPSEARCH's old step D3.

D2b is ADOPTED (`records/d2b/RESULT.md`: Gate A PROCEED, Gate B PASS, match NOT DOWN, mock PASSED; live flop net **N1** since
6608183, mock GATES at 3df72b1). So the baseline **B = N1** everywhere below, which is what every calibration here assumes.

## 1. The question
**Does a leaf net N2, trained on D1 + D1b + a new D1c (+1,000,008 turn-start rows, doubling the union to 2,000,016), beat the
live net N1 on the flop without making the whole bot worse?**

Why (records, not hopes):
- The net's error is the flop's error (IDEAS B). N1 beat N0 by **-0.585% of the pot**, CI [-0.757, -0.412], on 160 M1 spots.
- More data was the lever: -14.2% selection loss at D2b's doubling (N0 0.014849 -> w2048 wd 0 0.012736; `selection.json`),
  more than the curve's ~-10%. The curve's last doubling (0.01711 -> 0.01464) is a ratio of 0.855 and D2b's wd-0 doubling is 0.858 (exponents ~0.22; the curve's first doubling was steeper, 0.811): loss ~ rows^-0.22.
- N1's train loss at its best epoch is 0.002626 against a selection loss of 0.011799 (4.5x): variance-limited, which is my
  reading, not a measured fact.
- **What it is for.** A precision step: the standing priority is strength and precision, speed second. The effect is small at the
  flop and unmeasurable in play (section 4). D3 asks for a net that is better on labelled data (Gate A), no worse at the flop (Gate B)
  and not worse in play (match), and it does not promise a win-rate gain.

## 2. What D2b leaves (re-verified 9 Nov against the records; section 14 traces each)
| Item | Value |
|---|---|
| Union rows, train / selection / gate | 951,466 / 23,728 / 24,814 (1,000,008 in all); fingerprint 7d5a77c30aaad5d6 |
| Split | splitmix64 of the suit-canonical board, SPLIT_SEED 20260929: <2.5% selection, <5% gate; no board crosses a half |
| Selection loss N0 / N1 | 0.014849 / 0.011799 (-20.5%) |
| Data effect at wd 0 / wd effect / seed spread (one pair) | -14.2% / -7.4% / +1.17% (0.011937 vs 0.011799) |
| N1 | w2048-d5, wd 0.1, 200 epochs, batch 2048, lr 1e-3 OneCycle, bf16 hidden / f32 out, Huber, suit augmentation, seed 0, non-canonical; sha256 54bae749...f97d72; 2,900.5 s (950.3 s of it upload) |
| Gate B (N1 - N0, 160 spots) | mean -0.585%, **sd 1.114%**, CI [-0.757, -0.412]; old 40: -0.216% (sd 0.713), new 120: -0.708% (sd 1.196) |
| Match N1 - N0 (10,000 deals) | -7.24 bb/100, raw 95% CI [-23.80, +9.32]: NOT DOWN; reading: it catches a gross loss only |
| **Latency (the draft's headroom claim was stale)** | mock at N1: flop p95 **2.976 s** vs bar 3.0 (margin 0.024 s; N0's mock 2.888, SLUMBOT-2's live run 2.663); N1's p95 on the 479 speed jobs 2.995 s; match lock-in flop p95 2.376 s (N1) vs 2.380 s (N0) |
| Cheaper net | w1024 is 0.540x the time [0.529, 0.544] at +3.3% selection loss; reported only |
| D1b's PC rate | 0.740 solves/s (derived: 129 shards, 128,004 rows, 48.07 h); recording 250,002 situations took 2 h 44 min (`record.log`) |

## 3. The arms
1. **More turn-start data, D1c. THE ARM** (size +1,000,008, decided 9 Nov: section 12).
   - Expected selection loss at 2.0M rows: 2^-0.223 = 0.857, **-14.3%** (-10% if returns fade to exponent 0.15). In MAE (Gate A's
     statistic; the data effect has been smaller there, -9.8% at D2b's doubling against -14.2% in loss): exponent 0.12-0.18,
     **-8% to -12% in MAE**. Extrapolation, not a measurement.
   - A scout (S0a) first, because D1 and D1b are blueprint play, not what the live flop search queries (D1b PREREG "Known limit").
2. **Recipe pin from a sweep first, as its own registration** (decided 9 Nov). A dropout / EMA / EV-space-loss sweep on D1 + D1b,
   fixed grid, selection by its own rule, is registered separately and run first (~1 h GPU per net). N2's recipe is then pinned by
   an amendment here that may name only (a) N1's recipe or (b) the one winning recipe of that sweep, **before D1c's first label is
   read**. Until the sweep is registered and run, N2 = N1's recipe. D3 never runs a grid itself.
3. **Ensemble ceiling S0c** (N1 and `d2b-seed1`, mean of the two on the selection half, forward only, minutes). Reported; if >= 6%
   better, a distillation registration is proposed. It touches nothing in D3.
4. Not done, and why: cheaper labels (400 iterations fail D1's exploitability bar, 0.60% against < 0.5%; SC3 warm start is
   unbuilt and restarts the solver-parity story), M1 values as training data (flop-level values, a flop net would need millions of rows),
   a wider net (the latency bar binds: 2.976 s against 3.0 s).

## 4. What the effect is expected to buy (a limit stated up front)
- Selection loss -10% to -14%; MAE -8% to -12%.
- At the flop: D2b's one data point is 0.0285 points of the pot per 1% of selection loss (the old 40 alone: 0.0105), so -10% to
  -14% maps to roughly **-0.1% to -0.4% of the pot; central ~ -0.3%**. LEAF-NET's P had -7.1% selection loss (on D1's half; -6.5% on the union's) and an M1 mean of
  **-0.001%**: gains of this order can sit under M1's noise, so the low end is about 0.
- In play: SLUMBOT-2's CI over 50,000 hands is [-26.41, +1.33] bb/100. No play match can measure this. The 10,000-deal match vetoes a gross loss only.
- The user may rightly judge ~3 weeks of PC time better spent elsewhere; this is registered because the user said yes (section 12).

## 5. The design (staged; each stage only on the last one's pass; no interim look decides anything except the two futility looks in 5.2)
### 5.0 Code stage (tests first; no label, no net)
1. **Leaf-query hook.** The live flop path is `FlopSearch(net, leaves="sym")` (`scripts/slumbot_run.py` LEAVES; `FlopSearch.tables`, `src/parciver/flop_search.py`
   ~608-625): `SymLeaves` / `NetLeaves` call `evaluate = net_values(net)`, one batch per traversal (two traversers per iteration, 500 iterations; KERNELS.md:29: 9 leaves x 49
   turn cards = 441 rows per traversal in the M1 geometry). `FactoredLeaves` is not on the live path (the live `LEAVES` is "sym"). **Graph replay:** `FlopSearch.graph` defaults to True and `FlopSolver.solve` runs a Python callable only in the
   side-stream first iteration and during capture, so a logging wrapper would see iteration 1 (near-uniform ranges) and nothing else. **The hook therefore runs the harvest
   and S0a searches with `graph=False`** (bit-identical results, slower; the code stage measures the slowdown, the harvest hours below are re-stated from it before any run, and
   time is not a gate). `evaluate = net_values(net)` is a local of `FlopSearch.tables`, so the harvest script subclasses `FlopSearch` (or patches `flop_search.net_values` and clears `_tables`)
   to wrap it; the wrapper logs the inputs. **"The last iteration's queries" = the rows of both evaluate calls (one per traverser) of iteration 500, pooled.** **A query is one row exactly as the hook sees it, i.e. as `NetLeaves` (or `SymLeaves`: canonical-turn plus coset rows, 23-49 per leaf on symmetric flops) hands it to the net** (the
   unit-normalised ranges the net takes, board, stack, half), so a label is the solve of that very row at D1b's settings; the number of rows per call is read from the
   hook's first call and written into the harvest script's test, not assumed here. No play code changes: `scripts/slumbot_run.py` and the engine are untouched.
2. **Recorder:** `turn_data.py record --exclude` takes **two** situation files (D1's and D1b's), refuses a seed block touching either
   [520000, 538000) or [600000, 618000); a test proves the per-cell quota is reachable under the 999-chunk cap with both excluded.
3. **Memory-mapped loader** (section 8) and **`d3-check`**: N1's recipe trained again on the union with the new loader must give **N1 bit for
   bit** (as `d2b-check` reproduced N0). Needs the GPU for ~50 min; it reads D1 + D1b only, so it may run in the code stage.
4. **`d3` driver:** `data` (fingerprint, no values read), `look` (the futility nets), `train` (N2 and the seed-1 net), `select`, Gate A, the M1 jobs,
   adapted from `scripts/d2b*`.
5. **Flop-spot recorder and harvest script (new code, none of it exists today).** `turn_data.py record` records turn roots; `flop_m1.py record` is a single-stake
   self-play recorder with no cells. D3 needs a recorder of **flop spots over the 18 cells** (200bb / 100bb x 9 opponents), with its own seed block (800000), **every block 18,000 wide like turn_data's (match seed = seed + 1000 x cell + chunk): S0a [780000, 798000), the harvest [800000, 818000), D1c [700000, 718000)**, each refused against
   every other block (D1, D1b, M1's 550000 and the rating seeds) by a test, and the harvest generation path: **one source, no random twin** (`turn_data.py generate` writes pairs), plus **its acceptance** (A1-A3 adapted
   to its layout, A4 as D1b's). Pre-declared here, so no data picks them: the quota is 18 cells x 567 = 10,206 spots, of which **the spots are taken round-robin across the 18 cells (seeded within a cell) and the first 10,204 are used**
   (cells 0-15 get 567, cells 16 and 17 get 566); 49 queries per spot; 10,204 x 49 = 499,996 queries, **the last spot is truncated to its first 21 queries so exactly 499,968 remain**; the harvested rows
   are interleaved **one to one** with the recorded rows in shard order, and the **72 recorded rows left over go at the end of the last shard(s)**. The harvest blueprint for a cell
   is **v2-8B when it can play that cell's stake, else v2-2B** (decided by what loads, not by a result; reported per cell).
6. **Speed gate mode.** `scripts/flop_sym_speed.py --cand d2b` is hard-wired to N1 against a w1024 net and writes `records/d2b/speed.json`; it cannot time N2. A new `--cand d3`
   mode (A = N1, B = N2, same jobs file, **its own output paths, never `records/d2b/`**) is built and tested first.
7. **Gate B code.** `scripts/flop_m1.py` is hard-wired to D2b (`D2B = ("d2b0", "d2b1")`, `D2B_SPOTS` 160, `D2B_CARRIED` 40, `N0_SHA256`, the `gate-a.json` guard keyed to N1's
   sha). Parametrise the baseline/candidate pair (X_N1 the baseline, N2 the candidate), the sha guard, and the Gate A path, and **add the compare of re-made spots 0 and 1 of X_N1 (d2b1) against the stored rows (today only the carried d2b0 rows have a re-make path, `old_x`/`D2B_REMADE`)**, with tests; D2b's own commands must still give D2b's results.
8. Every piece gets the read-only subagent audit, one at a time, before use.

### 5.1 Stage 0: scouts (descriptive; they select only what 5.2 samples, never a gate)
- **S0a, what the live flop search asks the net.** On **fresh spots** (seed block 780000, 200 spots, disjoint from M1's 550000, D1's, D1b's,
  D1c's and the harvest's blocks; the recorder's refusal test proves it) run the live flop search with **N1 on v2-8B where it can play the cell's stake, else v2-2B** (5.0.5; the
  report says which per spot; with v2-2B it cannot see the difference the scout exists for) and log the leaf queries. Take **10
  queries per spot, uniform without replacement from the last iteration's queries, seeded: 2,000 queries**. Solve them at D1b's settings
  (500 iterations, gamma 5, `--terminals64` = Q4; 2,000 / 0.74 per s = 45 min GPU) and score N1 on them. These 2,000 are never training,
  selection or gate rows.
  - **Statistic:** r = N1's selection loss (the same function as `selection.json`) on the 2,000 queries / **0.002217**, N1's loss on the
    union's real-range rows (not the 0.011799 all-rows figure: half of those are random twins at 0.0214). 95% percentile bootstrap **over
    spots** (the 10 queries of a spot are one cluster), `default_rng(0)`, 10,000 resamples.
  - **Rule fixed here, before any query exists:** if the CI's **lower end of r is <= 1.5**, **D1c = D1b's recipe twice over: 18 cells x 27,778 =
    500,004 recorded situations, each with a random twin, = 1,000,008 rows.** If the lower end is **> 1.5**, **D1c = half recorded, half
    harvested: 18 x 13,890 = 250,020 recorded situations with twins = 500,040 rows, plus 499,968 harvested rows** (one source, no twin).
    S0a selects between these two branches and nothing else. Why 1.5: a judgement, about half way on a log scale between "same as the real
    ranges" (1) and the random twins' 9.7x (0.0214 / 0.002217); the real-range seed spread is 0.15%, so the line is far above noise.
  - **Harvest rule (fixed now, used only in the > 1.5 branch; the quota arithmetic is in 5.0.5):** flop spots of the new flop-spot recorder over the 18 cells, seed block
    **800000** (the first 10,204 spots of 10,206); per spot run the live flop search (N1 leaves, the blueprint rule of 5.0.5),
    take **49 queries uniform without replacement from the last iteration's, seeded**; label each at D1b's settings; exactly 499,968 queries kept (the last spot truncated, 5.0.5). Cost: 10,204 live flop searches at ~2.4 s (N1's median 2.384 s on `speed.json`) ~ 7 h GPU, then the labels at
    0.74/s ~ 188 h, which is inside the 375 h total.
  - **Blueprint for the recorded half: v2-2B** (as D1 and D1b, so the data stays one family).
- **S0c** as arm 3. Order and cost: minutes of GPU; no row is read.

### 5.2 Stage 1: D1c data (the PC; the cloud is not an option, user 8 Nov)
- **Recorded part:** `turn_data.py record` at seed **700000** (match seeds [700000, 718000)), `--exclude` D1's and D1b's `situations.npz`, 18 cells
  (200bb/100bb x 9 opponents), cell quota as above; duplicates dropped, a cell tops up until full (D1b's rule). ~5.5 h CPU (D1b: 2 h 44 min for
  half as many); it must not run beside the match.
- **Labels:** `turn_data.py generate --samples 1000008 --seed 700007 --terminals64` (the half branch: the same, with the harvested rows from seed 800007 and the
  two sources interleaved one to one in shard order, layout pinned and tested in 5.0.5 before it is used). **1,001 shards of 1,000 rows**
  (the last has 8). Stop rule: all 1,001 shards, each complete. Shards mix every cell (the order is a seeded permutation).
- **Rate and time:** 0.740 solves/s on the PC = **375 h = 15.6 days** of GPU with nothing else on it. D1b's rate is the only measured one; it is
  re-measured on D1c's first 20 shards and reported (the stop rule does not change).
- **Acceptance, as D1b:** `d1b_accept.py check` (A1-A3) and `verify` (A4) on D1c; D1b had mean exploitability 0.327% of the pot against a bar of
  0.5%, zero-sum max 9.4e-8, rows over 10% 0.0038%. Rows over 10% are counted and reported. A failed A1-A3 or A4 voids the data
  (section 7). No label's value is read beyond these statistics, **except in the two futility looks**.
- **Two futility looks (decided 9 Nov), at the first 251 and the first 501 complete shards** (~251k and ~501k rows; the union is then
  ~1.25M and ~1.50M rows). At each: generation pauses on a shard boundary (one GPU job at a time), A1-A3 are run on the shards so far,
  then **one net with N2's recipe (seed 0), trained on the union + the shards so far** with the same split hash, scored on the **union's
  original selection half** (23,728 rows, the same rows as N1's 0.011799). **Continue iff its selection loss <= 0.98 x N1's (-2%) at the first look,
  <= 0.96 x N1's (-4%) at the second.** If not, **D3 STOPS**: N1 stays, the shards are kept, nothing is adopted and no later look is added. Cost
  ~1.1 h and ~1.3 h GPU. Why these lines: the extrapolation gives -4.9% and -8.7% at those sizes; -2% and -4% are about 40-45% of
  it and 1.7x and 3.4x the one measured seed spread (1.17%). The look nets are never chosen from, adopted or used in any gate. The looks read labels
  only through training; that is the one registered exception to "no label read before the union is pinned".
- **Union fingerprint:** `d3 data` hashes D1 + D1b + D1c with no values read; an amendment pins it before N2 (D2b's Amendment 2 did this).
- **Disk (decided 9 Nov: free space first).** D1c on C: needs ~22.6 GB (shards ~18, situations ~4.5; the NVMe is the only drive training reads
  at speed). C: has **36.8 GiB free on 9 Nov 13:49** (the draft saw 77 on 8 Nov). **D1c's generation does not start until the user has
  freed room so that C: has >= 45 GiB free, or says otherwise**; the generator pauses and tells the user if C: free falls below 6 GiB. Gate B's
  oracle spots need ~27 GiB free (a further check at Gate B). Nothing is deleted by the assistant; the candidates are listed in the final report
  and in section 8. E: (USB HDD, 4.65 TiB free) can hold moved folders.

### 5.3 Stage 2: training
- **N2:** the recipe pinned in arm 2 (N1's unless the sweep amendment says otherwise); data = union + D1c, the same hash split (a D1c row whose canonical board hashes to the selection or gate share goes
  there, so no board is in two halves across any dataset). Expected sizes: train ~1.90M, selection ~47.5k, gate ~49.6k.
  Same code path as N1 (`d2b` trainer at e220bc0 or its audited successor); same 200 epochs. Est. ~1.6 h (N1's 2,900 s scaled 2x).
- **Seed-1 net** (decided 9 Nov): the same recipe and data, seed 1, **reported only, never used to choose**: it gives the seed noise at 2M rows
  (the one pair at 1M rows showed 1.17%). ~1.6 h GPU.
- **Stage 3, selection (one candidate, so a stop rule):** **if N2's loss on the combined selection half is not below N1's on the same rows, D3
  stops.** Reported: every net's loss on each dataset's selection half, real and random sources, train loss, the seed-1 net. No runner-up.

### 5.4 Stage 4: Gate A (D2b's statistic)
N2 against B = N1, both H_b-averaged, f32. D_i = MAE_N2(i) - MAE_N1(i), range weighted, % of the pot; mean with a 95% percentile bootstrap, 10,000
resamples of canonical boards with their samples, `default_rng(0)`. **Relative MAE change = D / mean MAE_N1** on the same rows (`gate-a.json`'s convention), not relative to the label mean.
- **Primary set: D1c's gate half** (~24.8k rows). Fresh ranges, **not fresh boards** (the split hashes the board, so it falls on the same
  ~406 boards as D2b's; disclosed as D2b did). **PROCEED needs the CI's upper end < 0 and the relative MAE change <= -4%.** Why -4%
  (decided 9 Nov, not D2b's -5% and not the draft's -3%): half the low end of the expected -8% to -12% at 2x data, ~17x the one measured
  MAE seed spread (0.24%); the bootstrap half-width at ~24.8k rows is ~0.8% of the MAE, so the bar is not limited by the CI.
- **No regression on the old gate halves (decided 9 Nov):** on D1's gate half (12,514 rows; the one G2 and LEAF-NET read) and on D1b's (12,300 rows; D2b's primary set), each separately, the same
  statistic D (and its CI, same bootstrap). **D3 stops if the CI's lower end of D (N2 - N1) is > 0 on either**, i.e. N2 is significantly worse than N1 there. (My
  reading of "no regression"; no margin is added. Reported beside the primary set either way.)
- G2's bar (MAE under half of the check-down's, 9.3228%) on D1's gate half must hold too.
- **In the > 1.5 branch** about half of D1c's gate half is harvested rows, a distribution N1 never trained on; the -4% bar and the CI condition are then judged on D1c's **recorded rows only**, and D on the harvested rows is reported beside it (the assistant's operationalisation, section 12).
- Otherwise D3 stops and N1 stays.

### 5.5 Stage 5: Gate B, M1 non-inferiority (decided 9 Nov: 160 spots)
- **Spots:** the first 160 of `flop_m1.py record`, seed 550000 (`runs/flop-eval-d2b/m1-spots.json`), bit for bit. No spot is chosen by a result.
- **Test (as D2b):** D = Loss(X_N2) - Loss(X_N1), both with sym leaves, 500 flop iterations, gamma 5, the same oracle, ISOMORPHISM on, paired over 160
  spots, z 1.96, e = 0.1% of the pot. **Pass iff the CI's upper end + e <= 0.25% of the pot.** A verdict only when all 160 are in.
- **Code:** `flop_m1.py` is parametrised for this pair first (5.0.7).
- **Rows:** X_N1 is on disk (`runs/flop-eval-d2b/`), so **only X_N2 is new: 160 solves at ~29.8 min each at 12 threads ~ 79 h CPU**, after ~1 h of GPU strategies. The baseline rows
  are valid only under engine build E58C7292 and an unchanged flop search, so (as D2b's Amendment 1) **spots 0 and 1 of X_N1 are re-made first and must equal the stored
  rows; any difference re-solves all of X_N1 (+79 h).**
- **The margin stays 0.25%.** Reported: the old 40 and new 120 apart; per-spot D; the five largest D with pot and board; whether the tail is N2's or N1's.
- **Power, from D2b's per-spot D** (bootstrapped, recentred, shifted; recomputed 9 Nov, a design check, not a registration number): n = 160, sd 1.11:
  pass **.38 if the true mean is 0, .86 at -0.1, 1.00 at -0.2 or lower**; false pass at +0.15 ~.01; superiority (upper < 0) .16 at -0.1, .66 at -0.2, .97 at -0.3.
  Non-inferiority needs sd <= 0.97% at n = 160 (the 40-spot rule). **Gate B cannot show a small flop gain; Gate A carries "better", Gate B carries "no harm".**
- **Inconclusive** (the CI crosses the line): not adopted; no extra spots after seeing it.
- **Chance of adopting, stated as D2b did:** ~.9 if N2 is truly -0.3% better at the flop (Gate A very likely, Gate B ~1.0, the match a veto); ~.35 if it is better on
  labels but only equal at the flop (Gate B alone: .38; the LEAF-NET pattern); ~0 if worse.

### 5.6 Stage 6: whole-bot safety and latency (decided 9 Nov: match included, latency rule below)
- **Smoke:** 50 deals, seed **777,007**, 1 and 4 threads must agree deal for deal (D2b's protocol) before the match.
- **Match:** L1 (live bot with N2) vs L0 (live bot with N1), 10,000 deals, baseline row seed 410,001, seats swapped, 4 threads, chunks of 100, 50 blocks of 200, bb/100 from
  L1's side, AIVAT reported. **DOWN iff the raw 95% CI's upper end < 0.** Not DOWN is needed. ~20 h GPU. It can only catch a gross loss (N1's match: raw +/-16.6 bb/100).
- **Latency gate (the rule accepted 9 Nov as "N2 must not be slower than N1"; the 1.02 is the assistant's):** `flop_sym_speed.py --cand d3` (5.0.6), A = N1, B = N2, the 479 jobs of `runs/flop-eval/m1-jobs-sym.pkl`,
  A B B A, same session, GPU otherwise idle. **Pass iff the flop-clustered 95% CI's upper end of the median ratio B/A is <= 1.02.** The 479 jobs cover **only 40 distinct flops**, so the
  cluster bootstrap has 40 clusters (`speed.json`: flops 40); that is disclosed, not fixed. Why: N2 has N1's ops, so the 1,000-hand mock's flop
  p95 (2.976 s at N1 against the 3.0 s bar) is near a coin flip for machine reasons and a failed mock would cost ~3 weeks.
- **Adoption:** (1) `scripts/slumbot_run.py` NET -> N2's `net.pt`, **committed with the user's yes**; (2) the registered 1,000-hand `--mock` on that commit, seed 20261024 (D2b's mock seed, reused on purpose so the two mocks are comparable): **every correctness
  check passes** (hands, chips, legal histories, refusals, winnings, parser), **turn and river p95 <= 2.0 s**; the **flop p95 is reported against 3.0 s and does not gate** (the speed A/B does),
  but if it exceeds 3.0 s the adoption waits for the user's explicit yes; the passing `mock_check.json` becomes `records/livesearch/mock_check.json`; a fail reverts NET to N1.
  (3) No fallback: no runner-up after a stop or fail. (4) **Pushing `main` is a separate yes.**

## 6. Decisions inside the design
- **Baseline B = N1.** If a later net is adopted before D3 reaches Gate A, D3 compares with the net that is live at Stage 4 only by amendment before any N2 label is read; otherwise N1.
- **One candidate, one seed, no grid in D3** (a grid buys selection bias; D2b ran 7 nets). The sweep is separate (arm 2).
- **Futility looks and S0a are the only places a number selects anything before the gates**, and each selects one registered branch or a stop.

## 7. VOID and bugs (stage by stage, as D2b)
- **Code stage:** a test failing or an audit finding is fixed before use; nothing reads data.
- **D1c recording:** a crash resumes; a recorded cell found short of its quota under the chunk cap is reported and stops the stage (no silent shortfall).
- **D1c generation:** a worker crash resumes per shard (a shard is kept only if complete); a shard run beside another GPU compute job is kept (labels are not timing); a failed A1-A3 or A4 voids D1c and it is reported.
- **S0a, looks, N2 and the seed-1 net, `d3-check`, the speed A/B, the match, the mock:** a crash, kill or reboot, or **another compute job on the GPU** (training, inference, a solver) voids that run; its cause is recorded and it
  is repeated. Desktop apps' GPU use is reported, it does not void.
- **Gate B:** M1 rows are not voided by machine load (two spots re-solved days apart came out bit-identical in every field but the clock). A killed solve resumes per spot. A commit stall delays, it does not void.
- **A bug found after any data:** that run is VOID and still reported; the fix and the rerun need an amendment first; both results stay.

## 8. Machine needs and wall clock
- **GPU:** one job at a time; 5.4 GiB was in use at the 9 Nov 13:49 read (36% load; another session's job or desktop apps; no D3 job is running).
- **Memory (the blocker, worse than the draft said).** The draft's 47-56 GB host peak was for 1.5M rows; at **~1.9M train rows it is roughly double D2b's 31-37 GB estimate**,
  and the GPU holds data in pools (12.9 GB at 951k rows on a 12 GiB card). The user chose (9 Nov) the **memory-mapped loader**: it maps the shards and builds one pool at a time.
  **Its test must report the host peak at 1.9M train rows** and `d3-check` must reproduce N1 bit for bit; **N2 is not trained until the measured peak is below the free commit at that
  time, and the user is told the number first.**
- **Disk:** C: 36.8 GiB free now; D1c ~22.6 GB; needs >= 45 GiB free to start (5.2). **Candidates the user may free, found by size (my grep; none verified safe to delete, none touched):**
  - **Not referenced outside `runs/` and `records/d3` by my grep:** `runs/blueprint-20260927-034129` 7.0 G, `runs/blueprint-20260927-032156` 3.4 G, Temp 3.4 G, pip cache 0.7 G.
  - **Pinned by records and rating receipts (the audit found these; moving or deleting breaks `rating.py verify` / replay of those receipts unless the receipt paths are also rewritten, because receipts pin absolute C: paths):**
    `runs/f1-stageA` 11 G (`records/preflop-open/f1-decide-...json`, F1-PREREG.md, `f1_chain.ps1`, `launch_f1.ps1`, a rating receipt; F1 is NOT ADOPTED),
    `runs/bp-8b-stage1` 7.2 G (`records/bp-8b/RESULT.md`, HANDOFF-ARCHIVE, a receipt), `runs/bp-8b-stage3` 3.6 G (v2-S3, not live; HANDOFF.md, `records/bp-8b/PREREG.md` and RESULT.md, a receipt).
    So "moving to E: is reversible" holds only with the receipt paths kept valid (e.g. a directory junction at the old path).
  - **Keep:** `runs/bp-8b-stage2` (v2-8B live), `runs/blueprint-20260927-141645` (v2-2B, the recorded half), `runs/turn-data*` and `runs/turn-net` (the union and the nets), `runs/blueprint-20260914-144924` and `-20260927-060745` (pinned in `scripts/rating.py`).
  Deleting is not reversible.
- **Oracle spots:** each needs its store + 11 GiB, up to 27.2 GiB; ~27 GiB must stay free through Gate B. No blueprint job beside training or M1.
- **Wall clock, serial, PC (my sums from the rates above):** code stage 1-2 working days (before any data) + `d3-check` 0.8 h + S0a ~1.1 h + S0c minutes + record 5.5 h + **D1c 375 h (15.6 d)** + looks 2.4 h
  + acceptance 0.3 h + N2 1.6 h (+ seed 1 1.6 h) + Gate A < 1 h + X_N2 strategies 1 h + **Gate B 79 h (3.3 d)** + smoke and match ~21 h + speed A/B 1 h + mock 0.7 h = **~20.5 days of compute, ~21-22 days with the code stage**. In the > 1.5 branch add the harvest searches (~7 h at graph on, more at graph=False; the recorded half saves ~2.7 h of labels) and ~2 GB of disk. The sweep (its own registration) and the user's
  disk work come before D1c and are not counted.

## 9. Order
Register (this commit) -> user's go for the code stage -> code, tests, audits, `d3-check` (no label of D1c, no D1c net) -> **the sweep, registered separately (user's yes)** -> recipe-pin amendment -> S0a, S0c -> pin D1c's branch (the pre-declared rule, no new decision) -> in the > 1.5 branch only: the flop-spot recording and the 10,204 harvest searches (~7 h GPU or more at graph=False)
-> the user frees disk -> user's go for compute -> record -> generate D1c (looks at 251 and 501 shards) -> acceptance A1-A4 -> union fingerprint pinned (amendment) -> N2 (+ seed 1) -> selection -> Gate A ->
X_N2 strategies (GPU) and 160 solves (CPU) -> Gate B -> smoke -> match -> speed A/B -> the user's yes to NET -> mock -> the user's yes to push. Each stage is reported when it ends.

## 10. What this does NOT decide
- A compact (w1024) net for latency (RM1): reported only; its own registration. A play gain (unmeasurable). Preflop search (RM2), turn iterations, the solver (SC3, SC4), anything about v2-8B or the Slumbot exam.
- Whether an ensemble distillation is worth building (S0c reports; any build is a new registration).

## 11. Not allowed after registration
- Changing the arms, the D1c rule (r, its 1.5 and both branches; S0a only selects between them), the harvest rule, the futility looks and their lines, the selection rule, gates, margins, spots, the latency rule or the order after any D3 data.
- Choosing N2 on gate-half or M1 numbers; swapping in seed 1 or a look net; trying a runner-up after a fail or stop.
- Dropping or reweighting a spot, or moving to a trimmed statistic after seeing its D.
- Reading a D1c label beyond the acceptance statistics and the two futility looks before the union is pinned.
- Anything touching `scripts/slumbot_run.py`, the engine or `runs/` before the stages above allow it.

## 12. The user's answers (9 Nov 2026, clickable rounds; every design choice below is the user's, apart from the operationalisations listed at the end of this section)
**Round 1.** Size and place: **+1,000,008 on the PC** (over +500,004 / +250,002; the draft had recommended +500,004). Memory: **memory-mapped loader** (not freeing ~55 GB). Futility: the user chose 1,000,000 and
asked it be considered ("take into consideration i chose the 1000000"), hence round 2. D1c recipe: **sweep first, as its own registration** (the draft had said no sweep before D3).
**Round 2.** Futility: **two looks, at +250,002 and +500,004** rows (here the first 251 and 501 shards: 5.2). Gate A: **-4% in MAE, plus no regression on the old gate halves**. Gate B: **160 spots**. Match: **include, as D2b**.
**Round 3.** D1c recipe rule and S0a: **scout, then the pre-declared rule** (5.1; the harvest rule in it was fixed by the assistant's question and accepted). Latency: **N2 must not be slower than N1** (5.6). Disk: **"lets see where we can free space first"** (5.2, 8: the user decides what goes).
Seed-1 net: **yes**.
**Standing from 8 Nov:** the cloud is not an option; the NET commit and pushing `main` need a separate yes at adoption. **The assistant's operationalisations of those answers (not the user's numbers; the user may change any of them before the code stage):** the no-regression wording (5.4: lower end of D > 0 stops, no margin); the
"-4%" read as D / mean MAE_N1; the futility lines -2% / -4% and their positions at shards 251 / 501 (rounding of +250,002 / +500,004 rows); the 1.5 line for S0a's r and its statistic; the 1.02 latency bar and the 479-job test;
the 45 GiB start threshold and the 6 GiB pause floor; the harvest quota arithmetic and blueprint rule (5.0.5); the new speed and Gate B code modes (5.0.6, 5.0.7); S0a's 200 spots x 10 queries, "the last iteration" as the harvest's source, the seed blocks 700000, 780000 and 800000 and their 18,000 width; judging Gate A on recorded rows only in the > 1.5 branch.

## 13. What the user still has to say yes to before compute starts
1. The code stage (loader, harvest, `d3` driver, tests and audits).
2. The **sweep's registration** (separate document) and its run, then the recipe-pin amendment.
3. **Free C:** >= 45 GiB (5.2).
4. **The go for S0a and S0c** (GPU free of other jobs; S0a, ~45 min, precedes D1c and fixes its branch).
5. **The go for D1c's compute** (record, in the > 1.5 branch the harvest searches, generate, the looks), with the GPU free of other jobs.
6. At adoption: the NET commit, and pushing `main`, each separately.

## 14. Number trace (every figure above, from a repo record unless marked mine)
| Number | Record |
|---|---|
| 0.014849 / 0.011799 / 0.012736 / 0.011937, -20.5%, -14.2%, -7.4%, +1.17%, loss 0.002217 (real rows), twins 0.0214, N1 train 0.0026255; MAE seed spread 0.24% | `records/d2b/selection.json`, `nets.json` |
| Gate B -0.585 [-0.757, -0.412], sd 1.114, old 40 / new 120, hours 55.0 / 79.35 | `records/d2b/m1-report.json` |
| Match -7.24 [-23.80, +9.32]; latency 2.376 s; mock 2.976 s, N0 mock 2.888, SLUMBOT-2 2.663, speed-job p95 2.995, w1024 0.540x [0.529, 0.544] | `records/d2b/RESULT.md`, `live.json`, `mock_check.json`, `speed.json` |
| Union 951,466 / 23,728 / 24,814, fingerprint 7d5a77c30aaad5d6; Gate A -1.05, G2 bar 9.3228 | `records/d2b/data.json`, `gate-a.json`, PREREG Amendment 2 |
| Curve 0.02109 / 0.01711 / 0.01464 | `runs/turn-net/curve.json` |
| D1b 129 shards in 48.07 h (RESULT.md says ~0.73 solves/s for the first shards), 501 shards, acceptance 0.327%; recording 9,788 s (`runs/turn-data-d1b/record.log`, not a record file); seed blocks [520000, 538000), [600000, 618000); `record --exclude` takes one PATH; match seed a.seed + 1000 * c + chunk, cap 999 | `records/d1b/RESULT.md`, `records/d1b/PREREG.md`, `scripts/turn_data.py` |
| 500 iterations, 2 traversers per iteration, 441 = 9 leaves x 49 turn cards (M1 geometry); live path `leaves="sym"` | `src/parciver/flop_search.py` (FlopSearch.tables, SymLeaves, NetLeaves), `KERNELS.md:29`, `scripts/slumbot_run.py` |
| Disk: C: 36.80 GiB free, E: 4,653 GiB; folder sizes | read 9 Nov 13:49 (`Get-PSDrive`, `du`); not in any record |
| Mine, not in a record: 0.740 solves/s (= 128,004 rows / 48.07 h), exponent 0.223 (the curve's last-doubling ratio 0.855 and D2b's wd-0 ratio 0.858 are exponents 0.226 and 0.221 and mix two recipes) and the extrapolations (-14.3%, -10%, MAE -8% to -12%), 0.0285 pp per 1% (one point), the power table (recomputed; the draft's table agrees within Monte Carlo noise), 1,000,008 = 18 x 27,778 x 2, 500,040 + 499,968 = 1,000,008, 10,204 = ceil(499,968 / 49), 375 h = 1,000,008 / 0.740 / 3600, ~21 days, 22.6 GB, the 1.9M-row memory scaling |

## Amendment 1: the recipe pin, d3s-rw (2026-11-10 16:07 SGT, before D1c's first label, S0a/S0c and any D1c compute)

Arm 2 (section 3) and section 5.3 left N2's recipe to a pin amendment that may name only (a) N1's recipe or (b) the one confirmed winner of the
sweep (`records/d3-sweep/PREREG.md`, commit 7d7dde2). The sweep ran on D1 + D1b only; no D1c label was read, no D1c net exists, and the gate halves were not read.
`records/d3-sweep/verdict.json` (sha256 d5a049e0a93fdfaca45ef1490b3b659cd5124a2e6cdf14547d4b0bd06250e2d1) and `RESULT.md` record the outcome. The user saw the verdict first and said to write this amendment.

**N2's recipe is `d3s-rw`: `{"dropout": 0.0, "ema": 0.0, "loss": "rw"}`.** Everything else is N1's, unchanged: width 2048, depth and batch and lr and 200 epochs from `LEAFNET_SETTINGS`,
wd 0.1, seed 0, data = union + D1c, the same hash split; pools are the driver's `pools_for` (4 at ~1.9M train rows; N1 and the sweep's nets used 2). `loss "rw"` trains on 0.5 plain Huber + 0.5 range-weighted Huber; N2 is selected (stage 3 and the looks) on the plain Huber as N1 was; Gate A is judged on MAE, the statistic the loss was meant to move.
`N2_ARM` in `scripts/d3.py` now reads `loss="rw"` (a test pins the whole recipe). The text of arm 2 and 5.3 stands as registered; where it says "N1's unless the sweep amendment says otherwise", this is that amendment.

| | seed 0 (selection, `selection.json`) | seed 1 (confirmation, `confirm.json`) |
|---|---|---|
| Control N1 | loss 0.011799, real rows 0.002217 | loss 0.011937, real rows 0.002213 |
| d3s-rw | loss 0.011320 (-4.06%, bar <= 0.011563), real 0.002181 | loss 0.011468 (bar <= 0.011817), real 0.002196 |
| Verdict | wins | confirmed |

The other four arms (dropout 0.05 and 0.10, EMA 0.999 and 0.9999) did not win; the combo was not called. Files: `selection.json` sha256 826b35e3597ed072f97400ec1515362b59e6204647018aaf9a3d7e4e540b92aa,
`confirm.json` e364da4bd25fcacc7fe8fb8b2aa0ca3c391203046c3de202c9fd8de0f6be69f5.

**The stand-in.** The range-weighted loss was the user's stand-in for the "EV-space loss" (choice 3 of the sweep registration; arm 2 above names it too). It is not the DEVN head (a later registration of its own), and a result for N2
is not a verdict on EV-space training.

**Limits, accepted in the sweep's registration and repeated:** winner's curse on the shared selection half; one seed pair; the real-row clause is close to a coin flip for a small effect;
the arm is trained toward MAE and judged on the plain Huber; and a recipe tuned at 951k rows is carried, not proven, at ~1.9M rows. Gate A on D1c's gate half tests it.

Nothing else changes: no arm, bar, gate, margin, branch rule or order of this registration, and N1's record is untouched. Section 6 stands: one candidate, one seed, no grid. The sweep is done; the
`d3-check` and `d3_sweep check` guards depend on the code trees, so a rerun on the final code is still the user's separate yes.

## Amendment 2: how d3-check is judged, by the net and not the file's bytes (2026-11-10 17:56 SGT, before any D1c label or compute)

Section 3 item 3 says `d3-check` "must give N1 bit for bit (as `d2b-check` reproduced N0)". The code stage turned that into "every compared field equal **and the
`net.pt` file's sha256 equal to N1's**". The run (`d3_load.py check`, commit 0d28864, 17:48 SGT; `records/d3/check.json` as it was written, sha256
fb4b4c2ad9c939e2516160a0d7507690afef6654eb40f0f3945d5d5dc67acd68 for the CRLF bytes on disk, 5c8feaef9e9970d534347deeb04adc94b949a8bca14d149c3ba136affb4407d0 for the LF blob committed
as 8516616, whose message and HANDOFF said "27 tensors": that counted `meta` keys too, it is 19) reported **NOT REPRODUCED on that one field only**:

- Equal to N1: the best epoch (194), the best selection loss, all 200 epochs' train, selection and MAE values, the config, and **all 19 entries of `state`**
  (every weight tensor, same dtype and `torch.equal`; checked on the CPU after the run). The two nets are therefore the same net.
- Different: the file's bytes (111,059,409 against 111,059,345) and so its sha256 (7aec31ff0ffe9f1e7ac72ba602a774ed1046b582f111d44e951e774dc313bc05 against N1's
  54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72). The only difference in the file is that its `meta` also holds `dropout: 0.0`, `ema: 0.0` and
  `loss: "huber"`, the three D3-SWEEP options (`scripts/turn_net.py`, `OPTION_KEYS`), which `vn.save` writes since the sweep's code stage and which N1 (e220bc0, 1 Nov) never had.

**The rule, from now on: `d3-check` passes when the best epoch, the best selection loss, every epoch's train and selection loss, the config, the net's width and depth, every tensor of
`state` (same dtype, bit-equal), and every `meta` key both nets hold are equal, and any key only one holds is one of the three options (`OPTION_KEYS`: dropout, ema, loss) at its off
default (`CFG_DEFAULTS`: 0.0, 0.0, "huber"; the type must match too, so `False` is not 0.0).** An option that is on, a missing or different tensor, or any other extra key fails. The file's sha256 is still computed and reported (`file_sha256`), no longer required.

This changes what the check proves in one respect and no other: it no longer proves the file's bytes are N1's, which holds for no net saved by code that adds a meta key. It does not widen
what a pass allows: N1's weights, history and config are reproduced exactly. The change is the user's choice (10 Nov question "How do we close d3-check?", answer "Tensor-level amendment"),
made after the result was seen and so not a pre-registered rule; it is named here as that, and it follows from a cause that does not depend on which way it helps (a metadata field, never a weight).

**What the code does:** `d3_load.py` gets `net_files_equal` (the rule above), `check_compare` takes it in place of the sha256, and a `recompare` command (a NOT REPRODUCED judgement leaves `check.json` untouched) judges the finished run again from the
two saved nets and `d3-check`'s record, on the CPU, training nothing: it moves the earlier `check.json` aside (kept, `check.json-void-1`) and writes a new one that carries the run's fields,
`recompared: true`, `run_reproduced` (false) and the code's commit. The run, its commit and its numbers stand; only the judgement is redone. Tests pin each case above.
Peak host memory of the run, reported only: working set 23.1 GiB, commit 15.0 GiB at 951,466 train rows (the sweep's check measured a 31.8 GiB peak commit under its own settings;
N2 trains at about 1.9M rows and 4 pools, so `d3 memory` measures it before `train --told`).

Nothing else changes: no arm, bar, gate, margin, branch rule or order of this registration. The sweep's own `check.json` is a separate record and is not touched; it guards only the sweep's
`train`, `select` and `confirm`, which are done, so a `d3_sweep check --redo` is needed only if any of them is ever run again.
