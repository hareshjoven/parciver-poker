# LBR-open and F2: result

Run 11 Oct 2026, 15:55–16:43 (`decide-20261011T075534Z.json`; per-deal arrays in
`runs/preflop-open/decide-20261011T075534Z/`), as PREREG-RUN.md fixes it.
- **Code:** commit `79109d1` (the pre-registration's commit), tree clean for the code under test.
- **Engine:** `E93347A5…` (this commit's build). The rating was re-verified on it first: `rating.py verify`
  replayed all six pool-v2 receipts bit-identically (every anchor match in full plus one 10,000-deal LBR
  chunk each) under `E93347A5573D899A+E819B132A991C676`, C6 vs pool-v1
  PASS, current = A2, audit OK, headline A2 PR 1665.6 (95% 1661.5..1669.8) VERIFIED. Full pytest: 601
  passed, 2 skipped, 0 failed.
- **Status:** the first run, complete, not VOID. It counts.

## Gates: all pass
- **G1:** standard peek-free LBR is bit-identical on v2 and F2 (+269.08 bb/100 both).
- **G2:** s = 300 (on the grid) is bit-identical on v2 and F2, for `all` and for `bp`.
- **G3:** `all` arms open 100,000 times; `bp` arms open 61,271 times on both bots (61.27%, against the
  root's registered 61.252%).
- **Descriptive:** standard LBR on v2 reproduces `translate-postfix/v2-2B-nopeek.npz` (3 Oct engine) per
  deal, exactly (max |diff| 0).

## Verdict: F2 is NOT adopted. (a) and (a′) fail: F2 is UP (worse) in all four arms
Δ = LBR-open_F2 − LBR-open_v2, bb/100 hands, paired over 100,000 deals (2 hands per deal; only the
small-blind hand can differ, so per 100 small-blind hands it is twice this):

| arm | Δ total | status | postflop part | preflop-ended part |
|---|---|---|---|---|
| 2bb, `all` | **+31.37** [+23.43, +39.32] | UP | +37.93 [+29.97, +45.88] | −6.55 [−6.66, −6.45] |
| 2.5bb, `all` | **+12.62** [+6.46, +18.79] | UP | +15.41 [+9.25, +21.58] | −2.79 [−2.86, −2.72] |
| 2bb, `bp` | **+20.65** [+14.22, +27.09] | UP | +24.70 [+18.26, +31.14] | −4.05 [−4.13, −3.96] |
| 2.5bb, `bp` | **+10.38** [+5.40, +15.36] | UP | +12.11 [+7.12, +17.09] | −1.72 [−1.78, −1.67] |
| 3bb (control), both | 0 | FLAT (identical) | 0 | 0 |

- (a) needs Δ(200, all) and Δ(250, all) DOWN: both are UP. **FAIL.**
- (a′) needs Δ(200, bp) and Δ(250, bp) not UP: both are UP. **FAIL.**
- (b), (c), (d) and the confirmation arm were **not run**: a departure from PREREG-RUN's Order, step 3–4,
  recorded here. The rule is a conjunction and (a) already fails, so none of them can change the
  verdict. (b) and (c) are identity checks for adopting F2; the confirmation arm "vetoes nothing" and
  exists to warn before an adoption. They can be run on request (`lbr_open.py confirm`,
  `pool_identity.py`, scorecard `--open-reading limp` + `lbr_open.py same-lbr`).

## Descriptive (decides nothing)
- **Each arm's LBR** (bb/100, 95% CI):

  | | std | 3bb all | 3bb bp | 2bb all | 2bb bp | 2.5bb all | 2.5bb bp |
  |---|---|---|---|---|---|---|---|
  | v2 | 269.08 | 272.43 | 263.00 | 241.04 | 241.47 | 254.52 | 250.42 |
  | F2 | 269.08 | 272.43 | 263.00 | 272.42 | 262.12 | 267.14 | 260.81 |

  CIs are about ±12 on each.
- **The reading's price, LBR-open(s) − LBR-open(3bb), per bot** (it mixes the misreading with the smaller
  pot, so FINDING ruled it decides nothing):
  - v2: 2bb all −31.39 [−36.26, −26.51], 2bb bp −21.53 [−25.53, −17.53], 2.5bb all −17.91
    [−21.47, −14.35], 2.5bb bp −12.57 [−15.49, −9.65]: all DOWN. LBR gains *less* by opening below the
    grid than by opening 3bb.
  - F2: −0.01, −0.87, −5.29, −2.19, all FLAT.
- **How the big blind answers the open** (`open_counts`: folds, raises; the rest call):

  | | `all` (100,000 opens) | `bp` (61,271 opens) |
  |---|---|---|
  | v2, every size (snap reads all three as 3bb) | fold 39.2, raise 16.5, call 44.3% | fold 39.5, raise 16.2% |
  | F2 at 2bb | fold 26.1, raise 25.0, call 49.0% | fold 26.3, raise 24.7% |
  | F2 at 2.5bb | fold 33.6, raise 20.1, call 46.3% | fold 33.9, raise 19.8% |

  - v2's fold rate matches FINDING's Slumbot sample (39.3%); its raise and call do not quite (Slumbot:
    14.7 and 45.9%, n = 2,560).
  - F2's defence at 2bb is 73.9%, FINDING's arithmetic 73.8%. That holds by construction (limp reads
    never fold, and 3bb reads play as v2), so it is a check of the code, not a finding.
- **Reading.**
  - At 2bb `all`, F2's wider defence saves 6.6 bb/100 preflop (13.1 points fewer folds, each giving up
    the 1bb blind) and loses about six times as much postflop (+37.9).
  - The likely cause is mostly FINDING's "known side effect". Under a limp reading, the big blind's
    abstract raise over a limp is played as a real 3-bet (`to_real` sizes it on the real pot). Its raise
    rate goes 16.5% → 25.0%, about 42% on limp reads. Those hands then play on in the limped-pot tree
    with a larger real pot, and these arms have no search.
  - The run does not split the postflop part by reading. Some deals that the agent read as 3bb differ
    too: 36,145 deals differ at 2bb, against about 33,300 limp reads. Peek-free LBR does not know the
    reading, so its own play changes on those deals.
  - F2's cost in the live config, where turn and river search re-solve with the real pot, is not
    measured: the confirmation arm was not run.
  - Against v2, LBR-open earns 12–31 bb/100 *less* by opening below the grid than at 3bb. v2's big blind
    acts the same at every size, so this price is mostly the smaller pot. It does not show that the
    misreading is harmless. It shows only that LBR-open's small open does not out-earn its grid open.

## What it decides
- F2 is not adopted. The live agent, the scorecard and the rating keep `snap` (the default); nothing
  changes for them. The engine keeps `with_open_reading` (default `snap`). With `snap` it plays as before:
  the rating's six receipts replay bit-identically on it, and standard LBR reproduces the 3 Oct arrays.
- **F1** (a new 200bb grid with 2bb and 2.5bb opens: per-street raise sizes, a new blueprint, days of
  CPU, a new rating) is allowed by FINDING's rule now that F2 failed. Given its cost, it starts only on
  the user's go (a project gate, not FINDING's). It would be judged by the same rule: Δ(a)/(a′) on these
  LBR-open arms, plus (d)'s `min_raiser`, `baseline_sizes_x0.6` and `baseline_realistic_sizes` rows.
  - Opinion, not a result: nothing here shows a leak for F1 to close. The price above is confounded,
    and F2's failure is mostly about the limped tree, which F1 would not use.
