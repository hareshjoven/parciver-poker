# LEAF-NET: a more precise turn-start value net at the same size (HANDOFF Next 3)

Registered 2026-10-12 (the commit that adds this file is the registration), before any LEAF-NET code, training
or data. Nothing here changes the live config until "Adoption" passes. Audited (read-only) before commit; the
audit's findings are folded in.

## Why
- M1 (FLOPSEARCH D4): the leaves carry the flop search. With check-down leaves it loses 3.5x what the blueprint
  does; with the net it removes 63% of the blueprint's flop loss. Better leaves are the most direct precision
  lever left in the live config.
- The live net N0 (`runs/turn-net/final-w2048-d5-e200/net.pt`, sha256 a8bf00be2fd1a3c7...; width 2,048, depth 5,
  200 epochs, best epoch 158; data 8df0fc79620a1820, 475,432 training samples) overfits:
  - train loss 0.0024 against selection loss 0.0145;
  - selection MAE 5.21% of the pot;
  - gate MAE 5.111% (real situations 2.876, random 7.345; 12,514 samples).
- N0 is trained with a uniform random suit relabelling of every sample, yet it is far from suit-symmetric (M1:
  isomorphic holdings up to 1.0 apart in probability).
  - FLOP-SYM averages N0 over the flop's own symmetries and the turn boards' stabilizers on 69% of flops.
  - The error stays wherever no symmetry of the job ties the inputs together.
- Training is cheap (~16 GPU-minutes per net, 945 s for N0), and a net of the same size runs at the same speed.
  Both changes below keep the architecture.

## The two changes (`value_net.ValueNet(width, depth, canonical=...)`, `turn_net.py --canonical --wd`)

### Canonical frame (`canonical=True`)
- The net relabels every input to its board's canonical suit frame, runs the inner network there and relabels
  the output back. The inner network is today's MLP plus the zero-sum layer, unchanged.
- **The frame:** k* is the SUIT_PERMS row whose relabelled, sorted board is smallest (`canonical_board`).
  - On a tie, k* is the lowest k: H_b, the board's stabilizer, is larger than the identity there.
  - It is computed as argmin over k of key·24 + k, where key = the sorted relabelled board's base-52 code, so
    the result does not rely on argmin's tie rule.
- **Graph-capturable:**
  - The frame comes from the board one-hot, inside `forward`, via topk, gather, sort and argmin. These are exact
    index operations: the one-hot is 0/1.
  - CARD_PERM, PAIR_PERM and PAIR_PERM_INV are non-persistent buffers. N0's state-dict keys are unchanged, and
    `forward` copies nothing from the host.
  - No nonzero, boolean indexing or `.item()` in `forward`.
- **What gets the frame for free:** `predict`, `net_values` and so the flop search (NetLeaves, SymLeaves) call
  `forward` and get it with no change to their code.
  - `FactoredLeaves` (`flop_leaves.py`) reads the MLP directly and would skip the frame. It refuses a canonical
    net (tested). It is not on the live path.
- **Equivariance:**
  - Where H_b is trivial, relabelled situations give the inner network the same input. The net is then exactly
    suit-equivariant there, bit for bit for inputs at the same row positions of the same batch shape.
  - On an unpaired rainbow flop (where FLOP-SYM does nothing: G is trivial), 46 of the 49 turns have a trivial
    H_b. The other 3 are the missing suit's cards that pair a flop card (|H_b| = 2; e.g. As Kh Qd + Ac). The
    dense path does not average them.
  - Where H_b is larger, the inner input is one of |H_b| relabellings. FLOP-SYM's coset average applies
    unchanged: the board set is the same, so the frame is the same, and the averaged inner inputs are still the
    distinct relabellings.
- **Training:** a uniform random relabelling per sample (as today), then `forward` canonicalizes. Training and
  inference run the same frame function. The zero-sum layer works in the frame, where r . v is unchanged by a
  relabelling.

### Weight decay (`--wd`)
- wd 0 uses `torch.optim.Adam` exactly as D2 did (the same code path).
- wd > 0 uses `torch.optim.AdamW` (PyTorch's decoupled decay: w <- w - lr_t·wd·w each step) on two groups:
  - the Linear weights with weight_decay = wd;
  - everything else (biases, PReLU slopes) with weight_decay = 0, set explicitly (AdamW's default is 0.01).
- The OneCycle schedule is the same.

## The grid and the selection (D2's protocol: the selection half picks, the gate half confirms)
- **8 nets:** canonical {off, on} x wd {0, 0.01, 0.1, 1.0}.
  - Fixed for all: width 2,048, depth 5, 200 epochs, batch 2,048, lr 1e-3, seed 0, D2's split.
  - The epoch kept is the one with the lowest selection loss (as D2's `final`).
  - Tags: `runs/turn-net/leafnet-c{0,1}-wd{0,0.01,0.1,1}/`.
- **Data check:** training refuses unless the data fingerprint is 8df0fc79620a1820 (D2's).
- **Records:** each record.json adds `canonical`, `wd`, the torch version and the commit.
- **(off, 0) is D2's recipe, retrained:** a check on the canonical=False / wd 0 path.
  - N0 was trained on the same torch (2.14.0+cu132, installed 3 Sep; N0 5 Oct).
  - Its best epoch and selection loss (the full float in record.json) must equal N0's record. D2's curve matched
    its sweep this way.
  - If they differ, the grid is not used until the cause is found and an amendment records it. Driver or cuBLAS
    changes are candidates; a code change on that path is a bug.
- **Pick P = the lowest selection loss of the 8.** If P is (off, 0), LEAF-NET stops: nothing beats N0's recipe.
- Expected (not a gate): canonical lowers the selection loss by a few percent; wd 1.0 is likely too strong.

## Gate A: the net's precision on the gate half (12,514 samples, used for no choice here)
- **The comparison:** P against N0, both in float32 (as G2 and the search run them), and both averaged over each
  sample's H_b as SymLeaves can at best.
  - Per sample, A(x) = mean over h in H_b of h^-1.N(h.x).
  - This removes the part of P's gain that FLOP-SYM's averaging already gives N0 on symmetric flops.
  - It is stricter than live play, where the 3 such turns of a rainbow flop and jobs that are not G-symmetric
    stay dense (not averaged). That is conservative.
- **The statistic:** per sample, D_i = MAE_P(i) - MAE_N0(i), the range-weighted MAE in % of the pot.
  - Mean of D, with a 95% percentile bootstrap CI: 10,000 resamples of canonical boards with their samples,
    `numpy.random.default_rng(0)`.
  - The board is the unit: the split is by board, and samples of one board are correlated.
- **PROCEED only if the CI's upper end < 0** (P superior). Otherwise LEAF-NET stops and N0 stays.
- P must also pass G2's bar: MAE under half the check-down's 18.646%.
- **Reported, not gating:**
  - the same comparison on the bare nets (no averaging);
  - the real/random split and the relative change;
  - the symmetry defect of both bare nets. Per gate sample: one random non-identity relabelling s
    (`default_rng(0)`, the same s for both nets), the range-weighted |N(s.x) - s.N(x)| in % of the pot. Mean and
    max, split by whether H_b is trivial (canonical nets: exactly 0 where it is).

## Gate B: M1 non-inferiority (decision precision; after Gate A)
- **The candidate:** the registered M1 harness (`scripts/flop_m1.py`), new candidate `leafnet` = sym leaves (FLOP-SYM
  C2, the live leaves) with P.
- **The baseline:** X_sym (N0; `m1-x-sym.npz`, `m1-v-sym.json`, loss 1.284% of the pot).
- **Fixed as in FLOP-SYM's M1:** the same 40 spots and menus, flop_iters 500, gamma 5, the same oracle and rule,
  and ISOMORPHISM on.
- **Harness changes (code, audited before data):**
  - `leafnet`'s pairing baseline is `sym`, not `net`.
  - The report checks X_leafnet's recorded net sha256 = P's and X_sym's = N0's. It does not require X_net's net
    path.
  - A leafnet spot's row is copied from `m1-v-sym.json` only if X_leafnet's strategies there are bit-identical to
    `m1-x-sym.npz`'s (not expected: the net changes every spot), and the report re-checks it.
  - z 1.96; output `records/leaf-net/m1-report-leafnet.json`. FLOPSEARCH's and FLOP-SYM's reports are not
    touched.
- **Engine pairing:**
  - X_sym's rows come from two older builds: 13 copied rainbow rows solved 6-7 Oct (X_net's) and 27 solved
    11-12 Oct. The engine was rebuilt since (E58C7292, 12 Oct, the AIVAT build).
  - **The check:** before the leafnet solve, two X_sym spots are solved again under the current engine, as a new
    candidate `symcheck`.
    - `m1-x-symcheck.npz` is a byte copy of `m1-x-sym.npz`; the solve uses `--threads 12` and copies no rows.
    - The two spots are the one with the lowest recorded `sec` among `m1-v-sym.json`'s 27 solved rows, and the
      one with the lowest `sec` among its 13 copied rainbow rows.
    - Their rows must equal `m1-v-sym.json`'s bit for bit in br, expl, lo, hi, value, profile, iters, curve,
      isomorphism and frozen. `sec` (wall time) and `copied` are excluded.
  - **On a mismatch:** all 40 spots are solved as `symcheck` before the leafnet solve, with no rainbow rows copied
    from `m1-v-net.json` (~17 more CPU-hours). That is the baseline.
- **The test:** D = Loss(X_leafnet) - Loss(X_sym), paired over the 40 spots; e = ORACLE_ERROR = 0.1% of the pot.
  - z 1.96: one candidate reaches M1, and the choice among 8 was made on the selection half. Gate A and Gate B
    must both pass (an intersection-union rule: no inflation).
  - **Pass (non-inferior): the CI's upper end + e <= 0.25% of the pot** (FLOP-SYM's margin).
  - A positive mean D within the margin passes, as in FLOP-SYM: Gate A is the superiority evidence, and M1
    guards the decisions against a net that is better on average but worse where play needs it.
  - The margin works as in FLOP-SYM's: after e, 0.15% is left for the mean plus 1.96 standard errors. The spread
    differs: here all 40 spots change, not 27.
- **Reported:** superior if the upper end + e < 0; Loss(X_leafnet) with its CI; per-spot D; CPU hours.
- **Cost:** ~18 CPU-hours (no spot copies expected), plus the two check spots. It never runs beside a
  blueprint-loading job (RAM).
- A verdict is given only when all 40 spots are in.

## Speed (after Gate B; only if P is canonical; the GPU otherwise idle)
- **The harness:** FLOP-SYM's (`scripts/flop_sym_speed.py`), extended to compare two nets with the same leaves.
  - `--net-a` = N0 and `--net-b` = P, each checked by sha256 (B's against X_leafnet's recorded net).
  - Jobs are X_leafnet's flop jobs. A = sym leaves with N0, B = sym leaves with P, in the order A B B A, with the
    same warm-up rules.
- Per job: ratio = (tB1 + tB2) / (tA1 + tA2).
- **Pass: over all jobs (rainbow included; the frame runs on every job), the median ratio's 95% CI upper end
  <= 1.02.** The CI is flop-clustered: 10,000 resamples, seed 0.
- The frame costs two gathers per call beside ~27.75M multiply-adds per row (5.54M + 16.78M + 5.43M), so under 1%
  is expected.
- A non-canonical P has the same ops as N0, so this step is skipped for it.

## Adoption (if Gates A and B pass, and speed when it applies)
- `scripts/slumbot_run.py` NET -> P's net.pt. run.json already records the net's sha256 and refuses a resume
  with another.
- **The mock gate:** LIVE-FLOP's / FLOP-SYM's is repeated. A registered 1,000-hand `--mock` run on committed play
  code must pass all of its checks and **flop decisions' p95 <= 3.0 s**.
  - `mock_check.json` is committed as `records/leaf-net/mock_check.json`, and GATES points at it.
  - A fail leaves the live config on N0.
- **No fallback:** after a fail at Gate A, Gate B, speed or the mock, no runner-up net is tried (e.g. a
  non-canonical net after a speed fail). Any new candidate needs a new registration.
- **Rating (reported, not a gate):** RATING.md Amendment 3's protocol on the new live config, label
  `live-leafnet`, 20,000 deals vs A2. Its play identity changes with the net.
  - Before NET is switched, `live-flopsym` is re-verified under the code with the canonical frame. Its replay
    runs N0 on the canonical=False path, which must be bit-identical.
  - Reported: `live-leafnet`'s PR (AIVAT and raw), and its paired per-deal change against `live-flopsym`. The
    change is computed from the two receipts' stored arrays: both are played on the same rating seeds, the same
    deals.
  - The rating is too coarse to gate a leaf change (+/-82 points per receipt). M1 is the gate.

## Order and compute
1. **Nothing under `src/parciver/`, `scripts/slumbot_run.py`, `scripts/rating.py`, the engine, N0, the rated
   blueprint (`runs/blueprint-20260927-141645/blueprint.bin`) or the .venv's packages (torch, triton) changes until
   the live-flopsym rating's receipt is written.**
   - The rating's identity covers them, and a change at its end means no receipt (`rating.py` refuses).
   - Then: the code, the tests below, an audit, and `rating.py verify` on pool-v2, so that live-flopsym and the
     blueprint receipts replay under the new code.
2. The grid, then P, then Gate A.
3. Gate B: the two engine-check spots, then strategy (GPU), then solve (CPU).
4. Speed, the re-verify of live-flopsym, the commit that switches NET, the mock, and the rating.
   - The re-verify must run before the switch: the mock plays only committed code, and once NET is P,
     `live_agent` refuses to replay live-flopsym.
   - After the switch, live-flopsym shows STALE for good (the net is in its play identity). The reported change
     is the one computed from the stored arrays.

Each run is reported when it ends; nothing is decided at an interim look.

## Tests before any data
- **`canonical=False`:**
  - forward is bit-identical to today's on random inputs with N0's weights;
  - a blob without the flag loads as canonical False;
  - the state-dict keys are unchanged.
- **`canonical=True`:**
  - Equivariance: with random weights and boards with trivial H_b, N(s.x) == s.N(x) bit for bit for random
    relabellings s (CPU float32, the same row positions of the same batch shape).
  - Direction:
    - The inner input is a valid situation. Its ranges are 0 on every holding that touches the inner (canonical)
      board, its `live` equals live_mask(inner board), and its board is `canonical_board(b)`.
    - Canonical forward equals an explicit reference: `value_net.permute(k*)`, the inner network, then
      relabelled back.
  - On boards with a larger H_b, the inner input of s.x is the inner input of x relabelled by a member of the
    canonical board's stabilizer.
  - r0 . v0 + r1 . v1 = 0 to rounding.
  - The frame index k* equals a Python first-minimum on 10,000 random boards and on every board class.
  - A CUDA graph replay equals eager execution: the flop search captures graphs.
  - `FactoredLeaves` refuses a canonical net.
- **Save, load and training:**
  - `save` / `load` keep the flag.
  - wd > 0 decays exactly the Linear weights, and the other group's weight_decay is 0.
  - wd 0 trains bit-identically to the D2 code path on a tiny run.
  - training refuses data with another fingerprint.
- **Suites:** the flop search's tests and selftests pass, and FLOP-SYM's random-evaluator tests pass with a
  canonical net. The M1 report's leafnet pairing and the speed script's two-net checks are tested on toy files.

## VOID and bugs
- A crash, kill or reboot; another compute job on the GPU (training, speed); or an overloaded oracle machine
  (M1): that run is VOID, its cause recorded, and it is repeated.
- A bug found after any data: that run is VOID and still reported. The fix and the rerun need an amendment here
  first, and both results stay in the record.

## Not allowed after this commit
- Changing the grid, the selection rule, the gates, the margins, the spots or the order after any LEAF-NET data.
- Choosing P on gate-half or M1 numbers.
- Trying a runner-up after a fail.

## Amendment 1 (12 Oct 22:35, with the code, before any LEAF-NET data)
- **No copied leafnet rows.** Gate B's "a leafnet spot's row is copied from `m1-v-sym.json` only if ..." is
  dropped: every leafnet spot is solved under the current engine, and the report refuses a copied leafnet row.
  Simpler, and every leafnet row then comes from the build that solves the baseline check (none were expected).
- **The harness, as coded:** `flop_m1.py symcheck-init` (the byte copies), `enginecheck` (picks the two spots,
  compares the rows, writes `records/leaf-net/enginecheck.json` with the baseline `sym` or `symcheck`), and
  `report` (Gate B against that baseline). `flop_sym_speed.py --cand leafnet --net N0 --net-b P` writes
  `records/leaf-net/speed.json`. Training and Gate A: `turn_net.py leafnet`, `turn_net.py gate-a`.
- **The order is enforced (the code audit; committed 12 Oct 23:08, d5a55d7; the time first written here, 23:40, was wrong):** `strategy --cand leafnet` and the report refuse
  unless `gate-a.json` says proceed for P's sha256. symcheck and leafnet solves record the engine build's sha256
  and the threads in their inputs (a resume under another build is refused); `enginecheck.json` records that build
  and the sha256 of `m1-v-sym.json` / `m1-x-sym.npz`, and the report refuses a leafnet (or symcheck) file solved
  under another build, or X_sym's files changed since the check. The speed step refuses a non-canonical P and
  reports each side's symmetric (non-dense) solves.
