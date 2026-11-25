# LEAF-EV: the leaf net predicts expected values (IDEAS SC5, scout)

Registered 2026-10-14 (the commit that adds this file is the registration), before any SC5 code, training
or data. The user's go: "both if it's fine" (14 Oct, SC2 and SC5 after LEAF-NET's Gate B). A scout: it changes
nothing live. A PROCEED only leads to Gate B and adoption, which an amendment registers with the user's go.
Audited (read-only) before commit; the audit's findings are folded in.

## Why
- **The leaf net is the largest error measured on the flop path.**
  - The flop solve is within 0.20% of the pot at 500 iterations on the check-down game (FLOPSEARCH D3).
  - D1's labels are within 0.23% (value MAE) of 3,000-iteration references (FLOPSEARCH: 120 validation
    solves).
  - The leaf net's value MAE is 4.650% of the pot for P and 5.107% for N0 (LEAF-NET Gate A).
  - How much of it shows in play is what M1 measures.
- Our net predicts each holding's value per unit of the opponent's **total** mass, in pots: v_p[h] / (M_q * pot)
  (`turn_values.py`:19-20). That is DeepStack's normalization.
  - Part of that number is card removal. A holding that blocks much of the opponent's range has a smaller
    value, and the net must learn that product.
- DEVN (AAAI 2023) splits it. The counterfactual value is EV x matchup, where the matchup is the opponent mass
  the holding does not block. The net predicts the EV; the matchup is computed exactly.
  - They report 3.37-8.39% lower loss without card abstraction (9.18-15.70% with buckets).
  - Their data were DeepStack-style random-range subgames, their nets 2-7 layers of 500-1,024. Half of D1 is
    real ranges, and our net is 5 x 2,048.
  - They train on EV targets. This scout tests only the factorization, under today's loss (below).

## The change (`value_net.ValueNet(width, depth, canonical=False, matchup=False)`; default unchanged)
- **`matchup=True`:** in `inner`, the output layer's values (float32, autocast off, dead holdings 0) are
  multiplied per player and holding by the matchup. Then DeepStack's zero-sum step runs, unchanged
  (`value_net.py`:174-175).
  - The matchup is m_p[h] = M_q - c_q[a] - c_q[b] + r_q[h], where:
    - r_q is the opponent's input range row as given;
    - M_q = r_q.sum(), not 1, since training ranges are stored in float16 (`turn_net.py`:55);
    - c_q = r_q @ INC is its mass per card (INC: the 1,326 x 52 holding-card incidence);
    - (a, b) are h's cards.
  - This is the fold terminals' inclusion-exclusion (`flop_search.py`:455-456).
  - It is computed in float32 with autocast off: the hidden layers train under bfloat16 autocast, and a
    matmul inside it would round to bfloat16.
- **What the network learns:** its raw output is the EV per unit of unblocked opponent mass, in pots. The
  matchup supplies card removal exactly.
- **The loss is unchanged:** today's Huber loss in today's value space, over the live holdings. This is a
  choice, made now, of one arm:
  - It isolates the parameterization. The loss, the selection metric and the zero-sum layer stay as they
    are.
  - No target is divided, so there is no 0/0 and no weighting choice. A holding whose opponent range is
    fully blocked has a matchup of 0 (up to float rounding) and a label of 0. Before the zero-sum step its
    output is 0 too, up to the same rounding; the step then shifts it like any live holding.
  - DEVN's own EV-space loss is not tested. One arm means no minimum-over-arms bias.
- **Canonical frames are refused with `matchup`** (a ValueError). P is c0; the pair is untested and not needed.
- **Equivariance:** the matchup commutes with suit relabelling (INC relabels with the holdings), so the H_b
  averaging and the symmetry checks apply unchanged.
- **Save and load:**
  - `save` writes `matchup: True` only for such nets, so a plain net's blob is unchanged.
  - `load` passes `blob.get("matchup", False)` on.
  - The INC buffer and the card indices are non-persistent, so the state-dict keys stay a plain net's.
- **Training records:** `train` gains `matchup=False`, written into `cfg`. `CFG_DEFAULTS` gains
  `matchup=False`, so earlier records still match on reuse.
- **What gets it for free:**
  - `predict`, `net_values` and so the flop search (NetLeaves, SymLeaves) call `forward`.
  - `FactoredLeaves` reads the MLP directly and would skip it, so it refuses a matchup net, as it refuses a
    canonical one. It is not on the live path.
- **Tests before any data** (`tests/test_value_net.py`, `tests/test_leaf_net.py`):
  - The default path is unchanged: the existing tests pass, and a plain net's forward and state dict are
    bit-identical.
  - The matchup equals a brute-force float64 sum over disjoint holdings, for ranges whose mass is not 1.
  - The zero-sum identity holds with `matchup`.
  - Under autocast the matchup is computed in float32.
  - A suit relabelling commutes with forward, up to float rounding.
  - Save and load round-trip, and a plain net's saved blob has no `matchup` key.
  - `canonical=True, matchup=True` raises.
  - `FactoredLeaves` refuses a matchup net.
  - CUDA-graph replay of a matchup net's leaf call is bit-identical to eager, as the flop solver captures it.

## The scout (`scripts/turn_net.py leafev`)
- **Data:** D1 through D2's split. Training refuses unless the fingerprint is 8df0fc79620a1820.
- **The baseline P is pinned:** `runs/turn-net/leafnet-c0-wd0.1/net.pt`, sha256 0c196d46...d45489
  (`records/leaf-net/gate-a.json`), checked at the start; and its `record.json` as quoted in step 1.
- **Step 1, reproduction:**
  - P's recipe is retrained with the new code, matchup off: c0, wd 0.1, width 2,048, depth 5, 200 epochs,
    batch 2,048, lr 1e-3, seed 0, the full training set.
  - Tag: `runs/turn-net/leafev-repro-c0-wd0.1/`.
  - Its best epoch and selection loss (the full float) must equal P's record (`leafnet-c0-wd0.1`: epoch 183,
    0.013437177158544897).
  - If they differ, nothing else runs until the cause is found and an amendment records it. Driver or cuBLAS
    changes are candidates; a code change on that path is a bug.
- **Step 2, the candidate S:** P's recipe with `matchup=True` (`runs/turn-net/leafev-c0-wd0.1/`). The epoch kept
  is the one with the lowest selection loss, as D2's `final`.
- **Step 3, selection:** S's selection loss must be below P's (0.013437177158544897). Otherwise SC5 stops.
- **Step 4, the gate half: LEAF-NET's Gate A statistic, S against P.**
  - `leafev` has its own gate step. It reuses Gate A's functions (`hb_average`, `weighted_mae`, `evaluate`,
    `symmetry_defect`, `board_bootstrap`) with S as the candidate and P as the baseline.
  - `cmd_gate_a` and `records/leaf-net/gate-a.json` are not touched.
  - 12,514 samples, used for no choice here.
  - Both nets run in float32, each averaged over each sample's H_b.
  - D_i = MAE_S(i) - MAE_P(i), the range-weighted MAE in % of the pot.
  - Mean, with a 95% percentile bootstrap CI: 10,000 resamples of canonical boards with their samples,
    `numpy.random.default_rng(0)`.
  - **PROCEED only if both hold:**
    - the averaged CI's upper end is < 0 (S superior);
    - S's bare MAE (no averaging) is under G2's bar: half the check-down's 18.646%, as Gate A checks it.
  - **Reported, not gating:** the bare nets' D and CI, the real/random split, the relative change, and both
    nets' symmetry defect.
- **Output:**
  - `records/leaf-ev/gate-a.json`: both sha256s, the data fingerprint, the commit, the torch version, and the
    reproduction's result.
  - Each net's `record.json` adds `matchup`.
- **Cost:** two trainings of ~15 GPU-minutes each (P: 896 s for 200 epochs) plus the gate half's evaluation.
  Under an hour (an estimate).

## After the scout
- **PROCEED:** an amendment registers, with the user's go:
  - Gate B: M1 non-inferiority against the live leaves' M1 row;
  - the speed check: the matchup adds a [B x 2, 1,326] x [1,326, 52] product per leaf call, expected under 2%
    of a call (an estimate);
  - adoption.

  If PROCEED, B (D1b)'s net trains with `matchup`.
- **STOP:** reported in IDEAS.md; SC5 closes for the factorization under today's loss.
  - A STOP does not test DEVN's EV-space loss, which stays listed in IDEAS as untested.

## Order
- **The code lives on its own branch** (`scout-sc5`, in the worktree `-sc5`; SC2
  has its own), and the scout runs from there. `rating.py` hashes every `src/parciver/**/*.py` into the play
  identity (`scripts/rating.py`:213-249), so main and the live receipts stay untouched.
  - It is run with `--data` set to the main checkout's `runs/turn-data`, so its nets land in the main
    checkout's `runs/turn-net/` (untracked).
  - **The result goes to main whatever the verdict:** `records/leaf-ev/gate-a.json` and the report, records
    only, no code.
  - Only a passed Gate B merges code into main, followed by `rating.py verify`, as the amendment will say.
  - A failed scout's branch is kept as an `archive/*` tag.
- **The scout runs after LEAF-NET's Gate B and its runbook step 5**, with the GPU otherwise idle.
  - It compares S with P whatever LEAF-NET's Gate B or mock found: P is the recipe it changes.
  - If P did not go live, the amendment's Gate B still goes against the live leaves' M1 row.
  - SC2 and SC5 run one at a time.
  - Training holds D1 in RAM (~9 GB or more), so it never runs beside an M1 oracle or a blueprint-loading job.

## VOID and bugs
- **VOID:** a crash, kill or reboot, or another compute job on the GPU. The run is VOID, its cause recorded, and
  it is repeated from scratch (same seed).
- **A bug found after any data:** that run is VOID and still reported. The fix and the rerun need an amendment
  here first, and both results stay in the record.

## Not allowed after this commit
- Changing the recipe, the seed, the selection rule, the gate statistic, the bar or the data after any SC5 data.
- Adding an arm after seeing results (an EV-space loss, another wd, canonical frames). A new arm needs a new
  registration.

## Amendment 1 (14 Oct 21:35, with the code, before any SC5 data)
- **The relabelling test, corrected.** "A suit relabelling commutes with forward" cannot hold: P is c0, a plain
  MLP, and its input layer is not suit-equivariant, with or without `matchup`. The registered test was wrong as
  written. What "Equivariance" above rests on is the matchup factor itself, so the test is instead:
  `matchups(permute(r)) == permute(matchups(r))` in float64, up to rounding (1e-12). The H_b averaging and the
  symmetry defect are measurements on the gate half, not assumptions, so nothing else changes.
- **Where the tests live:** `tests/test_leaf_ev.py` (new). The existing `tests/test_value_net.py` and
  `tests/test_leaf_net.py` run unchanged.

## Amendment 2 (15 Oct 06:43, from the code audit, before any SC5 data)
- **Every run trains both nets from scratch.** The code reused a finished net with matching settings, which
  contradicts "repeated from scratch" above. Now `leafev` never reuses or overwrites. It refuses to start while
  `records/leaf-ev/gate-a.json`, `runs/turn-net/leafev-repro-c0-wd0.1/` or `runs/turn-net/leafev-c0-wd0.1/`
  exists. A stopped run's record and nets are moved aside under new names (`gate-a-void-<n>.json`, `<tag>-void-<n>`)
  and kept, so both results stay in the record.
- **P is checked first.** P's sha256 and its record are checked before the data loads (~9 GB).
- **A non-finite loss** (training stops at once, as D2's): in P's recipe it is NOT REPRODUCED, so nothing else
  runs. In S there is no selection loss, so S is not below P's and SC5 stops. Either way the record keeps the error,
  and if its cause is a bug, the bug rule above applies.
- **Every stop after the start writes `gate-a.json`**, with its reason.
- **The record also holds:**
  - each training's commit and torch version, and its net's sha256;
  - the GPU state at the start and the end (`nvidia-smi`: load, memory, any other Python process; under WDDM the
    desktop's apps are listed and are not compute jobs).
- **At the gate:** both nets' sha256s must still be the ones trained and pinned, and each net file must be the one
  its record names (S: `matchup` on, its best epoch; P: `matchup` off, epoch 183). Otherwise the run is refused, with
  the reason in the record.
- **Tests:** `tests/test_leaf_ev.py` (the refusals, the from-scratch rerun, the NaN stops, the gate path on toy nets
  on the GPU).
