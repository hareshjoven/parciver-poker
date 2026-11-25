# D3-SWEEP: which training recipe trains N2 (dropout, EMA, a range-weighted loss), on D1 + D1b only

**Status: REGISTERED 9 Nov 2026 21:39 (registration only).** The user's yes in this session ("Sweep registration"), after D3's
code stage (PREREG 5.0, commits 9193ae6..1b319bc, HANDOFF). Written before any sweep code, training or data use. No net
has been trained or evaluated for it. Only existing records (`runs/turn-net/d2b-w2048-wd0.1`, `d2b-seed1`) and
the D3 PREREG were read. Nothing here changes the live config. Its own code stage and its run each need the user's
separate yes (see "The user's choices"). Audited read-only before going to the user; the audit's real findings are
folded in.

## The question this run decides
**Which one training recipe does N2 use?** D3 (records/d3/PREREG.md, arm 2 and 13) trains a single candidate N2 on D1 + D1b + D1c.
It never runs a grid itself: a grid read on the data that then judges the winner is a selection bias, and one candidate has
none. So the recipe is chosen here, on D1 + D1b alone (951,466 train rows, the selection half 23,728 rows), before D1c
has a single label read. The answer is one of two: **N1's recipe** (nothing beat it by the bar) or **one named winning
recipe**. An amendment to the D3 PREREG pins it (see "Order"), before D1c's first label is read.

**What is honestly expected (my reading, not a result).**
- There is room: N1's best epoch has train loss 0.00263 (with augmentation) against selection loss 0.01180, 4.5 times higher, and
  the selection loss was still falling slowly at epoch 150-200 (0.01201 -> 0.01180). A net this wide on 0.95M rows is
  overfitting in the plain sense, so regularisation and weight averaging have something to act on.
- It is not large. Weight decay 0.1 already took the union's selection loss from 0.012736 (wd 0) to 0.011799 (-7.4%); the seed spread alone is
  +1.17% (d2b-seed1 0.011937 against 0.011799). A winner will be 2-5% if there is one, and "no winner" is a likely and
  fully acceptable outcome: N2 then trains on N1's recipe, which is the cleaner experiment (data is the only change).
- The range-weighted loss is the least likely to help: it gives low-range holdings less gradient, and the search later raises those
  ranges. That is why it is mixed with the plain loss, not used alone.
- The range-weighted arm is also structurally disadvantaged: it is trained toward MAE-like error but judged on the plain Huber (SC5 showed the two can diverge:
  MAE -1.3% while the selection loss was +0.45%). That is accepted, and named here so a loss of that arm is not read as a verdict on EV-space training.
- Nothing here predicts strength. N2's value is decided by Gate A and B in D3, as before.

## Fixed for every arm (N1's recipe, `tn.LEAFNET_SETTINGS`)
w2048-d5, 200 epochs, batch 2048, lr 1e-3 OneCycle (2% warm-up), AdamW wd 0.1 on the Linear weights only, bf16 hidden / float32 output
layer, Huber (delta 1), suit augmentation, non-canonical, pools 2, seed 0 (the confirmation net: seed 1), data D1 + D1b
(`tn.D2B_UNION` fingerprint `7d5a77c30aaad5d6`; the driver refuses any other). Only the three options below move.
Rows go up once and are held for all nets (`held`), as in D2b's sweep.

## Arms
**The control is N1's own record** (`d2b-w2048-wd0.1`, no rerun): selection loss 0.011798965392366745 (epoch 194), real rows
0.002216822379196486, random 0.021381108084001496 (all on the 23,728-row selection half), float32. Its seed-1 twin
(`d2b-seed1`, 0.011936810236883902, epoch 196) is the control for the confirmation net. A rerun would be 1 h of GPU for a
number already recorded to the bit; the pre-check below ties the new code to the record instead.

| tag | dropout | EMA | loss | family |
|---|---|---|---|---|
| `d3s-drop05` | 0.05 | - | Huber | dropout |
| `d3s-drop10` | 0.10 | - | Huber | dropout |
| `d3s-ema999` | - | 0.999 | Huber | EMA |
| `d3s-ema9999` | - | 0.9999 | Huber | EMA |
| `d3s-rw` | - | - | 1/2 Huber + 1/2 range-weighted Huber | loss |
| `d3s-combo` (round 2, only if the rule below fires) | best winner's | best winner's | best winner's | - |
| `d3s-confirm` (seed 1, only for the overall winner) | the winner's | the winner's | the winner's | - |

Definitions (the code stage implements exactly these):
- **Dropout p:** inverted dropout (`F.dropout`, functional, driven by a plain attribute `p` that is not a module and not in the state dict) after each of the 5
  hidden PReLUs; none on the input or the output layer; off in eval and at `p = 0`. **`net.mlp` is never changed**: no `nn.Dropout` modules, so a saved net's state-dict keys are N1's and
  `value_net.load` (which builds `ValueNet(width, depth, canonical)`, p = 0) loads it unchanged. The recipe's `p` is kept in the record and the save's meta only.
- **EMA d:** a shadow copy of all parameters, `e <- d*e + (1-d)*w` after every optimiser step, `e` starting at the initial weights, no bias
  correction or warm-up. Every evaluation and every saved epoch uses `e`; the buffers (`scal_mean`, `scal_std`) are copied. With
  464 steps an epoch, d = 0.999 averages over ~2 epochs and d = 0.9999 over ~22.
- **Range-weighted Huber:** per player-sample `sum_i r_i * huber(pred_i - true_i)` over the live holdings (`r` the range, mass 1), averaged over
  the batch and both players; the training loss is `0.5 * plain + 0.5 * range-weighted`. It is the loss shaped like the
  search's own error (`value_net.range_mae`). It is not DEVN: DEVN's net head (the holding's expected value, the matchup computed
  exactly) is a different net and needs its own registration (IDEAS SC5 tested only the matchup factor under Huber,
  +0.45%, STOP). Not in this sweep.
- **Selection is always the plain Huber** on the selection half, in float32, eval mode, EMA weights if EMA: the same number
  N1's record holds, so every arm is comparable to it. The range-weighted number is reported only.

## Selection and the decision
Only the selection half (23,728 rows) of D1 + D1b decides anything. The gate halves (24,814 rows) are not read by the sweep.
No D1c row exists, none is read.
1. **Within a net:** the epoch with the lowest plain selection loss, as `train` already does.
2. **An arm wins** iff its best selection loss is **<= 0.98 x N1's (0.011563)** AND its loss on the real rows is **<= N1's real-row loss
   (0.002216822...)**. 2% is 1.7 times the seed spread; the real-row clause is Gate A's "no regression on what the bot sees", moved
   here.
3. **Round 2.** If two or more families (dropout, EMA, loss) each have a **winning** arm (rule 2), train `d3s-combo` once: the lowest-loss winning arm of each winning family, combined.
   If fewer than two families won, no combo is trained. The combo must itself pass the rule-2 clauses, and it goes to the confirmation **in place of the best single winner only if its
   selection loss is <= 0.99 x the best single winner's**; otherwise the best single winner goes (a noise-level combo never displaces a confirmable single recipe).
4. **Overall winner:** the best single winner, or the combo under the rule above. No winner -> the pin is N1's recipe.
5. **Confirmation.** The winner's recipe is trained once more with seed 1 (`d3s-confirm`). It is confirmed iff it is **<= 0.99 x 0.011936810236883902
   (0.011817)** and its real-row loss is <= d2b-seed1's real-row loss (0.0022133768046939204, in records/d2b/selection.json; `check` re-derives it, see below). Confirmed ->
   the pin is the winner. Not confirmed -> the pin is N1's recipe (a combo that fails does not fall back to a single recipe: the rule's order is the only one). **No second place and no other seed.**
6. A tie or a NaN: a non-finite loss stops that arm (`train` raises `FloatingPointError`); that is a **failed arm**, not VOID: it is recorded, it cannot win, and it is not retried (the same seed would repeat it). Exact ties go to the arm lower in the table.

The pin, whichever it is, is a fact about D1 + D1b at 951k rows. N2 trains on ~1.9M rows (about 928 steps an epoch, not 464), so a recipe tuned here is carried, not proven, there; D3's own
Gate A on D1c's gate half is what tests the result. **If EMA is pinned, it is pinned as its horizon in epochs** (0.999 = ~2, 0.9999 = ~22) and N2 uses the per-step `d` that gives the same epoch horizon at
its own steps per epoch; the amendment states that number.

**Limits, accepted and written into the verdict.**
- **Winner's curse.** The selection half decides the sweep and is the same 23,728 rows D3 reuses for its looks (<= 0.98 and <= 0.96 x N1) and its stage-3 stop rule. A recipe pinned for beating N1 on them has a
  head start there. D3's looks and the verdict note it; only Gate A on D1c's gate half is clean.
- **The confirmation is weak:** one seed pair gave the 1.17% spread, and 1% is 0.85 times it, so it removes some noise, not all. The real-row clause is close to a coin flip for a small true effect (real rows are ~9% of
  the selection loss and moved 0.16% between the two seeds), so it can sink an arm that gained on random rows. Both are accepted: the alternative is a bigger grid.

## Reported only (no decision rests on them)
Every arm: MAE (% of pot), the range-weighted loss, the real/random split of the selection loss, best epoch, train loss, the per-epoch
curves, time. The recipe's cost at inference is nil (dropout is off, the EMA is a weight set): the speed test of D3 is unaffected.

## Code tests first (before any data)
The code stage changes `src/parciver/value_net.py` and `scripts/turn_net.py` (options `dropout`, `ema`, `loss_rw` in `train`, the CLI
and the record's `cfg`), with a driver `scripts/d3_sweep.py` (`check`, `train`, `select`, `confirm`, `verdict`).
- **A golden test, written and passed on today's code before the code changes:** a tiny run (width 32, depth 2, 3 steps, CPU, seed 0) with the
  options off; its parameters' sha256 and the first losses are pinned in the test. The same test must pass after.
- The off path is the old path: no extra module in `net.mlp` (so a saved net's state-dict keys are N1's and `value_net.load` is untouched), no extra RNG
  draw, no extra pass over the data, the same `cfg` keys for the options off (`CFG_DEFAULTS` gets `dropout 0, ema 0, loss "huber"`, so old records
  still match under `reuse`).
- Dropout: zero in eval, inactive at p = 0, the active fraction ~ p, one RNG stream with the seed. EMA: equals the formula after k steps,
  eval and save read the EMA, the live weights keep training. Range-weighted: equals a hand sum on a tiny batch, 0.5 + 0.5 combination, dead
  holdings carry nothing.
- A saved EMA/dropout net loads in `value_net.load` and `predict` unchanged and gives the in-training eval number to float32.
- The golden values are pinned from this machine's CPU and torch build; a torch change rebuilds them by amendment, not by editing the test.
- `N2_ARM`, `net_row` (nets.json) and `d2b_train_arm` carry the recipe's three options, so N2's record names its recipe.
- The driver refuses: other data than `D2B_UNION`, an arm not in the table, a second `train` of a finished arm without `--redo`, an arm trained
  before `check` passed, a round 2 or a confirmation that the rule above does not call for, any other GPU job in flight, and a
  free commit below the peak that `check` measured at this size plus D3's 0.6 GiB margin (as D2b and D3: it tells the user first).
- An audit by a read-only subagent per piece, one at a time, before the user hears "done".

## Pre-check (before any arm)
`check` does three things, and an arm trains only after all three pass (`check.json`):
1. N1's recipe through the new code (options off), the first **3 epochs** of the 200-epoch schedule, reproduces `d2b-w2048-wd0.1`'s history for epochs 1-3 (`train_loss`, `val_loss`) to the bit. (The
   OneCycle schedule is the full run's, only stopped early; 3 epochs are mostly warm-up, so this ties the data path, the RNG and the new code's off path, not the whole schedule.)
2. It rescores both saved nets (N1 and `d2b-seed1`) by source on the selection half and equals `records/d2b/selection.json` to the bit: all, real (N1 0.002216822379196486; seed 1 0.0022133768046939204), random.
3. It records the peak commit of the process at this size (D3's `memory` way) for the guard below.
If any differs, nothing else is used until the cause is found and an amendment records it. A full 200-epoch rerun is not worth an hour. **D3's own `d3-check` (N1 bit for bit on the new loader, D3 PREREG 9) must
also be run on the final code, after this code stage, before N2 or a look**: the sweep changes `train` and `value_net`, and that check is what ties N2's path to the record.

## Machine needs and cost
One GPU, alone: any other GPU compute job voids the arm in flight (a desktop app's use is reported, not void; the D2b/D3 rule). Host memory as N1's training (pools 2,
~one training set): D2b only estimated it (31-37 GB), so `check` measures it and the arms refuse below that peak plus the 0.6 GiB margin, and say so. About 48-51 min a net (N1: 2,811-3,021 s; with pools 2 each epoch
re-uploads the pools, ~950 s of that total, so rows held saves only the host-side preparation): **round 1 (5 nets) ~4.3 h; combo +0.9 h; confirmation +0.9 h; the check ~10 min; ~6.2 h at most,** before the options' own small overhead (reported). The CPU stays free except the parent at
BelowNormal. No cloud.

## Order (each stage only on the last one's pass)
code stage, tests, audit -> `check` -> round 1 -> `select` (writes `selection.json`) -> round 2 if called for -> confirmation if there is a winner -> `verdict`
(`records/d3-sweep/verdict.json`, `RESULT.md`) -> **an amendment to records/d3/PREREG.md that names only (a) N1's recipe or (b) the one confirmed winner**, written and committed before D1c's first
label is read, and the D3 driver's N2 arm (`N2_ARM`, `scripts/d3.py`) updated to match under its own tests. The amendment also says that the range-weighted loss was the user's stand-in for D3's "EV-space loss" (choice 3). The user sees the verdict before the amendment.

## VOID and bugs
- **VOID is an external cause only:** a net stopped before epoch 200 by a killed process or a reboot, or trained while another GPU job ran. It is moved aside with the cause (a row in the sweep's own void list; `d3.void_net` refuses once a result
  file exists, so the sweep defines its own "result exists": `selection.json`), retrained from scratch with the same seed, and not read for any choice meanwhile. A NaN is a failed arm (rule 6), never VOID.
- A bug found in the new code after any arm's result is read: stop, and write an amendment first, naming the bug and the arms it could touch; only then fix with a test and rerun those arms from scratch. **Both results stay in the record.** The direction of the fix is never taken
  from which arm it would help.

## Not allowed after registration
- Changing the grid, the bars (0.98, 0.99, the real-row clauses), the loss mix, the EMA or dropout values, the epoch count or the seed, after any arm's loss is seen.
- Adding an arm, dropping an arm, or a different combo: the combo follows the rule above, nothing else.
- Choosing by anything but the plain selection loss and the stated clauses (MAE, the range-weighted number and the curves are reported only).
- Reading the gate halves, the D1c rows or labels, or anything of N2; tuning on Gate A.
- A third seed, a rerun to "see again", or a second place when the confirmation fails.
- Pinning a recipe that is not one of the two outcomes; changing N1's record (never touched).

## The user's choices (made by the registration; changeable only by an amendment before any arm's loss is seen)
1. **The grid** (above): dropout {0.05, 0.10}, EMA {0.999, 0.9999}, a mixed range-weighted loss, a rule-fired combo, one seed-1 confirmation. Recommended: this, ~6 h.
2. **The bars:** 2% on selection plus no regression on real rows, then 1% on seed 1. Recommended (1.7 and 0.9 times the seed spread; looser would let noise pick).
3. **The loss:** the range-weighted Huber stands in for D3's "EV-space loss" (the DEVN head is separate, a later registration of its own).
4. **Go points:** the code stage (no compute), the `check` (~20 min GPU, tell the user first), and the run (round 1 to the verdict, ~6 h GPU, one go) are each a separate yes.
