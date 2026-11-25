# D2b: a better turn-start net from D1 + D1b (HANDOFF Roadmap B)

**Status: REGISTERED 30 Oct 2026 20:50 SGT.** The user's yes, in this session, to "Register D2b as drafted, to start once memory is
free?": "Yes, register it". It covers the text committed at a16e0a8 and its five recommendations (below). Written before
any D2b code, training or data use. No net has been trained or evaluated for it, and D1b's labels have not been read for
it. Only row counts, split counts, file layouts and existing records were used. Nothing here changes the live config
until "Adoption" passes. Audited read-only before going to the user; the audit's real findings are folded in.

## The question this run decides
**Better, not cheaper.** Does a net of N0's size, trained on D1 + D1b, have lower error than N0 without making
flop decisions worse, and does the whole bot stay not DOWN with it?

Cheaper is measured and reported only, and cannot be adopted here (a compact net needs its own registration). Why:
- Speed binds nothing today. Flop p95 is under the 3.0 s bar (mock 2.888 s, live SLUMBOT-2 2.663 s), and the flop
  search is converged at 500 iterations (0.20% of the pot).
- A smaller net trades away error. RM1 (IDEAS): w1024-d5 is +6.8% in selection loss against w2048-d5 at 60 epochs, on D1.

**What is honestly expected.**
- More data is the lever for net error: labels are 0.23% of the pot from their references, the net ~5%. That the net's
  error is what costs the flop its loss is IDEAS' reading, not something M1 has confirmed.
- A gain in Gate A may not show up as strength. LEAF-NET's P was -9.0% in MAE, and its M1 mean was -0.001% of the pot.
- **The likely cost of a pass:** if N1's per-spot D looks like P's (mean ~0, the same tail), it passes Gate B about .36 of
  the time, so ~5 days of gates would have about a 2/3 chance of ending in a fail. It passes .82 of the time if its mean
  is -0.2% of the pot. The first stage (nets and Gate A) costs under a day and says whether D1b helped at all.

## Known
- **N0** (`runs/turn-net/final-w2048-d5-e200/net.pt`, sha256 a8bf00be2fd1a3c7...): width 2,048, depth 5, 200 epochs,
  best epoch 158, selection loss 0.014464557576207186, selection MAE 5.209% of the pot; data fingerprint
  8df0fc79620a1820; 475,432 training rows. Gate MAE 5.111% (real 2.876, random 7.345); check-down 18.646 (bar 9.323).
- **D2's curve (60 epochs):** selection loss 0.02109 / 0.01711 / 0.01464 at 25 / 50 / 100% of D1; a fit gives ~-10% at 2x
  data. An estimate: 3 points extrapolated. The last doubling was -14.5% in loss and -11.8% in MAE.
- **D1b** (`records/d1b/RESULT.md`): ACCEPTED, 500,004 rows, `runs/turn-data-d1b/`, situations.npz sha256
  f5dc4be3..., same recorder and 18 cells as D1 (v2-2B). Mean exploitability 0.327% of the pot (D1 0.332%).
- **LEAF-NET:** wd 0.1 lowered the selection loss to 0.013437 on D1 (the canonical frame raised it at every wd).
  Gate A passed (-9.0%). Gate B failed on one spot: D -0.001% [-0.440, +0.439] of the pot, upper + e = +0.539 > 0.25,
  with spot index 33 at P 9.24% against N0 1.18%.

## What the data contains (checked 30 Oct, no values read)
| | D1 | D1b |
|---|---|---|
| Terminals | float32 (`generate.json` has no `terminals` key; code 12ad87f-dirty, seed 520007) | Q4, all rows (`terminals: q4`; code 7839aa6, PC shards under 3e272db with the generator unchanged; seed 600007) |
| Patches | 1,200 rows replaced by Q3 re-solves (`resolve-q3/patch.npz`, keep-better of 1,919) | none (no patch folder) |
| Rows: train / select / gate | 475,432 / 12,058 / 12,514 | 476,034 / 11,670 / 12,300 |
| Canonical boards: train / select / gate | 15,555 / 398 / 404 | 15,550 / 398 / 405 |

Union: 951,466 / 23,728 / 24,814 rows. 16,283 boards are in both and 70 only in D1b, so **D1b adds ranges on nearly
the same boards, not new boards**.

## How the split works: no leakage
D2's split unchanged (`turn_net.split`: splitmix64 of the suit-canonical board, SPLIT_SEED 20260929, under 2.5%
selection, under 5% gate). It depends only on the board, so a board is on the same side in both datasets: no
held-out board of either dataset reaches training.
- D1's gate half has been read by G2 and by LEAF-NET's Gate A. **D1b's gate half has been scored by no net.** It is
  Gate A's primary set: fresh ranges and Q4 labels, not fresh boards (403 of its 405 boards are also in D1's gate half,
  read but never trained on). N0 never trained on them either.
- Every choice below uses the union's selection half; only Gate A reads a gate half.

## Q3 and Q4 terminals
Pool the rows as they stand: D1 as N0 saw it (float32 with the Q3 patch), D1b as accepted (Q4). No filter by
exploitability; D1b's ~19 rows over 10% stay in, as accepted, and are counted in the report.
- **Why it is safe to pool.** D1b's step 0 re-solved D1 rows and compared them to the stored labels (ordinary jobs, % of
  the pot, median / p95 / max): Q4 on the PC 0.030 / 0.165 / 1.26; float32 on the cloud 0.027 / 0.153 / 1.37
  (`records/d1b/step0/parity-*.json`). So Q4 moves a label about as much as moving float32 to another machine does,
  about 1/170 of the net's error at the median. (float32 on one machine repeats bit for bit.)
- **Disclosed:** those Q4 runs fail D1b's G4 as first written, a per-job bar against float32 labels; D1b's Amendment 1 is a
  post-hoc replacement, marked as such there.
- **Declined:** re-solving D1 under Q4 (~190 h on the PC at 0.73 solves/s) for a change of that size.
- **Reported control:** a D1b-only net (below) says what Q4 alone gives at N0's data size.

## Arms
Fixed for all: batch 2,048, lr 1e-3, 200 epochs, seed 0, D2's recipe (Huber, OneCycle with 2% warm-up, bfloat16 hidden,
float32 output, suit augmentation), non-canonical, keep the epoch with the lowest selection loss. 200 epochs on 2x data
means 464 steps an epoch, not 232; the epochs stay fixed so that only the data and wd change against N0's recipe.

| Tag | Net | Data | wd | Role |
|---|---|---|---|---|
| `d2b-check` | w2048-d5 | D1 | 0 | Pre-check: must equal N0's record to the bit (epoch 158, the full float) |
| `d2b-w2048-wd0` | w2048-d5 | D1 + D1b | 0 | **Candidate** |
| `d2b-w2048-wd0.1` | w2048-d5 | D1 + D1b | 0.1 | **Candidate** |
| `d2b-d1b-only` | w2048-d5 | D1b | 0 | Reported: Q4 at N0's data size |
| `d2b-w1024-wd0`, `d2b-w1024-wd0.1` | w1024-d5 | D1 + D1b | 0, 0.1 | Reported: the cheaper net's loss and speed |
| `d2b-seed1` | the pick | D1 + D1b | the pick's | Reported: seed spread. Never replaces the pick |

- wd is restricted to {0, 0.1}: 0 is D2's recipe and 0.1 was LEAF-NET's best on the selection half. The canonical
  frame is not retried (it raised the loss at every wd).
- LEAF-NET's P (`leafnet-c0-wd0.1`, D1 only) is scored too, not trained, as the wd-only control: it separates what the data
  adds from what wd adds.
- If `d2b-check` differs from N0's record, nothing else is used until the cause is found and an amendment records it.

## Selection
**The pick N1 = the lower selection loss of the two candidates**, in float32, on the union's selection half (23,728
rows). N0 and P are scored on the same rows. **If N1 is not below N0 there, D2b stops.** Reported: each net's loss on each
dataset's selection half, on real and random sources, and train loss. The other arms are reported on the selection
half only; they do not read a gate half, so it stays unused by any choice about a later compact net.

## Gate A: precision (LEAF-NET's statistic)
N1 against N0 in float32, both H_b-averaged as SymLeaves can at best. Per sample D_i = MAE_N1(i) - MAE_N0(i), range
weighted, % of the pot; mean of D with a 95% percentile bootstrap, 10,000 resamples of canonical boards with their
samples, `default_rng(0)`.
- **Primary set: D1b's gate half (12,300 rows).** Reported beside it: D1's gate half and the union.
- **PROCEED iff the CI's upper end < 0 and the relative change is <= -5%** on the primary set.
  - The -5% is new against LEAF-NET, and it is a judgement, not derived: the later gates cost ~5.6 days, and a smaller gain
    does not justify them or a live change. It looks at no D2b number.
  - It is not a high bar: wd 0.1 alone gave P -9.0% on D1's gate half, and the curve's last doubling was -11.8% in MAE. P's
    -9.0% still gave an M1 mean of ~0, so passing this says nothing about Gate B.
- N1 must also pass G2's bar on D1's gate half: MAE under half of the check-down's 18.646% (cached, no new baseline).
- Otherwise D2b stops and N0 stays.
- Reported: bare nets (no averaging); real and random; symmetry defect of both.

## Gate B: M1 non-inferiority on more spots
LEAF-NET's lesson: on 40 spots this gate is both underpowered and too easy to pass by luck. Its per-spot D is heavy
tailed (P: sd 1.419% of the pot, 0.558 without one spot), so the verdict mostly records whether the tail was drawn.

**Spots: the first 160 of the registered recorder** (`flop_m1.py record`, seed 550000, v2-2B's preflop, hero
alternating; 40 spots took 76 deals, so ~300 deals), recorded at the code stage, which reads no labels and no net. The
first 40 entries of `spots` must equal the stored `m1-spots.json`'s bit for bit, or stop. Each is deterministic per deal,
and new spots are not chosen by anyone.
- **Why v2-2B though live is v2-8B:** one population and one recipe, the 40 old rows stay valid, and the net's training
  ranges are v2-2B's too. Known limit: Gate B does not cover v2-8B's ranges; the match and the mock run the live bot.

**The test (unchanged but for n):** D = Loss(X_N1) - Loss(X_N0), both sym leaves, 500 flop iterations, gamma 5, the same
oracle and rule, ISOMORPHISM on; paired over the 160 spots, z 1.96, e = 0.1% of the pot. **Pass iff the CI's upper end + e
<= 0.25% of the pot.** A verdict only when all 160 are in.
- **Disclosed:** the margin stays 0.25 after P's failure, it is not widened. The spot count was chosen after seeing
  P's per-spot D (not N1's), using the table below. That is a design choice on old data, not a look at this run.
- **Rows.** In `runs/flop-eval-d2b/` (`runs/flop-eval` is untouched). Baseline X_N0: spots 0-39 are `m1-x-sym.npz`'s arrays,
  byte-equal, and their `m1-v-sym.json` rows are carried over under a header recording the old files' sha256s; that is valid
  under `records/leaf-net/enginecheck.json` (same engine build E58C7292, refused otherwise). Spots 40-159 are new, N0's
  strategies and solves. X_N1: all 160 new. No row is copied between candidates. `star` is not solved on new spots: D needs
  only values, and Loss itself is reported on the old 40 alone.
- Reported: the old 40 and new 120 separately; per-spot D; the five largest D with their pot and board class; whether
  the tail is N1's or N0's (on the old 40, N0's top Loss values are 3.28, 3.76, 3.86).

**Power, from a bootstrap of P's 40 per-spot D** (rough: 40 points cannot pin a tail; D shifted by a true mean; % of the pot):
| n | pass if true mean 0 | -0.1 | -0.2 | -0.3 | false pass if truly +0.15 | solves | CPU h |
|---|---|---|---|---|---|---|---|
| 40 | .37 | .36 | .41 | .57 | .25 | 40 | 16 |
| 80 | .30 | .44 | .62 | .74 | .125 | 120 | 48 |
| 120 | .33 | .53 | .73 | .87 | .088 | 200 | 80 |
| **160** | **.36** | **.61** | **.82** | **.93** | **.076** | **280** | **112** |
| 200 | .40 | .69 | .88 | .97 | .068 | 360 | 144 |
| 320 | .52 | .84 | .97 | 1.00 | .058 | 600 | 240 |

- +0.15 is the boundary after e, where the rule's nominal false-pass rate is 2.5% (one-sided z 1.96). Without the tail value
  the bootstrap gives .016-.022 at every n, so the excess is the heavy tail's. At 160, .076 is about 3x nominal.
- Without the tail value and recentred to mean 0, the pass rate is .39 / .70 / .87 / .95 / .98 / 1.00 at these n; at its
  own mean of -0.207 it is >= .997. So the tail is the whole difficulty, and catching a real tail is what this gate is for.
- **The recommendation is 160**, a judgement: .82 at -0.2 and a false-pass rate near its floor for the cost. 200 adds 32 CPU-h
  for +.06 to +.08.

## Whole-bot safety: a head-to-head
**L1 (the live bot with N1) against L0 (the live bot), 10,000 deals**, only after Gate B passes. LIVESEARCH P2's and
BP-8B's protocol: the 200bb baseline row (seed 410,001), first 10,000 deals, seats swapped, 4 threads, chunks of 100,
50 blocks of 200, bb/100 from L1's side, AIVAT reported, a 50-deal smoke on an unused seed at 1 and at 4 threads must
agree first. Each arm has its own search and its own net; both play v2-8B and the live schedules.
- **DOWN iff the raw 95% CI's upper end < 0.** Not DOWN is needed for adoption. AIVAT is reported, not gating.
- Why it is here: M1 tests flop decisions on v2-2B's ranges, and a net also serves the live bot on v2-8B's. D4 M2,
  LIVESEARCH P2, RIVER R1 and BP-8B ran one; FLOP-SYM's and LEAF-NET's designs did not.
- What it can do: catch a gross loss only. LIVESEARCH P2's interval (same blueprint in both arms, the closer analog) was
  about +/-9.1 bb/100 raw, +/-7.3 AIVAT; BP-8B's +/-19 raw, +/-12 AIVAT. Not DOWN is not a gain.

## Adoption (all of Gate A, Gate B, not DOWN)
1. `scripts/slumbot_run.py` NET -> N1's net.pt, committed with the user's yes (a live net change is the user's).
2. A registered 1,000-hand `--mock` on that commit, seed 20261024 (SLUMBOT-2's): every check passes, flop p95 <= 3.0 s, turn
   and river <= 2.0 s. `mock_check.json` is committed under `records/d2b/`, and also becomes
   `records/livesearch/mock_check.json`, which GATES reads. A fail reverts NET to N0.
3. No separate speed step: N1 has N0's ops, and the mock measures its latency.

**No fallback:** after any stop or fail no runner-up is tried (not the other wd, not seed 1, not a compact net). Any new
candidate needs a new registration.

## What this run does NOT decide
- Whether the bot is stronger against Slumbot, or at all: the gates test error and non-inferiority.
- The rating: the live receipt (PR 1860.4 mbb/hand) is already stale, and refreshing it is a separate call.
- A compact net's adoption: its numbers are reported for a later registration.
- RM2, the preflop leak (SCOUT), or new data for other streets.
- Whether labels should be redone under Q4.

## Reported speed of the cheaper net (not a gate)
`d2b-w1024-wd*`'s better loss against N1's; w1024-d5 is 9.68M multiply-adds per row against 27.75M (0.349x). Interleaved
A B B A on `runs/flop-eval/m1-jobs-sym.pkl` (these exist whatever Gate A says), the same sym leaves, via `flop_sym_speed.py`
(its refusal of a non-canonical net is lifted for this report only); the median ratio with its flop-clustered CI. Run once
after the nets, GPU otherwise idle; no decision rests on it. Net forward is 84% of a flop iteration (KERNELS), so ~0.45x
is expected if forward time scales with multiply-adds, which is unmeasured.

## Code, tests first
- **`scripts/turn_net.py`:** `load` for two dirs, rows tagged by dataset (each dataset's own index space and
  situations); `patch` applies to D1 only; a `d2b` command with the arms above; Gate A as a new output
  `records/d2b/gate-a.json`.
- **Inputs pinned at registration:** `records/d2b/d1b-manifest.sha256` holds the sha256 of D1b's 501 shards and situations.npz
  (files only, no values read; 502 lines; the manifest's own sha256 is bd73786ef100f80f...). The union's fingerprint is computed when the code stage first loads the data, before any
  net, and written here by an amendment; training refuses any other, and D1's must be 8df0fc79620a1820.
- **GPU pools.** A training row is 13,576 B on the GPU, so the union's 951,466 rows are 12.92 GB (12.03 GiB) and the RTX 4070
  has 12,282 MiB (11.99 GiB; D1 alone 6.45 GB). Each epoch takes one global permutation and cuts it on a batch boundary into 2
  pools, uploaded in turn; 1,194 rows drop each epoch, the last partial batch, as D2's rule does. The batches are those of a
  full shuffle, and labels stay float32. No subsampling and no rounding.
- **`scripts/flop_m1.py`:** candidates `d2b0` (X_N0 as above) and `d2b1` (sym leaves with N1), `--spots 160`, the carried rows'
  header, and the order enforced: `strategy --cand d2b1` and the report refuse unless `gate-a.json` says proceed for N1's
  sha256.
- **`scripts/d2b_live.py`:** the match, as `bp8b_live.py` with a net per arm (`livesearch.Arm` gains an optional net
  whose default is `sr.NET`).
- **Tests before any data:** the D1-only path trains bit-identically to today's on a tiny run (one pool, same RNG); the
  pools' batches equal a single full-permutation run's; `scal_mean` and `scal_std` are over all training rows in pool mode; the
  `held` cache key includes the dataset; dataset tags keep index spaces apart; the patch touches D1 rows only; the union's
  fingerprint changes if either part does; the report's guards and the carried rows on toy files; `Arm` with no net is unchanged.
- `scripts/slumbot_run.py` and the engine do not change before Adoption; the installed .pyd stays E58C7292.

## Machine needs
- **Memory is the blocker.** At 30 Oct 19:00 SGT the free commit was 6.9 GB (limit 94.6) with a VM host process at 38.6 GB. Training
  holds about 31-37 GB at its peak (18.0 raw float32 + 12.9 prepared rows, + a staged pool if it is mirrored on the host), so
  **~40 GB free commit** is asked for. Each oracle spot needs its store + 11 GiB, up to 27.2 GiB so far, and a larger spot may
  appear among the 120 new ones. A solve waits for memory rather than failing; on 14 Oct that was a ~14 h stall. **The user
  decides what to close;** the assistant closes no app, and unrelated processes are left alone.
- GPU free for training, the match and the mock; M1's solve is CPU only. No blueprint job beside training or M1 (RAM).

| Step | Machine | Estimate |
|---|---|---|
| Code, tests, audits, spots | CPU | before any net |
| 7 nets (~50 min each at 2x data, w1024 ~0.5x, pools' overhead to be measured; check and D1b-only ~16 min) | GPU | ~4.5 h |
| Gate A | GPU | under 1 h |
| Strategies (X_N0 on 120, X_N1 on 160, ~17 min per 40) | GPU | ~2 h |
| 280 solves, ~24 min each at 12 threads | CPU | ~112 h |
| Smoke and match | GPU, 4 threads | ~20 h |
| Mock | GPU | ~40 min |

**Under 1 day if Gate A stops it; ~5.8 days if everything runs.** The ~24 min per spot comes from 39 of LEAF-NET's
spots (the stalled one excluded); the new spots are of the same recorder, so the same is expected, not promised.

## Order
Each stage only on the last one's pass: code, tests and spots with audit -> `d2b-check` -> candidates and reported arms ->
selection -> Gate A -> strategies -> solves -> Gate B -> smoke and match -> the user's yes to the NET commit -> mock. Nothing
is decided at an interim look; each stage is reported when it ends.

## VOID and bugs
- A crash, kill or reboot during a net's training, the match or the mock: that run is VOID, its cause recorded, and it is
  repeated. So is any of them with another compute job on the GPU (training, inference, a solver); desktop apps' GPU use does
  not void a run and is reported.
- **M1's rows are not voided by machine load:** on the two spots of the engine check, re-solved days apart, every field but the
  clock came out bit-identical, so a slow or busy machine cannot void a bad Gate B. A killed solve resumes per spot.
- A bug found after any data: that run is VOID and still reported. The fix and the rerun need an amendment here first, and
  both results stay in the record.

## Not allowed after registration
- Changing the arms, selection rule, gates, margins, spots or order after any D2b data.
- Choosing N1 on gate-half or M1 numbers; replacing it with seed 1 or the other wd; trying a runner-up after a fail.
- Dropping or reweighting a spot, or moving to a trimmed statistic after seeing its D.

## The user's choices ("Yes, register it" took the assistant's recommendation on each)
1. **Better, with cheaper only reported.** Alternative: also let a compact net compete. Declined here for the reason above.
2. **160 spots** (~112 CPU-hours). Alternatives: 120 (~80 h, pass .73 at -0.2), 200 (~144 h, pass .88).
3. **Gate A's -5% minimum.** Alternative: CI upper < 0 only, as LEAF-NET, which can send a -1% net into 5.6 days of gates.
4. **Include the 10,000-deal match** (~20 h GPU). Alternative: M1 and the mock only, as FLOP-SYM.
5. **Memory:** the user frees ~40 GB of commit for training, and ~27+ GiB stays free through M1 (a VM host process is the largest holder).

## Amendment 1: the code stage's details (30 Oct to 1 Nov 2026 SGT, before any net, D1b label or M1 row)
Records what the registration left to the code. No arm, selection rule, gate, margin, spot or order changes. Not a decision the
user was asked for. Some entries add a stop (the X_N0 check, `d2b select`'s refusals). The VOID rule above stands as written, and two
entries (the finished-match rule, the lock once `selection.json` exists) make applying it there need a further amendment, so that a
result in hand is never moved aside without one on the record. Nothing here widens what counts as a valid run.
- **Spots:** `runs/flop-eval-d2b/m1-spots.json`, 160 spots from 290 deals, sha256 9370e1fa7d406732e0f8ee41fb0889148a5ef1494fc5367e8e40918fd28f5651;
  its first 40 equal `runs/flop-eval/m1-spots.json`'s bit for bit (`d2b-init`, `d2b-carry.json`). Every stage that reads them (strategies,
  solves, Gate B) re-hashes both files.
- **A check added to X_N0:** spots 2-39 are copied from `m1-x-sym.npz`; spots 0-1 are made again and must equal it in value and dtype
  (so -0.0 equals 0.0), or nothing is saved and D2b stops for a cause. Passing, those two also take `m1-x-sym.npz`'s own arrays, so all 40 are
  byte-equal as registered. `flop_search.py` changed on 19 Oct (f5e5d4f), after X_sym was made, so the two spots show today's code still plays
  N0 as it did then. Both candidates use seed 0 alone, 500 iterations, gamma 5, as X_sym did; other settings are refused.
- **Smoke seed:** 777,006, used nowhere else.
- **GPU VOID, nets:** `d2b train` looks at the GPU before each net, every 10 epochs and after its last. Another process of this Python
  seen during training stops it at the next look (about 2.5 minutes on the union) and voids that net; seen at the first look, before any
  training, it stops the run and voids nothing. A look that could not say (nvidia-smi absent or failed, or listing no client at all) is
  counted and reported, and one such look voids nothing; `d2b select` refuses a net with no look that could say, or with a look that saw
  such a process, or a kill that left any of the 22 looks out, so it is repeated.
- **GPU VOID, match:** the same looks around each chunk of 100 (a chunk is about 12 minutes); a process of this Python stops the match at
  the chunk's end and the chunk is not kept, its looks held in the stop's record. Looks that could not say are counted and reported. A match
  with none that could say has no automatic refusal: it is reported to the user, and voided only by an amendment if a job is otherwise known.
- **What the code cannot see:** Windows lists every desktop app as a GPU client, so a compute job of another program (another Python,
  LM Studio) cannot be told apart by code, and WSL's jobs may not be listed at all. A job that starts and ends between two looks is not seen,
  and a process listed as `[Insufficient Permissions]` cannot be named. Each look's list, with utilisation and memory, is kept: per net in
  `runs/turn-net/<net>/gpu.json` (`nets.json` holds each net's clients together), per chunk in the match's state and at its end `live.json`.
  Not kept: a first look that stops the run, a reused net's one look, and the opening look of a chunk cut by a kill or Ctrl-C. Both print
  each client as it first appears, and in all when a run ends (a run stopped by a look, a failed chunk or a kill prints no closing list; its
  lists are on disk). The assistant reads them as they print, stops a run if it sees a compute
  job, names it, and applies the VOID rule by hand; `d2b train`'s closing list is read before `d2b select`.
- **Repeating a VOID run:** a cut net or match is never continued. A rerun refuses it until `turn_net.py d2b train --void "<cause>" --tag <net>`
  or `d2b_live.py match --void "<cause>"` moves it aside (`<net>-void-<n>`, `state-void-<n>.json`) with its cause; both refuse while a process of
  this Python is on the GPU or nvidia-smi cannot say. Moved runs are kept and listed in `nets.json` and `live.json`. A finished net whose record
  is not its arm's is refused the same way, never trained over. Once `selection.json` exists no net is voided without a further amendment.
- **Finished-match rule:** `--void` does not move a match that played all its deals. A compute job found in a finished match's lists still
  voids it under the rule above, but only through a further amendment naming the job, and its result stays in the record.
- **The mock:** `slumbot_run.py` is play code, with no GPU looks and a `--dir` that continues a run, and is not changed here. The rule holds by hand: a
  cut mock, or one that shared the GPU with a compute job, is repeated in a fresh run, never with `--dir`.
- **Committed code:** `d2b train` loads the net code, then reads the commit once and refuses unless src and scripts are clean; every net in
  that run records it, whatever is committed meanwhile. Finished nets are reused across later commits (other sessions commit to main), so a
  change to the training code needs an amendment and a retrain, as above.
- **Not yet here:** the union's fingerprint, which needs the data loaded: Amendment 2, when `d2b data` has run.

## Amendment 2: the union's fingerprint (2026-11-01 13:24 SGT, before any net)

`d2b data` ran at 13:22-13:24 SGT 1 Nov 2026 and wrote `records/d2b/data.json`. It reads the rows to hash and count them; no label's value was
looked at. **The union is `7d5a77c30aaad5d6`**, now `D2B_UNION` in `scripts/turn_net.py`; training and the gates refuse any other.

| | fingerprint | train / selection / gate rows | boards | rows over 10% exploitability |
|---|---|---|---|---|
| Union | 7d5a77c30aaad5d6 | 951,466 / 23,728 / 24,814 | 15,621 / 400 / 406 | 0 |
| D1 | 8df0fc79620a1820 (as registered) | 475,432 / 12,058 / 12,514 | 15,555 / 398 / 404 | 0 |
| D1b | 870e6a2fb7e1c2fb | 476,034 / 11,670 / 12,300 | 15,550 / 398 / 405 | 0 |

Every row count equals the registered one. Nothing else changes.
