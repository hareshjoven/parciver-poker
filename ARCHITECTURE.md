# Architecture

A poker bot of this kind has two halves: an **offline blueprint**, a complete but coarse strategy for the whole
game, and an **online search** that refines the blueprint at the exact situation reached in a hand. This is the
same two-part design behind Libratus and Pluribus, scaled to one desktop.

```
 offline                                                    at the table
 -------                                                    ------------
 card abstraction ──┐                                       opponent's real bet
 (isomorphism +     │                                          │ translate to nearest grid size
  k-means buckets)  ├─> Rust MCCFR ──> blueprint (1.27 GB)     v
 bet abstraction ───┘   8e9 iterations        │            blueprint lookup ──> action        (preflop, or --no search)
                                              │                 │
                                              └── ranges ──> subgame solve on the GPU ──> action
                                                   flop: depth-limited, learned leaf values
                                                   turn, river: solved to the end of the hand
```

## 1. The game and its abstraction

Heads-up no-limit hold'em at 200 big blinds has on the order of 10^160 decision points, so the blueprint
solves a smaller game that approximates it.

- **Card abstraction** (`engine/src/isomorphism.rs`, `bucketing.rs`, `src/parciver/clustering.py`). Suits are
  interchangeable until a flush matters, so positions are first collapsed *exactly* by suit isomorphism: 1,326
  starting hands become 169 classes, 22,100 flops become 1,755. Preflop is not bucketed beyond that. Postflop, situations are grouped
  by a *potential-aware* feature: the distribution of a hand's equity one street later (a 30-bin histogram, so a
  made hand and a draw with equal equity now look different), clustered with k-means under Earth-Mover's distance
  on the GPU. That gives 10,000 buckets on the flop and on the turn (from 1.29 million and 13.96 million distinct
  situations) and 1,950 on the river (123 million situations, clustered on scalar equity).
- **Bet abstraction** (`engine/src/nlhe.rs`). Opens of half and full pot, a first raise of one pot, further
  pot-size re-raises, and all-in always available. A real bet the grid does not contain is *translated* when the bot
  reads it (`engine/src/play.rs`): a size between two grid sizes is mapped to one of its neighbours, and one
  above the largest is read as that size (or as all-in if it commits nearly the whole stack). The default policy
  is called `snap`. Translation is the abstraction's weak point; see [docs/translation-and-exploitability.md](docs/translation-and-exploitability.md).

## 2. The blueprint: Monte Carlo CFR in Rust

`engine/src/mccfr.rs` runs external-sampling **Linear MCCFR**: on each iteration one player's decisions are
traversed in full while the opponent's and chance's are sampled, regrets are accumulated with iteration-linear
weights, and the average strategy is the output. It runs in parallel across cores.

The solver is trusted because it was climbed to, not assumed (see the ladder in [RESULTS.md](RESULTS.md)):
tabular CFR/CFR+/Linear/DCFR first reproduce Kuhn's exact value and Leduc's 288 information sets under an
independent exact best-response oracle, the Rust port agrees with the Python reference to 1e-6, and only then
does the same engine run on hold'em. The 7-card evaluator is checked against all 133,784,560 hands.

The Python/Rust split follows measurement: Rust takes the branchy pointer-chasing tree traversal, hashing and
hand evaluation; NumPy/PyTorch keep the bandwidth-bound array math, where Rust was measured not to win.

## 3. Real-time search on the GPU

At the table the bot re-solves the current subgame from the ranges the blueprint implies, rather than trusting the
abstraction's coarse buckets and bet sizes (`src/parciver/gpu_fused.py`, `gpu_search.py`, `flop_search.py`).

- **River and turn** are solved to the end of the hand by a fused GPU CFR solver (PyTorch plus Triton kernels):
  2,000 iterations on the river, 400 on the turn. Terminal payoffs use a precise kernel ("Q4", measured 2.9x
  faster than the float64-precise reference) ([docs/gpu-kernels.md](docs/gpu-kernels.md)).
- **Flop** cannot be solved to the end in time, so the flop betting round is solved to depth and the value of each
  turn-start position is taken from a **learned value net** trained on 951,466 offline turn-start training rows
  (`scripts/turn_data.py`, `scripts/turn_net.py`, `src/parciver/value_net.py`), 500 iterations
  ([docs/flop-search.md](docs/flop-search.md)).
- Latency, measured over the 50,000-hand SLUMBOT-2 run (flop net N0): flop median 2.3 s, turn 0.9 s, river 0.16 s.

## 4. Measuring it

- **Rating** (`src/parciver/rating.py`, `scripts/rating.py`): a chip-linear strength number against a frozen
  anchor pool, with exact standard errors and AIVAT variance reduction ([docs/rating.md](docs/rating.md)).
- **LBR** (`engine/src/lbr.rs`): a local-best-response exploiter, an estimate (a lower bound) of how much a
  strong opponent could win from the bot.
- **Slumbot client** (`scripts/slumbot_run.py`, `src/parciver/slumbot.py`): plays the third-party server with
  every payoff re-computed locally, a parser check and a registered mock run gating each live run.
- **Training telemetry and dashboard** (`src/parciver/telemetry.py`, `src/parciver/dashboard/`).

## 5. Where things are

| path | what |
|---|---|
| `engine/src/` | game engine, abstraction, MCCFR, exploitability, LBR, hand evaluator (PyO3 module `parciver_engine`) |
| `src/parciver/games`, `solvers`, `eval` | Kuhn, Leduc; the CFR family; exact best-response exploitability |
| `src/parciver/play_small.py`, `play_nlhe.py`, `cli.py`, `assets.py` | the terminal games and the blueprint downloader |
| `src/parciver/gpu_*.py`, `flop_*.py`, `value_net.py`, `turn_values.py` | the real-time search |
| `scripts/` | experiment drivers; `slumbot_run.py` holds the live configuration |
| `records/` | pre-registrations, results and receipts |
