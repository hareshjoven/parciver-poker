# New here? Start with this page

You do not need to know poker, game theory or machine learning to use parciver. This page explains what it is,
walks you through a first session, and defines every word the README uses.

## What this project is, in four sentences

parciver is a computer opponent for heads-up (two-player) no-limit Texas hold'em. It was not taught poker
strategy by a person: it *learned* it by playing itself billions of times with an algorithm called
**Counterfactual Regret Minimization (CFR)**, which keeps track of what it wishes it had done at every decision
and gradually shifts toward the choices it regrets least. The result is a strategy that is very hard to exploit,
which is what "solving" poker means in practice. You can play against it in your terminal, and you can read how
it was built and measured.

## Your first ten minutes

Every step below is a command you type; `parciver doctor` tells you what is ready and what is next.

1. **Install the Python side** (a few seconds, no Rust needed), following the quick start in the
   [README](../README.md).
2. **Check your setup:**
   ```bash
   parciver doctor
   ```
   It lists what works, what is missing, and the exact command that fixes each gap. It changes nothing.
3. **Play the smallest poker game** (3 cards, nothing to download):
   ```bash
   parciver play kuhn
   ```
   The prompt tells you what is legal (for example `check/bet`); the first letter is enough, and `q` quits.
   It first prints how exploitable the opponent is. That number is the whole point of the project: it
   measures how much a perfect opponent could win.
4. **Watch the strategy** it just learned:
   ```bash
   parciver play leduc --peek
   ```
   After each hand it shows the probabilities it used. Notice that it is *mixed*: in the same spot it
   sometimes bets and sometimes checks, on purpose, so you cannot read it.
5. **Play real poker** once you want to. This needs the Rust engine and a one-time ~0.7 GB download; the
   README's [Install](../README.md#install) section has the steps, and `parciver doctor` shows which one you
   are on.
   ```bash
   parciver play nlhe
   ```

If something fails, run `parciver doctor` first and read the troubleshooting table in the README. When you ask
for help in an issue, paste the doctor's output.

## Reading the No-Limit screen

Before every decision you see a line of facts and a line of options. This is the very first decision of a hand, as
the small blind (the cards are an example; the numbers are what the program prints):

```
  [preflop] you: Ah Kd   board: -   pot: 150   to call: 50   behind: 19950
  you can: f fold | c call 50 | b <to> raise to 200-20000 | h half pot = 200 | p pot = 300 | a all in = 20000 | ? help | q quit
```

Later in a hand a `so far:` line between the two recaps what has been bet. Here are the parts:

| you see | it means |
|---|---|
| **[flop]** | the street (preflop, flop, turn, river); **you** are your two cards |
| **pot** | chips in the middle |
| **to call** | what it costs you to stay in the hand |
| **behind** | chips you still hold (your stack) |
| **board** | the shared cards; none before the flop |
| `f` / `c` | fold (give up) / call, or check when nothing is owed |
| `b 600` | bet or raise **to a total of** 600 chips on this street (`r 600` and a bare `600` work too) |
| `h`, `p`, `a` | half-pot, pot-sized, all in |
| `?` | prints the commands and the hand ranking again, then re-asks |

Chips are counted in the game's own units: one big blind is 100 chips, and stacks start at 20,000 (200 big
blinds). Nothing you type can hurt anything but your own stack, so experiment. The opponent plays a fixed mix
of bets, calls and folds; it never reads you or adapts.

## Poker in one minute

Each player gets two private cards. Five shared cards come out in stages: three (the **flop**), one (the
**turn**), one (the **river**), with a round of betting between stages. If nobody folds, the best five-card hand
from your two cards plus the board wins at the **showdown**. Hands from best to worst: straight flush, four of a
kind, full house, flush, straight, three of a kind, two pair, pair, high card.

"No-limit" means you may bet any amount up to your whole stack. "Heads-up" means just two players, so you
alternate between the two forced bets: the **small blind** (half a big blind) and the **big blind**.

## Glossary

| term | plain meaning |
|---|---|
| **CFR** | Counterfactual Regret Minimization: self-play that repeatedly shifts toward actions it regrets not having taken |
| **Nash equilibrium** | a strategy that no opponent can profitably exploit; in two-player zero-sum poker, playing it cannot lose in the long run |
| **exploitability** | how much a perfect opponent could win per hand against a strategy; 0 means unbeatable. Exact on Kuhn and Leduc, estimated on no-limit |
| **blueprint** | the big strategy table learned offline by self-play (the 0.7 GB download) |
| **abstraction** | grouping similar situations and card hands together so the game is small enough to solve; no-limit hold'em is far too large to solve exactly |
| **bucket** | one group of similar hands inside the abstraction (for example 10,000 flop buckets) |
| **search / subgame solving** | re-solving the part of the game you are actually in, in real time, more precisely than the blueprint |
| **MCCFR** | Monte Carlo CFR: samples a part of the game each iteration instead of walking all of it |
| **big blind (bb)** | the larger forced bet; the unit poker results are quoted in |
| **bb/100** | big blinds won (or lost) per 100 hands, the standard way to compare poker players. +10 is a strong winner; ±14 is the noise in a 50,000-hand match |
| **confidence interval (CI)** | the range a measured result could plausibly fall in; if it includes zero, the match did not prove a winner |
| **AIVAT** | a variance-reduction method that removes luck from the card deals so results converge with fewer hands |
| **pre-registered** | the test and the pass/fail rule were written down and committed before the data existed, so results cannot be cherry-picked |
| **Kuhn / Leduc** | two tiny poker games (3 cards and 6 cards) small enough to solve exactly; they are the checks the solver must pass before it touches real poker |
| **Slumbot** | a public no-limit hold'em bot (a 2018 competition winner) that the project benchmarks against over HTTP |

## Where to go next

- [README](../README.md): the results, the full install, the three play tiers.
- [ARCHITECTURE.md](../ARCHITECTURE.md): how the pieces fit together.
- [RESULTS.md](../RESULTS.md): every measurement, including the ones that did not favour the bot.
- [CONTRIBUTING.md](../CONTRIBUTING.md): reporting a problem or sending a change.
