# Continuing this work

Short version: yes, someone else can pick this up. The code is tested, the experiments are written down in a form
that says what was tried, what passed and what did not, and the open leads are listed below.

## What is and is not new here

Nothing in the algorithm list is new, and the project does not claim it is. Counterfactual regret minimisation,
Monte-Carlo CFR, card abstraction by clustering, depth-limited subgame solving with learned leaf values,
AIVAT and local best response all come from the published poker-AI literature (Libratus, Pluribus, DeepStack and
related work). This repository is an independent, documented implementation and evaluation of those ideas on one
desktop (one GPU, 16 CPU threads).

What the repository adds is the engineering and the evidence:

- a validation ladder (exact oracles on Kuhn and Leduc, a certified hand evaluator) before any hold'em result is trusted;
- every experiment pre-registered before its data existed, with the negative and inconclusive results kept
  (`records/`);
- a rating and evaluation harness with exact standard errors, and an honest headline: the live match against
  Slumbot is **unresolved**, not a win;
- the engineering that makes the pipeline fit one machine (Rust MCCFR, GPU subgame kernels, bit-identical speed work).

## Getting started

```bash
pip install -e ".[dev]" maturin && maturin develop --release -m engine/Cargo.toml   # needs Python 3.12+ and Rust
pytest                          # the CPU suite; tests that need a GPU or triton skip themselves
parciver play leduc             # tier 0: no downloads
```

Bigger pieces (blueprint, GPU search) need the release assets and a CUDA GPU; see [RELEASING.md](../RELEASING.md)
and [ARCHITECTURE.md](../ARCHITECTURE.md).

## Open leads (all grounded in `records/`)

| Lead | State | Where |
|---|---|---|
| A better flop value net (D3: another 1,000,008 turn-start training rows) | labelling stopped at about 500 shards by choice; the second look said CONTINUE (-7.10% against a -4% bar). Resuming means labelling more shards (about 1.4 h each on one GPU) and retraining | `records/d3` |
| Resolving the Slumbot result | both 50,000-hand matches have intervals containing zero; a decisive answer needs many more hands or a lower-variance estimator | `records/slumbot`, `records/slumbot-2` |
| The big-blind seat was negative in both matches | plausible leak in big-blind defence, not shown; the seat split mostly reflects position | `records/slumbot-2/RESULT.md` |
| More blueprint iterations or a finer abstraction | the 8B-iteration blueprint is in use; the 12.3B one was trained but is not the live one | `records/bp-8b` |
| Re-rating the current bot | the 1860.4 rating predates the 8B blueprint and the flop-net upgrades | `RESULTS.md` |
| Small preflop opens | two attempts (patching, retraining) did not pass their gates; understand why before a third | `records/preflop-open` |

`src/` and `scripts/` also contain a closed multiplayer-exploitation track (`rmapss*`); its later gates failed
and it is kept only for the record.

## Working conventions

- **Pre-register first.** Write the hypothesis, sample size and pass rule in `records/<topic>/PREREG.md`
  and commit it before generating data; file the outcome as `RESULT.md`, including when it fails.
- **Oracles before speed.** A faster implementation must reproduce the reference to the bit or pass an exact
  oracle test; never trade precision for speed.
- **Keep CI green.** The Rust job and the Python job (Ubuntu and Windows, CPU torch, ruff, pytest) run on every push.

Rebuilding the release assets or the public tree from the author's private lab repository is not required to work
on this one; the code here is self-contained.
