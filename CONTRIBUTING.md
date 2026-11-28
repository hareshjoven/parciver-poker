# Contributing

Thank you for looking. This is a research project with a strict rule about evidence, so a few notes will save you time.

## Reporting a problem

Run `parciver doctor` and paste its output in a new issue; the templates ask for it. A setup problem is a bug in the
instructions or the tooling, not in you: please report it, even if you found a way around it.

## Trying a change

```bash
pip install -e ".[dev]"          # plus the Rust engine, see the README's Install section
ruff check src scripts tests      # lint, as CI runs it
pytest -m "not slow"              # the fast tier: about 5 minutes
pytest                            # everything: about 22 minutes on a 16-thread desktop
cargo test --release --no-default-features --manifest-path engine/Cargo.toml
```

CI runs the lint, the full Python suite and the Rust tests on Linux and Windows for every pull request.

## What is welcome

- Fixes to the docs, the setup path and the error messages. If a beginner would have been stuck, it is in scope.
- Tests and bug fixes, with a test that fails before and passes after.
- A new solver, abstraction or search idea, as an **experiment**: the project pre-registers what it will measure and
  what result counts as a pass before it looks at data (see [`records/`](records) and [RESULTS.md](RESULTS.md)).
  Please open an issue describing the test and the pass rule first.

## What will not be merged

A strength claim without a pre-registered measurement, a change that alters a recorded result, or a download that
executes code. The tables and nets are numeric data only, and that is a guarantee the project keeps.
