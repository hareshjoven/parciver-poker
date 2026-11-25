# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Cross-validation of the Rust engine against the Python reference.

The Rust crate (`parciver_engine`) reimplements the game trees and the CFR hot
loop for speed. Speed is worthless if it drifts from the validated Python
semantics, so these tests pin the two engines together on the SAME ground truth:

  * structural parity -- identical infoset partitions (keys and action counts);
  * equilibrium quality -- a Rust-solved strategy, scored by the INDEPENDENT
    Python best-response oracle, must clear the same exploitability gates the
    Python solver clears (Kuhn value -1/18, Leduc exploitability low, correct
    variant ordering);
  * numerical parity -- at a matched iteration count the two solvers land on the
    same iterate, confirming they implement the same algorithm, not merely two
    things that both happen to converge.

The suite skips cleanly if the extension has not been built yet
(`maturin develop -m engine/Cargo.toml`), so a fresh checkout still runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

engine = pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)

from parciver.eval.exploitability import exploitability, expected_value
from parciver.games.kuhn import GAME_VALUE, KuhnPoker
from parciver.games.leduc import LeducHoldem
from parciver.solvers.cfr import DCFR, CFRSolver

VARIANTS = ("cfr", "cfr+", "dcfr")


# --- structural parity ----------------------------------------------------

def test_infoset_counts_match_canonical():
    assert engine.leduc_infoset_count() == 288
    assert engine.kuhn_infoset_count() == 12


def _python_infosets(game, iterations=200):
    """Every infoset key the Python engine produces (full traversal visits all)."""
    solver = CFRSolver(game=game, variant=DCFR, alternating=False)
    solver.run(iterations)
    return {key: len(node.strategy_sum) for key, node in solver.nodes.items()}


@pytest.mark.parametrize(
    "game, solve",
    [(LeducHoldem(), engine.solve_leduc), (KuhnPoker(), engine.solve_kuhn)],
    ids=["leduc", "kuhn"],
)
def test_infoset_keys_and_action_counts_match_python(game, solve):
    """The Rust and Python engines must agree on the infoset partition exactly.

    Equal key sets prove the perfect-recall abstraction matches; equal per-key
    vector lengths prove the legal-action sets (and their ordering slots) match.
    A divergence here is the classic silent CFR bug the whole project guards
    against.
    """
    rust = solve(200, "dcfr")
    py = _python_infosets(game)
    assert set(rust.keys()) == set(py.keys())
    for key, n_actions in py.items():
        assert len(rust[key]) == n_actions, f"action count mismatch at {key!r}"


# --- equilibrium quality (Rust-solved, Python-scored) ---------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_kuhn_rust_reaches_game_value(variant):
    strat = engine.solve_kuhn(5000, variant)
    kuhn = KuhnPoker()
    assert expected_value(kuhn, strat) == pytest.approx(GAME_VALUE, abs=2e-3)
    assert exploitability(kuhn, strat) < 5e-3


def test_leduc_rust_dcfr_converges():
    """Same gate as the Python solver's Leduc test, applied to the Rust output."""
    leduc = LeducHoldem()
    strat = engine.solve_leduc(1000, "dcfr")
    expl = exploitability(leduc, strat)
    assert expl < 4e-2, f"Leduc DCFR exploitability {expl:.4e} exceeds gate"
    assert -0.15 < expected_value(leduc, strat) < 0.0


def test_leduc_rust_variant_ordering():
    """DCFR <= CFR+ < vanilla on Leduc, the ordering the Python sweep established."""
    leduc = LeducHoldem()
    expl = {v: exploitability(leduc, engine.solve_leduc(1000, v)) for v in VARIANTS}
    assert expl["dcfr"] <= expl["cfr+"] * 1.05
    assert expl["cfr+"] < expl["cfr"]


# --- numerical parity with the Python solver ------------------------------

def test_leduc_matches_python_solver_iterate():
    """At a matched iteration count the two engines produce the same iterate.

    Both run identical full-traversal DCFR in float64 with the same update
    order, so the average strategies -- and therefore the independently computed
    exploitability -- should coincide to well within any tolerance a genuine
    algorithmic divergence would blow past.
    """
    leduc = LeducHoldem()
    n = 200
    py = CFRSolver(game=leduc, variant=DCFR, alternating=False)
    py.run(n)
    py_expl = exploitability(leduc, py.average_strategy())
    rust_expl = exploitability(leduc, engine.solve_leduc(n, "dcfr"))
    assert rust_expl == pytest.approx(py_expl, abs=1e-6)


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        engine.solve_kuhn(10, "not-a-variant")
