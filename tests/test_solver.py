# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Convergence gates for the CFR solvers.

These are the project's primary defence against silent correctness bugs. A CFR
implementation can look healthy while converging to a non-equilibrium; the exact
best-response exploitability computed here is what catches that, because it is
computed independently of the solver.

Thresholds are set with headroom above the empirically measured convergence so
the suite is a real regression gate, not a flaky one. Iteration counts are kept
modest to keep the pure-Python reference solver's runtime reasonable; the fast
Rust path (future) will let these run far longer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from parciver.eval.exploitability import exploitability, expected_value
from parciver.games.kuhn import GAME_VALUE, KuhnPoker
from parciver.games.leduc import LeducHoldem
from parciver.solvers.cfr import CFR_PLUS, DCFR, VANILLA, CFRSolver


@pytest.mark.parametrize("variant", [VANILLA, CFR_PLUS, DCFR], ids=lambda v: v.name)
def test_kuhn_reaches_game_value(variant):
    """Every variant must reproduce Kuhn's exact game value of -1/18."""
    game = KuhnPoker()
    solver = CFRSolver(game=game, variant=variant, alternating=False)
    solver.run(3000)
    value = expected_value(game, solver.average_strategy())
    assert value == pytest.approx(GAME_VALUE, abs=2e-3)


@pytest.mark.parametrize("variant", [VANILLA, CFR_PLUS, DCFR], ids=lambda v: v.name)
def test_kuhn_converges_to_equilibrium(variant):
    """Exploitability must fall well below a loose bound on tiny Kuhn."""
    game = KuhnPoker()
    solver = CFRSolver(game=game, variant=variant, alternating=False)
    solver.run(3000)
    assert exploitability(game, solver.average_strategy()) < 5e-3


def test_kuhn_exploitability_decreases():
    """Exploitability must be monotone-ish downward, not stuck or diverging."""
    game = KuhnPoker()
    solver = CFRSolver(game=game, variant=DCFR, alternating=False)
    solver.run(500)
    early = exploitability(game, solver.average_strategy())
    solver.run(4500)
    late = exploitability(game, solver.average_strategy())
    assert late < early / 2.0


def test_leduc_dcfr_converges():
    """Discounted CFR must reach low exploitability on Leduc.

    Measured ~1.6e-2 at 2000 iterations; the 4e-2 gate leaves headroom while
    still failing loudly on a regression that breaks convergence.
    """
    game = LeducHoldem()
    solver = CFRSolver(game=game, variant=DCFR, alternating=False)
    solver.run(1000)
    expl = exploitability(game, solver.average_strategy())
    assert expl < 4e-2, f"Leduc DCFR exploitability {expl:.4e} exceeds gate"


def test_leduc_on_policy_value_matches_known():
    """The converged on-policy value should sit near the known Leduc value.

    Leduc's game value to the first player is small and negative (~-0.086 chips
    in this ante/bet parameterisation); a badly broken engine or solver would
    not land here.
    """
    game = LeducHoldem()
    solver = CFRSolver(game=game, variant=DCFR, alternating=False)
    solver.run(1000)
    value = expected_value(game, solver.average_strategy())
    assert -0.15 < value < 0.0


def test_exploitability_oracle_is_zero_sum_consistent():
    """Best-response values must bracket the on-policy value (zero-sum sanity).

    For any strategy profile, player 0 best-responding can only do at least as
    well as following the profile, and player 1 best-responding can only push
    player 0's value down. This is a property of the exploitability oracle
    itself, independent of convergence.
    """
    game = KuhnPoker()
    solver = CFRSolver(game=game, variant=DCFR, alternating=False)
    solver.run(500)
    strat = solver.average_strategy()
    from parciver.eval.exploitability import best_response_value

    br0 = best_response_value(game, strat, 0)   # value to p0 when p0 best-responds
    br1 = best_response_value(game, strat, 1)   # value to p1 when p1 best-responds
    on_policy = expected_value(game, strat)     # value to p0 under the profile
    assert br0 >= on_policy - 1e-9
    assert -br1 <= on_policy + 1e-9
    # NashConv is non-negative.
    assert br0 + br1 >= -1e-9
