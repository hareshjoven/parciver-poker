"""R-MAPSS Step 1 — multiplayer exploitation spine (RMAPSS.md §33, 3p Kuhn).

Pure-Python + exact; no engine required (uses the Python N-player CFR + the exact
best-response oracle). Locks the spine's gates and the two structural findings that
make Step 1 honest: the independent (marginal) model is EXACT for card-independent
opponents and UNSAFE (best-responding to it loses) for card-dependent equilibrium
opponents (refinement R2).
"""
from __future__ import annotations

import pytest

from parciver.games.kuhn_multi import KuhnPokerN
from parciver.solvers.cfr_multi import MultiCFR
from parciver.eval.exploitability import nash_conv_multiplayer, on_policy_value
from parciver.rmapss.step1 import run_step1, clairvoyant_team_min, collect_infosets


@pytest.fixture(scope="module")
def rep():
    # One shared run (fewer CFR iters keeps the suite fast; gates are robust to it).
    return run_step1(cfr_iters=5000)


def test_cfr_multi_reaches_low_nashconv():
    # Independent Python N-player CFR (cross-checks cfr_np.rs): low NashConv on 3p Kuhn.
    game = KuhnPokerN(3)
    s = MultiCFR(game, variant="cfr+")
    s.run(5000)
    assert nash_conv_multiplayer(game, s.average_strategy()) < 0.02


def test_step1_all_gates_pass(rep):
    assert rep.all_pass, [k for k, v in rep.gates.items() if not v]


def test_marginal_is_exact_for_card_independent_and_unsafe_for_equilibrium(rep):
    by = {r.label: r for r in rep.results}

    # Card-INDEPENDENT opponents (station/maniac/nit/uniform): the marginal model
    # is exact, so A2 reaches the exact-BR ceiling.
    for label in ("station,station", "maniac,maniac", "nit,nit", "uniform,uniform"):
        r = by[label]
        assert abs(r.ev_a2 - r.ev_ceiling) < 1e-9, label

    # Equilibrium opponents are card-DEPENDENT: best-responding to the marginal
    # (A2) is strictly worse than CFR (R2) — the finding the safety gate protects.
    eq = by["eq,eq"]
    assert eq.ev_a2 < eq.ev_a0 - 1e-3
    # ...but the confidence gate switches the deviation off there (alpha == 0).
    assert eq.alpha == 0.0
    assert abs(eq.ev_final - eq.ev_a0) < 1e-9


def test_exact_br_is_the_ceiling_and_worstcase_is_a_lower_bound(rep):
    for r in rep.results:
        # exact BR to the fixed pair is the exploitation ceiling (§28)
        assert r.ev_a0 <= r.ev_ceiling + 1e-9
        assert r.ev_final <= r.ev_ceiling + 1e-9
        # clairvoyant-team worst case never exceeds realized EV
        assert r.worstcase_final <= r.ev_final + 1e-9
        # safety: deviating never erodes the worst case beyond the budget
        assert r.safety_ok


def test_clairvoyant_worstcase_le_on_policy():
    # Sanity: an omniscient adversary can only do >= as well as any fixed opponent.
    game = KuhnPokerN(3)
    s = MultiCFR(game, variant="cfr+")
    s.run(2000)
    eq = s.average_strategy()
    keys = collect_infosets(game)
    hero = {k: eq[k] for k in keys[0]}
    opp = {k: eq[k] for k in keys[1] + keys[2]}
    wc = clairvoyant_team_min(game, hero)
    on = on_policy_value(game, {**hero, **opp}, 0)
    assert wc <= on + 1e-9
