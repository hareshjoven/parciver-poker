# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""R-MAPSS Step 2 — interaction kernel + its exact controls (RMAPSS.md §34).

Pure-Python + exact. Two anchors:
* the ``FastTree`` used to make the kernel tractable is BYTE-FOR-BYTE the certified
  game-driven exact BR / on-policy value (``parciver.eval.exploitability``), and its
  fast CFR matches ``MultiCFR`` — so speed buys nothing at the cost of correctness;
* the Step-2 machinery's §28 controls hold on 3p Kuhn (the within-round FILTER): the
  correct-model MPC (A5) reproduces the exact ρ-ceiling at full horizon and never
  falls below the myopic baseline (A2.5), realised EV never exceeds the valid upper
  bound, and depth-1 == A2.5 (the fairness identity). A NO on the inter-opponent edge
  in single-round Kuhn is EXPECTED (R3) and is not a kill; the decisive test is Leduc.
"""
from __future__ import annotations

import pytest

from parciver.games.kuhn_multi import KuhnPokerN
from parciver.solvers.cfr_multi import MultiCFR
from parciver.eval import exploitability as E
from parciver.rmapss.fast_tree import FastTree
from parciver.rmapss.step2 import run_step2, collect_seat_infosets, _uniform


def test_fasttree_matches_certified_exploitability_bytewise():
    game = KuhnPokerN(3)
    ft = FastTree(game)
    s = MultiCFR(game, variant="cfr+")
    s.run(2000)
    strat = s.average_strategy()
    _, abk = collect_seat_infosets(game)
    full = {k: strat.get(k, _uniform(len(abk[k]))) for k in abk}
    for p in range(game.num_players):
        assert ft.on_policy_value(full, p) == E.on_policy_value(game, full, p)
        assert ft.best_response_value(full, p) == E.best_response_value(game, full, p)
        assert ft.best_response_strategy(full, p) == E.best_response_strategy(game, full, p)


def test_fasttree_cfr_matches_multicfr_nashconv():
    game = KuhnPokerN(3)
    ft = FastTree(game)
    fast_eq = ft.solve_cfr(3000, "cfr+")
    slow_eq = MultiCFR(game, variant="cfr+")
    slow_eq.run(3000)
    from parciver.eval.exploitability import nash_conv_multiplayer
    nc_fast = nash_conv_multiplayer(game, fast_eq)
    nc_slow = nash_conv_multiplayer(game, slow_eq.average_strategy())
    assert nc_fast < 0.02
    assert abs(nc_fast - nc_slow) < 1e-9   # identical recursion -> identical iterate


@pytest.fixture(scope="module")
def kuhn_rep():
    return run_step2(game_name="3p-kuhn", m=6, depths=(1, 2, 3), cfr_iters=2500)


def test_kuhn_controls_all_pass(kuhn_rep):
    ctl = {k: v for k, v in kuhn_rep.gates.items() if k.startswith("ctl")}
    assert all(ctl.values()), [k for k, v in ctl.items() if not v]


def test_kuhn_a5_full_horizon_equals_ceiling(kuhn_rep):
    # The correct-model MPC at full horizon reproduces the exact ρ-ceiling (§28:
    # planner is sound and A5's adaptation model IS the true ρ).
    assert all(r.a5_full_eq_ceiling for r in kuhn_rep.results)


def test_kuhn_a5_never_below_myopic_and_bounded(kuhn_rep):
    for r in kuhn_rep.results:
        assert r.ev["A5"] >= r.ev["A2.5"] - 1e-9          # shaping never hurts (correct model)
        assert r.ev["A5"] <= r.upper_bound + 1e-9          # valid upper bound holds
        assert r.ev["A4"] <= r.upper_bound + 1e-9
        assert r.ceiling >= r.ev["A2.5"] - 1e-9            # oracle >= myopic


def test_kuhn_depth1_is_the_myopic_baseline(kuhn_rep):
    # Fairness identity (§34.3): depth-1 planning == A2.5 for BOTH A4 and A5.
    for r in kuhn_rep.results:
        d1_a5 = dict(r.depths["A5"])[1]
        d1_a4 = dict(r.depths["A4"])[1]
        assert abs(d1_a5 - r.ev["A2.5"]) < 1e-9
        assert abs(d1_a4 - r.ev["A2.5"]) < 1e-9


def test_kuhn_cascade_depth_is_live(kuhn_rep):
    # The depth-ablated cascade actually does something: at least one family has a
    # deeper A5 value that differs from depth-1 (else the kernel is inert).
    changed = any(dict(r.depths["A5"]).get(2) != dict(r.depths["A5"])[1]
                  or dict(r.depths["A5"]).get(3) != dict(r.depths["A5"])[1]
                  for r in kuhn_rep.results)
    assert changed


def test_kuhn_verdict_is_a_filter_outcome(kuhn_rep):
    # Kuhn is the within-round filter; the outcome must be a filter verdict, never a
    # decisive GO/KILL (R3). Machinery-broken verdicts must not appear.
    assert kuhn_rep.role == "filter"
    assert kuhn_rep.verdict in ("FILTER-PASS", "FILTER-FLAT", "INCONCLUSIVE")
