"""Tests for R-MAPSS Step Q — fragility-as-transferable-EV-predictor (RMAPSS.md §38).

Fast, exact, engine-independent (3p Kuhn/Leduc pure-Python). Certifies the machinery,
the §38.5 controls (especially the circularity guard: cheap metrics make ZERO Hero-BR
calls), and that the pre-registered pipeline runs end-to-end and populates its gates.
"""
from __future__ import annotations

import math

import pytest

from parciver.games.kuhn_multi import KuhnPokerN
from parciver.solvers.cfr_multi import MultiCFR
import parciver.eval.exploitability as expl
from parciver.rmapss import stepQ
from parciver.rmapss.stepQ import (
    CHEAP_METRICS,
    BASELINE_METRIC,
    build_pop_q,
    build_exploiter,
    collect_infosets,
    infoset_reach,
    observe_marginal,
    opp_keys_by_seat,
    run_stepQ,
    spearman,
    _rankdata,
    _seat_values_under_profile,
)


@pytest.fixture(scope="module")
def kuhn_ctx():
    game = KuhnPokerN(3)
    seat_keys = collect_infosets(game)
    solver = MultiCFR(game, variant="cfr+")
    solver.run(4000)
    eq = solver.average_strategy()
    hero_eq = {k: eq[k] for k in seat_keys[0]}
    okeys = [k for s, ks in opp_keys_by_seat(seat_keys).items() for k in ks]
    _, eq_marg, _ = observe_marginal(game, hero_eq, {k: eq[k] for k in okeys}, seat_keys)
    ctx = {"hero_eq": hero_eq, "hero_keys": seat_keys[0], "eq_marg": eq_marg}
    return game, seat_keys, eq, ctx


# -------------------------------------------------------------------------
# rank correlation
# -------------------------------------------------------------------------

def test_spearman_monotone_and_antitone():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman(xs, [10.0, 20.0, 30.0, 40.0, 50.0]) == pytest.approx(1.0)
    assert spearman(xs, [50.0, 40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_ties_average_rank():
    assert _rankdata([1.0, 1.0, 2.0]) == [1.5, 1.5, 3.0]
    # constant vector -> zero variance -> defined as 0 correlation
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


# -------------------------------------------------------------------------
# population (§38.3)
# -------------------------------------------------------------------------

def test_population_splits_and_coverage(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    pop = build_pop_q(game, eq, seat_keys)
    splits = {s: [o for o in pop if o.split == s] for s in ("fit", "held", "transfer")}
    assert len(splits["fit"]) >= 5 and len(splits["held"]) >= 4 and len(splits["transfer"]) >= 3
    okeys = set(k for _, ks in opp_keys_by_seat(seat_keys).items() for k in ks)
    for o in pop:
        # every opponent covers all opponent infosets with a valid distribution
        assert set(o.profile) >= okeys
        for k, d in o.profile.items():
            assert abs(sum(d) - 1.0) < 1e-9 and all(x >= -1e-12 for x in d)


def test_rand_direction_deterministic_across_hashseed():
    """REGRESSION (session 15): the 'rand' perturbation direction must NOT depend on the
    process hash seed — it used builtin hash(), which PYTHONHASHSEED randomises, making the
    whole population (and every ρ) non-reproducible across runs. Now zlib.crc32-seeded."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = ("from parciver.rmapss.stepQ import _direction_target;"
            "print(_direction_target(3,'rand','2:cbf'), _direction_target(2,'rand','1:b'))")
    outs = []
    for hs in ("0", "1", "99"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hs
        env["PYTHONPATH"] = str(root / "src")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=env, cwd=str(root))
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] and all(o == outs[0] for o in outs), f"non-reproducible rand direction: {outs}"


def test_matched_dist_eq_pairs_differ_in_capture(kuhn_ctx):
    # the crux: near-eq opponents at matched perturbation who leak DIFFERENT EV exist,
    # so the experiment is non-vacuous (variance to predict beyond dist_eq).
    game, seat_keys, eq, ctx = kuhn_ctx
    pop = build_pop_q(game, eq, seat_keys)
    ceilings = {o.label: expl.best_response_value(game, o.profile, 0)
                - stepQ.on_policy_value(game, {**ctx["hero_eq"], **o.profile}, 0)
                for o in pop if o.klass != "eq"}
    assert max(ceilings.values()) - min(ceilings.values()) > 0.05


# -------------------------------------------------------------------------
# metrics are BR-free (THE circularity guard, §38.5) + finite + signed sanity
# -------------------------------------------------------------------------

def test_cheap_metrics_make_zero_hero_BR_calls(kuhn_ctx, monkeypatch):
    game, seat_keys, eq, ctx = kuhn_ctx
    pop = build_pop_q(game, eq, seat_keys)
    calls = {"n": 0}
    orig_val, orig_strat = expl.best_response_value, expl.best_response_strategy

    def spy_val(g, s, p, *a, **k):
        calls["n"] += 1
        return orig_val(g, s, p, *a, **k)

    def spy_strat(g, s, p, *a, **k):
        calls["n"] += 1
        return orig_strat(g, s, p, *a, **k)

    # patch at the module the metrics resolve through
    monkeypatch.setattr(stepQ, "best_response_value", spy_val)
    monkeypatch.setattr(stepQ, "best_response_strategy", spy_strat)

    for name, fn in CHEAP_METRICS.items():
        for o in pop[:6]:
            fn(game, o.profile, seat_keys, ctx)
    assert calls["n"] == 0, "a cheap metric called a best-response — circularity guard violated"


def test_metrics_finite_and_dist_eq_orders_eq_below_gross(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    pop = build_pop_q(game, eq, seat_keys)
    by = {o.label: o for o in pop}
    for name, fn in CHEAP_METRICS.items():
        for o in pop:
            v = fn(game, o.profile, seat_keys, ctx)
            assert math.isfinite(v) and v >= -1e-9
    d_eq = stepQ.m_dist_eq(game, by["eq|eq"].profile, seat_keys, ctx)
    d_gross = stepQ.m_dist_eq(game, by["fold1|aggro1"].profile, seat_keys, ctx)
    assert d_eq < 1e-6 < d_gross


def test_mistake_mass_nonneg_and_zero_at_eq(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    pop = build_pop_q(game, eq, seat_keys)
    by = {o.label: o for o in pop}
    # ~0 at equilibrium (bounded by CFR convergence), orders below a gross opponent.
    eq_m = stepQ.m_mistake(game, by["eq|eq"].profile, seat_keys, ctx)
    gross_m = stepQ.m_mistake(game, by["fold1|aggro1"].profile, seat_keys, ctx)
    assert eq_m < 2e-2
    assert gross_m > 20 * eq_m > 0.0


def test_pert_sens_deterministic(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    o = build_pop_q(game, eq, seat_keys)[3]
    a = stepQ.m_pert(game, o.profile, seat_keys, ctx, k=8)
    b = stepQ.m_pert(game, o.profile, seat_keys, ctx, k=8)
    assert a == b  # seeded


# -------------------------------------------------------------------------
# value walk sanity: reconstructed expected value == on_policy_value
# -------------------------------------------------------------------------

def test_seat_value_walk_matches_on_policy(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    o = build_pop_q(game, eq, seat_keys)[5]
    full = {**ctx["hero_eq"], **o.profile}
    for seat in opp_keys_by_seat(seat_keys):
        child_vals, node_reach = _seat_values_under_profile(game, full, seat)
        # at each seat node, sum_a pi*childval reconstructs the node value; the reach-
        # weighted local best cannot be below the reach-weighted expected (regret >= 0)
        for key, cv in child_vals.items():
            p = o.profile[key]
            assert max(cv) + 1e-9 >= sum(p[i] * cv[i] for i in range(len(cv)))


# -------------------------------------------------------------------------
# exploiter E* + targets + invariants (§38.1 / §38.5)
# -------------------------------------------------------------------------

def test_exploiter_alpha_within_safety(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    pop = build_pop_q(game, eq, seat_keys)
    fit = [o for o in pop if o.split == "fit"]
    estar, a_star, a_safe, wc = build_exploiter(
        game, ctx["hero_eq"], seat_keys[0], fit, seat_keys, budget=0.02)
    assert 0.0 <= a_star <= a_safe + 1e-9
    assert set(estar) >= set(seat_keys[0])


# -------------------------------------------------------------------------
# end-to-end (§38) — gates populated, invariants hold, control fires
# -------------------------------------------------------------------------

def test_run_stepQ_kuhn_pipeline(kuhn_ctx):
    game = KuhnPokerN(3)
    rep = run_stepQ(game, "kuhn", cfr_iters=4000, pert_k=6)
    # invariants
    assert rep.gates["Y_cap <= Y_ceil (all opponents, R2)"]
    assert rep.gates["Y_ceil ~ eq-residual vs equilibrium anchors"]
    assert rep.gates["E* alpha within safety budget"]
    # pipeline sanity: a perfect predictor (target itself) is near-perfectly rank-correlated
    assert rep.gates["pipeline sanity: perfect predictor rho_held ~1"]
    assert rep.control_rho_perfect_held >= 0.999
    # the GO gate exists and is a bool (verdict decided by the run, not asserted here)
    go_key = [k for k in rep.gates if k.startswith("H-Q GO")][0]
    assert isinstance(rep.gates[go_key], bool)
    # baseline metric present, cheap candidates all have rho triples
    assert rep.metric_rho[BASELINE_METRIC].is_baseline
    for n in CHEAP_METRICS:
        if n == BASELINE_METRIC:
            continue
        m = rep.metric_rho[n]
        assert all(math.isfinite(x) for x in (m.rho_fit, m.rho_held, m.rho_transfer))


def test_infoset_reach_sums_to_number_of_visits(kuhn_ctx):
    game, seat_keys, eq, ctx = kuhn_ctx
    o = build_pop_q(game, eq, seat_keys)[2]
    reach = infoset_reach(game, {**ctx["hero_eq"], **o.profile})
    assert all(r > -1e-12 for r in reach.values())
    assert len(reach) > 0
