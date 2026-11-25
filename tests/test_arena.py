# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Match runner (`PlayAgent.match_against`) validated against exact values.

On the small exact game the value of any fixed profile is computable exactly,
so a head-to-head estimate must land on it: an equilibrium agent against its
exact best response must lose exactly the best-response value, and the
equilibrium against itself in duplicate must be ~0. This tests dealing, seating,
the real-chip engine, payoffs and the bb/100 scaling in one go.
"""
import math

import pytest

pe = pytest.importorskip("parciver_engine")
if not hasattr(pe, "PlayAgent") or not hasattr(pe.PlayAgent, "match_against"):
    pytest.skip("engine built without the match runner", allow_module_level=True)

SB, BB, STACK = 1, 2, 4  # the small game's real stakes
DEALS = 60_000


@pytest.fixture(scope="module")
def eq():
    return pe.solve_nlhe_small(2000)


@pytest.mark.parametrize("br_seat", [0, 1])
def test_estimate_matches_exact_best_response_value(eq, br_seat):
    br_rows = pe.nlhe_small_best_response_strategy(eq, br_seat)
    exact = pe.nlhe_small_best_response_value(eq, br_seat)  # chips to the BR seat
    profile = dict(eq)
    profile.update(br_rows)
    eq_agent = pe.PlayAgent.from_strategy(eq, "small")
    br_agent = pe.PlayAgent.from_strategy(profile, "small")
    r = br_agent.match_against("agent", DEALS, SB, BB, STACK, mirror=False,
                               seat=br_seat, seed=3, other=eq_agent)
    expected = exact / BB * 100
    assert abs(r["bb_per_100"] - expected) < 4 * r["se"], (r, expected)
    assert r["a_sources"][0] > 0 and r["a_sources"][2] == 0  # all on-tree


def test_self_play_duplicate_is_exactly_zero(eq):
    # Same agent, same cards, same per-seat seeds in both seatings: the two plays
    # are identical with the seats swapped, so duplicate scoring cancels exactly.
    agent = pe.PlayAgent.from_strategy(eq, "small")
    r = agent.match_against("agent", DEALS, SB, BB, STACK, mirror=True, seed=9)
    assert r["bb_per_100"] == 0.0 and r["se"] == 0.0
    assert r["hands"] == 2 * DEALS


def test_baselines_run_and_are_deterministic(eq):
    agent = pe.PlayAgent.from_strategy(eq, "small")
    for opp in ["station", "pot_raiser", "min_raiser", "random", "perturbed"]:
        a = agent.match_against(opp, 2000, SB, BB, STACK, seed=1, perturb=1.7)
        b = agent.match_against(opp, 2000, SB, BB, STACK, seed=1, perturb=1.7)
        assert a == b and math.isfinite(a["bb_per_100"])


def test_full_grid_matches_run_off_grid():
    # No blueprint needed: an empty strategy plays the fallback everywhere, which
    # still exercises translation, the real engine and every baseline at 100bb.
    agent = pe.PlayAgent.from_strategy({}, "100bb_exact")
    for opp in ["station", "min_raiser", "random", "pot_raiser"]:
        r = agent.match_against(opp, 300, 50, 100, 10_000, seed=2)
        assert r["hands"] == 600 and math.isfinite(r["bb_per_100"])


def test_luck_corrected_score_keeps_the_mean():
    """VARRED.md test 6: the board-luck-corrected score `per_deal_bb_cv` is
    unbiased: its mean differs from the raw score's by less than 3 SE of the
    paired difference. Off by default, and on it changes no raw score."""
    agent = pe.PlayAgent.uniform()
    off = agent.match_against("random", 300, 50, 100, 10_000, seed=6)
    on = agent.match_against("random", 300, 50, 100, 10_000, seed=6, luck=True)
    assert "per_deal_bb_cv" not in off and on["per_deal_bb"] == off["per_deal_bb"]
    diff = [c - u for u, c in zip(on["per_deal_bb"], on["per_deal_bb_cv"])]
    n = len(diff)
    mean = sum(diff) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in diff) / (n - 1) / n)
    assert n == 300 and se > 0 and abs(mean) < 3 * se, (mean, se)


def test_aivat_score_keeps_the_mean():
    """records/rating-live/PREREG.md test 6 (suite size; the registered run is
    records/rating-live/test6.py): `per_deal_bb_aivat` (board and action luck
    removed) is unbiased: its mean differs from the raw score's by less than 3
    SE of the paired difference. A uniform agent mixes at every decision; its
    opponent (an empty strategy) plays the fallback. Off by default, and on it
    changes neither the raw nor the board-luck score; refused against a builtin."""
    agent, fixed = pe.PlayAgent.uniform(), pe.PlayAgent.from_strategy({}, "100bb_exact")
    run = lambda **k: agent.match_against("agent", 300, 50, 100, 10_000, seed=6, other=fixed, **k)
    luck, on = run(luck=True), run(aivat=True)
    assert "per_deal_bb_aivat" not in luck
    assert (on["per_deal_bb"], on["per_deal_bb_cv"]) == (luck["per_deal_bb"], luck["per_deal_bb_cv"])
    diff = [c - u for u, c in zip(on["per_deal_bb"], on["per_deal_bb_aivat"])]
    n = len(diff)
    mean = sum(diff) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in diff) / (n - 1) / n)
    assert n == 300 and se > 0 and abs(mean) < 3 * se, (mean, se)
    assert on["per_deal_bb_aivat"] != on["per_deal_bb_cv"]
    with pytest.raises(ValueError):
        agent.match_against("station", 4, 50, 100, 10_000, aivat=True)


def _merge(parts):
    return ([x for p in parts for x in p["per_deal_bb"]],
            [sum(v) for v in zip(*(p.get("a_sources", [0]) for p in parts))])


@pytest.mark.parametrize("opponent", ["station", "random", "min_raiser"])
def test_chunked_match_equals_one_call(opponent):
    """Deal i depends only on (seed, i): a match played in chunks via
    `first_deal` is identical to one call (what the live scorecard relies on)."""
    eq = pe.solve_nlhe_small(200)
    agent = pe.PlayAgent.from_strategy(eq, "small")
    whole = agent.match_against(opponent, 60, 1, 2, 4, seed=9)
    parts = [agent.match_against(opponent, 20, 1, 2, 4, seed=9, first_deal=k) for k in (0, 20, 40)]
    per_deal, sources = _merge(parts)
    assert per_deal == whole["per_deal_bb"]
    assert sources == whole["a_sources"]


def test_chunked_lbr_and_search_equal_one_call():
    eq = pe.solve_nlhe_small(200)
    agent = pe.PlayAgent.from_strategy(eq, "small")
    whole = agent.lbr(40, 1, 2, 4, seed=4)
    parts = [agent.lbr(20, 1, 2, 4, seed=4, first_deal=k) for k in (0, 20)]
    assert _merge(parts)[0] == whole["per_deal_bb"]
    # A searching agent: fresh agents (empty solve caches) per chunk change nothing.
    make = lambda: pe.PlayAgent.uniform().with_search(river_iters=10)  # noqa: E731
    whole = make().match_against("station", 24, 50, 100, 10_000, seed=3)
    parts = [make().match_against("station", 12, 50, 100, 10_000, seed=3, first_deal=k) for k in (0, 12)]
    per_deal, sources = _merge(parts)
    assert per_deal == whole["per_deal_bb"] and sources == whole["a_sources"]
    assert whole["a_sources"][3] > 0 and whole["a_search_failed"] == [0, 0]  # search ran, and never failed
