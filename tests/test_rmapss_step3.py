"""R-MAPSS Step 3 (S3.1) — engine exploitation-spine primitives on `nlhe_small`.

Export-level parity + §28 controls for the new `nlhe_small_*` exports added for
the HU exploitation spine (RMAPSS.md §40): exact best-response VALUE and STRATEGY
vs a fixed opponent, realized on-policy value, and the Hero-observable opponent
marginal. These certify the primitives; the full spine + go/no-go gates live in
`parciver.rmapss.step3` / `test` for sub-step 2.

The parity holds for ANY anchor, so a cheap MCCFR anchor keeps the test fast.
"""
import pytest

pe = pytest.importorskip("parciver_engine")


@pytest.fixture(scope="module")
def anchor():
    # Cheap approximate-equilibrium anchor (MCCFR is far faster than full DCFR).
    return pe.solve_nlhe_small_mccfr(40_000, "linear", 0)


def test_br_strategy_realizes_br_value(anchor):
    """The exact BR STRATEGY, realized against the fixed opponent, reproduces the
    exact BR VALUE — the load-bearing consistency between the two new exports."""
    for hero in (0, 1):
        br_val = pe.nlhe_small_best_response_value(anchor, hero)
        on_val = pe.nlhe_small_on_policy_value(anchor, hero)
        br_strat = pe.nlhe_small_best_response_strategy(anchor, hero)
        realized = pe.nlhe_small_on_policy_value({**anchor, **br_strat}, hero)
        assert br_val >= on_val - 1e-9, "BR value must be >= on-policy value"
        assert abs(realized - br_val) < 1e-9, "BR strategy must realize the BR value"
        # every br row is a valid one-hot distribution
        for v in br_strat.values():
            assert abs(sum(v) - 1.0) < 1e-9
            assert v.count(1.0) == 1 and all(x in (0.0, 1.0) for x in v)


def test_exploitability_is_br_average(anchor):
    """The exact per-player exploitability equals the mean of the two BR values —
    ties the standalone BR-value export to the certified `exploitability`."""
    expl = pe.nlhe_small_exploitability(anchor)
    br0 = pe.nlhe_small_best_response_value(anchor, 0)
    br1 = pe.nlhe_small_best_response_value(anchor, 1)
    assert abs(expl - (br0 + br1) / 2.0) < 1e-9


def test_observed_marginal_valid_and_ceiling_bounded(anchor):
    """The Hero-observed opponent marginal is a valid distribution per opponent
    infoset, and best-responding to it never beats the exact BR ceiling (§28: no
    hallucinated exploit)."""
    flat = pe.nlhe_small_observed_marginal(anchor, anchor, 0)
    assert len(flat) > 0
    for v in flat.values():
        assert abs(sum(v) - 1.0) < 1e-9
        assert all(x >= -1e-12 for x in v)
    # opponent keys sharing a Hero-observable context get the SAME marginal
    a2_strat = pe.nlhe_small_best_response_strategy(flat, 0)
    profile = {**anchor, **a2_strat}
    assert all(profile[k] == v for k, v in a2_strat.items())
    a2_realized = pe.nlhe_small_on_policy_value(profile, 0)
    ceiling = pe.nlhe_small_best_response_value(anchor, 0)
    assert a2_realized <= ceiling + 1e-9, "observed-model BR must not exceed exact BR"


def test_combined_observation_matches_separate_exports(anchor):
    # Both seats, including a distorted policy: independent export paths must
    # reproduce every key, probability and reach mass, not just a final verdict.
    uniform = {k: [1.0 / len(v)] * len(v) for k, v in anchor.items()}
    for hero in (0, 1):
        for opp in (anchor, uniform):
            flat, nodes, mass = pe.nlhe_small_observed_model(anchor, opp, hero)
            assert flat == pe.nlhe_small_observed_marginal(anchor, opp, hero)
            assert (nodes, mass) == pe.nlhe_small_observed_marginal_nodes(anchor, opp, hero)
