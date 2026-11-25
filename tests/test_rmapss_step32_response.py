# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""S3.2 sub-step 2: sampled fixed-opponent response, exact-control first.

The same response algorithm intended for the full abstraction is exercised only
on `nlhe_small` here, where the certified exact best response supplies a real
ceiling. This is an engineering/control gate, not a full-game S3.2 outcome.
"""
from __future__ import annotations

import pytest

pe = pytest.importorskip("parciver_engine")

from parciver.rmapss import step3  # noqa: E402


@pytest.fixture(scope="module")
def exact_fixture():
    anchor = pe.solve_nlhe_small_mccfr(120_000, "linear", 73)
    structure = step3.infoset_structure()
    p0, p1 = step3.split_keys(structure)
    boundary = pe.nlhe_small_observed_marginal(anchor, anchor, 0)
    return anchor, structure, p0, p1, boundary


@pytest.mark.parametrize("kind", ["nit", "station", "maniac"])
def test_sampled_response_stays_under_exact_br_and_captures_gain(exact_fixture, kind):
    anchor, structure, p0, p1, boundary = exact_fixture
    opponent = step3.build_opponent(
        structure, p1, anchor, kind, board_blind_flat=boundary
    )
    anchor_hero = {
        key: list(anchor.get(key, [1.0 / len(structure[key][1])] * len(structure[key][1])))
        for key in p0
    }
    fixed_profile = {**anchor, **opponent}
    baseline = pe.nlhe_small_on_policy_value({**anchor_hero, **opponent}, 0)
    exact = pe.nlhe_small_best_response_value(fixed_profile, 0)

    response = pe.nlhe_small_fixed_response(
        fixed_profile, 0, 500_000, "linear", 1_000 + len(kind), 1
    )
    realized = pe.nlhe_small_on_policy_value({**opponent, **response}, 0)
    assert set(response) <= set(p0)
    assert realized <= exact + 1e-9, "sampled response cannot exceed the exact BR"
    available = exact - baseline
    assert available > 0
    captured = (realized - baseline) / available
    assert captured >= 0.95, (kind, baseline, realized, exact, captured)


def test_sampled_response_rejects_invalid_request(exact_fixture):
    anchor, *_ = exact_fixture
    with pytest.raises(ValueError, match="responder"):
        pe.nlhe_small_fixed_response(anchor, 2, 10)
    with pytest.raises(ValueError, match="iterations"):
        pe.nlhe_small_fixed_response(anchor, 0, 0)
