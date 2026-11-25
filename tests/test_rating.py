"""The rating's math (RATING.md): known-answer meta-games, invariance, the paired interval."""
import numpy as np
import pytest

from parciver import rating as R

RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def test_rock_paper_scissors_is_uniform():
    p = R.nash(RPS)
    np.testing.assert_allclose(p, [1 / 3] * 3, atol=1e-12)
    assert R.check_nash(RPS, p) <= 1e-12


def test_transitive_pool_puts_all_weight_on_the_strongest():
    # 2 beats 1 beats 0.
    A = np.array([[0.0, -5.0, -9.0], [5.0, 0.0, -3.0], [9.0, 3.0, 0.0]])
    np.testing.assert_allclose(R.nash(A), [0, 0, 1], atol=1e-12)


def test_a_duplicated_anchor_splits_its_weight_and_changes_no_rating():
    A4 = np.zeros((4, 4))
    A4[:3, :3] = RPS
    A4[3, :3] = RPS[0]
    A4[:3, 3] = RPS[:, 0]  # anchor 3 is a copy of anchor 0 (rock)
    p4 = R.nash(A4)
    np.testing.assert_allclose(p4, [1 / 6, 1 / 3, 1 / 3, 1 / 6], atol=1e-6)
    # A candidate's rating against the pool is the same with and without the copy.
    rng = np.random.default_rng(0)
    n = 20_000
    x = {j: rng.normal(mu, 50.0, n) for j, mu in enumerate([10.0, -4.0, 7.0])}
    ref = {j: rng.normal(0.0, 50.0, n) for j in range(3)}
    x4, ref4 = dict(x), dict(ref)
    x4[3], ref4[3] = x[0], ref[0]
    r3 = R.paired(x, ref, dict(enumerate(R.nash(RPS))))
    r4 = R.paired(x4, ref4, dict(enumerate(p4)))
    # (The interval differs: a real copy is played on its own seed, i.e. more deals against "rock".)
    assert abs(r3["pr"] - r4["pr"]) < 1e-6


def test_the_reference_is_exactly_1500():
    rng = np.random.default_rng(1)
    ref = {j: rng.normal(0.0, 80.0, 12_000) for j in range(3)}
    r = R.paired(ref, ref, {0: 0.2, 1: 0.0, 2: 0.8})
    assert r["pr"] == 1500.0 and r["se"] == 0.0 and r["anchors"] == [0, 2]


def test_paired_interval_matches_the_formula():
    rng = np.random.default_rng(2)
    n = 50_000
    luck = rng.normal(0.0, 300.0, n)  # shared card luck cancels in the pair
    x = {0: luck + 25.0 + rng.normal(0, 20, n), 1: -luck + 5.0 + rng.normal(0, 20, n)}
    ref = {0: luck + rng.normal(0, 20, n), 1: -luck + rng.normal(0, 20, n)}
    w = {0: 0.25, 1: 0.75}
    r = R.paired(x, ref, w)
    d0, d1 = x[0] - ref[0], x[1] - ref[1]
    assert r["pr"] == pytest.approx(1500 + 0.25 * d0.mean() + 0.75 * d1.mean(), abs=1e-9)
    # Anchors are played on independent seeds: their variances add (no cross-anchor covariance term).
    assert r["se"] == pytest.approx(np.sqrt((0.25**2 * d0.var(ddof=1) + 0.75**2 * d1.var(ddof=1)) / n), rel=1e-12)
    assert r["lo"] < 1500 + 0.25 * 25 + 0.75 * 5 < r["hi"]
    # Pairing removes the luck: the interval is far tighter than unpaired.
    assert r["se"] < 0.2 * np.sqrt(0.25**2 * x[0].var() + 0.75**2 * x[1].var()) / np.sqrt(n) * np.sqrt(2)


def test_refuses_what_it_cannot_measure():
    a = {0: np.zeros(20_000)}
    with pytest.raises(R.RatingError):
        R.paired({0: np.zeros(100)}, {0: np.zeros(100)}, {0: 1.0})  # too few deals
    with pytest.raises(R.RatingError):
        R.paired(a, {0: np.zeros(19_999)}, {0: 1.0})  # unpaired lengths
    with pytest.raises(R.RatingError):
        R.paired({}, a, {0: 1.0})  # missing anchor
    with pytest.raises(R.RatingError):
        R.paired(a, a, {0: 0.5})  # weights not a distribution
    with pytest.raises(R.RatingError):
        R.nash(np.array([[0.0, 1.0], [1.0, 0.0]]))  # not zero-sum


def test_array_hash_catches_one_changed_value():
    v = np.linspace(-3, 3, 1001)
    h = R.array_sha(v)
    w = v.copy()
    w[500] = np.nextafter(w[500], 1.0)
    assert R.array_sha(w) != h and R.array_sha(v.copy()) == h


def test_a_longer_reference_pairs_its_shared_deals():
    # Amendment 2: the reference plays R > D deals; the first D are the candidate's deals.
    rng = np.random.default_rng(4)
    D, Rn = 20_000, 80_000
    luck = rng.normal(0.0, 300.0, Rn)
    ref = {0: luck + rng.normal(0, 20, Rn)}
    x = {0: luck[:D] + 7.0 + rng.normal(0, 20, D)}
    r = R.paired(x, ref, {0: 1.0})
    a, b = x[0], ref[0]
    c = np.cov(a, b[:D], ddof=1)[0, 1]
    var = a.var(ddof=1) / D + b.var(ddof=1) / Rn - 2 * c * D / (D * Rn)
    assert r["pr"] == pytest.approx(1500 + a.mean() - b.mean(), abs=1e-9)
    assert r["se"] == pytest.approx(np.sqrt(var), rel=1e-12)
    assert r["lo"] < 1507.0 < r["hi"]
    # The shared deals' luck cancels: tighter than the unpaired interval (only partly, as R > D).
    assert r["se"] < 0.85 * np.sqrt(a.var() / D + b.var() / Rn)


def test_change_is_the_difference_of_ratings_without_the_reference():
    rng = np.random.default_rng(5)
    n = 30_000
    luck = {j: rng.normal(0.0, 300.0, n) for j in range(2)}
    ref = {j: luck[j] + rng.normal(0, 20, n) for j in range(2)}
    x = {j: luck[j] + 10.0 + rng.normal(0, 20, n) for j in range(2)}
    y = {j: luck[j] + 4.0 + rng.normal(0, 20, n) for j in range(2)}
    w = {0: 0.4, 1: 0.6}
    c = R.change(x, y, w)
    assert c["mean"] == pytest.approx(R.paired(x, ref, w)["pr"] - R.paired(y, ref, w)["pr"], abs=1e-9)
    assert c["lo"] < 6.0 < c["hi"]
    with pytest.raises(R.RatingError):
        R.change({0: x[0]}, y, w)  # missing anchor
