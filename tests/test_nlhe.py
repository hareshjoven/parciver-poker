# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""HU NLHE tests (Phase 3c).

B2 — card-abstraction infoset key. The Rust NLHE game keys its information sets on
the Phase-3b bucket abstraction, loaded in Rust by ``parciver_engine.Abstraction``.
The correctness that matters is *parity*: the Rust ``Abstraction.bucket_of`` must
equal the Python reference ``parciver.bucketing.bucket_of`` for every situation,
and the preflop class must equal ``preflop_index``. Both sides call the same
``canonical_index``, so this certifies the Rust ``.npy`` reader loaded the
persisted table values faithfully and mapped each street correctly.

B3 — the solver correctness bridge. An exact best-response engine in Rust
(``exploit.rs``) is meta-gated against the independent Python oracle on Kuhn and
Leduc (they must agree to ~1e-9), then used to score a tiny full-traversal NLHE
game (``NlheConfig::small``): full CFR and external-sampling MCCFR must both drive
exact exploitability down toward zero and agree — the NLHE analog of the
Kuhn/Leduc gate, before any large blueprint solve.

The bucket-parity tests skip cleanly if the baked tables are absent (they are
gitignored; build them with ``scripts/build_buckets.py``); the B3 tests need only
the extension.
"""
from __future__ import annotations

import random

import pytest

pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)
pytest.importorskip("torch")

import parciver_engine as pe
from parciver import bucketing as B

STREETS = ("flop", "turn", "river")


def _tables_present() -> bool:
    d = B.default_out_dir()
    return all(B._paths(d, s)["buckets"].exists() for s in STREETS)


baked = pytest.mark.skipif(
    not _tables_present(),
    reason="baked bucket tables absent; run scripts/build_buckets.py",
)


@pytest.fixture(scope="module")
def abstraction():
    """The real flop/turn/river bucket tables, loaded once (≈277 MB resident)."""
    return pe.Abstraction(str(B.default_out_dir()))


@baked
def test_postflop_bucket_parity_rust_vs_python(abstraction):
    # For random canonical situations of each street, the Rust bucket must equal
    # the Python reference lookup table[canonical_index(...)].
    rng = random.Random(20260911)
    for street in STREETS:
        sid = B.STREET_IDS[street]
        n = pe.canonical_index_size(sid)
        for _ in range(300):
            i = rng.randrange(n)
            hand = list(pe.canonical_unindex_batch(sid, i, 1))
            hole, board = hand[:2], hand[2:]
            rust = abstraction.bucket_of(hole, board)
            py = B.bucket_of(street, hand)
            assert rust == py, f"{street} idx {i}: rust {rust} != python {py}"


@baked
def test_preflop_bucket_is_the_169_classes(abstraction):
    # An empty board => the preflop class, matching preflop_index exactly.
    rng = random.Random(7)
    for _ in range(500):
        a, b = rng.sample(range(52), 2)
        assert abstraction.bucket_of([a, b], []) == pe.preflop_index(a, b)
    # And every class 0..169 is reachable / in range.
    seen = {abstraction.bucket_of(list(pair), []) for pair in _all_holdings()}
    assert seen == set(range(169))


def _all_holdings():
    return ((a, b) for a in range(52) for b in range(a + 1, 52))


@baked
def test_bucket_of_is_suit_permutation_invariant(abstraction):
    # A consistent suit relabeling must not change any bucket, on any street.
    perm = [1, 2, 3, 0]
    relabel = lambda cs: [(c >> 2) * 4 + perm[c & 3] for c in cs]
    rng = random.Random(99)
    for street in STREETS:
        sid = B.STREET_IDS[street]
        n = pe.canonical_index_size(sid)
        for _ in range(100):
            hand = list(pe.canonical_unindex_batch(sid, rng.randrange(n), 1))
            base = abstraction.bucket_of(hand[:2], hand[2:])
            perm_hand = relabel(hand)
            assert abstraction.bucket_of(perm_hand[:2], perm_hand[2:]) == base


@baked
def test_bucket_ids_are_in_range(abstraction):
    # Postflop bucket ids fall within [0, k) for the street's k (from meta).
    import json

    rng = random.Random(3)
    d = B.default_out_dir()
    for street in STREETS:
        sid = B.STREET_IDS[street]
        k = json.loads(B._paths(d, street)["meta"].read_text())["k"]
        n = pe.canonical_index_size(sid)
        for _ in range(200):
            hand = list(pe.canonical_unindex_batch(sid, rng.randrange(n), 1))
            bucket = abstraction.bucket_of(hand[:2], hand[2:])
            assert 0 <= bucket < k


@baked
def test_bucket_of_rejects_bad_input(abstraction):
    with pytest.raises(ValueError):
        abstraction.bucket_of([0], [1, 2, 3])  # hole not 2 cards
    with pytest.raises(ValueError):
        abstraction.bucket_of([0, 1], [2, 3])  # board length 2 is not a street
    with pytest.raises(ValueError):
        abstraction.bucket_of([0, 52], [])  # card index out of range
    with pytest.raises(ValueError):
        abstraction.bucket_of([0, 1], [1, 2, 3])  # duplicate card (1 in hole+board)


def test_abstraction_load_errors_on_missing_dir(tmp_path):
    # A directory with no bucket tables must fail loudly at construction.
    with pytest.raises(ValueError):
        pe.Abstraction(str(tmp_path / "does_not_exist"))


# --- B3: exact best response + small NLHE cross-validation -----------------


def test_rust_best_response_matches_python_oracle():
    # META-GATE: the Rust exact best-response must equal the independent Python
    # exploitability oracle on Kuhn AND Leduc, which is what licenses trusting it
    # on the small NLHE game where there is no Python game to check against.
    from parciver.eval.exploitability import exploitability as py_expl
    from parciver.games.kuhn import KuhnPoker
    from parciver.games.leduc import LeducHoldem

    kuhn = pe.solve_kuhn(3000, "dcfr")
    leduc = pe.solve_leduc(500, "dcfr")
    assert abs(pe.kuhn_exploitability(kuhn) - py_expl(KuhnPoker(), kuhn)) < 1e-6
    assert abs(pe.leduc_exploitability(leduc) - py_expl(LeducHoldem(), leduc)) < 1e-6


def test_small_nlhe_is_small_and_solvable():
    # The gate game must stay tiny enough to full-traverse; a blow-up here (e.g. a
    # dealing/abstraction regression) would make CFR + exact BR intractable.
    n = pe.nlhe_small_infoset_count()
    assert 100 < n < 5000, f"small NLHE infoset count {n} out of expected range"


def test_small_nlhe_cfr_converges():
    # Full-traversal CFR must drive exact exploitability down monotonically.
    lo = pe.nlhe_small_exploitability(pe.solve_nlhe_small(60, "dcfr"))
    hi = pe.nlhe_small_exploitability(pe.solve_nlhe_small(250, "dcfr"))
    assert hi < lo, f"CFR exploitability did not fall: {lo} -> {hi}"
    assert hi < 0.008, f"CFR exploitability {hi} not low enough at 250 iters"


def test_small_nlhe_mccfr_converges_and_agrees_with_cfr():
    # External-sampling MCCFR (the scalable solver) must ALSO converge, and land in
    # the same low-exploitability neighborhood the full solver reaches — the
    # cross-validation that certifies MCCFR on an NLHE-shaped tree.
    coarse = pe.nlhe_small_exploitability(pe.solve_nlhe_small_mccfr(50_000, "linear", 0))
    fine = pe.nlhe_small_exploitability(pe.solve_nlhe_small_mccfr(500_000, "linear", 0))
    assert fine < coarse, f"MCCFR exploitability did not fall: {coarse} -> {fine}"
    assert fine < 0.006, f"MCCFR exploitability {fine} not low enough at 500k iters"

    cfr = pe.nlhe_small_exploitability(pe.solve_nlhe_small(250, "dcfr"))
    assert abs(fine - cfr) < 0.004, f"MCCFR {fine} and CFR {cfr} disagree"


def test_small_nlhe_mccfr_is_deterministic():
    a = pe.solve_nlhe_small_mccfr(20_000, "linear", 123)
    b = pe.solve_nlhe_small_mccfr(20_000, "linear", 123)
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] for k in a)
