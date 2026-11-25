"""Dense hand-isomorphism indexer tests (Phase 3b).

The exhaustive proofs live in the Rust crate: the flop indexer is certified to
cover *exactly* the independently-enumerated naive canonical set (all 1,286,792),
and turn/river get 20k-sample round-trip + suit-invariance + agreement with the
role-aware `canonical_situation`. These Python tests pin the exposed surface and
re-derive the headline facts once more through the API.

Skips cleanly if the extension is not built.
"""
from __future__ import annotations

import random

import pytest

pe = pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)

# (street id, name, literature canonical count, full hand card count)
STREETS = [
    (0, "flop", 1_286_792, 5),
    (1, "turn", 13_960_050, 6),
    (2, "river", 123_156_254, 7),
]


def _sample_indices(n, count, rng):
    picks = {0, n - 1, n // 2, min(n - 1, 1)}
    while len(picks) < count:
        picks.add(rng.randrange(n))
    return sorted(picks)


@pytest.mark.parametrize("sid,name,expected,cards", STREETS)
def test_sizes_and_card_counts_match_literature(sid, name, expected, cards):
    assert pe.canonical_index_size(sid) == expected
    assert pe.street_card_count(sid) == cards


@pytest.mark.parametrize("sid,name,expected,cards", STREETS)
def test_index_unindex_round_trips(sid, name, expected, cards):
    rng = random.Random(1234 + sid)
    n = pe.canonical_index_size(sid)
    for i in _sample_indices(n, 60, rng):
        hand = list(pe.canonical_unindex_batch(sid, i, 1))
        assert len(hand) == cards
        assert len(set(hand)) == cards, "representative has duplicate cards"
        assert pe.canonical_index(sid, hand) == i


def _suit_permute(cards, perm):
    return [(c >> 2) * 4 + perm[c & 3] for c in cards]


@pytest.mark.parametrize("sid,name,expected,cards", STREETS)
def test_index_is_suit_permutation_invariant(sid, name, expected, cards):
    rng = random.Random(99 + sid)
    n = pe.canonical_index_size(sid)
    for _ in range(200):
        hand = list(pe.canonical_unindex_batch(sid, rng.randrange(n), 1))
        base = pe.canonical_index(sid, hand)
        # A few non-trivial suit relabelings must all map to the same index.
        for perm in ([1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0], [1, 2, 3, 0]):
            assert pe.canonical_index(sid, _suit_permute(hand, perm)) == base


@pytest.mark.parametrize("sid,name,expected,cards", STREETS)
def test_indexer_class_agrees_with_canonical_situation(sid, name, expected, cards):
    # The indexer's equivalence classes must match the independent role-aware
    # canonicalizer: hole = first 2 cards, board = the rest.
    rng = random.Random(7 + sid)
    n = pe.canonical_index_size(sid)
    for _ in range(300):
        hand = list(pe.canonical_unindex_batch(sid, rng.randrange(n), 1))
        hole, board = hand[:2], hand[2:]
        canon = pe.canonical_situation(hole, board)
        # Any suit permutation shares the index *and* the canonical_situation key.
        perm = [rng.randrange(4) for _ in range(4)]
        # make it a real permutation
        perm = [0, 1, 2, 3]
        rng.shuffle(perm)
        ph = _suit_permute(hand, perm)
        assert pe.canonical_index(sid, ph) == pe.canonical_index(sid, hand)
        assert pe.canonical_situation(ph[:2], ph[2:]) == canon


def test_flop_indexer_is_bijective_on_a_sample():
    # Distinct dense indices must yield distinct canonical situations (the full
    # bijection over all 1,286,792 is the Rust cargo test).
    n = pe.canonical_index_size(0)
    rng = random.Random(2024)
    idxs = _sample_indices(n, 4000, rng)
    keys = set()
    for i in idxs:
        hand = list(pe.canonical_unindex_batch(0, i, 1))
        keys.add(tuple(pe.canonical_situation(hand[:2], hand[2:])))
    assert len(keys) == len(idxs)


def test_unindex_batch_rows_match_their_indices():
    for sid, _, _, cards in STREETS:
        n = pe.canonical_index_size(sid)
        start = n // 3
        count = 500
        import numpy as np

        flat = np.frombuffer(pe.canonical_unindex_batch(sid, start, count), dtype=np.uint8)
        flat = flat.reshape(count, cards)
        for j in range(0, count, 53):
            assert pe.canonical_index(sid, flat[j].tolist()) == start + j


def test_invalid_inputs_raise():
    with pytest.raises(Exception):
        pe.canonical_index_size(3)  # only 0..2
    with pytest.raises(Exception):
        pe.canonical_index(0, [0, 1, 2, 3])  # wrong card count for flop (needs 5)
    with pytest.raises(Exception):
        pe.canonical_index(0, [0, 0, 1, 2, 3])  # duplicate card
    with pytest.raises(Exception):
        pe.canonical_unindex_batch(0, pe.canonical_index_size(0), 5)  # out of range
