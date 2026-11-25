# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Equity-distribution feature-extraction tests (Phase 3b).

The heavy correctness proofs live in the Rust crate: `eval7_fast` is certified
bit-identical to the reference `eval7` (exhaustively over all 133,784,560 hands
in the `--ignored` gate, and over a 3M random sample every build), and the
histogram/canonical-situation logic has Rust unit tests. These pin the
Python-facing surface and the poker facts a human can eyeball.

Skips cleanly if the extension is not built
(`maturin develop -m engine/Cargo.toml`).
"""
from __future__ import annotations

import pytest

pe = pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)


def cards(*names):
    return [pe.parse_card(n) for n in names]


def hist_mean(h):
    bins = len(h)
    return sum(((b + 0.5) / bins) * m for b, m in enumerate(h))


# --- flop histograms -------------------------------------------------------

def test_flop_histogram_is_a_normalized_distribution():
    h = pe.flop_equity_histogram(cards("Ac", "Ad"), cards("Ah", "7d", "2c"), 50)
    assert len(h) == 50
    assert abs(sum(h) - 1.0) < 1e-5
    assert all(m >= 0.0 for m in h)


def test_flop_made_hand_beats_trash():
    # Flopped top set sits high; 72o on a broadway board is nearly dead.
    made = pe.flop_equity_histogram(cards("Ac", "Ad"), cards("Ah", "7d", "2c"), 50)
    trash = pe.flop_equity_histogram(cards("7c", "2d"), cards("Ah", "Kd", "Qs"), 50)
    assert hist_mean(made) > 0.85
    assert hist_mean(trash) < 0.35


def test_flop_extraction_is_deterministic():
    a = pe.flop_equity_histogram(cards("Ts", "Js"), cards("Qs", "9d", "2c"), 30)
    b = pe.flop_equity_histogram(cards("Ts", "Js"), cards("Qs", "9d", "2c"), 30)
    assert a == b


# --- turn histograms -------------------------------------------------------

def test_turn_histogram_orders_correctly():
    nuts = pe.turn_equity_histogram(cards("Ah", "Kh"), cards("Qh", "Jh", "2h", "3d"), 50)
    weak = pe.turn_equity_histogram(cards("7c", "2d"), cards("Ah", "Kd", "Qs", "Jc"), 50)
    assert abs(sum(nuts) - 1.0) < 1e-5
    assert hist_mean(nuts) > hist_mean(weak)
    assert hist_mean(nuts) > 0.9


# --- river scalar ----------------------------------------------------------

def test_river_equity_matches_engine():
    e = pe.river_equity(cards("Ac", "Ad"), cards("Ah", "7d", "2c", "Ts", "Js"))
    assert e > 0.9  # top set crushes a random hand on the river
    # Consistent with the general equity engine on the same 5-card board.
    assert abs(e - pe.equity_vs_random(cards("Ac", "Ad"), cards("Ah", "7d", "2c", "Ts", "Js"))) < 1e-12


# --- batch == single -------------------------------------------------------

def test_flop_batch_matches_single_rows():
    holes = cards("Ac", "Ad", "7c", "2d")
    flops = cards("Ah", "7d", "2s", "Ah", "Kd", "Qs")
    bins = 40
    batch = pe.flop_histograms_batch(holes, flops, bins)
    row0 = pe.flop_equity_histogram(holes[0:2], flops[0:3], bins)
    row1 = pe.flop_equity_histogram(holes[2:4], flops[3:6], bins)
    assert batch[0:bins] == row0
    assert batch[bins:2 * bins] == row1


def test_river_batch_matches_single_rows():
    holes = cards("Ac", "Ad", "7c", "2d")
    boards = cards("Ah", "7d", "2s", "Ts", "Js", "Kh", "Qd", "9s", "4c", "3h")
    batch = pe.river_equities_batch(holes, boards)
    assert abs(batch[0] - pe.river_equity(holes[0:2], boards[0:5])) < 1e-6
    assert abs(batch[1] - pe.river_equity(holes[2:4], boards[5:10])) < 1e-6


# --- canonical situation (role-aware) --------------------------------------

def test_canonical_situation_distinguishes_hole_from_board():
    # Same five cards, split two ways: holding the flush != flush on the board.
    a = pe.canonical_situation(cards("Ac", "Kc"), cards("Qc", "Jd", "2h"))
    b = pe.canonical_situation(cards("Qc", "Jd"), cards("Ac", "Kc", "2h"))
    assert a != b


def test_canonical_situation_is_suit_permutation_invariant():
    a = pe.canonical_situation(cards("As", "Ks"), cards("Qs", "Js", "2c"))
    b = pe.canonical_situation(cards("Ah", "Kh"), cards("Qh", "Jh", "2d"))
    assert a == b


# --- input validation ------------------------------------------------------

def test_feature_input_validation():
    with pytest.raises(ValueError):
        pe.flop_equity_histogram(cards("Ac"), cards("Kc", "Qc", "Jc"), 50)  # hole size
    with pytest.raises(ValueError):
        pe.flop_equity_histogram(cards("Ac", "Kd"), cards("Kc", "Qc"), 50)  # flop size
    with pytest.raises(ValueError):
        pe.flop_equity_histogram(cards("Ac", "Kd"), cards("Ac", "Qc", "Jc"), 50)  # dup card
    with pytest.raises(ValueError):
        pe.flop_equity_histogram(cards("Ac", "Kd"), cards("2c", "Qc", "Jc"), 0)  # bins >= 1
