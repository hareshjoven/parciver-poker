"""Equity and suit-isomorphism tests (Phase 3a).

The exhaustive/exact gates (1326→169 preflop classes, flops→1755, hero+villain
complementarity, board-plays ties, Monte-Carlo↔exact convergence) live in the
Rust crate where they run fast. These tests pin the Python-facing surface and
the poker facts a human can eyeball: the textbook preflop equity ladder, exact
showdowns, symmetry, and input validation.

Skips cleanly if the extension is not built
(`maturin develop -m engine/Cargo.toml`).
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)

from parciver import abstraction as ab


# --- preflop classes -------------------------------------------------------

def test_preflop_table_has_169_unique_classes():
    assert len(ab.PREFLOP_CLASS_NAMES) == 169
    assert len(set(ab.PREFLOP_CLASS_NAMES)) == 169
    assert ab.PREFLOP_CLASS_NAMES[:3] == ["22", "33", "44"]  # pairs first, by rank
    assert "AA" in ab.PREFLOP_CLASS_NAMES
    assert "AKs" in ab.PREFLOP_CLASS_NAMES
    assert "AKo" in ab.PREFLOP_CLASS_NAMES


def test_preflop_class_names_are_correct():
    assert ab.preflop_class_name("Ac", "Ad") == "AA"
    assert ab.preflop_class_name("Ac", "Kc") == "AKs"
    assert ab.preflop_class_name("Ac", "Kd") == "AKo"
    assert ab.preflop_class_name("7c", "2d") == "72o"
    # Card order is irrelevant.
    assert ab.preflop_class("Ac", "Kc") == ab.preflop_class("Kc", "Ac")


def test_card_helpers_round_trip():
    for name in ("As", "2c", "Td", "9h", "Kc"):
        assert ab.card_name(ab.card(name)) == name
    # ints pass through, strings parse.
    assert ab.card(0) == 0
    assert ab.card("2c") == 0


# --- suit isomorphism ------------------------------------------------------

def test_canonical_dedupes_isomorphic_boards():
    # Same ranks, all clubs vs all diamonds: isomorphic monotone flops.
    assert ab.canonical(["Ac", "Kc", "Qc"]) == ab.canonical(["Ad", "Kd", "Qd"])
    # A rainbow flop is NOT isomorphic to a monotone one of the same ranks.
    assert ab.canonical(["Ac", "Kd", "Qh"]) != ab.canonical(["Ac", "Kc", "Qc"])


# --- exact equity ----------------------------------------------------------

def test_exact_showdown_aces_beat_kings():
    # A pair of aces outright beats a pair of kings on this river: equity 1.0.
    e = ab.equity_vs_hand(["Ac", "Ad"], ["Kc", "Kd"],
                          ["2h", "7s", "Ts", "Jd", "3c"])
    assert e == 1.0


def test_turn_equities_are_complementary():
    # Over the same 44 river runouts the two players' equities sum to exactly 1.
    hero, villain = ["Ah", "Kh"], ["Qs", "Qd"]
    board = ["2h", "7h", "Ts", "Jd"]
    a = ab.equity_vs_hand(hero, villain, board)
    b = ab.equity_vs_hand(villain, hero, board)
    assert abs(a + b - 1.0) < 1e-12


def test_board_plays_gives_everyone_half():
    # Royal flush on the board: nobody can win, everybody chops.
    e = ab.equity(["2c", "3d"], ["Ah", "Kh", "Qh", "Jh", "Th"])
    assert e == 0.5


def test_default_dispatch_uses_exact_postflop_and_mc_preflop(monkeypatch):
    # Postflop (>=3 board cards) must be exact; preflop must sample. Confirm by
    # checking the exact path is reproducible and the sampled path depends on seed.
    flop = ["Ad", "7h", "2s"]
    assert ab.equity(["Ac", "Kc"], flop) == ab.equity(["Ac", "Kc"], flop)  # exact, stable
    a = ab.equity(["Ac", "Ad"], samples=20_000, seed=1)
    b = ab.equity(["Ac", "Ad"], samples=20_000, seed=2)
    assert a != b  # different seeds -> different sample -> (almost surely) differ


# --- textbook preflop equity ladder ---------------------------------------

@pytest.mark.parametrize(
    "hole, lo, hi",
    [
        (["Ac", "Ad"], 0.84, 0.86),  # AA ~85.2%
        (["Kc", "Kd"], 0.81, 0.84),  # KK ~82.4%
        (["Qc", "Qd"], 0.79, 0.81),  # QQ ~79.9%
        (["Ac", "Kc"], 0.66, 0.68),  # AKs ~67.0%
        (["7c", "2d"], 0.31, 0.36),  # 72o, the worst starting hand ~34.6%
    ],
)
def test_preflop_equity_ladder_matches_textbook(hole, lo, hi):
    e = ab.equity(hole, samples=300_000, seed=2024)
    assert lo <= e <= hi, f"{hole}: {e} not in [{lo}, {hi}]"


def test_stronger_preflop_hands_have_more_equity():
    seed = 777
    aa = ab.equity(["Ac", "Ad"], samples=300_000, seed=seed)
    kk = ab.equity(["Kc", "Kd"], samples=300_000, seed=seed)
    aks = ab.equity(["Ac", "Kc"], samples=300_000, seed=seed)
    trash = ab.equity(["7c", "2d"], samples=300_000, seed=seed)
    assert aa > kk > aks > trash


# --- input validation ------------------------------------------------------

def test_equity_rejects_bad_input():
    with pytest.raises(ValueError):
        ab.equity(["Ac"], ["Kc", "Qc", "Jc"])  # hole must be 2 cards
    with pytest.raises(ValueError):
        ab.equity(["Ac", "Ac"], ["Kc", "Qc", "Jc"])  # duplicate card
    with pytest.raises(ValueError):
        ab.equity(["Ac", "Kd"], ["Ac", "Qc", "Jc"])  # board collides with hole
