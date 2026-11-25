"""Python-API tests for the Rust 7-card hand evaluator.

The exhaustive correctness gates (every 5-card and every 7-card hand matched
against the known category counts) live in the Rust crate, where they run fast
enough to cover all 133,784,560 seven-card hands. These tests exercise the
exposed Python surface and pin the poker orderings a human can eyeball: category
ranking, kicker resolution, the ace-low wheel, and best-of-seven selection.

Skips cleanly if the extension is not built
(`maturin develop -m engine/Cargo.toml`).
"""
from __future__ import annotations

import pytest

engine = pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)


def hand(*cards: str) -> list[int]:
    return [engine.parse_card(c) for c in cards]


def cat(cards: list[int]) -> int:
    ev = engine.eval7 if len(cards) == 7 else engine.eval5
    return engine.hand_category(ev(cards))


# --- category ladder ------------------------------------------------------

def test_five_card_category_ladder():
    """One hand per category, strictly increasing in both category and score."""
    ladder = [
        hand("2c", "3d", "5h", "9s", "Kc"),  # 0 high card
        hand("2c", "2d", "5h", "9s", "Kc"),  # 1 one pair
        hand("2c", "2d", "5h", "5s", "Kc"),  # 2 two pair
        hand("2c", "2d", "2h", "9s", "Kc"),  # 3 three of a kind
        hand("2c", "3d", "4h", "5s", "6c"),  # 4 straight
        hand("2c", "5c", "7c", "9c", "Kc"),  # 5 flush
        hand("2c", "2d", "2h", "9s", "9c"),  # 6 full house
        hand("2c", "2d", "2h", "2s", "Kc"),  # 7 four of a kind
        hand("2c", "3c", "4c", "5c", "6c"),  # 8 straight flush
    ]
    scores = [engine.eval5(h) for h in ladder]
    assert [engine.hand_category(s) for s in scores] == list(range(9))
    assert scores == sorted(scores)
    assert scores == sorted(set(scores))  # strictly increasing


def test_category_names_cover_zero_to_eight():
    names = [engine.hand_category_name(c) for c in range(9)]
    assert names[0] == "high card"
    assert names[8] == "straight flush"
    assert len(set(names)) == 9


# --- tiebreaks ------------------------------------------------------------

def test_kicker_resolution():
    # Both a pair of aces; the K-Q-3 kickers beat K-Q-2.
    weaker = hand("Ac", "Ad", "Kh", "Qs", "2c")
    stronger = hand("As", "Ah", "Kd", "Qc", "3c")
    assert cat(weaker) == cat(stronger) == 1
    assert engine.eval5(weaker) < engine.eval5(stronger)


def test_flush_compared_by_high_cards():
    lower = hand("2c", "5c", "7c", "9c", "Jc")
    higher = hand("2d", "5d", "7d", "9d", "Qd")
    assert cat(lower) == cat(higher) == 5
    assert engine.eval5(lower) < engine.eval5(higher)


def test_wheel_is_the_lowest_straight():
    wheel = hand("Ac", "2d", "3h", "4s", "5c")
    six_high = hand("2c", "3d", "4h", "5s", "6c")
    assert cat(wheel) == 4  # counts as a straight, not ace-high garbage
    assert engine.eval5(wheel) < engine.eval5(six_high)


# --- 7-card best-of-five --------------------------------------------------

def test_eval7_selects_best_five():
    royal = hand("Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d")
    boat = hand("Ah", "Ac", "Ad", "Kh", "Kc", "2d", "3s")
    assert cat(royal) == 8
    assert cat(boat) == 6
    assert engine.eval7(royal) > engine.eval7(boat)


def test_eval7_ignores_distractor_cards():
    # The made straight on board should be found regardless of the two extra
    # low offsuit cards.
    straight = hand("5c", "6d", "7h", "8s", "9c", "2d", "3s")
    assert cat(straight) == 4


# --- input validation -----------------------------------------------------

def test_eval7_rejects_wrong_length():
    with pytest.raises(ValueError):
        engine.eval7(hand("Ah", "Kh", "Qh"))


def test_parse_card_rejects_garbage():
    with pytest.raises(ValueError):
        engine.parse_card("Zx")
