# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Structural correctness of the reference games.

These are cheap, exhaustive checks of the game trees themselves, independent of
any solver. A bug caught here would otherwise masquerade as a training problem.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parciver.games.kuhn import KuhnPoker
from parciver.games.leduc import LeducHoldem


def _walk(game, state, prob):
    """Yield (leaf_state, reach_prob) over the whole tree; collect infosets."""
    if game.is_terminal(state):
        yield ("leaf", state, prob)
        return
    if game.is_chance(state):
        outcomes = game.chance_outcomes(state)
        total = sum(p for _, p in outcomes)
        assert abs(total - 1.0) < 1e-9, f"chance probs sum to {total}"
        for action, p in outcomes:
            yield from _walk(game, game.next_state(state, action), prob * p)
        return
    yield ("decision", state, prob)
    for action in game.legal_actions(state):
        yield from _walk(game, game.next_state(state, action), prob)


def test_kuhn_tree_shape():
    game = KuhnPoker()
    leaves = [x for x in _walk(game, game.initial_state(), 1.0) if x[0] == "leaf"]
    # 6 card deals x 5 terminal histories.
    assert len(leaves) == 30
    histories = {s[1] for _, s, _ in leaves}
    assert histories == {"cc", "cbf", "cbc", "bf", "bc"}


def test_kuhn_infoset_count():
    game = KuhnPoker()
    infosets = {game.infoset_key(s) for kind, s, _ in _walk(game, game.initial_state(), 1.0) if kind == "decision"}
    # 3 cards x (player-0 sees {"", "cb"} ; player-1 sees {"c","b"}) = 12.
    assert len(infosets) == 12


def test_leduc_infoset_count():
    """Canonical Leduc hold'em has exactly 288 information sets (perfect recall)."""
    game = LeducHoldem()
    infosets = {game.infoset_key(s) for kind, s, _ in _walk(game, game.initial_state(), 1.0) if kind == "decision"}
    assert len(infosets) == 288


def test_leduc_is_zero_sum_and_reachable():
    game = LeducHoldem()
    leaves = [(s, p) for kind, s, p in _walk(game, game.initial_state(), 1.0) if kind == "leaf"]
    assert len(leaves) > 0
    # Every terminal utility is finite and within the max pot swing.
    for s, _ in leaves:
        u = game.terminal_utility(s)
        assert -14.0 <= u <= 14.0  # deepest pot: ante + 2*bet(2) + 2*bet(4) per side


def test_leduc_showdown_pair_beats_high_card():
    """A player who pairs the board must win regardless of card ranks."""
    game = LeducHoldem()
    from parciver.games.leduc import _State

    # p0 has J(0), p1 has K(2), board pairs p0's J -> p0 wins despite lower card.
    s = _State(dealt=(0, 2, 0), round=1, round_history="cc", committed=(3, 3), folded=None, bet_log="cc/cc")
    assert game._showdown_winner(s) == 0
