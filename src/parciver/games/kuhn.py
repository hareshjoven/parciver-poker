"""Kuhn poker: the smallest poker game with a known analytic Nash equilibrium.

Rules. Three cards (J=0, Q=1, K=2). Each player antes 1 and receives one private
card. Player 0 acts first and may check or bet 1.
  - check -> player 1 may check (showdown) or bet 1
      - after that bet, player 0 may fold or call (showdown)
  - bet   -> player 1 may fold or call (showdown)
Higher card wins at showdown.

This game exists here purely as a correctness oracle. Its equilibrium is a known
one-parameter family and its game value is exactly -1/18 to player 0, so a solver
that does not reproduce both numbers is broken.
"""
from __future__ import annotations

from typing import Hashable, Sequence

from .base import Game

#: Action encoding shared with Leduc: check/call, bet/raise, fold.
CHECK_CALL, BET, FOLD = "c", "b", "f"

#: Histories at which the hand is over.
_TERMINAL = {"cc", "cbf", "cbc", "bf", "bc"}

#: Payoff to player 0 when nobody folds is +/- this, keyed by history.
_SHOWDOWN_POT = {"cc": 1, "cbc": 2, "bc": 2}

#: Payoff to player 0 when someone folds.
_FOLD_UTILITY = {"cbf": -1, "bf": 1}

#: Exact game value to player 0 under optimal play.
GAME_VALUE = -1.0 / 18.0


class KuhnPoker(Game):
    """Kuhn poker as an extensive-form game.

    State is ``(dealt_cards, history)``. Cards are dealt by two sequential chance
    nodes so the solver exercises the same chance-sampling code paths it will use
    on hold'em.
    """

    num_cards = 3

    def initial_state(self) -> Hashable:
        return ((), "")

    # ---- chance ---------------------------------------------------------
    def is_chance(self, state) -> bool:
        dealt, _ = state
        return len(dealt) < 2

    def chance_outcomes(self, state) -> Sequence[tuple[Hashable, float]]:
        dealt, _ = state
        remaining = [c for c in range(self.num_cards) if c not in dealt]
        p = 1.0 / len(remaining)
        return [(c, p) for c in remaining]

    # ---- terminal -------------------------------------------------------
    def is_terminal(self, state) -> bool:
        dealt, history = state
        return len(dealt) == 2 and history in _TERMINAL

    def terminal_utility(self, state) -> float:
        dealt, history = state
        if history in _FOLD_UTILITY:
            return float(_FOLD_UTILITY[history])
        # Showdown: the higher card takes the pot.
        amount = _SHOWDOWN_POT[history]
        return float(amount) if dealt[0] > dealt[1] else float(-amount)

    # ---- decisions ------------------------------------------------------
    def current_player(self, state) -> int:
        _, history = state
        return len(history) % 2

    def legal_actions(self, state) -> Sequence[Hashable]:
        _, history = state
        # Facing a bet, the only choices are fold or call.
        if history.endswith(BET):
            return (FOLD, CHECK_CALL)
        return (CHECK_CALL, BET)

    def next_state(self, state, action) -> Hashable:
        dealt, history = state
        if self.is_chance(state):
            return (dealt + (action,), history)
        return (dealt, history + action)

    def infoset_key(self, state) -> str:
        dealt, history = state
        # The acting player sees only their own card plus the public history.
        return f"{dealt[self.current_player(state)]}:{history}"
