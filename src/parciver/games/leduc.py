"""Leduc hold'em: the standard small benchmark for CFR correctness.

Rules (2 players). A 6-card deck: two suits x three ranks (J, Q, K), i.e. two of
each rank. Each player antes 1 and is dealt one private card. There are two
betting rounds; a single public card is revealed before the second round.

Betting. Each round allows check/bet and call/raise with a fixed bet size (2 in
round 1, 4 in round 2) and a cap of two raises per round. At showdown a player
who pairs the public card wins; otherwise the higher private card wins; equal
ranks split.

Leduc is large enough to separate CFR variants sharply -- CFR+ reaches low
exploitability in a few hundred iterations while vanilla CFR needs far more --
yet small enough that exact best-response exploitability is cheap. That makes it
the primary correctness gate for the solvers in this project.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .base import Game

CHECK_CALL, BET_RAISE, FOLD = "c", "r", "f"

#: Ranks 0,1,2 == J,Q,K. Two copies of each in the deck.
_DECK = (0, 0, 1, 1, 2, 2)
_BET_SIZE = (2, 4)   # round 0, round 1
_MAX_RAISES = 2      # per round
_ANTE = 1


@dataclass(frozen=True)
class _State:
    """Immutable Leduc state.

    ``dealt`` holds cards in the order (player0, player1, [public]). ``round`` is
    0 pre-flop / 1 post-flop. ``round_history`` is the action string for the
    current round only, used for betting logic; ``bet_log`` is the full betting
    history across rounds (rounds joined by "/") and is what the infoset key uses
    so the game has PERFECT RECALL. ``committed`` tracks chips in the pot.
    """

    dealt: tuple[int, ...]
    round: int
    round_history: str
    committed: tuple[int, int]
    folded: int | None  # index of the player who folded, or None
    bet_log: str = ""


class LeducHoldem(Game):
    """Leduc hold'em as an extensive-form game."""

    def initial_state(self) -> Hashable:
        return _State(dealt=(), round=0, round_history="", committed=(_ANTE, _ANTE), folded=None, bet_log="")

    # ---- chance ---------------------------------------------------------
    def is_chance(self, state: _State) -> bool:
        # Deal two private cards, then the public card at the start of round 1.
        if len(state.dealt) < 2:
            return True
        if state.round == 1 and len(state.dealt) < 3:
            return True
        return False

    def chance_outcomes(self, state: _State) -> Sequence[tuple[Hashable, float]]:
        remaining = list(_DECK)
        for c in state.dealt:
            remaining.remove(c)
        # Distinct card values with multiplicities collapse to weighted outcomes.
        counts: dict[int, int] = {}
        for c in remaining:
            counts[c] = counts.get(c, 0) + 1
        total = len(remaining)
        return [(c, n / total) for c, n in sorted(counts.items())]

    # ---- round / betting bookkeeping ------------------------------------
    def _round_over(self, state: _State) -> bool:
        """True when the current betting round has closed."""
        h = state.round_history
        if h.endswith(CHECK_CALL):
            # A call always closes the round. A check ("c" by the first actor)
            # only closes it when the second player also checks ("cc").
            if h == CHECK_CALL:
                return False  # single check, opponent still to act
            return True
        if h == CHECK_CALL * 2:
            return True
        return False

    def is_terminal(self, state: _State) -> bool:
        if state.folded is not None:
            return True
        # Terminal after round 1 betting closes.
        return state.round == 1 and len(state.dealt) == 3 and self._round_over(state)

    def terminal_utility(self, state: _State) -> float:
        # Utility to player 0 is chips won (positive) or lost (negative).
        if state.folded is not None:
            winner = 1 - state.folded
            # Winner gains what the loser committed.
            gain = state.committed[state.folded]
            return float(gain) if winner == 0 else float(-gain)
        # Showdown.
        winner = self._showdown_winner(state)
        if winner is None:  # split pot: each gets back what they put in
            return 0.0
        gain = state.committed[1 - winner]
        return float(gain) if winner == 0 else float(-gain)

    def _showdown_winner(self, state: _State) -> int | None:
        p0, p1, board = state.dealt[0], state.dealt[1], state.dealt[2]
        p0_pair = p0 == board
        p1_pair = p1 == board
        if p0_pair != p1_pair:
            return 0 if p0_pair else 1
        # Neither paired (or both, impossible with one board card): high card.
        if p0 == p1:
            return None
        return 0 if p0 > p1 else 1

    # ---- decisions ------------------------------------------------------
    def current_player(self, state: _State) -> int:
        # Post-flop the first player to act is player 0; the number of actions so
        # far in the round determines whose turn it is.
        return len(state.round_history) % 2

    def _num_raises(self, state: _State) -> int:
        return state.round_history.count(BET_RAISE)

    def legal_actions(self, state: _State) -> Sequence[Hashable]:
        facing_bet = state.round_history.endswith(BET_RAISE)
        if facing_bet:
            if self._num_raises(state) < _MAX_RAISES:
                return (FOLD, CHECK_CALL, BET_RAISE)
            return (FOLD, CHECK_CALL)
        # Not facing a bet: check, or open a bet.
        return (CHECK_CALL, BET_RAISE)

    def next_state(self, state: _State, action) -> _State:
        if self.is_chance(state):
            return _State(
                dealt=state.dealt + (action,),
                round=state.round,
                round_history=state.round_history,
                committed=state.committed,
                folded=state.folded,
                bet_log=state.bet_log,
            )

        player = self.current_player(state)
        committed = list(state.committed)
        new_log = state.bet_log + action
        if action == FOLD:
            return _State(state.dealt, state.round, state.round_history + FOLD, tuple(committed), folded=player, bet_log=new_log)

        if action == BET_RAISE:
            # To raise, match the opponent then add one bet increment on top.
            opp = 1 - player
            call_amount = committed[opp] - committed[player]
            committed[player] += call_amount + _BET_SIZE[state.round]
        elif action == CHECK_CALL:
            opp = 1 - player
            committed[player] = committed[opp] if committed[opp] > committed[player] else committed[player]

        new_history = state.round_history + action
        new_state = _State(state.dealt, state.round, new_history, tuple(committed), folded=None, bet_log=new_log)

        # Advance to the next round when betting closes in round 0. The "/"
        # separator keeps the round-0 and round-1 histories distinct in the log.
        if state.round == 0 and self._round_over(new_state):
            return _State(state.dealt, 1, "", tuple(committed), folded=None, bet_log=new_log + "/")
        return new_state

    def infoset_key(self, state: _State) -> str:
        player = self.current_player(state)
        private = state.dealt[player]
        board = state.dealt[2] if len(state.dealt) >= 3 else "?"
        # Perfect recall: the acting player sees their card, the board (once
        # revealed), and the FULL betting history across both rounds. Dropping
        # the round-0 history here would silently create an imperfect-recall
        # game and break CFR's convergence guarantee.
        return f"{private}|{board}|{state.bet_log}"
