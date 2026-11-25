"""N-player Kuhn poker — the multiplayer oracle for the R-MAPSS track.

The independent Python mirror of the Rust ``kuhn_n::KuhnN`` (RMAPSS.md §33,
Step 0.2). It exists so ``parciver.eval.exploitability`` can score a Rust-solved
n-player strategy the same way it certifies the 2p engines — the project's
signature Rust-vs-Python parity gate.

Rules (standard n-player / OpenSpiel Kuhn): a deck of ``n+1`` cards ranked
``0..n``; each of ``n`` players is dealt one private card (one card undealt);
everyone antes 1; ONE betting round, one bet size of 1, NO raises. Seats act in
order; facing no bet a seat checks ``'c'`` or bets ``'b'``; once a seat bets the
other ``n-1`` seats respond (fold ``'f'`` / call ``'c'``) in wrap order starting
after the bettor, then showdown among the non-folded (highest card wins the pot).

Byte-for-byte with the Rust engine: the acting seat is ``len(history) % n`` in
both phases, the legal-action order is ``(c, b)`` pre-bet / ``(f, c)`` facing a
bet, and the infoset key is ``f"{card}:{history}"``. With ``n=2`` it reduces to
``parciver.games.kuhn`` exactly.
"""
from __future__ import annotations

from typing import Hashable, Sequence

from .base import Game

CHECK_CALL, BET, FOLD = "c", "b", "f"


class KuhnPokerN(Game):
    """n-player Kuhn on a deck of ``n+1`` cards."""

    def __init__(self, num_players: int = 3) -> None:
        if num_players < 2:
            raise ValueError("Kuhn needs at least 2 players")
        self.num_players = num_players
        self.num_cards = num_players + 1

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _bet_pos(history: str) -> int | None:
        i = history.find(BET)
        return i if i >= 0 else None

    # ---- chance ---------------------------------------------------------
    def initial_state(self) -> Hashable:
        return ((), "")

    def is_chance(self, state) -> bool:
        dealt, _ = state
        return len(dealt) < self.num_players

    def chance_outcomes(self, state) -> Sequence[tuple[Hashable, float]]:
        dealt, _ = state
        remaining = [c for c in range(self.num_cards) if c not in dealt]
        p = 1.0 / len(remaining)
        return [(c, p) for c in remaining]

    # ---- terminal -------------------------------------------------------
    def is_terminal(self, state) -> bool:
        dealt, history = state
        if len(dealt) != self.num_players:
            return False
        p = self._bet_pos(history)
        if p is None:
            return len(history) == self.num_players  # all checked
        return len(history) == p + self.num_players  # bettor + n-1 responders

    def terminal_utility(self, state) -> float:
        return self.terminal_utility_for(state, 0)

    def terminal_utility_for(self, state, player: int) -> float:
        dealt, history = state
        n = self.num_players
        contrib = [1.0] * n  # antes
        folded = [False] * n
        p = self._bet_pos(history)
        if p is not None:
            contrib[p] += 1.0  # the bettor
            for k, tok in enumerate(history[p + 1:]):
                seat = (p + 1 + k) % n
                if tok == CHECK_CALL:
                    contrib[seat] += 1.0  # called
                else:
                    folded[seat] = True  # 'f'
        pot = sum(contrib)
        winner, best = 0, -1
        for i in range(n):
            if not folded[i] and dealt[i] > best:
                best = dealt[i]
                winner = i
        u = -contrib[player]
        if player == winner:
            u += pot
        return u

    # ---- decisions ------------------------------------------------------
    def current_player(self, state) -> int:
        _, history = state
        return len(history) % self.num_players

    def legal_actions(self, state) -> Sequence[Hashable]:
        _, history = state
        if self._bet_pos(history) is not None:
            return (FOLD, CHECK_CALL)
        return (CHECK_CALL, BET)

    def next_state(self, state, action) -> Hashable:
        dealt, history = state
        if self.is_chance(state):
            return (dealt + (action,), history)
        return (dealt, history + action)

    def infoset_key(self, state) -> str:
        dealt, history = state
        return f"{dealt[self.current_player(state)]}:{history}"
