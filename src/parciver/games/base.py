"""Abstract extensive-form game interface.

Every game used by the solvers implements this. Keeping it minimal and explicit
makes the CFR implementations game-agnostic, so the same solver code that is
validated against Kuhn/Leduc (which have known equilibria) is the code that
later runs on hold'em.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, Sequence


class Game(ABC):
    """A two-player zero-sum extensive-form game with chance nodes.

    States are opaque to the solver; it only moves through them via this API.
    Utilities are always returned from player 0's perspective.
    """

    #: Number of players. The solvers here assume 2 (zero-sum).
    num_players: int = 2

    @abstractmethod
    def initial_state(self) -> Hashable:
        """Return the root state."""

    @abstractmethod
    def is_terminal(self, state: Hashable) -> bool:
        ...

    @abstractmethod
    def terminal_utility(self, state: Hashable) -> float:
        """Payoff to player 0 at a terminal state."""

    def terminal_utility_for(self, state: Hashable, player: int) -> float:
        """Payoff to ``player`` at a terminal state.

        The default is the 2-player zero-sum relation derived from the player-0
        payoff, so every existing 2p game is UNCHANGED. N-player general-sum
        games (the R-MAPSS track) override this. Mirrors the Rust
        ``Game::terminal_utility_for`` default byte-for-byte.
        """
        u = self.terminal_utility(state)
        return u if player == 0 else -u

    @abstractmethod
    def is_chance(self, state: Hashable) -> bool:
        ...

    @abstractmethod
    def chance_outcomes(self, state: Hashable) -> Sequence[tuple[Hashable, float]]:
        """Return (action, probability) pairs. Probabilities must sum to 1."""

    @abstractmethod
    def current_player(self, state: Hashable) -> int:
        """Acting player at a decision node (0 or 1)."""

    @abstractmethod
    def legal_actions(self, state: Hashable) -> Sequence[Hashable]:
        ...

    @abstractmethod
    def next_state(self, state: Hashable, action: Hashable) -> Hashable:
        ...

    @abstractmethod
    def infoset_key(self, state: Hashable) -> str:
        """Information set identifier for the acting player.

        Two states share a key exactly when the acting player cannot tell them
        apart. Getting this wrong is the single most common silent CFR bug.
        """
