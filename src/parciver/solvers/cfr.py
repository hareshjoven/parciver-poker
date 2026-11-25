# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Tabular CFR family: vanilla CFR, CFR+, Linear CFR and Discounted CFR.

All four are the same traversal differing only in how regrets and the average
strategy are weighted over time, so they are expressed here as one parameterised
solver. Keeping them in one place makes the variant comparison apples to apples,
and whichever variant wins on Leduc is the one worth porting to the fast path.

Discount parameters follow Brown and Sandholm, "Solving Imperfect-Information
Games via Discounted Regret Minimization" (AAAI 2019):
    alpha  exponent applied to POSITIVE accumulated regret
    beta   exponent applied to NEGATIVE accumulated regret
    gamma  exponent applied to the average-strategy weight
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from ..games.base import Game


@dataclass(frozen=True)
class Variant:
    """Weighting scheme for a CFR variant."""

    name: str
    alpha: float | None = None   # None => no discounting of positive regret
    beta: float | None = None
    gamma: float | None = None
    floor_regrets_at_zero: bool = False  # the "+" in CFR+
    linear_averaging: bool = False       # weight the average strategy by t


#: Vanilla CFR. Regrets accumulate unweighted and the average strategy is
#: uniform over iterations. Correct, but the slowest to converge.
VANILLA = Variant("cfr")

#: CFR+. Negative regrets are clamped to zero each update and the average
#: strategy is weighted linearly in t. Very strong in practice on small games.
CFR_PLUS = Variant("cfr+", floor_regrets_at_zero=True, linear_averaging=True)

#: Linear CFR. Both regrets and strategy contributions are weighted by t.
LINEAR = Variant("linear-cfr", alpha=1.0, beta=1.0, gamma=1.0)

#: Discounted CFR with the parameters the paper reports as best overall.
DCFR = Variant("dcfr", alpha=1.5, beta=0.0, gamma=2.0)

VARIANTS = {v.name: v for v in (VANILLA, CFR_PLUS, LINEAR, DCFR)}


@dataclass
class _Node:
    """Per-infoset accumulators."""

    regret_sum: np.ndarray
    strategy_sum: np.ndarray
    player: int  # which player owns this infoset


@dataclass
class CFRSolver:
    """Full-traversal (non-sampled) CFR over a small extensive-form game."""

    game: Game
    variant: Variant = DCFR
    #: Update one player per iteration rather than both. When enabled each player
    #: runs on its own clock (its accumulators are touched only on its own
    #: iterations -- see ``_apply_discounts``). Alternating clearly helps vanilla
    #: CFR, but in this implementation it currently UNDERPERFORMS for the
    #: discounted variants on Leduc (validated by the convergence sweep), so the
    #: default is simultaneous updates, under which DCFR is the strongest and all
    #: variants converge with the correct ordering. Revisit before the Rust port.
    alternating: bool = False
    nodes: dict[str, _Node] = field(default_factory=dict)
    iterations: int = 0
    #: Per-player update counts, so each player's discount schedule advances at
    #: its own rate rather than the global iteration rate.
    player_iters: list[int] = field(default_factory=lambda: [0, 0])

    # ---- strategy handling ------------------------------------------------
    def _node(self, key: str, num_actions: int, player: int) -> _Node:
        node = self.nodes.get(key)
        if node is None:
            node = _Node(np.zeros(num_actions), np.zeros(num_actions), player)
            self.nodes[key] = node
        return node

    @staticmethod
    def _regret_matching(regret_sum: np.ndarray) -> np.ndarray:
        """Convert accumulated regret into a strategy.

        Only positive regret contributes. If none is positive the strategy is
        uniform, which keeps the iterate well defined early on.
        """
        positive = np.maximum(regret_sum, 0.0)
        total = positive.sum()
        if total > 0.0:
            return positive / total
        return np.full(len(regret_sum), 1.0 / len(regret_sum))

    def average_strategy(self) -> dict[str, list[float]]:
        """The converged object: the AVERAGE strategy, not the current one.

        The CFR guarantee is on the time-average iterate. Reading out the
        current strategy instead is a classic and badly wrong shortcut.
        """
        out: dict[str, list[float]] = {}
        for key, node in self.nodes.items():
            total = node.strategy_sum.sum()
            if total > 0.0:
                out[key] = (node.strategy_sum / total).tolist()
            else:
                n = len(node.strategy_sum)
                out[key] = [1.0 / n] * n
        return out

    # ---- traversal --------------------------------------------------------
    def _traverse(
        self, state: Hashable, reach0: float, reach1: float, reach_chance: float, update_player: int | None
    ) -> float:
        """Return the expected value to player 0 below ``state``.

        ``update_player`` selects whose regrets and average strategy are updated
        on this pass. ``None`` updates both (simultaneous CFR); an int updates
        only that player (alternating CFR, as CFR+ and DCFR assume).
        """
        game = self.game
        if game.is_terminal(state):
            return game.terminal_utility(state)

        if game.is_chance(state):
            value = 0.0
            for action, prob in game.chance_outcomes(state):
                value += prob * self._traverse(
                    game.next_state(state, action), reach0, reach1, reach_chance * prob, update_player
                )
            return value

        player = game.current_player(state)
        actions = game.legal_actions(state)
        node = self._node(game.infoset_key(state), len(actions), player)
        strategy = self._regret_matching(node.regret_sum)

        action_values = np.empty(len(actions))
        value = 0.0
        for i, action in enumerate(actions):
            child = game.next_state(state, action)
            if player == 0:
                child_value = self._traverse(child, reach0 * strategy[i], reach1, reach_chance, update_player)
            else:
                child_value = self._traverse(child, reach0, reach1 * strategy[i], reach_chance, update_player)
            action_values[i] = child_value
            value += strategy[i] * child_value

        if update_player is None or player == update_player:
            # Counterfactual reach excludes the acting player's own contribution.
            # Including it here is the most common CFR implementation bug.
            cf_reach = (reach1 if player == 0 else reach0) * reach_chance
            # Utilities are stored from player 0's view, so player 1 maximises -u.
            sign = 1.0 if player == 0 else -1.0
            node.regret_sum += cf_reach * sign * (action_values - value)

            own_reach = reach0 if player == 0 else reach1
            node.strategy_sum += self._strategy_weight(self._clock[player]) * own_reach * strategy
        return value

    def _strategy_weight(self, t: int) -> float:
        """Weight applied to this iteration's contribution to the average strategy.

        The average-strategy guarantee wants later iterations (closer to the
        equilibrium) to dominate, so the contribution at update ``t`` is weighted
        by an INCREASING function t**gamma:
            vanilla -> gamma 0 (uniform averaging)
            CFR+/Linear -> gamma 1 (linear averaging)
            DCFR -> gamma 2
        Using a near-constant weight here silently degrades every variant to
        plain uniform averaging, erasing the whole point of CFR+/DCFR.
        ``t`` is the acting player's own update count so alternating and
        simultaneous updates weight consistently.
        """
        if self.variant.gamma is not None:
            gamma = self.variant.gamma
        elif self.variant.linear_averaging:
            gamma = 1.0
        else:
            gamma = 0.0
        return float(t) ** gamma

    def _apply_discounts(self, update_player: int | None) -> None:
        """Re-weight accumulated regret for the player(s) just updated.

        Crucially this touches only the nodes owned by the player who was
        updated this pass, advancing that player's own discount clock. Decaying
        every node on every global iteration (the naive approach) discounts a
        player's regrets twice as often as they are updated under alternating
        updates, which visibly slows CFR+/DCFR below vanilla.
        """
        variant = self.variant
        players = (0, 1) if update_player is None else (update_player,)
        for p in players:
            owned = [node for node in self.nodes.values() if node.player == p]
            if variant.floor_regrets_at_zero:
                for node in owned:
                    np.maximum(node.regret_sum, 0.0, out=node.regret_sum)
                continue
            if variant.alpha is None and variant.beta is None:
                continue
            t = float(self._clock[p])
            scale_pos = (t ** variant.alpha) / (t ** variant.alpha + 1.0) if variant.alpha is not None else 1.0
            scale_neg = (t ** variant.beta) / (t ** variant.beta + 1.0) if variant.beta is not None else 1.0
            for node in owned:
                regret = node.regret_sum
                np.multiply(regret, np.where(regret > 0.0, scale_pos, scale_neg), out=regret)

    @property
    def _clock(self) -> list[int]:
        """Per-player update count used by the weighting and discount schedules."""
        return self.player_iters

    def run(self, iterations: int) -> None:
        """Run ``iterations`` CFR passes.

        Alternating updates advance one player per iteration on its own clock;
        simultaneous updates advance both. The per-player clock is what the DCFR
        discount schedule and CFR+ linear averaging are stated against.
        """
        for _ in range(iterations):
            self.iterations += 1
            if self.alternating:
                update_player = self.iterations % self.game.num_players
                self.player_iters[update_player] += 1
            else:
                update_player = None
                for p in range(self.game.num_players):
                    self.player_iters[p] += 1
            self._traverse(self.game.initial_state(), 1.0, 1.0, 1.0, update_player)
            self._apply_discounts(update_player)
