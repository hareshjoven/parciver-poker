# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""N-player general-sum CFR (pure Python) for the R-MAPSS toy experiments.

The Python sibling of the Rust ``cfr_np.rs``. Toy games (3p Kuhn / Leduc) are
tiny, so orchestration-side Python is fine here and keeps the research code
engine-independent; it also cross-checks the Rust solver. Simultaneous updates
only (the validated path). For n>=3 CFR has NO equilibrium guarantee (Gibson
2013); the average strategy is an approximate low-NashConv profile whose exact
NashConv is always checked by ``parciver.eval.exploitability``.
"""
from __future__ import annotations

from typing import Hashable

from ..games.base import Game


class MultiCFR:
    """Full-traversal simultaneous CFR for a general-sum n-player game."""

    def __init__(self, game: Game, variant: str = "cfr+") -> None:
        self.game = game
        self.n = game.num_players
        self.variant = variant
        self.regret: dict[str, list[float]] = {}
        self.strat_sum: dict[str, list[float]] = {}
        self.t = 0

    @staticmethod
    def _regret_matching(r: list[float]) -> list[float]:
        pos = [x if x > 0 else 0.0 for x in r]
        s = sum(pos)
        if s > 0:
            return [x / s for x in pos]
        u = 1.0 / len(r)
        return [u] * len(r)

    def _traverse(self, state: Hashable, reach: list[float], reach_c: float) -> list[float]:
        g = self.game
        n = self.n
        if g.is_terminal(state):
            return [g.terminal_utility_for(state, i) for i in range(n)]
        if g.is_chance(state):
            val = [0.0] * n
            for a, p in g.chance_outcomes(state):
                cv = self._traverse(g.next_state(state, a), reach, reach_c * p)
                for i in range(n):
                    val[i] += p * cv[i]
            return val

        player = g.current_player(state)
        actions = g.legal_actions(state)
        na = len(actions)
        key = g.infoset_key(state)
        r = self.regret.setdefault(key, [0.0] * na)
        strat = self._regret_matching(r)

        avals: list[list[float]] = []
        val = [0.0] * n
        for a_i, a in enumerate(actions):
            cr = list(reach)
            cr[player] *= strat[a_i]
            cv = self._traverse(g.next_state(state, a), cr, reach_c)
            avals.append(cv)
            for i in range(n):
                val[i] += strat[a_i] * cv[i]

        cf = reach_c
        for j in range(n):
            if j != player:
                cf *= reach[j]
        ssum = self.strat_sum.setdefault(key, [0.0] * na)
        weight = float(self.t) if self.variant in ("cfr+", "linear-cfr") else 1.0
        for a_i in range(na):
            r[a_i] += cf * (avals[a_i][player] - val[player])
            ssum[a_i] += weight * reach[player] * strat[a_i]
        return val

    def run(self, iterations: int) -> None:
        root = self.game.initial_state()
        for _ in range(iterations):
            self.t += 1
            self._traverse(root, [1.0] * self.n, 1.0)
            if self.variant == "cfr+":  # floor regrets at zero
                for r in self.regret.values():
                    for i in range(len(r)):
                        if r[i] < 0.0:
                            r[i] = 0.0

    def average_strategy(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for key, s in self.strat_sum.items():
            tot = sum(s)
            out[key] = [x / tot for x in s] if tot > 0 else [1.0 / len(s)] * len(s)
        return out
