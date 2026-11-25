"""Precompiled game tree for fast repeated exact evaluation (R-MAPSS Step 2).

The Step-2 experiment evaluates on-policy value and exact best response THOUSANDS
of times over the same game (once per distinct policy/opponent profile). The
game-driven recursion in ``parciver.eval.exploitability`` re-walks the tree calling
``game.next_state`` / ``infoset_key`` / ``legal_actions`` (dataclass construction +
string formatting) at every node on every call — on 3p Leduc (~58k nodes) that is
~0.5 s per evaluation, which makes the kernel experiment intractable.

This module walks the game ONCE into flat arrays, then answers ``on_policy_value``,
``best_response_value`` and ``best_response_strategy`` with pure array passes and NO
game calls. The algorithm is BYTE-FOR-BYTE the two-pass, per-infoset best response
of ``exploitability.py`` (the per-infoset constraint, not per-state — otherwise a
clairvoyant responder inflates exploitability); it is validated for exact agreement
against that certified implementation in the tests. It assumes a TREE (each state
reached by one path), which holds for Kuhn/Leduc (ordered ``dealt`` tuples).
"""
from __future__ import annotations

from typing import Hashable, Mapping, Sequence

from ..games.base import Game

Strategy = Mapping[str, Sequence[float]]

_TERM, _CHANCE, _DEC = 0, 1, 2


class FastTree:
    """A flat, precompiled extensive-form game tree."""

    def __init__(self, game: Game) -> None:
        self.game = game
        self.n = game.num_players
        self.kind: list[int] = []
        self.tutil: list[tuple] = []            # terminal: per-player utility
        self.cchildren: list[list] = []         # chance: [(child_id, prob)]
        self.dplayer: list[int] = []            # decision: acting player
        self.dkey: list[str] = []               # decision: infoset key
        self.dchildren: list[list] = []         # decision: [child_id per action]
        self.nactions: list[int] = []           # decision: number of actions
        self.root = self._build(game.initial_state())

    def _add(self) -> int:
        self.kind.append(-1)
        self.tutil.append(())
        self.cchildren.append([])
        self.dplayer.append(-1)
        self.dkey.append("")
        self.dchildren.append([])
        self.nactions.append(0)
        return len(self.kind) - 1

    def _build(self, state: Hashable) -> int:
        g = self.game
        idx = self._add()
        if g.is_terminal(state):
            self.kind[idx] = _TERM
            self.tutil[idx] = tuple(g.terminal_utility_for(state, p) for p in range(self.n))
            return idx
        if g.is_chance(state):
            self.kind[idx] = _CHANCE
            self.cchildren[idx] = [(self._build(g.next_state(state, a)), float(p))
                                   for a, p in g.chance_outcomes(state)]
            return idx
        self.kind[idx] = _DEC
        self.dplayer[idx] = g.current_player(state)
        self.dkey[idx] = g.infoset_key(state)
        acts = g.legal_actions(state)
        self.nactions[idx] = len(acts)
        self.dchildren[idx] = [self._build(g.next_state(state, a)) for a in acts]
        return idx

    # -- reverse pre-order (children before parents) for bottom-up sweeps --
    def _rev(self) -> range:
        return range(len(self.kind) - 1, -1, -1)

    def on_policy_value(self, strategy: Strategy, player: int) -> float:
        """EV to ``player`` when ALL players follow ``strategy`` (uniform default)."""
        v = [0.0] * len(self.kind)
        for i in self._rev():
            k = self.kind[i]
            if k == _TERM:
                v[i] = self.tutil[i][player]
            elif k == _CHANCE:
                v[i] = sum(p * v[c] for c, p in self.cchildren[i])
            else:
                na = self.nactions[i]
                probs = strategy.get(self.dkey[i])
                ch = self.dchildren[i]
                if probs is None:
                    inv = 1.0 / na
                    v[i] = sum(v[c] for c in ch) * inv
                else:
                    v[i] = sum(probs[a] * v[ch[a]] for a in range(na))
        return v[self.root]

    # -- exact best response (two-pass, per-infoset) --
    def _br_common(self, strategy: Strategy, brp: int):
        infoset_states: dict[str, list[tuple[int, float]]] = {}

        def enumerate_states(node: int, opp_reach: float) -> None:
            if opp_reach == 0.0:
                return
            k = self.kind[node]
            if k == _TERM:
                return
            if k == _CHANCE:
                for c, p in self.cchildren[node]:
                    enumerate_states(c, opp_reach * p)
                return
            ch = self.dchildren[node]
            if self.dplayer[node] == brp:
                infoset_states.setdefault(self.dkey[node], []).append((node, opp_reach))
                for c in ch:
                    enumerate_states(c, opp_reach)
            else:
                na = self.nactions[node]
                probs = strategy.get(self.dkey[node])
                if probs is None:
                    inv = 1.0 / na
                    for c in ch:
                        enumerate_states(c, opp_reach * inv)
                else:
                    for a in range(na):
                        if probs[a] > 0.0:
                            enumerate_states(ch[a], opp_reach * probs[a])

        enumerate_states(self.root, 1.0)

        best_action: dict[str, int] = {}

        def chosen_action(key: str) -> int:
            if key in best_action:
                return best_action[key]
            states = infoset_states[key]
            na = self.nactions[states[0][0]]
            totals = [0.0] * na
            for node, opp_reach in states:
                ch = self.dchildren[node]
                for a in range(na):
                    totals[a] += opp_reach * subtree_value(ch[a])
            best = max(range(na), key=lambda a: totals[a])
            best_action[key] = best
            return best

        def subtree_value(node: int) -> float:
            k = self.kind[node]
            if k == _TERM:
                return self.tutil[node][brp]
            if k == _CHANCE:
                return sum(p * subtree_value(c) for c, p in self.cchildren[node])
            ch = self.dchildren[node]
            if self.dplayer[node] == brp:
                return subtree_value(ch[chosen_action(self.dkey[node])])
            na = self.nactions[node]
            probs = strategy.get(self.dkey[node])
            if probs is None:
                inv = 1.0 / na
                return sum(subtree_value(c) for c in ch) * inv
            return sum(probs[a] * subtree_value(ch[a]) for a in range(na) if probs[a] > 0.0)

        return infoset_states, best_action, chosen_action, subtree_value

    def best_response_value(self, strategy: Strategy, brp: int) -> float:
        _, _, _, subtree_value = self._br_common(strategy, brp)
        return subtree_value(self.root)

    def best_response_strategy(self, strategy: Strategy, brp: int) -> dict[str, list[float]]:
        infoset_states, _, chosen_action, subtree_value = self._br_common(strategy, brp)
        subtree_value(self.root)  # populate chosen_action over all reached infosets
        out: dict[str, list[float]] = {}
        for key, states in infoset_states.items():
            na = self.nactions[states[0][0]]
            oh = [0.0] * na
            oh[chosen_action(key)] = 1.0
            out[key] = oh
        return out

    # -- fast simultaneous CFR (mirrors parciver.solvers.cfr_multi on the flat tree) --
    def solve_cfr(self, iters: int, variant: str = "cfr+") -> dict[str, list[float]]:
        """Simultaneous n-player CFR / CFR+ on the flat tree; returns the average
        strategy. Byte-for-byte the recursion of ``MultiCFR`` (validated in tests),
        just ~10x faster (no game calls). No >=3p equilibrium guarantee (R5) — the
        exact NashConv is always checked separately."""
        n = self.n
        regret: dict[str, list[float]] = {}
        strat_sum: dict[str, list[float]] = {}

        def rm(r: list[float]) -> list[float]:
            pos = [x if x > 0 else 0.0 for x in r]
            s = sum(pos)
            if s > 0:
                return [x / s for x in pos]
            u = 1.0 / len(r)
            return [u] * len(r)

        state = {"t": 0}

        def traverse(node: int, reach: list[float], reach_c: float) -> list[float]:
            k = self.kind[node]
            if k == _TERM:
                return list(self.tutil[node])
            if k == _CHANCE:
                val = [0.0] * n
                for c, p in self.cchildren[node]:
                    cv = traverse(c, reach, reach_c * p)
                    for i in range(n):
                        val[i] += p * cv[i]
                return val
            player = self.dplayer[node]
            key = self.dkey[node]
            na = self.nactions[node]
            ch = self.dchildren[node]
            r = regret.setdefault(key, [0.0] * na)
            strat = rm(r)
            avals: list[list[float]] = []
            val = [0.0] * n
            for a in range(na):
                cr = list(reach)
                cr[player] *= strat[a]
                cv = traverse(ch[a], cr, reach_c)
                avals.append(cv)
                for i in range(n):
                    val[i] += strat[a] * cv[i]
            cf = reach_c
            for j in range(n):
                if j != player:
                    cf *= reach[j]
            ssum = strat_sum.setdefault(key, [0.0] * na)
            weight = float(state["t"]) if variant in ("cfr+", "linear-cfr") else 1.0
            for a in range(na):
                r[a] += cf * (avals[a][player] - val[player])
                ssum[a] += weight * reach[player] * strat[a]
            return val

        for _ in range(iters):
            state["t"] += 1
            traverse(self.root, [1.0] * n, 1.0)
            if variant == "cfr+":
                for r in regret.values():
                    for i in range(len(r)):
                        if r[i] < 0.0:
                            r[i] = 0.0

        out: dict[str, list[float]] = {}
        for key, s in strat_sum.items():
            tot = sum(s)
            out[key] = [x / tot for x in s] if tot > 0 else [1.0 / len(s)] * len(s)
        return out
