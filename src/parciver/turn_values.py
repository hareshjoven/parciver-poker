# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Turn-start values: the training data of the depth-limited flop search's
value net (FLOPSEARCH.md, D1).

A situation is the public state at the start of the turn's betting (4-card
board, each player's stake and remaining stack, who acts first) plus both
players' ranges over the 1,128 holdings off the board. Its values are both
players' counterfactual values per holding at the root of the turn subgame
the agent's turn search solves (same tree and menus as `subgame.rs`:
`turn_menu`, `turn_river_menu`), under the average strategy of the fused GPU
solver after a fixed number of DCFR iterations.

* `Recorder` collects real situations: it is the external subgame solver of
  an agent with turn search, so it sees exactly the ranges the search starts
  from (`subgame::ranges_at` at the round's start). It answers every job with
  uniform strategies, a valid answer that the search caches, so the turn and
  river play after the recorded state is not the agent's (irrelevant: the
  state is recorded before it, and deals are independent).
* `random_range` is DeepStack's range generator, for coverage.
* Values are stored per unit of the opponent's range mass, in pots:
  `v_p[h] / (M_q * pot)`. Zero-sum: with each range scaled to mass 1,
  `r0 . v0 + r1 . v1 = 0`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gpu_search import Hands, Menu, Spec, Tree

TURN_MENU = Menu([0.5, 1.0], [1.0], 2)  # subgame.rs `turn_menu`
TURN_RIVER_MENU = Menu([0.75], [], 1)  # subgame.rs `turn_river_menu`
HOLDINGS = 1128  # two-card holdings off a 4-card board


@dataclass
class Situation:
    board: tuple  # 4 card indices (rank * 4 + suit)
    half: int  # each player's stake in the pot
    stack: int  # each player's remaining stack
    first: int  # the seat that acts first on the turn
    bb: int  # the minimum bet
    ranges: np.ndarray  # [2, HOLDINGS] over `Hands(board).combos`

    @property
    def pot(self) -> int:
        return 2 * self.half

    def spec(self) -> Spec:
        return Spec((self.half, self.half), (self.stack, self.stack), self.first, self.bb, TURN_MENU, TURN_RIVER_MENU)


def _uniform_answer(job: dict) -> list:
    start, n = job["child_start"], len(job["ranges"][0])
    node = {s: i for i, s in enumerate(job["slot"]) if s >= 0}
    out = []
    for w in job["want"]:
        k = start[node[w] + 1] - start[node[w]]
        out.append([1.0 / k] * (k * n))
    return out


def root_of(job: dict, stack_total: int, bb: int) -> Situation:
    """The situation at the root of a turn subgame job (`FlatSubgame` dict).
    The root is the round's start (the search solves the round from its
    first action); its stakes are those of the check-down showdown, the
    smallest terminal pot."""
    kind = np.asarray(job["kind"])
    stake = np.asarray(job["stake"], dtype=np.int64).reshape(-1, 2)[kind == 0]
    lo = stake[np.argmin(stake.sum(1))]
    if lo[0] != lo[1]:
        raise ValueError(f"unequal stakes at the turn's start: {lo}")
    half = int(lo[0])
    return Situation(tuple(int(c) for c in job["board"]), half, stack_total - half, int(job["player"][0]), bb,
                     np.asarray(job["ranges"], dtype=np.float64))


class Recorder:
    """`PlayAgent.with_search(from_turn=True, solver=Recorder(...))`: records
    each distinct turn-round root the agent's search reaches. `stack_total`:
    the table's starting stack (stacks are equal at the turn's start)."""

    def __init__(self, stack_total: int, bb: int):
        self.stack_total, self.bb = stack_total, bb
        self.situations: list[Situation] = []
        self.first_decisions: list[dict] = []  # jobs with no earlier action this round (tests)
        self.keep_jobs = False
        self._seen: set = set()

    def __call__(self, job: dict) -> list:
        if len(job["board"]) == 4:
            sit = root_of(job, self.stack_total, self.bb)
            key = (sit.board, sit.half, sit.first, sit.ranges.tobytes())
            if key not in self._seen:
                self._seen.add(key)
                self.situations.append(sit)
            if self.keep_jobs and len(job["want"]) == 1:
                self.first_decisions.append(dict(job))
        return _uniform_answer(job)


# ---------------------------------------------------------------- ranges


def strength_order(hands: Hands) -> np.ndarray:
    """Holdings weakest first by their mean showdown rank (tied holdings share
    their average rank) over the river cards that leave them alive; equal
    means by index."""
    ranks = np.zeros(hands.n)
    for v in range(hands.views):
        live = hands.alive[v] > 0
        s = hands.strength[v][live]
        ss = np.sort(s)
        mid = (np.searchsorted(ss, s, "left") + np.searchsorted(ss, s, "right") - 1) / 2
        ranks[live] += mid / max(live.sum() - 1, 1)
    ranks /= hands.alive.sum(0)
    return np.argsort(ranks, kind="stable")


def random_range(order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DeepStack's generator: the mass 1 over holdings in `order` split
    recursively, a uniform share of each part to its weaker half."""
    out = np.zeros(len(order))

    def split(lo: int, hi: int, p: float):
        if hi - lo == 1:
            out[order[lo]] = p
            return
        mid = (lo + hi) // 2
        q = p * rng.random()
        split(lo, mid, q)
        split(mid, hi, p - q)

    split(0, len(order), 1.0)
    return out


# ---------------------------------------------------------------- solving


def solve(sit: Situation, hands: Hands, iters: int, dtype=None, tree: Tree | None = None, plan=None,
          gamma: int = 2, plus: bool = False, terminals64: bool = False) -> dict:
    """`iters` DCFR iterations (strategy-sum exponent `gamma`; DCFR+ with
    `plus`) of the turn subgame on the fused GPU solver.
    Returns `values` [2, n] (pots per unit of opponent mass), `game` (player
    0's value per dealt pair, pots) and `exploitability` (both best responses'
    gains per dealt pair, pots: search.rs `Solver::exploitability` / pot).
    With `plan` given, `hands` and `tree` are not used (may be None).
    `terminals64`: FusedSolver's (KERNELS Q4 for float32 solves)."""
    import torch

    from .gpu_fused import FusedSolver

    if plan is None:
        tree = tree or Tree(sit.spec(), hands)
    s = FusedSolver(hands, tree, list(sit.ranges), dtype=dtype or torch.float32, plan=plan, gamma=gamma, plus=plus,
                    terminals64=terminals64)
    for _ in range(iters):
        s.step()
    values, game = values_of(s, sit)
    return dict(values=values, game=game, exploitability=s.exploitability() / sit.pot)


def values_of(s, sit: Situation) -> tuple[np.ndarray, float]:
    """A `FusedSolver`'s root values under its average strategy, per holding
    in pots per unit of opponent mass, and player 0's value per dealt pair."""
    values = np.zeros((2, s.n))
    for p in (0, 1):
        g = s.value(p)
        if p == 0:
            game = g / sit.pot
        values[p] = s.val[0].double().cpu().numpy() / (float(np.sum(sit.ranges[1 - p])) * sit.pot)
    return values, game


def checkdown_values(sit: Situation, hands: Hands) -> np.ndarray:
    """Both players' values if nobody bets again (showdown equity x pot, net
    of the stake), in the units of `solve`'s values: the G2 baseline."""
    out = np.zeros((2, hands.n))
    for p in (0, 1):
        rq = np.asarray(sit.ranges[1 - p], dtype=np.float64)
        acc = np.zeros(hands.n)
        for v in range(hands.views):
            acc += hands.showdown_matrix(v) @ (rq * hands.alive[v])
        out[p] = acc * sit.half / (hands.views - 4) / (rq.sum() * sit.pot)
    return out


# ---------------------------------------------------------------- suit symmetry


def combos(board) -> list:
    """The holdings off `board` in the Rust (and `Hands`) order."""
    used = {int(c) for c in board}
    return [(a, b) for a in range(52) for b in range(a + 1, 52) if a not in used and b not in used]


def permute_cards(cards, perm) -> list:
    """Cards with suits relabelled by `perm` (suit s -> perm[s])."""
    return [(int(c) & ~3) | int(perm[int(c) & 3]) for c in cards]


def holding_map(board, perm) -> tuple[list, np.ndarray]:
    """The permuted board and, for each holding of the permuted board, the
    index of its preimage among the original board's holdings (so permuted
    per-holding data is `data[..., idx]`)."""
    new_board = permute_cards(board, perm)
    pre = {tuple(sorted(permute_cards(c, perm))): i for i, c in enumerate(combos(board))}
    return new_board, np.array([pre[c] for c in combos(new_board)])
