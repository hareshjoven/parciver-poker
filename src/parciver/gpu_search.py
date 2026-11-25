# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""GPU subgame solver (prototype): the real-time search's range-vs-range DCFR
(engine/src/search.rs) as tensor operations, to measure whether a GPU makes
turn (and later flop) solves fast enough to matter.

Same game, same algorithm as the Rust solver: every two-card holding off the
board is a hand in both ranges, card removal is exact, Discounted CFR with
alpha=1.5, beta=0, gamma=2 and alternating updates. What changes is the
organization:

* Showdowns and folds are dense matrix products. Per complete board `v`,
  `W_v[h, j] = sign(strength_h - strength_j)` over live, card-disjoint pairs,
  so every showdown terminal's values are `stake * W_v @ reach` (one batched
  matmul per iteration); folds use the card-disjointness matrix `C`.
* The tree is processed level by level: one pass down the depths carries both
  players' reaches to every node, all terminals are valued at once, one pass
  up combines values and updates regrets. Each level is a handful of batched
  gathers/scatters over all of its nodes, including the 48 copies of a river
  subtree below a turn chance node.

`RefSolver` is the oracle: the Rust solver's recursion, line for line, in
NumPy float64. `GpuSolver` must reproduce it (to rounding in float64) before
any of its speed counts. Trees and inputs mirror the Rust tests
(`board`, `range_`, `spec`, `small_turn_spec`, `turn_spec`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ALPHA = 1.5
GAMMA = 2.0
MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------ inputs


def splitmix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def unit(x: int) -> float:
    return (x >> 11) / float(1 << 53)


def board(seed: int) -> list[int]:
    """search.rs `tests_support::board`: 5 cards from a seeded shuffle."""
    deck = list(range(52))
    s = seed
    for i in range(5):
        s = splitmix(s)
        j = i + s % (52 - i)
        deck[i], deck[j] = deck[j], deck[i]
    return deck[:5]


def range_(n: int, seed: int) -> np.ndarray:
    """search.rs `tests_support::range`."""
    return np.array([unit(splitmix(seed ^ i)) for i in range(n)])


# ------------------------------------------------------------------ cards


class Hands:
    """Holdings off `board_cards` (4 = turn solve, 5 = river) in the Rust
    order, one view per complete board, and the matrices the solvers use."""

    def __init__(self, board_cards: list[int]):
        import parciver_engine as pe

        self.board = list(board_cards)
        used = 0
        for c in self.board:
            used |= 1 << c
        self.combos = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)
                                if not used >> a & 1 and not used >> b & 1], dtype=np.int64)
        n = len(self.combos)
        if len(self.board) == 5:
            cards = [None]
        else:
            cards = [c for c in range(52) if not used >> c & 1]
        self.view_cards = cards
        self.alive = np.ones((len(cards), n))
        self.strength = np.zeros((len(cards), n))
        for v, card in enumerate(cards):
            full = self.board + ([] if card is None else [card])
            for h, (x, y) in enumerate(self.combos):
                if card is not None and card in (x, y):
                    self.alive[v, h] = 0.0
                else:
                    self.strength[v, h] = pe.eval7([int(x), int(y), *full])

    @property
    def n(self) -> int:
        return len(self.combos)

    @property
    def views(self) -> int:
        return len(self.view_cards)

    @classmethod
    def from_arrays(cls, board_cards, strength, alive) -> "Hands":
        """Hands with showdown strengths and liveness given per view (as the
        Rust search computes them) instead of evaluated here."""
        h = cls.__new__(cls)
        h.board = [int(c) for c in board_cards]
        used = 0
        for c in h.board:
            used |= 1 << c
        h.combos = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)
                             if not used >> a & 1 and not used >> b & 1], dtype=np.int64)
        h.view_cards = [None] if len(h.board) == 5 else [c for c in range(52) if not used >> c & 1]
        h.strength = np.asarray(strength, dtype=np.float64).reshape(len(h.view_cards), len(h.combos))
        h.alive = np.asarray(alive, dtype=np.float64).reshape(h.strength.shape)
        return h

    @property
    def compat(self) -> np.ndarray:
        """Card-disjoint pairs (a holding is not disjoint from itself)."""
        if getattr(self, "_compat", None) is None:
            a, b = self.combos[:, 0], self.combos[:, 1]
            self._compat = ((a[:, None] != a[None, :]) & (a[:, None] != b[None, :])
                            & (b[:, None] != a[None, :]) & (b[:, None] != b[None, :])).astype(np.float64)
        return self._compat

    def compatible_mass(self, reach) -> np.ndarray:
        """`compat @ reach` in O(n): the total minus the holdings sharing a card."""
        reach = np.asarray(reach, dtype=np.float64)
        card = np.zeros(52)
        np.add.at(card, self.combos[:, 0], reach)
        np.add.at(card, self.combos[:, 1], reach)
        return reach.sum() - card[self.combos[:, 0]] - card[self.combos[:, 1]] + reach

    def showdown_matrix(self, v: int) -> np.ndarray:
        s, live = self.strength[v], self.alive[v]
        return np.sign(s[:, None] - s[None, :]) * self.compat * live[:, None] * live[None, :]


# ------------------------------------------------------------------ tree (port of search.rs Tree::build, no forced line)


@dataclass
class Menu:
    bet_fracs: list[float]
    raise_fracs: list[float]
    max_raises: int


@dataclass
class Spec:
    stake: tuple[int, int]
    stacks: tuple[int, int]
    first: int
    min_bet: int
    menu: Menu
    river: Menu | None = None


def spec() -> Spec:
    return Spec((1000, 1000), (9000, 9000), 1, 100, Menu([0.33, 0.75, 1.5], [1.0], 3))


def small_turn_spec() -> Spec:
    return Spec((1000, 1000), (9000, 9000), 1, 100, Menu([0.5, 1.0], [1.0], 2), Menu([0.75], [], 1))


def turn_spec() -> Spec:
    return Spec((1000, 1000), (9000, 9000), 1, 100, Menu([0.33, 0.75, 1.5], [1.0], 3), Menu([0.33, 0.75, 1.5], [1.0], 3))


def _round(x: float) -> int:
    """Rust f64::round (half away from zero) for the non-negative sizes here."""
    return int(math.floor(x + 0.5))


@dataclass
class Node:
    kind: str  # "T" terminal, "D" decision, "C" chance
    depth: int
    player: int = -1
    acts: list = field(default_factory=list)
    children: list = field(default_factory=list)
    slot: int = -1
    folded: int | None = None
    stake: tuple[int, int] = (0, 0)
    view: int = 0
    norm: float = 1.0
    kid_views: list = field(default_factory=list)


class Tree:
    def __init__(self, sp: Spec, hands: Hands):
        turn = hands.views > 1
        if turn != (sp.river is not None):
            raise ValueError("a river menu is required for a turn solve and only for one")
        self.sp, self.hands = sp, hands
        self.nodes: list[Node] = []
        self.slots = 0
        start = dict(stake=list(sp.stake), stacks=list(sp.stacks), to_act=sp.first, last_raise=0, raises=0,
                     acted=[False, False])
        self._grow(sp.menu, start, None if turn else 0, 0)

    @staticmethod
    def _legal(sp: Spec, menu: Menu, b: dict) -> list:
        p = b["to_act"]
        q = 1 - p
        c = b["stake"][q] - b["stake"][p]
        stack = b["stacks"][p]
        acts = []
        if c > 0:
            acts.append(("f", 0))
        acts.append(("c", 0))
        if stack > c and b["stacks"][q] > 0:
            pot = float(b["stake"][0] + b["stake"][1])
            floor = max(b["last_raise"], sp.min_bet)
            adds = []
            if b["raises"] < menu.max_raises:
                for f in (menu.bet_fracs if c == 0 else menu.raise_fracs):
                    add = max(_round(f * pot), floor) if c == 0 else c + max(_round(f * (pot + c)), floor)
                    if add < stack:
                        adds.append(add)
            adds.append(stack)
            acts.extend(("b", a) for a in sorted(set(adds)))
        return acts

    @staticmethod
    def _apply(b: dict, act) -> tuple[dict, object]:
        p = b["to_act"]
        q = 1 - p
        c = b["stake"][q] - b["stake"][p]
        n = dict(stake=list(b["stake"]), stacks=list(b["stacks"]), to_act=b["to_act"], last_raise=b["last_raise"],
                 raises=b["raises"], acted=list(b["acted"]))
        kind, add = act
        if kind == "f":
            return n, ("fold", p)
        if kind == "c":
            pay = min(c, b["stacks"][p])
            n["stake"][p] += pay
            n["stacks"][p] -= pay
            n["acted"][p] = True
        else:
            n["stake"][p] += add
            n["stacks"][p] -= add
            n["acted"][p] = True
            by = n["stake"][p] - b["stake"][q]
            if by > 0:
                n["last_raise"] = max(by, b["last_raise"])
                n["raises"] += 1
                n["acted"][q] = False
        n["to_act"] = q

        def owes(x):
            return n["stacks"][x] > 0 and (n["stake"][x] < n["stake"][1 - x] or not n["acted"][x])

        if not owes(0) and not owes(1):
            return n, ("round",)
        if n["stacks"][q] == 0 and n["stake"][p] >= n["stake"][q]:
            return n, ("round",)
        return n, None

    def _add(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _grow(self, menu: Menu, b: dict, view, depth: int) -> int:
        acts = self._legal(self.sp, menu, b)
        idx = self._add(Node("D", depth, player=b["to_act"], acts=acts, slot=self.slots))
        self.slots += 1
        kids = []
        for a in acts:
            nb, end = self._apply(b, a)
            if end is not None and end[0] == "fold":
                child = self._add(Node("T", depth + 1, folded=end[1], stake=tuple(nb["stake"]), view=0))
            elif end is not None:
                if view is not None:
                    child = self._add(Node("T", depth + 1, stake=tuple(nb["stake"]), view=view))
                else:
                    child = self._chance(nb, depth + 1)
            else:
                child = self._grow(menu, nb, view, depth + 1)
            kids.append(child)
        self.nodes[idx].children = kids
        return idx

    def _chance(self, b: dict, depth: int) -> int:
        idx = self._add(Node("C", depth, norm=float(self.hands.views - 4)))
        all_in = b["stacks"][0] == 0 or b["stacks"][1] == 0
        kids, views = [], []
        for v in range(self.hands.views):
            if all_in:
                child = self._add(Node("T", depth + 1, stake=tuple(b["stake"]), view=v))
            else:
                start = dict(stake=list(b["stake"]), stacks=list(b["stacks"]), to_act=self.sp.first, last_raise=0,
                             raises=0, acted=[False, False])
                child = self._grow(self.sp.river, start, v, depth + 1)
            kids.append(child)
            views.append(v)
        self.nodes[idx].children = kids
        self.nodes[idx].kid_views = views
        return idx


# ------------------------------------------------------------------ reference solver (search.rs, NumPy float64)


class RefSolver:
    """search.rs `Solver` without frozen nodes: the oracle. Other schedules:
    `plus`, DCFR+ (Xu et al. 2024): each regret clipped at 0 after its
    update, and the positive discount's t is t - 1. `shift`: the positive
    discount's t is max(t - shift, 0) (default 1 with `plus`, else 0).
    `restart`: the strategy sum restarts at every power of 4, as in
    postflop-solver (its schedule: shift 2, restart, gamma 3; its iteration
    t' counts from 0, so t' = t - 1): the sum weight's t is t' minus the
    largest power of 4 at or below t' (0 for t' = 0)."""

    def __init__(self, hands: Hands, tree: Tree, ranges, gamma: float = GAMMA, plus: bool = False,
                 shift: int | None = None, restart: bool = False):
        self.h, self.t = hands, tree
        self.ranges = [np.asarray(ranges[0], dtype=np.float64), np.asarray(ranges[1], dtype=np.float64)]
        self.gamma = gamma
        self.plus = plus
        self.shift = (1 if plus else 0) if shift is None else shift
        self.restart = restart
        n = hands.n
        self.regret, self.sum = {}, {}
        for node in tree.nodes:
            if node.kind == "D":
                self.regret[node.slot] = np.zeros((len(node.acts), n))
                self.sum[node.slot] = np.zeros((len(node.acts), n))
        self.W = [hands.showdown_matrix(v) for v in range(hands.views)]
        self.frozen = {}
        self.iter = 0

    def freeze(self, slot: int, strategy):
        """Play `strategy` ([actions, hands]) at `slot`: no regrets, no sums."""
        self.frozen[slot] = np.asarray(strategy, dtype=np.float64)

    def _terminal(self, node: Node, p: int, rq):
        if node.folded is not None:
            amount = -float(node.stake[p]) if node.folded == p else float(node.stake[1 - p])
            return (self.h.compat @ rq) * amount
        return (self.W[node.view] @ rq) * float(min(node.stake))

    @staticmethod
    def _current(r):
        pos = np.maximum(r, 0.0)
        tot = pos.sum(axis=0)
        k = r.shape[0]
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / k)

    def _cfr(self, i: int, p: int, rq, rp):
        node = self.t.nodes[i]
        if node.kind == "T":
            return self._terminal(node, p, rq)
        if node.kind == "C":
            out = np.zeros(self.h.n)
            for kid, v in zip(node.children, node.kid_views):
                live = self.h.alive[v]
                out += self._cfr(kid, p, rq * live, rp * live) * live
            return out / node.norm
        frozen = node.slot in self.frozen
        sig = self.frozen[node.slot] if frozen else self._current(self.regret[node.slot])
        if node.player == p:
            vals = [self._cfr(c, p, rq, rp * sig[a]) for a, c in enumerate(node.children)]
            v = sum(sig[a] * vals[a] for a in range(len(vals)))
            if not frozen:
                self.regret[node.slot] += np.array(vals) - v
                if self.plus:
                    np.maximum(self.regret[node.slot], 0.0, out=self.regret[node.slot])
            return v
        if not frozen:
            self.sum[node.slot] = self.sum[node.slot] * self.sum_w + rq * sig
        return sum(self._cfr(c, p, rq * sig[a], rp) for a, c in enumerate(node.children))

    def begin(self):
        """Start an iteration: count it and discount the regrets."""
        self.iter += 1
        t = float(self.iter)
        d = max(t - self.shift, 0.0)
        pos = d ** ALPHA / (d ** ALPHA + 1.0)
        for r in self.regret.values():
            r *= np.where(r > 0, pos, 0.5)
        x = t / (t + 1.0)
        if self.restart:
            tp, q = self.iter - 1, 1
            while q * 4 <= tp:
                q *= 4
            tg = float(tp - q) if tp > 0 else 0.0
            x = tg / (tg + 1.0)
        self.sum_w = x ** self.gamma

    def traverse(self, p: int):
        """One player's half of the iteration (alternating updates)."""
        self._cfr(0, p, self.ranges[1 - p], self.ranges[p])

    def step(self):
        self.begin()
        for p in (0, 1):
            self.traverse(p)


# ------------------------------------------------------------------ GPU solver


class GpuSolver:
    """The same DCFR, level-synchronous on a torch device."""

    def __init__(self, hands: Hands, tree: Tree, ranges, device: str = "cuda", dtype=None):
        import torch

        self.torch = torch
        self.dev = torch.device(device)
        self.dtype = dtype or torch.float32
        f = dict(device=self.dev, dtype=self.dtype)
        nodes = tree.nodes
        N, n = len(nodes), hands.n
        self.N, self.n = N, n
        dec = [i for i, x in enumerate(nodes) if x.kind == "D"]
        dec.sort(key=lambda i: nodes[i].slot)
        self.kmax = max(len(nodes[i].acts) for i in dec)
        D = len(dec)
        children = np.full((D, self.kmax), N, dtype=np.int64)  # N: a dummy row (value 0)
        mask = np.zeros((D, self.kmax))
        for d, i in enumerate(dec):
            k = len(nodes[i].children)
            children[d, :k] = nodes[i].children
            mask[d, :k] = 1.0
        depth = np.array([x.depth for x in nodes])
        self.max_depth = int(depth.max())
        t = lambda a, dt=None: torch.as_tensor(a, device=self.dev, dtype=dt)  # noqa: E731
        self.dec_node = t(np.array(dec), torch.long)
        self.dec_player = t(np.array([nodes[i].player for i in dec]), torch.long)
        self.children = t(children, torch.long)
        self.mask = t(mask).to(self.dtype)[:, :, None]
        self.kcount = self.mask.sum(1, keepdim=True)
        dec_depth = depth[dec]
        self.dec_levels = [t(np.nonzero(dec_depth == L)[0], torch.long) for L in range(self.max_depth + 1)]
        # Chance kids by the chance node's depth.
        ck = {L: ([], [], []) for L in range(self.max_depth + 1)}
        for i, x in enumerate(nodes):
            if x.kind == "C":
                for kid, v in zip(x.children, x.kid_views):
                    ck[x.depth][0].append(kid)
                    ck[x.depth][1].append(i)
                    ck[x.depth][2].append(v)
        self.chance_levels = [tuple(t(np.array(a, dtype=np.int64), torch.long) for a in ck[L]) for L in range(self.max_depth + 1)]
        # Per level: the chance nodes themselves and each kid's row among them,
        # so the upward sum needs no unique() (a device sync) or full-size buffer.
        self.chance_up = []
        for L in range(self.max_depth + 1):
            parents = sorted(set(ck[L][1]))
            row = {c: r for r, c in enumerate(parents)}
            self.chance_up.append((t(np.array(parents, dtype=np.int64), torch.long),
                                   t(np.array([row[c] for c in ck[L][1]], dtype=np.int64), torch.long)))
        self.norm = torch.ones(N + 1, 1, **f)
        # Each node's chance weight: 1 / norm per chance node above it (parents
        # precede their children in the node list).
        weight = np.ones(N + 1)
        for i, x in enumerate(nodes):
            if x.kind == "C":
                self.norm[i] = x.norm
            for c in x.children:
                weight[c] = weight[i] / (x.norm if x.kind == "C" else 1.0)
        self.weight = t(weight).to(self.dtype)
        # Terminals.
        folds = [i for i, x in enumerate(nodes) if x.kind == "T" and x.folded is not None]
        self.fold_nodes = t(np.array(folds, dtype=np.int64), torch.long)
        amt = np.zeros((2, len(folds)))
        for j, i in enumerate(folds):
            x = nodes[i]
            for p in (0, 1):
                amt[p, j] = -float(x.stake[p]) if x.folded == p else float(x.stake[1 - p])
        self.fold_amt = t(amt).to(self.dtype)[:, :, None]
        sds = [i for i, x in enumerate(nodes) if x.kind == "T" and x.folded is None]
        by_view = [[i for i in sds if nodes[i].view == v] for v in range(hands.views)]
        smax = max(len(b) for b in by_view)
        sd = np.full((hands.views, smax), N, dtype=np.int64)
        sdm = np.zeros((hands.views, smax))
        for v, b in enumerate(by_view):
            sd[v, :len(b)] = b
            sdm[v, :len(b)] = [float(min(nodes[i].stake)) for i in b]
        self.sd_nodes = t(sd, torch.long)
        self.sd_m = t(sdm).to(self.dtype)[:, :, None]
        self.C = t(hands.compat).to(self.dtype)
        self.W = torch.stack([t(hands.showdown_matrix(v)).to(self.dtype) for v in range(hands.views)])
        self.alive = t(hands.alive).to(self.dtype)
        self.ranges = [t(np.asarray(r)).to(self.dtype) for r in ranges]
        self.regret = torch.zeros(D, self.kmax, n, **f)
        self.sum = torch.zeros(D, self.kmax, n, **f)
        self.iter = 0

    def _current(self, r):
        pos = r.clamp(min=0) * self.mask
        tot = pos.sum(1, keepdim=True)
        return self.torch.where(tot > 0, pos / tot.clamp(min=1e-300 if self.dtype == self.torch.float64 else 1e-30),
                                self.mask / self.kcount)

    def _down(self, p: int, sig, both: bool = True, update_sum: float | None = None):
        """Reaches at every node: `rq` the opponent's (of `p`), `rp` p's own.
        `update_sum`: the strategy-sum discount (the opponent's nodes)."""
        torch = self.torch
        rq = torch.zeros(self.N + 1, self.n, device=self.dev, dtype=self.dtype)
        rp = torch.zeros_like(rq) if both else None
        rq[0] = self.ranges[1 - p]
        if both:
            rp[0] = self.ranges[p]
        for L in range(self.max_depth + 1):
            d = self.dec_levels[L]
            if len(d):
                nodes = self.dec_node[d]
                s = sig[d]
                isp = (self.dec_player[d] == p)[:, None, None]
                q = rq[nodes][:, None, :]
                crq = torch.where(isp, q.expand_as(s), q * s)
                ch = self.children[d].reshape(-1)
                rq[ch] = crq.reshape(-1, self.n)
                if both:
                    r = rp[nodes][:, None, :]
                    rp[ch] = torch.where(isp, r * s, r.expand_as(s)).reshape(-1, self.n)
                if update_sum is not None:
                    opp = ~isp
                    self.sum[d] = torch.where(opp, self.sum[d] * update_sum + q * s, self.sum[d])
            kid, parent, view = self.chance_levels[L]
            if len(kid):
                live = self.alive[view]
                rq[kid] = rq[parent] * live
                if both:
                    rp[kid] = rp[parent] * live
        return rq, rp

    def _terminals(self, p: int, rq):
        val = self.torch.zeros_like(rq)
        if len(self.fold_nodes):
            val[self.fold_nodes] = (rq[self.fold_nodes] @ self.C) * self.fold_amt[p]
        r = rq[self.sd_nodes]  # [V, S, n]
        vals = self.torch.bmm(r, self.W.transpose(1, 2)) * self.sd_m
        val[self.sd_nodes.reshape(-1)] = vals.reshape(-1, self.n)
        val[self.N] = 0.0
        return val

    def _chance_up(self, L: int, val):
        kid, _, view = self.chance_levels[L]
        if len(kid):
            nodes, row = self.chance_up[L]
            acc = self.torch.zeros(len(nodes), self.n, device=self.dev, dtype=self.dtype)
            acc.index_add_(0, row, val[kid] * self.alive[view])
            val[nodes] = acc / self.norm[nodes]

    def begin(self):
        """Start an iteration: count it and discount the regrets."""
        self.iter += 1
        t = float(self.iter)
        pos = t ** ALPHA / (t ** ALPHA + 1.0)
        # Scale in place in the solver's dtype (a `where` of two Python scalars
        # would build a float32 factor tensor).
        r = self.regret
        r.copy_(self.torch.where(r > 0, r * pos, r * 0.5))
        self.sum_w = (t / (t + 1.0)) ** GAMMA

    def traverse(self, p: int):
        """One player's half of the iteration (alternating updates)."""
        torch = self.torch
        sig = self._current(self.regret)
        rq, _ = self._down(p, sig, both=False, update_sum=self.sum_w)
        val = self._terminals(p, rq)
        for L in range(self.max_depth, -1, -1):
            self._chance_up(L, val)
            d = self.dec_levels[L]
            if len(d):
                vc = val[self.children[d]]  # [m, k, n]
                s = sig[d]
                isp = (self.dec_player[d] == p)[:, None, None]
                v = torch.where(isp[:, :, 0], (s * vc).sum(1), vc.sum(1))
                val[self.dec_node[d]] = v
                self.regret[d] += torch.where(isp, (vc - v[:, None, :]) * self.mask[d], 0.0)

    def step(self):
        self.begin()
        for p in (0, 1):
            self.traverse(p)

    def average(self):
        tot = self.sum.sum(1, keepdim=True)
        return self.torch.where(tot > 0, self.sum / tot.clamp(min=1e-300 if self.dtype == self.torch.float64 else 1e-30),
                                self.mask / self.kcount)

    def _joint_mass(self):
        return float(self.ranges[0] @ (self.C @ self.ranges[1]))

    def exploitability(self) -> float:
        """Both players' best-response values against the average strategy, in
        chips per dealt pair (search.rs `Solver::exploitability`)."""
        torch = self.torch
        avg = self.average()
        total = 0.0
        for p in (0, 1):
            rq, _ = self._down(p, avg, both=False)
            val = self._terminals(p, rq)
            for L in range(self.max_depth, -1, -1):
                self._chance_up(L, val)
                d = self.dec_levels[L]
                if len(d):
                    vc = val[self.children[d]]
                    isp = (self.dec_player[d] == p)[:, None]
                    best = torch.where(self.mask[d] > 0, vc, -torch.inf).amax(1)
                    val[self.dec_node[d]] = torch.where(isp, best, vc.sum(1))
            total += float(self.ranges[p] @ val[0])
        return total / self._joint_mass()

    def value(self, p: int) -> float:
        """Value to `p` of the average-strategy profile, chips per dealt pair."""
        avg = self.average()
        rq, rp = self._down(p, avg, both=True)
        val = self._terminals(p, rq)
        term = self.torch.zeros(self.N + 1, dtype=self.dtype, device=self.dev)
        idx = self.torch.cat([self.fold_nodes, self.sd_nodes.reshape(-1)])
        term[idx] = (rp[idx] * val[idx]).sum(1) * self.weight[idx]
        term[self.N] = 0.0
        return float(term.sum()) / self._joint_mass()
