# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Depth-limited flop search (FLOPSEARCH.md, D3): the flop round solved on the
GPU, its leaves valued by the turn-start value net (`value_net`).

The Rust search builds the flop-round subgame and sends it through
`FlatSubgame` (`PlayAgent.with_search(from_flop=True, solver=FlopSearch(...))`).
Where the round ends without a fold the tree has a leaf (kind 3: chips
behind, equal stakes) or a runout (kind 4: someone is all-in). This module
solves it with the Rust solver's DCFR (`gpu_search.RefSolver`: alpha 1.5,
negative regrets halved, strategy sums weighted (t / (t + 1)) ** gamma,
alternating updates), level by level over the tree:

* folds: inclusion-exclusion over cards, as `gpu_fused`;
* runouts: `min(stake) * E @ r_q`, with `E[h, q]` the mean showdown sign over
  the 990 turn and river pairs each disjoint pair sees (`flop_runouts`);
* leaves: for each of the 49 turn cards t, both reaches restricted to the
  holdings off t and scaled to mass 1 are the value net's input (the board
  plus t, pot and stacks at the leaf); its values (pots per unit of opponent
  mass) come back as chips, `cfv_p(h) = 1/45 sum_t v_t,p(h) M_t,q pot`, where
  `M_t,q` is the opponent's mass off t. Each disjoint pair sees 45 turn cards.
  A zero-mass range enters as uniform (its values are multiplied by 0 or
  belong to a holding nobody reaches).

The tree is small (a 200bb flop round has 9 leaves), so launching its kernels
one at a time from Python costs more than running them: `FlopSolver.solve`
captures one iteration as a CUDA graph and replays it (bit-identical).

`CheckdownValues` replaces the net with the exact values of a game with no
betting after the flop: with it the depth-limited solve is the Rust solver's
full solve of that game (`flop_checkdown`), the test of the leaf plumbing
above, and the X_eq ablation of FLOPSEARCH.md D4.
"""
from __future__ import annotations

import gc
import inspect
import threading
import time

import numpy as np
import torch

from . import value_net
from .gpu_fused import GPU_LOCK, SubgameGpu, drop_frames
from .gpu_search import ALPHA

GAMMA = 2.0
HOLDINGS = 1176  # two-card holdings off a flop, and turn-and-river pairs
TURNS = 49  # unseen cards on the flop
TURN_SHARE = 45  # turn cards a disjoint pair of holdings sees
RIVERS = 44  # river cards a disjoint pair sees on a turn board
_EXACT = {torch.float64}


def combos(board) -> np.ndarray:
    """[1176, 2]: the holdings (and the runouts) off a flop, in the Rust order."""
    used = {int(c) for c in board}
    return np.array([(a, b) for a in range(52) for b in range(a + 1, 52) if a not in used and b not in used],
                    dtype=np.int64)


def _sign_sum(strength: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """[n, n] int32: sum over the rows of `strength` ([k, n] int32, 0 = dead)
    of sign(s_h - s_q) over the pairs both live on that row."""
    n = strength.shape[1]
    out = torch.zeros(n, n, dtype=torch.int32, device=strength.device)
    for lo in range(0, len(strength), chunk):
        s = strength[lo:lo + chunk]
        a, b = s[:, :, None], s[:, None, :]
        win = ((a > b) & (b > 0)).to(torch.int8)  # b live, so a is
        lose = ((a < b) & (a > 0)).to(torch.int8)
        out += (win - lose).sum(0, dtype=torch.int32)
    return out


def _strengths(board, device) -> torch.Tensor:
    import parciver_engine as pe

    raw = np.frombuffer(pe.flop_runouts([int(c) for c in board]), dtype="<u4").reshape(HOLDINGS, HOLDINGS)
    if raw.max() >= 2 ** 31:
        raise ValueError("hand strengths overflow int32")
    return torch.as_tensor(raw.astype(np.int32), device=device)


def _compat(hold: np.ndarray) -> np.ndarray:
    a, b = hold[:, 0], hold[:, 1]
    return ((a[:, None] != a[None, :]) & (a[:, None] != b[None, :])
            & (b[:, None] != a[None, :]) & (b[:, None] != b[None, :]))


def runout_equity(board, device="cuda", dtype=torch.float32) -> torch.Tensor:
    """[1176, 1176] `E[h, q]`: the mean over the 990 turn and river pairs that
    miss both holdings of sign(strength_h - strength_q); 0 unless disjoint."""
    hold = combos(board)
    counts = _sign_sum(_strengths(board, device))
    compat = torch.as_tensor(_compat(hold), device=device)
    return (counts * compat).to(dtype) / (TURN_SHARE * RIVERS / 2)


class CheckdownValues:
    """A value net's interface (`net_values`) answered exactly for a game with
    no betting after the flop: each turn board's showdown equity over its 44
    river cards, x pot, net of the stake, per unit of opponent mass (pots).
    One flop only. Holds [49, 1176, 1176] counts (0.27 GB in float32)."""

    def __init__(self, board, device="cuda", dtype=torch.float64):
        self.flop = tuple(int(c) for c in board)
        hold = combos(board)
        self.turns = [c for c in range(52) if c not in self.flop]
        s = _strengths(board, device)
        pair = {tuple(p): i for i, p in enumerate(hold.tolist())}
        compat = torch.as_tensor(_compat(hold), device=device)
        w = torch.empty(TURNS, HOLDINGS, HOLDINGS, dtype=dtype, device=device)
        for i, t in enumerate(self.turns):
            rows = [pair[(min(t, r), max(t, r))] for r in self.turns if r != t]
            w[i] = (_sign_sum(s[rows]) * compat).to(dtype)
        self.w = w
        self.idx = torch.as_tensor(value_net.PAIR_INDEX[hold[:, 0], hold[:, 1]], device=device)
        self._checked = None

    def __call__(self, ranges, board, live, half, stack):
        if board is not self._checked:  # once per board tensor: a CUDA graph capture cannot read the device
            if tuple(board[0, :3].tolist()) != self.flop or board[:, 3].tolist() != self.turns:
                raise ValueError("CheckdownValues answers its own flop's 49 turn boards, in order")
            self._checked = board
        r = ranges[..., self.idx].to(self.w.dtype)  # [L, T, 2, n]
        L = r.shape[0]
        rq = r.flip(2).permute(1, 0, 2, 3).reshape(TURNS, L * 2, HOLDINGS)  # the opponent's, per turn card
        v = torch.bmm(rq, self.w.transpose(1, 2)) * (0.5 / RIVERS)  # [T, L * 2, n]
        v = v.view(TURNS, L, 2, HOLDINGS).permute(1, 0, 2, 3)
        out = ranges.new_zeros(ranges.shape)
        out[..., self.idx] = v.to(ranges.dtype)
        return out


def net_values(net):
    """The value net as a leaf evaluator: `f(ranges [L, T, 2, 1326] (each of
    mass 1), board [T, 4], live [T, 1326], half [L], stack [L])` -> values
    [L, T, 2, 1326] (pots per unit of opponent mass, dead holdings 0), in
    float32 as the net was gated (`value_net.predict`'s inputs)."""

    def f(ranges, board, live, half, stack):
        L, T = ranges.shape[:2]
        onehot = torch.zeros(T, 52, device=ranges.device)
        onehot.scatter_(1, board, 1.0)
        h, s = half.double(), stack.double()
        scal = torch.stack([h / (h + s), s / (2 * h)], -1).float()  # value_net.scalars
        with torch.no_grad():
            v = net(ranges.reshape(L * T, 2, -1).float(), onehot.repeat(L, 1), scal.repeat_interleave(T, 0),
                    live.repeat(L, 1))
        return v.view(L, T, 2, -1).to(ranges.dtype)

    return f


class NetLeaves:
    """Leaf values in chips from a leaf evaluator (`net_values`,
    `CheckdownValues`): `__call__(reach [L, 2, n], half [L], stack [L])` ->
    [L, 2, n], each player's counterfactual values at each leaf."""

    def __init__(self, board, evaluate, device="cuda", dtype=torch.float32):
        flop = [int(c) for c in board]
        hold = combos(flop)
        self.turns = [c for c in range(52) if c not in flop]
        self.evaluate = evaluate
        t = torch.as_tensor(self.turns, device=device)
        h = torch.as_tensor(hold, device=device)
        self.mask = ((h[None, :, 0] != t[:, None]) & (h[None, :, 1] != t[:, None])).to(dtype)  # [T, n]
        self.uniform = self.mask / self.mask.sum(1, keepdim=True)
        self.board = torch.as_tensor([flop + [c] for c in self.turns], device=device)  # [T, 4]
        self.live = torch.as_tensor(value_net.live_mask(self.board.cpu().numpy()), device=device)  # [T, 1326]
        self.idx = torch.as_tensor(value_net.PAIR_INDEX[hold[:, 0], hold[:, 1]], device=device)

    def __call__(self, reach, half, stack):
        L = reach.shape[0]
        r = reach[:, None] * self.mask[None, :, None, :]  # [L, T, 2, n]
        mass = r.sum(-1, keepdim=True)
        unit = torch.where(mass > 0, r / torch.where(mass > 0, mass, 1.0), self.uniform[None, :, None, :])
        x = unit.new_zeros(L, TURNS, 2, value_net.HOLDINGS)
        x[..., self.idx] = unit
        v = self.evaluate(x, self.board, self.live, half, stack)[..., self.idx] * self.mask[None, :, None, :]
        pot = (2 * half).to(v.dtype)
        return (v * mass.flip(2)).sum(1) * (pot / TURN_SHARE)[:, None, None]


def flop_group(board) -> list[int]:
    """The suit permutations (`value_net.SUIT_PERMS` rows, the identity first)
    that map the flop onto itself: its stabilizer G."""
    flop = {int(c) for c in board}
    return [k for k in range(len(value_net.SUIT_PERMS)) if set(value_net.CARD_PERM[k, list(flop)].tolist()) == flop]


def _compose(a: int, b: int) -> int:
    """The SUIT_PERMS row of `a` after `b` (suit s -> a[b[s]])."""
    p = value_net.SUIT_PERMS
    return int(np.flatnonzero((p == p[a][p[b]]).all(1))[0])


def sym_plan(board, cosets: bool = True) -> tuple:
    """FLOP-SYM's rows for a flop: (G, canonical turns, orbit sizes, each
    row's canonical turn position and relabelling (a SUIT_PERMS row), each
    canonical turn's rows). One row per G-orbit of turn cards; with `cosets`,
    one more per other coset of G_t (the members of G fixing the turn) in
    H_b (the turn board's stabilizer): relabellings that G_t's members give
    are the same input."""
    flop = [int(c) for c in board]
    group = flop_group(flop)
    turns, seen, size = [], set(), []
    for t in range(52):
        if t in flop or t in seen:
            continue
        orbit = {int(value_net.CARD_PERM[k, t]) for k in group}
        seen |= orbit
        turns.append(t)
        size.append(len(orbit))
    row_turn, row_perm, slots = [], [], []
    for j, t in enumerate(turns):
        b = flop + [t]
        reps = [0]
        if cosets:
            h_b = [k for k in range(len(value_net.SUIT_PERMS)) if set(value_net.CARD_PERM[k, b].tolist()) == set(b)]
            g_t = [k for k in group if value_net.CARD_PERM[k, t] == t]
            covered = set(g_t)
            for k in h_b:
                if k not in covered:
                    reps.append(k)
                    covered |= {_compose(k, g) for g in g_t}
        slots.append(list(range(len(row_turn), len(row_turn) + len(reps))))
        row_turn += [j] * len(reps)
        row_perm += reps
    return group, turns, size, row_turn, row_perm, slots


class SymLeaves:
    """`NetLeaves` made suit-symmetric (FLOP-SYM, `records/flop-sym/PREREG.md`)
    for ranges that the flop's stabilizer G leaves unchanged. One canonical
    turn card per G-orbit is evaluated; with `cosets` (C2) the net also runs
    on the leaf range relabelled by one member of each other coset of G_t in
    the turn board's own stabilizer H_b, each output relabelled back and the
    results averaged (C1 without). The canonical turns' contributions, each
    weighted by its orbit's size, are summed and averaged over G, once per
    holding orbit, so G-equal holdings get bit-equal values. `perm` [|G|, n]:
    holding i -> its image under each member of G; `canon` [n]: each
    holding's orbit representative. Same interface as `NetLeaves`."""

    def __init__(self, board, evaluate, device="cuda", dtype=torch.float32, cosets: bool = True):
        flop = [int(c) for c in board]
        group, turns, size, row_turn, row_perm, slots = sym_plan(flop, cosets)
        if len(group) < 2:
            raise ValueError("the flop's stabilizer is the identity: use NetLeaves")
        hold = combos(flop)
        pair = value_net.PAIR_INDEX[hold[:, 0], hold[:, 1]]
        pos = np.full(value_net.HOLDINGS, -1, dtype=np.int64)
        pos[pair] = np.arange(len(hold))
        perm = np.stack([pos[value_net.PAIR_PERM[k, pair]] for k in group])
        assert (perm >= 0).all() and sum(size) == TURNS
        self.group, self.evaluate = group, evaluate
        self.perm_host, self.canon_host = perm, perm.min(0)
        self.turns, self.rows = turns, len(row_turn)
        width = max(len(s) for s in slots)
        pad = [s + [self.rows] * (width - len(s)) for s in slots]  # row `rows`: zeros
        t = torch.as_tensor(turns, device=device)
        h = torch.as_tensor(hold, device=device)
        self.mask = ((h[None, :, 0] != t[:, None]) & (h[None, :, 1] != t[:, None])).to(dtype)  # [C, n]
        self.uniform = self.mask / self.mask.sum(1, keepdim=True)
        self.size = torch.as_tensor(size, dtype=dtype, device=device)  # [C]
        self.idx = torch.as_tensor(pair, device=device)
        self.row_turn = torch.as_tensor(row_turn, device=device)
        board_rows = np.array([flop + [turns[j]] for j in row_turn])  # a relabelling in H_b keeps the board's set
        self.board = torch.as_tensor(board_rows, device=device)  # [R, 4]
        self.live = torch.as_tensor(value_net.live_mask(board_rows), device=device)  # [R, 1326]
        self.to_row = torch.as_tensor(value_net.PAIR_PERM_INV[row_perm], device=device)  # [R, 1326]: input relabelled
        self.back = torch.as_tensor(value_net.PAIR_PERM[row_perm], device=device)  # [R, 1326]: output relabelled back
        self.slot = torch.as_tensor(pad, device=device)  # [C, width]
        self.count = torch.as_tensor([len(s) for s in slots], dtype=dtype, device=device)
        self.perm = torch.as_tensor(perm, device=device)
        self.canon = torch.as_tensor(self.canon_host, device=device)

    def symmetric(self, x) -> bool:
        """Whether G leaves `x` ([..., n], holdings last) exactly unchanged."""
        x = np.asarray(x, dtype=np.float64)
        return all(np.array_equal(x[..., p], x) for p in self.perm_host[1:])

    def average(self, x: np.ndarray) -> np.ndarray:
        """`x` ([..., n]) averaged over G, once per holding orbit: exactly symmetric."""
        return x[..., self.perm_host].mean(-2)[..., self.canon_host]

    def __call__(self, reach, half, stack):
        L, R = reach.shape[0], self.rows
        r = reach[:, None] * self.mask[None, :, None, :]  # [L, C, 2, n]
        mass = r.sum(-1, keepdim=True)
        unit = torch.where(mass > 0, r / torch.where(mass > 0, mass, 1.0), self.uniform[None, :, None, :])
        x = unit.new_zeros(L, len(self.turns), 2, value_net.HOLDINGS)
        x[..., self.idx] = unit
        x = torch.gather(x[:, self.row_turn], 3, self.to_row[None, :, None, :].expand(L, R, 2, -1))
        v = self.evaluate(x, self.board, self.live, half, stack)
        v = torch.gather(v, 3, self.back[None, :, None, :].expand(L, R, 2, -1))
        v = torch.cat([v, v.new_zeros(L, 1, 2, value_net.HOLDINGS)], 1)[:, self.slot].sum(2) / self.count[None, :, None, None]
        v = v[..., self.idx] * self.mask[None, :, None, :]  # [L, C, 2, n]
        u = (v * mass.flip(2) * self.size[None, :, None, None]).sum(1)  # [L, 2, n]
        u = u[..., self.perm].mean(-2)[..., self.canon]
        pot = (2 * half).to(u.dtype)
        return u * (pot / TURN_SHARE)[:, None, None]


class FlopSolver:
    """DCFR on a flop-round subgame (`FlatSubgame` dict), level by level.
    `leaves`: a `NetLeaves`, or an evaluator whose call takes `players`
    (`flop_leaves.FactoredLeaves`: asked for the traverser's values alone);
    `equity`: `runout_equity` of the job's board."""

    def __init__(self, job: dict, leaves, equity, device="cuda", dtype=torch.float32, gamma: float = GAMMA):
        dev = self.dev = torch.device(device)
        self.dtype, self.gamma, self.leaves = dtype, gamma, leaves
        self.one = "players" in inspect.signature(leaves).parameters  # the traverser's leaf values alone
        kind = np.asarray(job["kind"], dtype=np.int64)
        player = np.asarray(job["player"], dtype=np.int64)
        slot = np.asarray(job["slot"], dtype=np.int64)
        stake = np.asarray(job["stake"], dtype=np.int64).reshape(-1, 2)
        start = np.asarray(job["child_start"], dtype=np.int64)
        kids_flat = np.asarray(job["children"], dtype=np.int64)
        total = np.asarray(job["total"], dtype=np.int64)
        if len(job["board"]) != 3 or (kind == 2).any() or ((kind == 0) & (player < 0)).any():
            raise ValueError("not a flop-round subgame")
        N = self.N = len(kind)
        n = self.n = len(job["ranges"][0])
        dec = np.flatnonzero(kind == 1)
        if not np.array_equal(slot[dec], np.arange(len(dec))):
            raise ValueError("decision slots are not in node order")
        S = self.S = len(dec)
        kids = [kids_flat[start[i]:start[i + 1]] for i in range(N)]
        K = self.K = max(len(kids[i]) for i in dec)
        # Depth, and each node's path: the (slot, action) of each ancestor decision.
        depth = np.zeros(N, dtype=np.int64)
        path: list[list] = [[] for _ in range(N)]
        for i in range(N):
            for a, c in enumerate(kids[i]):
                if c <= i:
                    raise ValueError("a child before its parent")
                depth[c] = depth[i] + 1
                path[c] = path[i] + [(int(player[i]), int(slot[i]) * K + a)]
        ones = S * K  # the all-ones row appended to a flat strategy
        self.paths = []
        for o in (0, 1):
            mine = [[x for pl, x in path[i] if pl == o] for i in range(N)]
            width = max(1, max(len(m) for m in mine))
            self.paths.append(torch.as_tensor([m + [ones] * (width - len(m)) for m in mine], device=dev))
        self.kids = torch.full((S, K), N, dtype=torch.int64)
        self.amask = torch.zeros(S, K, dtype=dtype)
        for s, i in enumerate(dec):
            self.kids[s, :len(kids[i])] = torch.as_tensor(kids[i])
            self.amask[s, :len(kids[i])] = 1.0
        self.kids, self.amask = self.kids.to(dev), self.amask.to(dev)
        self.k = self.amask.sum(1)
        self.owner = torch.as_tensor(player[dec], device=dev)
        self.node_of = torch.as_tensor(dec, device=dev)
        self.levels = [np.flatnonzero(depth[dec] == d) for d in range(int(depth[dec].max()), -1, -1)]
        self.levels = [torch.as_tensor(x, device=dev) for x in self.levels]
        self.dk_host = [len(kids[i]) for i in dec]
        # Terminals.
        fold = np.flatnonzero(kind == 0)
        self.fold = torch.as_tensor(fold, device=dev)
        self.fold_player = torch.as_tensor(player[fold], device=dev)
        self.fold_stake = torch.as_tensor(stake[fold], dtype=dtype, device=dev)
        run = np.flatnonzero(kind == 4)
        self.run = torch.as_tensor(run, device=dev)
        self.run_stake = torch.as_tensor(stake[run].min(1), dtype=dtype, device=dev)
        leaf = np.flatnonzero(kind == 3)
        if (stake[leaf, 0] != stake[leaf, 1]).any() or ((total[None, :] - stake[leaf]) <= 0).any():
            raise ValueError("a leaf with unequal stakes or no chips behind")
        self.leaf = torch.as_tensor(leaf, device=dev)
        self.leaf_half = torch.as_tensor(stake[leaf, 0], dtype=dtype, device=dev)
        self.leaf_stack = torch.as_tensor((total[None, :] - stake[leaf]).min(1), dtype=dtype, device=dev)
        hold = combos(job["board"])
        inc = np.zeros((n, 52))
        inc[np.arange(n), hold[:, 0]] = 1.0
        inc[np.arange(n), hold[:, 1]] = 1.0
        self.inc = torch.as_tensor(inc, dtype=dtype, device=dev)
        self.ha, self.hb = torch.as_tensor(hold[:, 0], device=dev), torch.as_tensor(hold[:, 1], device=dev)
        self.equity = equity.to(dtype)
        self.ranges = torch.as_tensor(np.asarray(job["ranges"], dtype=np.float64), dtype=dtype, device=dev)
        self.regret = torch.zeros(S, K, n, dtype=dtype, device=dev)
        self.sum = torch.zeros(S, K, n, dtype=dtype, device=dev)
        self.fz = torch.zeros(S, dtype=torch.bool, device=dev)
        self.fstrat = torch.zeros(S, K, n, dtype=dtype, device=dev)
        self.val = torch.zeros(N + 1, n, dtype=dtype, device=dev)  # row N: padding (0)
        self.iter = 0
        self._t = torch.zeros(1, dtype=torch.int64, device=dev)  # iterations done, on the device
        self._half = self.regret.new_tensor(0.5)
        self.graph = None
        self._grow(1024)
        self._split()

    def _grow(self, n: int):
        """The discounts of iterations 1..n, on the device, computed on the
        host as the Rust solver does (so a captured graph can read them)."""
        t = [float(i) for i in range(1, n + 1)]
        self._pos = torch.tensor([x ** ALPHA / (x ** ALPHA + 1.0) for x in t], dtype=self.dtype, device=self.dev)
        self._w = torch.tensor([(x / (x + 1.0)) ** self.gamma for x in t], dtype=self.dtype, device=self.dev)
        self.graph = None  # it read the old tables

    def _split(self):
        """Per player p: the unfrozen slots p updates (regrets) and those of
        the other player whose strategy sums its traversal updates."""
        live = ~self.fz
        self.upd = [torch.nonzero(live & (self.owner == p)).flatten() for p in (0, 1)]
        self.avg_slots = [torch.nonzero(live & (self.owner != p)).flatten() for p in (0, 1)]

    def freeze(self, slot: int, strategy):
        """Play `strategy` (action-major, [actions, hands]) at `slot`: no
        regrets, no sums; the best responder plays it too. Before the first step."""
        s = torch.as_tensor(np.asarray(strategy, dtype=np.float64), dtype=self.dtype, device=self.dev).reshape(-1, self.n)
        if s.shape[0] != self.dk_host[slot]:
            raise ValueError(f"slot {slot} has {self.dk_host[slot]} actions, the strategy {s.shape[0]}")
        self.fstrat[slot].zero_()
        self.fstrat[slot, :s.shape[0]] = s
        self.fz[slot] = True
        self._split()

    def current(self):
        """Regret matching at every slot, frozen slots their strategy: [S, K, n]."""
        pos = self.regret.clamp(min=0)
        tot = pos.sum(1, keepdim=True)
        uni = (self.amask / self.k[:, None])[..., None]
        sig = torch.where(tot > 0, pos / torch.where(tot > 0, tot, 1.0), uni)
        return torch.where(self.fz[:, None, None], self.fstrat, sig)

    def average(self, slots=None):
        """The average strategy [S, K, n] (rows `slots` if given); frozen
        slots their strategy (search.rs `Solver::average`)."""
        s = self.sum if slots is None else self.sum[slots]
        fz = self.fz if slots is None else self.fz[slots]
        fs = self.fstrat if slots is None else self.fstrat[slots]
        am = self.amask if slots is None else self.amask[slots]
        k = am.sum(1)
        tot = s.sum(1, keepdim=True)
        avg = torch.where(tot > 0, s / torch.where(tot > 0, tot, 1.0), (am / k[:, None])[..., None])
        return torch.where(fz[:, None, None], fs, avg)

    def _reach(self, strat):
        """Both players' reach at every node under `strat` [S, K, n]: [2, N, n]."""
        flat = torch.cat([strat.reshape(-1, self.n), torch.ones(1, self.n, dtype=self.dtype, device=self.dev)])
        out = []
        for o in (0, 1):
            acc = self.ranges[o].expand(self.N, self.n)
            for d in range(self.paths[o].shape[1]):
                acc = acc * flat[self.paths[o][:, d]]
            out.append(acc)
        return torch.stack(out)

    def _terminals(self, p: int, reach):
        """Fold, runout and leaf values for `p` into `self.val`."""
        rq = reach[1 - p]
        if len(self.fold):
            r = rq[self.fold]
            card = r @ self.inc
            mass = r.sum(1, keepdim=True) - card[:, self.ha] - card[:, self.hb] + r
            amount = torch.where(self.fold_player == p, -self.fold_stake[:, p], self.fold_stake[:, 1 - p])
            self.val[self.fold] = mass * amount[:, None]
        if len(self.run):
            self.val[self.run] = (rq[self.run] @ self.equity.T) * self.run_stake[:, None]
        if len(self.leaf):
            both = reach[:, self.leaf].transpose(0, 1)  # [L, 2, n]
            if self.one:
                self.val[self.leaf] = self.leaves(both, self.leaf_half, self.leaf_stack, players=(p,))[:, 0]
            else:
                self.val[self.leaf] = self.leaves(both, self.leaf_half, self.leaf_stack)[:, p]

    def _up(self, p: int, strat, mode: str):
        """Values up the tree for `p`: "cfr" (expectation, regret update),
        "expect" (expectation) or "best" (best response)."""
        for lv in self.levels:
            cv = self.val[self.kids[lv]]  # [m, K, n]
            mine = (self.owner[lv] == p)[:, None, None]
            if mode == "best":
                best = torch.where(self.amask[lv][..., None] > 0, cv, torch.finfo(self.dtype).min).amax(1)
                play = (strat[lv] * cv).sum(1)  # a frozen node of p is played, not chosen
                v = torch.where(mine[:, 0], torch.where(self.fz[lv][:, None], play, best), cv.sum(1))
            else:
                w = torch.where(mine, strat[lv], self.amask[lv][..., None])
                v = (w * cv).sum(1)
            self.val[self.node_of[lv]] = v
            if mode == "cfr":
                self.regret[lv] += torch.where(mine & ~self.fz[lv][:, None, None], (cv - v[:, None]) * self.amask[lv][..., None], 0.0)

    def begin(self):
        """Start an iteration: count it and discount the regrets."""
        self.iter += 1
        if self.iter > len(self._pos):
            self._grow(2 * len(self._pos))
        self._begin_device()

    def _begin_device(self):
        pos, self.sum_w = self._pos.index_select(0, self._t), self._w.index_select(0, self._t)
        self._t += 1
        self.regret.mul_(torch.where(self.regret > 0, pos, self._half))

    def traverse(self, p: int):
        """One player's half of the iteration (alternating updates)."""
        sig = self.current()
        reach = self._reach(sig)
        q = self.avg_slots[p]
        self.sum[q] = self.sum[q] * self.sum_w + reach[1 - p][self.node_of[q]][:, None, :] * sig[q]
        self._terminals(p, reach)
        self._up(p, sig, "cfr")

    def step(self):
        self.begin()
        for p in (0, 1):
            self.traverse(p)

    def solve(self, iters: int, graph: bool = True):
        """`iters` steps. With `graph` (CUDA), the first is run on a side
        stream and the next captured as one CUDA graph, replayed for the rest
        (the tree is small, so launching its kernels one by one from Python
        costs more than running them). The same kernels on the same
        tensors: bit-identical to `step()`."""
        if not (graph and self.dev.type == "cuda"):
            for _ in range(iters):
                self.step()
            return
        if self.iter + iters > len(self._pos):
            self._grow(max(self.iter + iters, 2 * len(self._pos)))
        if self.graph is None and iters > 0:
            side = torch.cuda.Stream(self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):
                self.step()
            torch.cuda.current_stream(self.dev).wait_stream(side)
            g = torch.cuda.CUDAGraph()
            # No garbage collection while capturing (as FusedSolver.step): freeing
            # a Triton kernel unloads its module, which the capture forbids.
            with GPU_LOCK:
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(g):  # recorded, not run
                        self._begin_device()
                        for p in (0, 1):
                            self.traverse(p)
                finally:
                    if enabled:
                        gc.enable()
            self.graph = g
            iters -= 1
        for _ in range(iters):
            self.iter += 1
            self.graph.replay()

    def _joint(self) -> float:
        r0, r1 = self.ranges
        card = r1 @ self.inc
        return float(r0 @ (r1.sum() - card[self.ha] - card[self.hb] + r1))

    def value(self, p: int) -> float:
        """Value to `p` of the average profile, chips per dealt pair."""
        avg = self.average()
        reach = self._reach(avg)
        self._terminals(p, reach)
        self._up(p, avg, "expect")
        return float(self.ranges[p] @ self.val[0]) / self._joint()

    def exploitability(self) -> float:
        """Both best responses' values against the average profile, chips per
        dealt pair (search.rs `Solver::exploitability`). Exact when the leaf
        values do not depend on the responder's own range (check-down leaves);
        with a net the responder's range at a leaf is its average one."""
        avg = self.average()
        reach = self._reach(avg)
        total = 0.0
        for p in (0, 1):
            self._terminals(p, reach)
            self._up(p, avg, "best")
            total += float(self.ranges[p] @ self.val[0])
        return total / self._joint()


class FlopSearch:
    """The search's external subgame solver with flop search:
    `PlayAgent.with_search(from_flop=True, solver=FlopSearch(net))`. Flop jobs
    are solved here (net leaves, or exact check-down leaves with `net=None`:
    the X_eq ablation); turn and river jobs go to `turn` (a `SubgameGpu`).
    Runout equities and leaf tables are cached per flop. `graph`: replay each
    solve's iterations as a CUDA graph (`FlopSolver.solve`). `leaves`: the net
    leaves' path, "dense" (`NetLeaves` over `net_values`) or "factored"
    (`flop_leaves.FactoredLeaves` with its last stage in float64, KERNELS.md
    N1 + N1b), "sym" or "orbit" (FLOP-SYM's C2 and C1: `SymLeaves` with and
    without its cosets, for a job whose flop has a stabilizer beyond the
    identity and whose ranges and frozen strategies it leaves exactly
    unchanged; its returned strategies are averaged over the stabilizer.
    Any other job takes the dense path, bit-identical to "dense")."""

    def __init__(self, net=None, device="cuda", dtype=torch.float32, gamma: float = GAMMA, turn=None, cache: int = 4,
                 graph: bool = True, leaves: str = "dense"):
        if leaves not in ("dense", "factored", "sym", "orbit"):
            raise ValueError(f"leaves: 'dense', 'factored', 'sym' or 'orbit', not {leaves!r}")
        if leaves in ("sym", "orbit") and net is None:
            raise ValueError("symmetric leaves need a net")
        self.net, self.device, self.dtype, self.gamma, self.graph = net, device, dtype, gamma, graph
        self.leaves = leaves
        self.turn = turn or SubgameGpu(device=device, dtype=dtype)
        self.cache = cache
        self._tables: dict[tuple, tuple] = {}
        self._lock = threading.Lock()
        self.solves, self.seconds = 0, []
        self.symmetric = 0  # solves that took the symmetric path
        self.drift = 0.0  # the largest change the G-average made to a returned strategy

    def tables(self, board) -> tuple:
        key = tuple(int(c) for c in board)
        with self._lock:
            got = self._tables.get(key)
            if got is None:
                if len(self._tables) >= self.cache:
                    self._tables.clear()
                if self.net is not None and self.leaves == "factored":
                    from .flop_leaves import FactoredLeaves  # it imports this module

                    leaves = FactoredLeaves(key, self.net, self.device, last=torch.float64)  # in the net's dtype
                else:
                    evaluate = net_values(self.net) if self.net is not None else CheckdownValues(key, self.device, self.dtype)
                    leaves = NetLeaves(key, evaluate, self.device, self.dtype)
                sym = None
                if self.leaves in ("sym", "orbit") and len(flop_group(key)) > 1:
                    sym = SymLeaves(key, evaluate, self.device, self.dtype, cosets=self.leaves == "sym")
                got = self._tables[key] = (runout_equity(key, self.device, self.dtype), leaves, sym)
            return got

    def __call__(self, job: dict):
        if len(job["board"]) != 3:
            return self.turn(job)
        want = list(job["want"])
        with GPU_LOCK:
            t0 = time.perf_counter()
            s = equity = leaves = sym = None
            try:
                equity, leaves, sym = self.tables(job["board"])
                if sym is not None and not (sym.symmetric(job["ranges"]) and all(sym.symmetric(np.reshape(f, (-1, len(sym.canon_host))))
                                                                                for _, f in job["frozen"])):
                    sym = None  # a job G changes: the dense path
                s = FlopSolver(job, sym or leaves, equity, device=self.device, dtype=self.dtype, gamma=self.gamma)
                for slot, strategy in job["frozen"]:
                    s.freeze(slot, strategy)
                s.solve(int(job["iters"]), graph=self.graph)
                avg = s.average(torch.as_tensor(want, device=s.dev)).to(torch.float64).cpu().numpy()
                if sym is not None:
                    sym_avg = sym.average(avg)  # removes float-level asymmetry: exactly symmetric
                    self.drift = max(self.drift, float(np.abs(sym_avg - avg).max()))
                    avg = sym_avg
                    self.symmetric += 1
                dk = s.dk_host
            except BaseException as e:
                s = equity = leaves = sym = None  # a failed solve's solver and graph are freed here too
                drop_frames(e)
                raise
            # Freed under the lock (as SubgameGpu's): the solver's CUDA graph is reset as it goes, and a reset
            # while another match thread captures a graph invalidates that capture (RIVER R1 Amendment 2).
            s = equity = leaves = sym = None
            self.solves += 1
            self.seconds.append(time.perf_counter() - t0)
        return [avg[i, :dk[w]].reshape(-1).tolist() for i, w in enumerate(want)]
