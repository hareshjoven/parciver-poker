"""NumPy models of `gpu_fused`'s down pass and terminal kernels, and of the
candidates of KERNELS.md that change them (K1, K2, K3b, K4, Q3): each design's
algebra is checked here on the CPU before it is written in Triton
(tests/test_kernel_designs.py; `scripts/kernel_bench.py designs`).

Everything takes a `gpu_fused.host_plan` and works in the solver's rows
(decisions in slot order, chance nodes, folds, showdowns by view). The
"today" functions follow the kernels' expressions term by term; their float32
rounding is NumPy's (sequential prefix sums, pairwise sums), not the GPU's, so
float32 errors measured here are previews. The float64 runs are the test:
rule 5's tolerance against the dense matrices.
"""
from __future__ import annotations

import numpy as np


def uniform(plan, dtype=np.float32) -> np.ndarray:
    """[D, kmax, n]: regret matching on zero regrets, the first iteration's strategy."""
    k = np.asarray(plan.dk)
    sig = (np.arange(plan.kmax)[None, :] < k[:, None]) * (dtype(1.0) / k.astype(dtype))[:, None]
    return np.broadcast_to(sig[:, :, None].astype(dtype), (plan.D, plan.kmax, plan.n))


def skewed(plan, seed: int, power: float = 8.0, dtype=np.float32) -> np.ndarray:
    """[D, kmax, n]: a random strategy with most of each hand's weight on one
    action, as late iterations' are."""
    rng = np.random.default_rng(seed)
    x = rng.random((plan.D, plan.kmax, plan.n)) ** power
    x *= np.arange(plan.kmax)[None, :, None] < np.asarray(plan.dk)[:, None, None]
    return (x / x.sum(1, keepdims=True)).astype(dtype)


# ---------------------------------------------------------------- K1: reach rows aliased at the traverser's nodes


def reach_rows(plan, p: int):
    """K1's row table for traverser `p`. `rrow[row]`: the row that holds a
    node's opponent reach (itself, or its nearest ancestor that is the root or
    the child of an opponent or chance node: the traverser's own nodes pass
    the reach on unchanged). `levels`: per tree level, the decision rows and
    the chance nodes `_down` still runs on (the opponent's and chance)."""
    rrow = np.arange(plan.N)
    dk, dkid, dpl = np.asarray(plan.dk), np.asarray(plan.dkid), np.asarray(plan.dpl)
    levels = []
    for ld, md, lc, mc in plan.levels:  # from the root: a parent's entry is final before its children's
        ld, lc = np.asarray(ld)[:md], np.asarray(lc)[:mc]
        for d in ld[dpl[ld] == p]:
            rrow[dkid[d, :dk[d]]] = rrow[d]
        levels.append((ld[dpl[ld] != p], lc))
    return rrow, levels


def down(plan, p: int, strat, dtype=np.float32, alias: bool = False):
    """`_down` for traverser `p` at every level: the opponent's reach at each
    row under `strat` [D, kmax, n]. With `alias` (K1) only the children of
    opponent and chance nodes are written and every read goes through `rrow`.
    -> (rq [N, n], rrow, rows written)."""
    rq = np.zeros((plan.N, plan.n), dtype)
    rq[0] = np.asarray(plan.ranges[1 - p]).astype(dtype)
    alive = np.asarray(plan.alive).astype(dtype)
    dk, dkid, dpl = np.asarray(plan.dk), np.asarray(plan.dkid), np.asarray(plan.dpl)
    crow, ckid = np.asarray(plan.crow), np.asarray(plan.ckid)
    rrow = reach_rows(plan, p)[0] if alias else np.arange(plan.N)
    written = 0
    for ld, md, lc, mc in plan.levels:
        for d in np.asarray(ld)[:md]:
            kids = dkid[d, :dk[d]]
            if dpl[d] == p:
                if alias:
                    continue
                rq[kids] = rq[d]
            else:
                rq[kids] = rq[rrow[d]][None, :] * strat[d, :dk[d]]
            written += len(kids)
        for c in np.asarray(lc)[:mc]:
            rq[ckid[c]] = rq[rrow[crow[c]]][None, :] * alive
            written += len(ckid[c])
    return rq, rrow, written


# ---------------------------------------------------------------- folds


def fold_today(plan, p: int, r, dtype=np.float32):
    """`_fold` after the card-sum matmul: r [F, n] fold reach rows -> values."""
    r = r.astype(dtype)
    rc = r @ np.asarray(plan.inc).astype(dtype)  # [F, cards + 1]: per-card sums, the total last
    amt = np.asarray(plan.amt)[p, :plan.F].astype(dtype)
    return (rc[:, -1:] - rc[:, plan.ia] - rc[:, plan.ib] + r) * amt[:, None]


def fold_cards(plan, p: int, r, dtype=np.float32, sums=None):
    """K2: the per-card sums from each card's member list (a [cards, cards - 1]
    block of gathers summed along its rows) and the total from the row.
    `sums` (Q3): the dtype of the sums and their combination; the result is
    rounded to `dtype` once."""
    r = r.astype(dtype)
    sums = sums or dtype
    members = np.asarray(plan.clist)[0].reshape(-1, plan.L)  # any view's lists hold each card's hands
    card = r[:, members].sum(-1, dtype=sums)  # [F, cards]
    amt = np.asarray(plan.amt)[p, :plan.F].astype(dtype)
    return ((r.sum(-1, dtype=sums)[:, None] - card[:, plan.ia] - card[:, plan.ib] + r) * amt[:, None]).astype(dtype)


# ---------------------------------------------------------------- showdowns


def _prefix(x, dtype):
    """Inclusive prefix sums along the last axis with a leading 0 (scr[0] = 0)."""
    return np.concatenate([np.zeros(x.shape[:-1] + (1,), dtype), np.cumsum(x, -1, dtype=dtype)], -1)


def showdown(plan, r, dtype=np.float32, cards: bool = False, mask: bool = True, sums=None):
    """`_showdown`: r [S, n] showdown reach rows -> values. `cards` (K3b):
    each card's list has its own prefix sums, so a card's part is
    `P[la] + P[ha] - P[L]` on numbers no larger than that card's mass, instead
    of four reads of one prefix array over all the lists. `mask=False` (K4):
    the input is not multiplied by `alive` (it is already 0 on dead hands).
    `sums` (Q3): the dtype of the prefix sums and their combination; the
    result is rounded to `dtype` once."""
    r = r.astype(dtype)
    sums = sums or dtype
    n, L = plan.n, plan.L
    ia, ib = np.asarray(plan.ia), np.asarray(plan.ib)
    view = np.asarray(plan.sd_view)[:plan.S]
    m = np.asarray(plan.sd_m)[:plan.S].astype(dtype)
    out = np.zeros_like(r)
    for v in np.unique(view):
        at = np.flatnonzero(view == v)
        al = np.asarray(plan.alive)[v].astype(dtype)
        perm, clist = np.asarray(plan.perm)[v], np.asarray(plan.clist)[v]
        lo, hi, la, ha, lb, hb = np.asarray(plan.idx)[v]
        rs, rc = r[at][:, perm], r[at][:, clist]
        if mask:
            rs, rc = rs * al[perm], rc * al[clist]
        rs, rc = rs.astype(sums), rc.astype(sums)
        e1 = _prefix(rs, sums)
        full = e1[:, lo] + e1[:, hi] - e1[:, n:n + 1]
        if cards:
            P = _prefix(rc.reshape(len(at), -1, L), sums)  # [rows, cards, L + 1]
            part = (P[:, ia, la - ia * L] + P[:, ia, ha - ia * L] - P[:, ia, L]
                    + P[:, ib, lb - ib * L] + P[:, ib, hb - ib * L] - P[:, ib, L])
        else:
            e2 = _prefix(rc, sums)
            ca, cb = ia * L, ib * L
            part = (e2[:, la] + e2[:, ha] - e2[:, ca] - e2[:, ca + L]
                    + e2[:, lb] + e2[:, hb] - e2[:, cb] - e2[:, cb + L])
        out[at] = (full - part) * al * m[at][:, None]
    return out


# ---------------------------------------------------------------- the dense forms (the oracle)


def dense_fold(hands, plan, p: int, r) -> np.ndarray:
    return (r.astype(np.float64) @ hands.compat) * np.asarray(plan.amt)[p, :plan.F][:, None]  # compat is symmetric


def dense_showdown(hands, plan, r) -> np.ndarray:
    r = r.astype(np.float64)
    view = np.asarray(plan.sd_view)[:plan.S]
    out = np.zeros_like(r)
    for v in np.unique(view):
        at = np.flatnonzero(view == v)
        out[at] = (r[at] @ hands.showdown_matrix(int(v)).T) * np.asarray(plan.sd_m, dtype=np.float64)[at][:, None]
    return out


def scales(plan, p: int, fold_rows, showdown_rows):
    """Rule 2's scale per terminal row: the opponent's mass x the chips at stake."""
    f = fold_rows.astype(np.float64).sum(1) * np.abs(np.asarray(plan.amt)[p, :plan.F])
    s = showdown_rows.astype(np.float64).sum(1) * np.asarray(plan.sd_m, dtype=np.float64)[:plan.S]
    return f, s


def errors(got, want, scale) -> np.ndarray:
    """Rule 2's statistic per row: max_h |got - want| / scale (rows of scale 0 dropped)."""
    ok = scale > 0
    return np.abs(got.astype(np.float64) - want).max(1)[ok] / scale[ok]
