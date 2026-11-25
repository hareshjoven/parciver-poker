# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Fused GPU subgame solver: `gpu_search.GpuSolver`'s level-synchronous DCFR
as a few Triton kernels per tree level, with O(n) terminals and one CUDA graph
per iteration.

Why: the torch version spends ~90% of its GPU time moving [nodes, hands]
tensors through ~20 gather/elementwise passes per level (and the rest on n^2
terminal matmuls), and its wall time on small trees is kernel launches. Here:

* one kernel per level carries reaches down (regret matching, strategy-sum
  update and chance-card masking fused), one per level carries values up
  (strategy expectation, regret update, chance average fused);
* fold terminals use inclusion-exclusion over cards,
  `sum_{j disjoint from h} r_j = T - R[a] - R[b] + r_h`, with the per-card
  sums `R` from one [folds, hands] x [hands, cards] matmul;
* showdown terminals use prefix sums over hands sorted by strength, with the
  same card correction from per-card lists sorted by strength: O(n) per node
  instead of the dense `W_v @ r`;
* `terminals64=True` (KERNELS Q3, Q4): both sum and combine in float64
  (float32 solves: in pairs of float32, Q4; `q3=True` keeps Q3's float64)
  and round once at the store, so a float32 value is the exact one up to
  that rounding at every hand;
* `step()` replays a captured CUDA graph (the iteration count and discounts
  live on the device), so a small tree costs no Python per kernel.

Rows are renumbered (decisions in slot order, chance nodes, folds,
showdowns by view) so terminal blocks are contiguous; `regret`/`sum` keep
GpuSolver's [slot, action, hand] layout. Same algorithm, same oracle
(`gpu_search.RefSolver`).
"""
from __future__ import annotations

import gc
import threading
from types import SimpleNamespace

import numpy as np
import torch
import triton
import triton.language as tl

from .gpu_search import ALPHA, GAMMA, Hands, Tree

assert ALPHA == 1.5 and GAMMA == 2.0  # hard-coded in _prep (gamma: FusedSolver's option, default 2)

UPDATE, BEST, EXPECT = 0, 1, 2  # up-pass modes: CFR regret update, best response, plain expectation

# One solve on the GPU at a time. Torch releases the GIL, so the search's match
# threads would otherwise run CUDA work while another thread captures a graph
# (capture fails on any other work on the device).
GPU_LOCK = threading.RLock()


def drop_frames(e: BaseException) -> None:
    """Clear the frames an exception's traceback (and its causes' and contexts') keeps alive, so a solver and
    the CUDA graph it holds are freed now, under `GPU_LOCK`, not when the error is dropped outside it: a graph
    reset while another thread captures one invalidates that capture (RIVER R1 Amendment 2). The frame still
    running is skipped (`traceback.clear_frames`); its caller drops its own references."""
    import traceback

    stack, seen = [e], set()
    while stack:
        x = stack.pop()
        if x is None or id(x) in seen:
            continue
        seen.add(id(x))
        traceback.clear_frames(x.__traceback__)
        stack += [x.__cause__, x.__context__]


@triton.jit
def _prep(iter_ptr, scal_ptr, GAMMA: tl.constexpr, SHIFT: tl.constexpr, RESTART: tl.constexpr):
    """Count the iteration; scal = [positive-regret discount, strategy-sum
    discount (t / (t + 1)) ** GAMMA]. SHIFT: the regret discount's t is
    max(t - SHIFT, 0). RESTART: the sum's t counts from the last power of 4
    at or below t - 1, so the sum restarts there (`FusedSolver`)."""
    it = tl.load(iter_ptr) + 1
    tl.store(iter_ptr, it)
    t = it.to(tl.float64)
    if SHIFT == 0:
        d = t
    else:
        d = tl.maximum(t - SHIFT, 0.0)
    ta = d * tl.sqrt(d)
    tl.store(scal_ptr, ta / (ta + 1.0))
    if RESTART:
        tp = it - 1
        q = tp * 0 + 1
        for _ in tl.static_range(12):
            q = tl.where(q * 4 <= tp, q * 4, q)
        tg = tl.where(tp > 0, tp - q, 0).to(tl.float64)
        x = tg / (tg + 1.0)
    else:
        x = t / (t + 1.0)
    w = x
    for _ in tl.static_range(GAMMA - 1):
        w = w * x
    tl.store(scal_ptr + 1, w)


@triton.jit
def _wide(x):
    """Compute 16-bit storage in float32."""
    if tl.constexpr(x.dtype.is_fp16() or x.dtype.is_bf16()):
        return x.to(tl.float32)
    else:
        return x


@triton.jit
def _discount(r_ptr, scal_ptr, fz_ptr, total, per_slot, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < total
    m = m & (tl.load(fz_ptr + i // per_slot, mask=m, other=1) == 0)  # frozen slots hold their strategy
    r = _wide(tl.load(r_ptr + i, mask=m))
    pos = tl.load(scal_ptr).to(r.dtype)
    tl.store(r_ptr + i, tl.where(r > 0, r * pos, r * 0.5).to(r_ptr.dtype.element_ty), mask=m)


@triton.jit
def _strategy(s_ptr, d, a, hs, k, n, KS, m2, disc, scal_ptr, DISC: tl.constexpr):
    """Regret matching over a node's actions (on a strategy sum: its average).
    With DISC and `disc` (the node is not frozen), the regrets read are first
    discounted exactly as `_discount` would have stored them, which `step()`
    defers to each regret's first read of the iteration."""
    off = (d * KS + a[:, None]) * n + hs[None, :]
    x = _wide(tl.load(s_ptr + off, mask=m2, other=0.0))
    if DISC:
        if disc:
            f = tl.load(scal_ptr).to(x.dtype)
            x = _wide(tl.where(x > 0, x * f, x * 0.5).to(s_ptr.dtype.element_ty))
    pos = tl.maximum(x, 0.0)
    tot = tl.sum(pos, axis=0)
    sig = tl.where(tot[None, :] > 0, pos / tl.where(tot > 0, tot, 1.0)[None, :], 1.0 / k.to(x.dtype))
    return tl.where(m2, sig, 0.0), off, x


@triton.jit
def _down(rq_ptr, strat_ptr, sum_ptr, scal_ptr, dec_ptr, m_dec, dpl_ptr, dk_ptr, dkid_ptr, fz_ptr, keep_ptr,
          ch_ptr, crow_ptr, ckid_ptr, alive_ptr, V, n, KS, p,
          UPDATE_SUM: tl.constexpr, ALL_SUMS: tl.constexpr, DISC: tl.constexpr, KP: tl.constexpr,
          BLOCK: tl.constexpr):
    """One level down: the opponent's reach `rq` to every child. Decision rows
    are slots, so node `d` reads row `d`. A frozen slot (`fz`) holds its fixed
    strategy in both regret and sum, which are then never updated. Unless
    ALL_SUMS, only slots with `keep` set update their sums. DISC: the
    opponent's regrets are not yet discounted this iteration (`_strategy`)."""
    pid = tl.program_id(0)
    hs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mh = hs < n
    if pid < m_dec:
        d = tl.load(dec_ptr + pid).to(tl.int64)
        k = tl.load(dk_ptr + d)
        pl = tl.load(dpl_ptr + d)
        a = tl.arange(0, KP)
        m2 = (a[:, None] < k) & mh[None, :]
        q = tl.load(rq_ptr + d * n + hs, mask=mh, other=0.0)
        kids = tl.load(dkid_ptr + d * KS + a, mask=a < k, other=0).to(tl.int64)
        if pl == p:
            child = tl.broadcast_to(q[None, :], (KP, BLOCK))
        else:
            fz = tl.load(fz_ptr + d)
            sig, off, _x = _strategy(strat_ptr, d, a, hs, k, n, KS, m2, fz == 0, scal_ptr, DISC)
            child = q[None, :] * sig
            if UPDATE_SUM:
                if ALL_SUMS:
                    upd = fz == 0
                else:
                    upd = (fz == 0) & (tl.load(keep_ptr + d) != 0)
                if upd:
                    w = tl.load(scal_ptr + 1).to(q.dtype)
                    s = tl.load(sum_ptr + off, mask=m2, other=0.0)
                    tl.store(sum_ptr + off, s * w + child, mask=m2)
        tl.store(rq_ptr + kids[:, None] * n + hs[None, :], child, mask=m2)
    else:
        c = tl.load(ch_ptr + (pid - m_dec)).to(tl.int64)
        q = tl.load(rq_ptr + tl.load(crow_ptr + c).to(tl.int64) * n + hs, mask=mh, other=0.0)
        for cv in range(V):
            kid = tl.load(ckid_ptr + c * V + cv).to(tl.int64)
            al = tl.load(alive_ptr + cv * n + hs, mask=mh, other=0.0)
            tl.store(rq_ptr + kid * n + hs, q * al, mask=mh)


@triton.jit
def _up(val_ptr, strat_ptr, scal_ptr, dec_ptr, m_dec, dpl_ptr, dk_ptr, dkid_ptr, fz_ptr,
        ch_ptr, crow_ptr, ckid_ptr, alive_ptr, V, norm, n, KS, p,
        MODE: tl.constexpr, DISC: tl.constexpr, CLIP: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr):
    """One level up: node values from children; in UPDATE mode also the
    traverser's regrets (`strat_ptr` is then the regret tensor), discounted
    first if DISC (`_strategy`) and clipped at 0 after if CLIP (DCFR+)."""
    pid = tl.program_id(0)
    hs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mh = hs < n
    if pid < m_dec:
        d = tl.load(dec_ptr + pid).to(tl.int64)
        k = tl.load(dk_ptr + d)
        pl = tl.load(dpl_ptr + d)
        a = tl.arange(0, KP)
        m2 = (a[:, None] < k) & mh[None, :]
        kids = tl.load(dkid_ptr + d * KS + a, mask=a < k, other=0).to(tl.int64)
        vc = tl.load(val_ptr + kids[:, None] * n + hs[None, :], mask=m2, other=0.0)
        if pl == p:
            fz = tl.load(fz_ptr + d)
            sig, off, x = _strategy(strat_ptr, d, a, hs, k, n, KS, m2, fz == 0, scal_ptr, DISC)
            v = tl.sum(sig * vc, axis=0)
            if MODE == 1:  # a frozen node is not a choice for the best responder
                v = tl.where(fz != 0, v, tl.max(tl.where(m2, vc, float("-inf")), axis=0))
            if MODE == 0:
                r = x + (vc - v[None, :])
                if CLIP:
                    r = tl.maximum(r, 0.0)
                tl.store(strat_ptr + off, r.to(strat_ptr.dtype.element_ty), mask=m2 & (fz == 0))
        else:
            v = tl.sum(vc, axis=0)
        tl.store(val_ptr + d * n + hs, v, mask=mh)
    else:
        c = tl.load(ch_ptr + (pid - m_dec)).to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=val_ptr.dtype.element_ty)
        for cv in range(V):
            kid = tl.load(ckid_ptr + c * V + cv).to(tl.int64)
            al = tl.load(alive_ptr + cv * n + hs, mask=mh, other=0.0)
            acc += tl.load(val_ptr + kid * n + hs, mask=mh, other=0.0) * al
        tl.store(val_ptr + tl.load(crow_ptr + c).to(tl.int64) * n + hs, acc / norm, mask=mh)


@triton.jit
def _fold(rq_ptr, val_ptr, rc_ptr, amt_ptr, ia_ptr, ib_ptr, f_lo, F, n, RS, p, BLOCK: tl.constexpr):
    """Fold terminals: (C @ r)[h] = T - R[a_h] - R[b_h] + r_h, times the amount won."""
    f = tl.program_id(0).to(tl.int64)
    hs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mh = hs < n
    row = f_lo + f
    r = tl.load(rq_ptr + row * n + hs, mask=mh, other=0.0)
    base = rc_ptr + f * RS
    tot = tl.load(base + RS - 1)
    ra = tl.load(base + tl.load(ia_ptr + hs, mask=mh, other=0), mask=mh, other=0.0)
    rb = tl.load(base + tl.load(ib_ptr + hs, mask=mh, other=0), mask=mh, other=0.0)
    amt = tl.load(amt_ptr + p * F + f)
    tl.store(val_ptr + row * n + hs, (tot - ra - rb + r) * amt, mask=mh)


@triton.jit
def _fold64(rq_ptr, val_ptr, mem_ptr, amt_ptr, ia_ptr, ib_ptr, f_lo, F, n, NC, L, p,
            BN: tl.constexpr, BNC: tl.constexpr, BL: tl.constexpr):
    """`_fold` for `terminals64`, one node per program: (C @ r)[h] = T - R[a_h]
    - R[b_h] + r_h, times the amount won. The total T and the per-card sums R (over each card's
    L hands, `mem` = [NC, L] hand indices) are float64, and so is the
    combination: a float32 row's values are rounded once, at the store
    (KERNELS Q3: in float32 the cancellation leaves no relative precision
    where the value is small)."""
    f = tl.program_id(0).to(tl.int64)
    rbase = rq_ptr + (f_lo + f) * n
    i = tl.arange(0, BN)
    mi = i < n
    r = tl.load(rbase + i, mask=mi, other=0.0).to(tl.float64)
    c = tl.arange(0, BNC)[:, None]
    j = tl.arange(0, BL)[None, :]
    mc = (c < NC) & (j < L)
    mem = tl.load(mem_ptr + c * L + j, mask=mc, other=0)
    card = tl.sum(tl.load(rbase + mem, mask=mc, other=0.0).to(tl.float64), axis=1)
    ra = tl.gather(card, tl.load(ia_ptr + i, mask=mi, other=0), 0)
    rb = tl.gather(card, tl.load(ib_ptr + i, mask=mi, other=0), 0)
    amt = tl.load(amt_ptr + p * F + f).to(tl.float64)
    out = (tl.sum(r, axis=0) - ra - rb + r) * amt
    tl.store(val_ptr + (f_lo + f) * n + i, out.to(val_ptr.dtype.element_ty), mask=mi)


@triton.jit
def _showdown(rq_ptr, val_ptr, scr_ptr, perm_ptr, clist_ptr, idx_ptr, ia_ptr, ib_ptr, alive_ptr,
              view_ptr, m_ptr, s_lo, n, L, CL, SW, BN: tl.constexpr, BC: tl.constexpr):
    """Showdown terminals, one node per program: (W_v @ r)[h] = (weaker - stronger)
    over all hands minus the same over hands sharing a card with h (h itself
    ties, so it adds nothing). Sums come from inclusive prefix sums stored at
    scr[1..] (scr[0] = 0), over hands by strength and over per-card lists by
    strength; idx[v] = [lo, hi, la, ha, lb, hb] positions into them."""
    s = tl.program_id(0).to(tl.int64)
    row = s_lo + s
    v = tl.load(view_ptr + s).to(tl.int64)
    rbase = rq_ptr + row * n
    abase = alive_ptr + v * n
    e1 = scr_ptr + s * SW
    e2 = e1 + n + 1
    i = tl.arange(0, BN)
    mi = i < n
    pi = tl.load(perm_ptr + v * n + i, mask=mi, other=0)
    rs = tl.load(rbase + pi, mask=mi, other=0.0) * tl.load(abase + pi, mask=mi, other=0.0)
    tl.store(e1 + 1 + i, tl.cumsum(rs, axis=0), mask=mi)
    j = tl.arange(0, BC)
    mj = j < CL
    cj = tl.load(clist_ptr + v * CL + j, mask=mj, other=0)
    rc = tl.load(rbase + cj, mask=mj, other=0.0) * tl.load(abase + cj, mask=mj, other=0.0)
    tl.store(e2 + 1 + j, tl.cumsum(rc, axis=0), mask=mj)
    tl.debug_barrier()
    ib6 = idx_ptr + v * 6 * n
    lo = tl.load(ib6 + i, mask=mi, other=0)
    hi = tl.load(ib6 + n + i, mask=mi, other=0)
    full = tl.load(e1 + lo, mask=mi, other=0.0) + tl.load(e1 + hi, mask=mi, other=0.0) - tl.load(e1 + n)
    ca = tl.load(ia_ptr + i, mask=mi, other=0) * L
    cb = tl.load(ib_ptr + i, mask=mi, other=0) * L
    part = (tl.load(e2 + tl.load(ib6 + 2 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            + tl.load(e2 + tl.load(ib6 + 3 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            - tl.load(e2 + ca, mask=mi, other=0.0) - tl.load(e2 + ca + L, mask=mi, other=0.0)
            + tl.load(e2 + tl.load(ib6 + 4 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            + tl.load(e2 + tl.load(ib6 + 5 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            - tl.load(e2 + cb, mask=mi, other=0.0) - tl.load(e2 + cb + L, mask=mi, other=0.0))
    m = tl.load(m_ptr + s)
    tl.store(val_ptr + row * n + i, (full - part) * tl.load(abase + i, mask=mi, other=0.0) * m, mask=mi)


@triton.jit
def _showdown64(rq_ptr, val_ptr, scr_ptr, perm_ptr, clist_ptr, idx_ptr, ia_ptr, ib_ptr, alive_ptr,
                view_ptr, m_ptr, s_lo, n, L, CL, SW, BN: tl.constexpr, BC: tl.constexpr):
    """`_showdown` for `terminals64`: the same sums, with the sums, `scr` and
    the combination in float64; a float32 row's values are rounded once, at
    the store (as `_fold64`'s)."""
    s = tl.program_id(0).to(tl.int64)
    row = s_lo + s
    v = tl.load(view_ptr + s).to(tl.int64)
    rbase = rq_ptr + row * n
    abase = alive_ptr + v * n
    e1 = scr_ptr + s * SW
    e2 = e1 + n + 1
    i = tl.arange(0, BN)
    mi = i < n
    pi = tl.load(perm_ptr + v * n + i, mask=mi, other=0)
    rs = (tl.load(rbase + pi, mask=mi, other=0.0) * tl.load(abase + pi, mask=mi, other=0.0)).to(tl.float64)
    tl.store(e1 + 1 + i, tl.cumsum(rs, axis=0), mask=mi)
    j = tl.arange(0, BC)
    mj = j < CL
    cj = tl.load(clist_ptr + v * CL + j, mask=mj, other=0)
    rc = (tl.load(rbase + cj, mask=mj, other=0.0) * tl.load(abase + cj, mask=mj, other=0.0)).to(tl.float64)
    tl.store(e2 + 1 + j, tl.cumsum(rc, axis=0), mask=mj)
    tl.debug_barrier()
    ib6 = idx_ptr + v * 6 * n
    lo = tl.load(ib6 + i, mask=mi, other=0)
    hi = tl.load(ib6 + n + i, mask=mi, other=0)
    full = tl.load(e1 + lo, mask=mi, other=0.0) + tl.load(e1 + hi, mask=mi, other=0.0) - tl.load(e1 + n)
    ca = tl.load(ia_ptr + i, mask=mi, other=0) * L
    cb = tl.load(ib_ptr + i, mask=mi, other=0) * L
    part = (tl.load(e2 + tl.load(ib6 + 2 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            + tl.load(e2 + tl.load(ib6 + 3 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            - tl.load(e2 + ca, mask=mi, other=0.0) - tl.load(e2 + ca + L, mask=mi, other=0.0)
            + tl.load(e2 + tl.load(ib6 + 4 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            + tl.load(e2 + tl.load(ib6 + 5 * n + i, mask=mi, other=0), mask=mi, other=0.0)
            - tl.load(e2 + cb, mask=mi, other=0.0) - tl.load(e2 + cb + L, mask=mi, other=0.0))
    m = tl.load(m_ptr + s).to(tl.float64)
    out = (full - part) * tl.load(abase + i, mask=mi, other=0.0).to(tl.float64) * m
    tl.store(val_ptr + row * n + i, out.to(val_ptr.dtype.element_ty), mask=mi)


# KERNELS Q4: Q3's float64 sums as unevaluated pairs (hi, lo) of float32,
# |lo| <= ulp(hi)/2, for float32 solvers (float64 runs at 1/64 of float32's
# rate on this GPU). Error-free transformations: IEEE float32 adds in program
# order. Triton does not reassociate, the launches turn FMA contraction off
# (`enable_fp_fusion=False`, `_FF`), and the product with the amount is an
# explicit FMA (`_ff_round_mul`).
@triton.jit
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def _fast_two_sum(a, b):  # |a| >= |b| (or a = 0)
    s = a + b
    return s, b - (s - a)


@triton.jit
def _ff_add_pos(ah, al, bh, bl):
    """Pair add for terms of one sign (the reach sums): one TwoSum and the low parts."""
    s, e = _two_sum(ah, bh)
    return _fast_two_sum(s, e + (al + bl))


@triton.jit
def _ff_add(ah, al, bh, bl):
    """Pair add, any signs (AccurateDWPlusDW, Joldes, Muller, Popescu 2017): relative error <= 3u^2."""
    s, e = _two_sum(ah, bh)
    t, f = _two_sum(al, bl)
    s, e = _fast_two_sum(s, e + t)
    return _fast_two_sum(s, e + f)


@triton.jit
def _ff_round_mul(h, l, m):
    """The pair (h, l) times the float32 m, rounded once: RN(h*m + RN(l*m)) in
    one FMA. The small product's own rounding (<= 2^-48 |h m| while l*m is
    normal) can move only a near-tie of the one rounding (a double rounding,
    still within 2^-24 |(h + l) m|). With contraction on, the TwoProd form p + (fma(h, m, -p) + l*m)
    compiles to fma(h, m, fma(l, m, fma(h, m, -p))) and counts the product's
    error twice (Q4 audit, 6 Oct); this form has nothing left to contract."""
    return tl.fma(h, m, l * m)


@triton.jit
def _ld2(base, SW, off, mask):
    """A pair from `scr`'s hi plane at `off` and its lo plane at `SW + off`."""
    return tl.load(base + off, mask=mask, other=0.0), tl.load(base + SW + off, mask=mask, other=0.0)


_FF = dict(enable_fp_fusion=False)  # Q4's launches: no FMA contraction in the error-free transformations


@triton.jit
def _fold_ff(rq_ptr, val_ptr, mem_ptr, amt_ptr, ia_ptr, ib_ptr, f_lo, F, n, NC, L, p,
             BN: tl.constexpr, BNC: tl.constexpr, BL: tl.constexpr):
    """`_fold64` with pairs of float32: T and R as pairs (nonnegative sums), the
    combination T - R[a] - R[b] + r as a pair, rounded once with the amount."""
    f = tl.program_id(0).to(tl.int64)
    rbase = rq_ptr + (f_lo + f) * n
    i = tl.arange(0, BN)
    mi = i < n
    r = tl.load(rbase + i, mask=mi, other=0.0)
    c = tl.arange(0, BNC)[:, None]
    j = tl.arange(0, BL)[None, :]
    mc = (c < NC) & (j < L)
    mem = tl.load(mem_ptr + c * L + j, mask=mc, other=0)
    x = tl.load(rbase + mem, mask=mc, other=0.0)
    ch, cl = tl.reduce((x, tl.zeros_like(x)), 1, _ff_add_pos)
    th, tl_ = tl.reduce((r, tl.zeros_like(r)), 0, _ff_add_pos)
    ga = tl.load(ia_ptr + i, mask=mi, other=0)
    gb = tl.load(ib_ptr + i, mask=mi, other=0)
    h, l = _ff_add(th + tl.zeros_like(r), tl_ + tl.zeros_like(r), -tl.gather(ch, ga, 0), -tl.gather(cl, ga, 0))
    h, l = _ff_add(h, l, -tl.gather(ch, gb, 0), -tl.gather(cl, gb, 0))
    h, l = _ff_add(h, l, r, tl.zeros_like(r))
    out = _ff_round_mul(h, l, tl.load(amt_ptr + p * F + f))
    tl.store(val_ptr + (f_lo + f) * n + i, out, mask=mi)


@triton.jit
def _showdown_ff(rq_ptr, val_ptr, scr_ptr, perm_ptr, clist_ptr, idx_ptr, ia_ptr, ib_ptr, alive_ptr,
                 view_ptr, m_ptr, s_lo, n, L, CL, SW, BN: tl.constexpr, BC: tl.constexpr):
    """`_showdown64` with pairs of float32: both prefix sums as pairs (`scr`'s
    rows are [hi plane | lo plane], SW each), `full`, `part` and their
    difference as pairs, rounded once with the amount."""
    s = tl.program_id(0).to(tl.int64)
    row = s_lo + s
    v = tl.load(view_ptr + s).to(tl.int64)
    rbase = rq_ptr + row * n
    abase = alive_ptr + v * n
    e1 = scr_ptr + s * (2 * SW)
    e2 = e1 + n + 1
    i = tl.arange(0, BN)
    mi = i < n
    pi = tl.load(perm_ptr + v * n + i, mask=mi, other=0)
    rs = tl.load(rbase + pi, mask=mi, other=0.0) * tl.load(abase + pi, mask=mi, other=0.0)
    sh, sl = tl.associative_scan((rs, tl.zeros_like(rs)), 0, _ff_add_pos)
    tl.store(e1 + 1 + i, sh, mask=mi)
    tl.store(e1 + SW + 1 + i, sl, mask=mi)
    j = tl.arange(0, BC)
    mj = j < CL
    cj = tl.load(clist_ptr + v * CL + j, mask=mj, other=0)
    rc = tl.load(rbase + cj, mask=mj, other=0.0) * tl.load(abase + cj, mask=mj, other=0.0)
    ch, cl = tl.associative_scan((rc, tl.zeros_like(rc)), 0, _ff_add_pos)
    tl.store(e2 + 1 + j, ch, mask=mj)
    tl.store(e2 + SW + 1 + j, cl, mask=mj)
    tl.debug_barrier()
    ib6 = idx_ptr + v * 6 * n
    ah, al = _ld2(e1, SW, tl.load(ib6 + i, mask=mi, other=0), mi)
    bh, bl = _ld2(e1, SW, tl.load(ib6 + n + i, mask=mi, other=0), mi)
    th, tl_ = _ld2(e1, SW, n + tl.zeros_like(i), mi)
    fh, fl = _ff_add(ah, al, bh, bl)
    fh, fl = _ff_add(fh, fl, -th, -tl_)  # full
    ca = tl.load(ia_ptr + i, mask=mi, other=0) * L
    cb = tl.load(ib_ptr + i, mask=mi, other=0) * L
    # full - part: part's added terms subtracted, its subtracted terms added
    for k in tl.static_range(2, 6):
        xh, xl = _ld2(e2, SW, tl.load(ib6 + k * n + i, mask=mi, other=0), mi)
        fh, fl = _ff_add(fh, fl, -xh, -xl)
    xh, xl = _ld2(e2, SW, ca, mi)
    fh, fl = _ff_add(fh, fl, xh, xl)
    xh, xl = _ld2(e2, SW, ca + L, mi)
    fh, fl = _ff_add(fh, fl, xh, xl)
    xh, xl = _ld2(e2, SW, cb, mi)
    fh, fl = _ff_add(fh, fl, xh, xl)
    xh, xl = _ld2(e2, SW, cb + L, mi)
    fh, fl = _ff_add(fh, fl, xh, xl)
    # Times alive (0 or 1) after the rounding: exact, and a dead holding's zero
    # keeps its value's sign, as Q3's does (hi * 0 + lo * 0 would not).
    out = _ff_round_mul(fh, fl, tl.load(m_ptr + s)) * tl.load(abase + i, mask=mi, other=0.0)
    tl.store(val_ptr + row * n + i, out, mask=mi)


def showdown_index(hands: Hands):
    """Per view: hands by strength (`perm`), per-card hand lists by strength
    (`clist`, [cards, cards-1] flattened), and each hand's tie-range positions
    in both (`idx` = [lo, hi, la, ha, lb, hb], prefix-sum positions)."""
    n = hands.n
    used = set(hands.board)
    cards = [c for c in range(52) if c not in used]
    cid = {c: i for i, c in enumerate(cards)}
    NC = len(cards)
    L = NC - 1
    ia = np.array([cid[int(a)] for a, _ in hands.combos])
    ib = np.array([cid[int(b)] for _, b in hands.combos])
    members = [[] for _ in range(NC)]
    for h in range(n):
        members[ia[h]].append(h)
        members[ib[h]].append(h)
    M = np.array(members)  # [NC, L]
    assert M.shape == (NC, L)
    V = hands.views
    perm = np.zeros((V, n), dtype=np.int64)
    clist = np.zeros((V, NC * L), dtype=np.int64)
    idx = np.zeros((V, 6, n), dtype=np.int64)
    for v in range(V):
        s = hands.strength[v]
        perm[v] = np.argsort(s, kind="stable")
        ss = s[perm[v]]
        idx[v, 0] = np.searchsorted(ss, s, "left")
        idx[v, 1] = np.searchsorted(ss, s, "right")
        sm = s[M]
        o = np.argsort(sm, axis=1, kind="stable")
        Ms = np.take_along_axis(M, o, 1)
        sms = np.take_along_axis(sm, o, 1)
        clist[v] = Ms.reshape(-1)
        for c in range(NC):
            mem = M[c]
            left = c * L + np.searchsorted(sms[c], s[mem], "left")
            right = c * L + np.searchsorted(sms[c], s[mem], "right")
            first = ia[mem] == c
            idx[v, 2, mem[first]] = left[first]
            idx[v, 3, mem[first]] = right[first]
            idx[v, 4, mem[~first]] = left[~first]
            idx[v, 5, mem[~first]] = right[~first]
    return dict(cards=cards, ia=ia, ib=ib, L=L, perm=perm, clist=clist, idx=idx)


def _pow2(x: int) -> int:
    return 1 << max(0, int(x) - 1).bit_length()


def host_plan(hands: Hands, tree, ranges, scaled: bool = False) -> SimpleNamespace:
    """`FusedSolver`'s host-side preparation of a subgame: every numpy array
    it uploads, with no GPU work, so a caller can build it outside `GPU_LOCK`
    while another thread's solve runs. `scaled`: the solve stores 16-bit
    regrets (`FusedSolver`'s `regret_dtype`), so ranges and chips are scaled."""
    t = flatten(tree) if isinstance(tree, Tree) else tree
    kind = np.asarray(t["kind"], dtype=np.int64)
    player = np.asarray(t["player"], dtype=np.int64)
    slot = np.asarray(t["slot"], dtype=np.int64)
    stake = np.asarray(t["stake"], dtype=np.float64).reshape(-1, 2)
    view = np.asarray(t["view"], dtype=np.int64)
    start = np.asarray(t["child_start"], dtype=np.int64)
    children = np.asarray(t["children"], dtype=np.int64)
    N, n, V = len(kind), hands.n, hands.views
    cnt = np.diff(start)
    dec = np.nonzero(kind == 1)[0]
    dec = dec[np.argsort(slot[dec])]
    chance = np.nonzero(kind == 2)[0]
    folds = np.nonzero((kind == 0) & (player >= 0))[0]
    sds = np.nonzero((kind == 0) & (player < 0))[0]
    sds = sds[np.argsort(view[sds], kind="stable")]
    assert kind[0] == 1 and slot[0] == 0 and np.array_equal(slot[dec], np.arange(len(dec)))
    row = np.empty(N, dtype=np.int64)
    row[np.concatenate([dec, chance, folds, sds])] = np.arange(N)
    D, C, F, S = len(dec), len(chance), len(folds), len(sds)

    def kid_rows(nodes, width):
        j = np.arange(width)[None, :]
        ok = j < cnt[nodes][:, None]
        at = np.where(ok, start[nodes][:, None] + j, 0)
        return np.where(ok, row[children[np.minimum(at, max(len(children) - 1, 0))]], 0)

    kmax = int(cnt[dec].max())
    if C:
        assert (cnt[chance] == V).all()
    # Depth by level from the root (a parent precedes its children).
    depth = np.zeros(N, dtype=np.int64)
    frontier = np.array([0])
    lv = 0
    while len(frontier):
        depth[frontier] = lv
        c = cnt[frontier]
        frontier = children[np.repeat(start[frontier] - np.cumsum(c) + c, c) + np.arange(c.sum())]
        lv += 1
    max_depth = lv - 1
    levels = []
    for lv in range(max_depth + 1):
        ld = np.nonzero(depth[dec] == lv)[0]
        lc = np.nonzero(depth[chance] == lv)[0]
        if len(ld) + len(lc):
            levels.append((ld if len(ld) else np.zeros(1, np.int64), len(ld),
                           lc if len(lc) else np.zeros(1, np.int64), len(lc)))
    # Terminals.
    sx = getattr(hands, "_showdown_index", None) or showdown_index(hands)
    hands._showdown_index = sx
    NC = len(sx["cards"])
    amt = np.zeros((2, max(F, 1)))
    for p in (0, 1):
        amt[p, :F] = np.where(player[folds] == p, -stake[folds, p], stake[folds, 1 - p])
    chip = 1.0
    if scaled:
        # The starting pot: the smallest terminal pot (the check-down line has it).
        chip = float(stake[kind == 0].sum(1).min())
        ranges = [np.asarray(r, dtype=np.float64) / np.sum(r) for r in ranges]
        amt /= chip
    inc = np.zeros((n, NC + 1))
    inc[np.arange(n), sx["ia"]] = 1.0
    inc[np.arange(n), sx["ib"]] = 1.0
    inc[:, NC] = 1.0
    return SimpleNamespace(
        scaled=scaled, N=N, n=n, V=V, D=D, F=F, S=S, f_lo=D + C, s_lo=D + C + F,
        term_nodes=np.concatenate([folds, sds]).tolist(),  # tree indices of rows f_lo.. (terminal_values' order)
        kmax=kmax, dk=cnt[dec], dkid=kid_rows(dec, kmax), dpl=player[dec],
        crow=row[chance] if C else np.zeros(1, np.int64),
        ckid=kid_rows(chance, V) if C else np.zeros((1, V), np.int64),
        norm=float(t["norm"]) if C else 1.0, max_depth=max_depth, levels=levels,
        amt=amt, chip=chip, inc=inc, ia=sx["ia"], ib=sx["ib"],
        sd_view=view[sds] if S else [0], sd_m=stake[sds].min(1) / chip if S else [0.0],
        L=sx["L"], CL=NC * sx["L"], perm=sx["perm"], clist=sx["clist"], idx=sx["idx"], alive=hands.alive,
        ranges=[np.asarray(r) for r in ranges],
        joint=float(np.asarray(ranges[0], dtype=np.float64) @ hands.compatible_mass(ranges[1])))


class FusedSolver:
    """GpuSolver's DCFR in fused Triton kernels (CUDA only)."""

    def __init__(self, hands: Hands, tree, ranges, device: str = "cuda", dtype=None,
                 block: int = 256, graph: bool = True, regret_dtype=None, plan: SimpleNamespace | None = None,
                 gamma: int = 2, plus: bool = False, shift: int | None = None, restart: bool = False,
                 sums=None, terminals64: bool = False, q3: bool = False):
        """`tree`: a `gpu_search.Tree`, or a subgame in the Rust search's flat
        form (a dict with its node arrays, see `flatten`). `regret_dtype`: store
        regrets in 16 bits (torch.float16/bfloat16; math stays in `dtype`); the
        solve then runs on ranges scaled to mass 1 and chips in units of the
        starting pot, so values fit float16's range (strategies are unchanged
        by either scaling; reported values are converted back). `plan`: this
        subgame's `host_plan`, if the caller built it already. `gamma`: the
        strategy sum's DCFR discount exponent (an integer; 2 is search.rs's).
        `plus`, `shift`, `restart`: other schedules (`gpu_search.RefSolver`'s):
        DCFR+ (Xu et al. 2024); the regret discount's t shifted back by
        `shift` (default 1 with `plus`, else 0); the strategy sum restarted at
        every power of 4 (postflop-solver's is shift 2, restart, gamma 3).
        `sums`: keep the strategy sums of these slots only (a search reads back
        a few averages; the others' sum updates are ~12% of the memory
        traffic). Their averages are unchanged bit for bit; `average`,
        `exploitability` and `value` refuse to read sums that were not kept.
        `terminals64`: fold and showdown sums carried past float32 and rounded
        once, for offline data: float32 sums lose the relative
        precision of small values, which regret matching follows on hard rows
        (records/d1-tail). Float64 solvers sum in float64; float32 solvers in
        pairs of float32 (KERNELS Q4, ~48 bits), where Q3's float64 sums cost
        ~4x on this GPU's float64 rate. `q3`: float32 `terminals64` solves
        keep Q3's float64 kernels (results labelled Q3 pin them)."""
        if int(gamma) != gamma or gamma < 1:
            raise ValueError(f"gamma must be a positive integer, got {gamma}")
        shift = (1 if plus else 0) if shift is None else shift
        if int(shift) != shift or shift < 0:
            raise ValueError(f"shift must be a non-negative integer, got {shift}")
        self.gamma = int(gamma)
        self.plus, self.shift, self.restart = bool(plus), int(shift), bool(restart)
        self.dev = torch.device(device)
        self.dtype = dtype or torch.float32
        self.block = block
        self.use_graph = graph and self.dev.type == "cuda"
        self.graph = None
        f = dict(device=self.dev, dtype=self.dtype)
        i32 = dict(device=self.dev, dtype=torch.int32)
        h = plan if plan is not None else host_plan(hands, tree, ranges, scaled=regret_dtype is not None)
        assert h.scaled == (regret_dtype is not None), "the plan's scaling does not match regret_dtype"
        N, n, D = h.N, h.n, h.D
        self.N, self.n, self.V = N, n, h.V
        self.term_nodes = h.term_nodes
        self.kmax = h.kmax
        self.KP = max(2, _pow2(self.kmax))
        self.dk_host = h.dk
        self.dkid = torch.as_tensor(h.dkid, **i32)
        self.dk = torch.as_tensor(h.dk, **i32)
        self.dpl = torch.as_tensor(h.dpl, **i32)
        self.dfz = torch.zeros(D, **i32)
        if sums is not None:
            sums = list(sums)
            bad = [s for s in sums if s not in range(D)]
            if bad:
                raise ValueError(f"sum slots must lie in [0, {D}), got {bad}")
        self.sum_slots = None if sums is None or len(set(sums)) == D else frozenset(int(s) for s in sums)
        self.dkeep = None
        if self.sum_slots is not None:
            self.dkeep = torch.zeros(D, **i32)
            self.dkeep[sorted(self.sum_slots)] = 1
        self.crow = torch.as_tensor(h.crow, **i32)
        self.ckid = torch.as_tensor(h.ckid, **i32)
        self.norm = h.norm
        self.max_depth = h.max_depth
        self.levels = [(torch.as_tensor(ld, **i32), md, torch.as_tensor(lc, **i32), mc) for ld, md, lc, mc in h.levels]
        # Terminals.
        self.f_lo, self.F = h.f_lo, h.F
        self.regret_dtype = regret_dtype or self.dtype
        self.chip = h.chip
        self.amt = torch.as_tensor(h.amt, **f)
        self.terminals64 = bool(terminals64)
        if not self.terminals64:
            self.inc = torch.as_tensor(h.inc, **f)
            self.rc = torch.zeros(max(h.F, 1), h.inc.shape[1], **f)
        self.ia = torch.as_tensor(h.ia, **i32)
        self.ib = torch.as_tensor(h.ib, **i32)
        self.s_lo, self.S = h.s_lo, h.S
        self.sd_view = torch.as_tensor(h.sd_view, **i32)
        self.sd_m = torch.as_tensor(h.sd_m, **f)
        self.L, self.CL = h.L, h.CL
        self.SW = n + 1 + self.CL + 1
        # KERNELS Q4: float32 solves with `terminals64` sum in pairs of float32
        # (`scr` rows: hi plane, lo plane); float64 solves keep Q3's kernels.
        self.ff = self.terminals64 and self.dtype == torch.float32 and not q3
        self.scr = torch.zeros(max(h.S, 1), 2 * self.SW if self.ff else self.SW, device=self.dev,
                               dtype=torch.float64 if self.terminals64 and not self.ff else self.dtype)
        self.perm = torch.as_tensor(h.perm, **i32)
        self.clist = torch.as_tensor(h.clist, **i32)
        self.idx = torch.as_tensor(h.idx, **i32)
        self.alive = torch.as_tensor(h.alive, **f)
        self.ranges = [torch.as_tensor(r, **f) for r in h.ranges]
        self.joint = h.joint
        self.mask = (torch.arange(self.kmax, device=self.dev)[None, :] < self.dk[:, None]).to(self.dtype)[:, :, None]
        # State.
        self.rq = torch.zeros(N, n, **f)
        self.val = torch.zeros(N, n, **f)
        self.regret = torch.zeros(D, self.kmax, n, device=self.dev, dtype=self.regret_dtype)
        self.sum = torch.zeros(D, self.kmax, n, **f)
        self.iter_t = torch.zeros(1, dtype=torch.int64, device=self.dev)
        self.scal = torch.zeros(2, dtype=torch.float64, device=self.dev)
        self.iter = 0
    # ---------------------------------------------------------------- passes

    def _hb(self):
        return triton.cdiv(self.n, self.block)

    def _down(self, p: int, strat, update_sum: bool, disc: bool = False):
        self.rq[0].copy_(self.ranges[1 - p])
        keep = self.dfz if self.dkeep is None else self.dkeep  # not read when every sum is kept
        for ld, md, lc, mc in self.levels:
            _down[(md + mc, self._hb())](self.rq, strat, self.sum, self.scal, ld, md, self.dpl, self.dk, self.dkid, self.dfz,
                                         keep, lc, self.crow, self.ckid, self.alive, self.V, self.n, self.kmax, p,
                                         UPDATE_SUM=update_sum, ALL_SUMS=self.dkeep is None, DISC=disc, KP=self.KP,
                                         BLOCK=self.block)

    def _terminals(self, p: int):
        if self.terminals64:
            if self.F:  # each card's hands: any view's per-card lists (`clist[0]`)
                (_fold_ff if self.ff else _fold64)[(self.F,)](
                    self.rq, self.val, self.clist, self.amt, self.ia, self.ib, self.f_lo, self.F, self.n, self.L + 1,
                    self.L, p, BN=_pow2(self.n), BNC=_pow2(self.L + 1), BL=_pow2(self.L), num_warps=8,
                    **(_FF if self.ff else {}))
        elif self.F:
            torch.matmul(self.rq[self.f_lo:self.f_lo + self.F], self.inc, out=self.rc)
            _fold[(self.F, self._hb())](self.rq, self.val, self.rc, self.amt, self.ia, self.ib, self.f_lo, self.F,
                                        self.n, self.rc.shape[1], p, BLOCK=self.block)
        if self.S:
            sd = (_showdown_ff if self.ff else _showdown64) if self.terminals64 else _showdown
            sd[(self.S,)](self.rq, self.val, self.scr, self.perm, self.clist, self.idx, self.ia, self.ib,
                          self.alive, self.sd_view, self.sd_m, self.s_lo, self.n, self.L, self.CL, self.SW,
                          BN=_pow2(self.n), BC=_pow2(self.CL), num_warps=8, **(_FF if self.ff else {}))

    def _up(self, p: int, strat, mode: int, disc: bool = False):
        for ld, md, lc, mc in reversed(self.levels):
            _up[(md + mc, self._hb())](self.val, strat, self.scal, ld, md, self.dpl, self.dk, self.dkid, self.dfz, lc,
                                       self.crow, self.ckid, self.alive, self.V, self.norm, self.n, self.kmax, p,
                                       MODE=mode, DISC=disc, CLIP=self.plus, KP=self.KP, BLOCK=self.block)

    def _begin_device(self, discount: bool = True):
        _prep[(1,)](self.iter_t, self.scal, GAMMA=self.gamma, SHIFT=self.shift, RESTART=self.restart)
        if discount:
            total = self.regret.numel()
            _discount[(triton.cdiv(total, 4096),)](self.regret, self.scal, self.dfz, total, self.kmax * self.n, BLOCK=4096)

    def _traverse_device(self, p: int, disc_opponent: bool = False, disc_own: bool = False):
        self._down(p, self.regret, True, disc_opponent)
        self._terminals(p)
        self._up(p, self.regret, UPDATE, disc_own)

    def _step_device(self):
        # Bit-identical to begin(); traverse(0); traverse(1) without the pass
        # over every regret: each regret is discounted at its first read of the
        # iteration instead (regret matching reads the discounted value, the
        # update stores it). Player 1's regrets are first read by player 0's
        # down pass, and discounted in store only by player 1's up pass;
        # player 0's are read and stored by player 0's up pass.
        self._begin_device(discount=False)
        self._traverse_device(0, disc_opponent=True, disc_own=True)
        self._traverse_device(1, disc_opponent=False, disc_own=True)

    # ---------------------------------------------------------------- API (GpuSolver's)

    def freeze(self, slot: int, strategy):
        """Play `strategy` ([actions, hands], each column a distribution) at
        `slot`: no regret or strategy-sum updates there, and the best responder
        plays it too (search.rs `Solver::freeze`). Before the first step."""
        s = torch.as_tensor(np.asarray(strategy), device=self.dev, dtype=self.dtype).reshape(-1, self.n)
        if s.shape[0] != self.dk_host[slot]:
            raise ValueError(f"slot {slot} has {self.dk_host[slot]} actions, the strategy {s.shape[0]}")
        self.regret[slot].zero_()
        self.regret[slot, :s.shape[0]] = s
        self.sum[slot].copy_(self.regret[slot])
        self.dfz[slot] = 1

    def begin(self):
        """Start an iteration: count it and discount the regrets."""
        self.iter += 1
        self._begin_device()

    def traverse(self, p: int):
        """One player's half of the iteration (alternating updates)."""
        self._traverse_device(p)

    def step(self):
        self.iter += 1
        if not self.use_graph:
            self._step_device()
            return
        if self.graph is None:
            # A real step on a side stream compiles every kernel and warms
            # cuBLAS; capture then records (without running) the next one.
            side = torch.cuda.Stream(self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):
                self._step_device()
            torch.cuda.current_stream(self.dev).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            # No garbage collection while capturing: freeing a Triton kernel
            # unloads its module, which the capture forbids.
            with GPU_LOCK:
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(self.graph):
                        self._step_device()
                finally:
                    if enabled:
                        gc.enable()
            return
        self.graph.replay()

    def _need_sums(self, slots=None):
        """Refuse to read strategy sums the solve did not keep (`sums`); all of
        them when `slots` is None."""
        if self.sum_slots is None:
            return
        D = len(self.dk_host)
        if slots is None:
            asked = set(range(D))
        elif isinstance(slots, list) and all(type(s) is int for s in slots):  # the search's `want`: no tensor work
            asked = {s + D if s < 0 else s for s in slots}
        else:
            if torch.is_tensor(slots):
                slots = slots.cpu()  # one transfer, not one per slot
            # The slots `self.sum[slots]` reads, however they are given (tensor, bool mask, slice, `...`): index
            # slot numbers shaped like `sum`, its action and hand axes of size 1. That counts the slots only of
            # an index that leaves those axes whole (`1:` there would pick nothing here, all of them in `sum`),
            # so any other index is refused; its shape is checked on a meta tensor (no memory).
            at = torch.arange(D).view(D, 1, 1)
            picked = at[slots]
            read = torch.empty(self.sum.shape, device="meta")[slots]
            if picked.shape[-2:] != (1, 1) or read.shape[-2:] != self.sum.shape[1:]:
                raise ValueError("this solve kept some strategy sums only: index its slots only, "
                                 f"not {slots!r:.80}")
            asked = set(picked.reshape(-1).tolist())
        missing = sorted(asked - self.sum_slots)
        if missing:
            raise ValueError(f"this solve kept the strategy sums of slots {sorted(self.sum_slots)} only, "
                             f"not {missing[:8]}")

    def average(self, slots=None):
        """The average strategy, [slot, action, hand]; only the rows of
        `slots`, in that order, if given (a search reads back a few)."""
        self._need_sums(slots)
        s = self.sum if slots is None else self.sum[slots]
        m = self.mask if slots is None else self.mask[slots]
        tot = s.sum(1, keepdim=True)
        kc = m.sum(1, keepdim=True)
        return torch.where(tot > 0, s / tot.clamp(min=torch.finfo(self.dtype).tiny), m / kc)

    def exploitability(self) -> float:
        """Both players' best-response values against the average strategy, in
        chips per dealt pair (search.rs `Solver::exploitability`)."""
        self._need_sums()
        total = 0.0
        for p in (0, 1):
            self._down(p, self.sum, False)
            self._terminals(p)
            self._up(p, self.sum, BEST)
            total += float(self.ranges[p] @ self.val[0])
        return total / self.joint * self.chip

    def value(self, p: int) -> float:
        """Value to `p` of the average-strategy profile, chips per dealt pair."""
        self._need_sums()
        self._down(p, self.sum, False)
        self._terminals(p)
        self._up(p, self.sum, EXPECT)
        return float(self.ranges[p] @ self.val[0]) / self.joint * self.chip

    def terminal_values(self, p: int, rq_terminals):
        """Test hook: fold/showdown values for given terminal reaches, in the
        tree's terminal order (folds, then showdowns by view)."""
        self.rq[self.f_lo:].copy_(rq_terminals)
        self._terminals(p)
        return self.val[self.f_lo:].clone()


# ---------------------------------------------------------------- the search's subgames (engine/src/subgame.rs)


def flatten(tree: Tree) -> dict:
    """A `gpu_search.Tree`'s nodes in the Rust search's flat form."""
    kind, player, slot, stake, view, start, children = [], [], [], [], [], [0], []
    for x in tree.nodes:
        kind.append({"T": 0, "D": 1, "C": 2}[x.kind])
        player.append(x.player if x.kind == "D" else (x.folded if x.kind == "T" and x.folded is not None else -1))
        slot.append(x.slot if x.kind == "D" else -1)
        stake.append(tuple(x.stake))
        view.append(x.view)
        children.extend(x.children)
        start.append(len(children))
    norm = next((x.norm for x in tree.nodes if x.kind == "C"), 1.0)
    return dict(kind=kind, player=player, slot=slot, stake=stake, view=view, child_start=start,
                children=children, norm=norm)

def job_from_tree(hands: Hands, tree: Tree, ranges, iters: int, want, frozen=()) -> dict:
    """The flat job the Rust search would send for `tree` (tests, benchmarks)."""
    return dict(flatten(tree), board=list(hands.board),
                strength=hands.strength.reshape(-1).tolist(), alive=hands.alive.reshape(-1).astype(int).tolist(),
                ranges=(list(ranges[0]), list(ranges[1])), frozen=[(s, list(np.ravel(st))) for s, st in frozen],
                iters=iters, want=list(want))


class SubgameGpu:
    """The real-time search's subgame solver on the GPU, as the callable
    `PlayAgent.with_search(solver=...)` takes: solves a flat subgame from the
    Rust search for its `iters` DCFR iterations and returns the average
    strategy of each slot in `want` (action-major lists). Card views are
    cached per board (a hand's turn and river decisions reuse them).
    `terminals64`, `q3`: `FusedSolver`'s (KERNELS Q3/Q4; VERIFY V5 arm F).
    `turn_schedule`: `FusedSolver`'s schedule options (`gamma`, `plus`,
    `shift`, `restart`) for jobs on a 4-card board only (LIVESEARCH B1);
    river jobs keep the default schedule. `schedules`: the solves counted by
    (board length, and the schedule the solver itself ran: its `gamma`,
    `plus`, `shift`, `restart`). `times`: each solve's seconds inside the GPU
    lock, by board length."""

    SCHEDULE_KEYS = frozenset({"gamma", "plus", "shift", "restart"})

    def __init__(self, device: str = "cuda", dtype=None, cache: int = 64, terminals64: bool = False,
                 q3: bool = False, turn_schedule: dict | None = None):
        self.device, self.dtype, self.cache = device, dtype or torch.float32, cache
        self.terminals64, self.q3 = bool(terminals64), bool(q3)
        if turn_schedule is not None:
            bad = sorted(set(turn_schedule) - self.SCHEDULE_KEYS)
            if bad:
                raise ValueError(f"turn_schedule keys must be among {sorted(self.SCHEDULE_KEYS)}, got {bad}")
            turn_schedule = dict(turn_schedule)
            g, sh = turn_schedule.get("gamma", 2), turn_schedule.get("shift")
            if int(g) != g or g < 1 or sh is not None and (int(sh) != sh or sh < 0):  # FusedSolver's, before any solve
                raise ValueError(f"turn_schedule: gamma must be a positive integer and shift a non-negative one, "
                                 f"got {turn_schedule}")
        self.turn_schedule = turn_schedule
        self._hands: dict[tuple, Hands] = {}
        self.solves = 0
        self.schedules: dict[tuple, int] = {}
        self.seconds = 0.0
        self.times: dict[int, list[float]] = {}

    def hands(self, job: dict) -> Hands:
        key = tuple(job["board"])
        h = self._hands.get(key)
        if h is None:
            if len(self._hands) >= self.cache:
                self._hands.clear()
            h = self._hands[key] = Hands.from_arrays(job["board"], job["strength"], job["alive"])
        return h

    def __call__(self, job: dict):
        import time

        # The host-side preparation needs no GPU, so it runs before taking the
        # lock, overlapping the solve of whichever thread holds it.
        hands = self.hands(job)
        plan = host_plan(hands, job, job["ranges"])
        frozen = [(slot, np.asarray(strategy)) for slot, strategy in job["frozen"]]
        want = list(job["want"])
        street = len(job["board"])
        schedule = self.turn_schedule if street == 4 and self.turn_schedule is not None else {}
        # The match threads call in concurrently; solves take turns on the GPU.
        with GPU_LOCK:
            t0 = time.perf_counter()
            s = None
            try:
                s = FusedSolver(hands, job, job["ranges"], device=self.device, dtype=self.dtype, plan=plan, sums=want,
                                terminals64=self.terminals64, q3=self.q3, **schedule)
                ran = (street, s.gamma, s.plus, s.shift, s.restart)
                for slot, strategy in frozen:
                    s.freeze(slot, strategy)
                for _ in range(int(job["iters"])):
                    s.step()
                avg = s.average(want).to(torch.float64).cpu().numpy()
            except BaseException as e:
                s = None  # a failed solve's solver and graph are freed here too, under the lock
                drop_frames(e)
                raise
            s = None  # freed under the lock: its CUDA graph is reset as it goes
            self.solves += 1
            self.schedules[ran] = self.schedules.get(ran, 0) + 1
            dt = time.perf_counter() - t0
            self.seconds += dt
            self.times.setdefault(street, []).append(dt)
        return [avg[i, :plan.dk[w]].reshape(-1).tolist() for i, w in enumerate(want)]
