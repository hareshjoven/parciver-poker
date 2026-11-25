# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The parciver rating (RATING.md): one number for a bot configuration's strength.

    PR(x) = 1500 + sum_j p_j * [u(x, j) - u(A0, j)]      (1 point = 1 mbb/hand)

`u` is a mean result in mbb/hand from mirrored seeded deals, `p` the anchor
pool's maximum-entropy Nash weights (Nash averaging), A0 the 1500 reference.
x and A0 meet each anchor on the same seeds, so their shared deals are paired
(A0's row may be longer: Amendment 2). Everything here is pure (numpy);
scripts/rating.py plays the matches and keeps the receipts.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

BASE = 1500.0
Z95 = 1.959963984540054
# Below this many paired deals the normal interval is not trusted (RATING.md).
MIN_DEALS = 10_000


class RatingError(ValueError):
    pass


def per_hand_mbb(per_deal_bb, hands_per_deal: int = 2) -> np.ndarray:
    """An engine `per_deal_bb` list (big blinds won per deal) in mbb per hand."""
    return np.asarray(per_deal_bb, dtype=np.float64) * (1000.0 / hands_per_deal)


def array_sha(a) -> str:
    """sha256 of an array's float64 little-endian bytes (receipt integrity)."""
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def nash(A, tol: float = 1e-9) -> np.ndarray:
    """The maximum-entropy Nash equilibrium of the symmetric zero-sum game with
    antisymmetric payoff matrix `A` (A[i, j]: row i's payoff against j).

    Every equilibrium is found by support enumeration (for each support S,
    A[S, S] p = 0 with sum(p) = 1; kept when p >= 0 and A p <= 0, both up to
    `tol` times max|A|). The equilibria form a convex set; when there is more
    than one extreme point the entropy is maximized over their hull."""
    A = np.asarray(A, dtype=np.float64)
    k = A.shape[0]
    if A.shape != (k, k) or k == 0:
        raise RatingError(f"payoff matrix must be square, got {A.shape}")
    if k > 16:
        raise RatingError("support enumeration is for small pools (<= 16 anchors)")
    scale = max(float(np.abs(A).max()), 1e-300)
    if np.abs(A + A.T).max() > tol * scale:
        raise RatingError("payoff matrix is not antisymmetric")
    eq = []
    for size in range(1, k + 1):
        for S in itertools.combinations(range(k), size):
            S = list(S)
            M = np.vstack([A[np.ix_(S, S)], np.ones(size)])
            b = np.zeros(size + 1)
            b[-1] = 1.0
            q = np.linalg.lstsq(M, b, rcond=None)[0]
            if np.abs(A[np.ix_(S, S)] @ q).max() > tol * scale or abs(q.sum() - 1.0) > tol or q.min() < -tol:
                continue
            p = np.zeros(k)
            p[S] = np.maximum(q, 0.0)
            p /= p.sum()
            if (A @ p).max() > tol * scale:
                continue
            if not any(np.abs(p - e).max() <= 1e-9 for e in eq):
                eq.append(p)
    if not eq:
        raise RatingError("no equilibrium found (tolerance too tight?)")
    if len(eq) == 1:
        return eq[0]
    P = np.array(eq)
    lam = np.full(len(eq), 1.0 / len(eq))
    for _ in range(5000):
        q = np.maximum(lam @ P, 1e-300)
        grad = -(P * (np.log(q) + 1.0)).sum(1)
        lam = lam * np.exp(0.5 * (grad - grad.max()))
        lam /= lam.sum()
    p = lam @ P
    p[p < 1e-12] = 0.0
    p = p / p.sum()
    if (A @ p).max() > tol * scale:
        raise RatingError("the maximum-entropy mix of the equilibria is not an equilibrium (did not converge)")
    return p


def check_nash(A, p, tol: float = 1e-9) -> float:
    """max_i (A p)_i / max|A|: <= tol for an equilibrium of the zero-sum game."""
    A = np.asarray(A, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if p.min() < 0 or abs(p.sum() - 1.0) > 1e-12:
        raise RatingError("weights must be a distribution")
    return float((A @ p).max() / max(float(np.abs(A).max()), 1e-300))


def diff(x, y) -> tuple[float, float]:
    """mean(x) - mean(y) and its variance, for per-deal arrays on the same seed
    where deal i of both is the same deal (lengths may differ: the first
    min(len) deals are paired, the rest independent; RATING.md Amendment 2):
    var = var_x/Dx + var_y/Dy - 2 cov(x, y over the shared deals) m/(Dx Dy)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx, dy = len(x), len(y)
    m = min(dx, dy)
    if m < 2:
        raise RatingError("need at least 2 deals")
    if dx == dy:  # fully paired: the same value, computed on the differences (exactly 0 for identical rows)
        return float(x.mean() - y.mean()), float((x - y).var(ddof=1) / dx)
    c = float(np.cov(x[:m], y[:m], ddof=1)[0, 1])
    var = x.var(ddof=1) / dx + y.var(ddof=1) / dy - 2.0 * c * m / (dx * dy)
    return float(x.mean() - y.mean()), max(float(var), 0.0)


def _weighted(x: dict, y: dict, weights: dict, min_deals: int, x_longer_ok: bool):
    use = [j for j, w in weights.items() if w > 0]
    if not use:
        raise RatingError("no anchor has positive weight")
    if abs(sum(weights[j] for j in use) - 1.0) > 1e-9:
        raise RatingError("weights must sum to 1")
    missing = [j for j in use if j not in x or j not in y]
    if missing:
        raise RatingError(f"missing results against {missing}")
    mean, var, n = 0.0, 0.0, {}
    for j in use:
        if len(x[j]) < min_deals:
            raise RatingError(f"{len(x[j])} deals against {j} < {min_deals}: the interval would not be trusted")
        if not x_longer_ok and len(x[j]) > len(y[j]):
            raise RatingError(f"the reference has {len(y[j])} deals against {j}, fewer than the candidate's "
                              f"{len(x[j])}")
        d, v = diff(x[j], y[j])
        # Anchors are played on independent seeds: their variances add.
        mean += weights[j] * d
        var += weights[j] ** 2 * v
        n[j] = len(x[j])
    return mean, math.sqrt(var), use, n


def paired(x: dict, ref: dict, weights: dict, base: float = BASE, min_deals: int = MIN_DEALS) -> dict:
    """PR from per-deal mbb/hand arrays: `x[j]` the candidate against anchor j
    (D_j deals), `ref[j]` A0 against j on the same seed (R_j >= D_j deals; the
    first D_j are the same deals), `weights[j]` = p_j. Anchors with weight 0
    are not needed. Returns the rating, its standard error and 95% interval,
    and the candidate's deals (the smallest D_j)."""
    mean, se, use, n = _weighted(x, ref, weights, min_deals, x_longer_ok=False)
    return dict(pr=base + mean, se=se, lo=base + mean - Z95 * se, hi=base + mean + Z95 * se, n=min(n.values()),
                anchors=use, deals=n)


def change(x: dict, y: dict, weights: dict, min_deals: int = MIN_DEALS) -> dict:
    """PR(x) - PR(y) measured directly (the reference cancels): sum_j p_j
    [mean(x_j) - mean(y_j)] over each anchor's shared seed."""
    mean, se, use, n = _weighted(x, y, weights, min_deals, x_longer_ok=True)
    return dict(mean=mean, se=se, lo=mean - Z95 * se, hi=mean + Z95 * se, anchors=use)


def mean_ci(v) -> dict:
    """Mean, standard error and 95% interval of per-deal values."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n < 2:
        raise RatingError("need at least 2 deals")
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    return dict(mean=mean, se=se, lo=mean - Z95 * se, hi=mean + Z95 * se, n=n)

