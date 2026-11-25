# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""GPU k-means over equity-distribution histograms (Phase 3b).

Card abstraction buckets strategically-similar hands together. The modern,
potential-aware method (Ganzfried–Sandholm; the Libratus/Pluribus lineage)
represents each hand-on-a-board as a *distribution* over its next-street equity
and clusters those distributions under the **Earth-Mover's (Wasserstein-1)
distance** — so a drawing hand (low equity now, high variance) lands in a
different bucket from a made hand of the same current equity.

This is the dense-linear-algebra half of the pipeline, so it lives on the GPU
(PyTorch), matching the project's measured lesson that GPU/array math beats Rust
here while the branchy equity enumeration stays in Rust. For 1-D distributions
the EMD has a closed form — the L1 distance between CDFs — which is exact, cheap,
and trivially batched, so no optimal-transport solver is needed.

Centroids are updated to the arithmetic mean of their members' histograms. That
is the standard k-means-with-EMD update used in the abstraction literature; it is
the exact barycenter only under L2, but in practice recovers clusters well and
keeps the update a single fused reduction. (The exact 1-D Wasserstein barycenter
would average inverse-CDFs; noted here in case a later phase wants it.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _as_hist_tensor(x, device: torch.device) -> torch.Tensor:
    """Coerce input to a float32 [N, B] tensor of row-normalized histograms."""
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    if t.ndim != 2:
        raise ValueError(f"expected [N, B] histograms, got shape {tuple(t.shape)}")
    if not torch.isfinite(t).all() or (t < 0).any():
        raise ValueError("histograms must contain finite nonnegative mass")
    row = t.sum(dim=1, keepdim=True)
    if (row <= 0).any():
        raise ValueError("every histogram must have positive total mass")
    return t / row


def pick_device(device=None) -> torch.device:
    """Chosen device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def emd_distance_matrix(
    points: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk: int = 8192,
) -> torch.Tensor:
    """Pairwise 1-D EMD between `points` [N, B] and `centroids` [K, B].

    Both are row-normalized histograms over the same ordered bins. Returns an
    [N, K] matrix of Earth-Mover's distances in *bin units* (bin spacing = 1).

    1-D EMD is the L1 distance between the CDFs, so this is exactly
    ``torch.cdist(cumsum(points), cumsum(centroids), p=1)`` — computed chunked
    over points and via ``cdist`` (not an explicit ``[chunk, K, B]`` broadcast) so
    the transient memory stays ``O(chunk * K)`` rather than ``O(chunk * K * B)``.
    NOTE: the returned matrix is still ``[N, K]``; at abstraction scale
    (N in the millions, K in the thousands) never materialize it — use
    :func:`kmeans_emd`, which assigns in point-chunks and never holds ``[N, K]``.
    """
    pc = torch.cumsum(points, dim=1)
    cc = torch.cumsum(centroids, dim=1)
    n, k = pc.shape[0], cc.shape[0]
    out = torch.empty((n, k), dtype=points.dtype, device=points.device)
    for i in range(0, n, chunk):
        out[i : i + chunk] = torch.cdist(pc[i : i + chunk], cc, p=1)
    return out


def _kmeanspp_init(
    hists: torch.Tensor, k: int, generator: torch.Generator, *, hcum: torch.Tensor | None = None
) -> torch.Tensor:
    """k-means++ seeding under EMD (D^2 weighting) for a good, spread start.

    Each new-center pass is a single ``[N]`` distance-to-one-centroid (the L1
    distance between point CDFs and the chosen CDF), so memory stays ``O(N * B)``
    and never touches ``[N, K]`` — this scales to millions of points and thousands
    of centers. `hcum` (precomputed point CDFs) is reused if supplied.
    """
    n = hists.shape[0]
    if hcum is None:
        hcum = torch.cumsum(hists, dim=1)
    first = int(torch.randint(0, n, (1,), generator=generator, device=hists.device))
    chosen = [first]
    d2 = (hcum - hcum[first]).abs().sum(dim=1) ** 2  # [N]
    for _ in range(1, k):
        total = d2.sum()
        if total.item() == 0.0:
            raise ValueError(f"k={k} exceeds the number of distinct normalized histograms")
        probs = d2 / total
        idx = int(torch.multinomial(probs, 1, generator=generator))
        chosen.append(idx)
        nd = (hcum - hcum[idx]).abs().sum(dim=1) ** 2
        d2 = torch.minimum(d2, nd)
    return hists[torch.tensor(chosen, device=hists.device)].clone()


@dataclass
class KMeansResult:
    labels: torch.Tensor  # [N] int64 cluster id per point
    centroids: torch.Tensor  # [K, B] centroid histograms
    inertia: float  # sum of EMD (bin units) from each point to its centroid
    n_iter: int  # Lloyd iterations actually run
    converged: bool


def kmeans_1d(
    values,
    k: int,
    *,
    n_iter: int = 100,
    tol: float = 1e-9,
    device=None,
) -> KMeansResult:
    """1-D k-means on scalar values — the river bucketer.

    The river feature is a single equity scalar (a one-point distribution), so
    the EMD between two river hands is just ``|equity_i - equity_j|`` and k-means
    reduces to the exact 1-D case. Because Voronoi cells on a line are intervals,
    the returned buckets are **contiguous and monotone in equity** by
    construction: bucket id increases with equity. This is exact and fully
    deterministic — points are sorted once and each cluster is a segment whose
    mean is read off a prefix sum, so there are no nondeterministic CUDA atomics
    (cf. the one-hot trick in :func:`kmeans_emd`).

    Returns a :class:`KMeansResult` whose ``centroids`` is ``[k, 1]`` (ascending)
    so it prints and stores like the histogram centroids.
    """
    dev = pick_device(device)
    x = torch.as_tensor(values, dtype=torch.float32, device=dev).reshape(-1)
    n = x.numel()
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= N={n}, got k={k}")

    xs, _ = torch.sort(x)
    prefix = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64), xs.to(torch.float64).cumsum(0)])

    # Equal-frequency quantile seeding, then Lloyd on sorted segments.
    q = (torch.arange(k, device=dev, dtype=torch.float32) + 0.5) / k
    centroids = xs[(q * n).long().clamp_(max=n - 1)].clone()

    used_iters = n_iter
    converged = False
    for it in range(n_iter):
        centroids, _ = torch.sort(centroids)
        mids = (centroids[:-1] + centroids[1:]) * 0.5
        bnd = torch.searchsorted(xs, mids)  # k-1 split points into the sorted values
        edges = torch.cat([
            torch.zeros(1, device=dev, dtype=bnd.dtype),
            bnd,
            torch.full((1,), n, device=dev, dtype=bnd.dtype),
        ])
        counts = (edges[1:] - edges[:-1]).to(torch.float64)
        seg_sum = prefix[edges[1:]] - prefix[edges[:-1]]
        nonempty = counts > 0
        new = centroids.clone()
        new[nonempty] = (seg_sum[nonempty] / counts[nonempty]).to(torch.float32)
        # Re-seed any empty cluster into the widest-spanning current segment so
        # exactly k non-empty buckets come back.
        if (~nonempty).any():
            spans = xs[(edges[1:] - 1).clamp_(min=0)] - xs[edges[:-1].clamp_(max=n - 1)]
            for c in torch.nonzero(~nonempty, as_tuple=False).flatten().tolist():
                widest = int(spans.argmax())
                lo = int(edges[widest])
                hi = int(edges[widest + 1])
                new[c] = xs[(lo + hi) // 2]
                spans[widest] = -1.0
        shift = (new - centroids).abs().sum().item()
        centroids = new
        if shift <= tol and it > 0:
            converged = True
            used_iters = it + 1
            break

    centroids, _ = torch.sort(centroids)
    mids = (centroids[:-1] + centroids[1:]) * 0.5
    labels = torch.bucketize(x, mids)  # [N] in 0..k, monotone in equity
    inertia = (x - centroids[labels]).abs().sum().item()
    return KMeansResult(labels, centroids.unsqueeze(1), inertia, used_iters, converged)


def kmeans_emd(
    features,
    k: int,
    *,
    n_iter: int = 100,
    n_init: int = 3,
    seed: int = 0,
    tol: float = 1e-6,
    device=None,
    chunk: int = 16384,
) -> KMeansResult:
    """Cluster equity-distribution histograms with k-means under 1-D EMD.

    `features` is [N, B] histograms (rows need not be normalized; they are). Runs
    `n_init` seedings and keeps the lowest-inertia result. Fully deterministic
    given `seed`. Empty clusters are re-seeded to the points farthest from their
    centroid. Fewer than `k` distinct normalized histograms is rejected; callers
    must still gate the final assignment for nonempty clusters.

    **Scales to the abstraction sizes** (N in the millions, K in the thousands):
    assignment and the centroid accumulation run in point-chunks of `chunk`, so
    the peak transient is ``O(chunk * K)`` for the distance/one-hot blocks and the
    full ``[N, K]`` matrix is *never* allocated. The centroid update stays the
    deterministic one-hot + matmul (no nondeterministic CUDA atomic adds), so a
    fixed seed still gives bit-identical labels run to run. Per-point CDFs are
    precomputed once; distances use ``torch.cdist(p=1)`` on the CDFs (= 1-D EMD).
    """
    dev = pick_device(device)
    hists = _as_hist_tensor(features, dev)
    n, b = hists.shape
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= N={n}, got k={k}")
    if n_init < 1 or n_iter < 1 or chunk < 1:
        raise ValueError("n_init, n_iter and chunk must be positive")
    hcum = torch.cumsum(hists, dim=1)  # [N, B] point CDFs, reused every pass

    def assign(ccum: torch.Tensor, *, accumulate: bool = True):
        """Chunked assignment: return (labels [N], mindist [N], sums [K,B], counts [K])."""
        labels = torch.empty(n, dtype=torch.long, device=dev)
        mindist = torch.empty(n, dtype=hists.dtype, device=dev)
        sums = torch.zeros((k, b), dtype=hists.dtype, device=dev)
        counts = torch.zeros(k, dtype=hists.dtype, device=dev)
        for i in range(0, n, chunk):
            d = torch.cdist(hcum[i : i + chunk], ccum, p=1)  # [c, K]
            lab = d.argmin(dim=1)
            labels[i : i + chunk] = lab
            mindist[i : i + chunk] = d.gather(1, lab.unsqueeze(1)).squeeze(1)
            if accumulate:
                onehot = torch.zeros((lab.shape[0], k), dtype=hists.dtype, device=dev)
                onehot.scatter_(1, lab.unsqueeze(1), 1.0)
                sums += onehot.t() @ hists[i : i + chunk]
                counts += onehot.sum(dim=0)
        return labels, mindist, sums, counts

    best: KMeansResult | None = None
    for init in range(n_init):
        gen = torch.Generator(device=dev).manual_seed(seed + init)
        centroids = _kmeanspp_init(hists, k, gen, hcum=hcum)
        labels = torch.zeros(n, dtype=torch.long, device=dev)
        converged = False
        used_iters = n_iter
        for it in range(n_iter):
            labels, mindist, sums, counts = assign(torch.cumsum(centroids, dim=1))
            nonempty = counts > 0
            new_centroids = centroids.clone()
            new_centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
            # Re-seed empty clusters to the farthest-fit points (distinct).
            if (~nonempty).any():
                empties = torch.nonzero(~nonempty, as_tuple=False).flatten()
                worst = torch.topk(mindist, empties.numel()).indices
                new_centroids[empties] = hists[worst]
            shift = (new_centroids - centroids).abs().sum().item()
            centroids = new_centroids
            if shift <= tol and it > 0:
                converged = True
                used_iters = it + 1
                break
        # Final assignment against the converged centroids (labels above were vs
        # the pre-update centroids of the last iteration).
        labels, mindist, _, _ = assign(torch.cumsum(centroids, dim=1), accumulate=False)
        inertia = mindist.sum().item()
        if best is None or inertia < best.inertia:
            best = KMeansResult(labels.clone(), centroids.clone(), inertia, used_iters, converged)
    assert best is not None
    return best
