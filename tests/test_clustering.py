# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""GPU k-means / EMD clustering tests (Phase 3b).

The Earth-Mover's distance is cross-checked against an independent numpy
implementation and hand-computed values; k-means is checked by recovering
well-separated synthetic clusters and by producing sensible, contiguous equity
buckets over the real 169 preflop hands. Runs on CPU or GPU (whichever torch
finds); the numeric cross-check is pinned to CPU for exact float comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parciver.clustering import emd_distance_matrix, kmeans_emd


def _emd_numpy(p: np.ndarray, q: np.ndarray) -> float:
    """Independent 1-D EMD reference: L1 distance between CDFs (bin units)."""
    p = p / p.sum()
    q = q / q.sum()
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


# --- EMD correctness -------------------------------------------------------

def test_emd_hand_computed_cases():
    dev = torch.device("cpu")
    # All mass at bin 0 vs all mass at bin 3: move 1 unit of mass 3 bins -> 3.
    a = torch.tensor([[1.0, 0, 0, 0]], device=dev)
    b = torch.tensor([[0, 0, 0, 1.0]], device=dev)
    assert emd_distance_matrix(a, b).item() == pytest.approx(3.0)
    # Split mass moved two bins each: 0.5*2 + 0.5*2 = 2.0.
    c = torch.tensor([[0.5, 0.5, 0, 0]], device=dev)
    d = torch.tensor([[0, 0, 0.5, 0.5]], device=dev)
    assert emd_distance_matrix(c, d).item() == pytest.approx(2.0)


def test_emd_identity_and_symmetry():
    dev = torch.device("cpu")
    rng = np.random.default_rng(0)
    h = torch.tensor(rng.random((5, 12)), dtype=torch.float32, device=dev)
    h = h / h.sum(1, keepdim=True)
    d = emd_distance_matrix(h, h)
    assert torch.allclose(d.diagonal(), torch.zeros(5), atol=1e-6)  # d(x,x)=0
    assert torch.allclose(d, d.T, atol=1e-6)  # symmetric


def test_emd_matches_numpy_reference():
    dev = torch.device("cpu")
    rng = np.random.default_rng(42)
    P = rng.random((17, 20)).astype(np.float32)
    Q = rng.random((9, 20)).astype(np.float32)
    Pn = P / P.sum(1, keepdims=True)
    Qn = Q / Q.sum(1, keepdims=True)
    got = emd_distance_matrix(
        torch.tensor(Pn, device=dev), torch.tensor(Qn, device=dev)
    ).cpu().numpy()
    ref = np.array([[_emd_numpy(p, q) for q in Qn] for p in Pn])
    assert np.allclose(got, ref, atol=1e-5)


# --- k-means ---------------------------------------------------------------

def _make_clusters(seed: int):
    """Three well-separated archetype histograms + small noise; return (X, truth)."""
    rng = np.random.default_rng(seed)
    B = 24
    bins = np.arange(B)
    archetypes = []
    for center in (3, 12, 20):
        a = np.exp(-0.5 * ((bins - center) / 1.5) ** 2)
        archetypes.append(a / a.sum())
    X, truth = [], []
    for g, arch in enumerate(archetypes):
        for _ in range(40):
            noisy = arch + 0.01 * rng.random(B)
            X.append(noisy / noisy.sum())
            truth.append(g)
    return np.array(X, dtype=np.float32), np.array(truth)


def test_kmeans_recovers_separated_clusters():
    X, truth = _make_clusters(seed=1)
    res = kmeans_emd(X, k=3, seed=0, n_init=3)
    labels = res.labels.cpu().numpy()
    # Each true group must map to exactly one predicted cluster, and the three
    # groups to three distinct clusters (a perfect partition up to relabeling).
    mapping = {}
    for g in (0, 1, 2):
        preds = set(labels[truth == g].tolist())
        assert len(preds) == 1, f"group {g} split across clusters {preds}"
        mapping[g] = preds.pop()
    assert len(set(mapping.values())) == 3


def test_kmeans_is_deterministic_for_a_fixed_seed():
    X, _ = _make_clusters(seed=2)
    a = kmeans_emd(X, k=3, seed=7).labels.cpu().numpy()
    b = kmeans_emd(X, k=3, seed=7).labels.cpu().numpy()
    assert np.array_equal(a, b)


def test_kmeans_always_returns_k_clusters():
    # This fixture has enough distinct histograms for the requested clusters.
    X, _ = _make_clusters(seed=3)
    res = kmeans_emd(X, k=8, seed=0)
    counts = np.bincount(res.labels.cpu().numpy(), minlength=8)
    assert (counts > 0).all()


def test_kmeans_rejects_too_few_distinct_histograms():
    # Normalization makes the first two rows identical, too.
    with pytest.raises(ValueError, match="distinct normalized histograms"):
        kmeans_emd([[1., 0.], [2., 0.], [0., 1.]], 3, device="cpu", n_init=1)
    res = kmeans_emd([[1., 0.], [1., 0.]], 1, device="cpu", n_init=1)
    assert res.inertia == 0.0
    assert res.labels.tolist() == [0, 0]


@pytest.mark.parametrize("row", [[float("nan"), 1.], [float("inf"), 1.], [-1., 2.]])
def test_kmeans_rejects_invalid_histogram_mass(row):
    with pytest.raises(ValueError, match="finite nonnegative"):
        kmeans_emd([row], 1, device="cpu")


# --- end-to-end on real preflop equities -----------------------------------

def test_preflop_equity_buckets_are_ordered_and_contiguous():
    pytest.importorskip("parciver_engine")
    from parciver import abstraction as ab

    # One representative holding per preflop class, its equity vs a random hand.
    B = 50
    eqs, names = [], []
    for idx in range(169):
        name = ab.PREFLOP_CLASS_NAMES[idx]
        r1, r2 = name[0], name[1]
        if name.endswith("s"):
            hole = [r1 + "c", r2 + "c"]
        elif name.endswith("o"):
            hole = [r1 + "c", r2 + "d"]
        else:  # pair
            hole = [r1 + "c", r2 + "d"]
        eqs.append(ab.equity(hole, samples=30_000, seed=99))
        names.append(name)
    eqs = np.array(eqs)

    # Represent each hand as a one-hot histogram over equity bins, then cluster.
    hists = np.zeros((169, B), dtype=np.float32)
    bin_idx = np.clip((eqs * B).astype(int), 0, B - 1)
    hists[np.arange(169), bin_idx] = 1.0

    k = 8
    res = kmeans_emd(hists, k=k, seed=0, n_init=5)
    labels = res.labels.cpu().numpy()

    # AA (strongest) and 72o (near-weakest) must not share a bucket, and AA's
    # bucket must have the higher mean equity.
    aa = names.index("AA")
    trash = names.index("72o")
    assert labels[aa] != labels[trash]
    aa_mean = eqs[labels == labels[aa]].mean()
    trash_mean = eqs[labels == labels[trash]].mean()
    assert aa_mean > trash_mean

    # 1-D k-means partitions the equity line into intervals: sorting hands by
    # equity, the labels must form exactly (#used clusters) contiguous runs.
    order = np.argsort(eqs)
    sorted_labels = labels[order]
    runs = 1 + int((sorted_labels[1:] != sorted_labels[:-1]).sum())
    assert runs == len(set(labels.tolist()))
