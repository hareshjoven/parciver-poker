# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""End-to-end card-abstraction bucketing tests (Phase 3b).

Exercises the full driver on small slices: enumerate/index -> extract features
-> cluster -> persist -> gate, plus the resumable-checkpoint guarantee that makes
the multi-hour real runs safe, and the play-time bucket lookup.

Skips cleanly if the extension or torch is not available.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pytest

pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)
pytest.importorskip("torch")

import parciver_engine as pe
from parciver import bucketing as B
from parciver.clustering import kmeans_1d


def test_grouped_extrema_match_independent_masks():
    rng = np.random.default_rng(37)
    values = rng.random(1000).astype(np.float32)
    labels = rng.integers(0, 7, size=1000, dtype=np.uint16)
    expected = np.full((8, 2), np.nan)
    for b in range(7):
        members = values[labels == b]
        expected[b] = members.min(), members.max()
    np.testing.assert_array_equal(B._bucket_extrema(values, labels, 8), expected)


def test_build_flop_small_passes_gate(tmp_path):
    gate = B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    assert gate.ok
    assert gate.buckets_nonempty == 6
    assert gate.monotone
    assert gate.min_bucket_size > 0
    # Persisted table addresses dense indices and has one entry per situation.
    table = np.load(B._paths(tmp_path, "flop")["buckets"])
    assert table.shape == (150,)
    assert set(np.unique(table).tolist()) == set(range(6))
    meta = json.loads(B._paths(tmp_path, "flop")["meta"].read_text())
    assert meta["k"] == 6 and meta["n"] == 150 and meta["gate"]["ok"]


def test_build_river_small_passes_gate_and_intervals_disjoint(tmp_path):
    gate = B.build_street("river", k=6, out_dir=tmp_path, chunk=20000, limit=40000, log=False)
    assert gate.ok
    assert gate.buckets_nonempty == 6
    assert gate.monotone
    # River buckets are exact equity intervals: their ranges must be disjoint.
    assert gate.river_intervals_disjoint


def test_build_river_caps_k_at_distinct_equities(tmp_path):
    # River equity is discrete ((wins+ties/2)/C(45,2)), so it has a hard ceiling
    # on the number of distinct values. Requesting more buckets than that must
    # cap to the distinct count (an exact, lossless assignment) and still pass.
    gate = B.build_street("river", k=1_000_000, out_dir=tmp_path, chunk=20000, limit=40000, log=False)
    assert gate.ok
    assert gate.k < 1_000_000  # capped
    assert gate.buckets_nonempty == gate.k
    assert gate.monotone and gate.river_intervals_disjoint
    # Every bucket is a single distinct equity value -> exact (inertia 0).
    meta = json.loads(B._paths(tmp_path, "river")["meta"].read_text())
    assert meta["inertia"] == 0.0


def test_bucket_of_matches_the_table(tmp_path):
    B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    B._TABLE_CACHE.clear()
    table = np.load(B._paths(tmp_path, "flop")["buckets"])
    # For a few dense indices, bucket_of(the representative) hits the same row.
    for i in (0, 37, 149):
        hand = list(pe.canonical_unindex_batch(0, i, 1))
        assert B.bucket_of("flop", hand, out_dir=tmp_path) == int(table[i])


def test_bucket_of_is_suit_invariant(tmp_path):
    B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    B._TABLE_CACHE.clear()
    hand = list(pe.canonical_unindex_batch(0, 42, 1))
    perm = [1, 2, 3, 0]
    permuted = [(c >> 2) * 4 + perm[c & 3] for c in hand]
    assert B.bucket_of("flop", hand, out_dir=tmp_path) == B.bucket_of("flop", permuted, out_dir=tmp_path)


def test_extraction_is_resumable_and_deterministic(tmp_path):
    # Full clean run, then roll the checkpoint back, corrupt the tail, and resume:
    # the refilled features must be bit-identical (exact enumeration, no sampling).
    B.extract_features("flop", bins=20, out_dir=tmp_path, chunk=40, limit=200, log=False)
    p = B._paths(tmp_path, "flop")
    full = np.array(np.load(p["features"], mmap_mode="r"))

    feats = np.lib.format.open_memmap(p["features"], mode="r+")
    feats[120:] = -1.0
    feats.flush()
    del feats
    gc.collect()
    p["progress"].write_text(json.dumps({"street": "flop", "n": 200, "bins": 20, "rows_done": 120}))

    B.extract_features("flop", bins=20, out_dir=tmp_path, chunk=40, limit=200, log=False)
    resumed = np.array(np.load(p["features"], mmap_mode="r"))

    assert json.loads(p["progress"].read_text())["rows_done"] == 200
    assert np.array_equal(resumed, full)
    assert not (resumed == -1.0).any()
    assert np.allclose(resumed.sum(axis=1), 1.0, atol=1e-5)


def test_extraction_stopped_before_its_first_chunk_starts_afresh(tmp_path):
    # open_memmap writes the header, then sizes the file: a stop in between
    # leaves the progress (0 rows) and a file of the header alone.
    full = B.extract_features("flop", bins=20, out_dir=tmp_path / "full", chunk=40, limit=120, log=False)[0]
    with open(full, "rb") as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        head = f.tell()
    p = B._paths(tmp_path, "flop")
    p["features"].write_bytes(full.read_bytes()[:head])
    p["progress"].write_text(json.dumps({"street": "flop", "n": 120, "bins": 20, "rows_done": 0}))
    B.extract_features("flop", bins=20, out_dir=tmp_path, chunk=40, limit=120, log=False)
    gc.collect()
    assert p["features"].read_bytes() == full.read_bytes()
    assert json.loads(p["progress"].read_text())["rows_done"] == 120 and not list(tmp_path.glob("*.tmp"))


def test_features_under_other_settings_are_never_overwritten(tmp_path):
    # A smoke run (another limit or bins) pointed at finished tables must raise, not restart over them.
    B.extract_features("flop", bins=20, out_dir=tmp_path, chunk=40, limit=120, log=False)
    p = B._paths(tmp_path, "flop")
    before = p["features"].read_bytes(), p["progress"].read_bytes()
    for kw in (dict(bins=20, limit=80), dict(bins=10, limit=120)):
        with pytest.raises(ValueError, match="use another out_dir"):
            B.extract_features("flop", out_dir=tmp_path, chunk=40, log=False, **kw)
        gc.collect()
        assert (p["features"].read_bytes(), p["progress"].read_bytes()) == before
    # The same settings still resume (here: nothing left to do).
    assert B.extract_features("flop", bins=20, out_dir=tmp_path, chunk=40, limit=120, log=False)[1] == 120
    assert p["features"].read_bytes() == before[0]
    # Features without their progress file are refused too, not started over.
    p["progress"].unlink()
    with pytest.raises(ValueError, match="without"):
        B.extract_features("flop", bins=20, out_dir=tmp_path, chunk=40, limit=120, log=False)
    gc.collect()
    assert p["features"].read_bytes() == before[0]


def test_a_built_bucket_table_is_kept(tmp_path, monkeypatch):
    gate = B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    p = B._paths(tmp_path, "flop")
    before = p["buckets"].read_bytes(), p["meta"].read_bytes()
    real = B.kmeans_emd
    monkeypatch.setattr(B, "kmeans_emd", None)  # the same settings: reported, not clustered again
    assert B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False) == gate
    with pytest.raises(FileExistsError, match="build_buckets.py --force"):  # another k: refused
        B.build_street("flop", k=5, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    for other in (dict(seed=1), dict(n_init=5), dict(n_iter=50)):  # other k-means settings: refused
        with pytest.raises(FileExistsError, match="kmeans"):
            B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False, **other)
    assert (p["buckets"].read_bytes(), p["meta"].read_bytes()) == before
    # A table that is not the one its meta describes (an older overwrite that
    # stopped between them), or a table without its meta: refused, not kept.
    labels = np.load(p["buckets"])
    np.save(p["buckets"], np.where(labels == 0, 1, labels).astype(labels.dtype))
    with pytest.raises(FileExistsError, match="not the table its meta"):
        B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    p["buckets"].write_bytes(before[0])
    p["meta"].unlink()
    with pytest.raises(FileExistsError, match="has no flop_meta.json"):
        B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    p["meta"].write_bytes(before[1])
    # A meta written before k_requested and kmeans were recorded: its k is kept;
    # its k-means settings cannot be compared (kept, said so).
    legacy = {key: v for key, v in json.loads(p["meta"].read_text()).items() if key not in ("k_requested", "kmeans")}
    p["meta"].write_text(json.dumps(legacy))
    assert B.build_street("flop", k=6, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False, seed=1) == gate
    with pytest.raises(FileExistsError):
        B.build_street("flop", k=5, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    monkeypatch.setattr(B, "kmeans_emd", real)
    gate5 = B.build_street("flop", k=5, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False, overwrite=True)
    assert gate5.k == 5 and json.loads(p["meta"].read_text())["k_requested"] == 5
    assert not list(tmp_path.glob("*.tmp"))
    # An overwrite stopped part-way (here: after the new table, before its
    # centroids) leaves the table without a meta: refused, never kept.
    real_save = B._save_npy

    def stop(path, arr):
        if "centroids" in path.name:
            raise KeyboardInterrupt
        real_save(path, arr)

    monkeypatch.setattr(B, "_save_npy", stop)
    with pytest.raises(KeyboardInterrupt):
        B.build_street("flop", k=4, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False, overwrite=True)
    monkeypatch.setattr(B, "_save_npy", real_save)
    for k in (4, 5):
        with pytest.raises(FileExistsError, match="has no flop_meta.json"):
            B.build_street("flop", k=k, bins=20, out_dir=tmp_path, chunk=50, limit=150, log=False)
    # A river table capped below the k asked is the one asked for.
    river = B.build_street("river", k=1_000_000, out_dir=tmp_path, chunk=20000, limit=40000, log=False)
    assert river.k < 1_000_000
    assert B.build_street("river", k=1_000_000, out_dir=tmp_path, chunk=20000, limit=40000, log=False) == river
    q = B._paths(tmp_path, "river")  # an older river meta records the capped k alone: that k is kept
    q["meta"].write_text(json.dumps({key: v for key, v in json.loads(q["meta"].read_text()).items()
                                     if key not in ("k_requested", "kmeans")}))
    assert B.build_street("river", k=river.k, out_dir=tmp_path, chunk=20000, limit=40000, log=False) == river
    with pytest.raises(FileExistsError):
        B.build_street("river", k=1_000_000, out_dir=tmp_path, chunk=20000, limit=40000, log=False)


def test_kmeans_1d_recovers_separated_clusters_monotone():
    rng = np.random.default_rng(0)
    # Three well-separated 1-D bumps.
    vals = np.concatenate([
        rng.normal(0.1, 0.01, 500),
        rng.normal(0.5, 0.01, 500),
        rng.normal(0.9, 0.01, 500),
    ]).astype(np.float32)
    res = kmeans_1d(vals, 3)
    labels = res.labels.cpu().numpy()
    centroids = res.centroids.cpu().numpy().reshape(-1)
    # Exactly 3 non-empty buckets, centroids ascending (monotone in equity).
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    assert np.all(np.diff(centroids) > 0)
    # Labels are monotone in value: a higher value never gets a lower bucket.
    order = np.argsort(vals)
    assert np.all(np.diff(labels[order]) >= 0)


def test_kmeans_1d_deterministic():
    rng = np.random.default_rng(3)
    vals = rng.random(2000).astype(np.float32)
    a = kmeans_1d(vals, 7).labels.cpu().numpy()
    b = kmeans_1d(vals, 7).labels.cpu().numpy()
    assert np.array_equal(a, b)
