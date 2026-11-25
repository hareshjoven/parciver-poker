# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""End-to-end card-abstraction bucketing (Phase 3b).

Ties the three certified pieces together into the bucket tables the CFR solver
keys its infosets on:

  1. **enumerate/index** every suit-isomorphic canonical ``(hole, board)``
     situation of a street — the flop's 1,286,792, the turn's 13,960,050, the
     river's 123,156,254 — via the dense Waugh indexer in ``parciver_engine``
     (``canonical_unindex_batch`` materializes a block of situations from their
     dense indices, so no hash set of all classes is ever built);
  2. **extract** the equity-distribution feature per situation (flop/turn: a
     normalized histogram of next-street equity; river: the scalar equity),
     using the rayon-parallel batch evaluators — this is the multi-hour exact
     pass, so it is **chunked and checkpointed** to a memmap and fully resumable;
  3. **cluster** the features into ``k`` buckets (flop/turn under Earth-Mover's
     distance with :func:`parciver.clustering.kmeans_emd`; river with the exact
     1-D :func:`parciver.clustering.kmeans_1d`), then **persist** a bucket table
     (one ``uint16`` per dense index) plus centroids and metadata.

The table is addressed by the dense index, so play-time lookup is
``bucket_of(street, cards)`` = ``table[canonical_index(street, cards)]``.

Every street run ends on a gate (:func:`gate_street`): exactly ``k`` non-empty
buckets and a monotone per-bucket mean-equity ordering (and, for the river,
non-overlapping equity intervals).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

import parciver_engine as _engine

from .clustering import KMeansResult, kmeans_1d, kmeans_emd, pick_device
from .solver_watch import persist, write_atomic

# Street selectors match the Rust indexer: 0=flop, 1=turn, 2=river.
STREET_IDS = {"flop": 0, "turn": 1, "river": 2}
STREET_NAMES = {v: k for k, v in STREET_IDS.items()}
BOARD_CARDS = {0: 3, 1: 4, 2: 5}


def _street_id(street) -> int:
    if isinstance(street, str):
        try:
            return STREET_IDS[street]
        except KeyError:
            raise ValueError(f"unknown street {street!r}; use one of {list(STREET_IDS)}")
    if street not in STREET_NAMES:
        raise ValueError(f"street id {street} out of 0..3 (flop/turn/river)")
    return street


def default_out_dir() -> Path:
    """Repo ``data/abstraction`` directory (gitignored); created on demand."""
    # src/parciver/bucketing.py -> repo root is three parents up.
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "abstraction"


def _paths(out_dir: Path, name: str) -> dict[str, Path]:
    return {
        "features": out_dir / f"{name}_features.npy",
        "progress": out_dir / f"{name}_features.progress.json",
        "buckets": out_dir / f"{name}_buckets.npy",
        "centroids": out_dir / f"{name}_centroids.npy",
        "meta": out_dir / f"{name}_meta.json",
    }


# --- feature extraction (resumable) ----------------------------------------

def extract_features(
    street,
    *,
    bins: int = 30,
    out_dir: Path | None = None,
    chunk: int = 4096,
    limit: int | None = None,
    log: bool = True,
) -> tuple[Path, int]:
    """Extract the equity-distribution feature for every situation of a street.

    Writes a float32 memmap (``[n, bins]`` for flop/turn, ``[n]`` for the river)
    and checkpoints progress after every chunk, so an interrupted multi-hour run
    resumes exactly where it stopped. ``limit`` caps the number of situations
    (for smoke tests). Returns ``(feature_path, n_rows)``.

    Features already in ``out_dir`` under other settings (another ``limit`` or
    ``bins``) are never overwritten: that raises, so a smoke run pointed at a
    directory of finished tables cannot destroy hours of extraction. So do
    features without their progress file (written before the file is created).
    A run stopped before its first chunk was saved starts afresh: it has no
    rows to keep, and its file may be cut short (a header alone). A chunk's
    rows are on the disk before the progress counts them.
    """
    sid = _street_id(street)
    name = STREET_NAMES[sid]
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _paths(out_dir, name)

    n_total = int(_engine.canonical_index_size(sid))
    n = n_total if limit is None else min(limit, n_total)
    cards = _engine.street_card_count(sid)
    is_river = sid == 2
    shape = (n,) if is_river else (n, bins)

    # Resume if a compatible partial run exists, else start fresh. A run under
    # other settings is refused rather than restarted over the existing file.
    rows_done = 0
    if p["features"].exists() and not p["progress"].exists():
        raise ValueError(f"{p['features']} exists without {p['progress'].name}: its rows cannot be trusted or "
                         f"resumed; use another out_dir, or delete it first")
    prog = json.loads(p["progress"].read_text()) if p["features"].exists() else {}
    if prog.get("rows_done", 0) > 0:
        have = tuple(np.load(p["features"], mmap_mode="r").shape)
        if prog.get("n") != n or prog.get("bins") != (0 if is_river else bins) or have != shape:
            raise ValueError(
                f"{p['features']} holds {name} features of shape {have} (progress {prog}), not the "
                f"{shape} asked for: use another out_dir, or delete the {name}_* files there first"
            )
        rows_done = int(prog.get("rows_done", 0))
        feats = np.lib.format.open_memmap(p["features"], mode="r+")
    else:  # progress first: a stop in between leaves progress alone, or with a part-made file: both start afresh
        write_atomic(p["progress"], json.dumps({"street": name, "n": n, "bins": 0 if is_river else bins, "rows_done": 0}))
        feats = np.lib.format.open_memmap(p["features"], mode="w+", dtype=np.float32, shape=shape)

    if rows_done >= n:
        if log:
            print(f"[{name}] features already complete ({n:,} rows)")
        return p["features"], n

    t0 = time.perf_counter()
    start_rows = rows_done
    pos = rows_done
    while pos < n:
        count = min(chunk, n - pos)
        raw = _engine.canonical_unindex_batch(sid, pos, count)
        flat = np.frombuffer(raw, dtype=np.uint8).reshape(count, cards)
        holes = flat[:, :2].reshape(-1).tolist()
        board = flat[:, 2:].reshape(-1).tolist()
        if is_river:
            out = _engine.river_equities_batch(holes, board)
            feats[pos : pos + count] = np.asarray(out, dtype=np.float32)
        elif sid == 0:
            out = _engine.flop_histograms_batch(holes, board, bins)
            feats[pos : pos + count] = np.asarray(out, dtype=np.float32).reshape(count, bins)
        else:
            out = _engine.turn_histograms_batch(holes, board, bins)
            feats[pos : pos + count] = np.asarray(out, dtype=np.float32).reshape(count, bins)
        pos += count
        feats.flush()  # on Windows this only hands the pages to the file cache:
        with open(p["features"], "rb+") as f:  # on the disk before the progress counts them
            os.fsync(f.fileno())
        write_atomic(p["progress"], json.dumps({
            "street": name, "n": n, "bins": 0 if is_river else bins, "rows_done": pos,
        }))
        if log:
            done = pos - start_rows
            rate = done / max(time.perf_counter() - t0, 1e-9)
            eta = (n - pos) / max(rate, 1e-9)
            print(f"[{name}] {pos:,}/{n:,} ({100 * pos / n:5.1f}%)  "
                  f"{rate:,.0f}/s  ETA {eta / 60:6.1f} min", flush=True)
    if log:
        print(f"[{name}] extraction done: {n:,} rows in {(time.perf_counter() - t0) / 60:.1f} min")
    return p["features"], n


# --- clustering + persistence ----------------------------------------------

def _bin_centers(bins: int, device) -> torch.Tensor:
    return (torch.arange(bins, device=device, dtype=torch.float32) + 0.5) / bins


def _bucket_mean_equity(centroids: torch.Tensor, is_river: bool) -> np.ndarray:
    """Mean equity represented by each bucket centroid, in [0, 1]."""
    if is_river:
        return centroids.reshape(-1).cpu().numpy()
    centers = _bin_centers(centroids.shape[1], centroids.device)
    return (centroids * centers).sum(dim=1).cpu().numpy()


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """`np.save` through a temporary file and `persist`: never a torn table."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    persist(tmp, path)


def _table_matches(path: Path, meta: dict) -> bool:
    """Whether the bucket table at `path` is the one `meta` describes: its n
    labels, each bucket's size. Read in chunks (bincount widens to int64)."""
    table = np.load(path, mmap_mode="r")
    sizes = np.zeros(len(meta["bucket_sizes"]), np.int64)
    if table.shape != (meta["n"],):
        return False
    for s in range(0, len(table), 1 << 24):
        c = np.bincount(table[s:s + (1 << 24)], minlength=len(sizes))
        if len(c) > len(sizes):
            return False
        sizes += c
    return sizes.tolist() == meta["bucket_sizes"]


@dataclass
class StreetGate:
    street: str
    n: int
    k: int
    buckets_nonempty: int
    monotone: bool
    river_intervals_disjoint: bool
    min_bucket_size: int
    equity_min: float
    equity_max: float
    ok: bool


def cluster_and_persist(
    street,
    *,
    k: int,
    bins: int = 30,
    out_dir: Path | None = None,
    n_iter: int = 100,
    n_init: int = 3,
    seed: int = 0,
    device=None,
    log: bool = True,
    overwrite: bool = False,
) -> StreetGate:
    """Cluster a street's extracted features into ``k`` buckets and persist the
    bucket table, centroids, and metadata. Returns the gate result.

    A bucket table already in ``out_dir`` is kept: every blueprint built on it
    reads it, and a new clustering may label the situations differently. One
    built under these settings is returned as it is (its gate from the meta);
    under other settings this raises, unless ``overwrite``."""
    sid = _street_id(street)
    name = STREET_NAMES[sid]
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    p = _paths(out_dir, name)
    is_river = sid == 2

    if not p["features"].exists():
        raise FileNotFoundError(f"no features for {name}; run extract_features first ({p['features']})")
    feats = np.lib.format.open_memmap(p["features"], mode="r")
    n = feats.shape[0]
    k_asked = k
    # k-means' settings (flop, turn); the river's clustering is exact or 1-D, seedless.
    kmeans = None if is_river else dict(seed=int(seed), n_init=int(n_init), max_iter=int(n_iter))
    if p["buckets"].exists() and not overwrite:
        if not p["meta"].exists():  # the meta is written last: a build that stopped part-way
            raise FileExistsError(f"{p['buckets']} has no {p['meta'].name} (a build stopped before its end): "
                                  f"use another out_dir, or rebuild (build_buckets.py --force)")
        meta = json.loads(p["meta"].read_text())
        built = dict(n=meta.get("n"), k=meta.get("k_requested", meta.get("k")), bins=meta.get("bins"),
                     kmeans=meta.get("kmeans", kmeans))  # older metas did not record it
        asked = dict(n=int(n), k=k_asked, bins=0 if is_river else bins, kmeans=kmeans)
        if built == asked:
            if not _table_matches(p["buckets"], meta):
                raise FileExistsError(f"{p['buckets']} is not the table its meta describes (its n and bucket sizes; "
                                      f"an older build stopped part-way?): rebuild it (build_buckets.py --force)")
            if log:
                unrecorded = "kmeans" not in meta and not is_river
                print(f"[{name}] buckets already built ({meta.get('created')}) under these settings"
                      + (" (its k-means seed, restarts and iteration cap unrecorded; --force rebuilds)" if unrecorded else "")
                      + ": kept")
            return StreetGate(**meta["gate"])
        raise FileExistsError(
            f"{p['buckets']} holds {name} buckets built under {built}, not {asked}; every blueprint on "
            f"these tables reads them. Use another out_dir, or overwrite (build_buckets.py --force)"
        )
    # Flop/turn need k <= N; the river caps k to its distinct-value count below,
    # so a large requested k there is fine (it becomes the exact abstraction).
    if not is_river and k > n:
        raise ValueError(f"k={k} exceeds situation count {n}")

    dev = pick_device(device)
    t0 = time.perf_counter()
    if is_river:
        # River equity vs a random opponent is (wins + ties/2)/C(45,2), i.e. a
        # multiple of 1/990 — a DISCRETE feature with only ~1981 possible values.
        # So the river cannot have more buckets than distinct values, and at the
        # cap it needs no lossy clustering at all: one bucket per distinct equity
        # is the *exact*, information-lossless river abstraction. Below the cap we
        # group with 1-D k-means.
        eq = np.asarray(feats)
        distinct = np.unique(eq)
        n_distinct = int(distinct.size)
        if k >= n_distinct:
            if log and k > n_distinct:
                print(f"[{name}] only {n_distinct} distinct equity values exist; "
                      f"capping k {k} -> {n_distinct} (exact, lossless river abstraction)")
            k = n_distinct
            labels = torch.as_tensor(
                np.searchsorted(distinct, eq).astype(np.int64), device=dev
            )
            centroids = torch.as_tensor(distinct.reshape(-1, 1), device=dev)
            result = KMeansResult(labels, centroids, 0.0, 1, True)
        else:
            result = kmeans_1d(eq, k, device=dev)
    else:
        result = kmeans_emd(feats, k, n_iter=n_iter, n_init=n_init, seed=seed, device=dev)
    if log:
        print(f"[{name}] clustered {n:,} -> {k} buckets in {time.perf_counter() - t0:.1f}s "
              f"(inertia {result.inertia:.4g}, {result.n_iter} iters)")

    labels = result.labels.to(torch.int64)
    means = _bucket_mean_equity(result.centroids, is_river)

    # Relabel buckets in ascending mean-equity order so bucket id is monotone in
    # strength (the property the gate checks and the solver benefits from).
    order = np.argsort(means, kind="stable")
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    labels_np = remap[labels.cpu().numpy()].astype(np.uint16)
    centroids_np = result.centroids.cpu().numpy()[order]
    means = means[order]

    # Persist: the old meta goes first and the new one comes last, so a stop
    # in between leaves tables without a meta (refused), never a meta over
    # other tables (kept).
    p["meta"].unlink(missing_ok=True)
    _save_npy(p["buckets"], labels_np)
    _save_npy(p["centroids"], centroids_np)

    sizes = np.bincount(labels_np, minlength=k)
    # For the river, verify buckets are genuinely disjoint equity intervals
    # (the exact 1-D monotonicity property), computed from the real data.
    river_minmax = None
    if is_river:
        eq = np.asarray(feats)
        river_minmax = _bucket_extrema(eq, labels_np, k)
    gate = gate_street(sid, labels_np, means, sizes, is_river, river_minmax=river_minmax)
    meta = {
        "street": name,
        "n": int(n),
        "k": int(k),
        "k_requested": int(k_asked),
        "bins": 0 if is_river else int(bins),
        "kmeans": kmeans,
        "is_river": is_river,
        "inertia": float(result.inertia),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "bucket_mean_equity": means.tolist(),
        "bucket_sizes": sizes.tolist(),
        "gate": asdict(gate),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_atomic(p["meta"], json.dumps(meta, indent=2))
    if log:
        status = "PASS" if gate.ok else "FAIL"
        print(f"[{name}] gate {status}: {gate.buckets_nonempty}/{k} non-empty, "
              f"monotone={gate.monotone}, equity {gate.equity_min:.3f}..{gate.equity_max:.3f}, "
              f"min bucket {gate.min_bucket_size:,}")
    return gate


def _bucket_extrema(values, labels, k: int) -> np.ndarray:
    """Exact per-bucket min/max in O(N + K), including NaN for empty buckets."""
    out = np.empty((k, 2), dtype=np.float64)
    out[:, 0] = np.inf
    out[:, 1] = -np.inf
    np.minimum.at(out[:, 0], labels, values)
    np.maximum.at(out[:, 1], labels, values)
    out[out[:, 0] == np.inf] = np.nan
    return out


def gate_street(street, labels_np, means, sizes, is_river, river_minmax=None) -> StreetGate:
    """Bucket-count + monotone-equity gate over a persisted assignment."""
    sid = _street_id(street)
    k = len(means)
    nonempty = int((sizes > 0).sum())
    # After ascending-mean relabeling, means must be non-decreasing.
    monotone = bool(np.all(np.diff(means) >= -1e-6))
    # River buckets are equity intervals, so consecutive buckets' ranges must not
    # overlap: max(bucket i) <= min(bucket i+1). An exact monotonicity check
    # unique to the 1-D case; N/A (reported True) for the 2-D EMD streets.
    disjoint = True
    if is_river and river_minmax is not None:
        hi = river_minmax[:-1, 1]
        lo = river_minmax[1:, 0]
        valid = ~(np.isnan(hi) | np.isnan(lo))
        disjoint = bool(np.all(hi[valid] <= lo[valid] + 1e-6))
    ok = nonempty == k and monotone and int(sizes.min()) > 0 and (disjoint or not is_river)
    return StreetGate(
        street=STREET_NAMES[sid],
        n=int(len(labels_np)),
        k=int(k),
        buckets_nonempty=nonempty,
        monotone=monotone,
        river_intervals_disjoint=disjoint,
        min_bucket_size=int(sizes.min()),
        equity_min=float(means.min()),
        equity_max=float(means.max()),
        ok=bool(ok),
    )


def build_street(
    street,
    *,
    k: int,
    bins: int = 30,
    out_dir: Path | None = None,
    chunk: int = 4096,
    limit: int | None = None,
    n_iter: int = 100,
    n_init: int = 3,
    seed: int = 0,
    device=None,
    log: bool = True,
    overwrite: bool = False,
) -> StreetGate:
    """Full pipeline for one street: extract -> cluster -> persist -> gate."""
    extract_features(street, bins=bins, out_dir=out_dir, chunk=chunk, limit=limit, log=log)
    return cluster_and_persist(
        street, k=k, bins=bins, out_dir=out_dir,
        n_iter=n_iter, n_init=n_init, seed=seed, device=device, log=log, overwrite=overwrite,
    )


# --- play-time lookup ------------------------------------------------------

_TABLE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def load_bucket_table(street, out_dir: Path | None = None) -> np.ndarray:
    """Load (and cache) a persisted bucket table, indexed by dense index."""
    sid = _street_id(street)
    name = STREET_NAMES[sid]
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    key = (str(out_dir), sid)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = np.load(_paths(out_dir, name)["buckets"], mmap_mode="r")
    return _TABLE_CACHE[key]


def bucket_of(street, cards, out_dir: Path | None = None) -> int:
    """Bucket id of a concrete hand: ``table[canonical_index(street, cards)]``.

    ``cards`` is the full hand (hole first, then board) as 0..52 indices; any
    suit permutation maps to the same bucket.
    """
    sid = _street_id(street)
    idx = _engine.canonical_index(sid, list(cards))
    return int(load_bucket_table(sid, out_dir)[idx])
