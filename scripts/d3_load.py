# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D3's memory-mapped loader (records/d3/PREREG.md 5.0.3, section 8) and the load's measurements.

`turn_net.load_many` reads every shard into host arrays: 18 KB a row, 34 GB for the 1.9M training rows of D3 plus a
second float16 copy of the ranges in `Pools`. `load_mapped` returns the same dict with `values` and `ranges` as `Col`s:
read-only proxies over the shards' stored `.npy` members (np.savez does not compress), gathered a batch of rows at a
time. Everything else (index, source, expl, board, half, stack, cell, dataset) is small and stays in memory.

A `Col` indexes like the array it stands for (an int, a slice, an array of rows), applies the D1 tail's patches as an
overlay (`col[i] = row`), streams its bytes in the array's order (`chunks`, so `turn_net.fingerprint` is the same hash)
and is never turned into one array: `np.asarray(col)` raises. `turn_net.Pools` prepares a mapped column's float16
ranges a pool at a time (bit for bit what it prepares in one go). Reads are the page cache's: the working set a run
reports includes mapped pages, which Windows reclaims under pressure; `memory()` reports the private bytes beside it.

    python scripts/d3_load.py measure --data DIR [DIR ...]       # load, fingerprint, one pool of ranges, memory
    python scripts/d3_load.py check --data runs/turn-data        # d3-check: N1 again, bit for bit (GPU, ~50 min)
    python scripts/d3_load.py recompare --data runs/turn-data    # the finished d3-check judged by tensors + history (Amendment 2; CPU)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import turn_net as tn  # noqa: E402

MIXED = "interleave-1:1"  # turn_data's generate.json layout key for D3's rows
HARVEST_ONLY = "harvest-only"  # ... and for S0a's labelled queries (every row harvested)
SITUATION_COLUMNS = ("board", "half", "stack", "cell")


# ------------------------------------------------------------------ the shards' columns


def npz_members(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """`names`' arrays of an .npz: a stored (uncompressed) .npy member is memory-mapped in place, a compressed one is
    read. Nothing is read of a mapped member until it is indexed."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for name in names:
            info = z.getinfo(name + ".npy")
            if info.compress_type != zipfile.ZIP_STORED:
                with z.open(info) as m:
                    out[name] = np.lib.format.read_array(m)
                continue
            f.seek(info.header_offset)
            head = f.read(30)
            if head[:4] != b"PK\x03\x04":
                raise ValueError(f"{path.name}: no local header for {name}")
            skip = struct.unpack("<HH", head[26:30])  # the local name's and extra field's lengths
            f.seek(info.header_offset + 30 + sum(skip))
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"{path.name}: {name} is .npy format {version}")
            if fortran or dtype.hasobject:
                raise ValueError(f"{path.name}: {name} is not a plain C-order array")
            if int(np.prod(shape, dtype=np.int64)) == 0:
                out[name] = np.empty(shape, dtype)
            else:
                out[name] = np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape)
    return out


class Col:
    """Rows [lo, hi) of the concatenation of `parts` (read-only arrays of rows), plus the overlay's replaced rows."""

    def __init__(self, parts: list[np.ndarray], starts: np.ndarray | None = None, lo: int = 0, hi: int | None = None,
                 overlay: dict | None = None):
        self.parts = parts
        self.starts = starts if starts is not None else np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
        self.lo, self.hi = lo, int(self.starts[-1]) if hi is None else hi
        self.overlay = {} if overlay is None else overlay  # global row -> replacement row
        self.dtype = parts[0].dtype
        self.tail = tuple(int(x) for x in parts[0].shape[1:])
        for p in parts:
            assert p.dtype == self.dtype and tuple(p.shape[1:]) == self.tail, "the shards disagree on dtype or shape"

    # --- what a consumer asks of an array
    chunks_rows = 1 << 14

    @property
    def shape(self) -> tuple:
        return (self.hi - self.lo,) + self.tail

    @property
    def ndim(self) -> int:
        return 1 + len(self.tail)

    def __len__(self) -> int:
        return self.hi - self.lo

    def __array__(self, dtype=None, copy=None):
        raise TypeError("a mapped column is never turned into one array: index its rows, or stream `chunks()`")

    def __getitem__(self, key):
        n = len(self)
        if isinstance(key, slice):
            a, b, step = key.indices(n)
            if step != 1:
                raise IndexError("a mapped column slices with step 1 only")
            return Col(self.parts, self.starts, self.lo + a, self.lo + max(a, b), self.overlay)
        if isinstance(key, (int, np.integer)):
            if not -n <= key < n:
                raise IndexError(f"row {key} of {n}")
            return self._gather(np.array([self.lo + int(key) % n], dtype=np.int64))[0]
        key = np.asarray(key)
        if key.size == 0 and key.dtype.kind == "f":  # `col[[]]`: numpy reads an empty list as float64 and returns no rows
            key = key.astype(np.int64)
        if key.ndim == 0 and key.dtype.kind in "iu":
            return self[int(key)]
        if key.dtype == bool:
            if key.shape != (n,):
                raise IndexError("a boolean row mask must be one value per row")
            key = np.flatnonzero(key)
        if key.dtype.kind not in "iu":
            raise TypeError(f"rows are indexed by integers, not {key.dtype}")
        key = key.astype(np.int64)
        if key.ndim != 1:
            raise IndexError("a mapped column takes one-dimensional row indices")
        if len(key) and (key.min() < -n or key.max() >= n):
            raise IndexError(f"a row outside 0..{n - 1}")
        return self._gather(self.lo + np.where(key < 0, key + n, key))

    def __setitem__(self, key, value) -> None:
        """A row replaced (the D1 tail's patch): kept in the overlay, the shards untouched."""
        if not isinstance(key, (int, np.integer)):
            raise TypeError("a mapped column replaces one row at a time")
        n = len(self)
        if not -n <= key < n:
            raise IndexError(f"row {key} of {n}")
        row = np.array(value, dtype=self.dtype)
        if row.shape != self.tail:
            raise ValueError(f"a row of shape {row.shape}, the column's {self.tail}")
        self.overlay[self.lo + int(key) % n] = row

    def _gather(self, g: np.ndarray) -> np.ndarray:
        """Global rows `g` (any order, repeats allowed), the overlay applied."""
        out = np.empty((len(g),) + self.tail, self.dtype)
        shard = np.searchsorted(self.starts, g, side="right") - 1
        order = np.argsort(shard, kind="stable")
        cut = np.flatnonzero(np.diff(shard[order])) + 1
        for at in np.split(order, cut) if len(g) else []:
            s = int(shard[at[0]])
            out[at] = self.parts[s][g[at] - self.starts[s]]
        if self.overlay and len(g):
            keys = np.fromiter(self.overlay, np.int64, len(self.overlay))
            for p in np.flatnonzero(np.isin(g, keys)):
                out[p] = self.overlay[int(g[p])]
        return out

    def chunks(self):
        """The column's rows in order, a shard at a time, the overlay applied (so their bytes are the array's)."""
        first = int(np.searchsorted(self.starts, self.lo, side="right")) - 1
        for s in range(first, len(self.parts)):
            a, b = max(int(self.starts[s]), self.lo), min(int(self.starts[s + 1]), self.hi)
            if a >= self.hi:
                break
            if b <= a:
                continue
            x = self.parts[s][a - self.starts[s]:b - self.starts[s]]
            hit = [r for r in self.overlay if a <= r < b]
            if hit:
                x = np.array(x)
                for r in hit:
                    x[r - a] = self.overlay[r]
            yield x

    @staticmethod
    def concat(cols: list["Col"]) -> "Col":
        """One column of whole columns one after the other, their overlays carried over."""
        assert all(c.lo == 0 and c.hi == int(c.starts[-1]) for c in cols), "only whole columns concatenate"
        out = Col([p for c in cols for p in c.parts])
        at = 0
        for c in cols:
            out.overlay.update({at + r: v for r, v in c.overlay.items()})
            at += len(c)
        return out


# ------------------------------------------------------------------ the loader


def _situation_columns(data: Path, meta: dict, source: np.ndarray, index: np.ndarray, harvest: Path | None,
                       verify_harvest: bool) -> dict:
    """board, half, stack and cell of every row of a folder: its situation `index` in situations.npz, and for a mixed
    folder's harvested rows (source 2) the harvest's row `index`."""
    sit = np.load(data / "situations.npz")
    rec = source != 2
    har = None
    if (~rec).any():
        if meta.get("layout") not in (MIXED, HARVEST_ONLY):
            raise ValueError(f"{data}: rows of source 2 in a folder whose generate.json has no {MIXED!r} layout")
        if harvest is None:
            raise SystemExit(f"{data} is a mixed run: pass its harvest (generate.json harvest {str(meta.get('harvest'))[:12]})")
        if verify_harvest:
            from turn_data import sha256_tree

            got = sha256_tree(harvest)
            if got != meta.get("harvest"):
                raise SystemExit(f"{harvest} is not this run's harvest: sha256 {got[:12]}, generate.json {str(meta.get('harvest'))[:12]}")
        har = {k: np.load(harvest / f"{k}.npy") for k in SITUATION_COLUMNS if (harvest / f"{k}.npy").exists()}
    out = {}
    for k in SITUATION_COLUMNS:
        x = sit[k][index[rec]]
        y = np.empty((len(index),) + x.shape[1:], x.dtype)
        y[rec] = x
        if har is not None:
            if k not in har:
                raise SystemExit(f"{harvest}: no {k}.npy")
            h = har[k][index[~rec]]
            if not np.array_equal(h.astype(x.dtype).astype(h.dtype), h):
                raise ValueError(f"{harvest}: {k} does not survive the situations' dtype {x.dtype}")
            y[~rec] = h.astype(x.dtype)
        out[k] = y
    return out


def load_mapped(dirs: list[Path], max_shards: int | None = None, log=print, harvest: Path | None = None,
                verify_harvest: bool = True) -> dict:
    """`turn_net.load_many(dirs)`'s dict, `values` and `ranges` mapped (`Col`). The same checks, the same patches (a
    folder's own `resolve-q3`), the same row order, dtypes and `dataset`. `harvest`: the directory a mixed folder's
    harvested rows were generated from (checked against its generate.json). `max_shards`: None, one cap for every
    folder, or a list with one cap (int or None) per folder (a futility look: all of D1 and D1b, the first shards of D1c)."""
    caps = list(max_shards) if isinstance(max_shards, (list, tuple)) else [max_shards] * len(dirs)
    if len(caps) != len(dirs):
        raise ValueError(f"max_shards has {len(caps)} entries for {len(dirs)} folders")
    found = []
    for data, cap in zip(dirs, caps):
        every = [p for p in sorted(data.glob("shard-*.npz")) if not p.name.endswith(".tmp.npz")]
        if not every[:cap]:
            raise FileNotFoundError(f"no finished shards in {data}")
        found.append((data, every, every[:cap]))
    small = ("index", "source", "expl")
    cols = {k: [] for k in small}
    parts = {"values": [], "ranges": []}
    folder_of = []  # per folder: (first shard, last shard + 1) in `parts`
    for data, _, paths in found:
        first = len(parts["values"])
        for p in paths:
            z = npz_members(p, ("index", "source", "expl", "values", "ranges"))
            m = len(z["index"])
            v, r = z["values"], z["ranges"]
            if not (np.isfinite(v).all() and np.isfinite(r).all() and (r >= 0).all() and (r.sum(-1) > 0).all()):
                raise ValueError(f"{p.name}: a non-finite value, or a range that is non-finite, negative or of mass 0")
            for k in small:
                assert len(z[k]) == m, (p.name, k)
                assert not cols[k] or z[k].dtype == cols[k][0].dtype, (p.name, k, "dtype differs from the first shard's")
                cols[k].append(np.asarray(z[k]))
            for k in parts:
                assert len(z[k]) == m, (p.name, k)
                parts[k].append(z[k])
        folder_of.append((first, len(parts["values"])))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    sizes = [sum(len(c) for c in cols["index"][a:b]) for a, b in folder_of]
    folders = {k: [] for k in parts}
    start = 0
    for n, ((data, every, paths), (a, b), size) in enumerate(zip(found, folder_of, sizes)):
        end = start + size
        view = dict(index=d["index"][start:end], source=d["source"][start:end], expl=d["expl"][start:end])  # views: a patch writes through
        for k in parts:
            view[k] = Col(parts[k][a:b])
        for line in tn.patch(view, data, complete=len(paths) == len(every)):
            log(line)
        for k in parts:
            folders[k].append(view[k])
        meta_path = data / "generate.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        for k, x in _situation_columns(data, meta, view["source"], view["index"], harvest, verify_harvest).items():
            if k not in d:
                d[k] = np.empty((len(d["index"]),) + x.shape[1:], x.dtype)
            assert x.dtype == d[k].dtype, (data, k)
            d[k][start:end] = x
        start = end
    for k in parts:
        d[k] = Col.concat(folders[k])
    if len(found) > 1:
        d["dataset"] = np.repeat(np.arange(len(found), dtype=np.int8), sizes)
    d["board"] = d["board"].astype(np.int64)
    return d


# ------------------------------------------------------------------ measurements


def memory() -> dict:
    """This process's host memory in GiB: the peak working set (mapped shard pages included, reclaimable), the peak
    commit and the private bytes now (what only this process holds)."""
    g = 2.0 ** 30
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                       "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        proc = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        if not ctypes.windll.psapi.GetProcessMemoryInfo(proc, ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return dict(peak_working_set=c.PeakWorkingSetSize / g, peak_commit=c.PeakPagefileUsage / g, private=c.PrivateUsage / g)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / g
    return dict(peak_working_set=peak, peak_commit=peak, private=peak)


def cmd_measure(a) -> int:
    """Load the folders mapped, fingerprint them, prepare one pool of float16 ranges, and report the host memory."""
    import torch

    dirs = [Path(x) for x in a.data]
    t0 = time.time()
    d = load_mapped(dirs, harvest=Path(a.harvest) if a.harvest else None)
    t1 = time.time()
    n = len(d["index"])
    fp = tn.fingerprint(d)
    t2 = time.time()
    rows = np.sort(np.random.default_rng(0).permutation(n)[: a.pool])
    r16 = tn.Batches.prepared_ranges(d, rows)
    v = torch.from_numpy(d["values"][rows])
    t3 = time.time()
    mem = memory()
    res = dict(rows=n, fingerprint=fp, load_seconds=round(t1 - t0, 1), fingerprint_seconds=round(t2 - t1, 1),
               pool_rows=len(rows), pool_seconds=round(t3 - t2, 1), pool_gib=round((r16.nbytes + v.numel() * 4) / 2.0 ** 30, 2),
               **{k: round(x, 2) for k, x in mem.items()})
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


N1_TAG = "d2b-w2048-wd0.1"  # the net D2b selected: the recipe `check` repeats
N1_SHA = "54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72"  # records/d2b/gate-a.json "N1"
CHECK_TAG = "d3-check"
CHECK_JSON = Path(__file__).resolve().parents[1] / "records" / "d3" / "check.json"


def net_files_equal(path: Path, other: Path) -> dict:
    """Two `net.pt` files compared as nets (records/d3/PREREG.md Amendment 2): width, depth and every tensor of `state` bit-equal,
    the meta keys both hold equal, and the keys only one holds (the three D3-SWEEP options, written by code newer than N1) at
    their off defaults. The file's bytes are reported by the caller, not compared here."""
    import torch

    a, b = (torch.load(x, map_location="cpu", weights_only=False) for x in (path, other))
    sa, sb = a["state"], b["state"]
    tensors = set(sa) == set(sb) and all(sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]) for k in sa)
    shared = {k for k in a["meta"] if k in b["meta"]}
    extra = {k: v for m, o in ((a["meta"], b["meta"]), (b["meta"], a["meta"])) for k, v in m.items() if k not in o}
    shape = (a["width"], a["depth"]) == (b["width"], b["depth"])
    meta = all(a["meta"][k] == b["meta"][k] for k in shared)
    at_defaults = all(k in tn.OPTION_KEYS and type(v) is type(tn.CFG_DEFAULTS[k]) and v == tn.CFG_DEFAULTS[k] for k, v in extra.items())
    return dict(tensors=tensors, tensor_count=len(sa), shape=shape, meta=meta, extra_meta=extra, extra_at_defaults=at_defaults,
                ok=bool(tensors and shape and meta and at_defaults))


def check_compare(rec: dict, n1: dict, sha: str, net: dict) -> dict:
    """`rec` against N1's record and net: the best epoch and its full float, every epoch's train and selection loss, the
    net as a net (`net_files_equal`: Amendment 2 of the D3 PREREG; the file's own sha256 is reported as `file_sha256`, not required).
    `reproduced` is all of them equal (the clock fields differ and are not compared)."""
    keys = ("epoch", "train_loss", "val_loss", "val_mae")
    hist = lambda r: [{k: h[k] for k in keys} for h in r["history"]]  # noqa: E731
    same = dict(best_epoch=rec["best"]["epoch"] == n1["best"]["epoch"], best_val_loss=rec["best"]["val_loss"] == n1["best"]["val_loss"],
                history=hist(rec) == hist(n1), net=bool(net["ok"]),
                config=all(rec[k] == n1[k] for k in ("width", "depth", "wd", "seed", "data", "train_samples", "val_samples", "pools",
                                                      "epochs", "batch", "lr")))
    return dict(**same, reproduced=all(same.values()), file_sha256=sha == N1_SHA, net_detail=net)


def aside(path: Path) -> Path | None:
    """Move `path` (a file or a folder) to `<name>-void-<n>` beside it, keeping it; None if there is nothing there."""
    if not path.exists():
        return None
    n = 1
    while path.with_name(f"{path.name}-void-{n}").exists():
        n += 1
    return path.rename(path.with_name(f"{path.name}-void-{n}"))


def cmd_check(a) -> int:
    """`d3-check` (PREREG 5.0.3): N1's recipe trained again on D1 + D1b through `load_mapped`, compared with N1 bit for bit."""
    import hashlib

    import livesearch as ls

    runs = Path(a.data).parent
    tn.NETS = runs / "turn-net"  # `turn_net.main` sets it from --data; this does the same
    commit = tn._commit()
    if not tn.d2b_committed(commit):
        raise SystemExit(f"uncommitted changes under src or scripts, or git could not say ({commit!r}): commit first (the record holds the commit)")
    n1 = json.loads((tn.NETS / N1_TAG / "record.json").read_text())
    if hashlib.sha256((tn.NETS / N1_TAG / "net.pt").read_bytes()).hexdigest() != N1_SHA:
        raise SystemExit(f"{tn.NETS / N1_TAG / 'net.pt'} is not N1 ({N1_SHA})")
    if getattr(a, "redo", False):  # a finished or cut d3-check is kept beside the new one, never trained over or reused
        for old in (tn.NETS / CHECK_TAG, CHECK_JSON):
            if aside(old):
                print(f"moved aside: {old}")
    elif (tn.NETS / CHECK_TAG).exists() or CHECK_JSON.exists():
        raise SystemExit(f"{tn.NETS / CHECK_TAG} or {CHECK_JSON} exists: `train` would reuse that net as it stands, so a rerun "
                         f"could report REPRODUCED without training; `check --redo` moves both aside (kept) and trains again")
    tn.d2b_check_files(runs / tn.D2B_DIRS[1], tn.D2B_RECORDS / "d1b-manifest.sha256")
    t0 = time.time()
    d = tn.d2b_views(load_mapped([runs / x for x in tn.D2B_DIRS]))
    t_load = time.time() - t0
    if tn.fingerprint(d["d1"]) != tn.LEAFNET_DATA:
        raise SystemExit(f"D1's fingerprint is {tn.fingerprint(d['d1'])}, registered {tn.LEAFNET_DATA}")
    tn.d2b_refuse_other_data(d)
    parts = {k: tn.split(v["board"]) for k, v in d.items()}
    mem_loaded = memory()
    arm = next(x for x in tn.D2B_CANDIDATES if x["tag"] == N1_TAG)
    arm = dict(arm, tag=CHECK_TAG)
    seen, clients = [], set()

    def look():
        g = ls.gpu_state()
        seen.append(g)
        new = sorted(set(g["apps"]) - clients)
        if new:
            print(f"  {CHECK_TAG}: GPU clients new at this look (a compute job voids the run, read them): {new}", flush=True)
            clients.update(new)
        if len(seen) == 1 and (g["peers"] or tn.d2b_blind(g)):
            raise SystemExit(f"another process of this Python on the GPU ({g['peers']}), or nvidia-smi could not say "
                             f"(util {g['util']}, apps {g['apps']}): nothing trained; rerun when that is settled")
        if g["peers"]:
            raise SystemExit(f"another process of this Python on the GPU ({g['peers']}): this run is VOID (PREREG 7); record it, then rerun")

    rec = tn.d2b_train_arm(d, parts, arm, {}, look, None, commit=commit, device=getattr(a, "device", "cuda"))
    mem_end = memory()
    sha = hashlib.sha256((tn.NETS / CHECK_TAG / "net.pt").read_bytes()).hexdigest()
    res = dict(check_compare(rec, n1, sha, net_files_equal(tn.NETS / N1_TAG / "net.pt", tn.NETS / CHECK_TAG / "net.pt")), tag=CHECK_TAG, commit=commit, n1_commit=n1["commit"], sha256_check=sha, sha256_n1=N1_SHA,
               rows=len(d["union"]["index"]), train_rows=rec["train_samples"], load_seconds=round(t_load, 1), seconds=rec["history"][-1]["seconds"],
               upload_seconds=rec.get("upload_seconds"), memory_after_load_gib=mem_loaded, memory_end_gib=mem_end,
               gpu_looks=len(seen), gpu_apps=sorted(clients), gpu_peers=sorted({str(p) for g in seen for p in g["peers"] or []}),
               gpu_unknown=sum(tn.d2b_blind(g) for g in seen), torch=n1.get("torch"), torch_now=rec.get("torch"))
    tn._write(CHECK_JSON, json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "gpu_apps"}, indent=1))
    print(f"d3-check: {'REPRODUCED N1 bit for bit' if res['reproduced'] else 'NOT REPRODUCED'}; host peak working set "
          f"{mem_end['peak_working_set']:.1f} GiB, peak commit {mem_end['peak_commit']:.1f} GiB at {res['train_rows']} train rows")
    return 0 if res["reproduced"] else 1


def cmd_recompare(a) -> int:
    """Amendment 2: judge the finished d3-check again by the net's tensors and history, not the file's bytes. Nothing is trained or
    loaded to the GPU. The earlier check.json is moved aside (kept) and the new one carries its run's fields."""
    import hashlib

    runs = Path(a.data).parent
    tn.NETS = runs / "turn-net"
    commit = tn._commit()
    if not tn.d2b_committed(commit):
        raise SystemExit(f"uncommitted changes under src or scripts, or git could not say ({commit!r}): commit first")
    old = json.loads(CHECK_JSON.read_text())
    if old.get("recompared"):
        raise SystemExit(f"{CHECK_JSON} is already a recompare")
    n1 = json.loads((tn.NETS / N1_TAG / "record.json").read_text())
    rec = json.loads((tn.NETS / CHECK_TAG / "record.json").read_text())
    for tag, want in ((N1_TAG, N1_SHA), (CHECK_TAG, old["sha256_check"])):
        if hashlib.sha256((tn.NETS / tag / "net.pt").read_bytes()).hexdigest() != want:
            raise SystemExit(f"{tn.NETS / tag / 'net.pt'} is not the net the check recorded ({want})")
    new = dict(old, **check_compare(rec, n1, old["sha256_check"], net_files_equal(tn.NETS / N1_TAG / "net.pt", tn.NETS / CHECK_TAG / "net.pt")),
               recompared=True, run_commit=old["commit"], commit=commit, run_reproduced=old["reproduced"], run_sha256=old.get("sha256"))
    new.pop("sha256", None)
    if not new["reproduced"]:  # a failed judgement leaves the run's check.json as it is
        print(json.dumps({k: new[k] for k in ("best_epoch", "best_val_loss", "history", "config", "net", "net_detail")}, indent=1))
        print("d3-check recompared (tensors and history): NOT REPRODUCED; check.json untouched")
        return 1
    staged = CHECK_JSON.with_name(CHECK_JSON.name + ".new")
    tn._write(staged, json.dumps(new, indent=1))  # written first: a failure here leaves the run's check.json where it was
    print(f"moved aside: {aside(CHECK_JSON)}")
    staged.rename(CHECK_JSON)
    print(f"d3-check recompared (tensors and history): REPRODUCED N1; file bytes equal: {new['file_sha256']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--data", nargs="+", required=True)
    m.add_argument("--harvest", default=None)
    m.add_argument("--pool", type=int, default=65536, help="rows prepared and gathered for the pool measurement")
    m.add_argument("--out", default=None)
    c = sub.add_parser("check", help="d3-check: N1 retrained through the mapped loader (GPU, ~50 min)")
    c.add_argument("--data", required=True, help="D1's folder; D1b's is beside it")
    c.add_argument("--redo", action="store_true", help="move an earlier d3-check (net folder and check.json) aside, then train again")
    r = sub.add_parser("recompare", help="Amendment 2: judge the finished d3-check by tensors and history (CPU, seconds)")
    r.add_argument("--data", required=True, help="D1's folder; D1b's is beside it")
    a = ap.parse_args()
    return {"measure": cmd_measure, "check": cmd_check, "recompare": cmd_recompare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
