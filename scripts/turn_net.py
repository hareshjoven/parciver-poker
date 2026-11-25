# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The turn-start value net (FLOPSEARCH.md, D2), trained on `turn_data.py generate`'s shards.

  baseline  the check-down values of the held-out samples (G2's baseline), cached
            as `checkdown-val.npz` next to the shards
  train     one net: --width --depth [--fraction of the training samples]
  sweep     width x depth on selection loss (the choice goes to sweep.json)
  final     the chosen architecture retrained longer; the net with the lower
            selection loss, this or the sweep's, is the final one (final.json)
  curve     the chosen architecture at 25 / 50 / 100% of the training samples
  g2        gate G2 on the gate boards: the final net's range-weighted MAE must
            be under half the check-down baseline's (a cached value whose
            sample was generated again with other inputs is refused)
  leafnet   LEAF-NET's grid (records/leaf-net/PREREG.md): canonical {off, on} x
            weight decay {0, 0.01, 0.1, 1} at D2's final settings; D2's recipe
            first, which must reproduce N0's record; the pick (lowest selection
            loss) goes to leafnet.json
  gate-a    LEAF-NET's Gate A: the pick against N0 on the gate boards, both
            averaged over each board's suit stabilizer H_b
  d2b       D2b (records/d2b/PREREG.md), a stage at a time on D1 + D1b:
            data    the pinned files checked; each part's and the union's fingerprint
            train   d2b-check (must reproduce N0's record), the candidates, the
                    reported arms; finished nets are reused (records/d2b/nets.json)
            select  N1 = the candidate with the lower loss on the union's selection
                    half, N0 and P scored too (records/d2b/selection.json)
            gate-a  N1 against N0 on D1b's gate half (records/d2b/gate-a.json)

5% of the suit-canonical boards are held out (no held-out board is a suit
relabelling of a training board, so suit augmentation never crosses the split).
Half of them, the selection boards, pick every epoch and architecture; the
other half, the gate boards, are used only by `g2`, so the gate measures a net
that was not chosen on them (FLOPSEARCH.md, D2 protocol). Nets and records go
to runs/turn-net/<tag>/. No dashboard (the author paused them).

The D1 tail's kept re-solves (records/d1-tail/PREREG15.md: `resolve-q3/patch*.npz`
next to the shards) replace their rows' values and expl as the shards load. The
data's fingerprint then changes, so no record made before a patch is taken for
the patched data's.

A run that stops part-way leaves nothing torn: net.pt and the json records are
replaced in one step, record.json is rewritten every epoch, and `sweep` and
`curve` skip the nets a previous run finished under the same settings and data
(--fresh retrains them). A net is not resumed mid-training: it starts again."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NETS = ROOT / "runs" / "turn-net"  # set from --data in main()
SPLIT_SEED = 20260929
HELD_OUT = 0.05  # of the canonical boards: [0, 0.025) selection, [0.025, 0.05) gate
TRAIN, SELECT, GATE = 0, 1, 2
RANGE_SCALE = 2.0 ** 14  # training ranges are stored in float16 times this (FLOPSEARCH.md D2 amendment)
CHUNK = 16384  # samples normalized at a time in `Batches`
# LEAF-NET (records/leaf-net/PREREG.md), fixed at registration
LEAFNET_DATA = "8df0fc79620a1820"  # D2's data: the grid refuses any other
LEAFNET_N0 = "final-w2048-d5-e200"  # the live net and D2's recipe
LEAFNET_WD = (0.0, 0.01, 0.1, 1.0)
LEAFNET_SETTINGS = dict(width=2048, depth=5, epochs=200, batch=2048, lr=1e-3, seed=0)
LEAFNET_RESAMPLES = 10_000
CFG_DEFAULTS = dict(canonical=False, wd=0.0, pools=1, dropout=0.0, ema=0.0, loss="huber")  # what a record from before LEAF-NET, D2b or D3-SWEEP was trained with
OPTION_KEYS = ("dropout", "ema", "loss")  # the three D3-SWEEP options a record names (CFG_DEFAULTS: off)
LOSSES = ("huber", "rw")  # D3-SWEEP: the plain Huber (N1's), or half plain + half range-weighted Huber
# D2b (records/d2b/PREREG.md), fixed at registration
D2B_RECORDS = ROOT / "records" / "d2b"
D2B_DIRS = ("turn-data", "turn-data-d1b")  # under the runs folder: D1, D1b (dataset 0, 1)
D2B_MANIFEST = "bd73786ef100f80ffe69fa5d48316d39344cee36c6f6c4510e8a0c691a1ff144"  # d1b-manifest.sha256's own sha256
D2B_UNION: str | None = "7d5a77c30aaad5d6"  # the union's fingerprint, by amendment after `d2b data`; training refuses any other
D2B_POOLS = 2  # the union does not fit on the card at once (13,576 B a row; "GPU pools")
D2B_CHECK = dict(tag="d2b-check", width=2048, data="d1", wd=0.0)
D2B_CANDIDATES = (dict(tag="d2b-w2048-wd0", width=2048, data="union", wd=0.0),
                  dict(tag="d2b-w2048-wd0.1", width=2048, data="union", wd=0.1))
D2B_REPORTED = (dict(tag="d2b-d1b-only", width=2048, data="d1b", wd=0.0),
                dict(tag="d2b-w1024-wd0", width=1024, data="union", wd=0.0),
                dict(tag="d2b-w1024-wd0.1", width=1024, data="union", wd=0.1))
D2B_MIN_GAIN = -0.05  # Gate A: the primary set's relative change must be at or below this
D2B_GPU_EVERY = 10  # epochs between looks at the GPU's other clients ("VOID and bugs")


from parciver.solver_watch import persist, write_atomic as _write  # noqa: E402  (never a torn file)


def fingerprint(d: dict) -> str:
    """The samples' sha256 (values, ranges and their situations): a record or
    a sweep made on other data is never taken for this data's, even with the
    same sample counts. Seconds on the full data, once per net."""
    h = hashlib.sha256()
    for k in ("values", "ranges", "board", "half", "stack"):
        if hasattr(d[k], "chunks"):  # a mapped column (d3_load.Col): the same bytes, a shard at a time
            h.update((k + " " + d[k].dtype.str + " " + str(d[k].shape) + chr(10)).encode())
            for c in d[k].chunks():
                h.update(memoryview(np.ascontiguousarray(c)).cast("B"))
            continue
        x = np.ascontiguousarray(d[k])
        h.update(f"{k} {x.dtype.str} {x.shape}\n".encode())
        h.update(memoryview(x).cast("B"))
    return h.hexdigest()[:16]


def row_digests(d: dict, rows: np.ndarray) -> np.ndarray:
    """[len(rows)] uint64: each row's check-down inputs (its ranges, board,
    half and stack) hashed. A cached check-down value is paired with its row
    by (index, source); the digest refuses the pairing when that row was
    generated again with other inputs. Per row, so adding shards keeps the
    cache."""
    out = np.empty(len(rows), np.uint64)
    for n, r in enumerate(rows):
        h = hashlib.blake2b(digest_size=8)
        for k in ("ranges", "board", "half", "stack"):
            h.update(np.ascontiguousarray(d[k][r]).tobytes())
        out[n] = int.from_bytes(h.digest(), "little")
    return out


def load(data: Path, max_shards: int | None = None, log=print) -> dict:
    """Every finished shard, with its situation's public state, and the D1
    re-solves patched over it (`patch`). A shard with a non-finite value, or a
    range that cannot be normalized, stops the load: one such sample turns every
    net trained on it to NaN."""
    return load_many([data], max_shards, log)


def load_many(dirs: list[Path], max_shards: int | None = None, log=print) -> dict:
    """`load` of each folder in turn, the rows one after the other (D2b). A row's
    `index` and `source` are its own folder's: its situation is looked up there
    and its folder's patches touch only that folder's rows. With two or more
    folders `dataset` holds each row's position in `dirs`."""
    found = []
    for data in dirs:
        every = [p for p in sorted(data.glob("shard-*.npz")) if not p.name.endswith(".tmp.npz")]
        if not every[:max_shards]:
            raise FileNotFoundError(f"no finished shards in {data}")
        found.append((data, every, every[:max_shards]))
    keys = ("index", "source", "values", "ranges", "expl")
    sizes = {}
    for _, _, paths in found:
        for p in paths:
            with np.load(p) as z:
                sizes[p] = len(z["index"])
    total = sum(sizes.values())
    d, at = {}, 0  # filled in place: concatenating the shards would hold every sample twice
    for _, _, paths in found:
        for p in paths:
            m = sizes[p]
            with np.load(p) as z:
                cols = {k: z[k] for k in keys}
            v, r = cols["values"], cols["ranges"]
            if not (np.isfinite(v).all() and np.isfinite(r).all() and (r >= 0).all() and (r.sum(-1) > 0).all()):
                raise ValueError(f"{p.name}: a non-finite value, or a range that is non-finite, negative or of mass 0")
            for k, x in cols.items():
                if k not in d:
                    d[k] = np.empty((total,) + x.shape[1:], x.dtype)
                assert x.dtype == d[k].dtype and len(x) == m, (p.name, k)
                d[k][at:at + m] = x
            at += m
    if len(found) > 1:
        d["dataset"] = np.empty(total, np.int8)
    start = 0
    for n, (data, every, paths) in enumerate(found):
        end = start + sum(sizes[p] for p in paths)
        view = {k: d[k][start:end] for k in keys}  # views: a patch writes through to `d`
        for line in patch(view, data, complete=len(paths) == len(every)):
            log(line)
        sit = np.load(data / "situations.npz")
        j = view["index"]
        for k in ("board", "half", "stack", "cell"):
            x = sit[k][j]
            if k not in d:
                d[k] = np.empty((total,) + x.shape[1:], x.dtype)
            assert x.dtype == d[k].dtype, (data, k)
            d[k][start:end] = x
        if len(found) > 1:
            d["dataset"][start:end] = n
        start = end
    d["board"] = d["board"].astype(np.int64)
    return d


def patch(d: dict, data: Path, complete: bool) -> list[str]:
    """Apply the D1 tail's kept re-solves (records/d1-tail/PREREG15.md:
    `resolve-q3/patch*.npz`, each with its .json) over the loaded rows in place.
    A patch row replaces the values and expl of the row with its (index,
    source), and only that row: the one it re-solved (the same ranges, and the
    stored e it beat). `complete`: every shard is loaded, so every patch row
    must be found. One line per patch applied."""
    paths = sorted((data / "resolve-q3").glob("patch*.npz"))
    if not paths:
        return []
    assert np.isin(d["source"], (0, 1)).all()
    key = d["index"].astype(np.int64) * 2 + d["source"]
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    done, out = set(), []
    for p in paths:
        meta = p.with_suffix(".json")
        if not meta.exists():
            raise ValueError(f"{p.name}: no {meta.name} (written after the patch: the patch may be torn)")
        meta = json.loads(meta.read_text())
        with np.load(p) as z:
            z = {k: z[k] for k in z.files}
        n = len(z["index"])
        if meta.get("rows") != n:
            raise ValueError(f"{p.name}: {n} rows, its record says {meta.get('rows')}")
        if n and (z["values"].shape[1:] != d["values"].shape[1:] or z["values"].dtype != d["values"].dtype):
            raise ValueError(f"{p.name}: values {z['values'].dtype} {z['values'].shape[1:]} per row, the shards' "
                             f"{d['values'].dtype} {d['values'].shape[1:]}")
        used = 0
        for r in range(n):
            i, s = int(z["index"][r]), int(z["source"][r])
            k = i * 2 + s
            if s not in (0, 1) or k in done:
                raise ValueError(f"{p.name}: row ({i}, {s}) is patched twice, or its source is not 0 or 1")
            done.add(k)
            lo, hi = np.searchsorted(sorted_key, k, "left"), np.searchsorted(sorted_key, k, "right")
            if hi - lo > 1:
                raise ValueError(f"{p.name}: row ({i}, {s}) is in the shards {hi - lo} times")
            if hi == lo:
                if complete:
                    raise ValueError(f"{p.name}: row ({i}, {s}) is not in the shards")
                continue
            row = order[lo]
            if d["expl"][row] != z["e_stored"][r] or not np.array_equal(d["ranges"][row], z["ranges"][r]):
                raise ValueError(f"{p.name}: row ({i}, {s}) is not the row it re-solved (stored e or ranges differ)")
            if not (z["expl"][r] < z["e_stored"][r] and np.isfinite(z["values"][r]).all()):
                raise ValueError(f"{p.name}: row ({i}, {s}): e {z['expl'][r]} does not beat the stored "
                                 f"{z['e_stored'][r]}, or a value is non-finite")
            d["values"][row], d["expl"][row] = z["values"][r], z["expl"][r]
            used += 1
        out.append(f"{p.name}: {used} of its {n} re-solves patched over the shards (commit {meta.get('head')})")
    return out


def canonical_codes(board: np.ndarray) -> np.ndarray:
    """[M] int: the suit-canonical board class of each row (vectorized
    `value_net.canonical_board`)."""
    from parciver import value_net as vn

    b = np.sort(vn.CARD_PERM[:, board], axis=-1)  # [24, M, 4]
    code = ((b[..., 0] * 52 + b[..., 1]) * 52 + b[..., 2]) * 52 + b[..., 3]
    return code.min(0)


def split(board: np.ndarray) -> np.ndarray:
    """[M] int8: each row's part, TRAIN, SELECT or GATE. A canonical board's
    seeded hash (splitmix64), as a fraction of 2^53, places it: under 2.5%
    selection, under 5% gate. A fixed property of the board, the same whatever
    subset of the data is loaded."""
    x = canonical_codes(board).astype(np.uint64) ^ np.uint64(SPLIT_SEED)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(11)).astype(np.float64) / 2.0**53
    return np.where(u < HELD_OUT / 2, SELECT, np.where(u < HELD_OUT, GATE, TRAIN)).astype(np.int8)


# ---------------------------------------------------------------- the check-down baseline

def _checkdown(args):
    board, half, stack, ranges_list = args
    from parciver import turn_values as tv
    from parciver.gpu_search import Hands

    hands = Hands(list(int(c) for c in board))
    return [tv.checkdown_values(tv.Situation(tuple(int(c) for c in board), int(half), int(stack), 1, 100,
                                             r.astype(np.float64)), hands) for r in ranges_list]


def cmd_baseline(a) -> int:
    import concurrent.futures as cf

    d = load(Path(a.data))
    rows = np.flatnonzero(split(d["board"]) != TRAIN)
    by_j: dict = {}
    for r in rows:
        by_j.setdefault(int(d["index"][r]), []).append(int(r))
    jobs = [(d["board"][rs[0]], d["half"][rs[0]], d["stack"][rs[0]], [d["ranges"][r] for r in rs]) for rs in by_j.values()]
    out = np.zeros((len(rows), 2, d["values"].shape[-1]), dtype=np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers) as pool:
        for n, (rs, vals) in enumerate(zip(by_j.values(), pool.map(_checkdown, jobs, chunksize=16))):
            for r, v in zip(rs, vals):
                out[pos[r]] = v
            if (n + 1) % 2000 == 0:
                print(f"  {n + 1}/{len(jobs)} boards  {time.time() - t0:.0f}s", flush=True)
    # Rows are identified by (index, source), stable as shards are added; `inputs` refuses a row generated again.
    tmp = Path(a.data) / "checkdown-val.tmp.npz"
    np.savez(tmp, index=d["index"][rows], source=d["source"][rows], values=out, inputs=row_digests(d, rows))
    persist(tmp, Path(a.data) / "checkdown-val.npz")
    print(f"saved the check-down values of {len(rows)} held-out samples ({len(jobs)} situations)")
    return 0


# ---------------------------------------------------------------- training

class Batches:
    """Samples on the GPU in the solver's 1,128-holding form; batches in the
    net's 1,326 form, optionally suit-relabelled (a uniform one of the 24)."""

    def __init__(self, d: dict, rows: np.ndarray, device):
        import torch

        from parciver import value_net as vn

        self.torch, self.vn, self.dev = torch, vn, device
        self.ranges = torch.as_tensor(self.prepared_ranges(d, rows), device=device)
        self.values = torch.as_tensor(d["values"][rows], device=device)
        self.board = torch.as_tensor(d["board"][rows], device=device)
        self.scal = torch.as_tensor(vn.scalars(d["half"][rows], d["stack"][rows]), dtype=torch.float32, device=device)
        self.pairs = torch.as_tensor(vn.PAIRS, device=device)
        self.n = len(rows)

    @staticmethod
    def prepared_ranges(d: dict, rows: np.ndarray) -> np.ndarray:
        r16 = np.empty((len(rows),) + d["ranges"].shape[1:], np.float16)
        for s in range(0, len(rows), CHUNK):  # float64 a chunk at a time: each row is normalized on its own
            r = d["ranges"][rows[s:s + CHUNK]].astype(np.float64)
            r /= r.sum(-1, keepdims=True)
            # float16 times RANGE_SCALE: entries <= 1 stay in range and the small ones
            # keep float16's full precision instead of its subnormal steps of 2^-24.
            r *= RANGE_SCALE  # in place (exact)
            r16[s:s + CHUNK] = r.astype(np.float16)
        return r16

    def take(self, perm, s: int, batch: int):
        """The epoch's batch at position s of its permutation, as `get` takes it."""
        return perm[s:s + batch]

    def stored(self, idx):
        """The rows' stored ranges and values."""
        return self.ranges[idx], self.values[idx]

    def get(self, idx, augment: bool):
        torch, vn = self.torch, self.vn
        board = self.board[idx]
        live = ~(self.pairs[None, :, :, None] == board[:, None, None, :]).any(-1).any(-1)
        ranges, values = self.stored(idx)
        r = vn.expand(ranges.float() * (1 / RANGE_SCALE), live)
        v = vn.expand(values, live)
        onehot = torch.zeros(len(idx), 52, device=self.dev)
        onehot.scatter_(1, board, 1.0)
        if augment:
            k = torch.randint(0, 24, (len(idx),), device=self.dev)
            r, v, onehot, live = vn.permute(k, r, v, onehot, live)
        return r, v, onehot, self.scal[idx], live


class Pools(Batches):
    """`Batches` for training rows that do not fit on the GPU at once (D2b,
    records/d2b/PREREG.md, "GPU pools"). Boards and scalars stay on the device
    for every row. Ranges and values do not: an epoch's permutation is cut on
    batch boundaries into `pools` pools, and each is read in row order, uploaded
    over the last, and found through `where` when its batches come. The rows,
    their order in each batch and the random draws are those of one
    full-permutation run; only where the rows wait differs. The last partial
    batch drops, as in `train`."""

    COPY = 65536  # rows uploaded at a time

    def __init__(self, d: dict, rows: np.ndarray, device, pools: int, batch: int):
        import torch

        from parciver import value_net as vn

        self.torch, self.vn, self.dev = torch, vn, device
        self.host_ranges = self.prepared_ranges(d, rows)  # once: a mapped column (d3_load.Col) is read a chunk at a time
        self.host_values, self.rows = d["values"], rows
        self.board = torch.as_tensor(d["board"][rows], device=device)
        self.scal = torch.as_tensor(vn.scalars(d["half"][rows], d["stack"][rows]), dtype=torch.float32, device=device)
        self.pairs = torch.as_tensor(vn.PAIRS, device=device)
        self.n = len(rows)
        full = self.n // batch
        cuts = [k * full // pools * batch for k in range(pools + 1)]
        self.pool_end = {lo: hi for lo, hi in zip(cuts, cuts[1:]) if hi > lo}
        size = max(hi - lo for lo, hi in self.pool_end.items())
        self.ranges = torch.empty((size,) + self.host_ranges.shape[1:], dtype=torch.float16, device=device)
        self.values = torch.empty((size,) + self.host_values.shape[1:], dtype=torch.float32, device=device)
        self.where = torch.zeros(self.n, dtype=torch.long, device=device)
        self.upload_seconds = 0.0

    def upload(self, ids) -> None:
        """Rows `ids` (positions in `rows`) onto the device, over the last pool."""
        torch = self.torch
        t0 = time.time()
        ordered = ids.sort().values
        self.where[ordered] = torch.arange(len(ordered), device=self.dev)
        host = ordered.cpu().numpy()
        for s in range(0, len(host), self.COPY):
            at = host[s:s + self.COPY]
            self.ranges[s:s + len(at)].copy_(torch.from_numpy(self.host_ranges[at]))
            self.values[s:s + len(at)].copy_(torch.from_numpy(self.host_values[self.rows[at]]))
        self.upload_seconds += time.time() - t0

    def take(self, perm, s: int, batch: int):
        if s in self.pool_end:
            self.upload(perm[s:self.pool_end[s]])
        return perm[s:s + batch]

    def stored(self, idx):
        at = self.where[idx]
        return self.ranges[at], self.values[at]


def _loss(pred, true):
    """Huber loss over the live holdings. Both tensors are 0 at the dead ones and
    every sample has 2 x 1,128 live entries, so the mean over all 1,326 columns
    rescaled is exactly the mean over the live ones (and needs no mask)."""
    import torch.nn.functional as F

    from parciver import value_net as vn

    return F.huber_loss(pred, true) * (vn.HOLDINGS / vn.LIVE)


def _loss_rw(pred, true, ranges):
    """D3-SWEEP's range-weighted Huber: per player-sample, sum_i r_i * huber(pred_i - true_i) over the holdings (r is 0 at the
    dead ones, and the net's value there is 0), averaged over the batch and both players. Each r sums to 1, so it is on the
    plain loss's scale (a weighted mean of the same elementwise Huber)."""
    import torch.nn.functional as F

    return (F.huber_loss(pred, true, reduction="none") * ranges).sum(-1).mean()


def training_loss(kind: str, pred, true, ranges):
    """What training descends: the plain Huber (N1's), or `rw` = 0.5 plain + 0.5 range-weighted. Selection is always the plain one."""
    if kind == "huber":
        return _loss(pred, true)
    return 0.5 * _loss(pred, true) + 0.5 * _loss_rw(pred, true, ranges)


def evaluate(net, b: Batches, batch: int = 4096, amp: bool = False) -> dict:
    """Huber loss and range-weighted MAE (% of the pot) over every sample, no
    augmentation. `amp`: bfloat16 matmuls, as in training; off (the default,
    so epoch selection and G2 score the net the search runs), float32, as
    `value_net.predict` runs the net in the search."""
    import torch

    from parciver import value_net as vn

    net.eval()
    loss, mae = 0.0, []  # loss: a device scalar, read once at the end
    with torch.no_grad(), torch.autocast(torch.device(b.dev).type, dtype=torch.bfloat16, enabled=amp):
        for s in range(0, b.n, batch):
            idx = torch.arange(s, min(s + batch, b.n), device=b.dev)
            r, v, onehot, scal, live = b.get(idx, False)
            pred = net(r, onehot, scal, live)
            loss = loss + _loss(pred.float(), v) * len(idx)
            mae.append(vn.range_mae(pred.float(), v, r))
    net.train()
    mae = torch.cat(mae)
    return dict(loss=float(loss) / b.n, mae=100 * float(mae.mean()), mae_per_sample=mae)


def leafnet_tag(canonical: bool, wd: float) -> str:
    return f"leafnet-c{int(canonical)}-wd{wd:g}"


def optimizer(net, lr: float, wd: float):
    """Adam as D2 (wd 0: the same code path), else AdamW with decoupled decay
    on the Linear weights only (biases and PReLU slopes: weight_decay 0)."""
    import torch

    if wd == 0:
        return torch.optim.Adam(net.parameters(), lr=lr)
    decay = [m.weight for m in net.modules() if isinstance(m, torch.nn.Linear)]
    ids = {id(p) for p in decay}
    rest = [p for p in net.parameters() if id(p) not in ids]
    return torch.optim.AdamW([dict(params=decay, weight_decay=wd), dict(params=rest, weight_decay=0.0)], lr=lr)


def train(d: dict, part: np.ndarray, width: int, depth: int, fraction: float, epochs: int, batch: int, lr: float,
          seed: int, tag: str, log=print, device="cuda", reuse: bool = False, held: dict | None = None,
          canonical: bool = False, wd: float = 0.0, pools: int = 1, commit: str | None = None,
          dropout: float = 0.0, ema: float = 0.0, loss: str = "huber", stop_after: int | None = None) -> dict:
    """One net. `commit`: the code it runs on, if read before a long job began. `reuse`: a run finished under these settings, on these samples
    (their counts and `fingerprint`), is returned from its record.json instead
    of trained again; otherwise the old record goes first, so a stop before the
    first epoch ends never leaves it beside a new net. `held`: a dict
    that keeps the last samples put on the device, so consecutive nets on the
    same rows (a sweep's) share them. `canonical`, `wd`: LEAF-NET's frame and
    weight decay (records/leaf-net/PREREG.md). `pools` > 1: the training rows
    wait on the host (`Pools`; the batches are the same). D3-SWEEP (records/d3-sweep/PREREG.md; each off at its default, and then
    the path is the old one to the bit): `dropout` p after each hidden PReLU, in training only (`value_net`); `ema` d > 0: the net
    that is evaluated and saved is the exponential average e <- d e + (1 - d) w of the weights, taken after every optimiser step
    from the initial weights (the live weights keep training); `loss` "rw": train on 0.5 plain + 0.5 range-weighted Huber
    (`train_loss` in the history is then that mix; `val_loss`, the selection, is always the plain Huber).
    `stop_after` k: the run ends after epoch k, on the schedule of all `epochs` (D3-SWEEP's `check`: the first epochs of a
    full run, to the bit; a record cut so is never reused)."""
    import torch

    from parciver import value_net as vn

    if not (0.0 <= dropout < 1.0 and 0.0 <= ema < 1.0 and loss in LOSSES):
        raise ValueError(f"dropout {dropout} and ema {ema} are in [0, 1), loss one of {LOSSES}: got {loss!r}")
    if stop_after is not None and not 1 <= stop_after <= epochs:
        raise ValueError(f"stop_after {stop_after} is not in 1..{epochs}")
    torch.manual_seed(seed)
    dev = torch.device(device)
    tr_rows = np.flatnonzero(part == TRAIN)  # fractions are nested: prefixes of one permutation
    tr_rows = np.sort(np.random.default_rng(seed).permutation(tr_rows)[: int(round(fraction * len(tr_rows)))])
    va_rows = np.flatnonzero(part == SELECT)
    out = NETS / tag
    cfg = dict(tag=tag, width=width, depth=depth, fraction=fraction, train_samples=len(tr_rows),
               val_samples=len(va_rows), data=fingerprint(d), epochs=epochs, batch=batch, lr=lr, seed=seed,
               canonical=bool(canonical), wd=float(wd), pools=int(pools), dropout=float(dropout), ema=float(ema), loss=str(loss))
    if reuse and (out / "record.json").exists() and (out / "net.pt").exists():
        rec = json.loads((out / "record.json").read_text())
        if all(rec.get(k, CFG_DEFAULTS.get(k)) == v for k, v in cfg.items()) and len(rec.get("history", [])) == epochs:
            log(f"  [{tag}] already trained under these settings ({len(tr_rows)} samples): reusing its record")
            return rec
    (out / "record.json").unlink(missing_ok=True)
    key = (cfg["data"], fraction, seed, pools, batch, str(dev))
    if held is not None and held.get("key") == key:
        tr, va = held["tr"], held["va"]
    else:
        if held is not None:
            held.clear()  # before the new samples go up: never two training sets on the device
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        tr, va = Batches(d, tr_rows, dev) if pools == 1 else Pools(d, tr_rows, dev, pools, batch), Batches(d, va_rows, dev)
        if held is not None:
            held.update(key=key, tr=tr, va=va)
    uploaded = getattr(tr, "upload_seconds", 0.0)  # `held` keeps the count of the nets before this one
    net = vn.ValueNet(width, depth, canonical).to(dev)
    scal = tr.scal
    net.scal_mean.copy_(scal.mean(0))
    net.scal_std.copy_(scal.std(0))
    net.p = float(dropout)  # a plain attribute: not in the state dict
    shadow = None  # the EMA net (a copy, buffers included); with ema 0 nothing exists and nothing runs
    if ema > 0:
        shadow = copy.deepcopy(net)
        shadow.p = 0.0
        shadow.requires_grad_(False)
        live_w, shadow_w = list(net.parameters()), list(shadow.parameters())
    scored = net if shadow is None else shadow  # what is evaluated and saved
    opt = optimizer(net, lr, wd)
    env = dict(torch=torch.__version__, commit=commit or _commit())
    steps = epochs * max(tr.n // batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.02)
    out.mkdir(parents=True, exist_ok=True)
    best, hist, t0, step = None, [], time.time(), 0
    for ep in range(epochs):
        perm = torch.randperm(tr.n, device=dev)
        run = torch.zeros((), device=dev)
        for s in range(0, tr.n - batch + 1, batch):
            r, v, onehot, sc, live = tr.get(tr.take(perm, s, batch), True)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                pred = net(r, onehot, sc, live)
            lv = training_loss(loss, pred.float(), v, r)
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()
            sched.step()
            if shadow is not None:
                with torch.no_grad():
                    torch._foreach_mul_(shadow_w, ema)
                    torch._foreach_add_(shadow_w, live_w, alpha=1 - ema)
            step += 1
            run += lv.detach()
        e = evaluate(scored, va)
        hist.append(dict(epoch=ep + 1, train_loss=float(run) / max(tr.n // batch, 1), val_loss=e["loss"], val_mae=e["mae"],
                         seconds=round(time.time() - t0, 1)))
        log(f"  [{tag}] epoch {ep + 1}/{epochs}  train {hist[-1]['train_loss']:.5f}  val {e['loss']:.5f}  "
            f"val MAE {e['mae']:.3f}% pot  {time.time() - t0:.0f}s")
        if not (math.isfinite(hist[-1]["train_loss"]) and math.isfinite(e["loss"])):
            raise FloatingPointError(f"[{tag}] non-finite loss at epoch {ep + 1}: stopped (a NaN net never recovers, "
                                     f"and no later epoch could be chosen over it)")
        if best is None or e["loss"] < best["val_loss"]:
            best = dict(hist[-1])
            vn.save(scored, out / "net.pt.tmp", meta=dict(width=width, depth=depth, fraction=fraction, epoch=ep + 1,
                                                           canonical=bool(canonical), wd=float(wd), dropout=float(dropout),
                                                           ema=float(ema), loss=str(loss)))
            persist(out / "net.pt.tmp", out / "net.pt")
        rec = dict(cfg, best=best, history=hist, **env, **({"upload_seconds": tr.upload_seconds - uploaded} if pools > 1 else {}))
        _write(out / "record.json", json.dumps(rec, indent=1))  # every epoch: a stopped run shows how far it got
        if stop_after is not None and ep + 1 >= stop_after:
            break
    return rec


def _commit() -> str:
    import subprocess

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
    except OSError:
        return "unknown"
    return head + ("+dirty" if dirty else "")


def _data(a):
    d = load(Path(a.data), a.max_shards)
    return d, split(d["board"])


def cmd_train(a) -> int:
    d, part = _data(a)
    train(d, part, a.width, a.depth, a.fraction, a.epochs, a.batch, a.lr, a.seed, a.tag or f"w{a.width}-d{a.depth}")
    return 0


def cmd_sweep(a) -> int:
    d, part = _data(a)
    recs, held = [], {}
    for w in a.widths:
        for dp in a.depths:
            recs.append(train(d, part, w, dp, 1.0, a.epochs, a.batch, a.lr, a.seed, f"sweep-w{w}-d{dp}",
                              reuse=not a.fresh, held=held))
    recs.sort(key=lambda r: r["best"]["val_loss"])
    for r in recs:
        print(f"  width {r['width']:>5d} depth {r['depth']}: val loss {r['best']['val_loss']:.5f}  "
              f"MAE {r['best']['val_mae']:.3f}% pot  (epoch {r['best']['epoch']})")
    keys = ("tag", "width", "depth", "train_samples", "val_samples", "data", "epochs", "best")
    _write(NETS / "sweep.json", json.dumps([{k: r[k] for k in keys} for r in recs], indent=1))
    print(f"chosen: width {recs[0]['width']} depth {recs[0]['depth']}")
    return 0


def _chosen(d: dict, part: np.ndarray) -> dict:
    """The sweep's best record (lowest selection loss). Refused when the sweep
    saw other data than `d`'s (a preview on fewer shards, regenerated shards)
    or did not record which: its choice and its losses are not comparable with
    a net trained now."""
    c = json.loads((NETS / "sweep.json").read_text())[0]
    have = dict(train_samples=int((part == TRAIN).sum()), val_samples=int((part == SELECT).sum()), data=fingerprint(d))
    stale = {k: (c.get(k), v) for k, v in have.items() if c.get(k) != v}
    if stale:
        raise SystemExit(f"sweep.json was made on other data ({stale}: the sweep's, this data's): run `sweep` again")
    return c


def cmd_final(a) -> int:
    d, part = _data(a)
    c = _chosen(d, part)
    rec = train(d, part, c["width"], c["depth"], 1.0, a.epochs, a.batch, a.lr, a.seed,
                f"final-w{c['width']}-d{c['depth']}-e{a.epochs}")
    cands = [{k: r[k] for k in ("tag", "width", "depth", "best")} for r in (c, rec)]
    pick = min(cands, key=lambda r: r["best"]["val_loss"])
    _write(NETS / "final.json", json.dumps(dict(tag=pick["tag"], candidates=cands), indent=1))
    for r in cands:
        print(f"  {r['tag']}: selection loss {r['best']['val_loss']:.5f}  MAE {r['best']['val_mae']:.3f}% pot")
    print(f"final net: {pick['tag']}")
    return 0


def cmd_curve(a) -> int:
    d, part = _data(a)
    w, dp = a.width or _chosen(d, part)["width"], a.depth or _chosen(d, part)["depth"]
    recs = [train(d, part, w, dp, f, a.epochs, a.batch, a.lr, a.seed, f"curve-{int(100 * f)}", reuse=not a.fresh)
            for f in (0.25, 0.5, 1.0)]
    for r in recs:
        print(f"  {int(100 * r['fraction']):>3d}% ({r['train_samples']} samples): val MAE {r['best']['val_mae']:.3f}% pot")
    _write(NETS / "curve.json", json.dumps([{k: r[k] for k in ("tag", "fraction", "train_samples", "best")}
                                            for r in recs], indent=1))
    return 0


def cmd_leafnet(a) -> int:
    """The grid, D2's recipe first: it must reproduce N0's record (best epoch
    and selection loss, the full float) or nothing else is trained. Finished
    nets are reused (a stopped run resumes at the next net)."""
    d, part = _data(a)
    if fingerprint(d) != LEAFNET_DATA:
        print(f"the data is {fingerprint(d)}, not D2's {LEAFNET_DATA}: refused (records/leaf-net/PREREG.md)")
        return 1
    s, held, recs = LEAFNET_SETTINGS, {}, []
    n0 = json.loads((NETS / LEAFNET_N0 / "record.json").read_text())["best"]
    for canonical in (False, True):
        for wd in LEAFNET_WD:
            rec = train(d, part, s["width"], s["depth"], 1.0, s["epochs"], s["batch"], s["lr"], s["seed"],
                        leafnet_tag(canonical, wd), reuse=True, held=held, canonical=canonical, wd=wd)
            recs.append(rec)
            if not canonical and wd == 0:
                repro = rec["best"]["epoch"] == n0["epoch"] and rec["best"]["val_loss"] == n0["val_loss"]
                print(f"  D2's recipe: best epoch {rec['best']['epoch']}, selection loss {rec['best']['val_loss']!r}; "
                      f"N0 {n0['epoch']}, {n0['val_loss']!r}: {'REPRODUCED' if repro else 'NOT REPRODUCED'}")
                if not repro:
                    _write(NETS / "leafnet.json", json.dumps(dict(reproduced=False, d2_recipe=rec["best"], n0=n0), indent=1))
                    print("the grid is not used until the cause is found and an amendment records it (PREREG)")
                    return 1
    pick = min(recs, key=lambda r: r["best"]["val_loss"])
    rows = [dict(tag=r["tag"], canonical=r["canonical"], wd=r["wd"], best=r["best"]) for r in recs]
    for r in sorted(rows, key=lambda r: r["best"]["val_loss"]):
        print(f"  {r['tag']:<22s} selection loss {r['best']['val_loss']:.6f}  MAE {r['best']['val_mae']:.3f}% pot  "
              f"(epoch {r['best']['epoch']})")
    stop = pick["tag"] == leafnet_tag(False, 0.0)
    _write(NETS / "leafnet.json", json.dumps(dict(reproduced=True, pick=pick["tag"], stop=stop, nets=rows), indent=1))
    print(f"pick P: {pick['tag']}" + ("  -> LEAF-NET stops: nothing beats N0's recipe" if stop else ""))
    return 0


def hb_members(board: np.ndarray) -> list[list[int]]:
    """Each row's turn-board stabilizer H_b ([M, 4] boards): the SUIT_PERMS rows
    mapping the board onto itself, the identity first."""
    from parciver import value_net as vn

    out = []
    for b in np.asarray(board, dtype=np.int64):
        s = set(b.tolist())
        out.append([k for k in range(len(vn.SUIT_PERMS)) if set(vn.CARD_PERM[k, b].tolist()) == s])
    return out


def inverse_perm(k: int) -> int:
    """The SUIT_PERMS row undoing row k."""
    from parciver import value_net as vn

    p = vn.SUIT_PERMS
    return int(np.flatnonzero((p[:, p[k]] == np.arange(4)).all(1))[0])


def hb_average(net, b: Batches, hb: list[list[int]], batch: int = 4096) -> np.ndarray:
    """[M, 2, 1326] float64: each sample's values averaged over its H_b,
    mean over h of h^-1.N(h.x), float32 as the search runs the net. Every term
    is relabelled back to the sample's own holdings."""
    import torch

    from parciver import value_net as vn

    out = np.zeros((b.n, 2, vn.HOLDINGS))
    count = np.array([len(h) for h in hb])
    net.eval()
    with torch.no_grad():
        for m in range(int(count.max())):
            rows = np.flatnonzero(count > m)
            for s in range(0, len(rows), batch):
                sel = rows[s:s + batch]
                idx = torch.as_tensor(sel, device=b.dev)
                r, _, onehot, scal, live = b.get(idx, False)
                kk = torch.as_tensor([hb[i][m] for i in sel], device=b.dev)
                back = torch.as_tensor([inverse_perm(hb[i][m]) for i in sel], device=b.dev)
                r2, _, oh2, live2 = vn.permute(kk, r, r, onehot, live)
                v = net(r2, oh2, scal, live2).float()
                _, v, _, _ = vn.permute(back, r2, v, oh2, live2)
                out[sel] += v.double().cpu().numpy()
    return out / count[:, None, None]


def weighted_mae(pred: np.ndarray, b: Batches, batch: int = 4096) -> np.ndarray:
    """[M] % of the pot: `range_mae` of [M, 2, 1326] values against the
    samples' targets, with the ranges `evaluate` weights by (the net's input
    ranges), so an average and a bare forward are scored alike."""
    import torch

    from parciver import value_net as vn

    out = []
    for s in range(0, b.n, batch):
        idx = torch.arange(s, min(s + batch, b.n), device=b.dev)
        r, v, _, _, _ = b.get(idx, False)
        p = torch.as_tensor(pred[s:s + batch], device=b.dev)
        out.append(vn.range_mae(p, v.double(), r.double()).cpu().numpy())
    return 100 * np.concatenate(out)


def board_bootstrap(diff: np.ndarray, cluster: np.ndarray, resamples: int = LEAFNET_RESAMPLES, seed: int = 0) -> dict:
    """Mean of `diff` and its 95% percentile bootstrap CI, resampling clusters
    (canonical boards) with their samples (numpy default_rng(seed))."""
    ids, inv = np.unique(cluster, return_inverse=True)
    tot = np.bincount(inv, weights=diff, minlength=len(ids))
    cnt = np.bincount(inv, minlength=len(ids)).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, len(ids), len(ids))
        means[i] = tot[pick].sum() / cnt[pick].sum()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(diff.mean()), ci=[float(lo), float(hi)], boards=int(len(ids)), samples=int(len(diff)))


def symmetry_defect(net, b: Batches, seed: int = 0, batch: int = 4096) -> np.ndarray:
    """[M] % of the pot: per sample, one random non-identity relabelling s
    (default_rng(seed)), the range-weighted |N(s.x) - s.N(x)|."""
    import torch

    from parciver import value_net as vn

    ks = np.random.default_rng(seed).integers(1, len(vn.SUIT_PERMS), b.n)
    out = []
    net.eval()
    with torch.no_grad():
        for s in range(0, b.n, batch):
            idx = torch.arange(s, min(s + batch, b.n), device=b.dev)
            r, _, onehot, scal, live = b.get(idx, False)
            k = torch.as_tensor(ks[s:s + batch], device=b.dev)
            v = net(r, onehot, scal, live).float()
            r2, sv, oh2, live2 = vn.permute(k, r, v, onehot, live)
            out.append(vn.range_mae(net(r2, oh2, scal, live2).float(), sv, r2).double().cpu().numpy())
    return 100 * np.concatenate(out)


def gate_a_result(mae: dict, bare: dict, defect: dict, cluster: np.ndarray, trivial: np.ndarray, source: np.ndarray,
                  bar: float, cand: str = "P") -> dict:
    """Gate A from per-sample arrays (% of the pot; keys `cand` and N0): proceed
    iff the H_b-averaged D = cand - N0 has its board-clustered CI's upper end < 0
    and the candidate's bare MAE is under the G2 bar."""
    def summary(x):
        return dict(board_bootstrap(x[cand] - x["N0"], cluster), **{cand: float(x[cand].mean())}, N0=float(x["N0"].mean()),
                    relative=float(x[cand].mean() / x["N0"].mean() - 1),
                    **{name: {k: float(x[k][source == s].mean()) for k in x} for s, name in ((0, "real"), (1, "random"))})

    def spread(v, m):
        return dict(mean=float(v[m].mean()), max=float(v[m].max())) if m.any() else dict(mean=None, max=None)

    averaged, plain = summary(mae), summary(bare)
    return dict(samples=len(cluster), trivial_hb=int(trivial.sum()), averaged=averaged, bare=plain, g2_bar=bar,
                proceed=bool(averaged["ci"][1] < 0 and plain[cand] < bar),
                symmetry_defect={k: dict(trivial=spread(v, trivial), other=spread(v, ~trivial)) for k, v in defect.items()})


def cmd_gate_a(a) -> int:
    import torch

    from parciver import value_net as vn

    lf = json.loads((NETS / "leafnet.json").read_text())
    if not lf.get("reproduced") or lf.get("stop"):
        print(f"leafnet.json: no pick to gate (reproduced {lf.get('reproduced')}, stop {lf.get('stop')})")
        return 1
    d, part = _data(a)
    if fingerprint(d) != LEAFNET_DATA:
        print(f"the data is {fingerprint(d)}, not D2's {LEAFNET_DATA}: refused")
        return 1
    rows = np.flatnonzero(part == GATE)
    b = Batches(d, rows, torch.device("cuda"))
    hb = hb_members(d["board"][rows])
    trivial = np.array([len(h) == 1 for h in hb])
    nets = {"P": NETS / lf["pick"] / "net.pt", "N0": NETS / LEAFNET_N0 / "net.pt"}
    sha = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in nets.items()}
    mae, bare, defect = {}, {}, {}
    for k, path in nets.items():
        net, _ = vn.load(path)
        mae[k] = weighted_mae(hb_average(net, b, hb), b)
        bare[k] = evaluate(net, b)["mae_per_sample"].double().cpu().numpy() * 100
        defect[k] = symmetry_defect(net, b)
        del net
        torch.cuda.empty_cache()
    g2 = json.loads((NETS / LEAFNET_N0 / "g2.json").read_text())
    res = gate_a_result(mae, bare, defect, canonical_codes(d["board"][rows]), trivial, d["source"][rows],
                        0.5 * g2["baseline_mae"])
    res = dict(pick=lf["pick"], sha256=sha, data=LEAFNET_DATA, **res)
    passed, bar = res["proceed"], res["g2_bar"]
    out = ROOT / "records" / "leaf-net" / "gate-a.json"
    _write(out, json.dumps(res, indent=1))
    av = res["averaged"]
    print(f"Gate A ({lf['pick']} vs N0, H_b-averaged, {len(rows)} gate samples): MAE {av['P']:.3f} vs {av['N0']:.3f}% pot, "
          f"D {av['mean']:+.4f} [{av['ci'][0]:+.4f}, {av['ci'][1]:+.4f}] ({100 * av['relative']:+.1f}%)")
    print(f"  bare: {res['bare']['P']:.3f} vs {res['bare']['N0']:.3f}, D {res['bare']['mean']:+.4f} "
          f"[{res['bare']['ci'][0]:+.4f}, {res['bare']['ci'][1]:+.4f}]; G2 bar {bar:.3f}")
    print(f"  {'PROCEED' if passed else 'STOP: N0 stays'}  ({out})")
    return 0 if passed else 1


# ---------------------------------------------------------------- D2b (records/d2b/PREREG.md)

def d2b_check_files(data: Path, manifest: Path, want: str = D2B_MANIFEST) -> None:
    """Refuse unless `manifest` is the registered one, its files are exactly the
    folder's finished shards and situations.npz, byte for byte, and no patch
    would change them as they load (D1b has none)."""
    text = manifest.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(text).hexdigest() != want:
        raise SystemExit(f"{manifest} is not the registered manifest ({want})")
    if list((data / "resolve-q3").glob("patch*")):
        raise SystemExit(f"{data}: holds a patch; the registration has none for it")
    pinned = dict(reversed(line.split("  ", 1)) for line in text.decode().splitlines())
    have = {p.name for p in data.glob("shard-*.npz") if not p.name.endswith(".tmp.npz")} | {"situations.npz"}
    if set(pinned) != have:
        raise SystemExit(f"{data}: the files differ from the manifest's: {sorted(set(pinned) ^ have)[:5]}")
    for name, sha in pinned.items():
        h = hashlib.sha256()
        with open(data / name, "rb") as f:
            while block := f.read(1 << 24):
                h.update(block)
        if h.hexdigest() != sha:
            raise SystemExit(f"{data / name}: sha256 {h.hexdigest()}, pinned {sha}")


def d2b_views(d: dict) -> dict:
    """The union and each part as views of its rows (no copies)."""
    n1 = int((d["dataset"] == 0).sum())
    if not (d["dataset"][:n1] == 0).all():
        raise ValueError("D1's rows do not come first")
    return dict(union=d, d1={k: v[:n1] for k, v in d.items()}, d1b={k: v[n1:] for k, v in d.items()})


def d2b_data(a) -> tuple[dict, dict]:
    """The views of D1 + D1b, the pinned files and D1's fingerprint checked."""
    runs = Path(a.data).parent
    d2b_check_files(runs / D2B_DIRS[1], D2B_RECORDS / "d1b-manifest.sha256")
    views = d2b_views(load_many([runs / x for x in D2B_DIRS]))
    if fingerprint(views["d1"]) != LEAFNET_DATA:
        raise SystemExit(f"D1's fingerprint is {fingerprint(views['d1'])}, registered {LEAFNET_DATA}")
    return views, {k: split(v["board"]) for k, v in views.items()}


def d2b_refuse_other_data(views: dict) -> str:
    """Training and the gates run only on the union pinned by the amendment."""
    got = fingerprint(views["union"])
    if D2B_UNION is None or got != D2B_UNION:
        raise SystemExit(f"the union's fingerprint is {got}, pinned {D2B_UNION}: `d2b data`, then an amendment first")
    return got


def d2b_data_record(views: dict, parts: dict) -> dict:
    """Fingerprints and counts (rows and canonical boards by part); the rows
    over 10% exploitability, kept as accepted, counted."""
    out = {}
    for name, v in views.items():
        codes = canonical_codes(v["board"])
        out[name] = dict(fingerprint=fingerprint(v), rows_over_10pct=int((v["expl"] > 10).sum()),
                         **{p: dict(rows=int((parts[name] == s).sum()), boards=len(np.unique(codes[parts[name] == s])))
                            for p, s in (("train", TRAIN), ("select", SELECT), ("gate", GATE))})
    return out


def cmd_d2b_data(a) -> int:
    views, parts = d2b_data(a)
    res = d2b_data_record(views, parts)
    _write(D2B_RECORDS / "data.json", json.dumps(res, indent=1))
    for name, r in res.items():
        print(f"{name}: fingerprint {r['fingerprint']}; train / select / gate rows {r['train']['rows']} / "
              f"{r['select']['rows']} / {r['gate']['rows']}; boards {r['train']['boards']} / {r['select']['boards']} / "
              f"{r['gate']['boards']}; over 10% exploitability {r['rows_over_10pct']}")
    return 0


def d2b_train_arm(views: dict, parts: dict, arm: dict, held: dict, gpu=lambda: None, seed: int | None = None, log=print,
                  device="cuda", commit: str | None = None, stop_after: int | None = None) -> dict:
    """One arm at D2's settings; `gpu` looks at the GPU's clients before it, every
    D2B_GPU_EVERY epochs and after its last (a finished net is reused, and looked at only once)."""
    s = LEAFNET_SETTINGS
    epochs = []

    def watch(line):
        log(line)
        if " epoch " in line:
            if len(epochs) % D2B_GPU_EVERY == 0:
                gpu()
            epochs.append(line)

    gpu()
    rec = train(views[arm["data"]], parts[arm["data"]], arm["width"], s["depth"], 1.0, s["epochs"], s["batch"], s["lr"],
                s["seed"] if seed is None else seed, arm["tag"], log=watch, device=device, reuse=True, held=held,
                wd=arm["wd"], pools=D2B_POOLS if arm["data"] == "union" else 1, commit=commit,
                dropout=arm.get("dropout", 0.0), ema=arm.get("ema", 0.0), loss=arm.get("loss", "huber"),
                stop_after=stop_after)  # D3-SWEEP options: an arm without them is the old recipe
    if epochs:
        gpu()
    return rec


def d2b_reproduced(rec: dict, n0: dict) -> bool:
    """d2b-check against N0's record: the best epoch and the full float."""
    return rec["best"]["epoch"] == n0["epoch"] and rec["best"]["val_loss"] == n0["val_loss"]


def d2b_pick(recs: list[dict]) -> dict:
    """The candidate with the lower selection loss (a tie: the first, wd 0)."""
    return min(recs, key=lambda r: r["best"]["val_loss"])


def d2b_blind(g: dict) -> bool:
    """A GPU look that could not say what else ran. `util` None: nvidia-smi ran and failed, and then lists no processes
    (livesearch.busy). No apps: that query failed alone, since Windows lists every desktop app as a client."""
    return g["peers"] is None or g["util"] is None or not g["apps"]


def d2b_nets_record(check: dict, n0: dict, recs: list[dict], gpu_looks: dict) -> dict:
    keys = ("tag", "width", "depth", "wd", "seed", "data", "train_samples", "val_samples", "pools", "commit", "best")
    rows = [dict({k: r.get(k) for k in keys}, **{k: r.get(k, CFG_DEFAULTS[k]) for k in OPTION_KEYS}, seconds=r["history"][-1]["seconds"], upload_seconds=r.get("upload_seconds"),
                 train_loss_at_best=r["history"][r["best"]["epoch"] - 1]["train_loss"],
                 gpu_peers=sorted({p for g in gpu_looks.get(r["tag"], []) for p in map(str, g["peers"] or [])}),
                 gpu_looks=len(gpu_looks.get(r["tag"], [])),
                 gpu_unknown=sum(d2b_blind(g) for g in gpu_looks.get(r["tag"], [])),
                 gpu_apps=sorted({x for g in gpu_looks.get(r["tag"], []) for x in g["apps"]})) for r in recs]
    return dict(reproduced=d2b_reproduced(check, n0), n0=n0, nets=rows, voided=d2b_voided())


def d2b_committed(commit: str) -> bool:
    """A full commit hash alone: no `+dirty`, and no "unknown" or "" from a git that could not say."""
    return len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)


def d2b_nets_refused(rows: list[dict]) -> str | None:
    """Why these nets may not be selected from: one trained on uncommitted code, beside another process of this Python
    on the GPU (VOID), or with no look that could say. Another program's compute job shows only in `gpu_apps`, among
    the desktop apps Windows lists: read there."""
    for r in rows:
        if not d2b_committed(str(r["commit"])):
            return f"{r['tag']}: trained on uncommitted code ({r['commit']})"
        if r["gpu_peers"]:
            return f"{r['tag']}: another process of this Python on the GPU ({r['gpu_peers']}): VOID (PREREG, VOID and bugs)"
        if r["gpu_unknown"] == r["gpu_looks"]:
            return f"{r['tag']}: none of its {r['gpu_looks']} GPU looks could say what else was running"
    return None


def d2b_records_differ(rows: list[dict]) -> str | None:
    """Why nets.json no longer speaks for the nets on disk: a record cut short or trained again since it was written."""
    for r in rows:
        path = NETS / r["tag"] / "record.json"
        if not path.exists():
            return f"{r['tag']}: nets.json holds it, and it has no record.json"
        rec = json.loads(path.read_text())
        if len(rec["history"]) != LEAFNET_SETTINGS["epochs"] or rec["best"] != r["best"]:
            return f"{r['tag']}: its record.json is not what nets.json holds ({len(rec['history'])} epochs, best {rec['best']})"
    return None


def d2b_net_refused(tag: str) -> str | None:
    """Why a net's folder may not be used as it stands: its training was cut (PREREG: VOID, repeated, never continued),
    even at its last look, or looks saw another process of this Python on the GPU. None: nothing there, or a net that finished."""
    record, looks = NETS / tag / "record.json", NETS / tag / "gpu.json"
    if not record.exists() and not looks.exists():
        return None
    held = json.loads(looks.read_text()) if looks.exists() else []
    seen = sorted({p for g in held for p in g["peers"] or []})
    if seen:
        return f"{tag}: another process of this Python was on the GPU ({seen})"
    epochs = LEAFNET_SETTINGS["epochs"]
    done, want = len(json.loads(record.read_text())["history"]) if record.exists() else 0, 2 + math.ceil(epochs / D2B_GPU_EVERY)
    if done != epochs or len(held) != want:  # before it, each D2B_GPU_EVERY epochs from the first, after its last
        return f"{tag}: training was cut at epoch {done} of {epochs}, with {len(held)} of {want} GPU looks"
    if not (NETS / tag / "net.pt").exists():
        return f"{tag}: it finished and has no net.pt"
    return None


def d2b_net_differs(arm: dict, views: dict, seed: int | None = None) -> str | None:
    """Why a finished net is not this arm's: `train` would retrain over it, and both results stay (PREREG, VOID and bugs)."""
    path = NETS / arm["tag"] / "record.json"
    if not path.exists():
        return None
    rec, s = json.loads(path.read_text()), LEAFNET_SETTINGS
    want = dict(width=arm["width"], depth=s["depth"], fraction=1.0, data=fingerprint(views[arm["data"]]), epochs=s["epochs"],
                batch=s["batch"], lr=s["lr"], seed=s["seed"] if seed is None else seed, canonical=False, wd=float(arm["wd"]),
                pools=D2B_POOLS if arm["data"] == "union" else 1, dropout=float(arm.get("dropout", 0.0)),
                ema=float(arm.get("ema", 0.0)), loss=str(arm.get("loss", "huber")))
    differ = {k: (rec.get(k, CFG_DEFAULTS.get(k)), v) for k, v in want.items() if rec.get(k, CFG_DEFAULTS.get(k)) != v}
    return f"{arm['tag']}: its record is another arm's (record, arm): {differ}" if differ else None


def d2b_disk_refused(nets_json: dict) -> str | None:
    """Why nets.json no longer speaks for what is on disk: a net voided since it was written, or now refused or changed."""
    if nets_json["voided"] != d2b_voided():
        return f"nets.json lists {len(nets_json['voided'])} voided nets, {len(d2b_voided())} are on disk: rerun `d2b train`"
    for r in nets_json["nets"]:
        why = d2b_net_refused(r["tag"])
        if why:
            return why
    return d2b_records_differ(nets_json["nets"])


def d2b_voided() -> list[dict]:
    """The nets moved aside as VOID so far: their cause, and how far each got."""
    return [json.loads(p.read_text()) for p in sorted(NETS.glob("d2b-*-void-*/void.json"))]


def d2b_void_net(tag: str, cause: str, ls) -> int:
    """PREREG, VOID and bugs: a net's folder moved aside with its cause in it, kept and listed in nets.json; the next
    `d2b train` repeats that net."""
    need = [D2B_CHECK["tag"], *(x["tag"] for x in D2B_CANDIDATES), *(x["tag"] for x in D2B_REPORTED), "d2b-seed1"]
    if tag not in need or not cause.strip():
        raise SystemExit(f"--void needs a cause, and --tag one of {need}")
    folder = NETS / tag
    if not folder.exists():
        raise SystemExit(f"{folder}: no net to void")
    if (D2B_RECORDS / "selection.json").exists():
        raise SystemExit(f"{D2B_RECORDS / 'selection.json'} exists: voiding a net after the selection needs an amendment first")
    g = ls.gpu_state()
    if ls.busy(g) or d2b_blind(g):
        raise SystemExit("another process of this Python is on the GPU, or nvidia-smi could not say: stop it first, "
                         "it may be this net's training")
    record = folder / "record.json"
    rec = json.loads(record.read_text()) if record.exists() else {}
    n = 1 + len(list(NETS.glob(f"{tag}-void-*")))
    while (NETS / f"{tag}-void-{n}").exists():
        n += 1
    _write(folder / "void.json", json.dumps(dict(tag=tag, cause=cause, at=ls.utc(), folder=f"{tag}-void-{n}", commit=rec.get("commit"),
                                                 epochs=len(rec.get("history", [])), best=rec.get("best")), indent=1))
    folder.rename(NETS / f"{tag}-void-{n}")
    print(f"VOID: {tag} moved aside to {tag}-void-{n}: {cause}")
    return 0


def d2b_clients_line(nets_json: dict) -> str:
    rows = nets_json["nets"]
    return ("GPU clients seen during training (desktop apps do not void a net, a compute job does): "
            f"{sorted({x for r in rows for x in r['gpu_apps']})}; looks that could not say: "
            f"{sum(r['gpu_unknown'] for r in rows)} of {sum(r['gpu_looks'] for r in rows)}; voided nets: {len(nets_json['voided'])}")


def cmd_d2b_train(a) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import livesearch as ls

    if getattr(a, "void", None) is not None:
        return d2b_void_net(a.tag, a.void, ls)
    from parciver import value_net  # noqa: F401  # loaded before the commit is read, not inside the first `train`

    commit = _commit()  # once: the code loaded now is what every net runs on, whatever is committed meanwhile
    if not d2b_committed(commit):
        raise SystemExit(f"uncommitted changes under src or scripts, or git could not say ({commit!r}): commit first "
                         f"(every net's record holds the commit)")
    views, parts = d2b_data(a)
    d2b_refuse_other_data(views)
    n0 = json.loads((NETS / LEAFNET_N0 / "record.json").read_text())["best"]
    looks, recs, held, clients = {}, [], {}, set()

    def run(arm, seed=None):
        folder, seen = NETS / arm["tag"], []
        path = folder / "gpu.json"
        why = d2b_net_refused(arm["tag"]) or d2b_net_differs(arm, views, seed)
        if why:
            raise SystemExit(f"{why}: this run is VOID (PREREG, VOID and bugs). `d2b train --void \"<cause>\" --tag "
                             f"{arm['tag']}` keeps it with its cause, then this repeats it")
        fresh = not path.exists()

        def look():
            g = ls.gpu_state()
            seen.append(g)
            new = sorted(set(g["apps"]) - clients)
            if new:  # as they appear, so a compute job is seen while the net can still be stopped
                print(f"  {arm['tag']}: GPU clients new at this look (a compute job voids the net, read them): {new}", flush=True)
                clients.update(new)
            if len(seen) == 1 and g["peers"]:  # nothing trained: not a VOID, and a finished net stays as it is
                raise SystemExit(f"{arm['tag']}: another process of this Python on the GPU ({g['peers']}): "
                                 f"nothing trained, nothing changed; rerun when it has gone")
            if fresh or len(seen) > 1:  # one look at a net with looks: it is reused, and keeps those of its training
                folder.mkdir(parents=True, exist_ok=True)
                _write(path, json.dumps(seen, indent=1))
            if g["peers"]:
                raise SystemExit(f"{arm['tag']}: another process of this Python on the GPU ({g['peers']}): this "
                                 f"run is VOID (PREREG, VOID and bugs). `d2b train --void \"<cause>\" --tag {arm['tag']}`, then rerun")

        rec = d2b_train_arm(views, parts, arm, held, look, seed, commit=commit)
        looks[arm["tag"]] = json.loads(path.read_text())
        recs.append(rec)
        return rec

    def save():
        res = d2b_nets_record(check, n0, recs, looks)
        _write(D2B_RECORDS / "nets.json", json.dumps(res, indent=1))
        return res

    check = run(D2B_CHECK)
    res = save()
    print(f"d2b-check: best epoch {check['best']['epoch']}, selection loss {check['best']['val_loss']!r}; N0 {n0['epoch']}, "
          f"{n0['val_loss']!r}: {'REPRODUCED' if res['reproduced'] else 'NOT REPRODUCED'}")
    if not res["reproduced"]:
        print("nothing else is used until the cause is found and an amendment records it (PREREG)")
        print(d2b_clients_line(res))
        return 1
    cands = [run(arm) for arm in D2B_CANDIDATES]
    save()
    for arm in D2B_REPORTED:
        run(arm)
        save()
    pick = d2b_pick(cands)
    run(dict(tag="d2b-seed1", width=pick["width"], data="union", wd=pick["wd"]), seed=1)
    res = save()
    for r in recs:
        print(f"  {r['tag']:<18s} selection loss {r['best']['val_loss']:.6f}  MAE {r['best']['val_mae']:.3f}% pot  "
              f"(epoch {r['best']['epoch']})")
    print(d2b_clients_line(res) + ": read them before `d2b select`")
    return 0


def d2b_batches(views: dict, parts: dict, part: int, dev) -> dict:
    """One part of each dataset, in all and by source."""
    return {name: {label: Batches(v, np.flatnonzero((parts[name] == part) & keep), dev)
                   for label, keep in (("all", np.ones(len(v["source"]), bool)), ("real", v["source"] == 0),
                                       ("random", v["source"] == 1))} for name, v in views.items()}


def d2b_scores(net, batches: dict) -> dict:
    """float32 loss and MAE of a net on each of `d2b_batches`."""
    out = {}
    for name, sets in batches.items():
        for label, b in sets.items():
            e = evaluate(net, b)
            out.setdefault(name, {})[label] = dict(rows=b.n, loss=e["loss"], mae=e["mae"])
    return out


def d2b_selection(scores: dict, candidates: tuple[str, ...] = tuple(x["tag"] for x in D2B_CANDIDATES)) -> dict:
    """N1 = the candidate with the lower loss on the union's selection half (a tie:
    the first); it must be below N0 there or D2b stops."""
    loss = {t: scores[t]["union"]["all"]["loss"] for t in candidates}
    n1 = min(candidates, key=lambda t: loss[t])
    n0 = scores[LEAFNET_N0]["union"]["all"]["loss"]
    return dict(n1=n1, n1_loss=loss[n1], n0_loss=n0, stop=not loss[n1] < n0, relative=loss[n1] / n0 - 1)


def d2b_tag_sha(tag: str) -> str:
    return hashlib.sha256((NETS / tag / "net.pt").read_bytes()).hexdigest()


def cmd_d2b_select(a) -> int:
    import torch

    from parciver import value_net as vn

    nets_json = json.loads((D2B_RECORDS / "nets.json").read_text())
    if not nets_json.get("reproduced"):
        print("nets.json: d2b-check did not reproduce N0's record")
        return 1
    tags = [r["tag"] for r in nets_json["nets"]]
    need = [D2B_CHECK["tag"], *(x["tag"] for x in D2B_CANDIDATES), *(x["tag"] for x in D2B_REPORTED), "d2b-seed1"]
    if tags != need:
        print(f"nets.json holds {tags}, not the registered {need}")
        return 1
    refused = d2b_nets_refused(nets_json["nets"]) or d2b_disk_refused(nets_json)
    if refused:
        print(f"nets.json: {refused}")
        return 1
    print(d2b_clients_line(nets_json))
    views, parts = d2b_data(a)
    union = d2b_refuse_other_data(views)
    scores, match = {}, {}
    p_tag = json.loads((NETS / "leafnet.json").read_text())["pick"]
    batches = d2b_batches(views, parts, SELECT, torch.device("cuda"))
    for tag in (LEAFNET_N0, p_tag, *need):
        net, _ = vn.load(NETS / tag / "net.pt")
        scores[tag] = d2b_scores(net, batches)
        del net
        rec = json.loads((NETS / tag / "record.json").read_text())
        data = rec.get("data")
        # What its record says it scored, on the data it trained on: must come back to the bit from net.pt.
        name = next((k for k, v in dict(d1=LEAFNET_DATA, union=union).items() if v == data), "d1b")
        match[tag] = scores[tag][name]["all"]["loss"] == rec["best"]["val_loss"]
    sel = d2b_selection(scores)
    res = dict(sel, sha256={sel["n1"]: d2b_tag_sha(sel["n1"]), LEAFNET_N0: d2b_tag_sha(LEAFNET_N0)}, data=union, p=p_tag,
               cheaper_sha256={x["tag"]: d2b_tag_sha(x["tag"]) for x in D2B_REPORTED if x["width"] != D2B_CHECK["width"]},
               matches_record=match, scores=scores)
    _write(D2B_RECORDS / "selection.json", json.dumps(res, indent=1))
    for tag, s in scores.items():
        print(f"  {tag:<22s} union {s['union']['all']['loss']:.6f} (MAE {s['union']['all']['mae']:.3f}%)  "
              f"D1 {s['d1']['all']['loss']:.6f}  D1b {s['d1b']['all']['loss']:.6f}  record {'ok' if match[tag] else 'DIFFERS'}")
    print(f"N1 = {sel['n1']}: {sel['n1_loss']:.6f} vs N0 {sel['n0_loss']:.6f} ({100 * sel['relative']:+.2f}%)"
          + ("  -> D2b STOPS: N1 is not below N0" if sel["stop"] else ""))
    if not all(match.values()):
        print("a saved net does not score what its record says: no gate until the cause is found (PREREG, VOID and bugs)")
        return 1
    return 1 if sel["stop"] else 0


def d2b_gate_a_verdict(sets: dict, bar: float, min_gain: float = D2B_MIN_GAIN) -> dict:
    """PROCEED iff, on the primary set (D1b's gate half), the CI's upper end < 0
    and the relative change is <= min_gain, and N1's bare MAE on D1's gate half
    is under G2's bar."""
    av = sets["d1b"]["averaged"]
    below, gain, g2 = av["ci"][1] < 0, av["relative"] <= min_gain, sets["d1"]["bare"]["N1"] < bar
    return dict(primary="d1b", upper_below_zero=bool(below), gain_at_least_5pct=bool(gain), under_g2_bar=bool(g2),
                g2_bar=bar, min_gain=min_gain, proceed=bool(below and gain and g2))


def cmd_d2b_gate_a(a) -> int:
    import torch

    from parciver import value_net as vn

    sel = json.loads((D2B_RECORDS / "selection.json").read_text())
    if sel.get("stop", True) or not all(sel.get("matches_record", {"none": False}).values()):
        print(f"selection.json: no N1 to gate (stop {sel.get('stop')}, matches {sel.get('matches_record')})")
        return 1
    views, parts = d2b_data(a)
    union = d2b_refuse_other_data(views)
    d = views["union"]
    rows = np.flatnonzero(parts["union"] == GATE)
    b = Batches(d, rows, torch.device("cuda"))
    hb = hb_members(d["board"][rows])
    trivial = np.array([len(h) == 1 for h in hb])
    tags = {"N1": sel["n1"], "N0": LEAFNET_N0}
    sha = {k: d2b_tag_sha(t) for k, t in tags.items()}
    if sha["N1"] != sel["sha256"][sel["n1"]]:
        raise SystemExit(f"{sel['n1']}'s net.pt changed after the selection")
    mae, bare, defect = {}, {}, {}
    for k, tag in tags.items():
        net, _ = vn.load(NETS / tag / "net.pt")
        mae[k] = weighted_mae(hb_average(net, b, hb), b)
        bare[k] = evaluate(net, b)["mae_per_sample"].double().cpu().numpy() * 100
        defect[k] = symmetry_defect(net, b)
        del net
        torch.cuda.empty_cache()
    bar = 0.5 * json.loads((NETS / LEAFNET_N0 / "g2.json").read_text())["baseline_mae"]
    codes, source, dataset = canonical_codes(d["board"][rows]), d["source"][rows], d["dataset"][rows]
    sets = {}
    for name, m in (("d1b", dataset == 1), ("d1", dataset == 0), ("union", np.ones(len(rows), bool))):
        got = gate_a_result({k: v[m] for k, v in mae.items()}, {k: v[m] for k, v in bare.items()},
                            {k: v[m] for k, v in defect.items()}, codes[m], trivial[m], source[m], bar, cand="N1")
        got.pop("proceed")  # LEAF-NET's rule; D2b's is below
        sets[name] = got
    # Not a gate: N0 on D1's gate half, here among the union's batches, against LEAF-NET's scoring of it.
    leaf = json.loads((ROOT / "records" / "leaf-net" / "gate-a.json").read_text())["averaged"]["N0"]
    res = dict(d2b_gate_a_verdict(sets, bar), n1=sel["n1"], sha256=sha, data=union,
               n0_d1_minus_leafnet=sets["d1"]["averaged"]["N0"] - leaf, sets=sets)
    out = D2B_RECORDS / "gate-a.json"
    _write(out, json.dumps(res, indent=1))
    for name, s in sets.items():
        av = s["averaged"]
        print(f"Gate A {name} ({s['samples']} samples): MAE {av['N1']:.3f} vs {av['N0']:.3f}% pot, D {av['mean']:+.4f} "
              f"[{av['ci'][0]:+.4f}, {av['ci'][1]:+.4f}] ({100 * av['relative']:+.1f}%); bare {s['bare']['N1']:.3f} vs {s['bare']['N0']:.3f}")
    print(f"  {'PROCEED' if res['proceed'] else 'STOP: N0 stays'}  (upper < 0 {res['upper_below_zero']}, <= -5% "
          f"{res['gain_at_least_5pct']}, G2 bar {bar:.3f} {res['under_g2_bar']})  ({out})")
    return 0 if res["proceed"] else 1


def cmd_d2b(a) -> int:
    if (a.void is not None) != (a.tag is not None) or (a.void is not None and a.stage != "train"):
        raise SystemExit("--void and --tag go together, with `d2b train` alone")
    return {"data": cmd_d2b_data, "train": cmd_d2b_train, "select": cmd_d2b_select, "gate-a": cmd_d2b_gate_a}[a.stage](a)


def cached_checkdown(cd, d: dict, rows: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The cached check-down values of `rows` (`cd`: checkdown-val.npz), or
    why they cannot be used: a row not in the cache, a row whose inputs
    changed since, or a cache older than its input digests."""
    if "inputs" not in cd.files:
        return None, "the cached check-down values predate their input digests: run `baseline` again"
    key = {(int(i), int(s)): n for n, (i, s) in enumerate(zip(cd["index"], cd["source"]))}
    at = [key.get((int(d["index"][r]), int(d["source"][r]))) for r in rows]
    missing = sum(n is None for n in at)
    if missing:
        return None, f"{missing} gate samples have no cached check-down values: run `baseline` again"
    changed = int((cd["inputs"][at] != row_digests(d, rows)).sum())
    if changed:
        return None, f"{changed} gate samples have other inputs than their cached check-down values: run `baseline` again"
    return cd["values"][at], ""


def cmd_g2(a) -> int:
    import torch

    from parciver import value_net as vn

    tag = a.tag or json.loads((NETS / "final.json").read_text())["tag"]
    d, part = _data(a)
    rows = np.flatnonzero(part == GATE)
    if not len(rows):
        print("no gate samples in this data: nothing to gate")
        return 1
    with np.load(Path(a.data) / "checkdown-val.npz") as cd:
        base, why = cached_checkdown(cd, d, rows)
    if base is None:
        print(why)
        return 1
    data, net_data = fingerprint(d), json.loads((NETS / tag / "record.json").read_text()).get("data")
    if net_data != data:  # the gate boards are never trained on, whatever the shards: a note, not a refusal
        print(f"note: {tag} was trained on data {net_data}; the gate samples are from data {data}")
    net, meta = vn.load(NETS / tag / "net.pt")
    b = Batches(d, rows, torch.device("cuda"))
    e = evaluate(net, b, amp=False)
    net_mae = e["mae_per_sample"].double().cpu().numpy() * 100
    base_mae = vn.range_mae(base.astype(np.float64), d["values"][rows].astype(np.float64),
                            d["ranges"][rows].astype(np.float64)) * 100
    res = dict(tag=tag, net=meta, net_data=net_data, data=data, samples=len(rows),
               net_mae=float(net_mae.mean()), baseline_mae=float(base_mae.mean()),
               passed=bool(net_mae.mean() < 0.5 * base_mae.mean()))
    for src, name in ((0, "real"), (1, "random")):
        m = d["source"][rows] == src
        res[f"{name}_net_mae"], res[f"{name}_baseline_mae"] = float(net_mae[m].mean()), float(base_mae[m].mean())
    _write(NETS / tag / "g2.json", json.dumps(res, indent=1))
    print(f"G2 ({tag}) on {len(rows)} gate samples: net MAE {res['net_mae']:.3f}% pot, check-down "
          f"{res['baseline_mae']:.3f}% pot (bar {0.5 * res['baseline_mae']:.3f}): {'PASS' if res['passed'] else 'FAIL'}")
    print(f"  real {res['real_net_mae']:.3f} vs {res['real_baseline_mae']:.3f}; "
          f"random {res['random_net_mae']:.3f} vs {res['random_baseline_mae']:.3f}")
    return 0 if res["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "runs" / "turn-data"))
    ap.add_argument("--max-shards", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--workers", type=int, default=6)
    for name in ("train", "sweep", "final", "curve"):
        p = sub.add_parser(name)
        p.add_argument("--epochs", type=int, default=200 if name == "final" else 60)
        p.add_argument("--batch", type=int, default=2048)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--seed", type=int, default=0)
        if name == "sweep":
            p.add_argument("--widths", type=int, nargs="+", default=[512, 1024, 2048])
            p.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7])
        elif name != "final":  # curve: the sweep's choice by default
            p.add_argument("--width", type=int, required=name == "train")
            p.add_argument("--depth", type=int, required=name == "train")
        if name == "train":
            p.add_argument("--fraction", type=float, default=1.0)
            p.add_argument("--tag", default=None)
        if name in ("sweep", "curve"):
            p.add_argument("--fresh", action="store_true", help="retrain the nets a previous run finished")
    g = sub.add_parser("g2")
    g.add_argument("--tag", default=None, help="default: final.json's")
    sub.add_parser("leafnet")
    sub.add_parser("gate-a")
    g = sub.add_parser("d2b")
    g.add_argument("stage", choices=("data", "train", "select", "gate-a"))
    g.add_argument("--void", metavar="CAUSE", help="train: this net's run is VOID, for this cause: move it aside")
    g.add_argument("--tag", help="train --void: the net")
    a = ap.parse_args()
    global NETS
    NETS = Path(a.data).parent / "turn-net"
    return {"baseline": cmd_baseline, "train": cmd_train, "sweep": cmd_sweep, "final": cmd_final, "curve": cmd_curve,
            "g2": cmd_g2, "leafnet": cmd_leafnet, "gate-a": cmd_gate_a, "d2b": cmd_d2b}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
