# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D3's flop-spot recorder and leaf-query harvest (records/d3/PREREG.md 5.0.1 and 5.0.5, 5.1).

  record   play the blueprint (v2-8B where it can play the cell's stake, else v2-2B) over the 18 cells (200bb / 100bb x
           9 opponents) with a recording flop solver and keep the first job of each distinct flop-round root, per cell:
           `spots-<block>-cNN.pkl`, `spots-<block>.json` (blueprint per cell, deals, counts). Blocks: s0a, harvest.
  queries  replay the spots, round-robin across the cells, through the live flop search (N1 leaves, "sym", gamma 5,
           500 iterations, graph=False so a Python hook sees every iteration; `--fast`: a CUDA graph for iterations 1..499,
           the hook on the 500th only, the same rows bit for bit) and keep K queries per spot, uniform without
           replacement from the rows of iteration 500's two evaluate calls pooled (a query is one row as the hook sees it):
           `queries-<k>.npz` blocks of 100 spots, resumable.
  merge    join the blocks into `queries/` (one .npy per column, memory-mappable; `turn_data.py generate --harvest`)
           and `queries.json` (provenance).

Seed blocks (match seed = seed + 1000 * cell + chunk, 18 cells, so 18,000 wide) are in `BLOCKS` and every other footprint
of the repo in `FOOTPRINTS`; `clashes` refuses a block that touches another. Nothing here changes play code."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import turn_data as td  # noqa: E402

BP_8B = str(ROOT / "runs" / "bp-8b-stage2" / "blueprint.bin")
BP_8B_SHA256 = "fd27fe50324eba29f0454ab5c68604c2920f2bad18dd437544255d754c7ae071"  # records/bp-8b/RESULT.md
N1_SHA256 = "54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72"  # runs/turn-net/d2b-w2048-wd0.1/net.pt
ITERS, GAMMA, LEAVES = 500, 5.0, "sym"  # the live flop search (scripts/slumbot_run.py SEARCH, FLOP_GAMMA, LEAVES)
CHUNK = 500  # deals per recording chunk
CELLS = [(s, o) for s in td.STAKES for o in td.OPPONENTS]
N_CELLS = len(CELLS)
BLOCK = td.BLOCK_WIDTH  # 18,000
assert BLOCK == 1000 * N_CELLS

# The registered blocks (records/d3/PREREG.md 5.0.5). D1c is 700000; turn_data's table lacks it (its own record run).
BLOCKS = dict(td.SEED_BLOCKS, D1c=700000)
# The other seeds the repo has spent or fixed, as [start, end): a block must touch none of them (D1c's overlap with the
# rating's anchor seeds is known and benign: those matches are not a D3 gate; D1b's with the rating's calibration seeds
# was spent before this table existed). `KNOWN_OVERLAPS` is what the test expects, nothing else.
FOOTPRINTS = {
    "flop_eval calibration": (540000, 540001),
    "flop_m1 M1": (550000, 550001),
    "benchmarks 410xxx": (410000, 411000),
    "smoke 777_00x": (777003, 777007),
    "rating calibration": (600000, 600100),  # 600000 + 10 * min(i, j) + max(i, j), scripts/rating.py
    "rating anchors": (700000, 700010),  # 700000 + anchor index (<= 10 anchors)
}
KNOWN_OVERLAPS = {frozenset(("D1b", "rating calibration")), frozenset(("D1c", "rating anchors"))}

# One stream per block: spots round-robin over the cells (stream index s -> cell s % 18, rank s // 18).
STREAMS = {
    "s0a": dict(seed=BLOCKS["S0a"], spots=200, queries=10, rows=2000),
    "harvest": dict(seed=BLOCKS["harvest"], spots=10204, queries=49, rows=499968),
}


BLOCK_NAME = {"s0a": "S0a", "harvest": "harvest"}
assert all(STREAMS[k]["seed"] == BLOCKS[v] for k, v in BLOCK_NAME.items())


def spans() -> dict:
    """Every block and footprint as name -> [start, end)."""
    out = {n: (s, s + BLOCK) for n, s in BLOCKS.items()}
    out.update(FOOTPRINTS)
    return out


def overlaps() -> set:
    """Every pair of spans that touch."""
    sp = spans()
    names = sorted(sp)
    return {frozenset((a, b)) for i, a in enumerate(names) for b in names[i + 1:]
            if sp[a][0] < sp[b][1] and sp[b][0] < sp[a][1]}


def clashes(name: str) -> list[str]:
    """The other spans that block `name` touches, apart from the known benign overlaps."""
    return sorted(next(iter(p - {name})) for p in overlaps() if name in p and p not in KNOWN_OVERLAPS)


def stream_cell(s: int) -> tuple[int, int]:
    """Stream index -> (cell, rank within the cell)."""
    return s % N_CELLS, s // N_CELLS


def per_cell(spots: int) -> list[int]:
    """How many of the first `spots` stream spots each cell gets."""
    return [len(range(c, spots, N_CELLS)) for c in range(N_CELLS)]


def quota(block: str) -> int:
    """Spots recorded per cell for a block: the largest per-cell count of its stream (567 for the harvest)."""
    return max(per_cell(STREAMS[block]["spots"]))


def take(block: str, s: int) -> int:
    """Queries kept from stream spot `s`: the block's K, the last spot truncated so exactly `rows` remain."""
    cfg = STREAMS[block]
    return max(0, min(cfg["queries"], cfg["rows"] - cfg["queries"] * s))


# ------------------------------------------------------------------ the leaf-query hook


class _NoTurn:
    """The turn solver a harvest search must never reach: it only replays flop jobs."""

    def __call__(self, job):
        raise RuntimeError("a flop-harvest search was asked to solve a turn or river job")


class QueryLog:
    """Every evaluate call's rows: the count, and the last two calls' inputs (cloned, so later work cannot touch them)."""

    def __init__(self):
        self.on = True  # off: the fast search's graph iterations, whose Python callables must not be logged
        self.reset()

    def reset(self):
        self.calls, self.last = 0, []

    def add(self, ranges, board, live, half, stack):
        if not self.on:
            return
        self.calls += 1
        self.last = (self.last + [(ranges.clone(), board, live, half.clone(), stack.clone())])[-2:]


def make_search(net, hook: bool = True, fast: bool = False):
    """`FlopSearch(net, leaves="sym", gamma=5)` with graph=False and no turn solver; with `hook`, its leaf evaluators log
    every call to `.log` (`HarvestSearch`). Without it: the same search, unhooked (the test's reference).
    `fast` (hook only): iterations 1..N-1 run as the live search runs them (a CUDA graph, nothing logged) and the last
    one eagerly with the hook on, so `.log` holds exactly iteration N's two calls. The kernels and
    tensors are the same, so the logged rows are bit-identical to the graph=False search's (a test proves it)."""
    from parciver import flop_search as fsmod
    from parciver.flop_search import FlopSearch

    if not hook:
        return FlopSearch(net, gamma=GAMMA, turn=_NoTurn(), graph=False, leaves=LEAVES)

    class HarvestSearch(FlopSearch):
        def __init__(self):
            super().__init__(net, gamma=GAMMA, turn=_NoTurn(), graph=fast, leaves=LEAVES)
            self.log = QueryLog()
            self.fast = fast

        def __call__(self, job):
            if not self.fast:
                return super().__call__(job)
            log, real = self.log, fsmod.FlopSolver

            class LastEager(real):
                def solve(self, iters, graph=True):
                    log.on = False
                    try:
                        real.solve(self, iters - 1, graph=True)  # the side-stream step, the capture and the replays
                    finally:
                        log.on = True
                    self.step()  # iteration `iters`, in Python: the hook sees its two evaluate calls

            fsmod.FlopSolver = LastEager
            try:
                return super().__call__(job)
            finally:
                fsmod.FlopSolver = real

        def tables(self, board):
            got = super().tables(board)
            for obj in got[1:]:  # NetLeaves and (for a symmetric flop) SymLeaves
                if obj is not None and not getattr(obj, "_hooked", False):
                    obj.evaluate = self._wrap(obj.evaluate)
                    obj._hooked = True
            return got

        def _wrap(self, inner):
            def f(ranges, board, live, half, stack):
                self.log.add(ranges, board, live, half, stack)
                return inner(ranges, board, live, half, stack)

            return f

    return HarvestSearch()


def replay_job(job: dict) -> dict:
    """A recorded flop job as the live search gets it (`_KEYS` + hero as stored) with the live iteration count."""
    return dict(job, iters=ITERS)


def last_iteration(search, job: dict) -> list:
    """Solve `job` and return the last two evaluate calls (traverser 0 and 1 of iteration 500). Refuses unless the search
    made exactly two calls per iteration, i.e. the hook saw every iteration."""
    search.log.reset()
    search(replay_job(job))
    log = search.log
    want = 2 if getattr(search, "fast", False) else 2 * ITERS
    if log.calls != want:
        raise RuntimeError(f"{log.calls} evaluate calls for {ITERS} iterations: not two per iteration"
                           if want != 2 else f"{log.calls} evaluate calls in the fast search's last iteration, 2 expected")
    return log.last


def pool_sizes(last: list) -> list[int]:
    return [int(x.shape[0] * x.shape[1]) for x, *_ in last]


def draw(block: str, s: int, pool: int) -> np.ndarray:
    """The pool rows kept from stream spot `s`: `queries` drawn uniform without replacement, in drawn order, seeded by
    (block seed, s); the last harvest spot keeps the first `take` of them."""
    cfg = STREAMS[block]
    if pool < cfg["queries"]:
        raise RuntimeError(f"spot {s}: {pool} rows in the last iteration, {cfg['queries']} wanted")
    rng = np.random.default_rng([cfg["seed"], s])
    return rng.choice(pool, cfg["queries"], replace=False)[:take(block, s)]


def rows_to_situations(last: list, picks) -> dict:
    """Pool rows `picks` (call 1's rows then call 2's, each row l * T + t of its call) as turn situations: the
    unit-normalised ranges over the board's 1128 live holdings, the board, half and stack."""
    n0 = pool_sizes(last)[0]
    cols = {k: [] for k in ("board", "half", "stack", "ranges", "call", "row")}
    for p in picks:
        call, r = (0, int(p)) if p < n0 else (1, int(p) - n0)
        x, board, live, half, stack = last[call]
        l, t = divmod(r, x.shape[1])
        keep = (live[t] > 0).cpu().numpy()
        row = x[l, t].cpu().numpy()  # [2, 1326]
        if int(keep.sum()) != td_holdings() or np.any(row[:, ~keep] != 0):
            raise RuntimeError(f"row {p}: {int(keep.sum())} live holdings, or mass on a dead one")
        h, st = float(half[l]), float(stack[l])
        if h != int(h) or st != int(st) or h <= 0:
            raise RuntimeError(f"row {p}: half {h}, stack {st}: not whole chips")
        cols["board"].append(board[t].cpu().numpy().astype(np.int8))
        cols["half"].append(int(h))
        cols["stack"].append(int(st))
        cols["ranges"].append(row[:, keep].astype(np.float32))
        cols["call"].append(call)
        cols["row"].append(int(p))
    return cols


def td_holdings() -> int:
    from parciver import turn_values as tv

    return tv.HOLDINGS


# ------------------------------------------------------------------ the recorder


def sha256_of(path) -> str:
    return td.sha256_file(path)


def probe_blueprint(pe, abs_dir: str, stakes) -> tuple[str, str]:
    """(path, why) for a stake: v2-8B if its file is the registered one and it plays a deal at that stake, else v2-2B.
    Decided by what loads and plays, not by a result (PREREG 5.0.5)."""
    sb, bb, stack = stakes
    if Path(BP_8B).exists():
        try:
            if sha256_of(BP_8B) != BP_8B_SHA256:
                raise RuntimeError("not the registered v2-8B file")
            pe.SavedBlueprint(abs_dir, BP_8B).play_agent().match_against("agent", 1, sb, bb, stack, seed=0, threads=1)
            return BP_8B, "v2-8B plays the stake"
        except Exception as e:  # noqa: BLE001 - any failure to load or play is the answer
            return td.V2, f"v2-8B does not: {type(e).__name__}: {str(e)[:120]}"
    return td.V2, "v2-8B file missing"


def record_cell(pe, v1, bp_path: str, c: int, block: str, deals: int = CHUNK) -> dict:
    """Play cell `c` until it has `quota(block)` distinct flop-round roots (or 999 chunks); the picked jobs in rank order."""
    import flop_eval

    st, opp = CELLS[c]
    cfg = STREAMS[block]
    rec = flop_eval.FlopRecorder()
    agent = pe.SavedBlueprint(td.ABS, bp_path).play_agent().with_search(from_flop=True, solver=rec)
    want, chunk = quota(block), 0
    while len(rec.jobs) < want and chunk < td.MAX_CHUNKS:
        # threads=1: the jobs arrive in play order, so the picks are seeded
        td._match(pe, agent, v1, opp, deals, td.STAKES[st], cfg["seed"] + 1000 * c + chunk, 1)
        chunk += 1
    if len(rec.jobs) < want:
        raise SystemExit(f"cell {c} ({st} {opp}): {len(rec.jobs)} distinct flop roots after {chunk} chunks, short of {want}")
    pick = np.random.default_rng([cfg["seed"], c]).permutation(len(rec.jobs))[:want]
    return dict(block=block, cell=c, stakes=st, opponent=opp, seed=cfg["seed"], blueprint=bp_path, deals=chunk * deals,
                distinct=len(rec.jobs), jobs=[rec.jobs[i] for i in pick])


def cmd_record(a) -> int:
    import parciver_engine as pe

    if clashes(BLOCK_NAME[a.block]):
        print(f"block {a.block} touches {clashes(BLOCK_NAME[a.block])}")
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v1 = pe.SavedBlueprint(td.ABS, td.V1)
    by_stake, summary, t0 = {}, [], time.time()
    for c, (st, opp) in enumerate(CELLS):
        path = out / f"spots-{a.block}-c{c:02d}.pkl"
        if st not in by_stake:
            by_stake[st] = probe_blueprint(pe, td.ABS, td.STAKES[st])
            print(f"{st}: {by_stake[st][1]} -> {Path(by_stake[st][0]).parent.name}", flush=True)
        if path.exists():
            got = pickle.loads(path.read_bytes())
        else:
            got = record_cell(pe, v1, by_stake[st][0], c, a.block, a.chunk)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(got, protocol=4))
            tmp.replace(path)
        frozen = sum(1 for j in got["jobs"] if len(j["frozen"]))
        summary.append({k: got[k] for k in ("cell", "stakes", "opponent", "blueprint", "deals", "distinct")}
                       | dict(spots=len(got["jobs"]), frozen=frozen, why=by_stake[st][1]))
        print(f"  cell {c:2d} {st} {opp:>10s}: {len(got['jobs'])} of {got['distinct']} distinct roots ({got['deals']} deals), "
              f"{frozen} with a frozen strategy  {time.time() - t0:.0f}s", flush=True)
    (out / f"spots-{a.block}.json").write_text(json.dumps(dict(block=a.block, seed=STREAMS[a.block]["seed"], cells=summary,
                                                               bp_8b_sha256=BP_8B_SHA256), indent=1))
    print(f"recorded {sum(s['spots'] for s in summary)} spots over {N_CELLS} cells")
    return 0


# ------------------------------------------------------------------ the queries


def cmd_queries(a) -> int:
    import torch

    from parciver import value_net as vn

    cfg = STREAMS[a.block]
    spots, out = Path(a.spots), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net_sha = hashlib.sha256(Path(a.net).read_bytes()).hexdigest()
    if net_sha != N1_SHA256:
        print(f"--net is not N1 ({net_sha[:12]})")
        return 1
    net, _ = vn.load(a.net)
    search = make_search(net, fast=a.fast)
    cells: dict = {}
    t0, size = time.time(), 100
    for k in range(-(-cfg["spots"] // size)):
        path = out / f"queries-{k:03d}.npz"
        if path.exists():
            continue
        cols = {n: [] for n in ("board", "half", "stack", "ranges", "call", "row", "cell", "spot", "pool", "frozen")}
        for s in range(k * size, min((k + 1) * size, cfg["spots"])):
            c, rank = stream_cell(s)
            if c not in cells:
                cells[c] = pickle.loads((spots / f"spots-{a.block}-c{c:02d}.pkl").read_bytes())["jobs"]
            job = cells[c][rank]
            last = last_iteration(search, job)
            pool = sum(pool_sizes(last))
            got = rows_to_situations(last, draw(a.block, s, pool))
            n = len(got["half"])
            for key, v in got.items():
                cols[key] += v
            cols["cell"] += [c] * n
            cols["spot"] += [s] * n
            cols["pool"] += [pool] * n
            cols["frozen"] += [len(job["frozen"])] * n
            del last
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{n: np.asarray(v) for n, v in cols.items()})
        tmp.replace(path)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} block {k + 1}/{-(-cfg['spots'] // size)}  "
              f"{len(cols['half'])} queries  {time.time() - t0:.0f}s  gpu mem {torch.cuda.memory_reserved() / 2**30:.2f} GiB",
              flush=True)
    return 0


def cmd_merge(a) -> int:
    cfg = STREAMS[a.block]
    out = Path(a.out)
    files = sorted(out.glob("queries-[0-9][0-9][0-9].npz"))
    if len(files) != -(-cfg["spots"] // 100):
        print(f"{len(files)} query blocks, {-(-cfg['spots'] // 100)} wanted")
        return 1
    parts = [dict(np.load(f)) for f in files]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if len(cols["half"]) != cfg["rows"]:
        print(f"{len(cols['half'])} queries, exactly {cfg['rows']} wanted")
        return 1
    spot = cols["spot"]
    if not (np.diff(spot) >= 0).all() or len(np.unique(spot)) != cfg["spots"]:
        print("the queries are not in stream order, or a spot is missing")
        return 1
    dest = out / "queries"
    dest.mkdir(exist_ok=True)
    cols["first"] = np.ones(len(spot), dtype=np.int64)  # the seat that acts first on the turn
    cols["bb"] = np.full(len(spot), 100, dtype=np.int64)
    for k, v in cols.items():
        np.save(dest / f"{k}.npy", v)
    prov = dict(block=a.block, rows=int(len(spot)), spots=cfg["spots"], seed=cfg["seed"], per_spot=cfg["queries"],
                net_sha256=N1_SHA256, iters=ITERS, gamma=GAMMA, leaves=LEAVES, sha256=td.sha256_tree(dest),
                pool_sizes=sorted({int(p) for p in cols["pool"]}),
                frozen_spots=int(len({int(s) for s, f in zip(spot, cols["frozen"]) if f})),
                cells=json.loads((Path(a.spots) / f"spots-{a.block}.json").read_text())["cells"] if a.spots else None)
    (out / "queries.json").write_text(json.dumps(prov, indent=1))
    print(f"merged {prov['rows']} queries of {prov['spots']} spots to {dest} (sha256 {prov['sha256'][:12]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--block", choices=list(STREAMS), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--chunk", type=int, default=CHUNK)
    q = sub.add_parser("queries")
    q.add_argument("--block", choices=list(STREAMS), required=True)
    q.add_argument("--net", required=True)
    q.add_argument("--spots", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--fast", action="store_true", help="graph replay for iterations 1..499, the hook on the last only (bit-identical queries)")
    m = sub.add_parser("merge")
    m.add_argument("--block", choices=list(STREAMS), required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--spots", default=None)
    a = ap.parse_args()
    return {"record": cmd_record, "queries": cmd_queries, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
