# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Turn-start value data for the depth-limited flop search (FLOPSEARCH.md, D1).

  record    play v2-2B (turn search, recording solver) against itself and the
            scorecard's opponents at 200bb and 100bb stakes; save the distinct
            turn-round roots (`situations.npz`).
  generate  solve them, plus as many situations with DeepStack random ranges
            on the same public states, on the fused GPU solver; resumable
            shards (`shard-00000.npz`, ...). Progress in `generate.log`.
            `--terminals64`: KERNELS Q4 terminals (D1b). `--worker i --workers W`:
            this process makes the shards k with k % W == i, so W processes
            share one GPU (D1b); only worker 0 writes generate.json.
  generate --harvest PATH: D3's mixed layout (records/d3/PREREG.md 5.0.5): PATH is a harvest (the queries/ directory
            of scripts/d3_flop.py merge, or an .npz), `--samples` the total rows; the recorded situations (a pair each: the situation, then its
            random twin) and the harvested rows (one source, no twin) are interleaved one to one in shard order and the
            recorded rows left over come last (`mixed_layout`). Sources: 0 recorded, 1 its twin, 2 harvested.
  generate --harvest PATH --harvest-only: S0a's labels (records/d3/PREREG.md 5.1): every row is a harvested one (source 2, no
            recorded situation, an empty situations.npz made from the harvest's dtypes); `--samples` is the harvest's rows.
  record --exclude PATH [PATH ...]: drop situations already in any PATH (a
            situations.npz; D1b against D1, D1c against D1 and D1b; matched on
            float32 ranges, as stored). With --exclude, a --seed whose match
            seeds touch a registered block (SEED_BLOCKS) is refused.

Paths default to runs/turn-data. No dashboard (the author paused them)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ABS = str(ROOT / "data" / "abstraction")
V2 = str(ROOT / "runs" / "blueprint-20260927-141645" / "blueprint.bin")
V1 = str(ROOT / "runs" / "blueprint-20260914-144924" / "blueprint.bin")  # the scorecard baseline
STAKES = {"200bb": (50, 100, 20000), "100bb": (50, 100, 10000)}
# The scorecard's rows (runs/scorecard/scorecard-v2-200bb/benchmark.json), plus self-play.
OPPONENTS = ["self", "baseline", "nit", "station", "maniac", "min_raiser", "x0.6", "x1.5", "realistic"]
SEED = 520000  # the benchmarks use 410xxx
MAX_CHUNKS = 999  # match seed = seed + 1000 * cell + chunk: chunk 1000 would be the next cell's seeds
# Match-seed blocks already spent or reserved, each 1000 * 18 cells wide: [start, start + BLOCK_WIDTH).
BLOCK_WIDTH = 1000 * len(STAKES) * len(OPPONENTS)
MIXED = "interleave-1:1"  # generate.json layout of D3's rows
HARVEST_ONLY = "harvest-only"  # ... and of S0a's labelled queries
SEED_BLOCKS = {"D1": 520000, "D1b": 600000, "S0a": 780000, "harvest": 800000}  # D1c itself is 700000


def seed_clash(seed: int, blocks: dict | None = None) -> list[str]:
    """The registered blocks [start, start + BLOCK_WIDTH) that a record run at `seed` would touch."""
    blocks = SEED_BLOCKS if blocks is None else blocks
    return [f"{n} [{s}, {s + BLOCK_WIDTH})" for n, s in blocks.items() if seed < s + BLOCK_WIDTH and s < seed + BLOCK_WIDTH]


def _match(pe, agent, v1, opp: str, deals: int, stakes, seed: int, threads: int):
    sb, bb, stack = stakes
    kw = dict(seed=seed, threads=threads)
    if opp == "self":
        return agent.match_against("agent", deals, sb, bb, stack, **kw)
    if opp == "baseline":
        return agent.match_against("agent", deals, sb, bb, stack, other=v1.play_agent(), **kw)
    if opp in ("nit", "station", "maniac"):
        return agent.match_against("agent", deals, sb, bb, stack, other=v1.s32_policy().leak(opp, 0.2).play_agent(), **kw)
    if opp == "min_raiser":
        return agent.match_against("min_raiser", deals, sb, bb, stack, **kw)
    if opp in ("x0.6", "x1.5"):
        return agent.match_against("perturbed", deals, sb, bb, stack, other=v1.play_agent(), perturb=float(opp[1:]), **kw)
    if opp == "realistic":
        return agent.match_against("realistic", deals, sb, bb, stack, other=v1.play_agent(), **kw)
    raise ValueError(opp)


def _xkey(board, half, stack, first, ranges) -> bytes:
    """A situation's key against a stored situations.npz (ranges as float32, as stored)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray([int(c) for c in board] + [int(half), int(stack), int(first)], dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(ranges, dtype=np.float32).tobytes())
    return h.digest()


def _excluded(paths) -> set:
    """The keys of every situation in `paths` (None, one path, or a list of situations.npz)."""
    if paths is None:
        return set()
    keys: set = set()
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        d = np.load(path)
        board, half, stack, first, ranges = (d[k] for k in ("board", "half", "stack", "first", "ranges"))
        keys |= {_xkey(board[i], half[i], stack[i], first[i], ranges[i]) for i in range(len(half))}
    return keys


def cmd_record(a) -> int:
    import parciver_engine as pe

    from parciver import turn_values as tv

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "situations.npz"
    if path.exists():
        print(f"{path} exists")
        return 1
    cells = [(s, o) for s in STAKES for o in OPPONENTS]
    per_cell = -(-a.situations // len(cells))
    clash = seed_clash(a.seed) if a.exclude else []
    if clash:
        # match seeds are seed + 1000 * cell + chunk: a range touching a recorded block replays its deals
        print(f"--seed {a.seed}: its match seeds [{a.seed}, {a.seed + BLOCK_WIDTH}) overlap {', '.join(clash)}")
        return 1
    v2, v1 = pe.SavedBlueprint(ABS, V2), pe.SavedBlueprint(ABS, V1)
    rng = np.random.default_rng(a.seed)
    cols = {k: [] for k in ("board", "half", "stack", "first", "bb", "cell", "ranges")}
    seen: set = set()
    excluded = _excluded(a.exclude)
    if a.exclude:
        print(f"excluding {len(excluded)} distinct situations of {', '.join(map(str, a.exclude))}", flush=True)
    hits = []
    t0 = time.time()
    for c, (st, opp) in enumerate(cells):
        rec = tv.Recorder(STAKES[st][2], STAKES[st][1])
        agent = v2.play_agent().with_search(from_turn=True, solver=rec)
        chunk = 0
        while True:
            while len(rec.situations) < per_cell and chunk < MAX_CHUNKS:
                _match(pe, agent, v1, opp, a.chunk, STAKES[st], a.seed + 1000 * c + chunk, a.threads)
                chunk += 1
            fresh = [s for s in rec.situations if (s.board, s.half, s.stack, s.first, s.ranges.tobytes()) not in seen]
            if not excluded:
                break
            n0 = len(fresh)
            fresh = [s for s in fresh if _xkey(s.board, s.half, s.stack, s.first, s.ranges) not in excluded]
            if len(fresh) >= per_cell or chunk >= MAX_CHUNKS:
                hits.append(n0 - len(fresh))
                break
            # With --exclude, top up: record one more chunk until the cell has per_cell fresh situations.
            _match(pe, agent, v1, opp, a.chunk, STAKES[st], a.seed + 1000 * c + chunk, a.threads)
            chunk += 1
        if len(fresh) < per_cell:  # at the chunk cap
            print(f"  {st} {opp}: {len(fresh)} fresh situations after {chunk} chunks, short of {per_cell}")
            return 1
        pick = rng.permutation(len(fresh))[:per_cell]
        for i in sorted(pick):
            s = fresh[i]
            seen.add((s.board, s.half, s.stack, s.first, s.ranges.tobytes()))
            for k, v in (("board", s.board), ("half", s.half), ("stack", s.stack), ("first", s.first), ("bb", s.bb),
                         ("cell", c), ("ranges", s.ranges.astype(np.float32))):
                cols[k].append(v)
        print(f"  {st} {opp:>10s}: {len(pick)} of {len(rec.situations)} distinct ({chunk * a.chunk} deals)  "
              + (f"{hits[-1]} already in the excluded files  " if excluded else "") + f"{time.time() - t0:.0f}s", flush=True)
    order = np.random.default_rng(a.seed + 1).permutation(len(cols["half"]))  # shards mix every cell
    arrays = {k: np.asarray(v)[order] for k, v in cols.items()}
    arrays["board"] = arrays["board"].astype(np.int8)
    extra = dict(excluded_hits=np.asarray(hits)) if excluded else {}
    np.savez(path, **arrays, cells=np.array([f"{s}/{o}" for s, o in cells]), **extra)
    print(f"saved {len(order)} situations to {path}")
    return 0


def _row(data, j: int) -> dict:
    return {k: data[k][j] for k in ("board", "half", "stack", "first", "bb", "ranges")}


def _prepare(row: dict, j: int, seed: int):
    """Host-side work for real situation `j` (`row`: its columns): its
    random-range twin on the same public state (seeded by `j`) and both
    solver plans. Runs in a worker process: in a thread it would hold the GIL
    the solve loop needs (11% fewer solves per second, measured)."""
    from parciver import turn_values as tv
    from parciver.gpu_fused import host_plan
    from parciver.gpu_search import Hands, Tree

    base = dict(board=tuple(int(c) for c in row["board"]), half=int(row["half"]),
                stack=int(row["stack"]), first=int(row["first"]), bb=int(row["bb"]))
    hands = Hands(list(base["board"]))
    order = tv.strength_order(hands)
    rng = np.random.default_rng([seed, j])
    sits = [tv.Situation(ranges=row["ranges"].astype(np.float64), **base),
            tv.Situation(ranges=np.stack([tv.random_range(order, rng), tv.random_range(order, rng)]), **base)]
    tree = Tree(sits[0].spec(), hands)
    return sits, [host_plan(hands, tree, list(s.ranges)) for s in sits]


def _prepare_one(row: dict, j: int, seed: int, src: int):
    """`_prepare`'s situation `src` alone (0: the row's own ranges, 1: its random twin, drawn from the same
    `default_rng([seed, j])`, so bit for bit what `_prepare` makes), with its plan. A harvested row is source 0."""
    from parciver import turn_values as tv
    from parciver.gpu_fused import host_plan
    from parciver.gpu_search import Hands, Tree

    base = dict(board=tuple(int(c) for c in row["board"]), half=int(row["half"]),
                stack=int(row["stack"]), first=int(row["first"]), bb=int(row["bb"]))
    hands = Hands(list(base["board"]))
    if src == 0:
        sit = tv.Situation(ranges=row["ranges"].astype(np.float64), **base)
    elif src == 1:
        order = tv.strength_order(hands)
        rng = np.random.default_rng([seed, j])
        sit = tv.Situation(ranges=np.stack([tv.random_range(order, rng), tv.random_range(order, rng)]), **base)
    else:
        raise ValueError(f"source {src}: 0 or 1")
    return [sit], [host_plan(hands, Tree(sit.spec(), hands), list(sit.ranges))]


def mixed_layout(pairs: int, harvested: int) -> tuple[np.ndarray, np.ndarray]:
    """D3's row order (records/d3/PREREG.md 5.0.5): `pairs` recorded situations make 2 * pairs rows (a situation,
    then its random twin), interleaved one to one with `harvested` rows, the recorded rows left over at the end.
    Returns (source, index) per global row: source 0 / 1 = recorded situation `index` / its twin, 2 = harvested row
    `index`. Row 2k is recorded row k and row 2k + 1 harvested row k, for k < harvested."""
    rec = 2 * pairs
    if not 0 <= harvested <= rec:
        raise ValueError(f"{harvested} harvested rows against {rec} recorded: the recorded rows must cover them")
    src = np.empty(rec + harvested, dtype=np.int8)
    idx = np.empty(rec + harvested, dtype=np.int64)
    k = np.arange(harvested)
    src[0:2 * harvested:2], idx[0:2 * harvested:2] = k % 2, k // 2
    src[1:2 * harvested:2], idx[1:2 * harvested:2] = 2, k
    rest = np.arange(harvested, rec)
    src[2 * harvested:], idx[2 * harvested:] = rest % 2, rest // 2
    return src, idx


def harvest_only_layout(harvested: int) -> tuple[np.ndarray, np.ndarray]:
    """S0a's row order: every row source 2, row k the harvest's row k."""
    return np.full(harvested, 2, dtype=np.int8), np.arange(harvested, dtype=np.int64)


def merge_meta(old: dict | None, meta: dict) -> tuple[dict, list[str]]:
    """The generate.json to write, and the settings that clash with the shards
    an earlier run left (`old`: its generate.json, None in a new directory).
    Shards already written fix everything but the sample count (a run may be
    extended). `code` stays the first run's, which wrote the first shards;
    each resume's is appended to `resumed`."""
    if old is None:
        return meta, []
    old = {"gamma": 2, "plus": False, **old}  # written before these were recorded
    old = {"terminals": "float32", "workers": 1, **old}  # D1's, written before these were recorded
    meta = {"terminals": "float32", "workers": 1, **meta}
    clash = [f"{k}={old[k]}" for k in ("iters", "gamma", "plus", "shard", "seed", "dtype", "terminals", "workers")
             if old[k] != meta[k]]
    mixed = ("layout", "harvest", "samples") if (old.get("layout") or meta.get("layout")) else ()  # D3: samples fix the layout
    clash += [f"{k}={old.get(k)}" for k in mixed if old.get(k) != meta.get(k)]
    return dict(meta, code=old.get("code", "unknown"), resumed=old.get("resumed", []) + [meta["code"]]), clash


def cmd_generate(a) -> int:
    import concurrent.futures as cf
    import multiprocessing

    import torch

    from parciver import turn_values as tv

    import subprocess

    out = Path(a.out)
    only = getattr(a, "harvest_only", False)
    harvest = load_harvest(a.harvest) if a.harvest else None  # (an empty one is refused inside)
    if only and harvest is None:
        print("--harvest-only needs --harvest")
        return 1
    if only and a.samples != len(harvest["half"]):  # (checked before anything is written)
        print(f"--harvest-only: --samples {a.samples} must be the harvest's {len(harvest['half'])} rows")
        return 1
    if only and not (out / "situations.npz").exists():  # no recorded situation: an empty file with the harvest's dtypes
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "situations.npz", **{k: np.asarray(harvest[k][:0]) for k in ("board", "half", "stack", "cell") if k in harvest})
    data = dict(np.load(out / "situations.npz")) if not only else {k: np.asarray(harvest[k][:0]) for k in harvest}
    n_real = len(data["half"])
    n_h = len(harvest["half"]) if harvest is not None else 0
    if only:
        pairs = 0
    elif harvest is not None:
        if a.samples - n_h <= 0 or (a.samples - n_h) % 2 or (a.samples - n_h) // 2 > n_real:
            print(f"--samples {a.samples} with {n_h} harvested rows leaves {a.samples - n_h} recorded rows: "
                  f"an even number, at most {2 * n_real} (the recorded situations, a pair each)")
            return 1
        pairs = (a.samples - n_h) // 2
    else:
        pairs = min(n_real, a.samples // 2)
    log = open(out / "generate.log", "a", encoding="utf-8")
    git = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True, text=True)
    meta = dict(iters=a.iters, gamma=a.gamma, plus=False, samples=2 * pairs, shard=a.shard, seed=a.seed,
                dtype="float32", units="pots per unit of opponent range mass; ranges as stored (float32)",
                code=git.stdout.strip() or "unknown")
    if harvest is not None:
        meta.update(samples=a.samples, layout=HARVEST_ONLY if only else MIXED, harvest=sha256_tree(a.harvest), harvest_rows=n_h)
    if a.terminals64:
        meta["terminals"] = "q4"  # absent = D1's float32 terminals
    if not 0 <= a.worker < a.workers:
        print(f"--worker must lie in [0, {a.workers})")
        return 1
    if a.workers > 1:
        meta["workers"] = a.workers  # absent = 1; a clash on resume (two splits would share shards)
    mp = out / "generate.json"
    if a.worker > 0 and not mp.exists():
        print(f"start worker 0 first: {mp} does not exist yet")
        return 1
    meta, clash = merge_meta(json.loads(mp.read_text()) if mp.exists() else None, meta)
    if clash:
        print(f"existing shards use other settings: {', '.join(clash)}")
        return 1
    if a.worker == 0:  # written whole, then renamed: another worker never reads a partial file
        tmp = out / "generate.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(mp)
    if harvest is not None:
        return _generate_mixed(a, out, data, harvest, meta, log, pairs)
    per = a.shard // 2
    shards = -(-pairs // per)
    mine = [k for k in range(shards) if k % a.workers == a.worker]
    tag = f"w{a.worker}/{a.workers} " if a.workers > 1 else ""
    def complete(k: int) -> bool:
        # A shard is final once written, unless an earlier run with fewer samples cut it short.
        path = out / f"shard-{k:05d}.npz"
        return path.exists() and len(np.load(path)["index"]) == 2 * (min((k + 1) * per, pairs) - k * per)

    todo = [k for k in mine if not complete(k)]
    t0, solved = time.time(), 0
    # spawn on every platform: D1's (Windows) and the probe's, never fork after the GPU runtime is up
    with cf.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        for k in todo:
            path = out / f"shard-{k:05d}.npz"
            js = list(range(k * per, min((k + 1) * per, pairs)))
            fut = pool.submit(_prepare, _row(data, js[0]), js[0], a.seed)
            rows = {x: [] for x in ("index", "source", "values", "game", "expl", "ranges")}
            for i, j in enumerate(js):
                sits, plans = fut.result()
                if i + 1 < len(js):
                    fut = pool.submit(_prepare, _row(data, js[i + 1]), js[i + 1], a.seed)
                for src, (sit, plan) in enumerate(zip(sits, plans)):
                    r = tv.solve(sit, None, a.iters, dtype=torch.float32, plan=plan, gamma=a.gamma, plus=meta["plus"],
                                 terminals64=a.terminals64)
                    rows["index"].append(j)
                    rows["source"].append(src)
                    rows["values"].append(r["values"].astype(np.float32))
                    rows["game"].append(r["game"])
                    rows["expl"].append(r["exploitability"])
                    rows["ranges"].append(sit.ranges.astype(np.float32))
                    solved += 1
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **{x: np.asarray(v) for x, v in rows.items()})
            tmp.replace(path)
            el = time.time() - t0
            left = (len(todo) - todo.index(k) - 1) * a.shard * el / max(solved, 1)
            msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag}shard {k + 1}/{shards}  {solved / el:.2f} solves/s  "
                   f"mean expl {100 * np.mean(rows['expl']):.2f}% pot  eta {left / 3600:.1f} h  "
                   f"gpu mem {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    return 0


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path) -> str:
    """The sha256 of a harvest: its file's, or for a directory of .npy columns the hash of each file's name and bytes
    in name order (a 500k-row harvest is 4.5 GB: `load_harvest` memory-maps it)."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for f in sorted(path.glob("*.npy")):
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def load_harvest(path) -> dict:
    """A harvest (`d3_flop.py merge`'s directory of .npy columns, memory-mapped; or a .npz) as column -> array."""
    path = Path(path)
    got = {f.stem: np.load(f, mmap_mode="r") for f in sorted(path.glob("*.npy"))} if path.is_dir() else dict(np.load(path))
    if not {"half", "ranges"} <= set(got):  # an empty or wrong directory must never fall back to the pairs layout
        raise SystemExit(f"{path} is not a harvest: it has columns {sorted(got)}, needs at least half and ranges")
    return got


def _generate_mixed(a, out: Path, data: dict, harvest: dict, meta: dict, log, pairs: int) -> int:
    """`cmd_generate` for D3's mixed layout: shard k holds global rows [k * shard, (k + 1) * shard) of `mixed_layout`,
    each solved alone (a recorded situation and its twin are two rows, usually in one shard but not always)."""
    import concurrent.futures as cf
    import multiprocessing

    import torch

    from parciver import turn_values as tv

    n_h = len(harvest["half"])
    src, idx = harvest_only_layout(n_h) if meta.get("layout") == HARVEST_ONLY else mixed_layout(pairs, n_h)
    total = len(src)
    shards = -(-total // a.shard)
    mine = [k for k in range(shards) if k % a.workers == a.worker]
    tag = f"w{a.worker}/{a.workers} " if a.workers > 1 else ""

    def rows_of(k: int) -> range:
        return range(k * a.shard, min((k + 1) * a.shard, total))

    def complete(k: int) -> bool:
        path = out / f"shard-{k:05d}.npz"
        return path.exists() and len(np.load(path)["index"]) == len(rows_of(k))

    def job(g: int):
        d = harvest if src[g] == 2 else data
        return _row(d, int(idx[g])), int(idx[g]), (0 if src[g] == 2 else int(src[g]))

    todo = [k for k in mine if not complete(k)]
    t0, solved = time.time(), 0
    with cf.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        for k in todo:
            path = out / f"shard-{k:05d}.npz"
            gs = list(rows_of(k))
            row, j, s0 = job(gs[0])
            fut = pool.submit(_prepare_one, row, j, a.seed, s0)
            rows = {x: [] for x in ("index", "source", "values", "game", "expl", "ranges")}
            for i, g in enumerate(gs):
                sits, plans = fut.result()
                if i + 1 < len(gs):
                    row, j, s0 = job(gs[i + 1])
                    fut = pool.submit(_prepare_one, row, j, a.seed, s0)
                r = tv.solve(sits[0], None, a.iters, dtype=torch.float32, plan=plans[0], gamma=a.gamma, plus=meta["plus"],
                             terminals64=a.terminals64)
                rows["index"].append(int(idx[g]))
                rows["source"].append(int(src[g]))
                rows["values"].append(r["values"].astype(np.float32))
                rows["game"].append(r["game"])
                rows["expl"].append(r["exploitability"])
                rows["ranges"].append(sits[0].ranges.astype(np.float32))
                solved += 1
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **{x: np.asarray(v) for x, v in rows.items()})
            tmp.replace(path)
            el = time.time() - t0
            left = (len(todo) - todo.index(k) - 1) * a.shard * el / max(solved, 1)
            msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag}shard {k + 1}/{shards}  {solved / el:.2f} solves/s  "
                   f"mean expl {100 * np.mean(rows['expl']):.2f}% pot  eta {left / 3600:.1f} h  "
                   f"gpu mem {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "turn-data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--situations", type=int, default=250_000)
    r.add_argument("--chunk", type=int, default=2000)
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--exclude", nargs="+", default=None, help="situations.npz files whose situations are not recorded again")
    g = sub.add_parser("generate")
    g.add_argument("--iters", type=int, required=True)
    g.add_argument("--gamma", type=int, default=2, help="DCFR strategy-sum exponent (FusedSolver gamma)")
    g.add_argument("--samples", type=int, default=500_000)
    g.add_argument("--shard", type=int, default=1000)
    g.add_argument("--seed", type=int, default=SEED + 7)
    g.add_argument("--terminals64", action="store_true", help="KERNELS Q4 terminals (D1b)")
    g.add_argument("--harvest", default=None, help="D3: a harvest file (queries.npz); --samples is then the total rows")
    g.add_argument("--harvest-only", action="store_true", help="S0a: label the harvest's rows alone (no recorded situations)")
    g.add_argument("--worker", type=int, default=0)
    g.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    return {"record": cmd_record, "generate": cmd_generate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
