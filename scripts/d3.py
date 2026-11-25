"""D3's driver (records/d3/PREREG.md 5.0.4): every stage after the labels, on the memory-mapped loader (`d3_load.py`).

    d3.py data            the pinned files checked, each dataset's fingerprint and counts (records/d3/data.json); before
                          D1C_MANIFEST is pinned it writes records/d3/d1c-manifest.sha256 and prints the sha for the amendment
    d3.py memory          the host peak at N2's ~1.9M training rows (records/d3/memory.json): N2 is not trained without it (8)
    d3.py look N          futility look 1 (251 shards) or 2 (501): one net with N2's recipe on D1 + D1b + the first shards of
                          D1c, scored on the old selection half (23,728 rows) against N1's (records/d3/look{N}.json)
    d3.py train           N2, then the seed-1 net (reported only): refused unless the memory number was told (--told GIB)
    d3.py select          N2 against N1 on the combined selection half; N1 and N2 scored on every dataset's (selection.json)
    d3.py gate-a          N2 against N1, H_b-averaged, on the gate halves (gate-a.json)
    d3.py s0a             S0a: N1's loss on the 2,000 labelled queries / 0.002217, bootstrapped over spots -> D1c's branch
    d3.py s0c             S0c: the mean prediction of N1 and d2b-seed1 on the old selection half, forward only

D3's views of the three datasets are contiguous slices of one loaded dict (`dataset` 0, 1, 2); the union's split and the
fingerprints are D2b's rules (`turn_net`), so D1 + D1b hashes to D2B_UNION and N1 rescores to its recorded number to the bit.
Anything that needs a number pinned by an amendment (D1C_MANIFEST, D3_UNION) refuses until it is."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import d3_load as dl  # noqa: E402
import turn_net as tn  # noqa: E402

ROOT = tn.ROOT
RECORDS = ROOT / "records" / "d3"
D1C_DIR = "turn-data-d1c"  # under the runs folder, beside D1's and D1b's
D1C_MANIFEST: str | None = None  # d1c-manifest.sha256's own sha256, by amendment after `d3 data`
D3_UNION: str | None = None  # D1 + D1b + D1c's fingerprint, by amendment after `d3 data` (as D2B_UNION)
N2_ARM = dict(tag="d3-n2", width=2048, data="union", wd=0.1, dropout=0.0, ema=0.0, loss="rw")  # N1's recipe with the range-weighted loss: the d3-sweep pin (records/d3/PREREG.md Amendment 1)
SEED1_TAG = "d3-n2-seed1"
LOOKS = {1: dict(shards=251, bar=0.98), 2: dict(shards=501, bar=0.96)}  # PREREG 5.2: continue iff loss <= bar x N1's
SHARD_ROWS = 1000
OLD_SELECT_ROWS = 23728  # the union's original selection half
MIN_GAIN = -0.04  # Gate A's primary set: the relative MAE change must be at or below this (PREREG 5.4)
POOL_ROWS = 500_000  # training rows a GPU pool holds: N1's were ~475k at 13.6 KB a row on the 12 GiB card
S0A_SPOTS, S0A_QUERIES, S0A_BAR = 200, 10, 1.5
S0C_BAR = 0.06
GPU_ROW_BYTES = 13_576
MEM_MARGIN = 0.6  # GiB: `d3 memory` uploads one pool; the selection batches and the net come on top (about 0.5 GiB and the weights)


# ------------------------------------------------------------------ small pure pieces


def pools_for(train_rows: int) -> int:
    """GPU pools for a training set: N1's own (2) up to 1M rows, more as it grows (the batches and draws are the same)."""
    return max(tn.D2B_POOLS, math.ceil(train_rows / POOL_ROWS))


def free_commit_gib() -> float:
    """The host's free commit now: what a process may still allocate before the allocator fails (page file included)."""
    g = 2.0 ** 30
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_page", "avail_page",
                                                          "total_virtual", "avail_virtual", "avail_ext")]

        s = Status()
        s.length = ctypes.sizeof(s)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            raise OSError("GlobalMemoryStatusEx failed")
        return s.avail_page / g
    info = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    return (info["CommitLimit"] - info["Committed_AS"]) * 1024 / g


def memory_refused(rec: dict | None, train_rows: int, free_now: float, told: float | None) -> str | None:
    """Why N2 may not train yet (PREREG 8): the measured host peak at this many rows must exist, be below the free commit
    now, and have been told to the author (`--told` repeats the number, to a tenth of a GiB)."""
    if not rec:
        return "records/d3/memory.json does not exist: run `d3 memory` first"
    if rec.get("train_rows") != train_rows:
        return f"memory.json measured {rec.get('train_rows')} training rows, this run has {train_rows}: measure again"
    peak = rec["peak_commit"]
    if not peak + MEM_MARGIN < free_now:
        return (f"the measured peak commit {peak:.1f} GiB (+ {MEM_MARGIN} GiB for the selection batches and the net) "
                f"is not below the free commit now ({free_now:.1f} GiB)")
    if told is None or round(told, 1) != round(peak, 1):
        return f"tell the author the measured peak commit first ({peak:.1f} GiB, free now {free_now:.1f}), then pass --told {peak:.1f}"
    return None


def look_verdict(loss: float, n1_loss: float, bar: float) -> dict:
    """Continue iff the look net's loss on the old selection half is at most `bar` x N1's."""
    return dict(loss=loss, n1_loss=n1_loss, ratio=loss / n1_loss, bar=bar, relative=loss / n1_loss - 1,
                **{"continue": bool(loss <= bar * n1_loss)})


def s0a_branch(lower: float) -> str:
    """D1c's recipe (PREREG 5.1): r's CI lower end at most 1.5 -> the pairs branch, above it -> half harvested."""
    return "pairs" if lower <= S0A_BAR else "mixed"


def s0c_gain(ensemble: float, n1: float, seed1: float) -> dict:
    """The ensemble's relative gain over N1 (the rule's) and over the better single net (reported)."""
    gain = 1 - ensemble / n1
    return dict(gain_vs_n1=gain, gain_vs_best_single=1 - ensemble / min(n1, seed1), distill_proposed=bool(gain >= S0C_BAR))


def d3_views(d: dict) -> dict:
    """The union and each dataset as views of its rows (no copies): D1, D1b, D1c in `dataset` order, `d12` = D1 + D1b."""
    ds = np.asarray(d["dataset"])
    n = [int((ds == k).sum()) for k in range(3)]
    if not all(n) or (np.diff(ds) < 0).any():
        raise ValueError(f"datasets do not come as D1, D1b, D1c in order: rows {n}")
    a, b = n[0], n[0] + n[1]

    def cut(lo, hi):
        return {k: v[lo:hi] for k, v in d.items()}

    return dict(union=d, d12=cut(0, b), d1=cut(0, a), d1b=cut(a, b), d1c=cut(b, len(ds)))


def labels_of(source: np.ndarray) -> dict:
    """Row masks of the label sets: all rows, and by source (0 recorded, 1 its random twin, 2 harvested)."""
    return {"all": np.ones(len(source), bool), "real": source == 0, "random": source == 1, "harvested": source == 2}


def summary(n2: np.ndarray, n1: np.ndarray, cluster: np.ndarray) -> dict:
    """D = N2 - N1 per sample (% of the pot): mean, board-clustered bootstrap CI, both nets' means, relative = D / mean N1."""
    return dict(tn.board_bootstrap(n2 - n1, cluster), N2=float(n2.mean()), N1=float(n1.mean()),
                relative=float(n2.mean() / n1.mean() - 1))


def gate_verdict(sets: dict, bar: float, min_gain: float = MIN_GAIN) -> dict:
    """PROCEED iff, on D1c's primary set (recorded rows only), the CI's upper end < 0 and the relative change <= min_gain;
    N2 is not significantly worse than N1 on D1's or D1b's gate half (the CI's lower end of D not above 0); and N2's bare
    MAE on D1's gate half is under G2's bar."""
    av = sets["d1c_recorded"]["averaged"]
    below, gain = av["ci"][1] < 0, av["relative"] <= min_gain
    worse = [k for k in ("d1", "d1b") if sets[k]["averaged"]["ci"][0] > 0]
    g2 = sets["d1"]["bare"]["N2"] < bar
    return dict(primary="d1c_recorded", upper_below_zero=bool(below), gain_at_least_4pct=bool(gain), regressed_on=worse,
                under_g2_bar=bool(g2), g2_bar=bar, min_gain=min_gain, proceed=bool(below and gain and not worse and g2))


def sample_losses(net, b) -> np.ndarray:
    """[M] each sample's loss as `evaluate` computes it (huber over the live holdings), float64."""
    import torch
    import torch.nn.functional as F

    from parciver import value_net as vn

    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, b.n, 4096):
            idx = torch.arange(s, min(s + 4096, b.n), device=b.dev)
            r, v, onehot, scal, live = b.get(idx, False)
            pred = net(r, onehot, scal, live).float()
            out.append((F.huber_loss(pred, v, reduction="none").flatten(1).mean(1) * (vn.HOLDINGS / vn.LIVE)).double().cpu().numpy())
    net.train()
    return np.concatenate(out)


# ------------------------------------------------------------------ data


def paths(a) -> tuple[Path, list[Path]]:
    runs = Path(a.data).parent
    tn.NETS = runs / "turn-net"  # `turn_net.main` sets it from --data; this does the same
    return runs, [runs / x for x in (*tn.D2B_DIRS, D1C_DIR)]


def load_all(a, caps=None, pinned: bool = True, log=print) -> tuple[dict, dict, dict, dict]:
    """The three datasets mapped; the registered files checked (D1b's manifest, and D1c's when the whole of it is loaded);
    D1's and D1 + D1b's fingerprints checked, and the union's against D3_UNION when `pinned`.
    Returns (d, views, parts, fingerprints); `parts` split by the canonical-board hash, per view."""
    runs, dirs = paths(a)
    tn.d2b_check_files(dirs[1], tn.D2B_RECORDS / "d1b-manifest.sha256")
    if caps is None and pinned:
        if D1C_MANIFEST is None:
            raise SystemExit("D1C_MANIFEST is not pinned: `d3 data`, then an amendment first")
        tn.d2b_check_files(dirs[2], RECORDS / "d1c-manifest.sha256", want=D1C_MANIFEST)
    d = dl.load_mapped(dirs, max_shards=caps, harvest=Path(a.harvest) if a.harvest else None, log=log)
    views = d3_views(d)
    fps = dict(d1=tn.fingerprint(views["d1"]), d12=tn.fingerprint(views["d12"]))
    if fps["d1"] != tn.LEAFNET_DATA:
        raise SystemExit(f"D1's fingerprint is {fps['d1']}, registered {tn.LEAFNET_DATA}")
    if fps["d12"] != tn.D2B_UNION:
        raise SystemExit(f"D1 + D1b's fingerprint is {fps['d12']}, registered {tn.D2B_UNION}")
    fps["union"] = tn.fingerprint(d)
    if pinned and (D3_UNION is None or fps["union"] != D3_UNION):
        raise SystemExit(f"the union's fingerprint is {fps['union']}, pinned {D3_UNION}: `d3 data`, then an amendment first")
    parts = {k: tn.split(v["board"]) for k, v in views.items()}
    return d, views, parts, fps


D1C_ROWS = 1_000_008  # PREREG 5.2: all 1,001 shards, the last with 8 rows


def d1c_incomplete(d1c: Path) -> str | None:
    """Why D1c may not be pinned yet: every shard there (1,001, the last with 8 rows), no half-written one, the registered count."""
    meta = json.loads((d1c / "generate.json").read_text()) if (d1c / "generate.json").exists() else {}
    shards = sorted(d1c.glob("shard-*.npz"))
    if any(p.name.endswith(".tmp.npz") for p in shards) or list(d1c.glob("*.tmp*")):
        return "a half-written shard is there"
    if meta.get("layout") != dl.MIXED:
        return f"generate.json layout {meta.get('layout')!r}, not {dl.MIXED!r}"
    if meta.get("samples") != D1C_ROWS or len(shards) != math.ceil(D1C_ROWS / SHARD_ROWS):
        return f"{len(shards)} shards, generate.json samples {meta.get('samples')}: D1c is {D1C_ROWS} rows in {math.ceil(D1C_ROWS / SHARD_ROWS)} shards"
    with np.load(shards[-1]) as z:
        last = len(z["index"])
    return None if last == D1C_ROWS % SHARD_ROWS else f"the last shard has {last} rows, not {D1C_ROWS % SHARD_ROWS}"


def d1c_manifest_text(d1c: Path) -> str:
    """`<sha>  <name>` lines (LF) of D1c's finished shards and situations.npz, sorted by name."""
    names = [p.name for p in sorted(d1c.glob("shard-*.npz")) if not p.name.endswith(".tmp.npz")] + ["situations.npz"]
    lines = []
    for name in sorted(names):
        h = hashlib.sha256()
        with open(d1c / name, "rb") as f:
            while block := f.read(1 << 24):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {name}")
    return "\n".join(lines) + "\n"


def data_record(views: dict, parts: dict, fps: dict) -> dict:
    out = {}
    for name, v in views.items():
        codes = tn.canonical_codes(v["board"])
        out[name] = dict(fingerprint=fps.get(name) or tn.fingerprint(v), rows=int(len(v["index"])), rows_over_10pct=int((v["expl"] > 10).sum()),
                         by_source={s: int((v["source"] == k).sum()) for k, s in enumerate(("recorded", "random_twin", "harvested"))},
                         **{p: dict(rows=int((parts[name] == s).sum()), boards=len(np.unique(codes[parts[name] == s])))
                            for p, s in (("train", tn.TRAIN), ("select", tn.SELECT), ("gate", tn.GATE))})
    return out


def cmd_data(a) -> int:
    runs, dirs = paths(a)
    RECORDS.mkdir(parents=True, exist_ok=True)
    manifest = RECORDS / "d1c-manifest.sha256"
    if D1C_MANIFEST is None:
        if manifest.exists():
            raise SystemExit(f"{manifest} exists and D1C_MANIFEST is not pinned: pin it by amendment, or move the file aside")
        why = d1c_incomplete(dirs[2])
        if why:
            raise SystemExit(f"{dirs[2]}: {why}: not pinned")
        text = d1c_manifest_text(dirs[2])
        manifest.write_bytes(text.encode())
        sha = hashlib.sha256(text.encode()).hexdigest()
        print(f"{manifest}: {len(text.splitlines())} files; its sha256 {sha} is D1C_MANIFEST (pin it by amendment, then run `d3 data` again)")
        return 0
    d, views, parts, fps = load_all(a, pinned=False)
    res = data_record(views, parts, fps)
    meta = json.loads((dirs[2] / "generate.json").read_text())
    res["d1c_layout"] = meta.get("layout")
    res["d1c_shards"] = len(list(dirs[2].glob("shard-*.npz")))
    _write(RECORDS / "data.json", json.dumps(res, indent=1))
    for name, r in res.items():
        if isinstance(r, dict) and "train" in r:
            print(f"{name}: fingerprint {r['fingerprint']}; rows {r['rows']}; train / select / gate {r['train']['rows']} / "
                  f"{r['select']['rows']} / {r['gate']['rows']}; sources {r['by_source']}; over 10% {r['rows_over_10pct']}")
    print(f"the union's fingerprint {fps['union']} is D3_UNION (pin it by amendment before N2)")
    return 0


def _write(path: Path, text: str) -> None:
    tn._write(path, text)


def fresh(path: Path, redo: bool) -> None:
    """A registered result is written once: the file there is moved aside (kept) only on `--redo`, else the run refuses."""
    if not path.exists():
        return
    if not redo:
        raise SystemExit(f"{path} exists: `--redo` moves it aside (kept) and runs again")
    print(f"moved aside: {dl.aside(path)}")


# ------------------------------------------------------------------ training


def net_differs(arm: dict, fp: str, seed: int, pools: int) -> str | None:
    """Why a finished net is not this arm's: `train` would retrain over it, and both results stay (PREREG 7)."""
    path = tn.NETS / arm["tag"] / "record.json"
    if not path.exists():
        return None
    rec, s = json.loads(path.read_text()), tn.LEAFNET_SETTINGS
    want = dict(width=arm["width"], depth=s["depth"], fraction=1.0, data=fp, epochs=s["epochs"], batch=s["batch"], lr=s["lr"],
                seed=seed, canonical=False, wd=float(arm["wd"]), pools=pools, dropout=float(arm.get("dropout", 0.0)),
                ema=float(arm.get("ema", 0.0)), loss=str(arm.get("loss", "huber")))
    differ = {k: (rec.get(k, tn.CFG_DEFAULTS.get(k)), v) for k, v in want.items() if rec.get(k, tn.CFG_DEFAULTS.get(k)) != v}
    return f"{arm['tag']}: its record is another arm's (record, arm): {differ}" if differ else None


def run_net(d: dict, part: np.ndarray, arm: dict, seed: int, fp: str, commit: str, ls, clients: set, held: dict,
            device: str = "cuda") -> tuple[dict, list]:
    """One net at D2's settings with D3's pools; GPU looks before it, every D2B_GPU_EVERY epochs and after its last
    (22 per net). Returns (record, the looks)."""
    tag, folder = arm["tag"], tn.NETS / arm["tag"]
    path, seen = folder / "gpu.json", []
    pools = pools_for(int((part == tn.TRAIN).sum()))
    why = tn.d2b_net_refused(tag) or net_differs(arm, fp, seed, pools)
    if why:
        raise SystemExit(f"{why}: this run is VOID (PREREG 7). `d3 train --void \"<cause>\" --tag {tag}` keeps it with its cause, then this repeats it")
    fresh = not path.exists()

    def look():
        g = ls.gpu_state()
        seen.append(g)
        new = sorted(set(g["apps"]) - clients)
        if new:
            print(f"  {tag}: GPU clients new at this look (a compute job voids the net, read them): {new}", flush=True)
            clients.update(new)
        if len(seen) == 1 and g["peers"]:
            raise SystemExit(f"{tag}: another process of this Python on the GPU ({g['peers']}): nothing trained; rerun when it has gone")
        if fresh or len(seen) > 1:
            folder.mkdir(parents=True, exist_ok=True)
            _write(path, json.dumps(seen, indent=1))
        if g["peers"]:
            raise SystemExit(f"{tag}: another process of this Python on the GPU ({g['peers']}): this run is VOID. "
                             f"`d3 train --void \"<cause>\" --tag {tag}`, then rerun")

    s, epochs = tn.LEAFNET_SETTINGS, []

    def watch(line):
        print(line, flush=True)
        if " epoch " in line:
            if len(epochs) % tn.D2B_GPU_EVERY == 0:
                look()
            epochs.append(line)

    look()
    rec = tn.train(d, part, arm["width"], s["depth"], 1.0, s["epochs"], s["batch"], s["lr"], seed, tag, log=watch, device=device,
                   reuse=True, held=held, wd=arm["wd"], pools=pools, commit=commit, dropout=arm.get("dropout", 0.0),
                   ema=arm.get("ema", 0.0), loss=arm.get("loss", "huber"))
    if epochs:
        look()
    return rec, json.loads(path.read_text())


def net_row(rec: dict, looks: list) -> dict:
    keys = ("tag", "width", "depth", "wd", "seed", "data", "train_samples", "val_samples", "pools", "commit", "best")
    return dict({k: rec.get(k) for k in keys}, **{k: rec.get(k, tn.CFG_DEFAULTS[k]) for k in tn.OPTION_KEYS},
                seconds=rec["history"][-1]["seconds"], upload_seconds=rec.get("upload_seconds"), train_loss_at_best=rec["history"][rec["best"]["epoch"] - 1]["train_loss"],
                gpu_peers=sorted({str(p) for g in looks for p in g["peers"] or []}), gpu_looks=len(looks),
                gpu_unknown=sum(tn.d2b_blind(g) for g in looks), gpu_apps=sorted({x for g in looks for x in g["apps"]}))


def voided() -> list[dict]:
    """The D3 nets moved aside as VOID so far: their cause, and how far each got."""
    return [json.loads(p.read_text()) for p in sorted(tn.NETS.glob("d3-*-void-*/void.json"))]


def void_tags() -> list[str]:
    return [N2_ARM["tag"], SEED1_TAG, *(f"d3-look{n}" for n in LOOKS)]


def void_net(tag: str, cause: str, ls) -> int:
    """PREREG 7: a net's folder moved aside with its cause in it, kept and listed in nets.json; the next run repeats that net."""
    if tag not in void_tags() or not cause.strip():
        raise SystemExit(f"--void needs a cause, and --tag one of {void_tags()}")
    folder = tn.NETS / tag
    if not folder.exists():
        raise SystemExit(f"{folder}: no net to void")
    done = RECORDS / ("selection.json" if not tag.startswith("d3-look") else f"look{tag[-1]}.json")
    if done.exists():
        raise SystemExit(f"{done} exists: voiding a net after its result needs an amendment first")
    g = ls.gpu_state()
    if ls.busy(g) or tn.d2b_blind(g):
        raise SystemExit("another process of this Python is on the GPU, or nvidia-smi could not say: stop it first, it may be this net's training")
    rec = json.loads((folder / "record.json").read_text()) if (folder / "record.json").exists() else {}
    n = 1 + len(list(tn.NETS.glob(f"{tag}-void-*")))
    while (tn.NETS / f"{tag}-void-{n}").exists():
        n += 1
    _write(folder / "void.json", json.dumps(dict(tag=tag, cause=cause, at=ls.utc(), folder=f"{tag}-void-{n}", commit=rec.get("commit"),
                                                 epochs=len(rec.get("history", [])), best=rec.get("best")), indent=1))
    folder.rename(tn.NETS / f"{tag}-void-{n}")
    print(f"VOID: {tag} moved aside to {tag}-void-{n}: {cause}")
    return 0


def need_commit() -> str:
    commit = tn._commit()
    if not tn.d2b_committed(commit):
        raise SystemExit(f"uncommitted changes under src or scripts, or git could not say ({commit!r}): commit first (every net's record holds the commit)")
    return commit


def import_ls():
    import livesearch as ls

    return ls


def cmd_memory(a) -> int:
    """Build N2's training pools and upload the first: the host's peak commit at ~1.9M rows (PREREG 8)."""
    import torch

    d, views, parts, fps = load_all(a)
    rows = np.flatnonzero(parts["union"] == tn.TRAIN)
    pools, s = pools_for(len(rows)), tn.LEAFNET_SETTINGS
    free0, t0 = free_commit_gib(), time.time()
    dev = torch.device("cuda")
    b = tn.Pools(d, rows, dev, pools, s["batch"])
    b.take(torch.randperm(b.n, device=dev), 0, s["batch"])  # the first pool goes up: every mapped page it reads is touched
    mem = dl.memory()
    res = dict(train_rows=int(len(rows)), pools=pools, commit=tn._commit(), seconds=round(time.time() - t0, 1), free_commit_before=free0,
               free_commit_after=free_commit_gib(), union=fps["union"], **{k: round(x, 2) for k, x in mem.items()})
    _write(RECORDS / "memory.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"host peak at {res['train_rows']} training rows: working set {res['peak_working_set']:.1f} GiB, commit "
          f"{res['peak_commit']:.1f} GiB; free commit before {free0:.1f}. Tell the author, then `d3 train --told {res['peak_commit']:.1f}`")
    return 0


def cmd_look(a) -> int:
    """A futility look (PREREG 5.2): the net's loss on the old selection half against N1's, one registered line."""
    import torch

    from parciver import value_net as vn

    ls, spec = import_ls(), LOOKS[a.n]
    tag = f"d3-look{a.n}"
    commit = need_commit()
    runs, dirs = paths(a)
    out = RECORDS / f"look{a.n}.json"
    if getattr(a, "redo", False):
        for old in (tn.NETS / tag, out):
            if dl.aside(old):
                print(f"moved aside: {old}")
    elif out.exists() or (tn.NETS / tag).exists():
        raise SystemExit(f"{out} or {tn.NETS / tag} exists: `look --redo` moves both aside (kept) and trains again")
    sel = json.loads((tn.D2B_RECORDS / "selection.json").read_text())
    n1_loss = sel["n1_loss"]
    if sel["n1"] != dl.N1_TAG:
        raise SystemExit(f"records/d2b/selection.json names {sel['n1']}, not {dl.N1_TAG}")
    d, views, parts, fps = load_all(a, caps=[None, None, spec["shards"]], pinned=False)
    n_c = int((d["dataset"] == 2).sum())
    if n_c != spec["shards"] * SHARD_ROWS:
        raise SystemExit(f"D1c has {n_c} rows in the first {spec['shards']} shards, not {spec['shards'] * SHARD_ROWS}: not all complete yet")
    part = parts["union"]
    arm = dict(N2_ARM, tag=tag)
    rec, looks = run_net(d, part, arm, tn.LEAFNET_SETTINGS["seed"], fps["union"], commit, ls, set(), {})
    row = net_row(rec, looks)
    refused = tn.d2b_nets_refused([row])
    if refused:
        raise SystemExit(f"{refused}: the look is not a verdict")
    # The old selection half: the original union's selection rows, which the look's own selection half contains.
    old = np.flatnonzero(parts["d12"] == tn.SELECT)
    if len(old) != OLD_SELECT_ROWS:
        raise SystemExit(f"the old selection half has {len(old)} rows, not {OLD_SELECT_ROWS}")
    b = tn.Batches(views["d12"], old, torch.device("cuda"))
    net, _ = vn.load(tn.NETS / tag / "net.pt")
    got = tn.evaluate(net, b)
    del net
    n1, _ = vn.load(tn.NETS / dl.N1_TAG / "net.pt")
    n1_now = tn.evaluate(n1, b)["loss"]
    verdict = look_verdict(got["loss"], n1_loss, spec["bar"])
    res = dict(verdict, look=a.n, shards=spec["shards"], data=fps["union"], train_rows=rec["train_samples"], selection_rows=rec["val_samples"],
               old_selection_rows=len(old), n1_rescored=n1_now, n1_rescored_matches=n1_now == n1_loss, mae=got["mae"], net=row, commit=commit,
               note="best epoch chosen on the combined selection half; scored on the old half; never chosen from, adopted or used in a gate")
    _write(out, json.dumps(res, indent=1))
    print(f"look {a.n} ({spec['shards']} shards, {rec['train_samples']} train rows): loss {got['loss']:.6f} vs N1 {n1_loss:.6f} = "
          f"{100 * verdict['relative']:+.2f}% (bar {100 * (spec['bar'] - 1):+.0f}%): "
          f"{'CONTINUE' if verdict['continue'] else 'D3 STOPS: N1 stays'}" + ("" if res["n1_rescored_matches"] else "  (N1 does not rescore to its record: no verdict)"))
    if not res["n1_rescored_matches"]:
        return 1
    return 0 if verdict["continue"] else 2


def cmd_train(a) -> int:
    ls = import_ls()
    if a.void is not None:
        return void_net(a.tag, a.void, ls)
    from parciver import value_net as vn  # noqa: F401  # loaded before the commit is read, not inside the first `train`

    commit = need_commit()
    d, views, parts, fps = load_all(a)
    part = parts["union"]
    mem = RECORDS / "memory.json"
    why = memory_refused(json.loads(mem.read_text()) if mem.exists() else None, int((part == tn.TRAIN).sum()), free_commit_gib(), a.told)
    if why:
        raise SystemExit(why)
    clients, held, recs, looks = set(), {}, [], {}

    def save():
        res = dict(nets=[net_row(r, looks[r["tag"]]) for r in recs], voided=voided(), union=fps["union"])
        _write(RECORDS / "nets.json", json.dumps(res, indent=1))
        return res

    for tag, seed in ((N2_ARM["tag"], 0), (SEED1_TAG, 1)):
        rec, looks[tag] = run_net(d, part, dict(N2_ARM, tag=tag), seed, fps["union"], commit, ls, clients, held)
        recs.append(rec)
        res = save()
        print(f"  {tag}: selection loss {rec['best']['val_loss']:.6f} (epoch {rec['best']['epoch']}), train loss at best "
              f"{res['nets'][-1]['train_loss_at_best']:.6f}")
    print(tn.d2b_clients_line(res) + ": read them before `d3 select`")
    return 0


def nets_refused(nets_json: dict, fp: str) -> str | None:
    """Why nets.json may not be selected from: the rows, the disk, the void list, the data."""
    rows = nets_json["nets"]
    if [r["tag"] for r in rows] != [N2_ARM["tag"], SEED1_TAG]:
        return f"nets.json holds {[r['tag'] for r in rows]}, not the registered {[N2_ARM['tag'], SEED1_TAG]}"
    if nets_json.get("union") != fp:
        return f"nets.json was trained on data {nets_json.get('union')}, not {fp}"
    if nets_json["voided"] != voided():
        return f"nets.json lists {len(nets_json['voided'])} voided nets, {len(voided())} are on disk: rerun `d3 train`"
    for r in rows:
        why = tn.d2b_net_refused(r["tag"])
        if why:
            return why
    return tn.d2b_nets_refused(rows) or tn.d2b_records_differ(rows)


def sha_of(tag: str) -> str:
    return tn.d2b_tag_sha(tag)


def cmd_select(a) -> int:
    """N2 against N1 on the combined selection half; the stop rule (PREREG 5.3). Every net on every dataset's half, by source."""
    import torch

    from parciver import value_net as vn

    runs, dirs = paths(a)
    nets_json = json.loads((RECORDS / "nets.json").read_text())
    fresh(RECORDS / "selection.json", a.redo)
    d, views, parts, fps = load_all(a)
    why = nets_refused(nets_json, fps["union"])
    if why:
        print(f"nets.json: {why}")
        return 1
    print(tn.d2b_clients_line(nets_json))
    if sha_of(dl.N1_TAG) != dl.N1_SHA:
        raise SystemExit(f"{dl.N1_TAG}'s net.pt is not N1 ({dl.N1_SHA})")
    tags = [dl.N1_TAG, N2_ARM["tag"], SEED1_TAG]
    nets = {t: vn.load(tn.NETS / t / "net.pt")[0] for t in tags}
    dev, scores = torch.device("cuda"), {t: {} for t in tags}
    for name, v in views.items():
        for label, keep in labels_of(v["source"]).items():
            rows = np.flatnonzero((parts[name] == tn.SELECT) & keep)
            if not len(rows):
                continue
            b = tn.Batches(v, rows, dev)
            for t in tags:
                e = tn.evaluate(nets[t], b)
                scores[t].setdefault(name, {})[label] = dict(rows=b.n, loss=e["loss"], mae=e["mae"])
            del b
            torch.cuda.empty_cache()
    recs = {t: json.loads((tn.NETS / t / "record.json").read_text()) for t in tags}
    match = {t: scores[t]["d12" if t == dl.N1_TAG else "union"]["all"]["loss"] == recs[t]["best"]["val_loss"] for t in tags}
    n1_loss, n2_loss = scores[dl.N1_TAG]["union"]["all"]["loss"], scores[N2_ARM["tag"]]["union"]["all"]["loss"]
    res = dict(n2=N2_ARM["tag"], n1=dl.N1_TAG, n1_loss=n1_loss, n2_loss=n2_loss, stop=not n2_loss < n1_loss, relative=n2_loss / n1_loss - 1,
               n1_loss_old_half=scores[dl.N1_TAG]["d12"]["all"]["loss"], sha256={N2_ARM["tag"]: sha_of(N2_ARM["tag"]), "N1": dl.N1_SHA},
               seed1_sha256=sha_of(SEED1_TAG), data=fps["union"], matches_record=match,
               train_loss_at_best={t: recs[t]["history"][recs[t]["best"]["epoch"] - 1]["train_loss"] for t in tags}, scores=scores)
    _write(RECORDS / "selection.json", json.dumps(res, indent=1))
    for t in tags:
        s = scores[t]
        print(f"  {t:<18s} combined {s['union']['all']['loss']:.6f}  old half {s['d12']['all']['loss']:.6f}  D1c {s['d1c']['all']['loss']:.6f}"
              f"  record {'ok' if match[t] else 'DIFFERS'}")
    print(f"N2 {n2_loss:.6f} vs N1 {n1_loss:.6f} on the combined selection half ({100 * res['relative']:+.2f}%)"
          + ("  -> D3 STOPS: N2 is not below N1" if res["stop"] else ""))
    if not all(match.values()):
        print("a saved net does not score what its record says: no gate until the cause is found (PREREG 7)")
        return 1
    return 1 if res["stop"] else 0


def cmd_gate_a(a) -> int:
    import torch

    from parciver import value_net as vn

    sel = json.loads((RECORDS / "selection.json").read_text())
    fresh(RECORDS / "gate-a.json", a.redo)
    if sel.get("stop", True) or not all(sel.get("matches_record", {"none": False}).values()):
        print(f"selection.json: no N2 to gate (stop {sel.get('stop')}, matches {sel.get('matches_record')})")
        return 1
    d, views, parts, fps = load_all(a)
    if sel["data"] != fps["union"]:
        raise SystemExit(f"selection.json is for data {sel['data']}, not {fps['union']}")
    tags = {"N2": sel["n2"], "N1": dl.N1_TAG}
    sha = {k: sha_of(t) for k, t in tags.items()}
    if sha["N2"] != sel["sha256"][sel["n2"]] or sha["N1"] != dl.N1_SHA:
        raise SystemExit("a net.pt changed after the selection")
    rows = np.flatnonzero(parts["union"] == tn.GATE)
    b = tn.Batches(d, rows, torch.device("cuda"))
    hb = tn.hb_members(d["board"][rows])
    trivial = np.array([len(h) == 1 for h in hb])
    mae, bare, defect = {}, {}, {}
    for k, tag in tags.items():
        net, _ = vn.load(tn.NETS / tag / "net.pt")
        mae[k] = tn.weighted_mae(tn.hb_average(net, b, hb), b)
        bare[k] = tn.evaluate(net, b)["mae_per_sample"].double().cpu().numpy() * 100
        defect[k] = tn.symmetry_defect(net, b)
        del net
        torch.cuda.empty_cache()
    bar = 0.5 * json.loads((tn.NETS / tn.LEAFNET_N0 / "g2.json").read_text())["baseline_mae"]
    codes, source, dataset = tn.canonical_codes(d["board"][rows]), d["source"][rows], d["dataset"][rows]
    masks = {"d1c_recorded": (dataset == 2) & (source != 2), "d1c_harvested": (dataset == 2) & (source == 2), "d1c": dataset == 2,
             "d1": dataset == 0, "d1b": dataset == 1, "union": np.ones(len(rows), bool)}
    sets = {}
    for name, m in masks.items():
        if not m.any():
            continue
        sets[name] = dict(samples=int(m.sum()), trivial_hb=int(trivial[m].sum()), averaged=summary(mae["N2"][m], mae["N1"][m], codes[m]),
                          bare=summary(bare["N2"][m], bare["N1"][m], codes[m]),
                          symmetry_defect={k: dict(mean=float(v[m].mean()), max=float(v[m].max())) for k, v in defect.items()})
    res = dict(gate_verdict(sets, bar), n2=sel["n2"], sha256=sha, data=fps["union"], sets=sets)
    out = RECORDS / "gate-a.json"
    _write(out, json.dumps(res, indent=1))
    for name, s in sets.items():
        av = s["averaged"]
        print(f"Gate A {name} ({s['samples']} samples): MAE {av['N2']:.3f} vs {av['N1']:.3f}% pot, D {av['mean']:+.4f} "
              f"[{av['ci'][0]:+.4f}, {av['ci'][1]:+.4f}] ({100 * av['relative']:+.1f}%); bare {s['bare']['N2']:.3f} vs {s['bare']['N1']:.3f}")
    print(f"  {'PROCEED' if res['proceed'] else 'STOP: N1 stays'}  (upper < 0 {res['upper_below_zero']}, <= -4% {res['gain_at_least_4pct']}, "
          f"worse than N1 on {res['regressed_on']}, G2 bar {bar:.3f} {res['under_g2_bar']})  ({out})")
    return 0 if res["proceed"] else 1


# ------------------------------------------------------------------ the scouts


def cmd_s0a(a) -> int:
    """S0a (PREREG 5.1): N1's loss on the 2,000 labelled queries / N1's union-real loss; the CI's lower end picks D1c's branch."""
    import torch

    from parciver import value_net as vn

    ls, runs = import_ls(), Path(a.data).parent
    tn.NETS = runs / "turn-net"
    fresh(RECORDS / "s0a.json", a.redo)
    folder, harvest = Path(a.s0a_data), Path(a.harvest) if a.harvest else None
    if harvest is None:
        raise SystemExit("--harvest: the S0a harvest (the recorder's queries/)")
    meta = json.loads((folder / "generate.json").read_text())
    if meta.get("layout") != dl.HARVEST_ONLY:
        raise SystemExit(f"{folder}: layout {meta.get('layout')!r}, S0a's labels are {dl.HARVEST_ONLY!r}")
    d = dl.load_mapped([folder], harvest=harvest)
    n = len(d["index"])
    if n != S0A_SPOTS * S0A_QUERIES or not (d["source"] == 2).all() or not np.array_equal(d["index"], np.arange(n)):
        raise SystemExit(f"{folder}: {n} rows; S0a needs {S0A_SPOTS * S0A_QUERIES}, all harvested, row k = harvest row k")
    spot = np.load(harvest / "spot.npy")[d["index"]]
    if len(np.unique(spot)) != S0A_SPOTS or np.bincount(np.unique(spot, return_inverse=True)[1]).tolist() != [S0A_QUERIES] * S0A_SPOTS:
        raise SystemExit(f"{harvest}: the first {n} rows are not {S0A_SPOTS} spots of {S0A_QUERIES} queries")
    hr = np.load(harvest / "ranges.npy", mmap_mode="r")
    for s in range(0, n, 500):  # the labels were made from these ranges, not others
        if not np.array_equal(d["ranges"][np.arange(s, min(s + 500, n))], np.asarray(hr[s:s + 500]).astype(d["ranges"].dtype)):
            raise SystemExit(f"{folder}: the shards' ranges are not the harvest's (rows {s}..): refused")
    sel = json.loads((tn.D2B_RECORDS / "selection.json").read_text())
    base = sel["scores"][dl.N1_TAG]["union"]["real"]["loss"]
    if sha_of(dl.N1_TAG) != dl.N1_SHA:
        raise SystemExit(f"{dl.N1_TAG}'s net.pt is not N1 ({dl.N1_SHA})")
    g0 = ls.gpu_state()
    if g0["peers"] or tn.d2b_blind(g0):
        raise SystemExit(f"another process of this Python on the GPU ({g0['peers']}), or nvidia-smi could not say: nothing scored")
    net, _ = vn.load(tn.NETS / dl.N1_TAG / "net.pt")
    b = tn.Batches(d, np.arange(n), torch.device("cuda"))
    loss, per = tn.evaluate(net, b)["loss"], sample_losses(net, b)
    g1 = ls.gpu_state()
    boot = tn.board_bootstrap(per, spot)
    lo, hi = boot["ci"][0] / base, boot["ci"][1] / base
    res = dict(r=loss / base, ci=[lo, hi], branch=s0a_branch(lo), bar=S0A_BAR, loss=loss, loss_mean_of_samples=float(per.mean()),
               base_loss=base, spots=S0A_SPOTS, queries=n, commit=tn._commit(), harvest=meta.get("harvest"), gpu=[g0, g1],
               gpu_peers_after=sorted(str(p) for p in g1["peers"] or []))
    _write(RECORDS / "s0a.json", json.dumps(res, indent=1))
    print(f"S0a: N1's loss on {n} queries {loss:.6f} = {res['r']:.2f} x {base:.6f}; CI of r [{lo:.2f}, {hi:.2f}] -> "
          f"{res['branch'].upper()} branch (the line {S0A_BAR})")
    return 1 if res["gpu_peers_after"] else 0


def cmd_s0c(a) -> int:
    """S0c (PREREG arm 3): the mean prediction of N1 and d2b-seed1 on the old selection half, forward only, no D1c row."""
    import torch

    from parciver import value_net as vn

    runs, dirs = paths(a)
    fresh(RECORDS / "s0c.json", a.redo)
    tn.d2b_check_files(dirs[1], tn.D2B_RECORDS / "d1b-manifest.sha256")
    views = tn.d2b_views(dl.load_mapped(dirs[:2]))
    if tn.fingerprint(views["union"]) != tn.D2B_UNION:
        raise SystemExit("D1 + D1b is not D2b's union")
    rows = np.flatnonzero(tn.split(views["union"]["board"]) == tn.SELECT)
    b = tn.Batches(views["union"], rows, torch.device("cuda"))
    sel = json.loads((tn.D2B_RECORDS / "selection.json").read_text())
    net1, _ = vn.load(tn.NETS / dl.N1_TAG / "net.pt")
    net2, _ = vn.load(tn.NETS / "d2b-seed1" / "net.pt")
    l1, l2 = tn.evaluate(net1, b)["loss"], tn.evaluate(net2, b)["loss"]
    total = 0.0
    net1.eval(), net2.eval()
    with torch.no_grad():
        for s in range(0, b.n, 4096):
            idx = torch.arange(s, min(s + 4096, b.n), device=b.dev)
            r, v, onehot, scal, live = b.get(idx, False)
            pred = (net1(r, onehot, scal, live).float() + net2(r, onehot, scal, live).float()) / 2
            total = total + tn._loss(pred, v) * len(idx)
    ens = float(total) / b.n
    res = dict(s0c_gain(ens, l1, l2), n1_loss=l1, seed1_loss=l2, ensemble_loss=ens, rows=b.n, n1_matches_selection=l1 == sel["n1_loss"],
               n1_sha256=sha_of(dl.N1_TAG), seed1_sha256=sha_of("d2b-seed1"), commit=tn._commit())
    _write(RECORDS / "s0c.json", json.dumps(res, indent=1))
    print(f"S0c: N1 {l1:.6f}, d2b-seed1 {l2:.6f}, mean prediction {ens:.6f}: {100 * res['gain_vs_n1']:+.1f}% vs N1 "
          f"({100 * res['gain_vs_best_single']:+.1f}% vs the better single); distillation "
          f"{'PROPOSED (>= 6%)' if res['distill_proposed'] else 'not proposed'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "runs" / "turn-data"), help="D1's folder; D1b's and D1c's are beside it")
    ap.add_argument("--harvest", default=None, help="the harvest a mixed D1c (or S0a's labels) was generated from")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    sub.add_parser("memory")
    lk = sub.add_parser("look")
    lk.add_argument("n", type=int, choices=sorted(LOOKS))
    lk.add_argument("--redo", action="store_true", help="move an earlier look (net folder and json) aside, then train again")
    t = sub.add_parser("train")
    t.add_argument("--told", type=float, default=None, help="the measured peak commit (GiB) as told to the author")
    t.add_argument("--void", metavar="CAUSE", default=None, help="this net's run is VOID, for this cause: move it aside")
    t.add_argument("--tag", default=None, help="--void: the net")
    for name in ("select", "gate-a", "s0c"):
        sub.add_parser(name).add_argument("--redo", action="store_true", help="move an earlier result file aside (kept), then run again")
    s = sub.add_parser("s0a")
    s.add_argument("--s0a-data", required=True, help="the folder `turn_data generate --harvest-only` labelled")
    s.add_argument("--redo", action="store_true", help="move an earlier s0a.json aside (kept), then run again")
    a = ap.parse_args()
    if (a.cmd == "train") and (a.void is not None) != (a.tag is not None):
        raise SystemExit("--void and --tag go together")
    return {"data": cmd_data, "memory": cmd_memory, "look": cmd_look, "train": cmd_train, "select": cmd_select, "gate-a": cmd_gate_a,
            "s0a": cmd_s0a, "s0c": cmd_s0c}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
