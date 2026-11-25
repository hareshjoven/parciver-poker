# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D1b / D1c acceptance (records/d1b/PREREG.md "Acceptance" and Amendment 2; records/d3/PREREG.md 5.2 and 5.0.5), on the PC.

  check   A1-A3 on the CPU, over every shard; also reported for the cloud part (shards 0-371) and the
          PC part (372-500) of D1b. `--partial` checks whatever complete shards exist (a dry run, or D1c's
          futility looks: its verdict is "PARTIAL", never of record).
  verify  A4 on the GPU: 200 jobs uniform over the rows (D1b: the 372,000 rows of the cloud shards; D1c: all its rows;
          numpy seed 2026, drawn without replacement), each re-solved exactly as `turn_data.py generate` solves it (its
          `_prepare_one`, `turn_values.solve`, Q4 terminals); G2 and G3 on every job (d1b_probe.py's
          definitions) and the median of e_pc / e_cloud in [0.8, 1.25]. Reported beside it, not
          gating: Amendment 1's mean rule (mean e_cloud <= mean e_pc + 0.02% of the pot) and the
          ordinary-job (e_cloud <= 1.5%) median.

`--profile d1b` (default) is D1b's registered settings; `--profile d1c` is D1c's (1,000,008 rows, seed 700007, all PC). D1c is
either pairs (a recorded situation, then its random twin: the D1b layout) or, in the > 1.5 branch, the mixed layout of
`turn_data.mixed_layout` (sources 0/1 recorded and its twin, 2 harvested; `generate.json` says which, and `--harvest` names the
harvest `d3_flop.py merge` wrote). Rows over 10% of the pot are counted and reported, per source too.

Units: values in pots per unit of opponent range mass; exploitability as a fraction of the pot in the
shards, in % of the pot in the reports. Writes JSON (default records/d1b/accept-*.json or records/d3/accept-*.json);
exit 0 iff PASS."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DIR = ROOT / "runs" / "turn-data-d1b"
D1 = ROOT / "runs" / "turn-data"
CLOUD = 372  # shards 0..371 were made on the cloud (Amendment 2)
SETTINGS = dict(iters=500, gamma=5, plus=False, samples=500004, shard=1000, seed=600007, dtype="float32",
                terminals="q4", workers=4, code="7839aa6")
# D1c (records/d3/PREREG.md 5.2): all on the PC, so no cloud part, and one GPU job at a time; the code is whatever generated it
SETTINGS_D1C = dict(iters=500, gamma=5, plus=False, samples=1000008, shard=1000, seed=700007, dtype="float32",
                    terminals="q4")
PROFILES = {"d1b": dict(settings=SETTINGS, cloud=CLOUD, dir=DIR, out=ROOT / "records" / "d1b"),
            "d1c": dict(settings=SETTINGS_D1C, cloud=0, dir=ROOT / "runs" / "turn-data-d1c", out=ROOT / "records" / "d3",
                        harvest_rows=499968)}  # 5.0.5: 10,204 spots, 49 queries each, the last cut to 21
MIXED = "interleave-1:1"
ZS_TOL = 1e-5  # A2, pots per solve
A3_MEAN = 0.005  # A3, fraction of the pot
A4_JOBS, A4_SEED = 200, 2026
KEYS = {"index", "source", "values", "game", "expl", "ranges"}


def _layout(meta: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(source, index, shards): the global row order `turn_data.generate` writes for this run (shard k = rows
    [k * shard, (k + 1) * shard)). Pairs: each recorded situation then its random twin; mixed: `mixed_layout`."""
    import turn_data as td

    if meta.get("layout") == MIXED:
        n_h = int(meta["harvest_rows"])
        src, idx = td.mixed_layout((meta["samples"] - n_h) // 2, n_h)
    else:
        pairs = meta["samples"] // 2
        src, idx = np.tile(np.array([0, 1], dtype=np.int8), pairs), np.repeat(np.arange(pairs), 2)
    return src, idx, -(-len(src) // meta["shard"])


def _part_summary(expl: np.ndarray, zs: np.ndarray, finite: bool) -> dict:
    return dict(rows=int(len(expl)), finite=finite,
                zero_sum_max=float(zs.max()) if len(zs) else None,
                zero_sum_over_tol=int((~(zs <= ZS_TOL)).sum()),
                expl_mean_pct=100 * float(expl.mean()) if len(expl) else None,
                expl_median_pct=100 * float(np.median(expl)) if len(expl) else None,
                expl_p99_pct=100 * float(np.percentile(expl, 99)) if len(expl) else None,
                share_over_3pct=float((expl > 0.03).mean()) if len(expl) else None,
                share_over_10pct=float((expl > 0.10).mean()) if len(expl) else None)


def _settings_clash(meta: dict, settings: dict, harvest_rows: int | None = None) -> dict:
    bad = {k: (meta.get(k), v) for k, v in settings.items() if meta.get(k) != v}
    if meta.get("layout") not in (None, MIXED):
        bad["layout"] = (meta.get("layout"), MIXED)
    if meta.get("layout") == MIXED:
        n_h = meta.get("harvest_rows")
        if not isinstance(n_h, int) or not 0 < n_h <= meta["samples"] - n_h:
            bad["harvest_rows"] = (n_h, "0 < harvested <= recorded rows")
        elif harvest_rows is not None and n_h != harvest_rows:
            bad["harvest_rows"] = (n_h, harvest_rows)
    return bad


def _harvest_of(a, meta: dict) -> dict | None:
    """The harvest columns when the run is mixed (memory-mapped), else None; refused unless it is the one generated from."""
    import turn_data as td

    if meta.get("layout") != MIXED:
        return None
    if not a.harvest:
        raise SystemExit("the run is mixed (harvested rows): pass --harvest, the queries/ directory d3_flop.py merge wrote")
    got = td.sha256_tree(a.harvest)
    if got != meta.get("harvest"):
        raise SystemExit(f"--harvest {a.harvest} is not the harvest this run was generated from "
                         f"({got[:12]} against {str(meta.get('harvest'))[:12]})")
    h = td.load_harvest(a.harvest)
    if len(h["half"]) != meta["harvest_rows"]:
        raise SystemExit(f"the harvest has {len(h['half'])} rows, generate.json says {meta['harvest_rows']}")
    return h


def cmd_check(a) -> int:
    prof = PROFILES[a.profile]
    d = Path(a.dir or prof["dir"])
    meta = json.loads((d / "generate.json").read_text())
    bad_meta = _settings_clash(meta, prof["settings"], prof.get("harvest_rows"))
    src_all, idx_all, shards = _layout(meta)
    per = meta["shard"]
    cloud = prof["cloud"]
    sit_ranges = np.load(d / "situations.npz")["ranges"]
    harvest = _harvest_of(a, meta)
    a1_bad, a2_nonfinite, missing = [], [], []
    expl = {"cloud": [], "pc": []}
    zs = {"cloud": [], "pc": []}
    by_src: dict[int, list] = {}
    finite = {"cloud": True, "pc": True}
    t0 = time.time()
    for k in range(shards):
        path = d / f"shard-{k:05d}.npz"
        if not path.exists():
            missing.append(k)
            continue
        try:
            s = dict(np.load(path))
        except Exception as e:  # a corrupt or half-written file
            a1_bad.append(dict(shard=k, problems=[f"unreadable: {type(e).__name__}: {e}"]))
            continue
        part = "cloud" if k < cloud else "pc"
        idx, src = s["index"], s["source"]
        want_idx, want_src = idx_all[k * per:(k + 1) * per], src_all[k * per:(k + 1) * per]
        n = len(want_idx)
        problems = []
        if set(s) != KEYS:
            problems.append(f"fields {sorted(s)}")
        if not np.array_equal(idx, want_idx):
            problems.append("index")
        if not np.array_equal(src, want_src):
            problems.append("source")
        v, r, g, e = s["values"], s["ranges"], s["game"], s["expl"]
        if v.shape != (n, 2, 1128) or r.shape != (n, 2, 1128) or g.shape != (n,) or e.shape != (n,):
            problems.append(f"shapes {v.shape} {r.shape} {g.shape} {e.shape}")
        if v.dtype != np.float32 or r.dtype != np.float32:
            problems.append(f"dtypes {v.dtype} {r.dtype}")
        if not problems:  # source 0 is the recorded situation: its ranges are situations.npz's, bit for bit
            s0 = src == 0
            if not np.array_equal(r[s0], sit_ranges[idx[s0]]):
                problems.append("source-0 ranges differ from situations.npz")
            s2 = src == 2  # a harvested row: its ranges are the harvest's
            if s2.any() and (harvest is None or not np.array_equal(r[s2], np.asarray(harvest["ranges"])[idx[s2]])):
                problems.append("source-2 ranges differ from the harvest")
        if problems:
            a1_bad.append(dict(shard=k, problems=problems))
            continue
        ok = bool(np.all(np.isfinite(v)) and np.all(np.isfinite(g)) and np.all(np.isfinite(e)))
        if not ok:
            a2_nonfinite.append(k)
            finite[part] = False
        r64 = r.astype(np.float64)
        rn = r64 / r64.sum(axis=2, keepdims=True)
        z = np.abs(np.einsum("ijk,ijk->i", rn, v.astype(np.float64)))
        zs[part].append(z)
        expl[part].append(e)
        for x in np.unique(src):
            by_src.setdefault(int(x), []).append(e[src == x])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{shards} shards  {time.time() - t0:.0f} s", flush=True)
    parts = {}
    for p in ("cloud", "pc"):
        ex = np.concatenate(expl[p]) if expl[p] else np.zeros(0)
        z = np.concatenate(zs[p]) if zs[p] else np.zeros(0)
        parts[p] = _part_summary(ex, z, finite[p])
    ex_all = np.concatenate(expl["cloud"] + expl["pc"]) if expl["cloud"] + expl["pc"] else np.zeros(0)
    z_all = np.concatenate(zs["cloud"] + zs["pc"]) if zs["cloud"] + zs["pc"] else np.zeros(0)
    whole = _part_summary(ex_all, z_all, finite["cloud"] and finite["pc"])
    sources = {str(x): dict(rows=int(sum(len(c) for c in v)), expl_mean_pct=100 * float(np.concatenate(v).mean()),
                            share_over_10pct=float((np.concatenate(v) > 0.10).mean()))
               for x, v in sorted(by_src.items())}
    d1 = None
    if a.profile == "d1b" and D1.exists():  # D1 as stored in its shards (its tail patch, records/d1-tail, lives elsewhere)
        d1e = np.concatenate([np.load(p)["expl"] for p in sorted(D1.glob("shard-*.npz"))])
        d1 = dict(rows=int(len(d1e)), expl_mean_pct=100 * float(d1e.mean()),
                  share_over_3pct=float((d1e > 0.03).mean()), share_over_10pct=float((d1e > 0.10).mean()))
    a1 = not bad_meta and not a1_bad and not missing
    a2 = not a2_nonfinite and whole["zero_sum_over_tol"] == 0
    a3 = whole["expl_mean_pct"] is not None and whole["expl_mean_pct"] <= 100 * A3_MEAN
    if a.partial:
        a1 = not bad_meta and not a1_bad
        verdict = "PARTIAL"
    else:
        verdict = "PASS" if (a1 and a2 and a3) else "FAIL"
    out = dict(check="A1-A3", profile=a.profile, verdict=verdict, A1=a1, A2=a2, A3=a3, generate_json=meta,
               settings_clash=bad_meta, shards_expected=shards, shards_missing=len(missing), missing_first=missing[:20],
               A1_bad=a1_bad, A2_nonfinite_shards=a2_nonfinite, zero_sum_tol=ZS_TOL, whole=whole, parts=parts,
               sources=sources, d1=d1, seconds=round(time.time() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_path = Path(a.out or prof["out"] / "accept-a1a3.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verdict", "A1", "A2", "A3", "shards_missing", "whole", "parts", "sources", "d1")},
                     indent=1))
    return 0 if verdict == "PASS" else (0 if verdict == "PARTIAL" else 2)


def _draw(population: int) -> np.ndarray:
    rng = np.random.default_rng(A4_SEED)
    return np.sort(rng.choice(population, size=A4_JOBS, replace=False))


def cmd_verify(a) -> int:
    import torch

    from parciver import turn_values as tv
    from turn_data import _prepare_one, _row

    prof = PROFILES[a.profile]
    settings = prof["settings"]
    d = Path(a.dir or prof["dir"])
    meta = json.loads((d / "generate.json").read_text())
    assert all(meta.get(k) == v for k, v in settings.items()), meta
    src_all, idx_all, _ = _layout(meta)
    harvest = _harvest_of(a, meta)
    situ = dict(np.load(d / "situations.npz"))
    rows, t0 = [], time.time()
    picks = _draw(prof["cloud"] * settings["shard"] if prof["cloud"] else len(src_all))
    if a.jobs is not None:
        picks = picks[: a.jobs]
    cache = {}
    for n, row in enumerate(picks):
        k, i = divmod(int(row), settings["shard"])
        if k not in cache:
            cache = {k: dict(np.load(d / f"shard-{k:05d}.npz"))}
        s = cache[k]
        j, src = int(s["index"][i]), int(s["source"][i])
        assert (j, src) == (int(idx_all[row]), int(src_all[row])), (row, j, src)
        data = harvest if src == 2 else situ
        sits, plans = _prepare_one(_row(data, j), j, settings["seed"], 0 if src == 2 else src)
        sit = sits[0]
        g1 = bool(np.array_equal(sit.ranges.astype(np.float32), s["ranges"][i]))  # the same inputs
        r = tv.solve(sit, None, settings["iters"], dtype=torch.float32, plan=plans[0], gamma=settings["gamma"],
                     plus=False, terminals64=True)
        e_pc, e_c = 100 * float(r["exploitability"]), 100 * float(s["expl"][i])
        dgame = 100 * abs(float(r["game"]) - float(s["game"][i]))
        v_pc = r["values"].astype(np.float64)
        rr = sit.ranges
        dv = max(float(np.sum(rr[p] * np.abs(v_pc[p] - s["values"][i][p])) / np.sum(rr[p])) for p in (0, 1))
        rows.append(dict(row=int(row), shard=k, index=j, source=src, inputs_equal=g1, e_cloud=e_c, e_pc=e_pc,
                         dgame=dgame, d=100 * dv,
                         finite=bool(np.all(np.isfinite(r["values"])) and np.isfinite(r["game"])
                                     and np.isfinite(r["exploitability"]))))
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(picks)} jobs  {(n + 1) / (time.time() - t0):.2f} solves/s", flush=True)
    e_c = np.array([x["e_cloud"] for x in rows]); e_pc = np.array([x["e_pc"] for x in rows])
    g2 = all(x["finite"] for x in rows)
    g3_bad = [x["row"] for x in rows if not x["dgame"] <= x["e_pc"] + x["e_cloud"] + 1e-4]
    inputs_bad = [x["row"] for x in rows if not x["inputs_equal"]]
    ratio = float(np.median(e_pc / e_c))
    ordn = e_c <= 1.5
    ok = g2 and not g3_bad and not inputs_bad and 0.8 <= ratio <= 1.25
    full = a.jobs is None
    out = dict(check="A4", profile=a.profile, verdict=("PASS" if ok else "FAIL") if full else "PARTIAL", jobs=len(rows),
               G2=g2, G3=not g3_bad, G3_bad=g3_bad, inputs_bad=inputs_bad, median_e_pc_over_e_cloud=ratio,
               sources={str(x): sum(1 for r_ in rows if r_["source"] == x) for x in sorted({r_["source"] for r_ in rows})},
               reported=dict(amendment1_mean_rule=bool(e_c.mean() <= e_pc.mean() + 0.02),
                             e_cloud_mean=float(e_c.mean()), e_pc_mean=float(e_pc.mean()),
                             ordinary=int(ordn.sum()),
                             median_ratio_ordinary=float(np.median(e_pc[ordn] / e_c[ordn])) if ordn.any() else None,
                             d_median=float(np.median([x["d"] for x in rows])),
                             d_p95=float(np.percentile([x["d"] for x in rows], 95)),
                             dgame_max=max(x["dgame"] for x in rows)),
               torch=torch.__version__, device=torch.cuda.get_device_name(0), seconds=round(time.time() - t0, 1),
               when=time.strftime("%Y-%m-%d %H:%M:%S"), rows=rows)
    out_path = Path(a.out or prof["out"] / "accept-a4.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0 if out["verdict"] in ("PASS", "PARTIAL") else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "verify"):
        c = sub.add_parser(name)
        c.add_argument("--profile", choices=sorted(PROFILES), default="d1b")
        c.add_argument("--dir", default=None, help="the run directory (default: the profile's)")
        c.add_argument("--harvest", default=None, help="a mixed run's harvest: the queries/ directory (or .npz)")
        c.add_argument("--out", default=None)
        if name == "check":
            c.add_argument("--partial", action="store_true", help="a dry run over the complete shards that exist")
        else:
            c.add_argument("--jobs", type=int, default=None, help="only the first JOBS of the 200 (a dry run)")
    a = ap.parse_args()
    return {"check": cmd_check, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
