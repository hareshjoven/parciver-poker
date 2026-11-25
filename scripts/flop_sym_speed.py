# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""FLOP-SYM's speed test (records/flop-sym/PREREG.md, "Speed"): every flop job
the candidate's M1 strategy run solved (`m1-jobs-<cand>.pkl`), each solved by
the dense leaves (A) and the candidate's (B), interleaved per job A B B A,
500 iterations, gamma 5, the same net, in one process with the GPU otherwise
idle. Both modes' tables for a flop are built, untimed, before its first job
(or again if evicted), and each mode runs untimed warm-up solves on the
first job of each flop class (|G| and C's rows), repeated on the class's next
job while C's solve fell back to dense; no job is dropped. Per job:
ratio = (tB1 + tB2) / (tA1 + tA2).

Pass: on the jobs whose flop has a stabilizer beyond the identity (those that
fell back to dense counted, their number reported), the median ratio is below
1 and its 95% percentile-bootstrap CI (10,000 resamples of flops with their
jobs, seed 0) is entirely below 1; on the unpaired rainbow jobs both return
bit-identical results.

  python scripts/flop_sym_speed.py --cand sym --net runs/turn-net/final-w2048-d5-e200/net.pt

LEAF-NET (records/leaf-net/PREREG.md, "Speed"): `--cand leafnet --net N0
--net-b P` times the same sym leaves with two nets, A = N0 (X_sym's net) and
B = P (X_leafnet's), on X_leafnet's jobs, with the same order and warm-ups.
Pass: over all jobs, the median ratio's 95% CI (flop-clustered, as above) has
its upper end <= 1.02 (records/leaf-net/speed.json).

D2b (records/d2b/PREREG.md, "Reported speed of the cheaper net"): `--cand d2b
--net N1 --net-b <d2b-w1024-wd* with the better selection loss>` times A = N1
and B = the cheaper net on X_sym's jobs, in the same way. No bar: reported only
(records/d2b/speed.json).

D3 (records/d3/PREREG.md, 5.0.6 and "Latency gate"): `--cand d3 --net N1 --net-b N2` times A = N1 and B = N2 on the same
479 jobs of `m1-jobs-sym.pkl` (40 flops), the same sym leaves, A B B A. Pass: the flop-clustered 95% CI's upper end of the
median ratio B/A is <= 1.02. Own outputs, never `records/d2b/`: `runs/flop-eval-d3/d3-speed.json` and
`records/d3/speed.json` (not overwritten: a void run's file is moved away first, its cause recorded)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # this checkout's code, as flop_eval.py
from parciver import flop_search as fs  # noqa: E402
from parciver import value_net as vn  # noqa: E402

ITERS, GAMMA = 500, 5.0
LEAFNET_BAR = 1.02  # LEAF-NET's speed bar: the CI's upper end
N0_SHA256 = "a8bf00be2fd1a3c7b0d1b0796809256cd939305b68723e541ef2cd029238ef65"
LEAF_RECORDS = Path(__file__).resolve().parents[1] / "records" / "leaf-net"
D2B_RECORDS = Path(__file__).resolve().parents[1] / "records" / "d2b"
D2B_OUT = Path(__file__).resolve().parents[1] / "runs" / "flop-eval-d2b"  # runs/flop-eval is untouched
D2B_CHEAPER = ("d2b-w1024-wd0", "d2b-w1024-wd0.1")
N1_SHA256 = "54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72"  # d2b-w2048-wd0.1, the live flop net
D3_RECORDS = Path(__file__).resolve().parents[1] / "records" / "d3"
D3_OUT = Path(__file__).resolve().parents[1] / "runs" / "flop-eval-d3"
D3_JOBS, D3_FLOPS = 479, 40  # records/d3/PREREG.md: the jobs file's size, as speed.json of D2b measured it
D3_JOBS_SHA256 = "7d9bd201407d672e49879006d63e9a57429c037b362d2dc68ab2f178fd0d4b3c"  # runs/flop-eval/m1-jobs-sym.pkl, pinned 9 Nov 2026


def flop_bootstrap(rows: list[dict]) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the median ratio, resampling flops with
    their jobs (10,000 resamples, default_rng(0))."""
    flops = sorted({tuple(r["flop"]) for r in rows})
    if not flops:
        return float("nan"), float("nan")
    by_flop = [np.array([r["ratio"] for r in rows if tuple(r["flop"]) == f]) for f in flops]
    rng = np.random.default_rng(0)
    boot = [np.median(np.concatenate([by_flop[j] for j in rng.integers(0, len(flops), len(flops))])) for _ in range(10_000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", choices=("sym", "orbit", "leafnet", "d2b", "d3"), required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--net-b", help="leafnet: P, timed against --net (N0) with the same sym leaves; d2b: the cheaper net; d3: N2 (--net is N1)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "flop-eval"))
    a = ap.parse_args()
    out = Path(a.out)
    if (a.cand in ("leafnet", "d2b", "d3")) != (a.net_b is not None):
        raise SystemExit("--net-b goes with --cand leafnet, d2b or d3, and only with them")
    if a.cand == "d2b":
        d2b_check_nets(a.net, a.net_b)
        return d2b_speed(pickle.loads((out / "m1-jobs-sym.pkl").read_bytes()), a.net, a.net_b)
    if a.cand == "d3":
        d3_check_nets(a.net, a.net_b)
        raw = (out / "m1-jobs-sym.pkl").read_bytes()
        return d3_speed(pickle.loads(raw), a.net, a.net_b, hashlib.sha256(raw).hexdigest())
    jobs = pickle.loads((out / f"m1-jobs-{a.cand}.pkl").read_bytes())
    net_sha = hashlib.sha256(Path(a.net).read_bytes()).hexdigest()
    made = json.loads((out / f"m1-x-{a.cand}.json").read_text())
    if a.cand == "leafnet":
        b_sha = hashlib.sha256(Path(a.net_b).read_bytes()).hexdigest()
        if net_sha != N0_SHA256 or made.get("net_sha256") != b_sha:
            raise SystemExit("--net must be N0 and --net-b the net X_leafnet's jobs were made with")
        return leafnet_speed(out, jobs, a.net, a.net_b, net_sha, b_sha)
    if made.get("net_sha256") != net_sha:
        raise SystemExit(f"--net is not the net X_{a.cand}'s jobs were made with")
    net, _ = vn.load(a.net)
    solvers = {"A": fs.FlopSearch(net, gamma=GAMMA, leaves="dense"), "B": fs.FlopSearch(net, gamma=GAMMA, leaves=a.cand)}
    rows, warm, t0 = [], set(), time.time()
    for i, job in enumerate(jobs):
        job = dict(job, iters=ITERS)
        flop = tuple(int(c) for c in job["board"])
        for s in solvers.values():
            if flop not in s._tables:  # the tables, untimed
                s.tables(flop)
        group = len(fs.flop_group(flop))
        cls = f"{group}/{len(fs.sym_plan(flop, cosets=a.cand == 'sym')[3]) if group > 1 else fs.TURNS}"
        if cls not in warm:  # untimed warm-up solves per mode and flop class (lazy CUDA setup, the class's rows)
            before = solvers["B"].symmetric
            for s in solvers.values():
                s(job)
            if group == 1 or solvers["B"].symmetric > before:  # a job that fell back to dense does not warm the class
                warm.add(cls)
        sec, res = {"A": [], "B": []}, {}
        before = solvers["B"].symmetric
        for who in "ABBA":
            s = solvers[who]
            res[who] = s(job)
            sec[who].append(s.seconds[-1])
        took = solvers["B"].symmetric - before == 2  # both B solves took the symmetric path
        rainbow = group == 1
        rows.append(dict(job=i, flop=list(flop), group=group, cls=cls, rainbow=rainbow, symmetric=took,
                         a=sec["A"], b=sec["B"], ratio=float(sum(sec["B"]) / sum(sec["A"])),
                         identical=res["A"] == res["B"]))
        if i % 50 == 0:
            print(f"  job {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    kept = rows
    nr = [r for r in kept if not r["rainbow"]]
    sym = np.array([r["ratio"] for r in nr])
    flops = sorted({tuple(r["flop"]) for r in nr})
    lo, hi = flop_bootstrap(nr)  # the same resampling as before it was a function
    rain = [r for r in kept if r["rainbow"]]
    report = dict(cand=a.cand, net=a.net, net_sha256=net_sha, jobs=len(rows), symmetric_jobs=sum(r["symmetric"] for r in kept),
                  non_rainbow=dict(n=len(sym), flops=len(flops), fell_back=sum(not r["symmetric"] for r in nr),
                                   median_ratio=float(np.median(sym)) if len(sym) else None, ci=[float(lo), float(hi)]),
                  rainbow=dict(n=len(rain), identical=all(r["identical"] for r in rain),
                               median_ratio=float(np.median([r["ratio"] for r in rain])) if rain else None))
    report["by_class"] = {}  # flop class: |G| / C's net rows per leaf
    for c in sorted({r["cls"] for r in kept}):
        x = [r for r in kept if r["cls"] == c]
        ta, tb = [np.mean(r["a"]) for r in x], [np.mean(r["b"]) for r in x]
        report["by_class"][c] = dict(n=len(x), median_ratio=float(np.median([r["ratio"] for r in x])),
                                     solve_A=dict(median=float(np.median(ta)), p95=float(np.percentile(ta, 95))),
                                     solve_B=dict(median=float(np.median(tb)), p95=float(np.percentile(tb, 95))))
    report["passed"] = bool(len(sym) and np.median(sym) < 1 and hi < 1 and report["rainbow"]["identical"])
    for name, sel in (("A", "a"), ("B", "b")):
        t = np.array([np.mean(r[sel]) for r in kept])
        report[f"solve_{name}"] = dict(median=float(np.median(t)), p95=float(np.percentile(t, 95)))
    (out / f"sym-speed-{a.cand}.json").write_text(json.dumps(dict(report, rows=rows), indent=1))
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 5


def leafnet_speed(out: Path, jobs: list, net_a: str, net_b: str, a_sha: str, b_sha: str) -> int:
    """LEAF-NET: the sym leaves with N0 (A) and with P (B), A B B A per job,
    the tables and warm-ups as above (a class: |G| and the sym rows)."""
    na, _ = vn.load(net_a)
    nb, _ = vn.load(net_b)
    if not nb.canonical:
        raise SystemExit("P is not canonical: the speed step is skipped for it (records/leaf-net/PREREG.md)")
    rows, solvers = two_net_rows(jobs, na, nb)
    ratios = np.array([r["ratio"] for r in rows])
    lo, hi = flop_bootstrap(rows)
    report = dict(two_net_summary(rows, solvers, ratios, lo, hi), net_a=net_a, net_a_sha256=a_sha, net_b=net_b, net_b_sha256=b_sha,
                  bar=LEAFNET_BAR, passed=bool(hi <= LEAFNET_BAR))
    (out / "leafnet-speed.json").write_text(json.dumps(dict(report, rows=rows), indent=1))
    LEAF_RECORDS.mkdir(parents=True, exist_ok=True)
    (LEAF_RECORDS / "speed.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 5


def d2b_check_nets(net_a: str, net_b: str) -> None:
    """--net must be selection.json's N1 and --net-b the D2B_CHEAPER net with the
    lower loss on the union's selection half, the file still the one scored."""
    sel = json.loads((D2B_RECORDS / "selection.json").read_text())
    cheaper = min(D2B_CHEAPER, key=lambda t: sel["scores"][t]["union"]["all"]["loss"])
    a_sha, b_sha = (hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (net_a, net_b))
    if a_sha != sel["sha256"][sel["n1"]] or Path(net_b).parent.name != cheaper:
        raise SystemExit(f"--net must be N1 ({sel['n1']}) and --net-b {cheaper}'s net")
    if Path(net_a).parent.name != sel["n1"] or b_sha != sel["cheaper_sha256"][cheaper]:
        raise SystemExit(f"--net is not {sel['n1']}'s file, or {cheaper}'s net.pt changed after the selection")


def d2b_speed(jobs: list, net_a: str, net_b: str) -> int:
    """D2b: the same sym leaves with N1 (A) and the cheaper net (B); reported only,
    so the non-canonical refusal above is not asked here."""
    na, _ = vn.load(net_a)
    nb, _ = vn.load(net_b)
    rows, solvers = two_net_rows(jobs, na, nb)
    ratios = np.array([r["ratio"] for r in rows])
    lo, hi = flop_bootstrap(rows)
    report = dict(two_net_summary(rows, solvers, ratios, lo, hi), net_a=net_a, net_b=net_b,
                  net_a_sha256=hashlib.sha256(Path(net_a).read_bytes()).hexdigest(),
                  net_b_sha256=hashlib.sha256(Path(net_b).read_bytes()).hexdigest())
    D2B_OUT.mkdir(parents=True, exist_ok=True)
    (D2B_OUT / "d2b-speed.json").write_text(json.dumps(dict(report, rows=rows), indent=1))
    (D2B_RECORDS / "speed.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


def d3_check_nets(net_a: str, net_b: str) -> None:
    """--net must be N1 (the live flop net, by its sha256) and --net-b the N2 that `records/d3/selection.json`
    names (`n2`, `sha256[n2]`), the file still the one selected."""
    sel = json.loads((D3_RECORDS / "selection.json").read_text())
    a_sha, b_sha = (hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (net_a, net_b))
    if a_sha != N1_SHA256:
        raise SystemExit(f"--net must be N1 (sha256 {N1_SHA256})")
    if Path(net_b).parent.name != sel["n2"] or b_sha == N1_SHA256:
        raise SystemExit(f"--net-b must be N2 ({sel['n2']}'s net.pt), not N1")
    if b_sha != sel["sha256"][sel["n2"]]:
        raise SystemExit(f"--net-b is not the file selected: {sel['n2']}'s net.pt changed after the selection")


def d3_speed(jobs: list, net_a: str, net_b: str, jobs_sha256: str | None = None) -> int:
    """D3's latency gate: the sym leaves with N1 (A) and N2 (B), A B B A per job. The jobs file must be the registered
    one (479 jobs over 40 flops). Pass iff the flop-clustered CI's upper end of the median ratio is <= LEAFNET_BAR;
    exit 0 on a pass, 5 on a fail. Never writes under records/d2b/; refuses to overwrite records/d3/speed.json."""
    flops = len({tuple(int(c) for c in j["board"]) for j in jobs})
    if (len(jobs), flops) != (D3_JOBS, D3_FLOPS):
        raise SystemExit(f"the jobs file has {len(jobs)} jobs over {flops} flops, not the registered {D3_JOBS} over {D3_FLOPS}")
    if jobs_sha256 != D3_JOBS_SHA256:
        raise SystemExit(f"the jobs file is not the registered runs/flop-eval/m1-jobs-sym.pkl (sha256 {jobs_sha256}, registered {D3_JOBS_SHA256})")
    dest = D3_RECORDS / "speed.json"
    for f in (dest, D3_OUT / "d3-speed.json"):
        if f.exists():
            raise SystemExit(f"{f} exists: move a void run's files away (its cause recorded) before timing again")
    na, _ = vn.load(net_a)
    nb, _ = vn.load(net_b)
    rows, solvers = two_net_rows(jobs, na, nb)
    ratios = np.array([r["ratio"] for r in rows])
    lo, hi = flop_bootstrap(rows)
    report = dict(two_net_summary(rows, solvers, ratios, lo, hi), net_a=net_a, net_b=net_b,
                  net_a_sha256=hashlib.sha256(Path(net_a).read_bytes()).hexdigest(),
                  net_b_sha256=hashlib.sha256(Path(net_b).read_bytes()).hexdigest(), jobs_sha256=jobs_sha256,
                  bar=LEAFNET_BAR, passed=bool(hi <= LEAFNET_BAR))
    D3_OUT.mkdir(parents=True, exist_ok=True)
    (D3_OUT / "d3-speed.json").write_text(json.dumps(dict(report, rows=rows), indent=1))
    D3_RECORDS.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 5


def two_net_summary(rows: list[dict], solvers: dict, ratios: np.ndarray, lo: float, hi: float) -> dict:
    report = dict(jobs=len(rows), symmetric={w: s.symmetric for w, s in solvers.items()},
                  solves={w: s.solves for w, s in solvers.items()}, flops=len({tuple(r["flop"]) for r in rows}),
                  median_ratio=float(np.median(ratios)), ci=[lo, hi])
    report["by_class"] = {c: dict(n=sum(r["cls"] == c for r in rows),
                                  median_ratio=float(np.median([r["ratio"] for r in rows if r["cls"] == c])))
                          for c in sorted({r["cls"] for r in rows})}
    for name, sel in (("A", "a"), ("B", "b")):
        t = np.array([np.mean(r[sel]) for r in rows])
        report[f"solve_{name}"] = dict(median=float(np.median(t)), p95=float(np.percentile(t, 95)))
    return report


def two_net_rows(jobs: list, na, nb) -> tuple[list[dict], dict]:
    """The sym leaves with two nets, A B B A per job, the tables and warm-ups as
    in `main` (a class: |G| and the sym rows)."""
    solvers = {"A": fs.FlopSearch(na, gamma=GAMMA, leaves="sym"), "B": fs.FlopSearch(nb, gamma=GAMMA, leaves="sym")}
    rows, warm, t0 = [], set(), time.time()
    for i, job in enumerate(jobs):
        job = dict(job, iters=ITERS)
        flop = tuple(int(c) for c in job["board"])
        for s in solvers.values():
            if flop not in s._tables:
                s.tables(flop)
        group = len(fs.flop_group(flop))
        cls = f"{group}/{len(fs.sym_plan(flop)[3]) if group > 1 else fs.TURNS}"
        if cls not in warm:
            before = solvers["B"].symmetric
            for s in solvers.values():
                s(job)
            if group == 1 or solvers["B"].symmetric > before:
                warm.add(cls)
        sec = {"A": [], "B": []}
        for who in "ABBA":
            s = solvers[who]
            s(job)
            sec[who].append(s.seconds[-1])
        rows.append(dict(job=i, flop=list(flop), group=group, cls=cls, a=sec["A"], b=sec["B"],
                         ratio=float(sum(sec["B"]) / sum(sec["A"]))))
        if i % 50 == 0:
            print(f"  job {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    return rows, solvers


if __name__ == "__main__":
    raise SystemExit(main())
