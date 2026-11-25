# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D1b cloud probe (HANDOFF "Cloud GPU for D1b"): is another GPU fit to make D1b's data?

  extract  (on the PC) D1's first PAIRS real situations (index 0..PAIRS-1, shard 0) with their stored
           labels -> probe-jobs.npz, a few MB, for the cloud.
  parity   (on the cloud) re-solve those jobs (both sources) in D1's settings (500 iterations, gamma 5,
           float32, plain kernels; `--terminals64` for Q4) and compare with the stored labels.
  speed    (on the cloud) aggregate solves/s of D1's generate loop with `--procs` processes in parallel.

The parity rule is fixed here, before any cloud data (19 Oct 2026). Units: % of the pot.
  G1 inputs: each source-1 random range, regenerated from its seed, equals the stored one bit for bit
     (float32), and source 0's equals situations.npz's; so both solves see D1's exact inputs.
  G2 every value, game value and exploitability is finite.
  G3 every job: |game_c - game_s| <= e_c + e_s + 1e-4. A theorem for any two strategy profiles of the
     same game: |u0(s) - v*| <= max(g0, g1) <= e(s), e = both best responses' gains. A port that
     solves a different game (a wrong terminal, payoff or tree) breaks it wherever the error exceeds e.
  G4 on ordinary jobs (stored e <= 1.5%; above it float32 solves are chaotic, records/d1-tail
     PREREG8): the median of e_c / e_s lies in [0.8, 1.25], and no job has e_c > 2 e_s + 0.05.
  PASS iff G1-G4 all hold (in D1's mode; the Q4 run must pass G2-G3 and e_c <= e_s + 0.05 on
  ordinary jobs). Descriptive only: d = the own-range-weighted mean |v_c - v_s| per player (max of the
  two), by band; bit-identity of a repeated solve; the tail jobs.
Kernel correctness itself is the GPU test suite's job (oracle tests against the CPU solver)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

D1_SEED, ITERS, GAMMA = 520007, 500, 5
BASE = ("board", "half", "stack", "first", "bb")


def cmd_extract(a) -> int:
    d1 = Path(a.d1)
    meta = json.loads((d1 / "generate.json").read_text())
    assert (meta["iters"], meta["gamma"], meta["seed"], meta["dtype"]) == (ITERS, GAMMA, D1_SEED, "float32"), meta
    sit = np.load(d1 / "situations.npz")
    sh = np.load(d1 / "shard-00000.npz")
    keep = np.flatnonzero(sh["index"] < a.pairs)
    assert len(keep) == 2 * a.pairs and np.array_equal(sh["index"][keep], np.repeat(np.arange(a.pairs), 2))
    out = {k: sit[k][: a.pairs] for k in BASE}
    out["real_ranges"] = sit["ranges"][: a.pairs]
    for k in ("index", "source", "values", "game", "expl", "ranges"):
        out["s_" + k] = sh[k][keep]
    np.savez(a.out, **out)
    print(f"{a.out}: {a.pairs} situations, {len(keep)} jobs")
    return 0


def _row(data, j: int) -> dict:
    return {k: data[k][j] for k in (*BASE, "real_ranges")}


def _job(row: dict, j: int):
    """Both sources' Situations and plans for situation j (`row`: its columns), built as turn_data's
    `_prepare`."""
    from parciver import turn_values as tv
    from parciver.gpu_fused import host_plan
    from parciver.gpu_search import Hands, Tree

    base = dict(board=tuple(int(c) for c in row["board"]), half=int(row["half"]),
                stack=int(row["stack"]), first=int(row["first"]), bb=int(row["bb"]))
    hands = Hands(list(base["board"]))
    order = tv.strength_order(hands)
    rng = np.random.default_rng([D1_SEED, j])
    sits = [tv.Situation(ranges=row["real_ranges"].astype(np.float64), **base),
            tv.Situation(ranges=np.stack([tv.random_range(order, rng), tv.random_range(order, rng)]), **base)]
    tree = Tree(sits[0].spec(), hands)
    return sits, [host_plan(hands, tree, list(s.ranges)) for s in sits]


def _solve(sit, plan, t64: bool) -> dict:
    """turn_values.solve with D1's settings, plus the terminals64 switch."""
    import torch

    from parciver import turn_values as tv
    from parciver.gpu_fused import FusedSolver

    s = FusedSolver(None, None, list(sit.ranges), dtype=torch.float32, plan=plan, gamma=GAMMA, plus=False,
                    terminals64=t64)
    for _ in range(ITERS):
        s.step()
    values, game = tv.values_of(s, sit)
    return dict(values=values, game=game, exploitability=s.exploitability() / sit.pot)


def cmd_parity(a) -> int:
    import torch

    data = dict(np.load(a.jobs))
    n = len(data["half"]) if a.pairs is None else a.pairs
    rows, g1_bad, t0 = [], [], time.time()
    first = {}
    for j in range(n):
        sits, plans = _job(_row(data, j), j)
        for src in (0, 1):
            k = 2 * j + src
            assert data["s_index"][k] == j and data["s_source"][k] == src
            if not np.array_equal(sits[src].ranges.astype(np.float32), data["s_ranges"][k]):
                g1_bad.append(k)
            r = _solve(sits[src], plans[src], a.terminals64)
            if j < a.repeat:
                first[k] = r
            v_c, v_s = r["values"], data["s_values"][k].astype(np.float64)
            rr = sits[src].ranges
            d = max(float(np.sum(rr[p] * np.abs(v_c[p] - v_s[p])) / np.sum(rr[p])) for p in (0, 1))
            rows.append(dict(job=k, index=j, source=src, e_s=100 * float(data["s_expl"][k]),
                             e_c=100 * r["exploitability"], dgame=100 * abs(r["game"] - float(data["s_game"][k])),
                             d=100 * d, finite=bool(np.all(np.isfinite(v_c)) and np.isfinite(r["game"])
                                                     and np.isfinite(r["exploitability"]))))
        if (j + 1) % 10 == 0:
            print(f"{j + 1}/{n} situations  {2 * (j + 1) / (time.time() - t0):.2f} solves/s", flush=True)
    rep = []
    for k, r in first.items():  # a repeated solve: bit-identical on this GPU?
        j, src = divmod(k, 2)
        sits, plans = _job(_row(data, j), j)
        r2 = _solve(sits[src], plans[src], a.terminals64)
        rep.append(bool(np.array_equal(r["values"], r2["values"]) and r["game"] == r2["game"]))
    e_s = np.array([x["e_s"] for x in rows]); e_c = np.array([x["e_c"] for x in rows])
    ordn = e_s <= 1.5
    g2 = all(x["finite"] for x in rows)
    g3_bad = [x["job"] for x in rows if not x["dgame"] <= x["e_c"] + x["e_s"] + 1e-4]
    ratio = float(np.median(e_c[ordn] / e_s[ordn])) if ordn.any() else float("nan")
    if a.terminals64:
        g4_bad = [x["job"] for x, o in zip(rows, ordn) if o and not x["e_c"] <= x["e_s"] + 0.05]
        g4 = not g4_bad
    else:
        g4_bad = [x["job"] for x, o in zip(rows, ordn) if o and not x["e_c"] <= 2 * x["e_s"] + 0.05]
        g4 = 0.8 <= ratio <= 1.25 and not g4_bad
    g1 = not g1_bad
    verdict = (g1 and g2 and not g3_bad and g4) if not a.terminals64 else (g2 and not g3_bad and g4)
    d = np.array([x["d"] for x in rows])
    summ = dict(mode="Q4 terminals64" if a.terminals64 else "D1 float32", jobs=len(rows), ordinary=int(ordn.sum()),
                G1=g1, G1_bad=g1_bad, G2=g2, G3=not g3_bad, G3_bad=g3_bad, G4=g4, G4_bad=g4_bad,
                median_ratio_ordinary=ratio, PASS=bool(verdict),
                desc=dict(d_median_ordinary=float(np.median(d[ordn])) if ordn.any() else None,
                          d_p95_ordinary=float(np.percentile(d[ordn], 95)) if ordn.any() else None,
                          d_max=float(d.max()), dgame_max=max(x["dgame"] for x in rows),
                          e_c_mean=float(e_c.mean()), e_s_mean=float(e_s.mean()),
                          tail_jobs=int((~ordn).sum()), repeat_bit_identical=f"{sum(rep)}/{len(rep)}"),
                torch=torch.__version__, device=torch.cuda.get_device_name(0), seconds=round(time.time() - t0, 1))
    Path(a.out).write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print(json.dumps(summ, indent=1))
    return 0 if verdict else 2


def _worker(path: str, start: int, t64: bool, seconds: float, q, barrier) -> None:
    """D1's generate loop: the next situation's host work in a 1-process pool while this one solves.

    Every process solves once (Triton compile), then all wait at `barrier` and count
    from the same moment, so the summed rate is over one common window.
    """
    import concurrent.futures as cf

    import torch  # noqa: F401  (the GPU runtime in this process)

    data = dict(np.load(path))
    n = len(data["half"])
    done, t_first, t0 = 0, None, time.time()
    with cf.ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        j = start % n
        fut = pool.submit(_job, _row(data, j), j)
        while True:
            sits, plans = fut.result()
            j = (j + 1) % n
            fut = pool.submit(_job, _row(data, j), j)
            for sit, plan in zip(sits, plans):
                _solve(sit, plan, t64)
                if t_first is None:  # count from the common start after every first solve
                    barrier.wait(timeout=900)  # a process that died first breaks it for all
                    t_first = time.time()
                else:
                    done += 1
            if t_first is not None and time.time() - t_first >= seconds:
                break
    q.put((done, time.time() - t_first, time.time() - t0))


def cmd_speed(a) -> int:
    ctx = mp.get_context("spawn")
    q, barrier = ctx.Queue(), ctx.Barrier(a.procs)
    ps = [ctx.Process(target=_worker, args=(a.jobs, 37 * i, a.terminals64, a.seconds, q, barrier))
          for i in range(a.procs)]
    for p in ps:
        p.start()
    try:
        res = [q.get(timeout=a.seconds + 1200) for _ in ps]
    except Exception:  # queue.Empty: a process died; never hang the run
        for p in ps:
            p.terminate()
        print(json.dumps(dict(procs=a.procs, error="a process died", exitcodes=[p.exitcode for p in ps])),
              flush=True)
        return 2
    for p in ps:
        p.join()
    rate = sum(dn / el for dn, el, _ in res)
    out = dict(procs=a.procs, mode="Q4" if a.terminals64 else "D1 float32", solves_per_s=round(rate, 3),
               vs_pc_1_17=round(rate / 1.17, 2), per_proc=[round(dn / el, 3) for dn, el, _ in res])
    print(json.dumps(out), flush=True)
    with open(a.log, "a") as f:
        f.write(json.dumps(dict(out, at=time.strftime("%Y-%m-%d %H:%M:%S %Z"))) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--d1", default=str(ROOT / "runs" / "turn-data"))
    e.add_argument("--pairs", type=int, default=150)
    e.add_argument("--out", default="probe-jobs.npz")
    p = sub.add_parser("parity")
    p.add_argument("--jobs", default="probe-jobs.npz")
    p.add_argument("--pairs", type=int, default=None)
    p.add_argument("--repeat", type=int, default=10, help="re-solve the first REPEAT situations' jobs (bit identity)")
    p.add_argument("--terminals64", action="store_true")
    p.add_argument("--out", default="parity.json")
    s = sub.add_parser("speed")
    s.add_argument("--jobs", default="probe-jobs.npz")
    s.add_argument("--procs", type=int, default=1)
    s.add_argument("--seconds", type=float, default=120)
    s.add_argument("--terminals64", action="store_true")
    s.add_argument("--log", default="speed.jsonl")
    a = ap.parse_args()
    return {"extract": cmd_extract, "parity": cmd_parity, "speed": cmd_speed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
