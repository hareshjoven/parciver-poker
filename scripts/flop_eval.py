"""Depth-limited flop search: settings and evaluation (FLOPSEARCH.md, D3 amendment and D4).

  record     play v2-2B against itself at 200bb (flop search on, a recording
             solver) and save the first job of each distinct flop-round root
             (`spots-<seed>.json`): the round from its first action, both
             starting ranges as the search sees them.
  calibrate  the flop_iters / gamma rule: solve each spot's round on the GPU
             with check-down leaves (float32, as the search runs) and record
             the exploitability at each checkpoint (`calibrate.json`).
  latency    seconds per flop decision of the searched agent (flop + turn
             search, value net leaves) in self-play (`latency-<tag>.json`).
  m2         D4 M2: v2-2B with flop + turn search against v2-2B with turn
             search, mirrored, on the first 10,000 strength deals of the 200bb
             scorecard (`m2.json`, saved after every chunk, resumed on rerun).

Paths default to runs/flop-eval. M2's live dashboard is scripts/watch_m2.py (it reads m2.json)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ABS = str(ROOT / "data" / "abstraction")
V2 = str(ROOT / "runs" / "blueprint-20260927-141645" / "blueprint.bin")
STAKES = (50, 100, 20000)  # the 200bb benchmark
CALIBRATION_SEED = 540000  # D3 amendment; M1's spots use 550000
CHECKPOINTS = (100, 200, 300, 500, 1000)
GAMMAS = (2.0, 5.0)
_KEYS = ("board", "kind", "player", "slot", "stake", "total", "child_start", "children", "ranges", "frozen", "want")


class FlopRecorder:
    """`with_search(from_flop=True, solver=...)`: keeps the first job of each
    distinct flop-round root; answers every job uniformly."""

    def __init__(self):
        self.jobs: list[dict] = []
        self._seen: set = set()

    def __call__(self, job: dict) -> list:
        from parciver import turn_values as tv

        if len(job["board"]) == 3:
            key = (tuple(job["board"]), pot_of(job), tuple(job["total"]), job["player"][0],
                   np.asarray(job["ranges"], dtype=np.float64).tobytes())
            if key not in self._seen:
                self._seen.add(key)
                player = np.asarray(job["player"])
                node = {s: i for i, s in enumerate(job["slot"]) if s >= 0}
                self.jobs.append({**{k: job[k] for k in _KEYS}, "hero": int(player[node[job["want"][0]]])})
        return tv._uniform_answer(job)


def pot_of(job: dict) -> float:
    """The pot at the flop round's start: the check-down's, the smallest
    pot where the round ends without a fold (decision nodes carry no stake)."""
    kind = np.asarray(job["kind"])
    stake = np.asarray(job["stake"], dtype=np.int64).reshape(-1, 2)
    return float(stake[(kind == 3) | (kind == 4)].sum(1).min())


def cmd_record(a) -> int:
    import parciver_engine as pe

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"spots-{a.seed}.json"
    if path.exists():
        print(f"{path} exists")
        return 1
    rec = FlopRecorder()
    agent = pe.SavedBlueprint(ABS, V2).play_agent().with_search(from_flop=True, solver=rec)
    sb, bb, stack = STAKES
    chunk = 0
    while len(rec.jobs) < a.spots:
        # One thread: the jobs arrive in play order, so the first `spots` are seeded.
        agent.match_against("agent", a.chunk, sb, bb, stack, seed=a.seed, threads=1, first_deal=chunk * a.chunk)
        chunk += 1
    jobs = rec.jobs[:a.spots]
    path.write_text(json.dumps(dict(seed=a.seed, stakes=STAKES, deals=chunk * a.chunk, spots=jobs)))
    pots = [pot_of(j) / bb for j in jobs]
    print(f"saved {len(jobs)} flop-round roots ({chunk * a.chunk} deals) to {path}; "
          f"pot median {np.median(pots):.1f} bb, range {min(pots):.0f}-{max(pots):.0f} bb")
    return 0


def cmd_calibrate(a) -> int:
    import torch

    from parciver import flop_search as fs

    spots = json.loads((Path(a.out) / f"spots-{a.seed}.json").read_text())["spots"]
    path = Path(a.out) / "calibrate.json"
    rows = []
    t0 = time.time()
    for i, job in enumerate(spots):
        root = dict(job, frozen=[])  # the whole round, as calibration
        board = job["board"]
        equity = fs.runout_equity(board)
        leaves = fs.NetLeaves(board, fs.CheckdownValues(board, dtype=torch.float32))
        pot = pot_of(job)
        for gamma in GAMMAS:
            s = fs.FlopSolver(root, leaves, equity, gamma=gamma)
            expl = {}
            for n in CHECKPOINTS:
                s.solve(n - s.iter)  # the search's graph; the average at n does not see later iterations
                expl[n] = 100.0 * s.exploitability() / pot
            rows.append(dict(spot=i, gamma=gamma, pot=pot, leaves=int(len(s.leaf)), slots=s.S, expl=expl))
        del leaves, equity
        torch.cuda.empty_cache()
        print(f"  spot {i + 1}/{len(spots)}  pot {pot / 100:.0f}bb  {len(s.leaf)} leaves  "
              + "  ".join(f"g{r['gamma']:.0f}@1000 {r['expl'][1000]:.3f}%" for r in rows[-2:])
              + f"  {time.time() - t0:.0f}s", flush=True)
    table = {f"{g:.0f}": {n: float(np.mean([r["expl"][n] for r in rows if r["gamma"] == g])) for n in CHECKPOINTS}
             for g in GAMMAS}
    # The rule (FLOPSEARCH.md, D3 amendment): the smallest checkpoint whose mean
    # exploitability, with the better gamma there, is under 0.5% of the pot.
    pick = None
    for n in CHECKPOINTS:
        g = min(GAMMAS, key=lambda g: table[f"{g:.0f}"][n])
        if table[f"{g:.0f}"][n] < 0.5:
            pick = dict(iters=n, gamma=g, passed=True)
            break
    if pick is None:
        n = max(CHECKPOINTS)
        pick = dict(iters=n, gamma=min(GAMMAS, key=lambda g: table[f"{g:.0f}"][n]), passed=False)
    path.write_text(json.dumps(dict(seed=a.seed, table=table, pick=pick, rows=rows), indent=1))
    print("mean exploitability, % of the pot:")
    print("  iters " + "".join(f"  gamma {g:.0f}" for g in GAMMAS))
    for n in CHECKPOINTS:
        print(f"  {n:5d} " + "".join(f"  {table[f'{g:.0f}'][n]:7.3f}" for g in GAMMAS))
    print(f"pick: {pick}")
    return 0


def cmd_latency(a) -> int:
    import parciver_engine as pe
    import torch

    from parciver import flop_search as fs
    from parciver import value_net as vn

    from parciver.solver_watch import Watched

    net, _ = vn.load(a.net)
    search = fs.FlopSearch(net, gamma=a.gamma, **({"leaves": a.leaves} if a.leaves else {}))  # FlopSearch's default if not given
    watched = Watched(search)  # each call timed whole: preparation, the wait for the GPU, the solve
    agent = pe.SavedBlueprint(ABS, V2).play_agent().with_search(
        from_flop=True, from_turn=True, flop_iters=a.iters, turn_iters=a.turn_iters, solver=watched,
        rescue=False)  # no CPU redo of a failed turn or river solve in the timed wall
    sb, bb, stack = STAKES
    torch.cuda.synchronize()
    t0 = time.time()
    agent.match_against("agent", a.deals, sb, bb, stack, seed=a.seed, threads=a.threads)
    wall = time.time() - t0
    sec = np.asarray(search.seconds)  # the solves alone, inside the GPU lock
    call = np.asarray(watched.seconds.get(3, []))  # the flop calls whole, as the engine waits for them
    got = dict(net=str(a.net), leaves=search.leaves, iters=a.iters, gamma=a.gamma, deals=a.deals, flop_solves=len(sec), wall=wall,
               median=float(np.median(sec)), p95=float(np.percentile(sec, 95)), mean=float(sec.mean()),
               call_median=float(np.median(call)), call_p95=float(np.percentile(call, 95)),
               failures=watched.failures,
               turn_solves=search.turn.solves if hasattr(search.turn, "solves") else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"latency-{a.tag}.json").write_text(json.dumps(dict(got, seconds=sec.tolist(), call_seconds=call.tolist()),
                                                          indent=1))
    print(json.dumps(got))
    if watched.failures:
        print(f"{watched.failures} search solves failed (the blueprint decided those); last: {watched.last_error}")
        return 1
    return 0


M2_BENCHMARK = ROOT / "runs" / "scorecard" / "scorecard-v2-200bb" / "benchmark.json"
M2_BLOCKS = 50  # the scorecard's


def cmd_m2(a) -> int:
    """FLOPSEARCH "M2 protocol": both agents share one solver, so their turn and
    river searches are the same; only A searches the flop. A chunk in which any
    search failed stops the run before it is saved (`rescue=False`)."""
    import parciver_engine as pe

    from parciver import flop_search as fs
    from parciver import scorecard as sc
    from parciver import value_net as vn
    from parciver.solver_watch import Watched, write_atomic

    m = json.loads(Path(a.benchmark).read_text(encoding="utf-8"))
    st, hpd = m["stakes"], m["hands_per_deal"]
    row = next(r for r in m["rows"] if r["name"] == "baseline")
    if (st["sb"], st["bb"], st["stack"]) != STAKES or Path(m["abstraction"]["dir"]) != Path(ABS):
        raise SystemExit(f"{a.benchmark} is not the 200bb benchmark on {ABS}")
    if a.deals > row["deals"] or a.deals % M2_BLOCKS or a.deals % a.chunk:
        raise SystemExit(f"--deals must be at most {row['deals']} and a multiple of {M2_BLOCKS} and --chunk")
    cfg = dict(benchmark=m["benchmark_id"], manifest_hash=m["manifest_hash"], row="baseline", seed=row["seed"],
               deals=a.deals, hands_per_deal=hpd, blueprint=V2, blueprint_sha256=sc.sha256_file(V2),
               net=str(Path(a.net).resolve()), net_sha256=sc.sha256_file(a.net), flop_iters=a.iters, gamma=a.gamma,
               turn_iters=a.turn_iters, river_iters=a.river_iters, turn_solver="SubgameGpu float32 (shared)")
    out = Path(a.out) / "m2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if state.get("config", cfg) != cfg:
        raise SystemExit(f"{out} holds a different configuration: {state['config']}")
    state["config"] = cfg
    run = state.setdefault("run", {"per_deal_bb": [], "a_sources": [0, 0, 0, 0], "b_sources": [0, 0, 0, 0],
                                   "seconds": 0.0, "calls": {}, "flop_seconds": [], "chunks": []})
    if len(run["per_deal_bb"]) < a.deals:
        engine = sc.sha256_file(pe.parciver_engine.__file__)
        if run["per_deal_bb"] and run.get("engine") != engine:
            raise SystemExit(f"{out} was played on another engine build; new chunks would mix with its: use a new --out")
        run["engine"] = engine
        net, _ = vn.load(a.net)
        search = fs.FlopSearch(net, gamma=a.gamma, cache=max(4, 2 * a.threads), leaves="dense")  # M2's leaves; the cache changes no value
        solver = Watched(search)
        kw = dict(from_turn=True, turn_iters=a.turn_iters, river_iters=a.river_iters, solver=solver, rescue=False)
        bp = pe.SavedBlueprint(ABS, V2)
        flop = bp.play_agent().with_search(from_flop=True, flop_iters=a.iters, **kw)
        turn = bp.play_agent().with_search(**kw)
        while len(run["per_deal_bb"]) < a.deals:
            first = len(run["per_deal_bb"])
            size = min(a.chunk, a.deals - first)
            t, n_flop = time.time(), len(search.seconds)
            solver.take()
            for v in solver.seconds.values():
                v.clear()
            res = dict(flop.match_against("agent", size, st["sb"], st["bb"], st["stack"], other=turn,
                                          seed=row["seed"], threads=a.threads, first_deal=first))
            calls, failures = solver.take()
            solver.check(calls, failures, f"deals {first}..{first + size - 1}")
            failed = [sum(res[k]) for k in ("a_search_failed", "b_search_failed")]
            if any(failed):
                raise SystemExit(f"{sum(failed)} decisions in deals {first}..{first + size - 1} reached a failed "
                                 f"search ({res['a_search_failed']}, {res['b_search_failed']}); nothing of this "
                                 f"chunk was saved")
            by_street = {str(k): len(v) for k, v in solver.seconds.items()}
            if not by_street.get("3"):
                raise SystemExit(f"no flop search reached the solver in deals {first}..{first + size - 1}; "
                                 f"nothing of this chunk was saved")
            dt = time.time() - t
            run["per_deal_bb"] += list(res["per_deal_bb"])
            run["a_sources"] = [x + y for x, y in zip(run["a_sources"], res["a_sources"])]
            run["b_sources"] = [x + y for x, y in zip(run["b_sources"], res["b_sources"])]
            run["seconds"] += dt
            for k, n in by_street.items():
                run["calls"][k] = run["calls"].get(k, 0) + n
            run["flop_seconds"] += search.seconds[n_flop:]
            run["chunks"].append(dict(first=first, deals=size, seconds=dt, calls=by_street, ended=time.time()))
            write_atomic(out, json.dumps(state))
            done = len(run["per_deal_bb"])
            mean, se = sc.mean_se(run["per_deal_bb"], hpd)
            print(f"m2: {done}/{a.deals} deals, {mean:+.1f} ± {se:.1f} bb/100 so far, {run['seconds'] / done:.2f} "
                  f"s/deal, eta {run['seconds'] / done * (a.deals - done) / 60:.0f} min", flush=True)

    s = sc.summarize(sc.to_blocks(run["per_deal_bb"][:a.deals], M2_BLOCKS, hpd))
    down = s["hi"] < 0
    f = np.asarray(run["flop_seconds"])
    share = lambda src: src[3] / max(1, sum(src))  # [blueprint, missing_key, off_tree, search]
    state["summary"] = dict(s, down=down, b_holds=not down, a_search_share=share(run["a_sources"]),
                            b_search_share=share(run["b_sources"]), s_per_deal=run["seconds"] / a.deals,
                            flop_solves_per_deal=run["calls"].get("3", 0) / a.deals,
                            flop_median=float(np.median(f)), flop_p95=float(np.percentile(f, 95)))
    write_atomic(out, json.dumps(state))
    print(f"M2: flop + turn search - turn search = {s['mean']:+.2f} bb/100, 95% [{s['lo']:+.2f}, {s['hi']:+.2f}] "
          f"over {a.deals} deals: {'DOWN, (b) FAILS' if down else 'not DOWN, (b) HOLDS'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "flop-eval"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    r.add_argument("--spots", type=int, default=40)
    r.add_argument("--chunk", type=int, default=50)
    c = sub.add_parser("calibrate")
    c.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    lat = sub.add_parser("latency")
    lat.add_argument("--net", required=True)
    lat.add_argument("--tag", required=True)
    lat.add_argument("--iters", type=int, required=True)
    lat.add_argument("--gamma", type=float, required=True)
    lat.add_argument("--turn-iters", type=int, default=100)
    lat.add_argument("--deals", type=int, default=50)
    lat.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    lat.add_argument("--threads", type=int, default=1)
    lat.add_argument("--leaves", choices=("dense", "factored"), help="FlopSearch's net leaves (its default if omitted)")
    m2 = sub.add_parser("m2")
    m2.add_argument("--net", required=True)
    m2.add_argument("--iters", type=int, required=True)
    m2.add_argument("--gamma", type=float, required=True)
    m2.add_argument("--turn-iters", type=int, default=100)
    m2.add_argument("--river-iters", type=int, default=200)
    m2.add_argument("--benchmark", default=str(M2_BENCHMARK))
    m2.add_argument("--deals", type=int, default=10000)
    m2.add_argument("--chunk", type=int, default=250)
    m2.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    return {"record": cmd_record, "calibrate": cmd_calibrate, "latency": cmd_latency, "m2": cmd_m2}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
