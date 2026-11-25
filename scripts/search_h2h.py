# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Head-to-head strength with real-time search on both sides, paired with the no-search receipt.

Does the candidate still beat the baseline when both search? Plays a scorecard's
strength row (the candidate against the benchmark's baseline blueprint, same
deals and seed as the receipts) with both play agents searching the river, or
the turn and river (--mode turn; subgames on the GPU with --gpu, one solver
shared by both sides), and pairs it deal by deal with the candidate's no-search
receipt (the scorecard's block statistics). Results are saved after every chunk;
rerunning with the same --out resumes (--threads and --chunk may change). A
chunk in which a search failed stops the run before it is saved: any decision a
failed search reached (counted by the engine) and, with --gpu, any failed GPU
solve or none at all (a failed GPU solve is not redone on the CPU: `rescue=False`
fails fast). Both sides search with the same settings.

    .venv/Scripts/python.exe scripts/search_h2h.py --benchmark runs/scorecard/scorecard-v2-200bb/benchmark.json \
        --receipt <candidate receipt.json> --mode river --deals 5000 --out runs/.../search-h2h-river.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import parciver_engine as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parciver import scorecard as sc  # noqa: E402
from parciver.solver_watch import Watched, write_atomic  # noqa: E402

BLOCKS = 50


def line(name: str, s: dict, suffix: str = "") -> str:
    return f"  {name:<40s} {s['mean']:+7.1f} bb/100 [{s['lo']:+.1f}, {s['hi']:+.1f}]{suffix}"


def search_share(sources: list[int]) -> float:
    """Fraction of decisions made by search ([blueprint, missing_key, off_tree, search])."""
    return sources[3] / max(1, sum(sources))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--receipt", required=True, help="the candidate's no-search receipt of this benchmark")
    p.add_argument("--row", default="baseline", help="strength row against the baseline agent")
    p.add_argument("--mode", choices=["river", "turn"], default="river")
    p.add_argument("--deals", type=int, default=0, help="first N of the row's deals (0 = all)")
    p.add_argument("--river-iters", type=int, default=200)
    p.add_argument("--turn-iters", type=int, default=100)
    p.add_argument("--gpu", action="store_true", help="solve search subgames on the GPU")
    p.add_argument("--terminals64", action="store_true",
                   help="with --gpu: FusedSolver's float64 terminal sums (KERNELS Q3, pinned; ~4x slower; VERIFY V5 arm F)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--chunk", type=int, default=250, help="deals per saved chunk")
    p.add_argument("--out", required=True, help="result JSON (resumed if it exists)")
    args = p.parse_args()
    if args.terminals64 and not args.gpu:
        raise SystemExit("--terminals64 needs --gpu")

    bench = Path(args.benchmark)
    m = json.loads(bench.read_text(encoding="utf-8"))
    st, hpd = m["stakes"], m["hands_per_deal"]
    row = next((r for r in m["rows"] if r["name"] == args.row), None)
    if row is None or row["kind"] != "agent" or row["family"] is not None:
        raise SystemExit(f"{args.row!r} is not a strength row against the baseline agent of {bench}")
    r = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if r["manifest_hash"] != m["manifest_hash"]:
        raise SystemExit(f"{args.receipt} is not a receipt of {bench}")
    if r["candidate"].get("search"):
        raise SystemExit(f"{args.receipt} was measured with search; pair with a no-search receipt")
    deals = args.deals or row["deals"]
    if deals > row["deals"] or deals % BLOCKS or deals % args.chunk:
        raise SystemExit(f"--deals must be at most {row['deals']} and a multiple of {BLOCKS} and --chunk")
    none = r["raw"]["rows"][args.row]["per_deal_bb"][:deals]

    out = Path(args.out)
    state = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    cfg = {"benchmark": m["benchmark_id"], "manifest_hash": m["manifest_hash"], "row": args.row, "deals": deals,
           "candidate": r["candidate"]["path"], "baseline": m["baseline"]["path"], "mode": args.mode,
           "river_iters": args.river_iters, "turn_iters": args.turn_iters, "gpu": args.gpu}
    if args.terminals64:  # only then: earlier outputs keep their configuration
        cfg["terminals64"] = True
    if state.get("config", cfg) != cfg:
        raise SystemExit(f"{out} holds a different configuration: {state['config']}")
    state["config"] = cfg
    run = state.setdefault("run", {"per_deal_bb": [], "a_sources": [0, 0, 0, 0], "b_sources": [0, 0, 0, 0],
                                   "seconds": 0.0})

    if len(run["per_deal_bb"]) < deals:
        engine = sc.sha256_file(pe.parciver_engine.__file__)
        if run["per_deal_bb"] and run.get("engine") != engine:
            raise SystemExit(f"{out} was played on another engine build (or before builds were recorded); new "
                             f"chunks would mix with its: use a new --out")
        run["engine"] = engine
        kw = dict(from_turn=args.mode == "turn", river_iters=args.river_iters, turn_iters=args.turn_iters)
        solver = None
        if args.gpu:
            from parciver.gpu_fused import SubgameGpu

            kw["solver"] = solver = Watched(SubgameGpu(terminals64=args.terminals64, q3=True))  # V5 arm F is Q3 (KERNELS Q4)
            kw["rescue"] = False  # a failed GPU solve stops the chunk; no CPU solve of a chunk then refused
        abs_dir = m["abstraction"]["dir"]
        cand = pe.SavedBlueprint(abs_dir, cfg["candidate"]).play_agent().with_search(**kw)
        base = pe.SavedBlueprint(abs_dir, cfg["baseline"]).play_agent().with_search(**kw)
        while len(run["per_deal_bb"]) < deals:
            first = len(run["per_deal_bb"])
            size = min(args.chunk, deals - first)
            t = time.time()
            if solver is not None:
                solver.take()
            res = dict(cand.match_against("agent", size, st["sb"], st["bb"], st["stack"], other=base,
                                          seed=row["seed"], threads=args.threads, first_deal=first))
            if solver is not None:
                calls, failures = solver.take()
                solver.check(calls, failures, f"deals {first}..{first + size - 1}")
                if not calls:
                    raise SystemExit(f"no search solve reached the GPU solver in deals {first}..{first + size - 1}; "
                                     f"nothing of this chunk was saved")
                run["solver_calls"] = run.get("solver_calls", 0) + calls
            # [built-in solver redid it, blueprint decided] per side (engines before 1 Oct do not count).
            failed = [sum(res[k]) for k in ("a_search_failed", "b_search_failed") if k in res]
            if any(failed):
                raise SystemExit(f"{sum(failed)} decisions in deals {first}..{first + size - 1} reached a failed search "
                                 f"(candidate, baseline: {res['a_search_failed']}, {res['b_search_failed']} "
                                 f"[built-in solver, blueprint]); nothing of this chunk was saved")
            run["per_deal_bb"] += list(res["per_deal_bb"])
            run["a_sources"] = [x + y for x, y in zip(run["a_sources"], res["a_sources"])]
            run["b_sources"] = [x + y for x, y in zip(run["b_sources"], res["b_sources"])]
            run["seconds"] += time.time() - t
            write_atomic(out, json.dumps(state))
            done = len(run["per_deal_bb"])
            eta = run["seconds"] / done * (deals - done)
            mean, se = sc.mean_se(run["per_deal_bb"], hpd)
            print(f"{args.row}/{args.mode}: {done}/{deals} deals, {mean:+.1f} ± {se:.1f} bb/100 so far, "
                  f"{run['seconds'] / done:.2f} s/deal, eta {eta / 60:.0f} min", flush=True)

    def blocks(per_deal):
        return sc.to_blocks(per_deal, BLOCKS, hpd)

    cur = blocks(run["per_deal_bb"][:deals])
    s0, s, d = sc.summarize(blocks(none)), sc.summarize(cur), sc.paired_change(cur, blocks(none))
    print(f"\n{m['benchmark_id']} row {args.row}: {r['candidate']['label']} vs baseline (higher is better), "
          f"first {deals} deals; search={args.mode}, river {args.river_iters} iters, "
          f"turn {args.turn_iters} iters{' (GPU)' if args.gpu else ''}")
    print(line("no search (receipt)", s0))
    print(line(f"both sides search={args.mode}", s,
               f"  (search decides {search_share(run['a_sources']):.0%} / {search_share(run['b_sources']):.0%})"))
    print(line("change vs no search", d, f"  {sc.status(d['lo'], d['hi'], higher_is_better=True)}"))
    state["summary"] = {"no_search": s0, "search": s, "vs_no_search": d}
    write_atomic(out, json.dumps(state))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
