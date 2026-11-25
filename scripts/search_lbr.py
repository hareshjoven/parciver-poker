# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""LBR of blueprints with real-time search, paired with their no-search receipts.

Does search repair the v2 grid's extra exploitability? Runs a scorecard's LBR
(same deals and seed as its receipts) with the play agent searching the river,
or the turn and river (subgames on the GPU with --gpu), and pairs each run deal
by deal with the candidate's no-search receipt and with the other candidates at
the same search mode (the scorecard's block statistics). Results are saved after
every chunk; rerunning with the same --out resumes (a label's blueprint must be
the one it was run with). A chunk in which a search failed (at the agent's
decisions or in LBR's model of it; with --gpu the blueprint decides those:
`rescue=False`), or with --gpu none ran, stops the run before it is saved;
a run resumes only on the engine build it was played on.
Where the engine records it
(and a no-search receipt has it, or its `scorecard.py lbr-split` sidecar), also
reports POSTFLOP LBR: the total minus the hands whose betting ended preflop,
where LBR only calls (see src/parciver/scorecard.py).

    .venv/Scripts/python.exe scripts/search_lbr.py --benchmark runs/scorecard/scorecard-v2-200bb/benchmark.json \
        --candidate v1=runs/blueprint-20260914-144924/blueprint.bin=<v1 receipt.json> \
        --candidate v2=<blueprint.bin>=<receipt.json> --modes river,turn --gpu --out runs/.../search-lbr.json
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
CHUNK = 250
COUNTS = ("search_failed", "model_search_failed")  # [built-in solver, blueprint] each
STALE = "the engine predates search-failure counts (1 Oct; LBR's 2 Oct): rebuild it"


def line(name: str, s: dict, hpd: int, suffix: str = "") -> str:
    return f"  {name:<34s} {s['mean']:+7.1f} bb/100 [{s['lo']:+.1f}, {s['hi']:+.1f}]{suffix}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--candidate", action="append", required=True,
                   help="label=blueprint.bin=no-search receipt.json (a receipt of this benchmark)")
    p.add_argument("--modes", default="river,turn")
    p.add_argument("--deals", type=int, default=0, help="first N of the benchmark's LBR deals (0 = all)")
    p.add_argument("--river-iters", type=int, default=200)
    p.add_argument("--turn-iters", type=int, default=100)
    p.add_argument("--gpu", action="store_true", help="solve search subgames on the GPU")
    p.add_argument("--terminals64", action="store_true",
                   help="with --gpu: FusedSolver's float64 terminal sums (KERNELS Q3, pinned; ~4x slower; VERIFY V5 arm F)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", required=True, help="result JSON (resumed if it exists)")
    args = p.parse_args()
    if args.terminals64 and not args.gpu:
        raise SystemExit("--terminals64 needs --gpu")

    bench = Path(args.benchmark)
    m = json.loads(bench.read_text(encoding="utf-8"))
    st, lb, hpd = m["stakes"], m["lbr"], m["hands_per_deal"]
    deals = args.deals or lb["deals"]
    if deals % BLOCKS or deals % CHUNK:
        raise SystemExit(f"--deals must be a multiple of {BLOCKS} and {CHUNK}")
    out = Path(args.out)
    state = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"runs": {}}
    cfg = {"benchmark": m["benchmark_id"], "manifest_hash": m["manifest_hash"], "deals": deals,
           "river_iters": args.river_iters, "turn_iters": args.turn_iters, "gpu": args.gpu}
    if args.terminals64:  # only then: earlier outputs keep their configuration
        cfg["terminals64"] = True
    if state.get("config", cfg) != cfg:
        raise SystemExit(f"{out} holds a different configuration: {state['config']}")
    state["config"] = cfg
    solver = None
    if args.gpu:
        from parciver.gpu_fused import SubgameGpu

        solver = Watched(SubgameGpu(terminals64=args.terminals64, q3=True))  # V5 arm F is Q3 (KERNELS Q4)

    cands = []
    for c in args.candidate:
        label, path, receipt = c.split("=", 2)
        r = json.loads(Path(receipt).read_text(encoding="utf-8"))
        if r["manifest_hash"] != m["manifest_hash"]:
            raise SystemExit(f"{receipt} is not a receipt of {bench}")
        side = Path(receipt).with_name(Path(receipt).stem + ".lbr-split.json")
        split = sc.lbr_split_of(r, json.loads(side.read_text(encoding="utf-8")) if side.exists() else None)
        total = r["raw"]["lbr"]["per_deal_bb"][:deals]
        cands.append((label, path, total, None if split is None else [a - b for a, b in zip(total, split)]))

    modes = args.modes.split(",")
    bps, engine = {}, None
    for mode in modes:  # every candidate at the cheaper mode first
        for label, path, _, _ in cands:
            if label not in bps:
                bps[label] = pe.SavedBlueprint(m["abstraction"]["dir"], path)
            bp = bps[label]
            key = f"{label}/{mode}"
            run = state["runs"].setdefault(key, {"label": label, "blueprint": path, "mode": mode,
                                                  "per_deal_bb": [], "per_deal_preflop_bb": [], "seconds": 0.0})
            if run["blueprint"] != path:
                raise SystemExit(f"{out}: {key} was run with {run['blueprint']}, not {path}")
            if len(run["per_deal_bb"]) >= deals:
                continue
            if engine is None:  # before the first chunk: the build, and that it counts failed searches
                engine = sc.sha256_file(pe.parciver_engine.__file__)
                if not set(COUNTS) <= dict(pe.PlayAgent.uniform().lbr(2, 50, 100, 1_000, seed=0)).keys():
                    raise SystemExit(STALE)
            if run["per_deal_bb"] and run.get("engine") != engine:
                raise SystemExit(f"{out}: {key} was played on another engine build (or before failed searches "
                                 f"stopped a chunk); new chunks would mix with its: use a new --out")
            run["engine"] = engine
            kw = dict(from_turn=mode == "turn", river_iters=args.river_iters, turn_iters=args.turn_iters)
            if solver is not None:
                kw.update(solver=solver, rescue=False)  # a failed GPU solve stops the chunk: no CPU redo first
            agent = bp.play_agent().with_search(**kw)
            while len(run["per_deal_bb"]) < deals:
                first = len(run["per_deal_bb"])
                t = time.time()
                if solver is not None:
                    solver.take()
                res = dict(agent.lbr(CHUNK, st["sb"], st["bb"], st["stack"], seed=lb["seed"], threads=args.threads,
                                     first_deal=first))
                if solver is not None:
                    calls, failures = solver.take()
                    solver.check(calls, failures, f"{key} deals {first}..{first + CHUNK - 1}")
                    if not calls:
                        raise SystemExit(f"{key}: no search solve reached the GPU solver in deals {first}.."
                                         f"{first + CHUNK - 1}; nothing of this chunk was saved")
                # [built-in solver redid it, blueprint decided], at the agent's decisions and in LBR's model of it.
                if not set(COUNTS) <= res.keys():
                    raise SystemExit(STALE)
                if sum(res["search_failed"]) or sum(res["model_search_failed"]):
                    raise SystemExit(f"{key}: deals {first}..{first + CHUNK - 1} reached a failed search "
                                     f"(decisions {res['search_failed']}, LBR's model {res['model_search_failed']} "
                                     f"[built-in solver, blueprint]); nothing of this chunk was saved")
                if solver is not None:
                    run["solver_calls"] = run.get("solver_calls", 0) + calls
                run["per_deal_bb"] += list(res["per_deal_bb"])
                if "per_deal_preflop_bb" in run and "per_deal_preflop_bb" in res:
                    run["per_deal_preflop_bb"] += list(res["per_deal_preflop_bb"])
                else:  # a run resumed from before the split, or an older engine
                    run.pop("per_deal_preflop_bb", None)
                run["seconds"] += time.time() - t
                write_atomic(out, json.dumps(state))
                done = len(run["per_deal_bb"])
                eta = run["seconds"] / done * (deals - done)
                mean, se = sc.mean_se(run["per_deal_bb"], hpd)
                print(f"{key}: {done}/{deals} deals, LBR so far {mean:+.1f} ± {se:.1f}, "
                      f"{run['seconds'] / done:.2f} s/deal, eta {eta / 60:.0f} min", flush=True)

    def blocks(per_deal):
        return sc.to_blocks(per_deal, BLOCKS, hpd)

    def postflop(run):
        """A run's per-deal postflop LBR (first `deals`), or None without the split."""
        pre = run.get("per_deal_preflop_bb")
        if pre is None or len(pre) != len(run["per_deal_bb"]):
            return None
        return [a - b for a, b in zip(run["per_deal_bb"][:deals], pre[:deals])]

    legacy = ("made before runs recorded their engine build and failed searches were counted "
              "(replay with a new --out)")
    print(f"\n{m['benchmark_id']}: LBR (lower is better), first {deals} deals; river {args.river_iters} iters, "
          f"turn {args.turn_iters} iters{' (GPU)' if args.gpu else ''}")
    summary = {}
    for label, _, none, none_post in cands:
        n = len(none)  # the receipt holds the benchmark's deals only; --deals may run past them
        print(line(f"{label} no search (receipt, {n} deals)", sc.summarize(blocks(none)), hpd))
        if none_post is not None:
            print(line("  postflop", sc.summarize(blocks(none_post)), hpd))
        for mode in modes:
            run = state["runs"].get(f"{label}/{mode}")
            if not run or len(run["per_deal_bb"]) < deals:
                continue
            if run.get("engine") is None:
                print(f"{label} search={mode}: not summarized, {legacy}")
                continue
            per_deal = run["per_deal_bb"][:deals]
            s, d = sc.summarize(blocks(per_deal)), sc.paired_change(blocks(per_deal[:n]), blocks(none))
            summary[f"{label}/{mode}"] = {"lbr": s, "vs_no_search": d}
            print(line(f"{label} search={mode}", s, hpd))
            print(line(f"  change vs no search (first {n} deals)", d, hpd,
                       f"  {sc.status(d['lo'], d['hi'], higher_is_better=False)}"))
            post = postflop(run)
            if post is not None:
                s = sc.summarize(blocks(post))
                summary[f"{label}/{mode}"]["lbr_postflop"] = s
                print(line("  postflop", s, hpd))
                if none_post is not None:
                    d = sc.paired_change(blocks(post[:n]), blocks(none_post))
                    summary[f"{label}/{mode}"]["postflop_vs_no_search"] = d
                    print(line("  postflop change vs no search", d, hpd,
                               f"  {sc.status(d['lo'], d['hi'], higher_is_better=False)}"))
    if len(cands) == 2:
        (a, _, an, an_post), (b, _, bn, bn_post) = cands
        for mode in ["none"] + modes:
            if mode == "none":
                xa, xb, pa, pb = an, bn, an_post, bn_post
            else:
                ra, rb = state["runs"].get(f"{a}/{mode}"), state["runs"].get(f"{b}/{mode}")
                if not ra or not rb or len(ra["per_deal_bb"]) < deals or len(rb["per_deal_bb"]) < deals:
                    continue
                if ra.get("engine") is None or rb.get("engine") is None:
                    print(f"{b} - {a}, search={mode}: not compared, {legacy}")
                    continue
                if ra.get("engine") != rb.get("engine"):  # a finished run is not replayed on a new build
                    print(f"{b} - {a}, search={mode}: not compared, the two runs were played on different "
                          f"engine builds (replay one with a new --out)")
                    continue
                xa, xb, pa, pb = ra["per_deal_bb"][:deals], rb["per_deal_bb"][:deals], postflop(ra), postflop(rb)
            d = sc.paired_change(blocks(xb), blocks(xa))
            summary[f"{b}-{a}/{mode}"] = d
            print(line(f"{b} - {a}, search={mode}", d, hpd, f"  {sc.status(d['lo'], d['hi'], higher_is_better=False)}"))
            if pa is not None and pb is not None:
                d = sc.paired_change(blocks(pb), blocks(pa))
                summary[f"{b}-{a}/{mode}/postflop"] = d
                print(line("  postflop", d, hpd, f"  {sc.status(d['lo'], d['hi'], higher_is_better=False)}"))
    state["summary"] = summary
    write_atomic(out, json.dumps(state))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
