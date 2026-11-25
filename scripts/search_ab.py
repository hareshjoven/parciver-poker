# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""One blueprint under two search settings, paired (FLOPSEARCH M2/M3, LIVESEARCH P2/P3).

The same blueprint in both arms, each with its own search settings and its own
solver, on a scorecard benchmark's deals (the first --deals; mirrored, as every
match), measured one of three ways:
- a strength row (--row baseline): arm A against arm B. A's bb/100 against B
  with the scorecard's block CI (what A wins B loses).
- a resized-bets or realistic-sizes row (--row baseline_sizes_x1.5,
  baseline_realistic_sizes): each arm against the row's bot (the benchmark's
  baseline blueprint, resizing its bets) on the row's deals. Each arm's bb/100
  and A minus B, paired by deal (higher is better for A).
- --lbr: each arm's LBR on the benchmark's LBR deals and seed. Each arm's
  bb/100 and A minus B, paired by deal (lower is better for A), with the
  postflop part (LBR's winnings from hands whose betting went past preflop).
Results are saved after every chunk; rerunning with the same --out resumes
(--threads and --chunk may change; nothing that decides a deal may: the arms and
deals, the benchmark manifest, the contents of the blueprint, the abstraction,
the nets and the engine build, this script's and the loaded solver modules'
sources, and with a GPU arm torch, CUDA, the GPU, triton and numpy). Latency is measured apart,
on an idle GPU (scripts/flop_eval.py latency): here two arms share it. A chunk
stops the run before it is saved on any decision a failed search reached (the
engine's counts; with --lbr also any query of LBR's model of an arm that read
one; GPU arms run with rescue=False) and, for a GPU arm, on any failed solve or
none at all.

An arm is a JSON object: `with_search` settings (from_turn, river_iters,
turn_iters, exact_own_range, solve_round, from_flop, flop_iters,
flop_checkdown; the engine's defaults where left out), "solver": null (the
built-in solver), "gpu" (SubgameGpu: turn and river) or "flop" (FlopSearch:
the flop with "net", or check-down leaves without; turn and river on
SubgameGpu), "solver_kw" (the solver's constructor keywords; {"terminals64": true} is Q4 since
it landed, add "q3": true for Q3) and "translation"
(null: the play agent's default).

    # M2: flop + turn search against turn search (FLOPSEARCH: D4's flop search is
    # flop_iters 500 and FlopSearch(net, gamma=5))
    .venv/Scripts/python.exe scripts/search_ab.py --benchmark runs/scorecard/scorecard-v2-200bb/benchmark.json \
        --blueprint runs/blueprint-20260927-141645/blueprint.bin --deals 10000 \
        --a '{"from_flop": true, "from_turn": true, "flop_iters": 500, "solver": "flop", "net": "<net.pt>", "solver_kw": {"gamma": 5}}' \
        --b '{"from_turn": true, "solver": "gpu"}' --out runs/flop-eval/m2.json
    # M3: the same arms against the x1.5 bot (--deals 5000 --row baseline_sizes_x1.5), the
    # realistic-sizes bot (--row baseline_realistic_sizes), and their LBR (--lbr)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import parciver_engine as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parciver import scorecard as sc  # noqa: E402
from parciver.solver_watch import Watched, write_atomic  # noqa: E402

# `PlayAgent.with_search`'s settings and defaults (a test checks them against the engine's).
SEARCH = {"from_turn": False, "river_iters": 200, "turn_iters": 100, "exact_own_range": True, "solve_round": True,
          "from_flop": False, "flop_iters": 100, "flop_checkdown": False}
ARM = dict(SEARCH, solver=None, solver_kw={}, net=None, translation=None)
# The modules each solver is built from (their sources, and what they import, key a run).
SOLVER_MODULES = {None: (), "gpu": ("parciver.gpu_fused",), "flop": ("parciver.flop_search", "parciver.value_net")}
# Loaded, but deciding no deal: it only summarizes (a fix to it must not block a resume).
SUMMARY_ONLY = {"parciver.scorecard"}


def arm(text: str) -> dict:
    """An arm's settings with every default filled in (equal arms compare equal)."""
    try:
        got = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"an arm is a JSON object: {e}") from None
    if not isinstance(got, dict) or set(got) - set(ARM):
        raise SystemExit(f"an arm is a JSON object with keys from {sorted(ARM)}, got {text}")
    a = dict(ARM, **got)
    for k, default in SEARCH.items():  # JSON true is not an iteration count, nor 1 a flag
        if type(default) is bool and type(a[k]) is not bool:
            raise SystemExit(f"{k} is true or false, got {a[k]!r}")
        if type(default) is int and (type(a[k]) is not int or a[k] < 1):
            raise SystemExit(f"{k} is a positive integer, got {a[k]!r}")
    if a["solver"] not in (None, "gpu", "flop"):
        raise SystemExit(f"solver must be null, \"gpu\" or \"flop\", got {a['solver']!r}")
    if a["from_flop"] and not a["flop_checkdown"] and a["solver"] != "flop":
        raise SystemExit("from_flop needs the \"flop\" solver (SubgameGpu solves no flop) or flop_checkdown")
    if a["net"] is not None and (a["solver"] != "flop" or not a["from_flop"] or a["flop_checkdown"]):
        raise SystemExit("a net is for the flop solver's flop search (from_flop, not flop_checkdown)")
    if not isinstance(a["solver_kw"], dict) or (a["solver_kw"] and a["solver"] is None):
        raise SystemExit("solver_kw is an object of the GPU solver's constructor keywords")
    if a["translation"] is not None and not isinstance(a["translation"], str):
        raise SystemExit(f"translation is null or a name, got {a['translation']!r}")
    return a


def engine_digest() -> str:
    """The loaded engine extension's sha256: a run resumes on the same build only."""
    return sc.sha256_file(pe.parciver_engine.__file__)


def sources(arms) -> dict:
    """sha256 of this script and of every module under src/ loaded once the
    arms' solver modules are imported (their code and what it imports), but
    SUMMARY_ONLY's."""
    for a in arms:
        for name in SOLVER_MODULES[a["solver"]]:
            importlib.import_module(name)
    src = (ROOT / "src").resolve()
    pkg = Path(sys.modules["parciver"].__file__).resolve()
    if not pkg.is_relative_to(src):
        raise SystemExit(f"parciver loads from {pkg.parent}, not {src}: its sources would key nothing")
    files = {Path(__file__).resolve()}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and name not in SUMMARY_ONLY and Path(f).resolve().is_relative_to(src):
            files.add(Path(f).resolve())
    return {f.relative_to(ROOT.resolve()).as_posix(): sc.sha256_file(f) for f in sorted(files)}


def gpu_env(arms) -> dict | None:
    """torch, its CUDA, the GPU, triton (the kernels' compiler, a package torch
    does not pin) and numpy, where an arm solves on the GPU."""
    if not any(a["solver"] for a in arms):
        return None
    import numpy
    import torch
    import triton

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu, "triton": triton.__version__,
            "numpy": numpy.__version__}


def identity(m: dict, args, deals: int, mode: str = "h2h") -> dict:
    """What decides the run's deals; a resume needs all of it unchanged."""
    sc.check_manifest(m)
    d = Path(m["abstraction"]["dir"])
    if {name: sc.sha256_file(d / name) for name in m["abstraction"]["sha256"]} != m["abstraction"]["sha256"]:
        raise SystemExit(f"the abstraction files in {d} changed since the benchmark was made")
    if mode == "bot" and sc.sha256_file(m["baseline"]["path"]) != m["baseline"]["sha256"]:
        raise SystemExit(f"the benchmark's baseline blueprint {m['baseline']['path']} changed since it was made")
    arms = (args.a, args.b)
    return {"benchmark": m["benchmark_id"], "manifest_hash": m["manifest_hash"], "measure": mode, "row": args.row,
            "deals": deals,
            "blueprint": sc.sha256_file(args.blueprint), "a": args.a, "b": args.b, "engine": engine_digest(),
            "nets": {side: sc.sha256_file(a["net"]) for side, a in zip("ab", arms) if a["net"]},
            "sources": sources(arms), "gpu": gpu_env(arms)}


def build(a: dict, blueprint) -> tuple:
    """An arm's play agent and its watched solver (None for the built-in one)."""
    agent = blueprint.play_agent()
    if a["translation"] is not None:
        agent = agent.with_translation(a["translation"])
    kw = {k: a[k] for k in SEARCH}
    watched = None
    if a["solver"] is not None:
        if a["solver"] == "gpu":
            from parciver.gpu_fused import SubgameGpu

            solver = SubgameGpu(**a["solver_kw"])
        else:
            from parciver import flop_search as fs
            from parciver import value_net as vn

            solver = fs.FlopSearch(None if a["net"] is None else vn.load(a["net"])[0], **a["solver_kw"])
        watched = Watched(solver)
        kw.update(solver=watched, rescue=False)  # a failed GPU solve stops the chunk; no CPU redo first
    return agent.with_search(**kw), watched


def play(run: dict, save, agents: tuple, watched: dict, deals: int, chunk: int, stakes: dict, seed: int,
         threads: int, hands_per_deal: int, mode: str = "h2h", bot: tuple | None = None) -> None:
    """Play chunks until `run` holds `deals` deals, calling `save()` after each;
    nothing of a chunk that stops the run is kept. `mode`: "h2h" (A against B),
    "bot" (each arm against `bot`: `match_against`'s opponent kind, the bot's
    agent and its other keywords) or "lbr" (each arm's LBR). `watched`: the GPU
    arms' watched solvers by side ("a", "b")."""
    a, b = agents
    st = (stakes["sb"], stakes["bb"], stakes["stack"])
    while (first := len(run["per_deal_bb"] if mode == "h2h" else run["a"]["per_deal_bb"])) < deals:
        size = min(chunk, deals - first)
        what = f"deals {first}..{first + size - 1}"
        t = time.time()
        for w in watched.values():
            w.take()
        kw = dict(seed=seed, threads=threads, first_deal=first)
        if mode == "h2h":
            res = {"a": dict(a.match_against("agent", size, *st, other=b, **kw))}
        elif mode == "bot":
            kind, opp, bot_kw = bot
            res = {side: dict(arm.match_against(kind, size, *st, other=opp, **bot_kw, **kw))
                   for side, arm in (("a", a), ("b", b))}
        else:
            res = {side: dict(arm.lbr(size, *st, **kw)) for side, arm in (("a", a), ("b", b))}
        calls = {}
        for side, w in watched.items():
            calls[side], failures = w.take()
            w.check(calls[side], failures, f"arm {side.upper()} {what}")
            if not calls[side]:
                raise SystemExit(f"arm {side.upper()}: no search solve reached its solver in {what}; "
                                 f"nothing of this chunk was saved")
        need = ("search_failed", "model_search_failed") if mode == "lbr" else ("a_search_failed",)
        if any(k not in r for r in res.values() for k in need):
            raise SystemExit("the engine predates search-failure counts (1 Oct; LBR's 2 Oct): rebuild it")
        # Against the bot each arm is its match's "a"; the bot (which never searches) is read too.
        failed = ({"A": res["a"]["a_search_failed"], "B": res["a"]["b_search_failed"]} if mode == "h2h" else
                  {"A": res["a"]["search_failed"], "B": res["b"]["search_failed"],
                   "LBR's model of A": res["a"]["model_search_failed"],
                   "of B": res["b"]["model_search_failed"]} if mode == "lbr" else
                  {"A": res["a"]["a_search_failed"], "B": res["b"]["a_search_failed"],
                   "the bot against A": res["a"]["b_search_failed"], "against B": res["b"]["b_search_failed"]})
        if any(map(sum, failed.values())):
            raise SystemExit(f"{what} reached a failed search ("
                             f"{', '.join(f'{k} {v}' for k, v in failed.items())} [built-in solver, blueprint]); "
                             f"nothing of this chunk was saved")
        if mode == "h2h":
            run["per_deal_bb"] += list(res["a"]["per_deal_bb"])
            for side in "ab":
                run[f"{side}_sources"] = [x + y for x, y in zip(run[f"{side}_sources"], res["a"][f"{side}_sources"])]
        else:
            for side in "ab":
                arm_run, r = run[side], res[side]
                arm_run["per_deal_bb"] += list(r["per_deal_bb"])
                if mode == "lbr":
                    arm_run["per_deal_preflop_bb"] += list(r["per_deal_preflop_bb"])
                else:
                    arm_run["sources"] = [x + y for x, y in zip(arm_run["sources"], r["a_sources"])]
        for side in "ab":
            run["solver_calls"][side] += calls.get(side, 0)
        run["seconds"] += time.time() - t
        run["chunks"].append({"first": first, "deals": size, "threads": threads, "seconds": time.time() - t,
                              "solver_calls": calls, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save()
        done = first + size
        label, per_deal = ("A vs B", run["per_deal_bb"]) if mode == "h2h" else (
            f"A - B ({mode})", [x - y for x, y in zip(run["a"]["per_deal_bb"], run["b"]["per_deal_bb"])])
        mean, se = sc.mean_se(per_deal, hands_per_deal)
        print(f"{label}: {done}/{deals} deals, {mean:+.1f} ± {se:.1f} bb/100 so far, "
              f"{run['seconds'] / done:.2f} s/deal, eta {run['seconds'] / done * (deals - done) / 60:.0f} min",
              flush=True)


def new_run(mode: str = "h2h") -> dict:
    common = {"solver_calls": {"a": 0, "b": 0}, "seconds": 0.0, "chunks": []}
    if mode == "h2h":
        return {"per_deal_bb": [], "a_sources": [0, 0, 0, 0], "b_sources": [0, 0, 0, 0], **common}
    arm = (lambda: {"per_deal_bb": [], "per_deal_preflop_bb": []}) if mode == "lbr" else (
        lambda: {"per_deal_bb": [], "sources": [0, 0, 0, 0]})
    return {"a": arm(), "b": arm(), **common}


def search_share(sources: list[int]) -> float:
    """Fraction of decisions made by search ([blueprint, missing_key, off_tree, search])."""
    return sources[3] / max(1, sum(sources))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--blueprint", required=True, help="blueprint.bin both arms play")
    p.add_argument("--a", required=True, type=arm, help="arm A (JSON)")
    p.add_argument("--b", required=True, type=arm, help="arm B (JSON)")
    p.add_argument("--row", default=None,
                   help="row whose deals and seed are played: a strength row (default baseline), or a resized-bets "
                        "or realistic-sizes row (each arm against its bot)")
    p.add_argument("--lbr", action="store_true", help="each arm's LBR on the benchmark's LBR deals and seed")
    p.add_argument("--deals", type=int, default=0, help="first N of the row's (or LBR's) deals (0 = all)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--chunk", type=int, default=250, help="deals per saved chunk (at least 2)")
    p.add_argument("--out", required=True, help="result JSON (resumed if it exists)")
    args = p.parse_args()

    bench = Path(args.benchmark)
    m = json.loads(bench.read_text(encoding="utf-8"))
    st, hpd, blocks = m["stakes"], m["hands_per_deal"], m["blocks"]
    row = None
    if args.lbr:
        if args.row is not None:
            raise SystemExit("--lbr plays the benchmark's LBR deals, not a row's")
        mode, total, seed = "lbr", m["lbr"]["deals"], m["lbr"]["seed"]
    else:
        args.row = args.row or "baseline"
        row = next((r for r in m["rows"] if r["name"] == args.row), None)
        if row is not None and row["kind"] == "agent" and row["family"] is None:
            mode = "h2h"
        elif row is not None and row["kind"] in ("perturbed", "realistic"):
            mode = "bot"
        else:
            raise SystemExit(f"{args.row!r} is neither a strength row nor a resized-bets or realistic-sizes row "
                             f"of {bench}")
        total, seed = row["deals"], row["seed"]
    deals = args.deals or total
    if not 0 < deals <= total or deals % blocks:
        raise SystemExit(f"--deals must be at most {total} and a positive multiple of {blocks}")
    if args.a == args.b:
        raise SystemExit("the arms are the same")

    out = Path(args.out)
    state = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    cfg = identity(m, args, deals, mode)
    if state.get("config", cfg) != cfg:
        old = state["config"]
        changed = sorted(k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k))
        raise SystemExit(f"{out} holds a different configuration ({', '.join(changed)} differ): {old}")
    state["config"] = cfg
    run = state.setdefault("run", new_run(mode))
    if mode != "h2h" and len({len(v) for side in "ab" for k, v in run[side].items() if k.startswith("per_deal")}) > 1:
        raise SystemExit(f"{out}: the arms' saved per-deal lists differ in length (edited by hand?)")
    left = deals - len(run["per_deal_bb"] if mode == "h2h" else run["a"]["per_deal_bb"])
    if args.chunk < 2 or left % args.chunk == 1:  # the engine plays at least 2 deals a call
        raise SystemExit(f"--chunk must be at least 2 and leave no last chunk of 1 deal (of the {left} left)")

    for side in "ab":
        print(f"arm {side.upper()}: {json.dumps(cfg[side])}", flush=True)
    if left:
        abs_dir = m["abstraction"]["dir"]
        bp = pe.SavedBlueprint(abs_dir, args.blueprint)
        (a, wa), (b, wb) = build(args.a, bp), build(args.b, bp)
        watched = {side: w for side, w in (("a", wa), ("b", wb)) if w is not None}
        bot = None
        if mode == "bot":  # the scorecard's: the benchmark's baseline blueprint, resizing its bets
            base = bp if cfg["blueprint"] == m["baseline"]["sha256"] else pe.SavedBlueprint(abs_dir, m["baseline"]["path"])
            bot = (row["kind"], base.play_agent(), {"perturb": row["perturb"]} if row["kind"] == "perturbed" else {})
        play(run, lambda: write_atomic(out, json.dumps(state)), (a, b), watched, deals, args.chunk, st, seed,
             args.threads, hpd, mode=mode, bot=bot)

    print(f"\n{m['benchmark_id']} " + {"h2h": f"row {args.row}", "bot": f"row {args.row} (each arm against its bot)",
                                       "lbr": "LBR"}[mode] + f", first {deals} deals, same blueprint {args.blueprint}")
    if mode == "h2h":
        s = sc.summarize(sc.to_blocks(run["per_deal_bb"][:deals], blocks, hpd))
        for side in "ab":
            print(f"  {side.upper()}: {json.dumps(cfg[side])}  "
                  f"(search decides {search_share(run[f'{side}_sources']):.0%})")
        print(f"  A against B {s['mean']:+7.1f} bb/100 [{s['lo']:+.1f}, {s['hi']:+.1f}]  "
              f"{sc.status(s['lo'], s['hi'], higher_is_better=True)} (higher is better for A)")
        state["summary"] = {"a_vs_b": s}
    else:
        state["summary"] = pair_summary(run, deals, blocks, hpd, mode)
        for side in "ab":
            s = state["summary"][side]
            share = "" if mode == "lbr" else f"  (search decides {search_share(run[side]['sources']):.0%})"
            print(f"  {side.upper()} {s['mean']:+7.1f} bb/100 [{s['lo']:+.1f}, {s['hi']:+.1f}]"
                  f"{'' if mode == 'bot' else ' won by LBR'}: {json.dumps(cfg[side])}{share}")
        for key, what in (("a_minus_b", "A - B"), ("postflop_a_minus_b", "A - B, postflop part")):
            if key in state["summary"]:
                s = state["summary"][key]
                print(f"  {what} {s['mean']:+7.1f} bb/100 [{s['lo']:+.1f}, {s['hi']:+.1f}]  "
                      f"{sc.status(s['lo'], s['hi'], higher_is_better=mode == 'bot')} "
                      f"({'higher' if mode == 'bot' else 'lower'} is better for A)")
    write_atomic(out, json.dumps(state))
    print(f"wrote {out}")


def pair_summary(run: dict, deals: int, blocks: int, hpd: int, mode: str) -> dict:
    """Each arm's block CI and A - B paired by block (the same deals); for LBR
    also the postflop parts' (sc.lbr_parts: the hands whose betting went past
    preflop, where LBR exploits)."""
    blk = {side: sc.to_blocks(run[side]["per_deal_bb"][:deals], blocks, hpd) for side in "ab"}
    out = {side: sc.summarize(blk[side]) for side in "ab"}
    out["a_minus_b"] = sc.paired_change(blk["a"], blk["b"])
    if mode == "lbr":
        post = {side: sc.lbr_parts(run[side]["per_deal_bb"][:deals], run[side]["per_deal_preflop_bb"][:deals], blocks,
                                   hpd)["postflop"] for side in "ab"}
        out["postflop"] = {side: sc.summarize(post[side]) for side in "ab"}
        out["postflop_a_minus_b"] = sc.paired_change(post["a"], post["b"])
    return out


if __name__ == "__main__":
    main()
