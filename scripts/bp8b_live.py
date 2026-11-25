# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""BP-8B's live match (records/bp-8b/PREREG.md "Live match"): L1 (the live bot with v2-8B) against L0 (the live bot).

  smoke  50 deals of seed 777,005 at 1 and at 4 threads, fresh arms each, each run's arms proven; the per-deal
         results must agree.  -> runs/bp-8b-live/smoke.json (scratch)
  match  L1 against L0 on the 200bb scorecard's baseline deals 0-9,999 (LIVESEARCH P2's protocol, as RIVER R1's
         P2), chunks of 100, resumed on rerun.  -> runs/bp-8b-live/state.json; at the end records/bp-8b/live.json

L0 is the live bot as scripts/slumbot_run.py builds it (its blueprint, search, net and schedules); L1 the same with
v2-8B (runs/bp-8b-stage2/blueprint.bin, its registered sha256) in place of L0's blueprint. Built as LIVESEARCH's
arms (scripts/livesearch.py `Arm`), both `snap`; L1 is the engine's A, so the result is L1 - L0. DOWN iff the
95% interval's upper end is < 0. A chunk that fails (a proof, a search, CUDA) is recorded in the state and stops
the match; a rerun refuses it (Amendment 1). A proof's error names the arms as livesearch does: B1 is L1 (the
engine's A), B0 is L0. From main, the code committed:
    .venv/Scripts/python.exe -I scripts/bp8b_live.py smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import livesearch as ls  # noqa: E402  (puts src on the path)

REC = ROOT / "records" / "bp-8b"
RUNS = ROOT / "runs" / "bp-8b-live"
STATE = RUNS / "state.json"
CODE = ls.PLAY_CODE + ("scripts/bp8b_live.py",)

V2_8B = ROOT / "runs" / "bp-8b-stage2" / "blueprint.bin"
V2_8B_SHA = "fd27fe50324eba29f0454ab5c68604c2920f2bad18dd437544255d754c7ae071"  # records/bp-8b/RESULT.md
DEALS, CHUNK, THREADS, BLOCKS = 10_000, 100, 4, 50
SMOKE_SEED, SMOKE_DEALS = 777_005, 50


# ---------------------------------------------------------------- provenance and arms


def play_code() -> dict:
    """livesearch.play_code over this registration's code: git object ids at HEAD, refused if any is changed."""
    out = ls.git("status", "--porcelain", "-z", "--untracked-files=all", "--", *CODE)
    dirty = [x for x in out.split("\0") if x and (not x.startswith("??") or x.endswith((".py", ".rs", ".toml")))]
    if dirty:
        raise SystemExit(f"uncommitted play code: {dirty[:5]}: commit it first (every record holds the commit)")
    return {p: ls.git("rev-parse", f"HEAD:{p}").strip() for p in CODE}


def provenance() -> dict:
    return dict(ls.provenance(), play_code=play_code())


def blueprints(sr) -> dict:
    """L0's blueprint (the live runner's) and L1's (v2-8B), each by path and sha256: v2-8B's must be the registered
    one, and L0's must not be it (after an adoption there is no L0 to test against)."""
    l1 = ls.need_sha(V2_8B, V2_8B_SHA)
    l0 = ls.sha256(sr.BLUEPRINT)
    if l0 == V2_8B_SHA or Path(sr.BLUEPRINT).resolve() == V2_8B.resolve():
        raise SystemExit(f"the live runner already plays v2-8B ({sr.BLUEPRINT}): no L0 to match it against")
    return dict(l0=dict(path=str(sr.BLUEPRINT), sha256=l0), l1=dict(path=str(V2_8B), sha256=l1))


def build_arms(threads: int, bps: dict):
    """L0 and L1: LIVESEARCH's arm with the live search (its turn iterations and TURN_SCHEDULE), each on its own
    blueprint, both on the live abstraction."""
    import parciver_engine as pe

    sr = ls.runner()
    loaded = {}
    for arm in ("l0", "l1"):
        bp = pe.SavedBlueprint(str(sr.ABSTRACTION), bps[arm]["path"])
        if Path(bp.source_path()).resolve() != Path(bps[arm]["path"]).resolve():
            raise SystemExit(f"{arm}'s blueprint loaded from {bp.source_path()}, not {bps[arm]['path']}")
        loaded[arm] = bp
    if loaded["l0"].grid != loaded["l1"].grid:  # the engine builds the match's tree from A's (L1's) grid
        raise SystemExit(f"L0's grid {loaded['l0'].grid} is not L1's {loaded['l1'].grid}")
    l0 = ls.Arm("L0", sr, loaded["l0"], sr.SEARCH["turn_iters"], sr.TURN_SCHEDULE, threads)
    l1 = ls.Arm("L1", sr, loaded["l1"], sr.SEARCH["turn_iters"], sr.TURN_SCHEDULE, threads)
    infosets = {arm: int(bp.num_infosets()) for arm, bp in loaded.items()}
    return sr, pe, l0, l1, dict(infosets=infosets, grid=loaded["l1"].grid)


def benchmark(sr) -> tuple[dict, dict]:
    m = json.loads(ls.BENCHMARK.read_text(encoding="utf-8"))
    row = next(r for r in m["rows"] if r["name"] == "baseline")
    if ls.stakes_of(m) != (50, 100, 20000) or Path(m["abstraction"]["dir"]).resolve() != Path(sr.ABSTRACTION).resolve():
        raise SystemExit(f"{ls.BENCHMARK} is not the 200bb benchmark on {sr.ABSTRACTION}")
    if row["deals"] < DEALS:
        raise SystemExit(f"the baseline row has {row['deals']} deals, the match needs {DEALS}")
    return m, row


# ---------------------------------------------------------------- smoke


def cmd_smoke(a) -> int:
    from parciver import scorecard as sc

    prov = provenance()
    sr = ls.runner()
    m, _ = benchmark(sr)
    bps = blueprints(sr)
    runs, engine, loaded = {}, None, None
    for threads in (1, THREADS):
        # Fresh arms for each run: an agent's search cache would answer the replayed deals from the first run.
        _, pe, l0, l1, loaded = build_arms(THREADS, bps)
        engine = sc.sha256_file(pe.parciver_engine.__file__)
        res, info = ls.play(l0, l1, ls.stakes_of(m), SMOKE_SEED, 0, SMOKE_DEALS, threads)
        runs[str(threads)] = dict(per_deal_bb=list(res["per_deal_bb"]), per_deal_bb_aivat=list(res["per_deal_bb_aivat"]),
                                  seconds=info["seconds"], peak_mib=info["peak_mib"], l0=info["b0"], l1=info["b1"])
        print(f"smoke at {threads} thread(s): {info['seconds']:.0f} s, L1 {np.mean(res['per_deal_bb']):+.3f} bb/deal; "
              f"L0 solves {info['b0']['flop']}/{info['b0']['turn']}/{info['b0']['river']}, "
              f"L1 {info['b1']['flop']}/{info['b1']['turn']}/{info['b1']['river']} (flop/turn/river)", flush=True)
        del l0, l1, res, info
        gc.collect()
    one, many = runs["1"], runs[str(THREADS)]
    same = one["per_deal_bb"] == many["per_deal_bb"] and one["per_deal_bb_aivat"] == many["per_deal_bb_aivat"]
    ls.save(RUNS / "smoke.json", dict(**prov, engine=engine, blueprints=bps, **loaded, seed=SMOKE_SEED,
                                      deals=SMOKE_DEALS, threads=[1, THREADS], identical=same, runs=runs))
    print(f"smoke: 1 and {THREADS} threads {'identical deal for deal' if same else 'DIFFER'}")
    return 0 if same else 1


def smoke_passed(prov: dict, engine: str, bps: dict, threads: int) -> bool:
    """A smoke run on this play code, engine and both blueprints comes before the match at either thread count;
    4 threads also need it identical deal for deal."""
    path = RUNS / "smoke.json"
    if not path.exists():
        return False
    s = json.loads(path.read_text(encoding="utf-8"))
    return (s.get("threads") == [1, THREADS] and s.get("play_code") == prov["play_code"] and s.get("engine") == engine
            and s.get("blueprints") == bps and (threads == 1 or s.get("identical") is True))


# ---------------------------------------------------------------- the match


def config(threads: int, sr, pe, m: dict, row: dict, prov: dict, bps: dict) -> dict:
    """What a resumed match must match (RIVER R1's pins plus both blueprints' sha256): the deals, the arms'
    settings, the inputs' sha256, the engine build, the play code's git objects, torch and triton (not the commit:
    a docs commit must not block a resume)."""
    from parciver import scorecard as sc

    return dict(benchmark=m["benchmark_id"], manifest_hash=m["manifest_hash"], row="baseline", seed=row["seed"],
                deals=DEALS, hands_per_deal=m["hands_per_deal"], chunk=CHUNK, threads=threads, blocks=BLOCKS,
                aivat=True, blueprints=bps, abstraction=str(sr.ABSTRACTION), net=str(sr.NET),
                net_sha256=sc.sha256_file(sr.NET), leaves=sr.LEAVES, flop_gamma=sr.FLOP_GAMMA,
                terminals64=sr.TERMINALS64, translation="snap", turn_schedule=dict(sr.TURN_SCHEDULE),
                search=dict(sr.SEARCH), engine=sc.sha256_file(pe.parciver_engine.__file__),
                play_code=prov["play_code"], torch=prov["torch"], triton=prov["triton"])


def cmd_match(a) -> int:
    import parciver_engine as pe
    from parciver import scorecard as sc

    ls.fresh(REC / "live.json")
    prov = provenance()
    if a.threads not in (1, THREADS):
        raise SystemExit(f"the match runs at {THREADS} threads, or at 1 if the smoke run's differ")
    sr = ls.runner()
    m, row = benchmark(sr)
    bps = blueprints(sr)
    if not STATE.exists() and not smoke_passed(prov, sc.sha256_file(pe.parciver_engine.__file__), bps, a.threads):
        raise SystemExit(f"no smoke run on this code, engine and blueprints"
                         f"{'' if a.threads == 1 else f' shows 1 and {THREADS} threads identical'}: run `smoke` first"
                         + ("" if a.threads == 1 else ", or the match at --threads 1 if its runs differ"))
    cfg = config(a.threads, sr, pe, m, row, prov, bps)
    state = ls.resume(STATE, cfg)
    if state.get("failed"):
        raise SystemExit(f"the match stopped at a failed chunk, reported and not retried (Amendment 1): {state['failed']}")
    if "smoke_sha256" not in state:
        state["smoke_sha256"] = ls.sha256(RUNS / "smoke.json")
    state.setdefault("sessions", []).append(dict(started=ls.utc(), commit=prov["commit"]))
    run = state.setdefault("run", dict(per_deal_bb=[], per_deal_bb_aivat=[], aivat_mass_error=0.0, a_sources=[0] * 4,
                                       b_sources=[0] * 4, seconds=0.0, chunks=[]))
    hpd = m["hands_per_deal"]
    if len(run["per_deal_bb"]) < DEALS:
        _, _, l0, l1, loaded = build_arms(a.threads, bps)
        # The blueprints are read into RAM: hashed again once loaded, so a file replaced meanwhile is caught.
        try:
            now = blueprints(sr)  # raises itself if v2-8B is not the registered one, or L0 now plays it
        except SystemExit as e:
            now = str(e)
        why = (f"the blueprints changed while loading: {now}, pinned {bps}" if now != bps else
               f"the loaded blueprints are {loaded}, earlier sessions' {state['loaded']}"
               if state.setdefault("loaded", loaded) != loaded else None)
        if why:
            state["failed"] = dict(first=len(run["per_deal_bb"]), error=why, at=ls.utc())
            ls.save(STATE, state)
            raise SystemExit(why)
        while len(run["per_deal_bb"]) < DEALS:
            first = len(run["per_deal_bb"])
            g0 = ls.gpu_state()
            try:
                res, info = ls.play(l0, l1, ls.stakes_of(m), row["seed"], first, CHUNK, a.threads)
                if info.pop("jobs"):
                    raise SystemExit(f"deals {first}..{first + CHUNK - 1}: jobs kept with no tap set. Nothing saved.")
            except BaseException as e:
                # Recorded in the state, so a rerun cannot retry the chunk and hide it. A Ctrl-C is not a failure.
                if not isinstance(e, KeyboardInterrupt):
                    state["failed"] = dict(first=first, error=f"{type(e).__name__}: {e}", at=ls.utc())
                    ls.save(STATE, state)
                raise
            g1 = ls.gpu_state()
            chunk = dict(first=first, deals=CHUNK, seconds=info["seconds"], ended=ls.utc(), l0=info["b0"], l1=info["b1"],
                         peak_mib=info["peak_mib"], gpu_start=g0, gpu_end=g1, shared=ls.busy(g0) or ls.busy(g1),
                         times=dict(l0=info["times"]["b0"], l1=info["times"]["b1"]))
            run["per_deal_bb"] += list(res["per_deal_bb"])
            run["per_deal_bb_aivat"] += list(res["per_deal_bb_aivat"])
            run["aivat_mass_error"] = max(run["aivat_mass_error"], float(res["aivat_mass_error"]))
            run["a_sources"] = [x + y for x, y in zip(run["a_sources"], res["a_sources"])]
            run["b_sources"] = [x + y for x, y in zip(run["b_sources"], res["b_sources"])]
            run["seconds"] += info["seconds"]
            run["chunks"].append(chunk)
            ls.save(STATE, state)
            done = len(run["per_deal_bb"])
            mean, se = sc.mean_se(run["per_deal_bb"], hpd)
            print(f"match: {done}/{DEALS} deals, L1 - L0 {mean:+.2f} ± {se:.2f} bb/100 so far, "
                  f"{run['seconds'] / done:.2f} s/deal, eta {run['seconds'] / done * (DEALS - done) / 3600:.1f} h"
                  + ("  (GPU shared or unknown: this chunk's latency is left out)" if chunk["shared"] else ""), flush=True)
    return finish(state, hpd)


def summary_of(run: dict, hpd: int) -> dict:
    from parciver import scorecard as sc

    s = sc.summarize(sc.to_blocks(run["per_deal_bb"][:DEALS], BLOCKS, hpd))
    s_aivat = sc.summarize(sc.to_blocks(run["per_deal_bb_aivat"][:DEALS], BLOCKS, hpd))
    clean = [c for c in run["chunks"] if not c["shared"]]
    latency = {arm: {street: ls.stats([t for c in clean for t in c["times"][arm][street]]) for street in ("flop", "turn", "river")}
               for arm in ("l0", "l1")}
    solves = {arm: {street: sum(c[arm][street] for c in run["chunks"]) / DEALS for street in ("flop", "turn", "river")}
              for arm in ("l0", "l1")}
    share = lambda src: src[3] / max(1, sum(src))  # [blueprint, missing_key, off_tree, search]
    return dict(s, down=s["hi"] < 0, aivat=s_aivat, aivat_mass_error=run["aivat_mass_error"], s_per_deal=run["seconds"] / DEALS,
                latency_in_lock=latency, latency_chunks=len(clean), shared_chunks=[c["first"] for c in run["chunks"] if c["shared"]],
                solves_per_deal=solves, l1_search_share=share(run["a_sources"]), l0_search_share=share(run["b_sources"]))


def finish(state: dict, hpd: int) -> int:
    run = state["run"]
    if len(run["per_deal_bb"]) != DEALS or len(run["per_deal_bb_aivat"]) != DEALS:
        raise SystemExit(f"{len(run['per_deal_bb'])} deals played, the match is {DEALS}")
    summary = summary_of(run, hpd)
    state["summary"] = summary
    ls.save(STATE, state)
    ls.fresh(REC / "live.json")
    record = dict(registered="records/bp-8b/PREREG.md, Live match (Amendment 1)", config=state["config"],
                  sessions=state["sessions"], smoke_sha256=state.get("smoke_sha256"), loaded=state.get("loaded"),
                  failures=state.get("failures", []), code_history=state.get("code_history", []),
                  summary=summary, chunks=[{k: v for k, v in c.items() if k != "times"} for c in run["chunks"]],
                  per_deal_bb=run["per_deal_bb"], per_deal_bb_aivat=run["per_deal_bb_aivat"])
    ls.save(REC / "live.json", record)
    s, sa = summary, summary["aivat"]
    print(f"BP-8B live: L1 - L0 = {s['mean']:+.2f} bb/100, 95% [{s['lo']:+.2f}, {s['hi']:+.2f}] over {DEALS} deals: "
          f"{'DOWN' if s['down'] else 'not DOWN'}; AIVAT (reported) {sa['mean']:+.2f} [{sa['lo']:+.2f}, {sa['hi']:+.2f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("match")
    p.add_argument("--threads", type=int, default=THREADS, help="4 as registered; 1 if the smoke run's differ")
    a = ap.parse_args()
    return {"smoke": cmd_smoke, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
