# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""RIVER R1 (records/river-live/PREREG.md): the live river at 2,000 iterations instead of 200.

  smoke  50 deals of seed 777,004 at 1 and at 4 threads, fresh arms each, each run's arms proven; the per-deal
         results must agree.  -> runs/river-live/smoke.json (scratch)
  p2     R1 against R0 on the 200bb scorecard's baseline deals 0-9,999 (M2's protocol, as LIVESEARCH P2), chunks
         of 100, resumed on rerun; R0's river jobs in deals 500-799 kept for P1.
         -> runs/river-live/p2-state.json, p1-jobs-<first>.pkl; at the end records/river-live/p2.json
  p1     each kept job solved from scratch as R0 (200 iterations) and R1 (2,000), its exploitability (% of the
         pot), and a board-clustered bootstrap of the difference.  -> records/river-live/p1.json

R0 is the live bot as scripts/slumbot_run.py builds it; R1 the same with river_iters 2,000. Built as LIVESEARCH's
arms (scripts/livesearch.py `Arm`), river jobs tapped as the RIVER-ITERS scout's (scripts/river_iters.py). From
main, the code committed:
    .venv/Scripts/python.exe -I scripts/river_live.py smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import livesearch as ls  # noqa: E402  (puts src on the path)
import river_iters as ri  # noqa: E402

REC = ROOT / "records" / "river-live"
RUNS = ROOT / "runs" / "river-live"
STATE = RUNS / "p2-state.json"
CODE = ls.PLAY_CODE + ("scripts/river_iters.py", "scripts/river_live.py")

R0_RIVER, R1_RIVER = 200, 2000
DEALS, CHUNK, THREADS, BLOCKS = 10_000, 100, 4, 50
P1_FIRST, P1_END = 500, 800  # R0's river jobs in deals 500..799 (P2's chunks 5-7; the scout had 0-499)
SMOKE_SEED, SMOKE_DEALS = 777_004, 50


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


def search_kw(search: dict, river_iters: int) -> dict:
    """An arm's search settings: the live runner's, with `river_iters`."""
    return dict(search, river_iters=river_iters)


class RArm(ls.Arm):
    """R0 or R1: LIVESEARCH's arm with the live turn (its iterations and TURN_SCHEDULE) and `river_iters`."""

    def __init__(self, name: str, sr, bp, river_iters: int, threads: int):
        super().__init__(name, sr, bp, sr.SEARCH["turn_iters"], sr.TURN_SCHEDULE, threads)
        self.search_kw = search_kw(sr.SEARCH, river_iters)
        self.iters = {3: self.search_kw["flop_iters"], 4: self.search_kw["turn_iters"], 5: river_iters}
        base = bp.play_agent()
        if base.translation != "snap":
            raise SystemExit(f"the play agent's translation is {base.translation!r}, not the live 'snap'")
        self.agent = base.with_search(solver=self.watch, **self.search_kw)
        self.mark()


def build_arms(threads: int):
    import parciver_engine as pe

    sr = ls.runner()
    if sr.SEARCH["river_iters"] != R0_RIVER:
        raise SystemExit(f"the live river runs {sr.SEARCH['river_iters']} iterations, R0 is {R0_RIVER}")
    bp = pe.SavedBlueprint(str(sr.ABSTRACTION), str(sr.BLUEPRINT))
    # Every arm's tap is built in Arm.__init__: both become RiverTaps (only R0's keeps jobs, in P1's chunks).
    tap, ls.Tap = ls.Tap, ri.RiverTap
    try:
        r0 = RArm("R0", sr, bp, R0_RIVER, threads)
        r1 = RArm("R1", sr, bp, R1_RIVER, threads)
    finally:
        ls.Tap = tap
    return sr, pe, r0, r1


def rivers(info: dict) -> list[str]:
    """livesearch.prove asks each arm for a flop and a turn solve, not a river one: R1 and R0 differ only on the
    river, so a chunk (or the smoke run) in which either arm made no river solve proves nothing."""
    return [f"{arm}: no river solve" for arm, key in (("R0", "b0"), ("R1", "b1")) if info[key]["river"] < 1]


def keeps(first: int) -> bool:
    """A chunk whose R0 river jobs are kept for P1."""
    return P1_FIRST <= first < P1_END


def benchmark(sr) -> tuple[dict, dict]:
    m = json.loads(ls.BENCHMARK.read_text(encoding="utf-8"))
    row = next(r for r in m["rows"] if r["name"] == "baseline")
    if ls.stakes_of(m) != (50, 100, 20000) or Path(m["abstraction"]["dir"]).resolve() != Path(sr.ABSTRACTION).resolve():
        raise SystemExit(f"{ls.BENCHMARK} is not the 200bb benchmark on {sr.ABSTRACTION}")
    if row["deals"] < DEALS:
        raise SystemExit(f"the baseline row has {row['deals']} deals, P2 needs {DEALS}")
    return m, row


# ---------------------------------------------------------------- smoke


def cmd_smoke(a) -> int:
    from parciver import scorecard as sc

    prov = provenance()
    sr = ls.runner()
    m, _ = benchmark(sr)
    runs, engine = {}, None
    for threads in (1, THREADS):
        # Fresh arms for each run: an agent's search cache would answer the replayed deals from the first run.
        _, pe, r0, r1 = build_arms(THREADS)
        engine = sc.sha256_file(pe.parciver_engine.__file__)
        res, info = ls.play(r0, r1, ls.stakes_of(m), SMOKE_SEED, 0, SMOKE_DEALS, threads)
        if rivers(info):
            raise SystemExit(f"smoke at {threads} thread(s): " + "; ".join(rivers(info)))
        runs[str(threads)] = dict(per_deal_bb=list(res["per_deal_bb"]), per_deal_bb_aivat=list(res["per_deal_bb_aivat"]),
                                  seconds=info["seconds"], peak_mib=info["peak_mib"], r0=info["b0"], r1=info["b1"])
        print(f"smoke at {threads} thread(s): {info['seconds']:.0f} s, R1 {np.mean(res['per_deal_bb']):+.3f} bb/deal; "
              f"R0 solves {info['b0']['flop']}/{info['b0']['turn']}/{info['b0']['river']}, "
              f"R1 {info['b1']['flop']}/{info['b1']['turn']}/{info['b1']['river']} (flop/turn/river)", flush=True)
        del r0, r1, res, info
        gc.collect()
    one, many = runs["1"], runs[str(THREADS)]
    same = one["per_deal_bb"] == many["per_deal_bb"] and one["per_deal_bb_aivat"] == many["per_deal_bb_aivat"]
    ls.save(RUNS / "smoke.json", dict(**prov, engine=engine, seed=SMOKE_SEED, deals=SMOKE_DEALS,
                                      threads=[1, THREADS], identical=same, runs=runs))
    print(f"smoke: 1 and {THREADS} threads {'identical deal for deal' if same else 'DIFFER'}")
    return 0 if same else 1


def smoke_passed(prov: dict, engine: str, threads: int) -> bool:
    """A smoke run on this play code and engine comes before P2 at either thread count; 4 threads also need it
    identical deal for deal."""
    path = RUNS / "smoke.json"
    if not path.exists():
        return False
    s = json.loads(path.read_text(encoding="utf-8"))
    return (s.get("threads") == [1, THREADS] and s.get("play_code") == prov["play_code"] and s.get("engine") == engine
            and (threads == 1 or s.get("identical") is True))


# ---------------------------------------------------------------- P2


def p2_config(threads: int, sr, pe, m: dict, row: dict, prov: dict) -> dict:
    """What a resumed P2 must match: the deals, both arms' settings, the inputs' sha256, the engine build, the
    play code's git objects, torch and triton (not the commit: a docs commit must not block a resume)."""
    from parciver import scorecard as sc

    return dict(benchmark=m["benchmark_id"], manifest_hash=m["manifest_hash"], row="baseline", seed=row["seed"],
                deals=DEALS, hands_per_deal=m["hands_per_deal"], chunk=CHUNK, threads=threads, blocks=BLOCKS,
                aivat=True, p1_deals=[P1_FIRST, P1_END], blueprint=str(sr.BLUEPRINT),
                blueprint_sha256=sc.sha256_file(sr.BLUEPRINT), net=str(sr.NET), net_sha256=sc.sha256_file(sr.NET),
                leaves=sr.LEAVES, flop_gamma=sr.FLOP_GAMMA, terminals64=sr.TERMINALS64, translation="snap",
                turn_schedule=dict(sr.TURN_SCHEDULE), r0=dict(search=search_kw(sr.SEARCH, R0_RIVER)),
                r1=dict(search=search_kw(sr.SEARCH, R1_RIVER)), engine=sc.sha256_file(pe.parciver_engine.__file__),
                play_code=prov["play_code"], torch=prov["torch"], triton=prov["triton"])


def cmd_p2(a) -> int:
    import parciver_engine as pe
    from parciver import scorecard as sc
    from parciver.solver_watch import persist

    ls.fresh(REC / "p2.json")
    prov = provenance()
    if a.threads not in (1, THREADS):
        raise SystemExit(f"P2 runs at {THREADS} threads, or at 1 if the smoke run's differ")
    if not STATE.exists() and not smoke_passed(prov, sc.sha256_file(pe.parciver_engine.__file__), a.threads):
        raise SystemExit(f"no smoke run on this code and engine{'' if a.threads == 1 else f' shows 1 and {THREADS} threads identical'}: "
                         "run `smoke` first" + ("" if a.threads == 1 else ", or P2 at --threads 1 if its runs differ"))
    sr = ls.runner()
    m, row = benchmark(sr)
    cfg = p2_config(a.threads, sr, pe, m, row, prov)
    state = ls.resume(STATE, cfg)
    if state.get("failed"):
        raise SystemExit(f"P2 stopped at a failed chunk, reported and not retried (PREREG): {state['failed']}")
    if "smoke_sha256" not in state:
        state["smoke_sha256"] = ls.sha256(RUNS / "smoke.json")
    state.setdefault("sessions", []).append(dict(started=ls.utc(), commit=prov["commit"]))
    run = state.setdefault("run", dict(per_deal_bb=[], per_deal_bb_aivat=[], aivat_mass_error=0.0, a_sources=[0] * 4,
                                       b_sources=[0] * 4, seconds=0.0, chunks=[]))
    hpd = m["hands_per_deal"]
    if len(run["per_deal_bb"]) < DEALS:
        _, _, r0, r1 = build_arms(a.threads)
        while len(run["per_deal_bb"]) < DEALS:
            first = len(run["per_deal_bb"])
            keep = keeps(first)
            r0.tap.keep = keep
            g0 = ls.gpu_state()
            try:
                # A failed proof (SystemExit) is recorded in the state, so a rerun cannot retry the chunk and hide it.
                res, info = ls.play(r0, r1, ls.stakes_of(m), row["seed"], first, CHUNK, a.threads)
                jobs = info.pop("jobs")
                bad = rivers(info)
                if keep and (len(jobs) != info["b0"]["river"]
                             or any(len(j["board"]) != 5 or int(j["iters"]) != R0_RIVER for j in jobs)):
                    bad.append(f"R0's kept jobs are not its {info['b0']['river']} river jobs at {R0_RIVER} iterations")
                elif not keep and jobs:
                    bad.append(f"{len(jobs)} jobs kept outside P1's chunks")
                if bad:
                    raise SystemExit(f"deals {first}..{first + CHUNK - 1}: " + "; ".join(bad) + ". Nothing saved.")
            except SystemExit as e:
                state["failed"] = dict(first=first, error=str(e), at=ls.utc())
                ls.save(STATE, state)
                raise
            r0.tap.keep = False
            g1 = ls.gpu_state()
            chunk = dict(first=first, deals=CHUNK, seconds=info["seconds"], ended=ls.utc(), r0=info["b0"], r1=info["b1"],
                         peak_mib=info["peak_mib"], gpu_start=g0, gpu_end=g1, shared=ls.busy(g0) or ls.busy(g1),
                         times=dict(r0=info["times"]["b0"], r1=info["times"]["b1"]))
            if keep:
                f = RUNS / f"p1-jobs-{first:05d}.pkl"
                tmp = f.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump(jobs, fh, protocol=pickle.HIGHEST_PROTOCOL)
                persist(tmp, f)
                chunk["p1_jobs"] = dict(file=f.name, jobs=len(jobs), sha256=ls.sha256(f))
            del jobs
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
            print(f"p2: {done}/{DEALS} deals, R1 - R0 {mean:+.2f} ± {se:.2f} bb/100 so far, "
                  f"{run['seconds'] / done:.2f} s/deal, eta {run['seconds'] / done * (DEALS - done) / 3600:.1f} h"
                  + ("  (GPU shared or unknown: this chunk's latency is left out)" if chunk["shared"] else ""), flush=True)
    return p2_finish(state, hpd)


def p2_finish(state: dict, hpd: int) -> int:
    from parciver import scorecard as sc

    run = state["run"]
    s = sc.summarize(sc.to_blocks(run["per_deal_bb"][:DEALS], BLOCKS, hpd))
    s_aivat = sc.summarize(sc.to_blocks(run["per_deal_bb_aivat"][:DEALS], BLOCKS, hpd))
    down = s["hi"] < 0
    clean = [c for c in run["chunks"] if not c["shared"] and "p1_jobs" not in c]
    latency = {arm: {street: ls.stats([t for c in clean for t in c["times"][arm][street]]) for street in ("flop", "turn", "river")}
               for arm in ("r0", "r1")}
    solves = {arm: {street: sum(c[arm][street] for c in run["chunks"]) / DEALS for street in ("flop", "turn", "river")}
              for arm in ("r0", "r1")}
    share = lambda src: src[3] / max(1, sum(src))  # [blueprint, missing_key, off_tree, search]
    summary = dict(s, down=down, aivat=s_aivat, aivat_mass_error=run["aivat_mass_error"], s_per_deal=run["seconds"] / DEALS,
                   latency_in_lock=latency, latency_chunks=len(clean), shared_chunks=[c["first"] for c in run["chunks"] if c["shared"]],
                   solves_per_deal=solves, r1_search_share=share(run["a_sources"]), r0_search_share=share(run["b_sources"]))
    state["summary"] = summary
    ls.save(STATE, state)
    ls.fresh(REC / "p2.json")
    record = dict(registered="records/river-live/PREREG.md, P2", config=state["config"], sessions=state["sessions"],
                  smoke_sha256=state.get("smoke_sha256"), summary=summary,
                  cuda_failures=state.get("cuda_failures", []), code_history=state.get("code_history", []),
                  chunks=[{k: v for k, v in c.items() if k != "times"} for c in run["chunks"]],
                  per_deal_bb=run["per_deal_bb"], per_deal_bb_aivat=run["per_deal_bb_aivat"])
    ls.save(REC / "p2.json", record)
    print(f"P2: R1 - R0 = {s['mean']:+.2f} bb/100, 95% [{s['lo']:+.2f}, {s['hi']:+.2f}] over {DEALS} deals: "
          f"{'DOWN' if down else 'not DOWN'}; AIVAT (reported) {s_aivat['mean']:+.2f} [{s_aivat['lo']:+.2f}, {s_aivat['hi']:+.2f}]")
    return 0


# ---------------------------------------------------------------- P1


def p1_rows(solve, jobs: list) -> list[dict]:
    """Each job's row: R0's and R1's exploitability by `solve(job, iters)`, D = R1 - R0."""
    rows = []
    for job in jobs:
        e0, e1 = solve(job, R0_RIVER), solve(job, R1_RIVER)
        rows.append(dict(board=list(ri.river_key(job["board"])), digest=ls.job_digest(job), pot=ls.pot_of(job),
                         frozen=len(job["frozen"]), nodes=len(job["kind"]), r0=e0, r1=e1, d=e1 - e0))
    return rows


def p1_stat(rows: list[dict]) -> dict:
    rows.sort(key=lambda r: (tuple(r["board"]), r["digest"]))
    boot = ls.board_bootstrap([r["d"] for r in rows], [tuple(r["board"]) for r in rows])
    return dict(bootstrap=boot, holds=boot["hi"] < 0, r0_mean=float(np.mean([r["r0"] for r in rows])),
                r1_mean=float(np.mean([r["r1"] for r in rows])), r0_median=float(np.median([r["r0"] for r in rows])),
                r1_median=float(np.median([r["r1"] for r in rows])), r1_lower=int(sum(r["d"] < 0 for r in rows)))


def cmd_p1(a) -> int:
    from parciver import gpu_fused as gf
    from parciver import gpu_search as gs

    out = REC / "p1.json"
    ls.fresh(out)
    prov = provenance()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    cfg = state["config"]
    same = {k: (cfg[k], prov[k]) for k in ("play_code", "torch", "triton")}
    if any(x != y for x, y in same.values()):
        raise SystemExit(f"P1 must run P2's play code, torch and triton: {same}")
    kept = [c for c in state["run"]["chunks"] if keeps(c["first"])]
    if [c["first"] for c in kept] != list(range(P1_FIRST, P1_END, cfg["chunk"])) or not all("p1_jobs" in c for c in kept):
        raise SystemExit(f"P2's chunks of deals {P1_FIRST}..{P1_END - 1} are not all saved with their jobs")
    for c in kept:
        ls.need_sha(RUNS / c["p1_jobs"]["file"], c["p1_jobs"]["sha256"])
    gpu0 = ls.gpu_state()
    if ls.busy(gpu0):
        raise SystemExit(f"the GPU's other users: {gpu0['peers']} (None: unknown): P1 needs it alone")
    rows, t0 = [], time.time()
    for c in kept:
        with open(RUNS / c["p1_jobs"]["file"], "rb") as fh:
            jobs = pickle.load(fh)
        if len(jobs) != c["p1_jobs"]["jobs"] or any(len(j["board"]) != 5 or int(j["iters"]) != R0_RIVER for j in jobs):
            raise SystemExit(f"{c['p1_jobs']['file']}: not the {c['p1_jobs']['jobs']} R0 river jobs its record says")
        rows += p1_rows(lambda job, n: ls.solve_expl(gf, gs, job, n, None), jobs)
        print(f"p1: {len(rows)} jobs, {time.time() - t0:.0f} s", flush=True)
        del jobs
    st = p1_stat(rows)
    gpu1 = ls.gpu_state()
    result = dict(registered="records/river-live/PREREG.md, P1", **prov, p2_config=cfg,
                  jobs_files=[c["p1_jobs"] for c in kept], r0=dict(iters=R0_RIVER), r1=dict(iters=R1_RIVER), **st,
                  rows=rows, seconds=time.time() - t0, gpu_start=gpu0, gpu_end=gpu1, void=ls.busy(gpu1))
    ls.save(out, result, indent=1)
    b = st["bootstrap"]
    print(f"P1: R1 - R0 = {b['mean']:+.4f}% of the pot, 95% [{b['lo']:+.4f}, {b['hi']:+.4f}] over {b['jobs']} jobs on "
          f"{b['boards']} boards (R0 {st['r0_mean']:.3f}%, R1 {st['r1_mean']:.3f}%): "
          f"{'HOLDS' if st['holds'] else 'does NOT hold'}" + ("  (VOID: the GPU was not ours alone at the end)" if result["void"] else ""))
    return 2 if result["void"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p2 = sub.add_parser("p2")
    p2.add_argument("--threads", type=int, default=THREADS, help="4 as registered; 1 if the smoke run's differ")
    sub.add_parser("p1")
    a = ap.parse_args()
    return {"smoke": cmd_smoke, "p2": cmd_p2, "p1": cmd_p1}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
