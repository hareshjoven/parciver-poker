# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""RIVER-ITERS scout (records/river-iters/PREREG.md): how far is the live river solve from equilibrium?

  collect  B1 (the live config) against B0 on the 200bb scorecard's baseline deals 0-499, P2's protocol (4 threads,
           AIVAT on, chunks of 100, each chunk's arms proven); every B1 river job copied before its solve.
           -> runs/river-iters/jobs-<first>.pkl and runs/river-iters/collect.json (resumed on rerun)
  solve    each kept job solved from scratch (float32, Q4's terminals, its frozen slots held) with dcfr_g2 (the
           live river's) and shift1_g5 (the live turn's), exploitability (% of the pot) at 100-2000 iterations,
           a board-clustered bootstrap of each mean, and the registered reading.  -> records/river-iters/scout.json

Reuses scripts/livesearch.py (arms, proofs, tap, bootstrap); changes nothing the live bot runs. From main, the play
code committed:
    .venv/Scripts/python.exe -I scripts/river_iters.py collect
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import livesearch as ls  # noqa: E402  (puts src on the path)

REC = ROOT / "records" / "river-iters"
RUNS = ROOT / "runs" / "river-iters"
STATE = RUNS / "collect.json"
OUT = REC / "scout.json"
P2 = ls.REC / "p2.json"

DEALS, CHUNK, THREADS = 500, 100, 4
RIVER_ITERS = 200  # the live river's (slumbot_run.py SEARCH), checked against the runner below
ARMS = ("dcfr_g2", "shift1_g5")
CHECK = (100, 200, 300, 500, 1000, 2000)
WORTH, HALF = 0.50, 0.5  # PROCEED: E(dcfr_g2, 200) >= 0.50% of the pot, and some E <= half of it, intervals apart


class RiverTap(ls.Tap):
    """livesearch's Tap, keeping each river job (5-card board) instead of each turn job."""

    def __call__(self, job: dict):
        street = len(job["board"])
        kept = copy.deepcopy(dict(job)) if self.keep and street == 5 else None
        with self._lock:
            self.seen[f"{street}:{int(job['iters'])}"] += 1
            if kept is not None:
                self.jobs.append(kept)
        return self.solver(job)


def committed() -> None:
    """This scout's own code and registration, committed (the record holds the commit)."""
    dirty = ls.git("status", "--porcelain", "--", "scripts/river_iters.py", "records/river-iters/PREREG.md").strip()
    if dirty:
        raise SystemExit(f"uncommitted: {dirty}: commit the scout first")


def river_key(board) -> tuple:
    """A river board as a set: the flop's cards sorted, then the turn and river cards."""
    b = [int(c) for c in board]
    return (*sorted(b[:3]), b[3], b[4])


# ---------------------------------------------------------------- collect


class Void(Exception):
    """A failed proof or search (PREREG.md: VOID). It, or any crash in a chunk's match, is recorded in
    collect.json, which then refuses to go on."""


def play(b0, b1, stakes, seed: int, first: int, deals: int) -> tuple[dict, dict, list]:
    """livesearch.play, returning B1's kept jobs (it returns B0's): refused, nothing kept, if a search failed or
    an arm is not proven."""
    import torch

    b0.mark()
    b1.mark()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    res = dict(b1.agent.match_against("agent", deals, *stakes, other=b0.agent, seed=seed, threads=THREADS,
                                      first_deal=first, aivat=True))
    dt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2**20
    (c1, _, jobs), (c0, _, _) = b1.chunk(), b0.chunk()
    bad = ls.prove("B1", c1, b1.iters, b1.schedule) + ls.prove("B0", c0, b0.iters, b0.schedule)
    if any(res["a_search_failed"]) or any(res["b_search_failed"]):
        bad.append(f"decisions reached a failed search (B1 {res['a_search_failed']}, B0 {res['b_search_failed']})")
    for key in ("per_deal_bb", "per_deal_bb_aivat"):
        if len(res.get(key, ())) != deals:
            bad.append(f"the engine returned {len(res.get(key, ()))} {key} values for {deals} deals")
    if len(jobs) != c1["river"]:
        bad.append(f"B1: {len(jobs)} river jobs kept for {c1['river']} river solves")
    if any(len(j["board"]) != 5 or int(j["iters"]) != RIVER_ITERS for j in jobs):
        bad.append("B1: kept jobs that are not river jobs at the live iterations")
    if bad:
        raise Void(f"deals {first}..{first + deals - 1} of seed {seed}: " + "; ".join(bad)
                         + ". Nothing of this chunk was saved.")
    return res, dict(seconds=dt, b0=c0, b1=c1, peak_mib=peak), jobs


def cmd_collect(a) -> int:
    import parciver_engine as pe
    from parciver import scorecard as sc
    from parciver.solver_watch import persist

    ls.fresh(OUT)
    committed()
    prov = ls.provenance()
    arm, schedule, stage1 = ls.b1_settings()
    m = json.loads(ls.BENCHMARK.read_text(encoding="utf-8"))
    row = next(r for r in m["rows"] if r["name"] == "baseline")
    sr = ls.runner()
    if ls.stakes_of(m) != (50, 100, 20000) or Path(m["abstraction"]["dir"]).resolve() != Path(sr.ABSTRACTION).resolve():
        raise SystemExit(f"{ls.BENCHMARK} is not the 200bb benchmark on {sr.ABSTRACTION}")
    if sr.SEARCH["river_iters"] != RIVER_ITERS:
        raise SystemExit(f"the live river runs {sr.SEARCH['river_iters']} iterations, this scout {RIVER_ITERS}")
    if ls.resolved(ls.ARMS["dcfr_g2"]) != ls.resolved(None) or ls.ARMS["shift1_g5"] != schedule:
        raise SystemExit("dcfr_g2 is not the river's default schedule, or shift1_g5 not B1's turn schedule")
    cfg = dict(benchmark=m["benchmark_id"], manifest_hash=m["manifest_hash"], row="baseline", seed=row["seed"],
               deals=DEALS, chunk=CHUNK, threads=THREADS, b1_arm=arm, b1_schedule=schedule, stage1_sha256=stage1,
               search=dict(sr.SEARCH), blueprint=str(sr.BLUEPRINT), blueprint_sha256=sc.sha256_file(sr.BLUEPRINT),
               net=str(sr.NET), net_sha256=sc.sha256_file(sr.NET), leaves=sr.LEAVES, flop_gamma=sr.FLOP_GAMMA,
               terminals64=sr.TERMINALS64, scout_code=ls.git("rev-parse", "HEAD:scripts/river_iters.py").strip(),
               engine=sc.sha256_file(pe.parciver_engine.__file__),
               play_code=prov["play_code"], torch=prov["torch"], triton=prov["triton"])
    state = ls.resume(STATE, cfg)
    if state.get("void"):
        raise SystemExit(f"{STATE} is VOID ({state['void']['reason']}): this scout is over")
    state.setdefault("sessions", []).append(dict(started=ls.utc(), commit=prov["commit"]))
    chunks = state.setdefault("chunks", [])
    if len(chunks) * CHUNK < DEALS:
        # Every arm's solver is a Tap built in Arm.__init__: B1's (and B0's, which keeps nothing) become RiverTaps.
        tap, ls.Tap = ls.Tap, RiverTap
        try:
            _, _, b0, b1 = ls.build_arms(THREADS, schedule)
        finally:
            ls.Tap = tap
        b1.tap.keep = True
        while len(chunks) * CHUNK < DEALS:
            first = len(chunks) * CHUNK
            g0 = ls.gpu_state()
            try:
                res, info, jobs = play(b0, b1, ls.stakes_of(m), row["seed"], first, CHUNK)
            except Exception as e:  # a crash in the match is VOID too (PREREG.md), not retried
                state["void"] = dict(utc=ls.utc(), first=first, reason=str(e))
                ls.save(STATE, state)
                raise SystemExit(f"VOID: {e}")
            g1 = ls.gpu_state()
            f = RUNS / f"jobs-{first:05d}.pkl"
            tmp = f.with_suffix(".tmp")
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                pickle.dump(jobs, fh, protocol=pickle.HIGHEST_PROTOCOL)
            persist(tmp, f)
            chunks.append(dict(first=first, deals=CHUNK, seconds=info["seconds"], ended=ls.utc(), b0=info["b0"],
                               b1=info["b1"], peak_mib=info["peak_mib"], gpu_start=g0, gpu_end=g1,
                               shared=ls.busy(g0) or ls.busy(g1), per_deal_bb=list(res["per_deal_bb"]),
                               jobs=dict(file=f.name, n=len(jobs), sha256=ls.sha256(f))))
            ls.save(STATE, state)
            print(f"collect: deals {first}..{first + CHUNK - 1}: {len(jobs)} river jobs kept, "
                  f"{info['seconds']:.0f} s" + ("  (GPU shared or unknown)" if chunks[-1]["shared"] else ""), flush=True)
            del jobs
    # Descriptive: the same deals' results as P2's (the same play, so the same river jobs).
    p2 = json.loads(P2.read_text(encoding="utf-8"))
    ours = [x for c in chunks for x in c["per_deal_bb"]]
    state["reproduces_p2"] = ours == p2["per_deal_bb"][:DEALS]
    ls.save(STATE, state)
    n = sum(c["jobs"]["n"] for c in chunks)
    print(f"collect: {n} river jobs over {DEALS} deals ({n / DEALS:.3f} a deal); deals' results identical to P2's: "
          f"{state['reproduces_p2']}")
    return 0


# ---------------------------------------------------------------- solve


def curve(gf, gs, job: dict, kw: dict) -> tuple[list[float], list[float]]:
    """livesearch.solve_expl with checkpoints: exploitability (% of the pot) and the synchronised solve seconds
    (construction and steps, not the evaluations) at each of CHECK."""
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hands = gs.Hands.from_arrays(job["board"], job["strength"], job["alive"])
    s = gf.FusedSolver(hands, job, job["ranges"], dtype=torch.float32, terminals64=True, **kw)
    if not s.ff:
        raise SystemExit("the solver is not running Q4's terminals")
    for slot, strategy in job["frozen"]:
        s.freeze(slot, np.asarray(strategy))
    pot = ls.pot_of(job)
    expl, secs, spent = [], [], 0.0
    for it in range(1, max(CHECK) + 1):
        s.step()
        if it in CHECK:
            torch.cuda.synchronize()
            spent += time.perf_counter() - t0
            secs.append(spent)
            expl.append(100.0 * s.exploitability() / pot)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    del s
    return expl, secs


def reading(boot: dict) -> dict:
    """The registered rule (PREREG.md "Reading") on the bootstrapped means."""
    base = boot["dcfr_g2"][CHECK.index(200)]
    worth = base["mean"] >= WORTH
    found = [(n, arm) for n in CHECK for arm in ARMS
             if boot[arm][CHECK.index(n)]["mean"] <= HALF * base["mean"] and boot[arm][CHECK.index(n)]["hi"] < base["lo"]]
    candidate = dict(arm=found[0][1], iters=found[0][0]) if found else None
    return dict(e_live=base, worth=worth, halved=[dict(arm=x, iters=n) for n, x in found], candidate=candidate,
                verdict="PROCEED" if worth and candidate else "CLOSE")


def cmd_solve(a) -> int:
    from parciver import gpu_fused as gf
    from parciver import gpu_search as gs

    ls.fresh(OUT)
    committed()
    prov = ls.provenance()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("void"):
        raise SystemExit(f"{STATE} is VOID ({state['void']['reason']})")
    cfg, chunks = state["config"], state["chunks"]
    same = {k: (cfg["play_code"]["src"], prov["play_code"]["src"]) if k == "src" else (cfg[k], prov[k])
            for k in ("src", "torch", "triton")}
    if any(x != y for x, y in same.values()):
        raise SystemExit(f"the solves must run collect's solver code, torch and triton: {same}")
    if [c["first"] for c in chunks] != list(range(0, DEALS, CHUNK)):
        raise SystemExit(f"collect has chunks {[c['first'] for c in chunks]}: not all {DEALS} deals")
    for c in chunks:
        ls.need_sha(RUNS / c["jobs"]["file"], c["jobs"]["sha256"])
    kw = {arm: ls.ARMS[arm] for arm in ARMS}
    gpu0 = ls.gpu_state()
    if ls.busy(gpu0):
        raise SystemExit(f"the GPU's other users: {gpu0['peers']} (None: unknown): the solves need it alone")
    rows, t0, total = [], time.time(), sum(c["jobs"]["n"] for c in chunks)
    for c in chunks:
        with open(RUNS / c["jobs"]["file"], "rb") as fh:
            jobs = pickle.load(fh)
        if len(jobs) != c["jobs"]["n"] or any(len(j["board"]) != 5 or int(j["iters"]) != RIVER_ITERS for j in jobs):
            raise SystemExit(f"{c['jobs']['file']}: not the {c['jobs']['n']} river jobs its record says")
        for job in jobs:
            r = dict(board=list(river_key(job["board"])), digest=ls.job_digest(job), pot=ls.pot_of(job),
                     frozen=len(job["frozen"]), nodes=len(job["kind"]), expl={}, seconds={})
            for arm in ARMS:
                r["expl"][arm], r["seconds"][arm] = curve(gf, gs, job, kw[arm])
            rows.append(r)
            if len(rows) % 50 == 0:
                print(f"solve: {len(rows)}/{total} jobs, {time.time() - t0:.0f} s", flush=True)
        del jobs
    # Sorted by board and content hash: nothing depends on the threads' timing during collect.
    rows.sort(key=lambda r: (tuple(r["board"]), r["digest"]))
    boards = [tuple(r["board"]) for r in rows]
    boot = {arm: [ls.board_bootstrap([r["expl"][arm][j] for r in rows], boards) for j in range(len(CHECK))]
            for arm in ARMS}
    for arm in ARMS:
        for j, b in enumerate(boot[arm]):
            b["median"] = float(np.median([r["expl"][arm][j] for r in rows]))
            b["seconds_median"] = float(np.median([r["seconds"][arm][j] for r in rows]))
            b["seconds_p95"] = float(np.percentile([r["seconds"][arm][j] for r in rows], 95))
    read = reading(boot)
    gpu1 = ls.gpu_state()
    if ls.busy(gpu1):
        read["verdict"] = "VOID"
    result = dict(registered="records/river-iters/PREREG.md", **prov, collect_config=cfg,
                  collect_sessions=state["sessions"], reproduces_p2=state.get("reproduces_p2"),
                  jobs_files=[c["jobs"] for c in chunks], arms={arm: kw[arm] for arm in ARMS}, check=list(CHECK),
                  bootstrap=boot, reading=read, distinct_jobs=len({r["digest"] for r in rows}), rows=rows,
                  seconds_note="per checkpoint: hands, plan, construction and steps, evaluations excluded; the first "
                               "job's compile included; not comparable with the live in-lock times",
                  seconds=time.time() - t0, gpu_start=gpu0, gpu_end=gpu1, void=ls.busy(gpu1))
    ls.save(OUT, result, indent=1)
    for arm in ARMS:
        print(f"{arm}: " + ", ".join(f"{n}: {b['mean']:.3f}% [{b['lo']:.3f}, {b['hi']:.3f}]" for n, b in zip(CHECK, boot[arm])))
    e = read["e_live"]
    print(f"RIVER-ITERS: E(dcfr_g2, 200) = {e['mean']:.3f}% of the pot [{e['lo']:.3f}, {e['hi']:.3f}] over "
          f"{e['jobs']} jobs on {e['boards']} boards; candidate {read['candidate']}: {read['verdict']}"
          + ("  (VOID: the GPU was not ours alone at the end)" if result["void"] else ""))
    return 2 if result["void"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    sub.add_parser("solve")
    a = ap.parse_args()
    return {"collect": cmd_collect, "solve": cmd_solve}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
