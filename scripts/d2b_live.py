# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D2b's whole-bot match (records/d2b/PREREG.md, "Whole-bot safety"): L1 (the live bot with N1) against L0 (the live bot).

  smoke  50 deals of seed 777,006 at 1 and at 4 threads, fresh arms each, each run's arms proven; the per-deal
         results must agree.  -> runs/d2b-live/smoke.json (scratch)
  match  L1 against L0 on the 200bb scorecard's baseline deals 0-9,999 (BP-8B's protocol), chunks of 100.
         -> runs/d2b-live/state.json; at the end records/d2b/live.json
         match --void "cause": PREREG's VOID rule. A match cut by a crash, kill or reboot, or stopped by a failed chunk
         or another compute job on the GPU, is not resumed: this moves its state aside as state-void-<n>.json with the
         cause, and the next `match` repeats it from deal 0. The record lists every voided run.

L0 is the live bot as scripts/slumbot_run.py builds it; L1 the same with Gate A's N1 (its sha256 in
records/d2b/gate-a.json) in place of L0's net. Both play the live blueprint, each with its own search and net. Built
as LIVESEARCH's arms, both `snap`; L1 is the engine's A, so the result is L1 - L0. DOWN iff the raw 95% interval's
upper end is < 0. Refused before Gate B has passed for N1. A chunk that fails is recorded in the state and stops the
match; a rerun refuses it. Another process of this Python on the GPU stops it the same way; another program's compute
job shows only in each chunk's `apps`, read there. A proof's error names the arms as livesearch does: B1 is L1, B0 is
L0. From main, the code committed:
    .venv/Scripts/python.exe -I scripts/d2b_live.py smoke
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
import bp8b_live as bp8  # noqa: E402

REC = ROOT / "records" / "d2b"
RUNS = ROOT / "runs" / "d2b-live"
STATE = RUNS / "state.json"
NETS = ROOT / "runs" / "turn-net"
CODE = ls.PLAY_CODE + ("scripts/d2b_live.py", "scripts/bp8b_live.py")
DEALS, CHUNK, THREADS, BLOCKS = bp8.DEALS, bp8.CHUNK, bp8.THREADS, bp8.BLOCKS
SMOKE_SEED, SMOKE_DEALS = 777_006, 50


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


def nets(sr) -> dict:
    """L0's net (the live runner's, N0 by Gate A's record) and L1's (N1), each by path and sha256; refused unless
    Gate A and Gate B passed for N1, and if the runner already plays N1 (no L0 to test against)."""
    g = json.loads((REC / "gate-a.json").read_text())
    b = json.loads((REC / "m1-report.json").read_text())
    if not g.get("proceed"):
        raise SystemExit("records/d2b/gate-a.json: Gate A has not passed")
    if not (b.get("final") and b.get("non_inferior")) or b.get("sha256") != g["sha256"]:
        raise SystemExit("records/d2b/m1-report.json: Gate B has not passed for Gate A's nets")
    l1_path = NETS / g["n1"] / "net.pt"
    l1 = ls.need_sha(l1_path, g["sha256"]["N1"])
    l0 = ls.sha256(sr.NET)
    if l0 == l1:
        raise SystemExit(f"the live runner already plays N1 ({sr.NET}): no L0 to match it against")
    if l0 != g["sha256"]["N0"]:
        raise SystemExit(f"the live runner's net {sr.NET} is not the N0 Gate A compared with")
    return dict(l0=dict(path=str(sr.NET), sha256=l0), l1=dict(path=str(l1_path), sha256=l1))


def build_arms(threads: int, pinned: dict):
    """L0 and L1: LIVESEARCH's arm with the live search and the live blueprint, each with its own net."""
    import parciver_engine as pe

    sr = ls.runner()
    bp = pe.SavedBlueprint(str(sr.ABSTRACTION), str(sr.BLUEPRINT))
    if Path(bp.source_path()).resolve() != Path(sr.BLUEPRINT).resolve():
        raise SystemExit(f"the blueprint loaded from {bp.source_path()}, not {sr.BLUEPRINT}")
    l0 = ls.Arm("L0", sr, bp, sr.SEARCH["turn_iters"], sr.TURN_SCHEDULE, threads, net=pinned["l0"]["path"])
    l1 = ls.Arm("L1", sr, bp, sr.SEARCH["turn_iters"], sr.TURN_SCHEDULE, threads, net=pinned["l1"]["path"])
    return sr, pe, l0, l1, dict(infosets=int(bp.num_infosets()), grid=bp.grid)


def live_blueprint(sr) -> None:
    """Both arms play the live blueprint, which is v2-8B (the registration's)."""
    ls.need_sha(Path(sr.BLUEPRINT), bp8.V2_8B_SHA)


# ---------------------------------------------------------------- smoke


def cmd_smoke(a) -> int:
    from parciver import scorecard as sc

    prov = provenance()
    sr = ls.runner()
    m, _ = bp8.benchmark(sr)
    live_blueprint(sr)
    pinned = nets(sr)
    runs, engine, loaded = {}, None, None
    for threads in (1, THREADS):
        # Fresh arms for each run: an agent's search cache would answer the replayed deals from the first run.
        _, pe, l0, l1, loaded = build_arms(THREADS, pinned)
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
    ls.save(RUNS / "smoke.json", dict(**prov, engine=engine, nets=pinned, **loaded, seed=SMOKE_SEED,
                                      deals=SMOKE_DEALS, threads=[1, THREADS], identical=same, runs=runs))
    print(f"smoke: 1 and {THREADS} threads {'identical deal for deal' if same else 'DIFFER'}")
    return 0 if same else 1


def smoke_passed(prov: dict, engine: str, pinned: dict, threads: int) -> bool:
    """A smoke run on this play code, engine and both nets comes before the match at either thread count; 4 threads
    also need it identical deal for deal."""
    path = RUNS / "smoke.json"
    if not path.exists():
        return False
    s = json.loads(path.read_text(encoding="utf-8"))
    return (s.get("threads") == [1, THREADS] and s.get("play_code") == prov["play_code"] and s.get("engine") == engine
            and s.get("nets") == pinned and (threads == 1 or s.get("identical") is True))


# ---------------------------------------------------------------- the match


def config(threads: int, sr, pe, m: dict, row: dict, prov: dict, pinned: dict) -> dict:
    """What a state on disk must match: the deals, the arms' settings, the inputs' sha256, the engine build, the play
    code's git objects, torch and triton (not the commit: a docs commit must not block writing the record)."""
    from parciver import scorecard as sc

    return dict(benchmark=m["benchmark_id"], manifest_hash=m["manifest_hash"], row="baseline", seed=row["seed"],
                deals=DEALS, hands_per_deal=m["hands_per_deal"], chunk=CHUNK, threads=threads, blocks=BLOCKS,
                aivat=True, nets=pinned, blueprint=dict(path=str(sr.BLUEPRINT), sha256=ls.sha256(sr.BLUEPRINT)),
                abstraction=str(sr.ABSTRACTION), leaves=sr.LEAVES, flop_gamma=sr.FLOP_GAMMA,
                terminals64=sr.TERMINALS64, translation="snap", turn_schedule=dict(sr.TURN_SCHEDULE),
                search=dict(sr.SEARCH), engine=sc.sha256_file(pe.parciver_engine.__file__),
                play_code=prov["play_code"], torch=prov["torch"], triton=prov["triton"])


def voided() -> list[dict]:
    """The runs voided so far: their cause, and how far each got."""
    out = []
    for path in sorted(RUNS.glob("state-void-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        s = json.loads(path.read_text(encoding="utf-8"))
        out.append(dict(file=path.name, cause=s["void"]["cause"], at=s["void"]["at"], failed=s.get("failed"),
                        deals=len(s.get("run", {}).get("per_deal_bb", [])), sessions=s.get("sessions", [])))
    return out


def void_run(cause: str) -> int:
    """PREREG, VOID and bugs: the unfinished match moved aside, its cause in it; the next `match` repeats it."""
    if not cause.strip():
        raise SystemExit("--void needs a cause")
    if not STATE.exists():
        raise SystemExit(f"{STATE}: no match to void")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if "summary" in state or len(state.get("run", {}).get("per_deal_bb", [])) >= DEALS:
        raise SystemExit("this match played all its deals: --void does not move it. A compute job found in its lists "
                         "voids it through a further amendment (PREREG, Amendment 1), and its result stays in the record")
    g = ls.gpu_state()
    if ls.busy(g) or not g["apps"]:  # no apps: nvidia-smi's list failed, since Windows lists every desktop app
        raise SystemExit("another process of this Python is on the GPU, or nvidia-smi could not say: stop it first, it may be this match")
    state["void"] = dict(cause=cause, at=ls.utc())
    path = RUNS / f"state-void-{len(voided()) + 1}.json"
    ls.fresh(path)
    ls.save(path, state)
    STATE.unlink()
    print(f"VOID: {len(state.get('run', {}).get('per_deal_bb', []))} deals moved aside to {path}: {cause}")
    return 0


def cmd_match(a) -> int:
    import parciver_engine as pe
    from parciver import scorecard as sc

    if a.void is not None:
        return void_run(a.void)
    ls.fresh(REC / "live.json")
    prov = provenance()
    if a.threads not in (1, THREADS):
        raise SystemExit(f"the match runs at {THREADS} threads, or at 1 if the smoke run's differ")
    sr = ls.runner()
    m, row = bp8.benchmark(sr)
    live_blueprint(sr)
    pinned = nets(sr)
    if not STATE.exists() and not smoke_passed(prov, sc.sha256_file(pe.parciver_engine.__file__), pinned, a.threads):
        raise SystemExit(f"no smoke run on this code, engine and nets"
                         f"{'' if a.threads == 1 else f' shows 1 and {THREADS} threads identical'}: run `smoke` first"
                         + ("" if a.threads == 1 else ", or the match at --threads 1 if its runs differ"))
    cfg = config(a.threads, sr, pe, m, row, prov, pinned)
    state = ls.resume(STATE, cfg)
    if state.get("failed"):
        raise SystemExit(f"the match stopped at a failed chunk, reported and not retried: {state['failed']}. "
                         f"It is VOID: `match --void \"<cause>\"`, then `match` repeats it")
    played = len(state.get("run", {}).get("per_deal_bb", []))
    if played < DEALS and "run" in state:
        raise SystemExit(f"this match was cut at deal {played} of {DEALS} and is not resumed: it is VOID (PREREG, VOID "
                         f"and bugs). `match --void \"<cause>\"`, then `match` repeats it")
    if "smoke_sha256" not in state:
        state["smoke_sha256"] = ls.sha256(RUNS / "smoke.json")
    run = state.setdefault("run", dict(per_deal_bb=[], per_deal_bb_aivat=[], aivat_mass_error=0.0, a_sources=[0] * 4,
                                       b_sources=[0] * 4, seconds=0.0, chunks=[]))
    hpd = m["hands_per_deal"]
    if len(run["per_deal_bb"]) < DEALS:
        state.setdefault("sessions", []).append(dict(started=ls.utc(), commit=prov["commit"]))
        ls.save(STATE, state)  # from here a stop leaves a state, so even one in the first chunk is seen as a cut
        _, _, l0, l1, loaded = build_arms(a.threads, pinned)
        # The nets and the blueprint are read into memory: hashed again once loaded, so a file replaced meanwhile is caught.
        try:
            now = nets(sr)
        except SystemExit as e:
            now = str(e)
        why = (f"the nets changed while loading: {now}, pinned {pinned}" if now != pinned else
               "the blueprint changed while loading" if ls.sha256(sr.BLUEPRINT) != cfg["blueprint"]["sha256"] else
               f"the loaded blueprint is {loaded}, earlier sessions' {state['loaded']}"
               if state.setdefault("loaded", loaded) != loaded else None)
        if why:
            state["failed"] = dict(first=len(run["per_deal_bb"]), error=why, at=ls.utc())
            ls.save(STATE, state)
            raise SystemExit(why)
        clients = set()

        def new_clients(g, first):  # as they appear: a finished match cannot be moved by --void
            new = sorted(set(g["apps"]) - clients)
            if new:
                print(f"match: GPU clients new around deal {first} (a compute job voids the match, read them): {new}", flush=True)
                clients.update(new)

        while len(run["per_deal_bb"]) < DEALS:
            first = len(run["per_deal_bb"])
            g0 = ls.gpu_state()
            new_clients(g0, first)
            try:
                res, info = ls.play(l0, l1, ls.stakes_of(m), row["seed"], first, CHUNK, a.threads)
                if info.pop("jobs"):
                    raise SystemExit(f"deals {first}..{first + CHUNK - 1}: jobs kept with no tap set. Nothing saved.")
            except BaseException as e:
                # Recorded in the state, so a rerun cannot retry the chunk and hide it. A Ctrl-C is a cut, not a failure.
                if not isinstance(e, KeyboardInterrupt):
                    state["failed"] = dict(first=first, error=f"{type(e).__name__}: {e}", at=ls.utc(), gpu_start=g0)
                    ls.save(STATE, state)
                raise
            g1 = ls.gpu_state()
            new_clients(g1, first)
            peers = sorted({*(g0["peers"] or []), *(g1["peers"] or [])})
            if peers:
                state["failed"] = dict(first=first, error=f"another process of this Python on the GPU ({peers}): VOID", at=ls.utc(),
                                   gpu_start=g0, gpu_end=g1)
                ls.save(STATE, state)
                raise SystemExit(state["failed"]["error"])
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


def finish(state: dict, hpd: int) -> int:
    run = state["run"]
    if len(run["per_deal_bb"]) != DEALS or len(run["per_deal_bb_aivat"]) != DEALS:
        raise SystemExit(f"{len(run['per_deal_bb'])} deals played, the match is {DEALS}")
    summary = bp8.summary_of(run, hpd)
    state["summary"] = summary
    apps = sorted({x for c in run["chunks"] for g in (c["gpu_start"], c["gpu_end"]) for x in g["apps"]})
    ls.save(STATE, state)
    ls.fresh(REC / "live.json")
    record = dict(registered="records/d2b/PREREG.md, Whole-bot safety", config=state["config"],
                  sessions=state["sessions"], voided=voided(), gpu_apps=apps, smoke_sha256=state.get("smoke_sha256"), loaded=state.get("loaded"),
                  failures=state.get("failures", []), code_history=state.get("code_history", []),
                  summary=summary, chunks=[{k: v for k, v in c.items() if k != "times"} for c in run["chunks"]],
                  per_deal_bb=run["per_deal_bb"], per_deal_bb_aivat=run["per_deal_bb_aivat"])
    ls.save(REC / "live.json", record)
    s, sa = summary, summary["aivat"]
    print(f"D2b live: L1 - L0 = {s['mean']:+.2f} bb/100, 95% [{s['lo']:+.2f}, {s['hi']:+.2f}] over {DEALS} deals: "
          f"{'DOWN' if s['down'] else 'not DOWN'}; AIVAT (reported) {sa['mean']:+.2f} [{sa['lo']:+.2f}, {sa['hi']:+.2f}]")
    looks = [g for c in run["chunks"] for g in (c["gpu_start"], c["gpu_end"])]
    print(f"GPU clients seen (desktop apps do not void the match, a compute job does: read them): {apps}; looks that "
          f"could not say: {sum(ls.busy(g) or not g['apps'] for g in looks)} of {len(looks)}; voided matches: {len(voided())}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("match")
    p.add_argument("--threads", type=int, default=THREADS, help="4 as registered; 1 if the smoke run's differ")
    p.add_argument("--void", metavar="CAUSE", help="the unfinished match is VOID, for this cause: move it aside")
    a = ap.parse_args()
    return {"smoke": cmd_smoke, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
