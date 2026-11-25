#!/usr/bin/env python
"""R-MAPSS §35.A — the RICH-BAIT DECIDER, runnable + watchable.

Settles whether §34.5's "H2 not supported" was a MENU ARTIFACT. §34.5's ρ-ceiling
was computed over a 5-archetype Hero menu — only a lower bound on the true optimal
exploitation. This runs a genuinely RICH bait search (large pure + mixed sample +
coordinate ascent) on the 1-step-ahead Stackelberg objective, recomputes a much
tighter ceiling, and applies the pre-registered §35.A gate:

    rich ceiling STILL == A2.5 (within δ) in Kuhn AND Leduc  ->  the null is REAL
    headroom appears                                          ->  MENU-ARTIFACT, and
        decompose single-opponent (star) vs the multiplayer A5-A4 gap (the H2 signal)

STANDING VISUALIZATION REQUIREMENT (RMAPSS §35.D): this auto-launches the dashboard
and OPENS THE BROWSER on a purpose-built experiment view (headroom-vs-δ, the
EV-vs-exact-BR frontier, the ablation compare, and the future-exploitability screen).
``--no-dashboard`` opts out; ``--no-browser`` keeps the server but does not open a tab.

    .venv/Scripts/python.exe scripts/rmapss_step2a.py --game 3p-kuhn
    .venv/Scripts/python.exe scripts/rmapss_step2a.py --game 3p-leduc --n-pure 800 --n-mixed 800
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import telemetry as T  # noqa: E402
from parciver.rmapss.step2_bait import run_decider, BaitConfig  # noqa: E402

_TIER_CHAR = {"br": "b", "archetype": "a", "pure": "p", "mixed": "m", "coord": "c"}


def _spawn_dashboard(run_dir: Path, port: int):
    """Start scripts/dashboard.py as a detached process watching ``run_dir`` (so it
    survives this runner and stays up for post-run review), and open the browser on
    the experiment view. Returns the Popen (or None)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"),
           "--run", str(run_dir), "--port", str(port), "--no-browser"]
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # detach from Ctrl-C
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[dashboard] could not spawn: {exc}", flush=True)
        return None
    return proc


def _downsample_frontier(pts: list[tuple], cap: int = 700) -> list[list]:
    """Keep every br/archetype/coord point (few, decision-relevant) + a uniform
    sample of the pure/mixed cloud, so the emitted scatter stays small."""
    keep = [p for p in pts if p[2] in ("b", "a", "c")]
    cloud = [p for p in pts if p[2] in ("p", "m")]
    budget = max(0, cap - len(keep))
    if len(cloud) > budget and budget > 0:
        step = len(cloud) / budget
        cloud = [cloud[int(i * step)] for i in range(budget)]
    out = keep + cloud
    return [[round(im, 6), round(fu, 6), t] for im, fu, t in out]


def _run_one(game_name: str, args) -> object:
    run_dir = args.run_dir or (ROOT / "runs" / T.make_run_id(f"rmapss-step2a-{game_name}"))
    run_dir = Path(run_dir)
    print(f"\n[rmapss-step2a] {game_name} run dir: {run_dir}", flush=True)

    exp_url = f"http://127.0.0.1:{args.port}/static/experiment.html"
    if not args.no_dashboard:
        run_dir.mkdir(parents=True, exist_ok=True)
        _spawn_dashboard(run_dir, args.port)
        print(f"[dashboard] {exp_url}  (watching {run_dir.name})", flush=True)
        if not args.no_browser:
            try:
                webbrowser.open(exp_url)
            except Exception:
                pass

    cfg = BaitConfig(n_pure=args.n_pure, n_mixed=args.n_mixed, coord=not args.no_coord,
                     coord_budget=args.coord_budget, coord_max_keys=args.coord_max_keys,
                     seed=args.seed)

    with T.RunWriter(run_dir) as w:
        w.start(run_id=run_dir.name, variant="rich-bait-decider", game=game_name,
                config={
                    "kind": "experiment", "experiment": "rich-bait-decider",
                    "note": "R-MAPSS §35.A: rich-bait 1-step Stackelberg ceiling vs myopic BR",
                    "m": args.m, "delta": args.delta, "depth": args.depth,
                    "n_states": args.n_states, "n_pure": args.n_pure, "n_mixed": args.n_mixed,
                    "coord": not args.no_coord, "cfr_iters": args.cfr_iters,
                })
        t0 = time.time()
        state_counter = {"n": 0}
        frontier_buf: list[tuple] = []
        eval_count = {"n": 0, "best": -1e18, "last_emit": 0}

        def on_eval(fam, g):
            frontier_buf.append((g.immediate, g.future, _TIER_CHAR.get(g.tier, "p")))
            eval_count["n"] += 1
            if g.total > eval_count["best"]:
                eval_count["best"] = g.total
            # throttle a lightweight search-progress line so long Leduc states stay live
            if eval_count["n"] - eval_count["last_emit"] >= 250:
                eval_count["last_emit"] = eval_count["n"]
                w.metrics(iteration=state_counter["n"], elapsed_s=time.time() - t0,
                          candidates_evaluated=eval_count["n"], search_best_total=eval_count["best"],
                          progress=1, family=fam.label)

        def on_state(fam, t, sd):
            state_counter["n"] += 1
            it = state_counter["n"]
            w.metrics(
                iteration=it, elapsed_s=time.time() - t0,
                headroom=sd.headroom, screen_gap=sd.screen_gap,
                lb_pure=sd.lb_pure, lb_mixed=sd.lb_mixed, lb_coord=sd.lb_coord,
                best_total=sd.best_total, best_immediate=sd.best_immediate, best_future=sd.best_future,
                myopic_val=sd.myopic_val, myopic_immediate=sd.myopic_immediate,
                myopic_future=sd.myopic_future, n_distinct_succ=sd.n_distinct_succ,
                n_candidates=sd.n_candidates, delta=args.delta,
                family=fam.label, split=fam.split, state_ix=t,
            )
            w.checkpoint(iteration=it, experiment={
                "family": fam.label, "split": fam.split, "state_ix": t,
                "delta": args.delta,
                "myopic": [round(sd.myopic_immediate, 6), round(sd.myopic_future, 6)],
                "headroom": round(sd.headroom, 6), "screen_gap": round(sd.screen_gap, 6),
                "lb": [round(sd.lb_pure, 6), round(sd.lb_mixed, 6), round(sd.lb_coord, 6)],
                "n_distinct_succ": sd.n_distinct_succ,
                "frontier": _downsample_frontier(frontier_buf),
            })
            frontier_buf.clear()
            eval_count["last_emit"] = eval_count["n"]

        def on_family(fam, fr):
            w.milestone(
                "opponent",
                f"{fr.label} [{fr.split}]: headroom {fr.headroom:+.4f} "
                f"(screen {fr.screen_gap:+.4f}) @state {fr.best_state_ix}; "
                f"controls imm={fr.control_immediate_ok} seed={fr.control_seed_ok}")

        rep = run_decider(game_name=game_name, m=args.m, delta=args.delta,
                          cfr_iters=args.cfr_iters, depth=args.depth, n_states=args.n_states,
                          cfg=cfg, do_escalate=not args.no_escalate, max_baits=args.max_baits,
                          on_family=on_family, on_state=on_state, on_eval=on_eval)

        for name, ok in rep.gates.items():
            w.milestone("gate" if ok else "warning", f"[{'PASS' if ok else 'FAIL'}] {name}")
        if rep.escalation is not None:
            e = rep.escalation
            w.milestone("milestone",
                        f"ESCALATION (rich menu, depth {e.depth}): mean headroom(held) "
                        f"{e.mean_headroom_held:+.4f}, mean A5-A4(held) {e.mean_a5_minus_a4_held:+.4f} "
                        f"[multiplayer signal]; controls_ok={e.controls_ok}")
            w.checkpoint(iteration=state_counter["n"] + 1, escalation={
                "depth": e.depth,
                "mean_headroom_held": round(e.mean_headroom_held, 6),
                "mean_a5_minus_a4_held": round(e.mean_a5_minus_a4_held, 6),
                "per_family": [{k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in r.items()} for r in e.per_family],
            })
        w.milestone("milestone", f"VERDICT[{rep.role}] = {rep.verdict}")
        w.end(state_counter["n"], reason="completed",
              summary={"verdict": rep.verdict, "role": rep.role, **rep.detail})

    _print_report(game_name, rep, run_dir, exp_url, args.no_dashboard)
    return rep


def _print_report(game_name, rep, run_dir, exp_url, no_dash) -> None:
    print(f"\n=== §35.A rich-bait decider · {game_name}  (role {rep.role}, m={rep.m}, "
          f"delta={rep.delta}, states={rep.n_states}, "
          f"n_pure={rep.cfg.n_pure}, n_mixed={rep.cfg.n_mixed}, coord={rep.cfg.coord}) ===")
    hdr = (f"{'family':16}{'split':6}{'headroom':>10}{'screen_gap':>11}"
           f"{'lb_pure':>9}{'lb_mixed':>9}{'lb_coord':>9}{'@state':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rep.results:
        print(f"{r.label:16}{r.split:6}{r.headroom:10.4f}{r.screen_gap:11.4f}"
              f"{max(s.lb_pure for s in r.per_state):9.4f}{max(s.lb_mixed for s in r.per_state):9.4f}"
              f"{max(s.lb_coord for s in r.per_state):9.4f}{r.best_state_ix:7d}")
    print("\nHeld-out summary:")
    for k, v in rep.detail.items():
        print(f"  {k:46} {v:.4f}")
    print("\nGates / controls:")
    for name, ok in rep.gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if rep.escalation is not None:
        e = rep.escalation
        print(f"\nESCALATION (headroom found -> rich menu, depth {e.depth}):")
        print(f"  mean headroom (held):    {e.mean_headroom_held:+.4f}")
        print(f"  mean A5-A4 (held):       {e.mean_a5_minus_a4_held:+.4f}   <- the MULTIPLAYER (H2) signal")
        print(f"  controls_ok:             {e.controls_ok}")
        eh = (f"{'family':16}{'split':6}{'A2.5':>9}{'A4':>9}{'A5':>9}{'ceil':>9}{'A5-A4':>9}")
        print("  " + eh)
        for r in e.per_family:
            print(f"  {r['label']:16}{r['split']:6}{r['A2.5']:9.4f}{r['A4']:9.4f}{r['A5']:9.4f}"
                  f"{r['ceiling']:9.4f}{r['a5_minus_a4']:+9.4f}")
    print(f"\nVERDICT [{rep.role}]: {rep.verdict}")
    if not no_dash:
        print(f"Watch/review it: {exp_url}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-MAPSS §35.A rich-bait decider")
    ap.add_argument("--game", choices=["3p-kuhn", "3p-leduc", "both"], default="3p-kuhn")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--depth", type=int, default=3, help="cascade depth for the escalation A4/A5")
    ap.add_argument("--n-states", type=int, default=None, help="myopic-trajectory states to probe (default m)")
    ap.add_argument("--cfr-iters", type=int, default=6000)
    ap.add_argument("--n-pure", type=int, default=400)
    ap.add_argument("--n-mixed", type=int, default=400)
    ap.add_argument("--no-coord", action="store_true", help="disable coordinate ascent")
    ap.add_argument("--coord-budget", type=int, default=4000)
    ap.add_argument("--coord-max-keys", type=int, default=0, help="0=all reached Hero infosets; else top-N by reach")
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--max-baits", type=int, default=24, help="cap enriched-menu size for the escalation DP (small on Leduc)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    games = ["3p-kuhn", "3p-leduc"] if args.game == "both" else [args.game]
    reps = {}
    base_port = args.port
    base_run_dir = args.run_dir
    for i, g in enumerate(games):
        args.port = base_port + i          # a distinct dashboard per game for --game both
        args.run_dir = base_run_dir        # None -> auto per game
        # Leduc traversals are ~1000x Kuhn's; a lighter CFR reference suffices there.
        if g == "3p-leduc" and args.cfr_iters > 800 and args.game == "both":
            saved = args.cfr_iters
            args.cfr_iters = 500
            reps[g] = _run_one(g, args)
            args.cfr_iters = saved
        else:
            reps[g] = _run_one(g, args)

    print("\n" + "=" * 62)
    print("R-MAPSS §35.A rich-bait decider summary:")
    for g, rep in reps.items():
        extra = ""
        if rep.escalation is not None:
            extra = f"  (rich A5-A4 held = {rep.escalation.mean_a5_minus_a4_held:+.4f})"
        print(f"  {g:10} [{rep.role:8}] -> {rep.verdict}{extra}")
    print("\n§35.A gate reading:")
    print("  NULL-REAL in Kuhn AND Leduc  => the §34.5 null is genuine (not a menu artifact).")
    print("  MENU-ARTIFACT (headroom)     => decompose; A5-A4>0 on held-out Leduc is the H2 signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
