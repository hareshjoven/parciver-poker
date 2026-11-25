#!/usr/bin/env python
"""R-MAPSS §35.B — the NO-REGRET-LEARNER decider, runnable + watchable.

The last pre-registered shot at a clean multiplayer edge H2. §35.A left H2 "open": under
dense lagged-BR the rich-bait ceiling revealed large shaping headroom, but it was dominantly
single-opponent (A4−A2.5); the multiplayer slice A5−A4 was certified nonzero yet sub-δ. This
replaces lagged-BR with PRINCIPLED no-regret learners (RM / RM+ / Hedge) adapting to the
JOINT field — learners carry cumulative regret (MEMORY), so a bait persists and the opp→opp
cascade gains a dynamical channel. It applies the pre-registered §35.C gate:

    A5−A4 > δ, held-out, exact-controlled, transferable to >=1 un-fit learner family,
    (Leduc) depth-earns-cost, in Kuhn (FILTER) THEN Leduc (DECISIVE)  ->  GREENLIGHT H2
    otherwise                                                          ->  KILL H2 (ship H1)

STANDING VISUALIZATION (RMAPSS §35.D): auto-launches the dashboard and OPENS THE BROWSER on a
purpose-built experiment view. Panels: the learners' LEARNING TRAJECTORY (aggression + regret
norm + Hero-EV vs exact-BR ceiling + opp→opp coupling per block), the EV-vs-exact-BR frontier,
the pure→+mixed→+coord ablation, the future-exploitability screen, live search progress, and
the A5−A4 (multiplayer) decomposition. ``--no-dashboard`` opts out; ``--no-browser`` keeps the
server without opening a tab.

    .venv/Scripts/python.exe scripts/rmapss_step2b.py --game 3p-kuhn
    .venv/Scripts/python.exe scripts/rmapss_step2b.py --game 3p-leduc --n-pure 150 --n-mixed 150 --leduc-families
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
from parciver.rmapss.step2_bait import BaitConfig  # noqa: E402
from parciver.rmapss.step2_learn import (  # noqa: E402
    run_learn_decider, default_learn_families, leduc_learn_families, leduc_confirm_families,
)

_TIER_CHAR = {"br": "b", "archetype": "a", "pure": "p", "mixed": "m", "coord": "c"}


def _spawn_dashboard(run_dir: Path, port: int):
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"),
           "--run", str(run_dir), "--port", str(port), "--no-browser"]
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        return subprocess.Popen(cmd, cwd=str(ROOT), **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[dashboard] could not spawn: {exc}", flush=True)
        return None


def _downsample_frontier(pts: list[tuple], cap: int = 700) -> list[list]:
    keep = [p for p in pts if p[2] in ("b", "a", "c")]
    cloud = [p for p in pts if p[2] in ("p", "m")]
    budget = max(0, cap - len(keep))
    if len(cloud) > budget and budget > 0:
        step = len(cloud) / budget
        cloud = [cloud[int(i * step)] for i in range(budget)]
    return [[round(im, 6), round(fu, 6), t] for im, fu, t in (keep + cloud)]


def _run_one(game_name: str, args) -> object:
    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id(f"rmapss-step2b-{game_name}")))
    print(f"\n[rmapss-step2b] {game_name} run dir: {run_dir}", flush=True)

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
    if game_name == "3p-leduc" and getattr(args, "confirm", False):
        families = leduc_confirm_families()
    elif game_name == "3p-leduc" and args.leduc_families:
        families = leduc_learn_families()
    else:
        families = default_learn_families()

    with T.RunWriter(run_dir) as w:
        w.start(run_id=run_dir.name, variant="no-regret-learner-decider", game=game_name,
                config={
                    "kind": "experiment", "experiment": "no-regret-learner-decider",
                    "note": "R-MAPSS §35.B: no-regret learners (RM/RM+/Hedge); A5-A4 = the H2 signal",
                    "m": args.m, "delta": args.delta, "depths": list(args.depths),
                    "n_states": args.m, "n_pure": args.n_pure, "n_mixed": args.n_mixed,
                    "coord": not args.no_coord, "cfr_iters": args.cfr_iters,
                    "learner": "RM (fit) + RM+/Hedge (held)",
                })
        t0 = time.time()
        state_counter = {"n": 0}
        frontier_buf: list[tuple] = []
        eval_count = {"n": 0, "best": -1e18, "last_emit": 0}

        def on_eval(fam, g):
            frontier_buf.append((g.immediate, g.future, _TIER_CHAR.get(g.tier, "p")))
            eval_count["n"] += 1
            eval_count["best"] = max(eval_count["best"], g.total)
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
                "family": fam.label, "split": fam.split, "state_ix": t, "delta": args.delta,
                "myopic": [round(sd.myopic_immediate, 6), round(sd.myopic_future, 6)],
                "headroom": round(sd.headroom, 6), "screen_gap": round(sd.screen_gap, 6),
                "lb": [round(sd.lb_pure, 6), round(sd.lb_mixed, 6), round(sd.lb_coord, 6)],
                "n_distinct_succ": sd.n_distinct_succ,
                "frontier": _downsample_frontier(frontier_buf),
            })
            frontier_buf.clear()
            eval_count["last_emit"] = eval_count["n"]

        def on_family(fam, fr):
            w.milestone("opponent",
                        f"{fam.label} [{fam.split}/{fam.rule_split}]: bait headroom "
                        f"{fr.headroom:+.4f} (screen {fr.screen_gap:+.4f}) @state {fr.best_state_ix}")

        rep = run_learn_decider(
            game_name=game_name, m=args.m, depths=tuple(args.depths), delta=args.delta,
            cfr_iters=args.cfr_iters, cfg=cfg, max_baits=args.max_baits,
            do_ceiling=not args.no_ceiling, ctl_horizon=args.ctl_horizon, families=families,
            on_family=on_family, on_state=on_state, on_eval=on_eval)

        # -- emit the learning trajectories (the §35.D headline panel) --
        seat_labels = [f"opp{ i+1 }" for i in range(len(next(iter(rep.trajectories.values()), [{}])[0].get("aggression", [])))] if rep.trajectories else []
        for label, blocks in rep.trajectories.items():
            w.checkpoint(iteration=state_counter["n"] + 1, trajectory={
                "family": label, "seats": seat_labels, "blocks": blocks,
            })
            w.milestone("milestone", f"learning trajectory [{label}]: "
                        f"coupling {blocks[0]['coupling']:+.3f}->{blocks[-1]['coupling']:+.3f} over {len(blocks)} blocks")

        # -- emit the A5-A4 decomposition (reuses the escalation panel) --
        per_family = [{
            "label": f.label, "split": ("held" if (f.split == "held" or f.rule_split == "held") else "fit"),
            "A2.5": round(f.a25, 6), "A4": round(f.a4, 6), "A5": round(f.a5, 6),
            "ceiling": round(f.a25 + f.headroom_lb, 6), "a5_minus_a4": round(f.a5_minus_a4, 6),
        } for f in rep.families]
        w.checkpoint(iteration=state_counter["n"] + 2, escalation={
            "depth": rep.best_depth,
            "mean_headroom_held": round(rep.detail["mean finite-menu headroom lb (held)"], 6),
            "mean_a5_minus_a4_held": round(rep.detail["mean A5-A4 (held)"], 6),
            "per_family": per_family,
        })

        for name, ok in rep.gates.items():
            w.milestone("gate" if ok else "warning", f"[{'PASS' if ok else 'FAIL'}] {name}")
        w.milestone("milestone", f"VERDICT[{rep.role}] = {rep.verdict}")
        w.end(state_counter["n"], reason="completed",
              summary={"verdict": rep.verdict, "role": rep.role, **rep.detail})

    _print_report(game_name, rep, run_dir, exp_url, args.no_dashboard)
    return rep


def _print_report(game_name, rep, run_dir, exp_url, no_dash) -> None:
    print(f"\n=== §35.B no-regret-learner decider · {game_name}  (role {rep.role}, m={rep.m}, "
          f"delta={rep.delta}, depth*={rep.best_depth}) ===")
    hdr = f"{'family':20}{'split':6}{'rule':6}{'A2.5':>9}{'A4':>9}{'A5':>9}{'A5-A4':>9}{'A5-A2.5':>9}"
    print(hdr)
    print("-" * len(hdr))
    for f in rep.families:
        print(f"{f.label:20}{f.split:6}{f.rule_split:6}{f.a25:9.4f}{f.a4:9.4f}{f.a5:9.4f}"
              f"{f.a5_minus_a4:+9.4f}{f.a5_minus_a25:+9.4f}")
    print("\nHeld-out summary:")
    for k, v in rep.detail.items():
        print(f"  {k:56} {v:+.4f}")
    # per-depth breakdown (the confirmation lever: does more depth reveal A5-A4 or shaping?)
    depth_set = sorted({d for f in rep.families for d, _ in f.depths.get("A5", [])})
    if len(depth_set) > 1:
        print("\nPer-depth A5-A2.5 (shaping) / A5-A4 (multiplayer), HELD families:")
        for f in rep.families:
            if f.split != "held":
                continue
            a5d = dict(f.depths["A5"]); a4d = dict(f.depths["A4"])
            cells = "  ".join(f"d{d}: sh={a5d[d]-f.a25:+.4f} mp={a5d[d]-a4d[d]:+.4f}" for d in depth_set)
            print(f"  {f.label:20} {cells}")
    print("\nGates / controls:")
    for name, ok in rep.gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT [{rep.role}]: {rep.verdict}")
    if not no_dash:
        print(f"Watch/review it: {exp_url}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-MAPSS §35.B no-regret-learner decider")
    ap.add_argument("--game", choices=["3p-kuhn", "3p-leduc", "both"], default="3p-kuhn")
    ap.add_argument("--m", type=int, default=4, help="lifted-game horizon (gate regime; §35.B.LOCK=4)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--cfr-iters", type=int, default=4000)
    ap.add_argument("--n-pure", type=int, default=200)
    ap.add_argument("--n-mixed", type=int, default=200)
    ap.add_argument("--no-coord", action="store_true")
    ap.add_argument("--coord-budget", type=int, default=1500)
    ap.add_argument("--coord-max-keys", type=int, default=0)
    ap.add_argument("--max-baits", type=int, default=8, help="cap enriched-menu size (small on Leduc)")
    ap.add_argument("--no-ceiling", action="store_true", help="skip the finite-menu ceiling DP (Leduc)")
    ap.add_argument("--ctl-horizon", type=int, default=None, help="self-consistency control horizon (default min(m,3))")
    ap.add_argument("--leduc-families", action="store_true", help="use the trimmed Leduc family set")
    ap.add_argument("--confirm", action="store_true",
                    help="§37.3 confirmation: Leduc confirm family set (1 fit + 3 held incl un-fit)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    games = ["3p-kuhn", "3p-leduc"] if args.game == "both" else [args.game]
    reps = {}
    base_port, base_run_dir = args.port, args.run_dir
    for i, g in enumerate(games):
        args.port = base_port + i
        args.run_dir = base_run_dir
        if g == "3p-leduc" and args.game == "both":
            args.leduc_families = True
            if args.cfr_iters > 800:
                saved = args.cfr_iters; args.cfr_iters = 600
                reps[g] = _run_one(g, args); args.cfr_iters = saved
                continue
        reps[g] = _run_one(g, args)

    print("\n" + "=" * 62)
    print("R-MAPSS §35.B no-regret-learner decider summary:")
    for g, rep in reps.items():
        print(f"  {g:10} [{rep.role:8}] -> {rep.verdict}   "
              f"(A5-A4 held mean={rep.detail['mean A5-A4 (held)']:+.4f}, "
              f"max={rep.detail['max A5-A4 (held)']:+.4f}, "
              f"un-fit>delta={int(rep.detail['un-fit-learner families A5-A4 > delta'])})")
    print("\nsec.35.C gate reading:")
    print("  GO/FILTER-GO  => A5-A4 > delta held-out, transfers to an un-fit learner; Leduc GO => greenlight H2.")
    print("  KILL/FILTER-FLAT => A5-A4 sub-delta or non-transferable; Leduc KILL => ship the H1 spine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
