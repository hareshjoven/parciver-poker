# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""R-MAPSS Step Q — fragility-as-transferable-EV-predictor runner (RMAPSS.md §38).

Runs the pre-registered H-Q experiment: does a CHEAP, best-response-free fragility metric
of an approximate-equilibrium opponent PREDICT the exact extractable EV out-of-sample,
beyond the trivial distance-from-eq baseline? 3p Kuhn is the FILTER, 3p Leduc the DECISIVE
game. Gated primary target = the exact ceiling ``Y_ceil`` (§38.DEV1, user-locked); the α=1
realized transferable capture is reported alongside. Gate (§38.4): a cheap metric (not
``dist_eq``) with |ρ_held| ≥ ρ* AND beating ``dist_eq`` by ≥ Δρ AND sign-stable (fit/held/
transfer), in BOTH games. ρ*=0.70, Δρ=0.15.

STANDING VISUALIZATION (RMAPSS §38.7 / §35.D / memory parciver-auto-dashboard-viz):
auto-launches the dashboard and OPENS THE BROWSER on a purpose-built fragility view
(``/static/experimentQ.html``): the fragility→EV scatter (held highlighted), ρ_held bars vs
the baseline + the ρ* line, the ceiling-vs-realized-capture gap, the cheap-beats-expensive
panel, and the live per-opponent table. ``--no-dashboard`` opts out; ``--no-browser`` keeps
the server without a tab (NOT used by default — the standing rule forbids it).

The 3p Leduc eq is slow in pure Python, so it is CACHED (runs/_eqcache/) and reused.

    .venv/Scripts/python.exe scripts/rmapss_stepQ.py                 # both games
    .venv/Scripts/python.exe scripts/rmapss_stepQ.py --game 3p-kuhn  # filter only
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):          # Windows console is cp1252; our prints use →/§
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from parciver import telemetry as T  # noqa: E402
from parciver.games.kuhn_multi import KuhnPokerN  # noqa: E402
from parciver.games.leduc_multi import LeducHoldemN  # noqa: E402
from parciver.rmapss.stepQ import (  # noqa: E402
    CHEAP_METRICS, BASELINE_METRIC, run_stepQ, solve_eq,
)

# Leduc CFR+ is slow (pure Python); 16000 iters (via FastTree.solve_cfr, cached) reaches
# NashConv ~0.038 — clean enough for a control-valid decisive run (§38.DEV2 F1).
GAMES = {"3p-kuhn": (KuhnPokerN, 20000), "3p-leduc": (LeducHoldemN, 16000)}
EQ_CACHE = ROOT / "runs" / "_eqcache"


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


def _cached_eq(game, game_name: str, cfr_iters: int, w=None):
    EQ_CACHE.mkdir(parents=True, exist_ok=True)
    key = EQ_CACHE / f"{game_name}-cfr{cfr_iters}.pkl"
    if key.is_file():
        print(f"[eq] cache hit {key.name}", flush=True)
        with open(key, "rb") as fh:
            return pickle.load(fh)
    print(f"[eq] solving {game_name} eq ({cfr_iters} CFR+ iters — slow on Leduc)…", flush=True)
    if w is not None:
        w.milestone("phase_change", f"solving {game_name} equilibrium reference ({cfr_iters} CFR+ iters)…")
    t = time.time()
    eq = solve_eq(game, cfr_iters)
    print(f"[eq] solved in {time.time() - t:.1f}s", flush=True)
    with open(key, "wb") as fh:
        pickle.dump(eq, fh)
    return eq


def _stepq_payload(rep) -> dict:
    metrics = []
    base_abs = abs(rep.metric_rho[BASELINE_METRIC].rho_held)
    for name, m in rep.metric_rho.items():
        metrics.append({
            "name": name, "is_baseline": (name == BASELINE_METRIC), "is_ref": name.endswith("*"),
            "rho_fit": round(m.rho_fit, 4), "rho_held": round(m.rho_held, 4),
            "rho_transfer": round(m.rho_transfer, 4), "rho_cap_held": round(m.rho_cap_held, 4),
        })
    rows = [{
        "label": r.label, "split": r.split, "klass": r.klass,
        "y_ceil": round(r.y_ceil, 5), "y_cap": round(r.y_cap, 5),
        "m": {k: round(v, 5) for k, v in r.metrics.items()},
    } for r in rep.rows]
    return {
        "game": rep.game_name, "baseline": BASELINE_METRIC,
        "eq_nashconv": round(rep.eq_nashconv, 5), "perfect_rho": round(rep.control_rho_perfect_held, 4),
        "alpha_star": round(rep.alpha_star, 4), "alpha_safe": round(rep.alpha_safe, 4),
        "baseline_abs_rho_held": round(base_abs, 4),
        "metrics": metrics, "rows": rows,
        "verdict": rep.verdict, "gates": {k: bool(v) for k, v in rep.gates.items()},
        "n_held": sum(1 for r in rep.rows if r.split == "held" and r.klass != "eq"),
    }


def _print_report(rep) -> None:
    print(f"\n=== §38 H-Q fragility→EV predictor · {rep.game_name}  (eq NashConv {rep.eq_nashconv:.4f}) ===")
    hdr = f"{'metric':16}{'rho_fit':>9}{'rho_held':>9}{'rho_tr':>8}{'rho_cap':>8}"
    print(hdr + "   note")
    print("-" * (len(hdr) + 8))
    for name, m in rep.metric_rho.items():
        note = "BASELINE" if name == BASELINE_METRIC else ("reference (expensive)" if name.endswith("*") else "")
        print(f"{name:16}{m.rho_fit:+9.3f}{m.rho_held:+9.3f}{m.rho_transfer:+8.2f}{m.rho_cap_held:+8.2f}   {note}")
    print("\nGates:")
    for k, v in rep.gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"VERDICT[{rep.game_name}] = {rep.verdict}")


def _combined_verdict(reps: dict, rho_star: float, delta_rho: float) -> tuple[str, list[str]]:
    """H-Q GO iff a cheap metric passes in BOTH games with a CONSISTENT sign (§38.4)."""
    per_game_pass = {}
    for gname, rep in reps.items():
        base = abs(rep.metric_rho[BASELINE_METRIC].rho_held)
        winners = {}
        for name, m in rep.metric_rho.items():
            if name == BASELINE_METRIC or name.endswith("*"):
                continue
            import math
            sstable = (m.rho_fit == 0 or m.rho_held == 0 or
                       math.copysign(1, m.rho_fit) == math.copysign(1, m.rho_held))
            tstable = (m.rho_transfer == 0 or m.rho_held == 0 or
                       math.copysign(1, m.rho_transfer) == math.copysign(1, m.rho_held))
            if (abs(m.rho_held) >= rho_star and abs(m.rho_held) - base >= delta_rho
                    and sstable and tstable):
                winners[name] = m.rho_held
        per_game_pass[gname] = winners
    if len(reps) < 2:
        only = next(iter(per_game_pass.values()))
        return ("FILTER-" + ("PASS" if only else "FLAT"), list(only))
    import math
    common = set.intersection(*[set(w) for w in per_game_pass.values()]) if per_game_pass else set()
    consistent = [n for n in common
                  if len({math.copysign(1, per_game_pass[g][n]) for g in per_game_pass}) == 1]
    return (("GO" if consistent else "KILL"), consistent)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R-MAPSS §38 H-Q fragility→EV-predictor runner.")
    ap.add_argument("--game", choices=["3p-kuhn", "3p-leduc", "both"], default="both")
    ap.add_argument("--cfr-iters", type=int, default=0, help="override per-game default")
    ap.add_argument("--pert-k", type=int, default=8)
    ap.add_argument("--rho-star", type=float, default=0.70)
    ap.add_argument("--delta-rho", type=float, default=0.15)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    games = ["3p-kuhn", "3p-leduc"] if args.game == "both" else [args.game]
    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id("rmapss-stepQ")))
    run_dir.mkdir(parents=True, exist_ok=True)
    exp_url = f"http://127.0.0.1:{args.port}/static/experimentQ.html"
    print(f"[rmapss-stepQ] run dir: {run_dir}", flush=True)

    if not args.no_dashboard:
        _spawn_dashboard(run_dir, args.port)
        print(f"[dashboard] {exp_url}  (watching {run_dir.name})", flush=True)
        if not args.no_browser:
            try:
                webbrowser.open(exp_url)
            except Exception:
                pass

    reps: dict = {}
    with T.RunWriter(run_dir) as w:
        w.start(run_id=run_dir.name, variant="fragility-ev-predictor", game=("+".join(games)),
                config={
                    "kind": "experiment", "experiment": "fragility-ev-predictor",
                    "note": "R-MAPSS §38 H-Q: cheap BR-free fragility predicts exact extractable EV",
                    "rho_star": args.rho_star, "delta_rho": args.delta_rho,
                    "baseline": BASELINE_METRIC, "target": "Y_ceil (gated) + Y_cap α=1 (reported)",
                    "metrics": list(CHEAP_METRICS) + ["nashconv_opp*"],
                    "games": games,
                })
        t0 = time.time()
        it = 0
        for gname in games:
            cls, default_iters = GAMES[gname]
            cfr_iters = args.cfr_iters or default_iters
            game = cls(3)
            eq = _cached_eq(game, gname, cfr_iters, w)
            w.milestone("phase_change", f"{gname}: computing metrics + exact targets over the population…")
            rep = run_stepQ(game, gname, cfr_iters=cfr_iters, rho_star=args.rho_star,
                            delta_rho=args.delta_rho, pert_k=args.pert_k, eq=eq)
            reps[gname] = rep
            it += 1
            w.metrics(iteration=it, elapsed_s=time.time() - t0)
            w.checkpoint(iteration=it, stepq=_stepq_payload(rep))
            for k, ok in rep.gates.items():
                w.milestone("gate" if ok else "warning", f"[{gname}] [{'PASS' if ok else 'FAIL'}] {k}")
            w.milestone("milestone", f"VERDICT[{gname}] = {rep.verdict}")
            _print_report(rep)

        verdict, winners = _combined_verdict(reps, args.rho_star, args.delta_rho)
        it += 1
        w.checkpoint(iteration=it, stepq_final={
            "verdict": verdict, "winners": winners,
            "games": list(reps), "rho_star": args.rho_star, "delta_rho": args.delta_rho,
        })
        w.milestone("milestone", f"H-Q {'FILTER' if len(reps) < 2 else 'DECISIVE'} VERDICT = {verdict}"
                    + (f"  ({', '.join(winners)})" if winners else ""))
        w.end(it, reason="completed", summary={"verdict": verdict, "winners": winners})

    print(f"\n{'='*70}\nH-Q VERDICT ({'both games' if len(reps) == 2 else games[0]}) = {verdict}"
          + (f"   winners: {', '.join(winners)}" if winners else "") + f"\n{'='*70}")
    if not args.no_dashboard:
        print(f"[dashboard] view: {exp_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
