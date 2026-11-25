#!/usr/bin/env python
"""R-MAPSS Step 2 — the inter-opponent interaction KERNEL, runnable + watchable.

Runs the Step-2 experiment (RMAPSS.md §34): adaptive 3p opponents under the locked
rule ρ (dense lagged-BR to all seats), a shared Hero menu, and the depth-ablated
kernel (A2.5 myopic / A4 star-model cascade / A5 dense-interaction-graph cascade)
against the exact ρ-ceiling and a valid upper bound. Emits to the telemetry contract
(``kind="experiment"``) so it is watchable on ``scripts/dashboard.py`` next to the
blueprint solve, and prints the ablation table + the pre-registered go/no-go verdict.

Kuhn is the within-round FILTER (a NO there is not a kill, R3); 3p Leduc is DECISIVE.

    .venv/Scripts/python.exe scripts/rmapss_step2.py --game both
    .venv/Scripts/python.exe scripts/dashboard.py --run runs/<printed-dir> --port 8771
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import telemetry as T  # noqa: E402
from parciver.rmapss.step2 import run_step2  # noqa: E402


def _run_one(game_name: str, args) -> "object":
    run_dir = args.run_dir or (ROOT / "runs" / T.make_run_id(f"rmapss-step2-{game_name}"))
    print(f"\n[rmapss-step2] {game_name} run dir: {run_dir}", flush=True)

    with T.RunWriter(run_dir) as w:
        w.start(
            run_id=run_dir.name, variant="interaction-kernel", game=game_name,
            config={
                "kind": "experiment",
                "note": "R-MAPSS Step 2: adaptive opponents (dense lagged-BR) + depth-ablated interaction kernel",
                "m": args.m, "depths": list(args.depths), "delta": args.delta,
                "cfr_iters": args.cfr_iters,
            },
        )
        t0 = time.time()

        def on_family(i, fam, fr, a25, a4, a5, ceil):
            # EV-vs-exact-BR frontier + ablation compare, per family (dashboard panels):
            #  * ceiling gap  = how much of ρ's exploitation is on the table
            #  * A5-A2.5      = shaping edge vs the myopic baseline
            #  * A5-A4        = the inter-opponent (multiplayer) edge — the bet
            w.metrics(
                iteration=i + 1,
                exploitability=max(0.0, fr.upper_bound - fr.ev["A5"]),  # slack under the hard bound
                elapsed_s=time.time() - t0,
                ev_a25=fr.ev["A2.5"], ev_a4=fr.ev["A4"], ev_a5=fr.ev["A5"],
                ev_ceiling=fr.ceiling, upper_bound=fr.upper_bound,
                a5_minus_a25=fr.a5_minus_a25, a5_minus_a4=fr.a5_minus_a4,
                headroom=fr.headroom, split=fr.split, family=fr.label,
            )
            w.milestone(
                "opponent",
                f"{fam.label} [{fr.split}]: A2.5 {fr.ev['A2.5']:+.3f} -> A5 {fr.ev['A5']:+.3f} "
                f"(ceil {fr.ceiling:+.3f}); A5-A2.5 {fr.a5_minus_a25:+.3f}, "
                f"A5-A4 {fr.a5_minus_a4:+.3f}", iteration=i + 1)

        rep = run_step2(game_name=game_name, m=args.m, depths=tuple(args.depths),
                        delta=args.delta, cfr_iters=args.cfr_iters, on_family=on_family)

        for name, ok in rep.gates.items():
            w.milestone("gate" if ok else "warning", f"[{'PASS' if ok else 'FAIL'}] {name}")
        w.milestone("milestone", f"VERDICT[{rep.role}] = {rep.verdict}")
        w.end(len(rep.results), reason="completed",
              summary={"verdict": rep.verdict, "role": rep.role,
                       "selected_depth": rep.best_depth, **rep.detail})

    _print_report(game_name, rep, run_dir)
    return rep


def _print_report(game_name, rep, run_dir) -> None:
    print(f"\n=== {game_name}  (role: {rep.role}, m={rep.m}, "
          f"selected planning depth={rep.best_depth}, delta={rep.delta}) ===")
    hdr = (f"{'family':16}{'split':6}{'A2.5':>9}{'A4':>9}{'A5':>9}"
           f"{'ceil':>9}{'UB':>9}{'A5-A25':>9}{'A5-A4':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rep.results:
        print(f"{r.label:16}{r.split:6}{r.ev['A2.5']:9.4f}{r.ev['A4']:9.4f}{r.ev['A5']:9.4f}"
              f"{r.ceiling:9.4f}{r.upper_bound:9.4f}{r.a5_minus_a25:+9.4f}{r.a5_minus_a4:+9.4f}")
    print("\nHeld-out summary:")
    for k, v in rep.detail.items():
        print(f"  {k:42} {v:.4f}")
    print("\nGates / controls:")
    for name, ok in rep.gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT [{rep.role}]: {rep.verdict}")
    print(f"Watch it: .venv/Scripts/python.exe scripts/dashboard.py --run {run_dir} --port 8771")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-MAPSS Step 2 interaction kernel")
    ap.add_argument("--game", choices=["3p-kuhn", "3p-leduc", "both"], default="both")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--cfr-iters", type=int, default=6000)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    games = ["3p-kuhn", "3p-leduc"] if args.game == "both" else [args.game]
    # Leduc traversals are ~1000x Kuhn's; a smaller CFR reference suffices there.
    reps = {}
    for g in games:
        if g == "3p-leduc" and args.cfr_iters > 800 and args.game == "both":
            saved = args.cfr_iters
            args.cfr_iters = 500
            reps[g] = _run_one(g, args)
            args.cfr_iters = saved
        else:
            reps[g] = _run_one(g, args)

    print("\n" + "=" * 60)
    print("R-MAPSS Step 2 summary (pre-registered §34 go/no-go):")
    for g, rep in reps.items():
        print(f"  {g:10} [{rep.role:8}] -> {rep.verdict}")
    decisive = reps.get("3p-leduc")
    if decisive is not None:
        print(f"\nDECISIVE (Leduc) verdict on hypothesis H2: {decisive.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
