#!/usr/bin/env python
"""R-MAPSS Step 1 — multiplayer exploitation spine (3p Kuhn), runnable + watchable.

Runs the Step-1 experiment (RMAPSS.md §33): a diverse fixed 3p opponent
population, an independent (marginal) opponent model, a confidence-scheduled
safety-gated deviation, all controlled by the exact N-player best response. Emits
to the telemetry contract (a generic ``kind="experiment"`` run) so it is watchable
on ``scripts/dashboard.py`` alongside the blueprint solve, and prints the ablation
table + gate verdicts.

    .venv/Scripts/python.exe scripts/rmapss_step1.py
    .venv/Scripts/python.exe scripts/dashboard.py --run runs/<the-printed-dir> --port 8770
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import telemetry as T  # noqa: E402
from parciver.rmapss.step1 import run_step1  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-MAPSS Step 1 exploitation spine")
    ap.add_argument("--cfr-iters", type=int, default=8000)
    ap.add_argument("--budget", type=float, default=0.02, help="exploitability budget for the alpha-gate")
    ap.add_argument("--kappa", type=float, default=0.35, help="deviation-signal -> alpha scale")
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    run_dir = args.run_dir or (ROOT / "runs" / T.make_run_id("rmapss-step1"))
    print(f"[rmapss-step1] run dir: {run_dir}", flush=True)

    with T.RunWriter(run_dir) as w:
        w.start(
            run_id=run_dir.name, variant="exploit-spine", game="3p-kuhn",
            config={
                "kind": "experiment",
                "note": "R-MAPSS Step 1: independent model + confidence-scheduled safety-gated deviation",
                "budget": args.budget, "kappa": args.kappa, "cfr_iters": args.cfr_iters,
            },
        )
        t0 = time.time()

        def on_metric(i, pair, r):
            # `exploitability` = Hero's worst-case erosion from deviating (want <= budget):
            # watch it stay under the budget line on the dashboard's exploitability chart.
            w.metrics(
                iteration=i + 1,
                exploitability=max(0.0, r.worstcase_cfr - r.worstcase_final),
                elapsed_s=time.time() - t0,
                ev_a0=r.ev_a0, ev_final=r.ev_final, ev_ceiling=r.ev_ceiling,
                alpha=r.alpha, exploit_gain=r.exploit_gain, klass=r.klass, split=r.split,
            )
            w.milestone("opponent", f"{pair.label} [{r.klass}/{r.split}]: "
                        f"CFR {r.ev_a0:+.3f} -> dev {r.ev_final:+.3f} (ceiling {r.ev_ceiling:+.3f}), "
                        f"alpha={r.alpha:.2f}, gain {r.exploit_gain:+.3f}", iteration=i + 1)

        rep = run_step1(cfr_iters=args.cfr_iters, budget=args.budget, kappa=args.kappa, on_metric=on_metric)

        for name, ok in rep.gates.items():
            w.milestone("gate" if ok else "warning", f"[{'PASS' if ok else 'FAIL'}] {name}")
        w.end(len(rep.results), reason="completed",
              summary={"all_pass": rep.all_pass, **rep.detail})

    # ---- console report ----
    print(f"\neq NashConv (exact) = {rep.eq_nashconv:.5f}\n")
    hdr = f"{'opponent pair':16} {'class':11} {'split':5} {'A0(CFR)':>8} {'A2(indep)':>10} {'final':>8} {'A2.5(ceil)':>11} {'alpha':>6} {'gain':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rep.results:
        print(f"{r.label:16} {r.klass:11} {r.split:5} {r.ev_a0:8.4f} {r.ev_a2:10.4f} "
              f"{r.ev_final:8.4f} {r.ev_ceiling:11.4f} {r.alpha:6.2f} {r.exploit_gain:+8.4f}")
    print("\nAblation (mean EV over class):")
    for klass in ("suboptimal", "equilibrium"):
        rs = [r for r in rep.results if r.klass == klass]
        a0 = sum(r.ev_a0 for r in rs) / len(rs)
        a2 = sum(r.ev_a2 for r in rs) / len(rs)
        fin = sum(r.ev_final for r in rs) / len(rs)
        ceil = sum(r.ev_ceiling for r in rs) / len(rs)
        print(f"  {klass:11}  A0={a0:+.4f}  A1(sched-only)={a0:+.4f}  A2(indep)={a2:+.4f}  "
              f"final={fin:+.4f}  A2.5(ceiling)={ceil:+.4f}")
    print("\nGates:")
    for name, ok in rep.gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\n" + ("ALL PASS" if rep.all_pass else "SOME GATES FAILED"))
    print(f"\nWatch it: .venv/Scripts/python.exe scripts/dashboard.py --run {run_dir} --port 8770")
    return 0 if rep.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
