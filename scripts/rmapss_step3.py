#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""R-MAPSS Step 3 / S3.1 — the HU exploitation SPINE on real reduced NLHE, runnable + watchable.

Ships + scales the validated H1 spine (RMAPSS.md §40) from toy 3p Kuhn onto a real
(reduced) heads-up NLHE game — `NlheConfig::small` (456 infosets, flop-only, 2 bb
stacks) — where exact best-response exploitability is computable and certified (B3),
so the safety control is EXACT (not an abstraction proxy). Per fixed opponent it
observes the Hero-visible action marginal, best-responds to it, and deviates from
the CFR-equilibrium anchor toward that BR by the confidence-scheduled, safety-gated
α = min(α_conf, α_safe). It applies exploratory regression gates (§41):

    beats the anchor vs HELD-OUT observably-distorted opponents (> δ_sub),
    matches the anchor vs near-eq (|gain| ≤ tol),
    matches the historical hole-blind model boundary (§41.DEV1),
    keeps EXACT worst-case exploitability within an S-scaled budget on EVERY opp,
    and never lets the observed-model BR exceed the exact BR ceiling (§28)
    -> PASS exploratory regression (NOT a confirmation or permission to scale)

STANDING VISUALIZATION (RMAPSS §40.8): auto-launches the dashboard and OPENS THE
BROWSER on a purpose-built experiment view. Panels: per-opponent EV-vs-exact-BR
frontier, the α schedule (α_conf vs α_safe), the safety-budget headroom
(exploitability margin), and the realized edge vs the field. ``--no-dashboard``
opts out; ``--no-browser`` keeps the server without opening a tab.

    .venv/Scripts/python.exe scripts/rmapss_step3.py                 # 2M anchor (headline)
    .venv/Scripts/python.exe scripts/rmapss_step3.py --anchor-iters 300000 --coarse
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from copy import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import telemetry as T  # noqa: E402
from parciver.rmapss import step3  # noqa: E402

_COARSE = (1.0, 0.7, 0.45, 0.25, 0.1, 0.0)


def _spawn_dashboard(run_dir: Path, port: int):
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"),
           "--run", str(run_dir), "--port", str(port), "--no-browser"]
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        return subprocess.Popen(cmd, cwd=str(ROOT), **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[dashboard] could not spawn: {exc}", flush=True)
        return None


def _row_dict(res, base_expl: float) -> dict:
    return {
        "label": res.label, "klass": res.klass, "split": res.split,
        "ev_a0": round(res.ev_a0, 6), "ev_ceiling": round(res.ev_ceiling, 6),
        "ev_a2": round(res.ev_a2, 6), "ev_final": round(res.ev_final, 6),
        "exploit_gain": round(res.exploit_gain, 6), "ceiling_gain": round(res.ceiling_gain, 6),
        "marginal_gap": round(res.marginal_gap, 6),
        "alpha": round(res.alpha, 4), "alpha_conf": round(res.alpha_conf, 4),
        "alpha_safe": round(res.alpha_safe, 4), "signal": round(res.signal, 5),
        "expl_final": round(res.expl_final, 6), "dexpl": round(res.expl_final - base_expl, 6),
        "safety_ok": bool(res.safety_ok),
    }


def run(args) -> object:
    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id("rmapss-step3")))
    print(f"\n[rmapss-step3] run dir: {run_dir}", flush=True)

    exp_url = f"http://127.0.0.1:{args.port}/static/experiment3.html"
    if not args.no_dashboard:
        run_dir.mkdir(parents=True, exist_ok=True)
        _spawn_dashboard(run_dir, args.port)
        print(f"[dashboard] {exp_url}  (watching {run_dir.name})", flush=True)
        if not args.no_browser:
            try:
                webbrowser.open(exp_url)
            except Exception:
                pass

    grid = _COARSE if args.coarse else step3.ALPHA_GRID
    spec, c_sub, c_tol, c_bud = step3.protocol_settings(args.protocol)
    with T.RunWriter(run_dir) as w:
        w.start(run_id=run_dir.name, variant="hu-exploitation-spine", game="nlhe_small",
                config={
                    "kind": "experiment", "experiment": "hu-exploitation-spine",
                    "note": "R-MAPSS S3.1 (§40/§41): confidence-scheduled, safety-gated single-"
                            "opponent exploitation on real reduced NLHE; exact BR ceiling + exact "
                            "exploitability are the §28 controls.",
                    "kappa": step3.KAPPA, "anchor_iters": args.anchor_iters,
                    "c_sub": c_sub, "c_tol": c_tol, "c_bud": c_bud,
                    "protocol": args.protocol, "calibration_split": "fit",
                    "anchor_seed": args.seed, "alpha_grid": list(grid),
                    "population": [list(row) for row in spec],
                    "infosets": 456,
                })
        t0 = time.time()
        base_expl_holder = {"v": 0.0}
        prog = {"n": 0}
        n_total = len(spec)

        def on_thresholds(rep):
            base_expl_holder["v"] = rep.base_expl
            w.checkpoint(iteration=0, step3_meta={
                "game": "nlhe_small", "scale_S": round(rep.scale_S, 6),
                "base_expl": round(rep.base_expl, 6), "delta_sub": round(rep.delta_sub, 6),
                "tol_eq": round(rep.tol_eq, 6), "budget": round(rep.budget, 6),
                "kappa": step3.KAPPA, "anchor_iters": args.anchor_iters,
            })
            w.milestone("phase_change",
                        f"scale S={rep.scale_S:.3f} -> delta_sub={rep.delta_sub:.4f} / "
                        f"tol={rep.tol_eq:.4f} / budget={rep.budget:.4f} (anchor expl {rep.base_expl:.4f})")

        def on_metric(i, opp, res):
            prog["n"] += 1
            be = base_expl_holder["v"]
            w.metrics(iteration=i + 1, elapsed_s=time.time() - t0,
                      progress=prog["n"] / n_total, opponent=res.label,
                      exploit_gain=res.exploit_gain, alpha=res.alpha)
            w.checkpoint(iteration=i + 1, step3_row=_row_dict(res, be))
            w.milestone("opponent",
                        f"{res.label} [{res.klass}/{res.split}]: gain {res.exploit_gain:+.3f} "
                        f"(ceiling {res.ceiling_gain:+.3f}, alpha {res.alpha:.2f}, "
                        f"dexpl {res.expl_final - be:+.4f})")

        rep = step3.run_step3(anchor_iters=args.anchor_iters, anchor_seed=args.seed,
                              on_thresholds=on_thresholds, on_metric=on_metric,
                              alpha_grid=grid, protocol=args.protocol)

        # -- the authoritative final payload (gates + verdict + full table) --
        rows = [_row_dict(r, rep.base_expl) for r in rep.results]
        w.checkpoint(iteration=n_total + 1, step3={
            "game": "nlhe_small", "verdict": rep.verdict, "all_pass": rep.all_pass,
            "protocol": rep.protocol, "calibration_split": "fit",
            "scale_S": round(rep.scale_S, 6), "base_expl": round(rep.base_expl, 6),
            "delta_sub": round(rep.delta_sub, 6), "tol_eq": round(rep.tol_eq, 6),
            "budget": round(rep.budget, 6), "kappa": step3.KAPPA,
            "gates": {k: bool(v) for k, v in rep.gates.items()},
            "detail": {k: round(v, 6) for k, v in rep.detail.items()},
            "rows": rows,
        })
        for name, ok in rep.gates.items():
            w.milestone("gate" if ok else "warning", f"[{'PASS' if ok else 'FAIL'}] {name}")
        w.milestone("milestone", f"VERDICT = {rep.verdict}")
        w.end(n_total + 1, reason="completed",
              summary={"verdict": rep.verdict, "all_pass": rep.all_pass,
                       "protocol": rep.protocol, **rep.detail})

    _print_report(rep, run_dir, exp_url, args.no_dashboard)
    return rep


def _print_report(rep, run_dir, exp_url, no_dash) -> None:
    print(f"\n=== R-MAPSS S3.1 · HU exploitation spine on nlhe_small "
          f"(S={rep.scale_S:.3f}, anchor expl={rep.base_expl:.4f}) ===")
    print(f"thresholds: delta_sub={rep.delta_sub:.4f}  tol={rep.tol_eq:.4f}  budget={rep.budget:.4f}")
    hdr = (f"{'opponent':16}{'class':6}{'split':6}{'ev_a0':>8}{'ceil':>8}{'ev_a2':>8}"
           f"{'final':>8}{'gain':>8}{'a':>6}{'aconf':>6}{'asafe':>6}{'sig':>6}{'dexpl':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rep.results:
        print(f"{r.label:16}{r.klass[:5]:6}{r.split:6}{r.ev_a0:8.3f}{r.ev_ceiling:8.3f}"
              f"{r.ev_a2:8.3f}{r.ev_final:8.3f}{r.exploit_gain:+8.3f}{r.alpha:6.2f}"
              f"{r.alpha_conf:6.2f}{r.alpha_safe:6.2f}{r.signal:6.2f}"
              f"{r.expl_final - rep.base_expl:+8.4f}")
    print("\nDetail:")
    for k, v in rep.detail.items():
        print(f"  {k:52} {v:+.4f}")
    print("\nGates / controls:")
    for name, ok in rep.gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT: {rep.verdict}")
    if not no_dash:
        print(f"Watch/review it: {exp_url}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-MAPSS S3.1 HU exploitation spine on nlhe_small")
    ap.add_argument("--anchor-iters", type=int, default=2_000_000,
                    help="MCCFR iterations for the CFR-equilibrium anchor (headline: 2e6)")
    ap.add_argument("--coarse", action="store_true",
                    help="use the coarse alpha grid (faster; the gate-audit setting)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--protocol", choices=(step3.PROTOCOL, step3.CONFIRMATION_PROTOCOL),
                    default=step3.PROTOCOL)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8774)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    if args.protocol == step3.CONFIRMATION_PROTOCOL:
        if args.seed is not None or args.coarse or args.anchor_iters != step3.CONFIRMATION_ITERS:
            ap.error("confirmation-v1 freezes all three seeds, 2M iterations and the full alpha grid")
        return run_confirmation(args)
    args.seed = 0 if args.seed is None else args.seed

    rep = run(args)
    print("\n" + "=" * 62)
    print("R-MAPSS S3.1 summary:")
    print(f"  nlhe_small [{'PASS' if rep.all_pass else 'FAIL'}] -> {rep.verdict}")
    print(f"  mean held-out exploit gain = {rep.detail['mean exploit gain (suboptimal HELD-OUT)']:+.4f}  "
          f"(S={rep.scale_S:.3f}); max eq |gain| = {rep.detail['max |exploit gain| (equilibrium)']:.4f}")
    print("  Exploratory regression only; a frozen confirmation protocol is required before S3.2.")
    return 0 if rep.all_pass else 1


def run_confirmation(args) -> int:
    """Run every locked seed, recording failures without changing the protocol."""
    parent = Path(args.run_dir or (ROOT / "runs" / T.make_run_id("rmapss-step3-confirmation")))
    reports = []
    with T.RunWriter(parent) as writer:
        writer.start(run_id=parent.name, variant="hu-confirmation", game="nlhe_small",
                     config={"kind": "experiment", "protocol": step3.CONFIRMATION_PROTOCOL,
                             "seeds": list(step3.CONFIRMATION_SEEDS),
                             "anchor_iters": step3.CONFIRMATION_ITERS,
                             "fractions": [0.10, 0.05, 0.10], "calibration_split": "fit",
                             "kappa": step3.KAPPA, "alpha_grid": list(step3.ALPHA_GRID),
                             "population": [list(row) for row in step3.CONFIRMATION_SPEC]})
        for index, seed in enumerate(step3.CONFIRMATION_SEEDS):
            per_seed = copy(args)
            per_seed.seed = seed
            per_seed.port = args.port + index
            per_seed.run_dir = parent / f"seed-{seed}"
            rep = run(per_seed)
            reports.append(rep)
            writer.checkpoint(iteration=index + 1, confirmation_seed={
                "seed": seed, "run_dir": str(per_seed.run_dir), "all_pass": rep.all_pass,
                "verdict": rep.verdict, "scale_S": rep.scale_S,
                "gates": rep.gates, "detail": rep.detail})
        passed = len(reports) == len(step3.CONFIRMATION_SEEDS) and all(r.all_pass for r in reports)
        verdict = "PASS (narrow S3.1 confirmation)" if passed else "FAIL (confirmation gate breach)"
        writer.end(len(reports), reason="completed", summary={
            "protocol": step3.CONFIRMATION_PROTOCOL, "all_pass": passed, "verdict": verdict,
            "seeds": list(step3.CONFIRMATION_SEEDS)})
    print(f"\nCONFIRMATION: {verdict}\nAggregate evidence: {parent}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
