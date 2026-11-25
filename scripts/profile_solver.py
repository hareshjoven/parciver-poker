# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Profile the full-game MCCFR response solver (the S3.4 downstream hot path).

This is a DEV/measurement tool, not part of any locked protocol. It times
`S32Policy.fixed_response` — the external-sampling best-response solve that
dominates S3.4 downstream cost — across thread counts, reporting throughput
(updates/s), parallel speedup and efficiency. The scaling curve tells us whether
the solver is lock/bandwidth bound (efficiency falls off) or compute bound
(near-linear), which decides whether "more cores", "tighten the CPU hot loop", or
"GPU redesign" is the right lever.

DO NOT run while a locked research run is in flight: a solve saturates all cores
and would both slow that run and skew these numbers. Run it only when the machine
is otherwise idle (i.e., after the S3.4 run releases the engine).

Usage:
    .venv/Scripts/python.exe scripts/profile_solver.py            # default sweep
    .venv/Scripts/python.exe scripts/profile_solver.py --iters 500000 --threads 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="profile full-game fixed_response scaling")
    parser.add_argument("--iters", type=int, default=300_000,
                        help="external-sampling updates per solve (default 300k; "
                             "the locked S3.4 budget is 5,000,000)")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--family", default="nit")
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=90_001)
    parser.add_argument("--repeats", type=int, default=1,
                        help="solves per thread count; reports the best (least noisy) time")
    args = parser.parse_args(argv)

    import parciver_engine as pe
    from parciver.rmapss import step34

    step34.verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(step34.ABSTRACTION_DIR), str(step34.BLUEPRINT))
    anchor = saved.s32_policy()
    opponent = anchor.leak(args.family, args.strength)

    print(f"full-game fixed_response | opp={args.family}({args.strength}) seat={args.seat} "
          f"iters={args.iters:,} | best of {args.repeats}")
    print(f"{'threads':>8} {'time_s':>10} {'updates/s':>14} {'speedup':>9} {'efficiency':>11}")

    baseline = None
    for threads in args.threads:
        best = float("inf")
        for _ in range(args.repeats):
            started = time.perf_counter()
            opponent.fixed_response(args.seat, args.iters, "linear", args.seed, threads)
            best = min(best, time.perf_counter() - started)
        ups = args.iters / best
        if baseline is None:
            baseline = best
        speedup = baseline / best
        eff = speedup / threads
        print(f"{threads:>8} {best:>10.3f} {ups:>14,.0f} {speedup:>8.2f}x {eff*100:>10.1f}%")

    print("\nReading it: efficiency near 100% => compute-bound (more cores scale ~linearly).")
    print("Efficiency dropping as threads rise => lock-contention / memory-bandwidth bound")
    print("(then tightening the CPU hot loop — fewer lock acquisitions, one key build per")
    print("node, interned infoset ids — buys more than raw cores).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
