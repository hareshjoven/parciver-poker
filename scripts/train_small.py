#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Real training telemetry from the small B3 NLHE game — the live convergence proof.

This is the *real-data* producer for the dashboard (the other is
``replay_demo.py``, a full-scale simulator). It drives the actual Rust MCCFR solver
on ``NlheConfig::small`` — the tiny flop-only heads-up game the B3 gate cross-
validates — and emits genuine telemetry to a run log the viewer tails.

Why this game: it is the one NLHE instance where exact best-response
**exploitability** is tractable (``nlhe_small_exploitability``), so the convergence
curve the dashboard plots is ground truth, not a proxy — and the whole thing
converges in seconds, so you can watch a real solver drive exploitability toward
zero live before B4's multi-day blueprint solve exists.

Each point on the curve is an independent MCCFR solve of a growing iteration budget
(the Rust entry point returns the final average strategy, so we resolve at a
geometric schedule of budgets and score each). That makes the exploitability-vs-
iterations curve real and the throughput (it/s) a real measurement of the solver.

Run it, then point the viewer at the printed run directory::

    .venv/Scripts/python.exe scripts/train_small.py
    .venv/Scripts/python.exe scripts/dashboard.py --run <printed run dir>

or let the dashboard launch it for you (``scripts/dashboard.py --demo small``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import parciver_engine as engine  # noqa: E402

from parciver import telemetry as T  # noqa: E402
from parciver.resources import ResourceSampler  # noqa: E402


def geometric_schedule(start: int, stop: int, points: int) -> list[int]:
    """`points` iteration budgets from `start` to `stop`, roughly geometric and
    strictly increasing (so a log-scale x-axis is evenly filled)."""
    if points < 2:
        return [stop]
    ratio = (stop / start) ** (1.0 / (points - 1))
    out: list[int] = []
    v = float(start)
    for _ in range(points):
        iv = int(round(v))
        if not out or iv > out[-1]:
            out.append(iv)
        v *= ratio
    if out[-1] != stop:
        out.append(stop)
    return out


def strategy_l1_delta(prev: dict, cur: dict) -> float | None:
    """Mean per-infoset L1 change of the average strategy between two snapshots,
    over the infosets they share. This is the same average-strategy stability signal
    B4 reports on the full game (there via periodic checkpoints), so the dashboard's
    stability panel is exercised on real data here."""
    if not prev:
        return None
    shared = prev.keys() & cur.keys()
    if not shared:
        return None
    total = 0.0
    for k in shared:
        a, b = prev[k], cur[k]
        m = min(len(a), len(b))
        total += sum(abs(a[i] - b[i]) for i in range(m))
    return total / len(shared)


def infosets_by_street(strategy: dict) -> dict:
    """Count infosets per betting street from the key prefix (``street|card|hist``).
    A real coverage-by-street breakdown for the dashboard, derived from the solver's
    own keys."""
    counts = {"P": 0, "F": 0, "T": 0, "R": 0}
    for k in strategy:
        s = k[:1]
        if s in counts:
            counts[s] += 1
    return {s: c for s, c in counts.items() if c > 0}


def run(run_dir: Path, *, variant: str, seed: int, iters_max: int, points: int,
        min_step_seconds: float) -> None:
    schedule = geometric_schedule(200, iters_max, points)
    config = {
        "hand_names": T.preflop_hand_names(),
        "abstraction": {"preflop": 169, "flop": 10000, "turn": 10000, "river": 1950},
        "stack_bb": 2, "sb": 1, "bb": 2, "deck_size": 7, "board_cards": 3,
        "bet_grid": "1.0pot open", "solver": "external-sampling MCCFR",
        "note": "small B3 cross-validation game — exact exploitability is the hero curve",
    }
    sampler = ResourceSampler(threads=engine_thread_hint())

    with T.RunWriter(run_dir) as w:
        w.start(run_id=run_dir.name, variant=variant, game="nlhe-small",
                config=config, target_iters=iters_max)
        w.milestone("phase_change", f"MCCFR on the small game ({engine.nlhe_small_infoset_count()} infosets), "
                                    f"exact exploitability every step", iteration=0)

        prev_strategy: dict = {}
        first_expl = None
        best_expl = float("inf")
        for iters in schedule:
            t0 = time.perf_counter()
            strategy = engine.solve_nlhe_small_mccfr(iters, variant, seed)
            dt = time.perf_counter() - t0
            expl = engine.nlhe_small_exploitability(strategy)
            it_per_s = iters / dt if dt > 0 else 0.0
            if first_expl is None:
                first_expl = expl
            best_expl = min(best_expl, expl)

            w.metrics(
                iteration=iters,
                it_per_s=it_per_s,
                exploitability=expl,
                infosets=len(strategy),
                strategy_l1_delta=strategy_l1_delta(prev_strategy, strategy),
                resources=sampler.sample(),
            )
            w.checkpoint(iteration=iters, infosets_by_street=infosets_by_street(strategy))
            print(f"[train_small] iters={iters:>9,}  expl={expl:.6f}  "
                  f"{it_per_s:,.0f} it/s  infosets={len(strategy)}", flush=True)
            prev_strategy = strategy

            # Keep the live feel readable even though small solves finish instantly.
            remaining = min_step_seconds - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

        reduction = (first_expl / best_expl) if best_expl > 0 else float("inf")
        w.milestone("checkpoint_saved", f"converged: exploitability {first_expl:.4f} -> {best_expl:.5f} "
                                        f"({reduction:.0f}x)", iteration=iters_max)
        w.end(iteration=iters_max, reason="completed", summary={
            "final_exploitability": best_expl,
            "initial_exploitability": first_expl,
            "reduction_factor": reduction,
            "infosets": engine.nlhe_small_infoset_count(),
            "variant": variant,
        })
    print(f"[train_small] done. run dir: {run_dir}")


def engine_thread_hint() -> int:
    """Best-effort worker-thread count for the resource panel. The Rust MCCFR here
    runs single-threaded per solve, but report the machine's logical cores so the
    panel reads sensibly during B4 (which uses rayon)."""
    import os
    return os.cpu_count() or 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit real dashboard telemetry from the small NLHE game.")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="run directory (default: runs/<auto id>)")
    ap.add_argument("--variant", default="linear", choices=["linear", "mccfr", "mccfr+"],
                    help="MCCFR variant (default: linear = Pluribus-style)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters-max", type=int, default=2_000_000,
                    help="largest MCCFR budget on the curve (default: 2,000,000)")
    ap.add_argument("--points", type=int, default=48,
                    help="number of points on the convergence curve (default: 48)")
    ap.add_argument("--min-step-seconds", type=float, default=0.4,
                    help="floor on wall time per point so the live feed is watchable")
    args = ap.parse_args(argv)

    run_dir = args.run_dir or (ROOT / "runs" / T.make_run_id("small"))
    run_dir = Path(run_dir)
    print(f"[train_small] run dir: {run_dir}")
    run(run_dir, variant=args.variant, seed=args.seed, iters_max=args.iters_max,
        points=args.points, min_step_seconds=args.min_step_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
