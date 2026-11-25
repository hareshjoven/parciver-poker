#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Launch the live training dashboard viewer.

The viewer is a local, no-build-step web app that tails a run's append-only
telemetry log and renders it (convergence, the 13x13 preflop range grid, throughput
and coverage, resources, head-to-head, and a completion banner). It is decoupled
from training: it only reads ``runs/<id>/events.jsonl``, so it can never slow or lag
a solve. This is a LOCAL app, not a hosted page — Artifacts are sandboxed and
cannot read local live telemetry.

Usage::

    # Watch a specific run (e.g. the B4 blueprint solve, once it is running):
    scripts/dashboard.py --run runs/<id>

    # Watch the most recent run:
    scripts/dashboard.py --latest

    # Develop/marvel before B4 exists — launch a producer AND the viewer together:
    scripts/dashboard.py --demo small   # real data: MCCFR on the small B3 game
    scripts/dashboard.py --demo full    # simulated full-scale 100bb run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import telemetry as T  # noqa: E402
from parciver.dashboard import server as dash_server  # noqa: E402

RUNS_DIR = ROOT / "runs"


def latest_run() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "events.jsonl").exists()]
    if not runs:
        return None
    return max(runs, key=lambda d: (d / "events.jsonl").stat().st_mtime)


_PRODUCERS = {"small": "train_small.py", "full": "replay_demo.py", "blueprint": "train_blueprint.py"}


def launch_producer(kind: str, run_dir: Path, extra: list[str]) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / "scripts" / _PRODUCERS[kind]), "--run-dir", str(run_dir), *extra]
    if kind == "blueprint":
        cmd.append("--no-dashboard")  # this viewer is the dashboard; the trainer's own would double it
    print(f"[dashboard] launching producer: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live training dashboard viewer.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", type=Path, help="run directory to watch")
    g.add_argument("--latest", action="store_true", help="watch the most recent run")
    g.add_argument("--demo", choices=["small", "full", "blueprint"],
                   help="launch a producer and watch it: small=real MCCFR on the B3 game, "
                        "full=simulated full-scale run, blueprint=the REAL B4 solve (extra args, e.g. "
                        "--grid 200bb, go to the trainer)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    # Any unrecognized args are forwarded verbatim to the --demo producer
    # (e.g. `--fast`, `--duration 60`, `--iters-max 5000000`).
    args, producer_args = ap.parse_known_args(argv)

    producer: subprocess.Popen | None = None
    if args.demo:
        prefix = {"small": "small", "full": "demo", "blueprint": "blueprint"}[args.demo]
        run_dir = RUNS_DIR / T.make_run_id(prefix)
        run_dir.mkdir(parents=True, exist_ok=True)
        producer = launch_producer(args.demo, run_dir, producer_args)
        # The blueprint loads ~277MB of tables before its first event; give it a
        # moment so the browser has run_start to render on connect.
        time.sleep(2.0 if args.demo == "blueprint" else 0.3)
    elif args.run:
        run_dir = args.run
    elif args.latest:
        run_dir = latest_run()
        if run_dir is None:
            print("no runs found under runs/. Try --demo small or --demo full.", file=sys.stderr)
            return 1
    else:
        run_dir = latest_run()
        if run_dir is None:
            print("No run selected and none found. Use --run <dir>, --latest, or --demo {small,full}.",
                  file=sys.stderr)
            return 2
        print(f"[dashboard] no run given; watching latest: {run_dir}")

    try:
        dash_server.serve(run_dir, host=args.host, port=args.port,
                          open_browser=not args.no_browser, block=True)
    finally:
        if producer is not None and producer.poll() is None:
            print("[dashboard] stopping producer…")
            producer.terminate()
            try:
                producer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                producer.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
