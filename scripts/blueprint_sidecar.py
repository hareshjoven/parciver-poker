#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Blueprint dashboard sidecar — fills the action-frequency, bet-sizing, and
head-to-head panels WITHOUT touching the running solve.

The trainer periodically saves the average strategy to ``runs/<id>/blueprint.bin``.
This sidecar is a separate, read-only process: on a timer it reads that snapshot,
runs the Rust ``BlueprintAnalyzer`` (Monte-Carlo self-play for per-street action /
bet-size mixes, plus a head-to-head match vs a calling-station baseline for bb/100),
and APPENDS the results as telemetry-contract events to a SEPARATE
``runs/<id>/panels.jsonl``.

Two deliberate isolation choices:
  * it never writes ``events.jsonl`` (the trainer owns that) — a second writer on one
    append-only log risks interleaved/torn lines, so the sidecar uses its own file
    and the dashboard merges the two streams;
  * it reads ``blueprint.bin`` (a file the trainer flushed), never the trainer's live
    memory, so it cannot slow or perturb the solve. The analysis is single-threaded
    and only re-runs when ``blueprint.bin`` actually changes, so it steals ~1 core for
    ~a second every few minutes.

Because a fresh engine build cannot overwrite the ``.pyd`` the trainer has loaded and
locked on Windows, ``--engine-path`` prepends a scratch install of the new engine to
``sys.path`` so THIS process imports the new ``BlueprintAnalyzer`` while the trainer
keeps running its already-loaded copy.

    python scripts/blueprint_sidecar.py --latest
    python scripts/blueprint_sidecar.py --run-dir runs/blueprint-... --interval 60
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest_run_dir(runs: Path) -> Path | None:
    cands = [p for p in runs.glob("blueprint-*") if p.is_dir()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


class PanelWriter:
    """Minimal append-only writer for ``panels.jsonl`` — the sidecar's own log,
    kept separate from the trainer's ``events.jsonl``. Validates every event against
    the shared telemetry contract before writing, so the dashboard can consume it
    identically."""

    def __init__(self, path: Path, validate) -> None:
        self.path = path
        self._validate = validate
        self._seq = 0

    def emit(self, type_: str, payload: dict) -> None:
        event = {"v": 1, "type": type_, "seq": self._seq, "ts": time.time()}
        for k in ("v", "type", "seq", "ts"):
            payload.pop(k, None)
        event.update(payload)
        self._validate(event)  # fail loudly rather than emit garbage
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
        self._seq += 1


def run(run_dir: Path, *, abstraction_dir: Path, interval: float,
        self_play_hands: int, hh_hands: int, seed: int) -> None:
    import parciver_engine as engine  # noqa: E402  (may come from --engine-path)
    from parciver import telemetry as T  # noqa: E402

    if not hasattr(engine, "BlueprintAnalyzer"):
        raise SystemExit(
            "[sidecar] this parciver_engine has no BlueprintAnalyzer — build the new "
            "engine and pass --engine-path <scratch install>, or `maturin develop` it "
            "once the trainer is stopped."
        )

    print(f"[sidecar] loading abstraction from {abstraction_dir} ...", flush=True)
    analyzer = engine.BlueprintAnalyzer(str(abstraction_dir))
    print("[sidecar] ready.", flush=True)

    blueprint_path = run_dir / "blueprint.bin"
    panels = PanelWriter(run_dir / "panels.jsonl", T.validate_event)
    # Read-only tail of the trainer's log, purely to stamp our panel events with the
    # run's current iteration (the x-axis the dashboard aligns on).
    ev_reader = T.RunReader(run_dir)

    def _refresh_iteration(prev: int | None) -> int | None:
        cur = prev
        try:
            for ev in ev_reader.read_new(validate=False):
                if ev.get("type") in ("metrics", "checkpoint") and isinstance(ev.get("iteration"), int):
                    cur = ev["iteration"]
        except (json.JSONDecodeError, ValueError):
            pass
        return cur

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True
        print("\n[sidecar] stop requested; exiting after this cycle.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except Exception:
            pass

    last_mtime = 0.0
    cur_iter = _refresh_iteration(None)
    print(f"[sidecar] watching {run_dir.name}; panels -> panels.jsonl every {interval:.0f}s", flush=True)

    while not stop["flag"]:
        cur_iter = _refresh_iteration(cur_iter)

        if blueprint_path.exists():
            mtime = blueprint_path.stat().st_mtime
            # Only recompute when the trainer has actually written a new snapshot.
            if mtime != last_mtime and cur_iter is not None:
                last_mtime = mtime
                t0 = time.perf_counter()
                res = analyzer.analyze(str(blueprint_path), self_play_hands, hh_hands, seed)
                dt = time.perf_counter() - t0
                panels.emit("checkpoint", {
                    "iteration": int(cur_iter),
                    "action_freq": res["action_freq"],
                    "bet_size_freq": res["bet_size_freq"],
                })
                panels.emit("metrics", {
                    "iteration": int(cur_iter),
                    "hh_bb_per_100": res["hh_bb_per_100"],
                })
                print(f"[sidecar] iter={cur_iter:,} hh={res['hh_bb_per_100']:+.1f} bb/100 "
                      f"(analyzed in {dt:.2f}s)", flush=True)

        # Sleep in small slices so a stop signal is responsive.
        slept = 0.0
        while slept < interval and not stop["flag"]:
            time.sleep(min(0.5, interval - slept))
            slept += 0.5


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Blueprint dashboard sidecar (fills the 3 idle panels).")
    ap.add_argument("--run-dir", type=Path, default=None, help="run dir to attach to")
    ap.add_argument("--latest", action="store_true", help="attach to the newest runs/blueprint-* dir")
    ap.add_argument("--abstraction-dir", type=Path, default=ROOT / "data" / "abstraction")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between panel updates")
    ap.add_argument("--self-play-hands", type=int, default=20000)
    ap.add_argument("--hh-hands", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-path", type=Path, default=None,
                    help="prepend this dir to sys.path so a freshly-built engine is "
                         "imported without overwriting the trainer's loaded .pyd")
    args = ap.parse_args(argv)

    if args.engine_path is not None:
        sys.path.insert(0, str(args.engine_path))
    sys.path.insert(0, str(ROOT / "src"))

    run_dir = args.run_dir
    if run_dir is None and args.latest:
        run_dir = _latest_run_dir(ROOT / "runs")
    if run_dir is None:
        raise SystemExit("[sidecar] specify --run-dir <dir> or --latest")
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise SystemExit(f"[sidecar] run dir does not exist: {run_dir}")

    run(run_dir, abstraction_dir=args.abstraction_dir, interval=args.interval,
        self_play_hands=args.self_play_hands, hh_hands=args.hh_hands, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
