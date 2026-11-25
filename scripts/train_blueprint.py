#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""B4 — the real 100bb heads-up blueprint solve, emitting live telemetry.

This runs external-sampling MCCFR on the FULL 52-card heads-up game
(`NlheConfig::pluribus_100bb`) keyed on the Phase-3b bucket abstraction (preflop 169
/ flop 10000 / turn 10000 / river 1950), via the Rust `parciver_engine.Blueprint`
solver. It is the blueprint the whole project has been building toward.

Exact exploitability is intractable on this game (that is exactly why B3 built the
small-game bridge), so convergence is judged by proxies watched live on the
dashboard: the variant-normalized average regret (trends to 0), the preflop range
grid crystallizing into a sane strategy, infoset coverage growing, and throughput.

The solve runs in chunks sized to take about `--tick-seconds` each (all threads busy); between
chunks it appends telemetry to `runs/<id>/events.jsonl`, and every `--save-minutes` /
`--checkpoint-minutes` it saves the average strategy (the durable blueprint,
`runs/<id>/blueprint.bin`) / the resumable state (`checkpoint.bin`). Those cadences are in
seconds, not iterations: in iterations they left the threads idle most of the time once the
engine got fast (a 20,000-iteration chunk is a fraction of a second on 16 threads, then a
single-threaded table scan and multi-GB checkpoint writes every few seconds). Nothing here blocks on the viewer — the
dashboard just tails the log. Designed to run for a long time (hours to days);
stop it whenever the range grid and convergence look good.

    .venv/Scripts/python.exe scripts/train_blueprint.py            # start the solve (opens the dashboard)
    .venv/Scripts/python.exe scripts/train_blueprint.py --grid 200bb   # 200bb, sized 3-/4-/5-bets
    .venv/Scripts/python.exe scripts/dashboard.py --latest        # re-open the dashboard of the latest run

The dashboard is served from this process (it stops with the solve; `--no-browser` serves
without opening it, `--no-dashboard` skips it — events are written either way).
"""
from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import parciver_engine as engine  # noqa: E402

from parciver import telemetry as T  # noqa: E402
from parciver.resources import ResourceSampler  # noqa: E402


def grid_l1_delta(prev, cur) -> float | None:
    """Mean per-visited-cell L1 change of the preflop range grid between two
    checkpoints — a real average-strategy stability signal (how much the preflop
    strategy is still moving). A cell that is unvisited in either grid is skipped."""
    if prev is None:
        return None
    total = 0.0
    n = 0
    for a, b in zip(prev, cur):
        if sum(a) == 0.0 or sum(b) == 0.0:
            continue
        total += sum(abs(x - y) for x, y in zip(a, b))
        n += 1
    return total / n if n else None


GRID_TEXT = {
    "100bb": "open {0.5,1.0}xpot, raise {1.0}xpot, all-in",
    "200bb": "open {0.5,1.0}xpot, raise {1.0}xpot, 3 more raises {1.0}xpot, all-in",
    "200bb-f1": "open {0.5,1.0}xpot, raise {1.0}xpot (the SB's open {0.5,0.75,1.0}xpot), 3 more raises {1.0}xpot, "
                "all-in",
}


FIRST_CHUNK = 20_000


def next_chunk(chunk: int, seconds: float, tick_seconds: float) -> int:
    """The chunk that should take about `tick_seconds`, given that `chunk` took `seconds`;
    moves at most 4x per tick so one slow chunk (a save, the OS) can't swing it far."""
    want = chunk * tick_seconds / max(seconds, 1e-3)
    return max(1_000, int(min(max(want, chunk / 4), chunk * 4)) // 1_000 * 1_000)


def run(run_dir: Path, *, abstraction_dir: Path, variant: str, seed: int,
        chunk: int, checkpoint_every: int, save_minutes: float,
        target_iters: int, max_seconds: float, threads: int = 0,
        resume_from: Path | None = None, checkpoint_minutes: float = 30.0,
        grid: str = "100bb", tick_seconds: float = 10.0) -> None:
    """`chunk` 0 sizes chunks to about `tick_seconds` each; `save_minutes` /
    `checkpoint_minutes` 0 save only at the end."""
    print(f"[blueprint] loading abstraction from {abstraction_dir} ...", flush=True)
    t_load = time.perf_counter()
    if resume_from is not None:
        ckpt = resume_from / "checkpoint.bin" if resume_from.is_dir() else resume_from
        if not ckpt.exists():
            raise SystemExit(f"[blueprint] --resume: no checkpoint at {ckpt}")
        if not hasattr(engine.Blueprint, "variant"):  # checked before the (long) load
            raise SystemExit("[blueprint] --resume needs an engine that reports the checkpoint's variant "
                             "(2 Oct): rebuild it (maturin develop --release)")
        try:
            bp = engine.Blueprint.resume(str(abstraction_dir), str(ckpt), threads)
        except ValueError as e:
            # A refused checkpoint: each save keeps the one before it (trained
            # on the same tables, so no help where those are the problem).
            prev = ckpt.with_name(ckpt.name + ".previous")
            corrupt = "resume failed" in str(e) and "different abstraction" not in str(e)
            hint = f"; the checkpoint before it: --resume {prev}" if corrupt and prev.exists() else ""
            raise SystemExit(f"[blueprint] --resume: {e}{hint}") from e
        print(f"[blueprint] RESUMED from {ckpt} @ {bp.iterations():,} iterations "
              f"({bp.num_infosets():,} infosets, grid {bp.grid})", flush=True)
        if bp.grid != grid:
            print(f"[blueprint] note: the checkpoint's grid {bp.grid!r} is used (--grid {grid!r} ignored)",
                  flush=True)
        if bp.variant != variant:
            print(f"[blueprint] note: the checkpoint's variant {bp.variant!r} is used (--variant {variant!r} ignored)",
                  flush=True)
        variant = bp.variant  # what the telemetry records
    else:
        bp = engine.Blueprint(str(abstraction_dir), variant, seed, threads, grid=grid)
    n_threads = bp.threads()
    print(f"[blueprint] ready in {time.perf_counter() - t_load:.1f}s "
          f"({n_threads} worker thread{'s' if n_threads != 1 else ''})", flush=True)

    config = {
        "hand_names": T.preflop_hand_names(),
        "abstraction": {"preflop": 169, "flop": 10000, "turn": 10000, "river": 1950},
        "stack_bb": 200 if bp.grid.startswith("200bb") else 100, "sb": 50, "bb": 100, "deck_size": 52,
        "board_cards": 5, "grid": bp.grid, "bet_grid": GRID_TEXT.get(bp.grid, bp.grid),
        "solver": f"external-sampling {variant} MCCFR",
        "threads": n_threads,
        "note": f"REAL {bp.grid} HU blueprint solve (B4), parallel MCCFR. Exact "
                "exploitability intractable -> convergence judged by avg-regret "
                "proxy + range sanity.",
    }
    model_path = run_dir / "blueprint.bin"
    state_path = run_dir / "checkpoint.bin"
    sampler = ResourceSampler(threads=n_threads)

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True
        print("\n[blueprint] stop requested; will checkpoint and exit after this chunk.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except Exception:
            pass

    prev_grid = None
    pending_l1 = None
    auto_chunk = chunk <= 0
    chunk = FIRST_CHUNK if auto_chunk else chunk
    t_start = time.perf_counter()
    last_save = last_state = t_start
    chunk_idx = 0
    start_iters = bp.iterations()
    with T.RunWriter(run_dir) as w:
        w.start(run_id=run_dir.name, variant=variant, game=f"HU-NLHE-{bp.grid}",
                config=config, target_iters=target_iters,
                resumed_from=(str(resume_from) if resume_from is not None else None))
        started = (f"blueprint solve RESUMED @ {start_iters:,} iters "
                   f"({n_threads} threads)" if resume_from is not None else
                   f"blueprint solve started (real {bp.grid} heads-up MCCFR, {n_threads} threads)")
        w.milestone("phase_change", started, iteration=start_iters)

        while True:
            t0 = time.perf_counter()
            bp.run(chunk)
            dt = time.perf_counter() - t0
            iters = bp.iterations()
            it_per_s = chunk / dt if dt > 0 else 0.0
            ran = chunk
            if auto_chunk:
                chunk = next_chunk(chunk, dt, tick_seconds)
            mean_abs_regret = bp.avg_regret()

            w.metrics(
                iteration=iters,
                it_per_s=it_per_s,
                mean_abs_regret=mean_abs_regret,
                wall_it_per_s=(iters - start_iters) / max(time.perf_counter() - t_start, 1e-9),
                strategy_l1_delta=pending_l1,
                infosets=bp.num_infosets(),
                resources=sampler.sample(),
            )
            pending_l1 = None
            chunk_idx += 1

            if chunk_idx % checkpoint_every == 0:
                grid = bp.preflop_range_grid()
                pending_l1 = grid_l1_delta(prev_grid, grid)
                prev_grid = grid
                by_street = list(bp.infosets_by_street())
                w.checkpoint(
                    iteration=iters,
                    range_grid={"actions": ["fold", "call", "raise"], "cells": grid},
                    infosets_by_street={s: c for s, c in zip("PFTR", by_street) if c > 0},
                    checkpoint_path=str(model_path),
                )
                wall = (iters - start_iters) / max(time.perf_counter() - t_start, 1e-9)
                print(f"[blueprint] iters={iters:>12,} infosets={bp.num_infosets():>11,} "
                      f"{it_per_s:>8,.0f} it/s solving, {wall:>8,.0f} it/s overall  "
                      f"regret={mean_abs_regret:.3e}  chunk={ran:,}", flush=True)

            now = time.perf_counter()
            if save_minutes > 0 and now - last_save >= save_minutes * 60:
                last_save = now
                bp.save(str(model_path))
                w.milestone("checkpoint_saved",
                            f"blueprint saved @ {iters:,} iterations "
                            f"({bp.num_infosets():,} infosets)", iteration=iters)

            # Full-state checkpoint (resumable) — larger than the average-only
            # blueprint, written less often; makes a multi-day solve restart-safe.
            if checkpoint_minutes > 0 and now - last_state >= checkpoint_minutes * 60:
                last_state = now
                bp.save_checkpoint(str(state_path))
                w.milestone("state_checkpoint",
                            f"resumable state saved @ {iters:,} iterations", iteration=iters)

            elapsed = time.perf_counter() - t_start
            done = (iters >= target_iters) or stop["flag"] or \
                   (max_seconds > 0 and elapsed >= max_seconds)
            if done:
                reason = "completed" if iters >= target_iters else "interrupted"
                bp.save(str(model_path))
                # Always leave a resumable full-state checkpoint on exit, so an
                # interrupted long solve can be picked up with --resume.
                bp.save_checkpoint(str(state_path))
                w.milestone("checkpoint_saved", f"final blueprint saved to {model_path.name}",
                            iteration=iters)
                w.end(iteration=iters, reason=reason, summary={
                    "final_mean_abs_regret": mean_abs_regret,
                    "infosets": bp.num_infosets(),
                    "wall_seconds": elapsed,
                    "variant": variant,
                    "blueprint_path": str(model_path),
                })
                print(f"[blueprint] {reason} @ {iters:,} iterations; blueprint at {model_path}", flush=True)
                break


DEFAULT_PORT = 8765


def _free_port(start: int) -> int:
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"[blueprint] no free dashboard port in {start}..{start + 29}")


def _serve_dashboard(run_dir: Path, port: int, open_browser: bool):
    """Serve the live dashboard for `run_dir` from a background thread of this
    process; it reads only the run's events.jsonl, so it cannot slow the solve."""
    from parciver.dashboard import server as dash_server
    port = _free_port(port)
    httpd = dash_server.serve(run_dir, port=port, open_browser=False, block=False)
    url = f"http://127.0.0.1:{port}/"
    print(f"[blueprint] dashboard: {url}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the real HU blueprint solve (B4).")
    ap.add_argument("--abstraction-dir", type=Path, default=ROOT / "data" / "abstraction")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--variant", default="linear", choices=["linear", "mccfr", "mccfr+"])
    ap.add_argument("--grid", default="100bb", choices=list(GRID_TEXT),
                    help="bet grid (and stack depth): 100bb (the original: 3-bets are all-in) or "
                         "200bb (sized 3-/4-/5-bets; the GTO Wizard depth) or 200bb-f1 (200bb, but the SB opens "
                         "to 2, 2.5 or 3bb: records/preflop-open/F1-PREREG.md). Stored in the files; "
                         "a resume uses the checkpoint's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="parallel worker threads (0=all logical cores, 1=deterministic serial)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume a prior solve: a run dir (uses its checkpoint.bin) or a "
                         "checkpoint.bin path. Telemetry starts a fresh run dir; iteration "
                         "count continues from the checkpoint, and so do its variant and grid.")
    ap.add_argument("--chunk", type=int, default=0,
                    help="MCCFR iterations per telemetry tick (0=sized to --tick-seconds)")
    ap.add_argument("--tick-seconds", type=float, default=10.0, help="target solving time per tick")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="ticks between range-grid checkpoints")
    ap.add_argument("--save-minutes", type=float, default=15.0,
                    help="minutes between blueprint.bin saves (0=only at the end)")
    ap.add_argument("--checkpoint-minutes", type=float, default=30.0,
                    help="minutes between resumable checkpoint.bin saves (0=only at the end)")
    ap.add_argument("--target-iters", type=int, default=1_000_000_000,
                    help="iteration budget (for ETA/progress); the run stops here or on Ctrl-C")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-clock cap (0=unlimited); used by the smoke gate")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="dashboard port (the next free one is used)")
    ap.add_argument("--no-browser", action="store_true", help="serve the dashboard but do not open it")
    ap.add_argument("--no-dashboard", action="store_true", help="no dashboard server (events are still written)")
    args = ap.parse_args(argv)
    if args.checkpoint_every < 1:
        ap.error("--checkpoint-every must be at least 1")

    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id("blueprint")))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[blueprint] run dir: {run_dir}", flush=True)
    httpd = None if args.no_dashboard else _serve_dashboard(run_dir, args.port, not args.no_browser)
    try:
        run(run_dir, abstraction_dir=args.abstraction_dir, variant=args.variant, seed=args.seed,
            chunk=args.chunk, checkpoint_every=args.checkpoint_every, save_minutes=args.save_minutes,
            target_iters=args.target_iters, max_seconds=args.max_seconds, threads=args.threads,
            resume_from=args.resume, checkpoint_minutes=args.checkpoint_minutes, grid=args.grid,
            tick_seconds=args.tick_seconds)
    finally:
        if httpd is not None:
            time.sleep(3.0)  # let open pages receive the final event before the server stops
            httpd.shutdown()
            print(f"[blueprint] dashboard stopped; re-open it with: scripts/dashboard.py --run {run_dir}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
