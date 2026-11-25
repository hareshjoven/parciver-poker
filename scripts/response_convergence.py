#!/usr/bin/env python
"""Response-solve convergence study — bot engineering, NOT a locked R-MAPSS protocol.

Question: how good is the full-game fixed-opponent response (external-sampling
MCCFR vs a fixed opponent, both seats, blended over the anchor) as a function of
its iteration budget, and does gating out rarely-reached response infosets (they
fall back to the anchor) make it better per unit of compute?

Why it matters for the bot: the response solve IS the exploitation step. S3.4's
5M-iteration "best-known response" lost 26-32 bb/100 to the anchor itself
(RMAPSS §49), although in the abstracted game the anchor is a feasible response,
so a converged best response to the anchor cannot lose to it.

Design (E1 = iteration ladder, E2 = reach gating, sharing every solve):
  * opponents: the anchor (null: converged gain must be >= 0) and held leaks
    nit(0.35), station(0.35);
  * N (iterations per seat) ladder, ascending; `--reps` independent solve seeds;
  * each solve yields one candidate per reach threshold tau (`fixed_response_gated`;
    tau=0 == the S3.x construction exactly);
  * each candidate is scored by `paired_evaluate_blocks` vs the anchor, with the
    SAME evaluation seed for every N and tau of an (opponent, rep) — common random
    numbers, so curves compare like with like.

Predictions and the decision rule are written into run_start BEFORE any result
(see PREDICTIONS). Results: runs/response-convergence-*/results.json + a live
dashboard (auto-opens; --no-browser to suppress).

    .venv/Scripts/python.exe scripts/response_convergence.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import parciver_engine as pe  # noqa: E402
from parciver.resources import ResourceSampler  # noqa: E402
from parciver.rmapss import step34  # noqa: E402
from parciver.telemetry import RunWriter, make_run_id  # noqa: E402
from response_decision import decide  # noqa: E402

OPPONENTS = {  # label -> (family, strength); None = the anchor itself (null)
    "anchor": None,
    "nit(0.35)": ("nit", 0.35),
    "station(0.35)": ("station", 0.35),
}
DEFAULT_LADDER = [1_000_000, 4_000_000, 16_000_000, 64_000_000]
DEFAULT_TAUS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5]
BASE_SEED = 7_100
EVAL_BLOCK = 10_000
Z99 = 2.576

PREDICTIONS = {
    "P1": "tau=0: gain vs the anchor (null) rises with N and is negative at 1M/4M; "
          "gains vs nit/station rise with N.",
    "P2": "reach gating (tau in 1e-7..1e-6) raises the null gain over tau=0 at N<=16M "
          "(paired difference > 0), with the largest effect at small N.",
    "P3": "gating does not cost more than the evaluation interval on nit/station at the "
          "largest N.",
    "decision_rule": "recommend the cheapest (N, tau) whose null gain 99% lower bound is "
                     ">= -2 bb/100, then maximize the mean leak gain; report if none qualifies.",
}


def _spawn_dashboard(run_dir: Path, port: int) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"),
           "--run", str(run_dir), "--port", str(port), "--no-browser"]
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(cmd, cwd=str(ROOT), **kwargs)


def _measure(candidate, anchor, opponent, hands: int, seed: int, threads: int) -> dict:
    m = dict(candidate.paired_evaluate_blocks(anchor, opponent, hands, seed, EVAL_BLOCK, threads))
    blocks = [float(x) for x in m["block_gains_bb_per_100"]]
    se = statistics.stdev(blocks) / math.sqrt(len(blocks))
    return {"gain": float(m["gain_bb_per_100"]), "se": se,
            "lo99": float(m["gain_bb_per_100"]) - Z99 * se,
            "hi99": float(m["gain_bb_per_100"]) + Z99 * se, "blocks": blocks}


class Heartbeat(threading.Thread):
    """Emit live resources + in-solve progress while a blocking solve runs."""

    def __init__(self, writer: RunWriter, lock: threading.Lock, sampler: ResourceSampler):
        super().__init__(daemon=True)
        self.writer, self.lock, self.sampler = writer, lock, sampler
        self.stop = threading.Event()
        self.solve = None  # (started_at, expected_s, label) of the solve in flight
        self.updates_done = 0

    def run(self) -> None:
        while not self.stop.wait(2.0):
            fields = {"resources": self.sampler.sample()}
            if self.solve:
                started, expected, label = self.solve
                fields["rc_solve"] = {"label": label, "elapsed_s": time.time() - started,
                                      "expected_s": expected}
            with self.lock:
                if self.stop.is_set():  # run_end may already be written
                    break
                self.writer.metrics(self.updates_done, **fields)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ladder", type=int, nargs="+", default=DEFAULT_LADDER,
                    help="iterations PER SEAT, ascending")
    ap.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS)
    ap.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=list(OPPONENTS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--eval-hands", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.ladder != sorted(args.ladder) or args.taus[0] != 0.0:
        ap.error("--ladder must be ascending and --taus must start with 0 (the ungated baseline)")

    run_dir = ROOT / "runs" / make_run_id("response-convergence")
    run_dir.mkdir(parents=True)
    dashboard = _spawn_dashboard(run_dir, args.port)
    url = f"http://127.0.0.1:{args.port}/static/response.html"
    print(f"[dashboard] {url}  (watching {run_dir.name})", flush=True)

    step34.verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(step34.ABSTRACTION_DIR), str(step34.BLUEPRINT))
    anchor = saved.s32_policy()
    opponents = {label: (anchor if OPPONENTS[label] is None else anchor.leak(*OPPONENTS[label]))
                 for label in args.opponents}

    units = [(n, label, rep) for n in args.ladder for label in args.opponents
             for rep in range(args.reps)]
    total_updates = sum(2 * n for n, _, _ in units)
    config = {
        "kind": "experiment", "study": "response-convergence",
        "ladder": args.ladder, "taus": args.taus, "opponents": args.opponents,
        "reps": args.reps, "eval_hands": args.eval_hands, "eval_block": EVAL_BLOCK,
        "threads": args.threads, "units": len(units), "total_updates": total_updates,
        "predictions": PREDICTIONS, "blueprint": str(step34.BLUEPRINT),
        "construction": "anchor.blend(resp_seat0,1).blend(resp_seat1,1); tau>0 drops "
                        "response infosets with estimated reach < tau (anchor fallback)",
    }
    lock = threading.Lock()
    sampler = ResourceSampler(threads=args.threads)
    rows: list[dict] = []
    with RunWriter(run_dir) as writer:
        writer.start(variant="linear", game="nlhe-100bb-abstracted", config=config,
                     target_iters=total_updates)
        if not args.no_browser:
            webbrowser.open(url)
        beat = Heartbeat(writer, lock, sampler)
        beat.start()
        rate = 250_000.0  # updates/s prior; replaced by the measured rate after one solve
        started = time.time()
        try:
            for index, (n, label, rep) in enumerate(units):
                opponent = opponents[label]
                opp_index = args.opponents.index(label)
                eval_seed = BASE_SEED * 1_000 + opp_index * 10 + rep  # fixed across N and tau
                with lock:
                    writer.checkpoint(beat.updates_done, rc_progress={
                        "completed": index, "total": len(units), "n": n, "opponent": label,
                        "rep": rep, "updates_done": beat.updates_done,
                        "total_updates": total_updates, "rate": rate})
                seat_policies = []
                kept_by_seat = []
                total_by_seat = []
                solve_s = 0.0
                for seat in (0, 1):
                    seed = BASE_SEED + 100_000 * rep + 1_000 * opp_index + 10 * args.ladder.index(n) + seat
                    beat.solve = (time.time(), n / rate, f"{label} · N={n:,} · rep {rep} · seat {seat}")
                    t0 = time.time()
                    policies, kept, total = opponent.fixed_response_gated(
                        seat, n, "linear", seed, args.threads, args.taus)
                    dt = time.time() - t0
                    solve_s += dt
                    rate = n / dt
                    beat.solve = None
                    beat.updates_done += n
                    seat_policies.append(policies)
                    kept_by_seat.append(kept)
                    total_by_seat.append(total)
                    with lock:
                        writer.metrics(beat.updates_done, it_per_s=rate, infosets=total)
                candidates = []
                for ti, tau in enumerate(args.taus):
                    cand = anchor.blend(seat_policies[0][ti], 1.0).blend(seat_policies[1][ti], 1.0)
                    m = _measure(cand, anchor, opponent, args.eval_hands, eval_seed, args.threads)
                    kept_frac = sum(k[ti] for k in kept_by_seat) / max(1, sum(total_by_seat))
                    candidates.append({"tau": tau, "kept_frac": kept_frac, **m})
                base_blocks = candidates[0]["blocks"]
                for c in candidates:  # paired difference vs the ungated candidate (same hands)
                    diffs = [a - b for a, b in zip(c["blocks"], base_blocks)]
                    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if c["tau"] else 0.0
                    c["delta_vs_tau0"] = statistics.fmean(diffs)
                    c["delta_se"] = se
                row = {"n": n, "opponent": label, "rep": rep, "solve_s": solve_s,
                       "infosets": total_by_seat, "eval_seed": eval_seed,
                       "candidates": [{k: v for k, v in c.items() if k != "blocks"} for c in candidates]}
                rows.append(row)
                (run_dir / "results.json").write_text(json.dumps(
                    {"config": config, "rows": rows}, indent=1), encoding="utf-8")
                best = max(candidates, key=lambda c: c["gain"])
                with lock:
                    writer.checkpoint(beat.updates_done, rc_row=row, rc_progress={
                        "completed": index + 1, "total": len(units), "n": n, "opponent": label,
                        "rep": rep, "updates_done": beat.updates_done,
                        "total_updates": total_updates, "rate": rate})
                    writer.milestone("note", f"{label} N={n:,} rep {rep}: tau=0 "
                                     f"{candidates[0]['gain']:+.1f}, best tau={best['tau']:g} "
                                     f"{best['gain']:+.1f} bb/100 ({solve_s:.0f}s solve)",
                                     iteration=beat.updates_done)
                print(f"[{index + 1}/{len(units)}] {label:14s} N={n:>11,} rep {rep}  " + "  ".join(
                    f"tau={c['tau']:g}:{c['gain']:+7.1f}±{Z99 * c['se']:.1f}" for c in candidates),
                    flush=True)
            decision = decide(config, rows)
            (run_dir / "results.json").write_text(json.dumps(
                {"config": config, "rows": rows, "decision": decision}, indent=1), encoding="utf-8")
            print(f"[decision] {decision['verdict']}", flush=True)
            summary = {"rows": len(rows), "elapsed_s": time.time() - started,
                       "results": str(run_dir / "results.json"), "decision": decision["verdict"]}
            with lock:
                beat.stop.set()
                writer.end(beat.updates_done, reason="completed", summary=summary)
        except BaseException as exc:
            beat.stop.set()
            with lock:
                writer.end(beat.updates_done, reason="interrupted" if isinstance(
                    exc, KeyboardInterrupt) else "error", summary={"error": repr(exc)})
            raise
    print(f"[done] {run_dir / 'results.json'}  (dashboard pid {dashboard.pid} still serving {url})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
