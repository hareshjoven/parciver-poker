#!/usr/bin/env python
"""Full-scale run SIMULATOR — every dashboard panel, marvellable before B4 exists.

The real small-game producer (``train_small.py``) gives genuine convergence data
but a tiny 7-card game with no 169-hand preflop structure. To build, polish, and
stress-test the panels that only the *full* blueprint solve will populate — the
13x13 preflop range grid, per-street action/bet-size distributions, the head-to-
head proxy, a hundred-million-iteration throughput/coverage trace — this simulator
emits a realistic long run to the very same telemetry contract the viewer reads and
B4 will write. Nothing here touches the solver; it is a believable *shape* of a run
so the UI can be finished and proven lag-free ahead of time.

What's modelled (all decaying/rising the way a real CFR run behaves):
  * throughput with jitter, iteration climbing toward ``--target-iters``;
  * a convergence proxy ``mean_abs_regret`` ~ C / sqrt(iters) and an average-
    strategy ``strategy_l1_delta`` that shrinks as the blueprint stabilises;
  * infoset coverage saturating toward the abstraction size;
  * the preflop **range grid** sharpening from noise toward a realistic heads-up
    button range (premiums raise, trash folds, a mixed middle) — this is the piece
    to marvel at, and it visibly crystallises as "training" proceeds;
  * per-street action and bet-size distributions that drift into shape;
  * a head-to-head ``bb/100`` climbing from losing to comfortably winning;
  * live machine resources (these are REAL — sampled from this box).

Pacing: by default the run is spread over ``--duration`` seconds so you can watch it
evolve; ``--fast`` emits as quickly as possible to flood the pipeline and prove the
viewer coalesces without lagging. Point the dashboard at the printed run directory,
or use ``scripts/dashboard.py --demo full`` to launch both together.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import telemetry as T  # noqa: E402
from parciver.resources import ResourceSampler  # noqa: E402

RANK_VAL = {r: 14 - i for i, r in enumerate("AKQJT98765432")}


# --------------------------------------------------------------------------
# A believable target preflop range (heads-up button / small blind RFI).
# --------------------------------------------------------------------------

def hand_strength(name: str) -> float:
    """A 0..1 preflop strength score for a canonical hand name (``AA``/``AKs``/
    ``72o``). A compact Chen-flavoured heuristic — enough to give the range grid a
    realistic, recognisable shape without calling the equity engine."""
    if len(name) == 2:  # a pair, e.g. "AA"
        v = RANK_VAL[name[0]]
        return min(1.0, 0.55 + (v - 2) / 12.0 * 0.45)
    hi, lo = RANK_VAL[name[0]], RANK_VAL[name[1]]
    suited = name.endswith("s")
    gap = hi - lo
    base = (hi - 2) / 12.0 * 0.5 + (lo - 2) / 12.0 * 0.28
    if suited:
        base += 0.10
    if gap == 1:
        base += 0.07  # connected
    elif gap == 2:
        base += 0.03
    base -= max(0, gap - 2) * 0.012  # disconnected penalty
    if hi == RANK_VAL["A"]:
        base += 0.05  # ace blockers / high card
    return float(max(0.0, min(1.0, base)))


def target_mix(strength: float) -> np.ndarray:
    """Target [fold, call, raise] action mix for a hand of the given strength — a
    wide, aggressive button range: premiums pure-raise, a mixed middle, trash folds."""
    if strength > 0.72:
        fold, call, raise_ = 0.0, 0.06, 0.94
    elif strength > 0.55:
        fold, call, raise_ = 0.06, 0.16, 0.78
    elif strength > 0.42:
        fold, call, raise_ = 0.30, 0.22, 0.48
    elif strength > 0.32:
        fold, call, raise_ = 0.60, 0.16, 0.24
    else:
        fold = min(0.98, 0.80 + (0.32 - strength))
        raise_ = (1.0 - fold) * 0.55  # a few bluff-raises survive at the bottom
        call = 1.0 - fold - raise_
    m = np.array([fold, call, raise_], dtype=np.float64)
    return m / m.sum()


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class RangeModel:
    """Evolves the 169 preflop action mixes from noisy-uniform toward the target
    range as training progresses, with a little live shimmer so the grid feels
    alive."""

    def __init__(self, names: list[str], seed: int) -> None:
        self.names = names
        self.rng = np.random.default_rng(seed)
        self.strength = np.array([hand_strength(n) for n in names])
        self.target = np.stack([target_mix(s) for s in self.strength])  # [169,3]
        self.uniform = np.full((len(names), 3), 1 / 3)

    def grid(self, progress: float) -> list[list[float]]:
        # Sharpen from uniform to target; convergence is fast early, fine-tuning late.
        w = smoothstep(progress) ** 0.7
        mix = (1 - w) * self.uniform + w * self.target
        # Per-checkpoint shimmer that fades as the range locks in.
        noise = self.rng.normal(0, 0.02 * (1 - w) + 0.004, mix.shape)
        mix = np.clip(mix + noise, 0.0, None)
        mix /= mix.sum(axis=1, keepdims=True)
        return mix.tolist()


# --------------------------------------------------------------------------
# Per-street distributions that drift into shape.
# --------------------------------------------------------------------------

def action_freq(progress: float, rng: np.random.Generator) -> dict:
    w = smoothstep(progress)

    def blend(early: dict, late: dict) -> dict:
        out = {k: (1 - w) * early[k] + w * late[k] for k in late}
        out = {k: max(0.0, v + rng.normal(0, 0.01 * (1 - w) + 0.003)) for k, v in out.items()}
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    return {
        "P": blend({"f": .34, "c": .33, "r": .33}, {"f": .28, "c": .18, "r": .54}),
        "F": blend({"x": .34, "c": .33, "r": .23, "A": .10}, {"x": .40, "c": .22, "r": .30, "A": .08}),
        "T": blend({"x": .34, "c": .33, "r": .23, "A": .10}, {"x": .44, "c": .24, "r": .24, "A": .08}),
        "R": blend({"x": .34, "c": .33, "r": .23, "A": .10}, {"x": .46, "c": .28, "r": .16, "A": .10}),
    }


def bet_size_freq(progress: float, rng: np.random.Generator) -> dict:
    w = smoothstep(progress)

    def blend(early: dict, late: dict) -> dict:
        out = {k: (1 - w) * early[k] + w * late[k] for k in late}
        out = {k: max(0.0, v + rng.normal(0, 0.01 * (1 - w) + 0.003)) for k, v in out.items()}
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    return {
        "P": blend({"1.0pot": .5, "allin": .5}, {"1.0pot": .82, "allin": .18}),
        "F": blend({"0.5pot": .34, "1.0pot": .33, "allin": .33}, {"0.5pot": .54, "1.0pot": .38, "allin": .08}),
        "T": blend({"0.5pot": .34, "1.0pot": .33, "allin": .33}, {"0.5pot": .46, "1.0pot": .42, "allin": .12}),
        "R": blend({"0.5pot": .34, "1.0pot": .33, "allin": .33}, {"0.5pot": .38, "1.0pot": .44, "allin": .18}),
    }


def infosets_by_street(total: int, progress: float) -> dict:
    # Preflop saturates first, later streets fill in as coverage deepens.
    frac = {"P": 0.02, "F": 0.30, "T": 0.34, "R": 0.34}
    speed = {"P": 6.0, "F": 3.0, "T": 2.2, "R": 1.6}
    return {s: int(total * frac[s] * (1 - math.exp(-speed[s] * progress))) for s in "PFTR"}


# --------------------------------------------------------------------------
# The simulated run.
# --------------------------------------------------------------------------

def run(run_dir: Path, *, target_iters: int, duration: float, tick: float,
        checkpoint_every: int, fast: bool, seed: int) -> None:
    names = T.preflop_hand_names()
    rng = np.random.default_rng(seed)
    ranges = RangeModel(names, seed)

    infoset_total = 42_000_000  # order-of-magnitude for the 100bb HU abstraction
    base_rate = 520_000.0       # ~it/s baseline for the blueprint solve
    r0 = 0.42                    # mean|regret| scale

    sampler = ResourceSampler(threads=16)
    n_ticks = max(2, int(round(duration / tick))) if not fast else 900
    start = time.perf_counter()

    with T.RunWriter(run_dir) as w:
        config = {
            "hand_names": names,
            "abstraction": {"preflop": 169, "flop": 10000, "turn": 10000, "river": 1950},
            "stack_bb": 100, "sb": 50, "bb": 100, "deck_size": 52, "board_cards": 5,
            "bet_grid": "open {0.5,1.0}xpot, raise {1.0}xpot, all-in",
            "solver": "external-sampling Linear MCCFR (simulated)",
            "note": "SIMULATED full-scale run for dashboard development (not a real solve)",
        }
        w.start(run_id=run_dir.name, variant="linear-mccfr", game="HU-NLHE-100bb",
                config=config, target_iters=target_iters)
        w.milestone("phase_change", "blueprint solve started (simulated 100bb heads-up)", iteration=0)

        peak_gpu = 0.0
        last_hh = -22.0
        for i in range(n_ticks + 1):
            progress = i / n_ticks
            iters = int(progress * target_iters)

            # Throughput: warm-up ramp then steady with jitter.
            warm = min(1.0, progress * 6)
            it_per_s = base_rate * (0.35 + 0.65 * warm) * (1 + rng.normal(0, 0.06))
            it_per_s = max(1000.0, it_per_s)

            # Convergence proxies.
            mean_regret = r0 / math.sqrt(1 + iters / 5e5) * (1 + rng.normal(0, 0.03))
            l1_delta = 0.9 / (1 + iters / 3e5) * (1 + rng.normal(0, 0.05)) if iters > 0 else None
            infosets = int(infoset_total * (1 - math.exp(-3.2 * progress)))

            # Head-to-head: climb from losing toward a winning plateau, monotone-ish.
            hh_target = -22.0 + 40.0 * smoothstep(progress)
            last_hh += (hh_target - last_hh) * 0.5 + rng.normal(0, 0.6)

            res = sampler.sample()
            if res.get("gpu_pct"):
                peak_gpu = max(peak_gpu, res["gpu_pct"])

            w.metrics(
                iteration=iters, it_per_s=it_per_s, mean_abs_regret=mean_regret,
                strategy_l1_delta=l1_delta, infosets=infosets, hh_bb_per_100=last_hh,
                resources=res,
            )

            if i % checkpoint_every == 0 or i == n_ticks:
                w.checkpoint(
                    iteration=iters,
                    range_grid={"actions": ["fold", "call", "raise"], "cells": ranges.grid(progress)},
                    action_freq=action_freq(progress, rng),
                    bet_size_freq=bet_size_freq(progress, rng),
                    infosets_by_street=infosets_by_street(infoset_total, progress),
                    checkpoint_path=f"checkpoints/{run_dir.name}/iter_{iters}.blueprint",
                )
                if i > 0:
                    w.milestone("checkpoint_saved", f"strategy snapshot @ {iters:,} iterations",
                                iteration=iters)

            if not fast:
                target_t = start + (i + 1) * tick
                sleep = target_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)

        w.milestone("phase_change", "blueprint solve complete", iteration=target_iters)
        w.end(iteration=target_iters, reason="completed", summary={
            "final_mean_abs_regret": mean_regret,
            "final_hh_bb_per_100": last_hh,
            "infosets": infosets,
            "peak_gpu_pct": peak_gpu,
            "wall_seconds": time.perf_counter() - start,
            "variant": "linear-mccfr (simulated)",
        })
    print(f"[replay_demo] done. run dir: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Simulate a full-scale blueprint run for the dashboard.")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--target-iters", type=int, default=100_000_000,
                    help="the run's headline iteration count (default: 100,000,000)")
    ap.add_argument("--duration", type=float, default=180.0,
                    help="wall seconds to spread the run over (ignored with --fast)")
    ap.add_argument("--tick", type=float, default=0.15, help="metrics emit cadence (seconds)")
    ap.add_argument("--checkpoint-every", type=int, default=12, help="ticks between checkpoints")
    ap.add_argument("--fast", action="store_true",
                    help="emit as fast as possible (stress the viewer's coalescing)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id("demo")))
    print(f"[replay_demo] run dir: {run_dir}  (target {args.target_iters:,} iters, "
          f"{'fast' if args.fast else f'{args.duration:.0f}s'})")
    run(run_dir, target_iters=args.target_iters, duration=args.duration, tick=args.tick,
        checkpoint_every=args.checkpoint_every, fast=args.fast, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
