#!/usr/bin/env python
"""Evaluate the play agent: head-to-head vs baselines + LBR exploitability.

Every number is the agent's (or LBR's) winnings in bb/100 with a standard error,
on the real-chip engine at real stakes. What each line means:

  station / pot_raiser / min_raiser / random  — simple opponents; the agent should
      beat all of them. min_raiser is the translation stress test (the grid's
      smallest raise is pot-sized, so min-raises are always off-grid).
  agent_sizes_xF — the blueprint's own strategy with bet sizes scaled by F (off
      the grid): the cost of action translation; 0 = no cost.
  LBR — local best response against the agent: a LOWER BOUND on exploitability
      (higher = more exploitable). Preflop call-only, so it probes postflop play.

Also reported: how often the agent was on the blueprint vs. in the fallback
(off-tree / missing key) vs. in real-time search.

`--search river` (or `turn`) evaluates the blueprint WITH real-time search on
river (turn + river) decisions; the off-grid rows then play against the plain
blueprint.

    .venv/Scripts/python.exe scripts/evaluate_bot.py            # real blueprint
    .venv/Scripts/python.exe scripts/evaluate_bot.py --quick    # ~1/10 the hands
    .venv/Scripts/python.exe scripts/evaluate_bot.py --quick --search river
    python scripts/evaluate_bot.py --demo                        # no blueprint needed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import parciver_engine as pe  # noqa: E402

SB, BB, STACK = 50, 100, 10_000
BASELINES = ["station", "pot_raiser", "min_raiser", "random"]
PERTURB = [0.6, 1.5]


def load_agent(demo: bool):
    if demo:
        # Empty strategy on the exact-card grid: every decision is the fallback.
        return pe.PlayAgent.from_strategy({}, "100bb_exact"), "demo (fallback only)"
    from parciver.rmapss import step34
    step34.verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(step34.ABSTRACTION_DIR), str(step34.BLUEPRINT))
    return saved.play_agent(), str(step34.BLUEPRINT)


def share(sources: list[int]) -> str:
    total = sum(sources) or 1
    text = (f"blueprint {100 * sources[0] / total:.1f}%  missing {100 * sources[1] / total:.1f}%  "
            f"off-tree {100 * sources[2] / total:.1f}%")
    if len(sources) > 3 and sources[3]:
        text += f"  search {100 * sources[3] / total:.1f}%"
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deals", type=int, default=100_000, help="duplicate deals per matchup")
    ap.add_argument("--lbr-deals", type=int, default=5_000)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="1/10 of the hands")
    ap.add_argument("--demo", action="store_true", help="no blueprint: fallback-only agent")
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    ap.add_argument("--search", choices=["none", "river", "turn"], default="none",
                    help="real-time search on river (or turn+river) decisions")
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--turn-iters", type=int, default=100)
    args = ap.parse_args(argv)
    scale = 10 if args.quick else 1
    deals, lbr_deals = max(2, args.deals // scale), max(2, args.lbr_deals // scale)

    agent, source = load_agent(args.demo)
    plain = agent
    if args.search != "none":
        agent = plain.with_search(from_turn=args.search == "turn", river_iters=args.river_iters,
                                  turn_iters=args.turn_iters)
        source += f" + search({args.search}, river_iters={args.river_iters}, turn_iters={args.turn_iters})"
    print(f"agent: {source}\nstakes {SB}/{BB}, {STACK // BB}bb deep; {deals:,} duplicate deals per matchup\n")
    report = {"agent": source, "stakes": [SB, BB, STACK], "rows": []}
    rows = [(opp, {}) for opp in BASELINES] + [("perturbed", {"perturb": f, "other": plain}) for f in PERTURB]
    for opp, extra in rows:
        t = time.time()
        r = agent.match_against(opp, deals, SB, BB, STACK, seed=args.seed, threads=args.threads, **extra)
        print(f"{r['opponent']:>18s}: {r['bb_per_100']:+9.1f} ± {r['se']:6.1f} bb/100   "
              f"[{share(r['a_sources'])}]  {time.time() - t:5.0f}s", flush=True)
        report["rows"].append({**r, "seconds": time.time() - t})
    t = time.time()
    r = agent.lbr(lbr_deals, SB, BB, STACK, seed=args.seed, threads=args.threads)
    print(f"{'LBR (exploit >=)':>18s}: {r['bb_per_100']:+9.1f} ± {r['se']:6.1f} bb/100   "
          f"({r['hands']:,} hands)  {time.time() - t:5.0f}s")
    report["lbr"] = {**r, "seconds": time.time() - t}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
