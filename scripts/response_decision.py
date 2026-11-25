#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Apply the response-convergence study's pre-registered decision rule.

Rule (PREDICTIONS["decision_rule"] in response_convergence.py): recommend the
cheapest (N, tau) whose null gain 99% lower bound is >= -2 bb/100, then maximize
the mean leak gain; report if none qualifies.

"Null gain" = the candidate's gain against the anchor itself; its 99% lower bound
is the replicate mean minus Z99 * sqrt(sum se^2) / reps (evaluation noise only,
the same aggregation the dashboard plots). "Cheapest" = smallest N (compute is
linear in N; tau is free). "Leak gain" = mean over every non-anchor opponent and
replicate. No engine import, so it runs on any results.json:

    python scripts/response_decision.py runs/response-convergence-*/results.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z99 = 2.576
NULL_FLOOR = -2.0  # bb/100


def _aggregate(rows: list[dict], opponent: str, n: int, ti: int) -> dict | None:
    cands = [r["candidates"][ti] for r in rows if r["opponent"] == opponent and r["n"] == n]
    if not cands:
        return None
    mean = sum(c["gain"] for c in cands) / len(cands)
    half = Z99 * math.sqrt(sum(c["se"] ** 2 for c in cands)) / len(cands)
    return {"mean": mean, "lo99": mean - half, "reps": len(cands)}


def decide(config: dict, rows: list[dict]) -> dict:
    """Every (N, tau) cell with its null bound and leak gain, plus the recommendation."""
    leaks = [o for o in config["opponents"] if o != "anchor"]
    cells = []
    for n in config["ladder"]:
        for ti, tau in enumerate(config["taus"]):
            null = _aggregate(rows, "anchor", n, ti)
            leak = [a for a in (_aggregate(rows, o, n, ti) for o in leaks) if a]
            cells.append({
                "n": n, "tau": tau,
                "null_gain": null and null["mean"], "null_lo99": null and null["lo99"],
                "leak_gain": sum(a["mean"] for a in leak) / len(leak) if leak else None,
                "leak_opponents": len(leak), "complete": bool(null) and len(leak) == len(leaks),
            })
    if "anchor" not in config["opponents"]:
        return {"cells": cells, "recommendation": None,
                "verdict": "no null (anchor) rows: the rule cannot be applied"}
    qualifying = [c for c in cells if c["complete"] and c["null_lo99"] >= NULL_FLOOR]
    if not qualifying:
        return {"cells": cells, "recommendation": None,
                "verdict": f"NO (N, tau) has null 99% lower bound >= {NULL_FLOOR:+g} bb/100"}
    cheapest = min(c["n"] for c in qualifying)
    best = max((c for c in qualifying if c["n"] == cheapest),
               key=lambda c: (c["leak_gain"] if c["leak_gain"] is not None else -math.inf))
    leak = "n/a" if best["leak_gain"] is None else f"{best['leak_gain']:+.1f}"
    return {"cells": cells, "recommendation": best,
            "verdict": f"N={best['n']:,}/seat, tau={best['tau']:g}: null {best['null_gain']:+.1f} "
                       f"(99% lo {best['null_lo99']:+.1f}), mean leak gain {leak} bb/100"}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    data = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result = decide(data["config"], data["rows"])
    print(f"{'N/seat':>12} {'tau':>7} {'null':>8} {'null lo99':>10} {'leak':>8}")
    for c in result["cells"]:
        if c["null_gain"] is None and c["leak_gain"] is None:
            continue
        fmt = lambda v: "—" if v is None else f"{v:+.1f}"  # noqa: E731
        mark = " *" if c is result["recommendation"] else ""
        print(f"{c['n']:>12,} {c['tau']:>7g} {fmt(c['null_gain']):>8} {fmt(c['null_lo99']):>10} "
              f"{fmt(c['leak_gain']):>8}{mark}")
    print(f"decision: {result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
