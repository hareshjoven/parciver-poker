#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Replay a scorecard benchmark's LBR deals with every action recorded, for several
versions of the bot, to see WHERE LBR wins (see `PlayAgent.lbr_trace`).

    .venv/Scripts/python.exe scripts/lbr_trace.py --benchmark runs/scorecard/scorecard-v1-quick/benchmark.json --candidate runs/blueprint-20260914-144924/blueprint.bin

Agents (default all): `plain` (blueprint, no search), `search-v1` (river search, own
range modelled from the policy, round forced: the first version), `search-exact` (exact
own range, round forced), `search` (exact own range, round solved: the current one).
Same deals and seeds as the benchmark's LBR row, so each agent's per-deal results are
checked against any receipt of the same bot in the benchmark directory.
Writes `trace-<utc>.json.gz` next to the benchmark and prints a summary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import scorecard as sc  # noqa: E402

AGENTS = {
    "plain": None,
    "search-v1": {"own_range": "model", "round": "forced"},
    "search-exact": {"own_range": "exact", "round": "forced"},
    "search": {"own_range": "exact", "round": "solve"},
}
# What receipts from before an option existed played.
SEARCH_DEFAULTS = {"own_range": "model", "round": "forced"}
BIG_POT_BB = 50.0


def build(saved, name: str, river_iters: int):
    agent = saved.play_agent()
    cfg = AGENTS[name]
    if cfg is None:
        return agent, None
    search = {"streets": "river", "river_iters": river_iters, "turn_iters": 100, **cfg}
    return agent.with_search(river_iters=river_iters, exact_own_range=cfg["own_range"] == "exact",
                             solve_round=cfg["round"] == "solve"), search


def matching_receipt(bench_dir: Path, cand_sha: str, search: dict | None, deals: int):
    """A receipt of the same bot on this benchmark (same LBR deals), if any."""
    for p in sorted(bench_dir.glob("receipt-*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rs = r.get("candidate", {}).get("search")
        if rs is not None:
            rs = {**SEARCH_DEFAULTS, **rs}
        same = (r.get("candidate", {}).get("sha256", "").upper() == cand_sha.upper() and rs == search)
        per = r.get("raw", {}).get("lbr", {}).get("per_deal_bb")
        if same and per is not None and len(per) == deals:
            return p, per
    return None, None


def summarize(hands: list[dict], hpd: int) -> dict:
    per: dict[int, float] = {}
    for h in hands:
        per[h["deal"]] = per.get(h["deal"], 0.0) + h["net_bb"]
    vals = [per[k] for k in sorted(per)]
    n = len(vals)
    mean = sum(vals) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1) / n) if n > 1 else float("nan")
    out = {"bb_per_100": mean / hpd * 100, "se": se / hpd * 100, "per_deal_bb": vals}
    # Agent decisions on the river, split by how many river actions it already took.
    first, later, calls, big = [], [], [], {"hands": 0, "lbr_net_bb": 0.0}
    for h in hands:
        acted = 0
        if abs(h["net_bb"]) >= BIG_POT_BB:
            big["hands"] += 1
            big["lbr_net_bb"] += h["net_bb"]
        for s in h["steps"]:
            if s["actor"] != "agent" or s["street"] != 5:
                continue
            d = s.get("diag")
            if d is not None:
                (first if acted == 0 else later).append(d["own_tv"])
                if s["action"] == "call" and s["owed"] > 0:
                    calls.append((d["eq_model"], d["eq_actual"], s["owed"] / (s["pot"] + s["owed"])))
            acted += 1
    mean_ = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    out["river"] = {
        "own_tv_first": mean_(first), "n_first": len(first),
        "own_tv_later": mean_(later), "n_later": len(later),
        "calls": len(calls),
        "call_eq_model": mean_([c[0] for c in calls]),
        "call_eq_actual": mean_([c[1] for c in calls]),
        "call_price": mean_([c[2] for c in calls]),
    }
    out["big_pots"] = big
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--agents", default=",".join(AGENTS), help=f"comma list of {sorted(AGENTS)}")
    ap.add_argument("--deals", type=int, default=0, help="first N LBR deals (0 = all of the benchmark's)")
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    import parciver_engine as pe

    manifest_path = Path(args.benchmark)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sc.check_manifest(manifest)
    st, lb, hpd = manifest["stakes"], manifest["lbr"], manifest["hands_per_deal"]
    deals = args.deals or lb["deals"]
    names = [a.strip() for a in args.agents.split(",") if a.strip()]
    for a in names:
        if a not in AGENTS:
            ap.error(f"unknown agent {a!r}")
    cand = Path(args.candidate).resolve()
    cand_sha = sc.sha256_file(cand)
    saved = pe.SavedBlueprint(manifest["abstraction"]["dir"], str(cand))

    result = {"benchmark_id": manifest["benchmark_id"], "manifest_hash": manifest["manifest_hash"],
              "candidate": {"path": str(cand), "sha256": cand_sha}, "lbr_seed": lb["seed"], "deals": deals,
              "stakes": st, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "agents": {}}
    for name in names:
        agent, search = build(saved, name, args.river_iters)
        t = time.time()
        hands = agent.lbr_trace(list(range(deals)), st["sb"], st["bb"], st["stack"], seed=lb["seed"],
                                threads=args.threads)
        elapsed = time.time() - t
        s = summarize(hands, hpd)
        rp, per = matching_receipt(manifest_path.parent, cand_sha, search, deals)
        check = "no receipt of this bot to check against"
        if per is not None:
            ok = all(abs(a - b) < 1e-9 for a, b in zip(per, s["per_deal_bb"]))
            check = f"{'MATCHES' if ok else 'DIFFERS FROM'} {rp.name}"
            if not ok:
                print(f"WARNING: {name}: replay differs from {rp.name}", file=sys.stderr)
        rv = s["river"]
        print(f"{name:12s} LBR {s['bb_per_100']:+7.1f} ± {1.96 * s['se']:.1f} bb/100  ({elapsed:.0f}s; {check})")
        print(f"{'':12s} river own-range error (TV): first decision {rv['own_tv_first']:.3f} (n={rv['n_first']}), "
              f"later {rv['own_tv_later']:.3f} (n={rv['n_later']})")
        print(f"{'':12s} river calls: {rv['calls']}, equity vs modelled range {rv['call_eq_model']:.3f}, "
              f"vs LBR's actual hand {rv['call_eq_actual']:.3f}, price {rv['call_price']:.3f}")
        print(f"{'':12s} hands with |net| >= {BIG_POT_BB:.0f}bb: {s['big_pots']['hands']}, "
              f"LBR net {s['big_pots']['lbr_net_bb']:+.0f}bb")
        result["agents"][name] = {"search": search, "elapsed_s": elapsed, "summary": {k: v for k, v in s.items()
                                  if k != "per_deal_bb"}, "per_deal_bb": s["per_deal_bb"], "check": check,
                                  "hands": hands}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = manifest_path.parent / f"trace-{stamp}.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"\ntrace: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
