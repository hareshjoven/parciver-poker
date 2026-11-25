#!/usr/bin/env python
"""V2LEAK (see V2LEAK.md): where does LBR win more against one bot than another?

Stage 1 replays a scorecard benchmark's LBR stream (no search) for several bots with every action
recorded, labels every hand, and splits the paired per-deal gap between two bots into additive
classes:

    .venv/Scripts/python.exe scripts/v2leak.py trace --benchmark <S>/benchmark.json --deals 100000 \
        --bot v1=<v1 blueprint.bin> --bot v2-2B=<2B blueprint.bin> --out-dir <dir> [--sizes grid]
    .venv/Scripts/python.exe scripts/v2leak.py report --out-dir <dir> --pair v2-2B:v1
    .venv/Scripts/python.exe scripts/v2leak.py delta --out-dir <dir> --pair v2-2B:v1 --vs v2-2B-grid:v1-grid

`trace` writes `<dir>/<label>[-<sizes>][-nopeek][-standin].npz` (one row per hand; `--translation standin`:
the agent reads oversized bets as TRANSLATE.md's option does) and checks deals 0..4999 against any
receipt of the same blueprint in the benchmark directory (the bit-identity gate). `report` prints the
decomposition with 95% CIs (per-deal paired differences).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import scorecard as sc  # noqa: E402

LBR_FRACS = (0.5, 1.0, 2.0)
# The agent's grid per blueprint grid name: opening fracs, first-raise fracs, re-raise fracs, re-raise cap.
GRIDS = {
    "100bb": ((0.5, 1.0), (1.0,), (), 0),
    "200bb": ((0.5, 1.0), (1.0,), (1.0,), 3),
}
Z_NAMES = ["Z0 LBR never bet/raised postflop", "Z1 all LBR bets on the agent's grid", "Z2 some LBR bet off the grid"]
E_NAMES = ["E0 ended preflop", "E1 agent folded postflop", "E2 LBR folded postflop", "E3 showdown"]
P_NAMES = ["P0 no preflop raise", "P1 agent raised preflop", "P2 ended preflop"]
SEAT_NAMES = ["agent SB", "agent BB"]
STREET = {0: 0, 3: 1, 4: 2, 5: 3}
STREET_NAMES = ["preflop", "flop", "turn", "river"]


def rnd(x: float) -> int:
    """Rust's f64::round (half away from zero) for non-negative values."""
    return int(math.floor(x + 0.5))


class Replay:
    """The real betting state, rebuilt from a traced hand's steps (checked against each step's
    recorded pot and amount owed)."""

    def __init__(self, sb: int, bb: int, stack: int):
        self.bb, self.stack = bb, stack
        self.committed = [sb, bb]
        self.contrib = [sb, bb]
        self.last_raise, self.street_bets, self.street = bb, 1, 0

    def enter(self, street: int):
        if street != self.street:
            self.street, self.contrib, self.last_raise, self.street_bets = street, [0, 0], 0, 0

    def owed(self, p: int) -> int:
        return max(self.contrib) - self.contrib[p]

    def remaining(self, p: int) -> int:
        return self.stack - self.committed[p]

    def floor(self) -> int:
        return max(self.last_raise, self.bb)

    def amount(self, p: int, f: float) -> int:
        """Chips to add for a pot-fraction-`f` bet/raise (the grid's and LBR's formula)."""
        pot, c = sum(self.committed), self.owed(p)
        if c == 0:
            return max(rnd(f * pot), self.bb)
        return c + max(rnd(f * (pot + c)), self.floor())

    def grid_fracs(self, p: int, grid) -> tuple:
        opens, raises, reraises, cap = grid
        if self.owed(p) == 0:
            return opens
        if self.street_bets <= 1:
            return raises
        if self.street_bets <= 1 + cap:
            return reraises
        return ()

    def apply(self, p: int, action: str, add: int):
        c = self.owed(p)
        if action == "call":
            d = min(c, self.remaining(p))
            self.committed[p] += d
            self.contrib[p] += d
        elif action == "bet":
            if add <= c:
                return self.apply(p, "call", 0)
            before = max(self.contrib)
            stackp = self.remaining(p)
            self.committed[p] += add
            self.contrib[p] += add
            by = self.contrib[p] - before
            if by > 0:
                self.last_raise = max(by, self.last_raise) if add == stackp else by
                self.street_bets += 1


def label_hand(h: dict, st: dict, grid) -> dict:
    lbr = h["lbr_seat"]
    agent = 1 - lbr
    r = Replay(st["sb"], st["bb"], st["stack"])
    steps = h["steps"]
    lbr_bets = on = off = 0
    frac_counts = {"0.5": 0, "1.0": 0, "2.0": 0, "allin": 0}
    agent_raised_pre = False
    off_tree = decisions = 0
    flop_pot = 0
    # agent responses to LBR postflop bets: (street, on_grid, folded)
    responses = []
    pending = None
    for s in steps:
        r.enter(s["street"])
        p = lbr if s["actor"] == "lbr" else agent
        if sum(r.committed) != s["pot"] or r.owed(p) != s["owed"]:
            raise RuntimeError(f"replay out of step at deal {h['deal']}: {s} vs committed {r.committed}")
        if s["street"] == 3 and flop_pot == 0:
            flop_pot = s["pot"]
        if s["actor"] == "agent":
            decisions += 1
            off_tree += s.get("source") == "off_tree"
            if pending is not None:
                responses.append((pending[0], pending[1], s["action"] == "fold"))
                pending = None
            if s["street"] == 0 and s["action"] == "bet":
                agent_raised_pre = True
        elif s["action"] == "bet" and s["street"] > 0:
            d = s["add"]
            stackp = r.remaining(p)
            allin = d >= stackp
            grid_amts = {r.amount(p, f) for f in r.grid_fracs(p, grid)}
            ok = allin or d in grid_amts
            lbr_bets += 1
            on += ok
            off += not ok
            if allin:
                frac_counts["allin"] += 1
            else:
                for f in LBR_FRACS:
                    if min(r.amount(p, f), stackp) == d:
                        frac_counts[f"{f:.1f}"] += 1
                        break
            pending = (STREET[s["street"]], ok)
        r.apply(p, s["action"], s["add"])
    postflop = any(s["street"] > 0 for s in steps)
    last = steps[-1]
    if not postflop:
        e = 0
    elif last["action"] == "fold":
        e = 1 if last["actor"] == "agent" else 2
    else:
        e = 3
    z = 0 if lbr_bets == 0 else (2 if off else 1)
    p_ = 2 if not postflop else (1 if agent_raised_pre else 0)
    return {
        "deal": h["deal"], "seat": agent, "net": h["net_bb"], "z": z, "e": e, "p": p_,
        "end_street": STREET[last["street"]], "flop_pot": flop_pot / st["bb"],
        "final_pot": sum(r.committed) / st["bb"], "off_tree": off_tree, "decisions": decisions,
        "f05": frac_counts["0.5"], "f10": frac_counts["1.0"], "f20": frac_counts["2.0"], "fai": frac_counts["allin"],
        "responses": responses,
    }


def cmd_trace(args) -> int:
    import parciver_engine as pe

    manifest_path = Path(args.benchmark)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sc.check_manifest(manifest)
    st, lb = manifest["stakes"], manifest["lbr"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.bot:
        label, path = spec.split("=", 1)
        cand = Path(path).resolve()
        sha = sc.sha256_file(cand)
        saved = pe.SavedBlueprint(manifest["abstraction"]["dir"], str(cand))
        grid = GRIDS[saved.grid or "100bb"]
        agent = saved.play_agent()
        if args.translation != "snap":
            agent = agent.with_translation(args.translation)
        kw = {} if args.sizes == "fixed" else {"lbr_sizes": args.sizes}
        if args.peek == "no":
            kw["lbr_peek"] = False
        rows: list[dict] = []
        t0 = time.time()
        for first in range(0, args.deals, args.chunk):
            deals = list(range(first, min(first + args.chunk, args.deals)))
            hands = agent.lbr_trace(deals, st["sb"], st["bb"], st["stack"], seed=lb["seed"], threads=args.threads, **kw)
            rows.extend(label_hand(h, st, grid) for h in hands)
            print(f"  {label}: {len(rows) // 2}/{args.deals} deals, {time.time() - t0:.0f}s", flush=True)
        rows.sort(key=lambda x: (x["deal"], x["seat"]))
        per_deal = np.zeros(args.deals)
        for x in rows:
            per_deal[x["deal"]] += x["net"]
        gate = "no receipt of this blueprint in the benchmark dir"
        if not kw and args.translation == "snap":
            for rp in sorted(manifest_path.parent.glob("receipt-*.json")):
                rec = json.loads(rp.read_text(encoding="utf-8"))
                c = rec.get("candidate", {})
                if c.get("sha256", "").upper() != sha.upper() or c.get("search") or c.get("translation"):
                    continue
                ref = np.array(rec["raw"]["lbr"]["per_deal_bb"])
                n = min(len(ref), args.deals)
                ok = bool(np.all(np.abs(per_deal[:n] - ref[:n]) < 1e-9))
                gate = f"{'MATCHES' if ok else 'DIFFERS FROM'} {rp.name} on deals 0..{n - 1}"
                if not ok:
                    print(f"GATE FAILED: {label}: {gate}", file=sys.stderr)
                    return 1
                break
        resp = [(i, *t) for i, x in enumerate(rows) for t in x["responses"]]
        cols = [k for k in rows[0] if k != "responses"]
        name = (label + ("" if args.sizes == "fixed" else f"-{args.sizes}") + ("" if args.peek == "yes" else "-nopeek")
                + ("" if args.translation == "snap" else f"-{args.translation}"))
        np.savez_compressed(
            out_dir / f"{name}.npz",
            **{k: np.array([x[k] for x in rows]) for k in cols},
            resp=np.array(resp, dtype=np.int64).reshape(-1, 4),
            meta=json.dumps({"label": label, "path": str(cand), "sha256": sha, "grid": saved.grid,
                             "sizes": args.sizes, "peek": args.peek, "translation": args.translation,
                             "deals": args.deals, "gate": gate,
                             "benchmark": manifest["benchmark_id"], "seed": lb["seed"], "stakes": st}),
        )
        bb100 = per_deal.mean() / 2 * 100
        se = per_deal.std(ddof=1) / math.sqrt(args.deals) / 2 * 100
        print(f"{name}: LBR {bb100:+.1f} ± {1.96 * se:.1f} bb/100 over {args.deals} deals; gate: {gate}", flush=True)
    return 0


def load(out_dir: Path, name: str) -> dict:
    z = np.load(out_dir / f"{name}.npz")
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def per_deal_class(d: dict, key: str, k: int, n: int) -> np.ndarray:
    """[deals, k] per-deal net of the hands in each class (bb)."""
    out = np.zeros((n, k))
    np.add.at(out, (d["deal"], d[key]), d["net"])
    return out


def ci(x: np.ndarray) -> tuple[float, float]:
    """Mean and 1.96 SE of per-deal values, in bb/100 (2 hands per deal)."""
    return x.mean() / 2 * 100, 1.96 * x.std(ddof=1) / math.sqrt(len(x)) / 2 * 100


def cmd_report(args) -> int:
    out_dir = Path(args.out_dir)
    a_name, b_name = args.pair.split(":")
    a, b = load(out_dir, a_name), load(out_dir, b_name)
    n = min(a["meta"]["deals"], b["meta"]["deals"])
    for d in (a, b):
        keep = d["deal"] < n
        for k in list(d):
            if k not in ("meta", "resp") and isinstance(d[k], np.ndarray) and len(d[k]) == len(keep):
                d[k] = d[k][keep]
    tot_a = np.bincount(a["deal"], weights=a["net"], minlength=n)
    tot_b = np.bincount(b["deal"], weights=b["net"], minlength=n)
    g, gse = ci(tot_a - tot_b)
    print(f"{a_name} vs {b_name}, {n} deals ({a['meta']['benchmark']}, sizes {a['meta']['sizes']}/{b['meta']['sizes']})")
    print(f"  LBR {a_name} {ci(tot_a)[0]:+.1f} ± {ci(tot_a)[1]:.1f}   {b_name} {ci(tot_b)[0]:+.1f} ± {ci(tot_b)[1]:.1f}")
    print(f"  gap G = {g:+.1f} ± {gse:.1f} bb/100")
    post_a, post_b = per_deal_total(a, n, a["e"] > 0), per_deal_total(b, n, b["e"] > 0)
    pm, ph = ci(post_a - post_b)
    print(f"  postflop (E0 excluded): LBR {a_name} {ci(post_a)[0]:+.1f} ± {ci(post_a)[1]:.1f}   {b_name} "
          f"{ci(post_b)[0]:+.1f} ± {ci(post_b)[1]:.1f}   gap {pm:+.1f} [{pm - ph:+.1f}, {pm + ph:+.1f}] bb/100\n")
    for key, names in (("z", Z_NAMES), ("e", E_NAMES), ("p", P_NAMES), ("seat", SEAT_NAMES)):
        k = len(names)
        ca, cb = per_deal_class(a, key, k, n), per_deal_class(b, key, k, n)
        print(f"  axis {key}: class               share {a_name:>7s} {b_name:>7s}    contrib {a_name:>7s} {b_name:>7s}   gap (CI)            of G")
        for c in range(k):
            sa, sb_ = (a[key] == c).mean() * 100, (b[key] == c).mean() * 100
            m, h = ci(ca[:, c] - cb[:, c])
            print(f"    {names[c]:36s} {sa:6.1f}% {sb_:6.1f}%    {ci(ca[:, c])[0]:+8.1f} {ci(cb[:, c])[0]:+8.1f}"
                  f"   {m:+7.1f} [{m - h:+.1f}, {m + h:+.1f}]  {100 * m / g if g else float('nan'):5.0f}%")
        print()
    # Descriptive: agent fold rate facing LBR postflop bets, by street and grid status.
    print("  agent fold rate facing an LBR postflop bet/raise (street, on/off grid): n, fold%")
    for d, name in ((a, a_name), (b, b_name)):
        r = d["resp"]
        cells = []
        for s in (1, 2, 3):
            for ok in (1, 0):
                m = (r[:, 1] == s) & (r[:, 2] == ok)
                cells.append(f"{STREET_NAMES[s][0]}{'on' if ok else 'off'} {m.sum():6d} {100 * r[m, 3].mean() if m.any() else float('nan'):5.1f}%")
        print(f"    {name:8s} " + " | ".join(cells))
    for d, name in ((a, a_name), (b, b_name)):
        tot = d["decisions"].sum()
        fr = {k: int(d[k].sum()) for k in ("f05", "f10", "f20", "fai")}
        print(f"  {name}: agent off-tree decisions {100 * d['off_tree'].sum() / tot:.2f}% of {tot};"
              f" LBR postflop bet sizes {fr}; mean flop pot {d['flop_pot'][d['p'] < 2].mean():.1f}bb")
    return 0


def per_deal_total(d: dict, n: int, mask=None) -> np.ndarray:
    keep = d["deal"] < n if mask is None else (d["deal"] < n) & mask
    return np.bincount(d["deal"][keep], weights=d["net"][keep], minlength=n)


def cmd_delta(args) -> int:
    """Gap a-b against gap c-d on the same deals, paired per deal (all hands, and postflop only:
    hands that ended preflop play identically under both LBR size options)."""
    out_dir = Path(args.out_dir)
    (a, b), (c, d) = [[load(out_dir, x) for x in p.split(":")] for p in (args.pair, args.vs)]
    n = min(x["meta"]["deals"] for x in (a, b, c, d))
    for label, post in (("all hands", False), ("postflop only (E0 excluded)", True)):
        tot = [per_deal_total(x, n, (x["e"] > 0) if post else None) for x in (a, b, c, d)]
        g1, g2 = tot[0] - tot[1], tot[2] - tot[3]
        (m1, h1), (m2, h2), (md, hd) = ci(g1), ci(g2), ci(g1 - g2)
        print(f"{label}: G {args.pair} {m1:+.1f} ± {h1:.1f} | G {args.vs} {m2:+.1f} ± {h2:.1f}"
              f" | dG {md:+.1f} [{md - hd:+.1f}, {md + hd:+.1f}] = {100 * md / m1 if m1 else float('nan'):.0f}% of G;"
              f" G_vs/G = {m2 / m1 if m1 else float('nan'):.2f}")
        for x, name in zip((a, b, c, d), (*args.pair.split(":"), *args.vs.split(":"))):
            m, h = ci(per_deal_total(x, n, (x["e"] > 0) if post else None))
            print(f"    LBR {name:12s} {m:+7.1f} ± {h:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--benchmark", required=True)
    t.add_argument("--bot", action="append", required=True, help="label=blueprint.bin (repeatable)")
    t.add_argument("--deals", type=int, default=100_000)
    t.add_argument("--chunk", type=int, default=5000)
    t.add_argument("--threads", type=int, default=0)
    t.add_argument("--sizes", default="fixed", choices=["fixed", "grid"])
    t.add_argument("--peek", default="yes", choices=["yes", "no"],
                   help="no: LBR does not read the agent's seeded translation of its bets")
    t.add_argument("--translation", default="snap", choices=["snap", "standin"],
                   help="how the agent reads bets above its largest grid size (TRANSLATE.md)")
    t.add_argument("--out-dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--pair", required=True, help="a:b (npz names), gap = a - b")
    dl = sub.add_parser("delta")
    dl.add_argument("--out-dir", required=True)
    dl.add_argument("--pair", required=True, help="a:b")
    dl.add_argument("--vs", required=True, help="c:d; prints (a-b) - (c-d)")
    args = ap.parse_args(argv)
    return {"trace": cmd_trace, "report": cmd_report, "delta": cmd_delta}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
