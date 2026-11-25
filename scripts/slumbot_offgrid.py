# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""SLUMBOT-2's descriptive reading (records/slumbot-2/PREREG.md, "Flop decisions facing an off-grid bet"): how many
of our searched flop decisions face a Slumbot bet off the grid, and how long they take. Decides nothing.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -I scripts/slumbot_offgrid.py --hands <hands.jsonl or .xz>
        [--json records/slumbot-2/offgrid.json]

Off the grid, as in scripts/v2leak.py: not an all-in, and not the amount the 200bb blueprint grid gives at the REAL
betting state (a bet: 0.5 or 1.0 pot; a raise: 1.0 of the pot after calling; the grid's formula, rounded, floors
included). So a preflop size the bot translated still has its flop bets judged against the real pot. The state comes
from parciver.slumbot.parse; it is checked against the run's own record: each decision's recorded bet options must
equal what this script computes at that state under the flop search's menu (engine/src/subgame.rs flop_menu: the same
fractions as the grid, but after two bets on the street only the all-in). That checks the replayed state and the
formula, not the grid's re-raise branch or the off-grid test itself (tests/test_slumbot_offgrid.py). Counted hands
that finished, or a mock's.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import slumbot as S  # noqa: E402

OPENS, RAISES, RERAISES, CAP = (0.5, 1.0), (1.0,), (1.0,), 3  # v2leak.GRIDS["200bb"], the blueprint's grid
BB, STACK = S.BB, S.STACK


def rnd(x: float) -> int:
    """Rust's f64::round (half away from zero) for non-negative values."""
    return int(math.floor(x + 0.5))


def grid_amounts(pot: int, owed: int, floor: int, street_bets: int) -> list[int]:
    """Chips to add for each sized grid bet or raise at a state (nlhe.rs sized_delta and sized_fracs)."""
    if owed == 0:
        return [max(rnd(f * pot), BB) for f in OPENS]
    fracs = RAISES if street_bets <= 1 else RERAISES if street_bets <= 1 + CAP else ()
    return [owed + max(rnd(f * (pot + owed)), floor) for f in fracs]


def replay_flop(at: str):
    """The real state at the end of history `at` (a flop decision) and Slumbot's last action on the flop: (pot, owed,
    floor, street_bets at the decision; the last bet as (add, all_in, off_grid, pot fraction), or None if none is
    to be answered)."""
    head, flop = at.split("/")
    s = S.parse(head + "/")
    pot = sum(s.total)
    contrib, last_raise, bets, last = [0, 0], 0, 0, None
    p, stacks = 0, [STACK - s.total[0], STACK - s.total[1]]
    i = 0
    while i < len(flop):
        c = flop[i]
        i += 1
        owed = max(contrib) - contrib[p]
        if c == "k":
            pass
        elif c == "c":
            contrib[p] += owed
            pot += owed
            stacks[p] -= owed
            last = None
        else:
            j = i
            while i < len(flop) and flop[i].isdigit():
                i += 1
            to = int(flop[j:i])
            add = to - contrib[p]
            amounts = grid_amounts(pot, owed, max(last_raise, BB), bets)
            all_in = add >= stacks[p]
            frac = (add - owed) / (pot + owed)
            last = (add, all_in, not all_in and add not in amounts, frac)
            by = to - max(contrib)
            last_raise = max(by, last_raise) if all_in else by
            bets += 1
            contrib[p] = to
            pot += add
            stacks[p] -= add
        p = 1 - p
    return pot, max(contrib) - contrib[p], max(last_raise, BB), bets, last, stacks[p]


def stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return dict(n=len(xs), median=float(np.median(xs)), p95=float(np.percentile(xs, 95)), mean=float(np.mean(xs)),
                max=float(np.max(xs)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True, help="hands.jsonl or .jsonl.xz")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    opener = lzma.open if args.hands.endswith(".xz") else open
    with opener(args.hands, "rt", encoding="utf-8") as f:
        hands = [json.loads(line) for line in f if line.strip()]

    cls, first, fracs, disagree, checked = {}, {}, [], 0, 0
    for h in hands:
        if h["phase"] not in ("counted", "mock") or h.get("lost"):  # a live run's counted hands, or a mock's
            continue
        seen = False
        for d in h.get("decisions", []):
            if d["street"] != 1 or not d.get("solves"):
                continue
            pot, owed, floor, bets, last, stack = replay_flop(d["at"])
            sized = [a for a in grid_amounts(pot, owed, floor, bets) if a < stack] if bets <= 1 else []
            want = sorted({*sized, *([stack] if stack > owed else [])})  # the flop search's menu: two bets, then all-in
            got = sorted(o[1] for o in d["options"] if o[0] == "bet")
            checked += 1
            disagree += want != got
            name = ("no bet to answer" if last is None else "all-in" if last[1] else
                    "off the grid" if last[2] else "on the grid")
            cls.setdefault(name, []).append(d["seconds"])
            first.setdefault((name, not seen), []).append(d["seconds"])
            if last is not None and last[2]:
                fracs.append(last[3])
            seen = True

    n = sum(len(v) for v in cls.values())
    facing = n - len(cls.get("no bet to answer", []))
    off = len(cls.get("off the grid", []))
    out = dict(searched_flop_decisions=n, facing_a_bet=facing, off_grid=off, off_grid_share_of_all=off / n,
               off_grid_share_of_facing=off / facing, record_check=dict(checked=checked, disagree=disagree),
               seconds={k: stats(v) for k, v in sorted(cls.items())},
               seconds_first_flop_decision={k: stats(v) for (k, f), v in sorted(first.items()) if f},
               seconds_later_flop_decision={k: stats(v) for (k, f), v in sorted(first.items()) if not f},
               off_grid_pot_fraction=dict(min=min(fracs), median=float(np.median(fracs)), max=max(fracs)))
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
