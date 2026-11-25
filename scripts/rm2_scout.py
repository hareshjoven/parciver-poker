# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""RM2 scout (IDEAS.md, RM2: "measure a preflop leak first"): how our big blind met Slumbot's 2bb open in the
50,000-hand run (records/slumbot/hands.jsonl.xz). Descriptive only; it decides nothing.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -I scripts/rm2_scout.py [--json records/rm2/scout.json]

Slumbot's API shows its hole cards in every hand, folds included, so each hand our big blind folded to the 2bb
open has an exact preflop all-in equity against the hand Slumbot actually held. The check-down counterfactual
"call, then check every street to showdown" then scores (pot 4bb) 4*eq - 2 bb, against -1 bb for the fold:
a gain of 4*eq - 1 bb per hand. Slumbot bets and folds postflop, so that is not what calling would earn; the
same check-down score of the hands we did call is compared with what they really earned (all-in adjusted, as
in scripts/slumbot_report.py) to show how far real play is from it. Counted hands only (the pilot is out).
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import parciver_engine as pe  # noqa: E402

from parciver import slumbot as S  # noqa: E402
from slumbot_report import adjusted  # noqa: E402

Z = 1.959963984540054
RANKS = "23456789TJQKA"


def mean_ci(xs: list[float]) -> dict:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return dict(n=n, mean=m, lo=m - Z * sd / math.sqrt(n), hi=m + Z * sd / math.sqrt(n))


def hand_class(hole: list[str]) -> str:
    a, b = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default=str(ROOT / "records/slumbot/hands.jsonl.xz"), help="hands.jsonl or .jsonl.xz")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    opener = lzma.open if args.hands.endswith(".xz") else open  # a run folder's hands.jsonl, or the record's xz
    with opener(args.hands, "rt", encoding="utf-8") as f:
        hands = [json.loads(line) for line in f if line.strip()]
    counted = [h for h in hands if h["phase"] == "counted" and not h.get("lost")]
    n_bb = sum(h["seat"] == "BB" for h in counted)
    spots = []  # our BB's first decision facing Slumbot's open to 2bb
    for h in counted:
        if h["seat"] != "BB" or not h["decisions"] or h["decisions"][0]["at"] != "b200":
            continue
        d = h["decisions"][0]
        probs = {o[0]: o[2] for o in d["options"] if o[0] in ("fold", "call")}
        spots.append(dict(h=h, act=d["options"][d["pick"]][0], p_fold=probs.get("fold", 0.0)))
    assert all(s["h"].get("bot_hole") for s in spots), "a hand without Slumbot's cards"

    def eq(s):
        h = s["h"]
        return pe.equity_vs_hand([S.card_index(c) for c in h["hole"]], [S.card_index(c) for c in h["bot_hole"]], [])

    with ThreadPoolExecutor(8) as ex:
        for s, e in zip(spots, ex.map(eq, spots)):
            s["eq"] = e

    out: dict = dict(hands_counted=len(counted), bb_hands=n_bb, facing_2bb=len(spots))
    # the policy's own fold frequency (mean probability) and the sampled actions
    out["policy_fold_mean"] = sum(s["p_fold"] for s in spots) / len(spots)
    out["actions"] = {a: sum(s["act"] == a for s in spots) for a in sorted({s["act"] for s in spots})}
    folds = [s for s in spots if s["act"] == "fold"]
    calls = [s for s in spots if s["act"] == "call"]
    out["fold"] = dict(
        n=len(folds),
        eq=mean_ci([s["eq"] for s in folds]),
        share_eq_ge_25=sum(s["eq"] >= 0.25 for s in folds) / len(folds),
        share_eq_ge_33=sum(s["eq"] >= 1 / 3 for s in folds) / len(folds),
        share_eq_ge_40=sum(s["eq"] >= 0.40 for s in folds) / len(folds),
        checkdown_gain_bb=mean_ci([4 * s["eq"] - 1 for s in folds]),
    )
    # how far real postflop play is from the check-down score, on the hands we called
    real = [adjusted(s["h"])[0] / 100 for s in calls]
    cd = [4 * s["eq"] - 2 for s in calls]
    out["call"] = dict(
        n=len(calls),
        eq=mean_ci([s["eq"] for s in calls]),
        checkdown_bb=mean_ci(cd),
        real_adjusted_bb=mean_ci(real),
        real_minus_checkdown_bb=mean_ci([r - c for r, c in zip(real, cd)]),
    )
    # by our fold probability: hands the policy folds only sometimes are the margin
    bands = [(0.0, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, 1.01)]
    out["by_p_fold"] = []
    for lo, hi in bands:
        b = [s for s in spots if lo <= s["p_fold"] < hi]
        if not b:
            continue
        fb = [s for s in b if s["act"] == "fold"]
        cb = [s for s in b if s["act"] == "call"]
        out["by_p_fold"].append(dict(
            p_fold=[lo, min(hi, 1.0)], n=len(b), folded=len(fb), eq_all=mean_ci([s["eq"] for s in b])["mean"],
            fold_checkdown_gain_bb=mean_ci([4 * s["eq"] - 1 for s in fb]) if len(fb) > 1 else None,
            call_real_bb=mean_ci([adjusted(s["h"])[0] / 100 for s in cb]) if len(cb) > 1 else None,
            call_checkdown_bb=mean_ci([4 * s["eq"] - 2 for s in cb]) if len(cb) > 1 else None,
        ))
    # classes folded most often (count of folds), with their mean equity against Slumbot's real hands
    cls: dict[str, list[float]] = {}
    for s in folds:
        cls.setdefault(hand_class(s["h"]["hole"]), []).append(s["eq"])
    top = sorted(cls.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    out["folded_classes_by_eq"] = [dict(cls=k, n=len(v), eq=sum(v) / len(v)) for k, v in top if len(v) >= 20][:25]
    # what folding cost in the bb/100 of the whole BB seat, at the check-down score (an upper-ish
    # indicator, not a cost estimate): total check-down gain of the folds / BB hands * 100
    out["fold_checkdown_gain_bb100_bb_seat"] = sum(4 * s["eq"] - 1 for s in folds) / n_bb * 100

    text = json.dumps(out, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
