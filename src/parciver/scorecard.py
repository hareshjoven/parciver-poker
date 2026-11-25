# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Bot scorecard: is the playing bot getting better, and at what cost?

Measures the WHOLE bot at the real-table boundary (the play agent: blueprint +
action translation + fallback, and later search), not a blueprint in isolation.

* Headline, STRENGTH: the candidate bot's duplicate bb/100 against the frozen
  baseline bot. The baseline scores exactly 0 against itself (identical play in
  both seatings cancels).
* Guard, EXPLOITABILITY: local best response against the candidate, a lower
  bound on how exploitable it is (lower is better). Shown beside the headline,
  never folded into it: a bot can win more against the field while becoming
  easier to exploit, and the report must show that. Also split into hands
  whose betting ended before the flop and the rest (POSTFLOP): LBR only checks
  or calls preflop, so the preflop part is not exploitation (it calls a
  preflop shove with any two cards, V2LEAK.md) and differs between bots that
  shove preflop at different rates; the postflop part compares them fairly.
* Guard, OFF-TREE: share of the candidate's decisions in the headline match that
  left the blueprint (off-tree or missing key). Decisions made by real-time
  search are neither (source index 3) and do not count as off-tree.
* Rows: the same match against fixed opponents built from the BASELINE (grid
  leaks: nit / station / maniac) and off-grid players (min-raiser, the baseline
  with its bet sizes rescaled, and with realistic sizes), so a headline gain
  cannot hide where losses are. Rows where the candidate left the blueprint
  show their off-tree share.

Every row uses fixed seeds and fixed deals, so any two checkpoints of the same
benchmark are compared on identical cards: the change since a previous receipt
is a paired per-block difference. Uncertainty: `BLOCKS` block means per row,
standard error with ddof=1, approximate 95% normal interval. The status line is
a development diagnostic on this benchmark, not a promotion decision.

This module is pure bookkeeping and statistics (no engine calls); the CLI is
`scripts/scorecard.py`.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass

SCHEMA = 1


@functools.lru_cache(maxsize=None)
def t95(df: int) -> float:
    """The two-sided 95% Student t quantile at `df` degrees of freedom (2.0096
    at 49, 50 blocks; the author's choice, 4 Oct: z 1.96 was ~2.5% too narrow).
    Bisection on the CDF, the density integrated by Simpson's rule."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def tail(t):  # P(0 < T < t)
        n = 4000
        f = [c * (1 + (t * k / n) ** 2 / df) ** (-(df + 1) / 2) for k in range(n + 1)]
        return t / n / 3 * (f[0] + f[n] + 4 * sum(f[1:n:2]) + 2 * sum(f[2:n:2]))
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) < 0.475 else (lo, mid)
    return (lo + hi) / 2


@dataclass(frozen=True)
class Row:
    """One fixed matchup. `kind`: agent (the baseline bot), leak (baseline leak
    policy), builtin (engine baseline opponent by name), perturbed (baseline bot
    with bet sizes scaled by `perturb`), realistic (baseline bot with each size
    redrawn from the sizes people and other solvers use)."""

    name: str
    kind: str
    deals: int
    seed: int
    family: str | None = None
    strength: float | None = None
    builtin: str | None = None
    perturb: float | None = None


HEADLINE = "baseline"

# v2 adds the realistic-sizing row; a v1 benchmark keeps its own rows (they
# live in its manifest), so v1 histories stay valid and never mix with v2.
BENCHMARK_ID = "scorecard-v2"
DEFAULT_ROWS = (
    Row(HEADLINE, "agent", 50_000, 410_001),
    Row("nit(0.20)", "leak", 25_000, 410_019, family="nit", strength=0.20),
    Row("station(0.20)", "leak", 25_000, 410_043, family="station", strength=0.20),
    Row("maniac(0.20)", "leak", 25_000, 410_069, family="maniac", strength=0.20),
    Row("min_raiser", "builtin", 25_000, 410_083, builtin="min_raiser"),
    Row("baseline_sizes_x0.6", "perturbed", 25_000, 410_091, perturb=0.6),
    Row("baseline_sizes_x1.5", "perturbed", 25_000, 410_097, perturb=1.5),
    Row("baseline_realistic_sizes", "realistic", 25_000, 410_107),
)
DEFAULT_LBR = {"deals": 5_000, "seed": 410_101}
BLOCKS = 50
STAKES = {"sb": 50, "bb": 100, "stack": 10_000}


class ScorecardError(ValueError):
    """Raised instead of producing a score from invalid or incomplete data."""


# ---------------------------------------------------------------- statistics


def summarize(blocks: list[float]) -> dict:
    """Mean of block values with a ddof=1 standard error and a 95% t interval."""
    b = len(blocks)
    if b < 2 or not all(math.isfinite(x) for x in blocks):
        raise ScorecardError(f"need >= 2 finite blocks, got {b}")
    mean = sum(blocks) / b
    var = sum((x - mean) ** 2 for x in blocks) / (b - 1)
    se = math.sqrt(var / b)
    q = t95(b - 1)
    return {"mean": mean, "se": se, "lo": mean - q * se, "hi": mean + q * se, "blocks": b}


def to_blocks(per_deal_bb: list[float], blocks: int, hands_per_deal: int) -> list[float]:
    """Split per-deal net big blinds into `blocks` equal blocks, each as bb/100 hands."""
    n = len(per_deal_bb)
    if n == 0 or n % blocks:
        raise ScorecardError(f"{n} deals do not split into {blocks} equal blocks")
    size = n // blocks
    return [sum(per_deal_bb[i * size:(i + 1) * size]) / size * 100.0 / hands_per_deal
            for i in range(blocks)]


def status(lo: float, hi: float, higher_is_better: bool = True) -> str:
    """Descriptive direction of a change interval (never 'equal')."""
    if lo > 0:
        return "UP ON THIS BENCHMARK" if higher_is_better else "MORE EXPLOITABLE ON THIS BENCHMARK"
    if hi < 0:
        return "DOWN ON THIS BENCHMARK" if higher_is_better else "LESS EXPLOITABLE ON THIS BENCHMARK"
    return "UNCLEAR / NO RESOLVED CHANGE"


def paired_change(current: list[float], previous: list[float]) -> dict:
    """Per-block paired difference current - previous (same fixed deals)."""
    if len(current) != len(previous):
        raise ScorecardError("block counts differ between receipts")
    d = [a - b for a, b in zip(current, previous)]
    if all(x == 0 for x in d):
        return {"mean": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0, "blocks": len(d)}
    return summarize(d)


# ---------------------------------------------------------------- validation


def validate_measurement(result: dict, deals: int, blocks: int, hands_per_deal: int) -> list[float]:
    """Check an engine result and return its block values (bb/100)."""
    per_deal = result.get("per_deal_bb")
    if per_deal is None or len(per_deal) != deals:
        raise ScorecardError(f"expected {deals} deals, got {None if per_deal is None else len(per_deal)}")
    if not all(math.isfinite(x) for x in per_deal):
        raise ScorecardError("non-finite deal result")
    if not math.isfinite(result.get("bb_per_100", float("nan"))):
        raise ScorecardError("non-finite summary")
    b = to_blocks(per_deal, blocks, hands_per_deal)
    if abs(sum(b) / len(b) - result["bb_per_100"]) > 1e-6 * max(1.0, abs(result["bb_per_100"])):
        raise ScorecardError("block mean disagrees with the engine summary")
    return b


MATCH_COUNTS = ("a_search_failed", "b_search_failed")  # match_against: the candidate, the other side
LBR_COUNTS = ("search_failed", "model_search_failed")  # lbr: the candidate, LBR's model of it


def check_search_failures(result: dict, keys: tuple[str, ...], where: str, counted: bool) -> None:
    """Refuse an engine result in which any decision reached a failed search (each
    count is [built-in solver redid it, blueprint decided]): the agent measured is
    then not the one named (as search_h2h / search_lbr refuse). `counted`: the
    counts must be present (a search-enabled candidate; engines before 1 Oct do
    not count)."""
    missing = [k for k in keys if k not in result]
    if counted and missing:
        raise ScorecardError(f"{where}: no {', '.join(missing)} (the engine predates search-failure counts)")
    bad = {k: list(result[k]) for k in keys if k in result and any(result[k])}
    if bad:
        raise ScorecardError(f"{where}: decisions reached a failed search {bad} "
                             f"[built-in solver, blueprint]; the run stops and keeps nothing")


LBR_PARTS = ("postflop", "preflop")


def lbr_parts(per_deal_bb: list[float], per_deal_preflop_bb: list[float] | None, blocks: int,
              hands_per_deal: int) -> dict | None:
    """Block values (bb/100) of LBR's two parts: `preflop`, the hands whose betting
    ended before the flop, and `postflop`, the rest. None without the split
    (results from engines before it was recorded)."""
    pre = per_deal_preflop_bb
    if pre is None:
        return None
    if len(pre) != len(per_deal_bb):
        raise ScorecardError(f"LBR preflop part has {len(pre)} deals, the total {len(per_deal_bb)}")
    if not all(math.isfinite(x) for x in pre):
        raise ScorecardError("non-finite LBR preflop part")
    post = [t - p for t, p in zip(per_deal_bb, pre)]
    return {"postflop": to_blocks(post, blocks, hands_per_deal), "preflop": to_blocks(pre, blocks, hands_per_deal)}


def lbr_split_of(receipt: dict, sidecar: dict | None = None) -> list[float] | None:
    """A receipt's per-deal LBR preflop part: recorded in the receipt, or from a
    sidecar (`scripts/scorecard.py lbr-split`) that reproduced its LBR exactly."""
    pre = receipt.get("raw", {}).get("lbr", {}).get("per_deal_preflop_bb")
    if pre is not None or sidecar is None:
        return pre
    if sidecar.get("receipt_hash") != canonical_hash(receipt):
        raise ScorecardError("LBR split sidecar belongs to another receipt")
    pre = sidecar["per_deal_preflop_bb"]
    if len(pre) != len(receipt["raw"]["lbr"]["per_deal_bb"]):
        raise ScorecardError("LBR split sidecar has the wrong number of deals")
    return pre


def lbr_part_scores(receipt: dict, split: list[float] | None, hands_per_deal: int = 2) -> dict:
    """`exploitability_lbr_<part>` scores of a receipt: its own, else from `split`
    (then also checked against the receipt's total LBR blocks)."""
    s = receipt["scores"]
    have = {f"exploitability_lbr_{p}": s[f"exploitability_lbr_{p}"] for p in LBR_PARTS
            if f"exploitability_lbr_{p}" in s}
    if have or split is None:
        return have
    per_deal, total = receipt["raw"]["lbr"]["per_deal_bb"], s["exploitability_lbr"]["blocks"]
    if to_blocks(per_deal, len(total), hands_per_deal) != total:
        raise ScorecardError("receipt LBR blocks do not match its per-deal results")
    parts = lbr_parts(per_deal, split, len(total), hands_per_deal)
    return {f"exploitability_lbr_{p}": {"blocks": b, **{k: v for k, v in summarize(b).items() if k != "blocks"}}
            for p, b in parts.items()}


# ---------------------------------------------------------------- identities


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def make_manifest(benchmark_id: str, baseline: dict, abstraction: dict, rows=DEFAULT_ROWS,
                  lbr=DEFAULT_LBR, blocks: int = BLOCKS, stakes=STAKES) -> dict:
    for r in rows:
        if r.deals % blocks:
            raise ScorecardError(f"row {r.name}: {r.deals} deals not divisible by {blocks} blocks")
    if lbr["deals"] % blocks:
        raise ScorecardError("LBR deals not divisible by the block count")
    if sum(r.name == HEADLINE for r in rows) != 1:
        raise ScorecardError("exactly one headline row required")
    body = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "baseline": baseline,
        "abstraction": abstraction,
        "stakes": dict(stakes),
        "rows": [asdict(r) for r in rows],
        "lbr": dict(lbr),
        "blocks": blocks,
        "hands_per_deal": 2,
    }
    return {**body, "manifest_hash": canonical_hash(body)}


def check_manifest(manifest: dict) -> None:
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if canonical_hash(body) != manifest.get("manifest_hash"):
        raise ScorecardError("benchmark manifest was edited (hash mismatch)")


def rows_of(manifest: dict) -> list[Row]:
    return [Row(**r) for r in manifest["rows"]]


# ---------------------------------------------------------------- receipts


def score_receipt(manifest: dict, measurements: dict, lbr_result: dict, headline_sources: list[int]) -> dict:
    """Validate every measurement and compute the scores. `measurements` maps row
    name -> engine result dict; all rows must be present (no partial headline)."""
    blocks, hpd = manifest["blocks"], manifest["hands_per_deal"]
    rows_out = {}
    for r in rows_of(manifest):
        if r.name not in measurements:
            raise ScorecardError(f"row {r.name} missing: no score without every row")
        b = validate_measurement(measurements[r.name], r.deals, blocks, hpd)
        rows_out[r.name] = {"blocks": b, **{k: v for k, v in summarize(b).items() if k != "blocks"},
                            "sources": measurements[r.name].get("a_sources")}
    lb = validate_measurement(lbr_result, manifest["lbr"]["deals"], blocks, hpd)
    total = sum(headline_sources)
    out = {
        "strength": {k: rows_out[HEADLINE][k] for k in ("mean", "se", "lo", "hi")},
        "exploitability_lbr": {"blocks": lb, **{k: v for k, v in summarize(lb).items() if k != "blocks"}},
        "off_tree_share": (headline_sources[1] + headline_sources[2]) / total if total else None,
        "rows": rows_out,
    }
    parts = lbr_parts(lbr_result["per_deal_bb"], lbr_result.get("per_deal_preflop_bb"), blocks, hpd)
    for p, b in (parts or {}).items():
        out[f"exploitability_lbr_{p}"] = {"blocks": b, **{k: v for k, v in summarize(b).items() if k != "blocks"}}
    return out


def compare(receipt: dict, previous: dict, previous_split: list[float] | None = None,
            hands_per_deal: int = 2) -> dict:
    """Paired changes vs a previous completed receipt of the SAME benchmark.
    `previous_split`: the previous receipt's LBR preflop part when the receipt
    itself predates the split (`lbr_split_of`)."""
    if previous.get("state") != "complete":
        raise ScorecardError("previous receipt is not complete")
    if previous.get("manifest_hash") != receipt["manifest_hash"]:
        raise ScorecardError("previous receipt is from a different benchmark definition")
    s_cur, s_prev = receipt["scores"], previous["scores"]
    strength = paired_change(s_cur["rows"][HEADLINE]["blocks"], s_prev["rows"][HEADLINE]["blocks"])
    lbr = paired_change(s_cur["exploitability_lbr"]["blocks"], s_prev["exploitability_lbr"]["blocks"])
    rows = {}
    for name, row in s_cur["rows"].items():
        prev_row = s_prev["rows"].get(name)
        if prev_row is not None:
            c = paired_change(row["blocks"], prev_row["blocks"])
            rows[name] = {**c, "status": status(c["lo"], c["hi"], True)}
    out = {
        "previous_label": previous["candidate"]["label"],
        "previous_receipt_hash": canonical_hash(previous),
        "strength": {**strength, "status": status(strength["lo"], strength["hi"], True)},
        "exploitability_lbr": {**lbr, "status": status(lbr["lo"], lbr["hi"], False)},
        "rows": rows,
    }
    # Postflop LBR only: the preflop part is not exploitation (see the module doc).
    key = "exploitability_lbr_postflop"
    prev_parts = lbr_part_scores(previous, previous_split, hands_per_deal)
    if key in s_cur and key in prev_parts:
        c = paired_change(s_cur[key]["blocks"], prev_parts[key]["blocks"])
        out[key] = {**c, "status": status(c["lo"], c["hi"], False)}
    return out


def mean_se(per_deal_bb: list[float], hands_per_deal: int) -> tuple[float, float]:
    """bb/100 mean and per-deal standard error, as the engine reports them."""
    n = len(per_deal_bb)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(per_deal_bb) / n
    var = sum((x - mean) ** 2 for x in per_deal_bb) / (n - 1)
    scale = 100.0 / hands_per_deal
    return mean * scale, math.sqrt(var / n) * scale


def merge_chunks(parts: list[dict], hands_per_deal: int) -> dict:
    """One engine result from consecutive `first_deal` chunks of the same run."""
    out = dict(parts[0])
    out["per_deal_bb"] = [x for p in parts for x in p["per_deal_bb"]]
    # Per-deal lists some results lack: kept where every chunk has them.
    for key in ("per_deal_preflop_bb", "per_deal_bb_cv"):
        out.pop(key, None)
        if all(key in p for p in parts):
            out[key] = [x for p in parts for x in p[key]]
    for key in ("a_sources", "b_sources"):
        if key in out:
            out[key] = [sum(v) for v in zip(*(p[key] for p in parts))]
    # Counted only by newer engines: kept where every chunk has it.
    for key in ("a_search_failed", "b_search_failed", "search_failed", "model_search_failed"):
        out.pop(key, None)
        if all(key in p for p in parts):
            out[key] = [sum(v) for v in zip(*(p[key] for p in parts))]
    if "hands" in out:
        out["hands"] = sum(p["hands"] for p in parts)
    out["bb_per_100"], out["se"] = mean_se(out["per_deal_bb"], hands_per_deal)
    return out


def summary_of(receipt: dict, split: list[float] | None = None) -> dict:
    """Compact view of a completed receipt (for history tables and the dashboard).
    `split`: the LBR preflop part of a receipt that predates it (`lbr_split_of`)."""
    s = receipt["scores"]
    pick = lambda d: {k: d[k] for k in ("mean", "lo", "hi")}  # noqa: E731
    out = {
        "label": receipt["candidate"]["label"],
        "started_utc": receipt["started_utc"],
        "search": receipt["candidate"].get("search"),
        "strength": pick(s["strength"]),
        "lbr": pick(s["exploitability_lbr"]),
        "off_tree_share": s["off_tree_share"],
        "rows": {name: pick(r) for name, r in s["rows"].items()},
    }
    for key, v in lbr_part_scores(receipt, split).items():
        out[key.replace("exploitability_", "")] = pick(v)
    return out


def render(receipt: dict) -> str:
    s = receipt["scores"]
    st, ex = s["strength"], s["exploitability_lbr"]
    lines = [
        f"Strength vs frozen baseline bot: {st['mean']:+.1f} bb/100   (approx. 95% [{st['lo']:+.1f}, {st['hi']:+.1f}])",
        f"Exploitability (LBR, lower is better): {ex['mean']:+.1f} bb/100   [{ex['lo']:+.1f}, {ex['hi']:+.1f}]",
    ]
    if "exploitability_lbr_postflop" in s:
        po, pr = s["exploitability_lbr_postflop"], s["exploitability_lbr_preflop"]
        lines.append(f"  of which postflop (exploitation): {po['mean']:+.1f} [{po['lo']:+.1f}, {po['hi']:+.1f}]; "
                     f"preflop-ended hands (LBR only calls): {pr['mean']:+.1f} [{pr['lo']:+.1f}, {pr['hi']:+.1f}]")
    if s["off_tree_share"] is not None:
        lines.append(f"Off-tree / fallback decisions: {100 * s['off_tree_share']:.1f}%")
    ch = receipt.get("change")
    if ch:
        c, e = ch["strength"], ch["exploitability_lbr"]
        lines.append(f"Change vs {ch['previous_label']}: strength {c['mean']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}] "
                     f"-> {c['status']}; LBR {e['mean']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}] -> {e['status']}")
        if "exploitability_lbr_postflop" in ch:
            e = ch["exploitability_lbr_postflop"]
            lines.append(f"  postflop LBR {e['mean']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}] -> {e['status']}")
    else:
        lines.append("Change: NO PREVIOUS COMPARISON")
    lines.append("Rows (candidate bb/100 vs each opponent):")
    for name, r in s["rows"].items():
        line = f"  {name:>24s}: {r['mean']:+8.1f}  [{r['lo']:+.1f}, {r['hi']:+.1f}]"
        off = row_off_tree(r)
        if off:
            line += f"  off-tree {100 * off:.1f}%"
        lines.append(line)
    lines.append(f"Benchmark: {receipt['benchmark_id']} | scope: the play agent at "
                 f"{stakes_text(receipt.get('stakes', STAKES))} heads-up")
    return "\n".join(lines)


def row_off_tree(row: dict) -> float | None:
    """Share of the candidate's decisions in a row that left the blueprint
    (off-tree or missing key; search decisions do not count)."""
    src = row.get("sources")
    total = sum(src) if src else 0
    return (src[1] + src[2]) / total if total else None


def stakes_text(stakes: dict) -> str:
    """'50/100, 100bb' (receipts from before stakes were recorded are 100bb)."""
    return f"{stakes['sb']}/{stakes['bb']}, {stakes['stack'] // stakes['bb']}bb"
