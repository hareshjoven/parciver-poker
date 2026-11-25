# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Local best response (`PlayAgent.lbr`) checked against exact values.

LBR is a legal strategy, so its winnings against a fixed agent can never exceed
that agent's exact best-response value. On the small game both are known: LBR
must stay under the bound (up to sampling error) and must find real value
against a badly exploitable agent.
"""
import math

import pytest

pe = pytest.importorskip("parciver_engine")
if not hasattr(pe, "PlayAgent") or not hasattr(pe.PlayAgent, "lbr"):
    pytest.skip("engine built without LBR", allow_module_level=True)

SB, BB, STACK = 1, 2, 4
DEALS = 40_000


@pytest.fixture(scope="module")
def strategies():
    eq = pe.solve_nlhe_small(2000)
    uniform = {k: [1.0 / len(v)] * len(v) for k, v in eq.items()}
    return {"eq": eq, "uniform": uniform}


@pytest.mark.parametrize("name", ["eq", "uniform"])
@pytest.mark.parametrize("lbr_seat", [0, 1])
def test_lbr_never_beats_the_exact_best_response(strategies, name, lbr_seat):
    strat = strategies[name]
    agent = pe.PlayAgent.from_strategy(strat, "small")
    r = agent.lbr(DEALS, SB, BB, STACK, mirror=False, seat=lbr_seat, seed=4)
    bound = pe.nlhe_small_best_response_value(strat, lbr_seat) / BB * 100
    assert r["bb_per_100"] <= bound + 4 * r["se"], (r, bound)


def test_lbr_exploits_a_uniform_agent(strategies):
    agent = pe.PlayAgent.from_strategy(strategies["uniform"], "small")
    r = agent.lbr(DEALS, SB, BB, STACK, seed=5)
    assert r["bb_per_100"] > 10 * r["se"] and r["bb_per_100"] > 20


# ------------------------------------------------------------------ hand traces

REAL = (50, 100, 10_000)


def _net_by_deal(hands):
    per = {}
    for h in hands:
        per[h["deal"]] = per.get(h["deal"], 0.0) + h["net_bb"]
    return per


@pytest.mark.parametrize("exact,solve", [(True, True), (True, False), (False, False)])
def test_trace_replays_lbr_exactly_and_diagnoses_turn_and_river(exact, solve):
    agent = pe.PlayAgent.uniform().with_search(river_iters=20, exact_own_range=exact, solve_round=solve)
    deals = list(range(30))
    whole = agent.lbr(len(deals), *REAL, seed=11)
    hands = agent.lbr_trace(deals, *REAL, seed=11)
    assert len(hands) == 2 * len(deals)
    per = _net_by_deal(hands)
    assert [per[i] for i in deals] == pytest.approx(whole["per_deal_bb"], abs=1e-12)
    diagnosed = 0
    for h in hands:
        assert len(h["cards"]) == 9 and len(set(h["cards"])) == 9
        for s in h["steps"]:
            assert s["actor"] in ("lbr", "agent")
            if s["actor"] == "agent":
                assert abs(sum(p for *_, p in s["options"]) - 1.0) < 1e-9
                assert ("diag" in s) == (s["street"] >= 4)
                if s["street"] == 5:
                    assert s["source"] == "search"
            if "diag" in s:
                d = s["diag"]
                diagnosed += 1
                assert 0.0 <= d["eq_model"] <= 1.0 and 0.0 <= d["eq_actual"] <= 1.0
                assert 0.0 <= d["own_tv"] <= 1.0
                if exact:
                    # The own range the decision assumes IS what LBR tracks
                    # (solved round: the solve's, carried by its frozen strategy).
                    assert d["own_tv"] < 1e-9, d
    assert diagnosed > 5


def test_lbr_preflop_part_is_the_hands_that_ended_preflop():
    """`per_deal_preflop_bb`: the traced hands whose last action came before the
    flop, and chunks of a run reproduce it exactly."""
    agent = pe.PlayAgent.uniform()  # folds and shoves preflop often, and plays on
    deals = list(range(60))
    whole = agent.lbr(len(deals), *REAL, seed=7)
    hands = agent.lbr_trace(deals, *REAL, seed=7)
    pre = {i: 0.0 for i in deals}
    ended = 0
    for h in hands:
        if h["steps"][-1]["street"] == 0:
            pre[h["deal"]] += h["net_bb"]
            ended += 1
    assert 10 <= ended <= len(hands) - 10
    assert whole["per_deal_preflop_bb"] == pytest.approx([pre[i] for i in deals], abs=1e-12)
    a = agent.lbr(30, *REAL, seed=7, first_deal=0)
    b = agent.lbr(30, *REAL, seed=7, first_deal=30, threads=1)
    assert a["per_deal_preflop_bb"] + b["per_deal_preflop_bb"] == whole["per_deal_preflop_bb"]
    assert a["per_deal_bb"] + b["per_deal_bb"] == whole["per_deal_bb"]


def test_standin_translation_option():
    """TRANSLATE.md: "standin" only changes how oversized bets are read, so LBR
    that bets only the agent's grid sizes (and all-in) plays it exactly like
    "snap"; with LBR's own sizes (2x pot is above every grid size) it differs,
    and both LBR models run on it. The option survives `with_search`."""
    snap = pe.PlayAgent.uniform()
    standin = snap.with_translation("standin")
    assert (snap.translation, standin.translation) == ("snap", "standin")
    assert standin.with_search(river_iters=20).translation == "standin"
    assert snap.with_search(river_iters=20).with_translation("standin").translation == "standin"
    with pytest.raises(ValueError):
        snap.with_translation("nearest")
    n = 200
    grid = [a.lbr(n, *REAL, seed=3, lbr_sizes="grid") for a in (snap, standin)]
    assert grid[0]["per_deal_bb"] == grid[1]["per_deal_bb"]
    own = [a.lbr(n, *REAL, seed=3) for a in (snap, standin)]
    assert own[0]["per_deal_bb"] != own[1]["per_deal_bb"]
    # Hands that end preflop never see an oversized bet.
    assert own[0]["per_deal_preflop_bb"] == own[1]["per_deal_preflop_bb"]
    blind = standin.lbr(n, *REAL, seed=3, lbr_peek=False)
    assert len(blind["per_deal_bb"]) == n and all(map(math.isfinite, blind["per_deal_bb"]))
    # Traces replay the run.
    hands = standin.lbr_trace(list(range(n)), *REAL, seed=3)
    per = _net_by_deal(hands)
    assert [per[i] for i in range(n)] == pytest.approx(own[1]["per_deal_bb"], abs=1e-12)


def _trace_cli():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("lbr_trace_cli", root / "scripts" / "lbr_trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trace_summary_and_receipt_matching(tmp_path):
    import json
    cli = _trace_cli()
    agent = pe.PlayAgent.uniform().with_search(river_iters=20)
    hands = agent.lbr_trace(list(range(12)), *REAL, seed=5)
    s = cli.summarize(hands, 2)
    per = _net_by_deal(hands)
    assert s["per_deal_bb"] == [per[i] for i in range(12)]
    assert s["bb_per_100"] == pytest.approx(sum(per.values()) / 12 / 2 * 100)
    # Receipts: same bot and deal count -> found; an older receipt without
    # `own_range` / `round` played the first version ("model" / "forced").
    search = {"streets": "river", "river_iters": 200, "turn_iters": 100, "own_range": "model", "round": "forced"}
    v1 = {k: v for k, v in search.items() if k not in ("own_range", "round")}
    exact = {**v1, "own_range": "exact"}
    for name, sha, srch in (("receipt-a.json", "ab", v1), ("receipt-b.json", "cd", v1), ("receipt-c.json", "ab", exact)):
        (tmp_path / name).write_text(json.dumps({"candidate": {"sha256": sha, "search": srch},
                                                 "raw": {"lbr": {"per_deal_bb": [1.0] * 12}}}))
    p, got = cli.matching_receipt(tmp_path, "AB", search, 12)
    assert p.name == "receipt-a.json" and got == [1.0] * 12
    p, _ = cli.matching_receipt(tmp_path, "AB", {**search, "own_range": "exact"}, 12)
    assert p.name == "receipt-c.json"
    assert cli.matching_receipt(tmp_path, "AB", {**search, "own_range": "exact", "round": "solve"}, 12) == (None, None)
    assert cli.matching_receipt(tmp_path, "AB", search, 13) == (None, None)
