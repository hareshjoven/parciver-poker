"""scripts/slumbot_run.py's mock check (records/slumbot/PREREG.md; LIVE-FLOP; LIVESEARCH.md, Amendment 3): the
flop, turn and river decisions' p95 bars, each street on its own; and the live search record's turn schedule
(rating.py). CPU only: fake hands, the parser check stubbed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sr = _load("slumbot_run_gates", ROOT / "scripts" / "slumbot_run.py")


def _run(tmp_path, monkeypatch, flop=1.0, turn=1.0, river=1.0, streets=(1, 2, 3), slow=None):
    """A mock run of 40 hands, each with one searched decision a street at the given seconds; `slow`: a street
    whose decisions take 2.5 s in hands 1-4 (10% of that street, 5% of turn and river pooled)."""
    monkeypatch.setattr(sr.S, "parser_check", lambda parse: dict(passed=True, prefixes=1))
    monkeypatch.setattr(sr.S, "refuses", lambda parse, action: False)
    secs = {0: 0.01, 1: flop, 2: turn, 3: river}
    hands = [dict(hand=k, session=1, winnings=10 if k % 2 else -10, check="ok", action="", seconds=1.0,
                  decisions=[dict(street=0, seconds=0.01, solves=0)]
                  + [dict(street=st, seconds=2.5 if st == slow and k <= 4 else secs[st], solves=1) for st in streets])
             for k in range(1, 41)]
    (tmp_path / "hands.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hands))
    cfg = dict(commit="c", engine_sha256="e", blueprint_sha256="b",
               search=dict(net_sha256="n", turn_iters=sr.SEARCH["turn_iters"], turn_schedule=sr.TURN_SCHEDULE,
                           river_iters=sr.SEARCH["river_iters"]))
    mock = SimpleNamespace(bot_net=0, refused=[])
    tally = SimpleNamespace(lost={"mock": 0})
    fs = SimpleNamespace(leaves="sym", solves=40, symmetric=0, drift=None)
    ok, msg = sr.mock_check(tmp_path, cfg, mock, None, tally, fs)
    return ok, msg, json.loads((tmp_path / "mock_check.json").read_text())


def test_the_live_config_is_b1():
    assert sr.SEARCH["turn_iters"] == 400 and sr.SEARCH["river_iters"] == 2000  # RIVER R1
    stage1 = json.loads((ROOT / "records" / "livesearch" / "stage1.json").read_text(encoding="utf-8"))
    assert sr.TURN_SCHEDULE == stage1["b1"]["schedule"] == dict(gamma=5, shift=1)
    assert sr.TURN_P95 == sr.RIVER_P95 == 2.0 and sr.FLOP_P95 == 3.0
    assert sr.GATES == ROOT / "records" / "livesearch" / "mock_check.json"


def test_under_every_bar_passes(tmp_path, monkeypatch):
    ok, msg, out = _run(tmp_path, monkeypatch, flop=2.9, turn=1.9, river=1.9)
    assert ok and out["passed"] and out["failed"] == []
    assert (out["turn"]["n"], out["river"]["n"], out["flop"]["n"]) == (40, 40, 40)
    assert out["turn"]["bar"] == out["river"]["bar"] == 2.0 and out["turn_river"]["n"] == 80
    assert out["turn_schedule"] == dict(gamma=5, shift=1) and out["turn_iters"] == 400
    assert out["river_iters"] == 2000
    assert "turn p95 1.900 s" in msg and "river p95 1.900 s" in msg


@pytest.mark.parametrize("street", [2, 3])
def test_each_street_is_gated_on_its_own(tmp_path, monkeypatch, street):
    # The other street's fast decisions do not pull the slow one under its bar: pooled, the p95 passes.
    ok, msg, out = _run(tmp_path, monkeypatch, slow=street)
    name, other = ("turn", "river") if street == 2 else ("river", "turn")
    assert not ok and out["failed"] == [f"{name} decisions' p95 2.500 s over 2.0 s"]
    assert out[other]["p95"] == 1.0 and out["turn_river"]["p95"] < 2.0


def test_no_searched_turn_or_river_decision_fails(tmp_path, monkeypatch):
    ok, msg, out = _run(tmp_path, monkeypatch, streets=(1,))
    assert not ok and "no searched turn decision" in out["failed"] and "no searched river decision" in out["failed"]


def test_a_p95_at_the_bar_passes(tmp_path, monkeypatch):
    ok, msg, out = _run(tmp_path, monkeypatch, flop=3.0, turn=2.0, river=2.0)
    assert ok and out["failed"] == []


def test_the_flop_bar_is_unchanged(tmp_path, monkeypatch):
    ok, msg, out = _run(tmp_path, monkeypatch, flop=3.1)
    assert not ok and out["failed"] == ["flop decisions' p95 3.100 s over 3.0 s"]


def test_the_live_search_record_carries_a_schedule_other_than_the_default(tmp_path):
    rating = _load("rating_gates", ROOT / "scripts" / "rating.py")
    net = tmp_path / "net.pt"
    net.write_bytes(b"net")
    base = dict(NET=net, SEARCH=dict(sr.SEARCH), LEAVES="sym", TERMINALS64=True, FLOP_GAMMA=5.0)
    before = rating.live_search_record(SimpleNamespace(**base), fresh=True)
    assert "turn_schedule" not in before  # a runner before B1 (no TURN_SCHEDULE)
    for default in (None, {}, dict(gamma=2)):
        assert rating.live_search_record(SimpleNamespace(**base, TURN_SCHEDULE=default), fresh=True) == before
    rec = rating.live_search_record(SimpleNamespace(**base, TURN_SCHEDULE=dict(gamma=5, shift=1)), fresh=True)
    assert rec == dict(before, turn_schedule=dict(gamma=5, shift=1))
