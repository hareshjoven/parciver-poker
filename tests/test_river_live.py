"""RIVER R1 (records/river-live/PREREG.md): scripts/river_live.py's harness. R1 differs from R0 only in
river_iters; a chunk's proof fails a river job at the wrong iterations; the tap keeps R0's river jobs in P1's
chunks only; the resume refusal; P1's statistic on toy data."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("river_live_cli", ROOT / "scripts" / "river_live.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)
ls, ri = rl.ls, rl.ri

LIVE = dict(from_flop=True, flop_iters=500, from_turn=True, turn_iters=400, river_iters=200, rescue=False)
SCHED = dict(gamma=5, shift=1)
T, R = "4:gamma5:plus0:shift1:restart0", "5:gamma2:plus0:shift0:restart0"


def _counts(river_iters, **over):
    c = dict(calls=10, failures=0, last_error=None, seen={"3:500": 2, "4:400": 5, f"5:{river_iters}": 3}, flop=2,
             turn=5, river=3, ran={T: 5, R: 3}, symmetric=0)
    c.update(over)
    return c


def test_r1_differs_from_r0_only_in_river_iters():
    r0, r1 = rl.search_kw(LIVE, rl.R0_RIVER), rl.search_kw(LIVE, rl.R1_RIVER)
    assert r0 == LIVE  # R0 is the live runner's search
    assert {k for k in r1 if r1[k] != r0[k]} == {"river_iters"} and r1["river_iters"] == 2000
    assert LIVE["river_iters"] == 200  # the input is not changed


def test_the_live_runner_is_r1():  # R0 until the adoption (PREREG "Adoption" 1); river_live.py p2 now refuses
    sr = ls.runner()
    assert rl.search_kw(sr.SEARCH, rl.R1_RIVER) == dict(sr.SEARCH)
    assert dict(sr.TURN_SCHEDULE) == SCHED


def test_a_chunk_proves_the_river_iterations():
    it0, it1 = {3: 500, 4: 400, 5: 200}, {3: 500, 4: 400, 5: 2000}
    assert ls.run_key(4, ls.resolved(SCHED)) == T and ls.run_key(5, ls.resolved(None)) == R
    assert ls.prove("R0", _counts(200), it0, SCHED) == []
    assert ls.prove("R1", _counts(2000), it1, SCHED) == []
    assert ls.prove("R1", _counts(200), it1, SCHED)  # R0's river on R1
    assert ls.prove("R0", _counts(2000), it0, SCHED)  # R1's river on R0
    assert ls.prove("R1", _counts(2000, seen={"3:500": 2, "4:400": 5, "5:2000": 2, "5:200": 1}), it1, SCHED)
    assert ls.prove("R1", _counts(2000, ran={T: 5, R: 2, "5:gamma5:plus0:shift1:restart0": 1}), it1, SCHED)


def test_the_tap_keeps_river_jobs_in_p1s_chunks_only():
    assert [f for f in range(0, rl.DEALS, rl.CHUNK) if rl.keeps(f)] == [500, 600, 700]
    tap = ri.RiverTap(lambda job: "ok")
    tap.keep = True
    for board in ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]):
        assert tap(dict(board=board, iters=7)) == "ok"
    seen, jobs = tap.take()
    assert seen == {"3:7": 1, "4:7": 1, "5:7": 1} and [j["board"] for j in jobs] == [[1, 2, 3, 4, 5]]
    tap.keep = False
    tap(dict(board=[1, 2, 3, 4, 5], iters=7))
    assert tap.take()[1] == []


def test_the_code_covers_the_scripts():
    assert set(ls.PLAY_CODE) < set(rl.CODE)
    assert {"scripts/river_iters.py", "scripts/river_live.py"} < set(rl.CODE)


def test_resume_refuses_another_configuration(tmp_path):
    path = tmp_path / "state.json"
    cfg = dict(seed=410001, r1=dict(search=rl.search_kw(LIVE, 2000)))
    path.write_text(json.dumps(dict(config=cfg, run={})), encoding="utf-8")
    assert ls.resume(path, cfg)["config"] == cfg
    with pytest.raises(SystemExit, match="r1"):
        ls.resume(path, dict(cfg, r1=dict(search=rl.search_kw(LIVE, 1000))))


def test_p1_rows_and_statistic():
    jobs = [dict(board=[b, 1, 2, 3, 4], kind=[0, 1], stake=[100, 100, 50, 50], frozen=[], iters=200, ranges=[[1.0]])
            for b in (9, 9, 10, 11)]
    calls = []

    def solve(job, n):
        calls.append(n)
        return (2.0 if n == rl.R0_RIVER else 0.5) + 0.01 * job["board"][0]

    rows = rl.p1_rows(solve, jobs)
    assert calls == [200, 2000] * 4
    assert all(abs(r["d"] + 1.5) < 1e-12 and r["pot"] == 200.0 for r in rows)
    assert rows[0]["board"] == [1, 2, 9, 3, 4]  # the flop sorted, then turn and river
    st = rl.p1_stat(rows)
    assert st["holds"] and st["bootstrap"]["boards"] == 3 and st["bootstrap"]["jobs"] == 4 and st["r1_lower"] == 4
    worse = [dict(r, d=0.1) for r in rows]
    assert not rl.p1_stat(worse)["holds"]


def test_a_chunk_needs_a_river_solve_from_each_arm():
    assert rl.rivers(dict(b0=dict(river=3), b1=dict(river=1))) == []
    assert rl.rivers(dict(b0=dict(river=0), b1=dict(river=2))) == ["R0: no river solve"]
    assert rl.rivers(dict(b0=dict(river=1), b1=dict(river=0))) == ["R1: no river solve"]


def test_build_arms_swaps_the_tap_and_restores_it(monkeypatch):
    seen, old = [], ls.Tap

    class FakeArm:
        def __init__(self, name, sr, bp, river_iters, threads):
            seen.append((name, ls.Tap, river_iters, threads))
            if name == "R1" and fail:
                raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "parciver_engine", types.SimpleNamespace(SavedBlueprint=lambda a, b: "bp"))
    monkeypatch.setattr(rl, "RArm", FakeArm)
    fail = False
    with pytest.raises(SystemExit, match="R0 is 200"):  # R1 adopted: P2's arms are no longer built
        rl.build_arms(4)
    sr = ls.runner()
    sr.SEARCH = dict(sr.SEARCH, river_iters=rl.R0_RIVER)  # the live runner as P2 ran it
    monkeypatch.setattr(ls, "runner", lambda: sr)
    rl.build_arms(4)
    assert seen == [("R0", ri.RiverTap, 200, 4), ("R1", ri.RiverTap, 2000, 4)] and ls.Tap is old
    fail = True
    with pytest.raises(RuntimeError):
        rl.build_arms(4)
    assert ls.Tap is old


def test_p2_needs_a_smoke_run_on_this_code(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "RUNS", tmp_path)
    prov = dict(play_code={"src": "abc"})
    assert not rl.smoke_passed(prov, "E", 1)  # no smoke run: not even at 1 thread
    rec = dict(threads=[1, 4], play_code=prov["play_code"], engine="E", identical=False)
    (tmp_path / "smoke.json").write_text(json.dumps(rec), encoding="utf-8")
    assert rl.smoke_passed(prov, "E", 1) and not rl.smoke_passed(prov, "E", 4)
    (tmp_path / "smoke.json").write_text(json.dumps(dict(rec, identical=True)), encoding="utf-8")
    assert rl.smoke_passed(prov, "E", 4)
    assert not rl.smoke_passed(prov, "F", 4) and not rl.smoke_passed(dict(play_code={"src": "x"}), "E", 4)
