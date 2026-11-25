"""scripts/search_lbr.py: a chunk that read a failed search is never saved, and
a run resumes only on its engine build (a stand-in engine: the real one is
tested in tests/test_search_ab.py)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("parciver_engine")
ROOT = Path(__file__).resolve().parents[1]
DEALS = 500  # two of the script's chunks; 50 blocks


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("search_lbr_cli", ROOT / "scripts" / "search_lbr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_failed_search_stops_the_chunk_unsaved_and_a_run_keeps_its_engine(cli, tmp_path, monkeypatch):
    bench, receipt, out = tmp_path / "benchmark.json", tmp_path / "receipt.json", tmp_path / "out.json"
    build = tmp_path / "engine.pyd"
    build.write_bytes(b"build 1")
    bench.write_text(json.dumps({"stakes": {"sb": 50, "bb": 100, "stack": 10_000}, "lbr": {"deals": DEALS, "seed": 11},
                                 "hands_per_deal": 2, "benchmark_id": "b", "manifest_hash": "h",
                                 "abstraction": {"dir": str(tmp_path)}}), encoding="utf-8")
    receipt.write_text(json.dumps({"manifest_hash": "h", "raw": {"lbr": {"per_deal_bb": [0.0] * DEALS}}}), encoding="utf-8")
    argv = ["search_lbr.py", "--benchmark", str(bench), "--candidate", f"v=bp.bin={receipt}", "--modes", "river",
            "--out", str(out)]

    played = []  # (deals, first deal) of every lbr call

    def run(late=None, **counts):
        """The script against an engine returning `counts` (from deal 250 on `late`'s instead, if given)."""
        played.clear()

        def lbr(deals, sb, bb, stack, first_deal=0, **kw):
            played.append((deals, first_deal))
            c = late if late is not None and first_deal >= 250 else counts
            return {"per_deal_bb": [1.0] * deals, "per_deal_preflop_bb": [0.0] * deals, **c}
        agent = types.SimpleNamespace(lbr=lbr)
        agent.with_search = lambda **kw: agent
        bp = types.SimpleNamespace(play_agent=lambda: agent)
        monkeypatch.setattr(cli, "pe", types.SimpleNamespace(
            SavedBlueprint=lambda abstraction, path: bp, PlayAgent=types.SimpleNamespace(uniform=lambda: agent),
            parciver_engine=types.SimpleNamespace(__file__=str(build))))
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()

    def deals_saved():
        return len(json.loads(out.read_text(encoding="utf-8"))["runs"]["v/river"]["per_deal_bb"])

    ok = {"search_failed": [0, 0], "model_search_failed": [0, 0]}
    for counts, match in ((dict(ok, model_search_failed=[0, 3]), r"reached a failed search \(decisions \[0, 0\], "
                                                                   r"LBR's model \[0, 3\]"),
                          (dict(ok, model_search_failed=[1, 0]), r"LBR's model \[1, 0\]"),
                          (dict(ok, search_failed=[1, 0]), r"reached a failed search \(decisions \[1, 0\]"),
                          (dict(ok, search_failed=[0, 2]), r"reached a failed search \(decisions \[0, 2\]"),
                          ({"search_failed": [0, 0]}, "predates search-failure counts"),
                          ({}, "predates search-failure counts")):
        with pytest.raises(SystemExit, match=match):
            run(**counts)
        assert not out.exists()
        if "predates" in match:  # found before playing a chunk
            assert played == [(2, 0)]
    # The second chunk fails: the first is kept.
    with pytest.raises(SystemExit, match=r"deals 250\.\.499 reached a failed search"):
        run(late=dict(ok, model_search_failed=[0, 1]), **ok)
    assert deals_saved() == 250
    with pytest.raises(SystemExit, match="predates search-failure counts"):  # each chunk is checked too
        run(late={}, **ok)
    assert played == [(2, 0), (250, 250)] and deals_saved() == 250
    # Not resumed on another build, nor if it predates the record of its build.
    build.write_bytes(b"build 2")
    with pytest.raises(SystemExit, match="another engine build"):
        run(**ok)
    build.write_bytes(b"build 1")
    state = json.loads(out.read_text(encoding="utf-8"))
    old = dict(state, runs={"v/river": {k: v for k, v in state["runs"]["v/river"].items() if k != "engine"}})
    out.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="another engine build"):
        run(**ok)
    out.write_text(json.dumps(state), encoding="utf-8")
    assert deals_saved() == 250
    run(**ok)
    assert json.loads(out.read_text(encoding="utf-8"))["runs"]["v/river"]["per_deal_bb"] == [1.0] * DEALS


def test_two_candidates_are_compared_only_on_one_engine_build(cli, tmp_path, monkeypatch, capsys):
    bench, receipt, out = tmp_path / "benchmark.json", tmp_path / "receipt.json", tmp_path / "out.json"
    build = tmp_path / "engine.pyd"
    bench.write_text(json.dumps({"stakes": {"sb": 50, "bb": 100, "stack": 10_000}, "lbr": {"deals": DEALS, "seed": 11},
                                 "hands_per_deal": 2, "benchmark_id": "b", "manifest_hash": "h",
                                 "abstraction": {"dir": str(tmp_path)}}), encoding="utf-8")
    receipt.write_text(json.dumps({"manifest_hash": "h", "raw": {"lbr": {"per_deal_bb": [0.0] * DEALS}}}), encoding="utf-8")
    argv = ["search_lbr.py", "--benchmark", str(bench), "--candidate", f"a=a.bin={receipt}",
            "--candidate", f"b=b.bin={receipt}", "--modes", "river", "--out", str(out)]

    def run(tag):
        def lbr(deals, sb, bb, stack, first_deal=0, **kw):
            return {"per_deal_bb": [float(first_deal % 7)] * deals, "search_failed": [0, 0],
                    "model_search_failed": [0, 0]}
        agent = types.SimpleNamespace(lbr=lbr)
        agent.with_search = lambda **kw: agent
        bp = types.SimpleNamespace(play_agent=lambda: agent)
        build.write_bytes(tag)
        monkeypatch.setattr(cli, "pe", types.SimpleNamespace(
            SavedBlueprint=lambda abstraction, path: bp, PlayAgent=types.SimpleNamespace(uniform=lambda: agent),
            parciver_engine=types.SimpleNamespace(__file__=str(build))))
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        return json.loads(out.read_text(encoding="utf-8"))

    state = run(b"build 1")
    assert {"b-a/none", "b-a/river"} <= state["summary"].keys()
    # b is replayed on a new build; a, finished, is not: their deals no longer pair.
    state["runs"].pop("b/river")
    out.write_text(json.dumps(state), encoding="utf-8")
    capsys.readouterr()
    state = run(b"build 2")
    assert state["runs"]["a/river"]["engine"] != state["runs"]["b/river"]["engine"]
    assert "b-a/none" in state["summary"] and not any(k.startswith("b-a/river") for k in state["summary"])
    assert "b - a, search=river: not compared, the two runs were played on different engine builds" in capsys.readouterr().out
