"""scripts/search_ab.py: one blueprint under two search settings, head to head
(FLOPSEARCH M2, LIVESEARCH P2). CPU arms and a stand-in GPU solver here."""
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

pe = pytest.importorskip("parciver_engine")
ROOT = Path(__file__).resolve().parents[1]
STAKES = {"sb": 50, "bb": 100, "stack": 10_000}


@pytest.fixture(scope="module")
def ab():
    spec = importlib.util.spec_from_file_location("search_ab_cli", ROOT / "scripts" / "search_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_arm_is_the_engines_settings_with_its_defaults(ab):
    sig = inspect.signature(pe.PlayAgent.with_search).parameters
    assert {k: sig[k].default for k in ab.SEARCH} == ab.SEARCH
    assert set(sig) - set(ab.SEARCH) == {"self", "solver", "rescue"}
    assert ab.arm("{}") == ab.ARM
    assert ab.arm('{"from_turn": true, "turn_iters": 100}') == ab.arm('{"from_turn": true}') == dict(ab.ARM, from_turn=True)
    assert ab.arm('{"solver": "flop"}')["net"] is None  # check-down leaves
    for good in ('{"from_flop": true, "solver": "flop", "net": "n.pt"}', '{"from_flop": true, "flop_checkdown": true}',
                 '{"from_flop": true, "flop_checkdown": true, "solver": "gpu"}', '{"translation": "standin"}'):
        ab.arm(good)
    for bad in ('[1]', '{"from_turn": tru}', '{"turn_iter": 100}', '{"solver": "tpu"}', '{"net": "n.pt"}',
                '{"net": "n.pt", "solver": "gpu"}', '{"solver_kw": {"cache": 8}}', '{"solver": "gpu", "solver_kw": 1}',
                # JSON true is not 1 iteration, nor 1 a flag; no zero iterations
                '{"river_iters": true}', '{"from_turn": 1}', '{"turn_iters": 0}', '{"flop_iters": 2.5}',
                # a flop search needs the flop solver or check-down leaves (SubgameGpu solves no flop)
                '{"from_flop": true}', '{"from_flop": true, "solver": "gpu"}',
                # a net no flop search uses
                '{"solver": "flop", "net": "n.pt"}', '{"from_flop": true, "flop_checkdown": true, "solver": "flop", "net": "n.pt"}',
                '{"translation": 3}'):
        with pytest.raises(SystemExit):
            ab.arm(bad)


def _arms():  # (uniform play seldom reaches the river: both search from the turn)
    return (pe.PlayAgent.uniform().with_search(from_turn=True, turn_iters=3, river_iters=10),
            pe.PlayAgent.uniform().with_search(from_turn=True, turn_iters=2, river_iters=10))


def test_chunks_resumed_equal_one_match(ab):
    a, b = _arms()
    whole = dict(a.match_against("agent", 24, 50, 100, 10_000, other=b, seed=3))
    assert whole["a_sources"][3] > 0 and whole["b_sources"][3] > 0  # both arms searched
    run, saves = ab.new_run(), []

    def save():
        saves.append(len(run["per_deal_bb"]))

    ab.play(run, save, _arms(), {}, 12, 6, STAKES, 3, 0, 2)
    ab.play(run, save, _arms(), {}, 24, 12, STAKES, 3, 0, 2)  # resumed with fresh agents and another chunk size
    assert saves == [6, 12, 24]
    assert run["per_deal_bb"] == whole["per_deal_bb"]
    assert run["a_sources"] == whole["a_sources"] and run["b_sources"] == whole["b_sources"]


def test_a_gpu_arm_has_no_rescue_and_its_failure_stops_the_chunk_unsaved(ab, monkeypatch):
    class Broken:  # stands in for SubgameGpu: no GPU here
        def __init__(self, **kw):
            self.kw = kw

        def __call__(self, job):
            raise RuntimeError("no GPU")

    monkeypatch.setitem(sys.modules, "parciver.gpu_fused", types.SimpleNamespace(SubgameGpu=Broken))
    bp = types.SimpleNamespace(play_agent=pe.PlayAgent.uniform)

    def arms():  # fresh agents: an agent keeps its failed solves (tried once per public state)
        a, wa = ab.build(ab.arm('{"from_turn": true, "turn_iters": 2, "solver": "gpu", "solver_kw": {"cache": 8}}'), bp)
        b, wb = ab.build(ab.arm('{"river_iters": 10}'), bp)
        assert wb is None and wa.solver.kw == {"cache": 8}
        return a, b, wa

    a, b, _ = arms()
    res = dict(a.match_against("agent", 24, 50, 100, 10_000, other=b, seed=3))
    assert res["a_search_failed"][0] == 0 and res["a_search_failed"][1] > 0  # the blueprint, not the CPU, decided
    a, b, wa = arms()
    run, saves = ab.new_run(), []
    with pytest.raises(SystemExit, match="search solves failed"):
        ab.play(run, lambda: saves.append(1), (a, b), {"a": wa}, 24, 24, STAKES, 3, 0, 2)
    assert run == ab.new_run() and saves == []
    # The engine's counts stop the chunk too: a failing solver no Watched sees ...
    a, b, _ = arms()
    with pytest.raises(SystemExit, match="reached a failed search"):
        ab.play(run, lambda: saves.append(1), (a, b), {}, 24, 24, STAKES, 3, 0, 2)
    # ... and a Watched no solve reaches (not the agent's solver) stops it.
    a, b, _ = arms()
    a = pe.PlayAgent.uniform().with_search(from_turn=True, turn_iters=2, river_iters=10)
    idle = ab.Watched(Broken())
    with pytest.raises(SystemExit, match="no search solve reached"):
        ab.play(run, lambda: saves.append(1), (a, b), {"a": idle}, 24, 24, STAKES, 3, 0, 2)
    assert run == ab.new_run() and saves == []


def test_chunks_resumed_equal_one_match_against_a_bot_and_one_lbr(ab):
    # (Uniform arms act alike in both seats, scoring 0 a deal, until one searches: seed 3's deal 15.)
    bot = pe.PlayAgent.uniform("200bb")  # a bot unlike either arm (another grid: other actions)
    a, b = _arms()
    whole = {side: dict(arm.match_against("perturbed", 24, 50, 100, 10_000, other=bot, perturb=1.5, seed=3))
             for side, arm in (("a", a), ("b", b))}
    assert whole["a"]["per_deal_bb"] != whole["b"]["per_deal_bb"] and whole["a"]["a_sources"][3] > 0
    run, saves = ab.new_run("bot"), []
    for deals, chunk in ((12, 6), (24, 12)):  # resumed with fresh agents and another chunk size
        ab.play(run, lambda: saves.append(len(run["a"]["per_deal_bb"])), _arms(), {}, deals, chunk, STAKES, 3, 0, 2,
                mode="bot", bot=("perturbed", bot, {"perturb": 1.5}))
    assert saves == [6, 12, 24]
    for side in "ab":
        assert run[side] == {"per_deal_bb": whole[side]["per_deal_bb"], "sources": whole[side]["a_sources"]}
    whole = {side: dict(arm.lbr(12, 50, 100, 10_000, seed=11)) for side, arm in (("a", a), ("b", b))}
    assert whole["a"]["per_deal_bb"] != whole["b"]["per_deal_bb"]
    run, saves = ab.new_run("lbr"), []
    for deals, chunk in ((6, 2), (12, 6)):
        ab.play(run, lambda: saves.append(len(run["a"]["per_deal_bb"])), _arms(), {}, deals, chunk, STAKES, 11, 0, 2,
                mode="lbr")
    assert saves == [2, 4, 6, 12]
    for side in "ab":
        assert run[side] == {k: whole[side][k] for k in ("per_deal_bb", "per_deal_preflop_bb")}


def test_a_failed_search_against_a_bot_or_lbr_stops_the_chunk_unsaved(ab):
    class Broken:
        def __call__(self, job):
            raise RuntimeError("no GPU")

    broken = pe.PlayAgent.uniform().with_search(from_turn=True, turn_iters=2, river_iters=10, solver=Broken(), rescue=False)
    for arms, match in (((broken, _arms()[1]), r"\(A \[0, [1-9]"), ((_arms()[0], broken), r"\(A \[0, 0\], B \[0, [1-9]")):
        for mode, deals, seed, bot in (("bot", 24, 3, ("realistic", pe.PlayAgent.uniform(), {})), ("lbr", 12, 11, None)):
            run = ab.new_run(mode)
            with pytest.raises(SystemExit, match=r"reached a failed search " + match):
                ab.play(run, lambda: None, arms, {}, deals, deals, STAKES, seed, 0, 2, mode=mode, bot=bot)
            assert run == ab.new_run(mode)

    # LBR's model of an arm reading a failed search (e.g. its answer to a bet LBR weighs but does not
    # make) stops it too; an engine that does not count the model's is refused.
    def counted(**counts):
        def lbr(deals, sb, bb, stack, **kw):
            return {"per_deal_bb": [0.0] * deals, "per_deal_preflop_bb": [0.0] * deals, **counts}
        return types.SimpleNamespace(lbr=lbr)

    ok = {"search_failed": [0, 0], "model_search_failed": [0, 0]}
    for arms, match in (((counted(**ok), counted(**dict(ok, model_search_failed=[2, 0]))),
                         r"reached a failed search \(A \[0, 0\], B \[0, 0\], LBR's model of A \[0, 0\], of B \[2, 0\]"),
                        ((counted(**dict(ok, model_search_failed=[0, 1])), counted(**ok)),
                         r"reached a failed search \(A \[0, 0\], B \[0, 0\], LBR's model of A \[0, 1\], of B \[0, 0\]"),
                        ((counted(**ok), counted(search_failed=[0, 0])), "predates search-failure counts")):
        run = ab.new_run("lbr")
        with pytest.raises(SystemExit, match=match):
            ab.play(run, lambda: None, arms, {}, 4, 4, STAKES, 11, 0, 2, mode="lbr")
        assert run == ab.new_run("lbr")
    ab.play(run, lambda: None, (counted(**ok), counted(**ok)), {}, 4, 4, STAKES, 11, 0, 2, mode="lbr")
    assert run["a"]["per_deal_bb"] == [0.0] * 4
    # The message names the chunk that stopped (here the second); the first stays saved.
    with pytest.raises(SystemExit, match=r"^deals 4\.\.7 reached a failed search \(A \[0, 0\], B \[0, 0\], "
                                         r"LBR's model of A \[0, 0\], of B \[0, 1\]"):
        ab.play(run, lambda: None, (counted(**ok), counted(**dict(ok, model_search_failed=[0, 1]))), {}, 8, 4, STAKES,
                11, 0, 2, mode="lbr")
    assert run["a"]["per_deal_bb"] == run["b"]["per_deal_bb"] == [0.0] * 4


def test_the_lbr_summary_pairs_each_arms_own_postflop_part(ab):
    # (The CLI test's arms search from the turn alike: their preflop parts are equal there.)
    run = {"a": {"per_deal_bb": [3.0, 1.0, 2.0, 0.0], "per_deal_preflop_bb": [1.0, 0.0, 0.0, 0.0]},
           "b": {"per_deal_bb": [1.0, 1.0, 0.0, 4.0], "per_deal_preflop_bb": [0.0, 1.0, 0.0, 2.0]}}
    got = ab.pair_summary(run, 4, 2, 2, "lbr")  # blocks of 2 deals, 2 hands a deal: 50 bb/100 a bb
    assert [got[s]["mean"] for s in "ab"] == [75.0, 75.0] and got["a_minus_b"]["mean"] == 0.0
    assert [got["postflop"][s]["mean"] for s in "ab"] == [62.5, 37.5]  # blocks [75, 50] and [25, 50]
    assert got["postflop_a_minus_b"] == ab.sc.summarize([50.0, 0.0])
    assert "postflop" not in ab.pair_summary(run, 4, 2, 2, "bot")


def _bench(tmp_path, ab, blocks=50):
    """A small benchmark manifest (hashed as the scorecard makes them), its abstraction files, its
    baseline blueprint file and a blueprint file."""
    absd = tmp_path / "abs"
    absd.mkdir()
    (absd / "flop_buckets.npy").write_bytes(b"flop")
    (tmp_path / "base.bin").write_bytes(b"baseline")
    m = {"benchmark_id": "t", "stakes": STAKES, "hands_per_deal": 2, "blocks": blocks,
         "baseline": {"path": str(tmp_path / "base.bin"), "sha256": ab.sc.sha256_file(tmp_path / "base.bin")},
         "abstraction": {"dir": str(absd), "sha256": {"flop_buckets.npy": ab.sc.sha256_file(absd / "flop_buckets.npy")}},
         "rows": [{"name": "baseline", "kind": "agent", "family": None, "deals": 100, "seed": 3, "perturb": None},
                  {"name": "nit(0.20)", "kind": "leak", "family": "nit", "deals": 100, "seed": 4, "perturb": None},
                  {"name": "baseline_sizes_x1.5", "kind": "perturbed", "family": None, "deals": 100, "seed": 9,
                   "perturb": 1.5},
                  {"name": "baseline_realistic_sizes", "kind": "realistic", "family": None, "deals": 100, "seed": 7,
                   "perturb": None}],
         "lbr": {"deals": 100, "seed": 11}}
    m["manifest_hash"] = ab.sc.canonical_hash(m)
    bench = tmp_path / "benchmark.json"
    bench.write_text(json.dumps(m))
    (tmp_path / "bp.bin").write_bytes(b"blueprint")
    return bench, m


def test_main_saves_resumes_and_refuses_another_configuration(ab, tmp_path, monkeypatch):
    bench, m = _bench(tmp_path, ab)
    monkeypatch.setattr(ab.pe, "SavedBlueprint", lambda d, p: types.SimpleNamespace(play_agent=pe.PlayAgent.uniform))
    out = tmp_path / "ab.json"
    argv = ["search_ab.py", "--benchmark", str(bench), "--blueprint", str(tmp_path / "bp.bin"),
            "--a", '{"from_turn": true, "turn_iters": 3, "river_iters": 10}',
            "--b", '{"from_turn": true, "turn_iters": 2, "river_iters": 10}', "--deals", "50", "--chunk", "25",
            "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    real_write = ab.write_atomic

    def write_then_stop(path, text):  # a run stopped right after its first chunk is saved
        real_write(path, text)
        raise KeyboardInterrupt

    monkeypatch.setattr(ab, "write_atomic", write_then_stop)
    with pytest.raises(KeyboardInterrupt):
        ab.main()
    monkeypatch.setattr(ab, "write_atomic", real_write)
    assert len(json.loads(out.read_text(encoding="utf-8"))["run"]["per_deal_bb"]) == 25
    monkeypatch.setattr(sys, "argv", argv[:-3] + ["24", "--out", str(out)])
    with pytest.raises(SystemExit, match=r"no last chunk of 1 deal \(of the 25 left\)"):  # the engine plays 2 or more
        ab.main()
    monkeypatch.setattr(sys, "argv", argv[:-3] + ["40", "--out", str(out)])
    ab.main()  # resumed, with a chunk larger than what is left
    got = json.loads(out.read_text(encoding="utf-8"))
    a, b = _arms()
    whole = dict(a.match_against("agent", 50, 50, 100, 10_000, other=b, seed=3))
    assert got["run"]["per_deal_bb"] == whole["per_deal_bb"]
    assert [(c["first"], c["deals"]) for c in got["run"]["chunks"]] == [(0, 25), (25, 25)]
    assert got["summary"]["a_vs_b"] == ab.sc.summarize(ab.sc.to_blocks(whole["per_deal_bb"], 50, 2))
    src = got["config"]["sources"]
    assert "scripts/search_ab.py" in src and "src/parciver/solver_watch.py" in src
    assert "src/parciver/scorecard.py" not in src and got["config"]["gpu"] is None
    ab.main()  # finished: the summary again, nothing played
    assert json.loads(out.read_text(encoding="utf-8")) == got

    def refused(match, argv=argv):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises((SystemExit, ab.sc.ScorecardError), match=match):
            ab.main()
        assert json.loads(out.read_text(encoding="utf-8")) == got

    refused(r"\(a differ\)", argv[:6] + ['{"from_turn": true, "turn_iters": 4, "river_iters": 10}'] + argv[7:])
    refused("the arms are the same", argv[:8] + [argv[6]] + argv[9:])
    refused("positive multiple of 50", argv[:-5] + ["60"] + argv[-4:])
    monkeypatch.setattr(ab, "engine_digest", lambda: "another build")
    refused(r"\(engine differ\)")
    monkeypatch.undo()
    monkeypatch.setattr(ab, "sources", lambda arms: {"scripts/search_ab.py": "edited"})
    refused(r"\(sources differ\)")
    monkeypatch.undo()
    monkeypatch.setattr(ab, "gpu_env", lambda arms: {"torch": "upgraded"})
    refused(r"\(gpu differ\)")
    monkeypatch.undo()
    monkeypatch.setattr(ab.pe, "SavedBlueprint", lambda d, p: types.SimpleNamespace(play_agent=pe.PlayAgent.uniform))
    (tmp_path / "bp.bin").write_bytes(b"retrained in place")
    refused(r"\(blueprint differ\)")
    (tmp_path / "bp.bin").write_bytes(b"blueprint")
    (tmp_path / "abs" / "flop_buckets.npy").write_bytes(b"rebuilt")
    refused("abstraction files")
    (tmp_path / "abs" / "flop_buckets.npy").write_bytes(b"flop")
    bench.write_text(json.dumps(dict(m, rows=[dict(m["rows"][0], seed=4)])))  # edited, hash kept
    refused("manifest was edited")


def test_main_plays_each_arm_against_the_rows_bot_or_lbr_paired(ab, tmp_path, monkeypatch):
    bench, m = _bench(tmp_path, ab, blocks=2)
    bot = pe.PlayAgent.uniform("200bb")
    loaded = []

    def saved(d, p):
        loaded.append(Path(p).name)
        return types.SimpleNamespace(play_agent=(lambda: bot) if Path(p).name == "base.bin" else pe.PlayAgent.uniform)

    monkeypatch.setattr(ab.pe, "SavedBlueprint", saved)
    argv = ["search_ab.py", "--benchmark", str(bench), "--blueprint", str(tmp_path / "bp.bin"),
            "--a", '{"from_turn": true, "turn_iters": 3, "river_iters": 10}',
            "--b", '{"from_turn": true, "turn_iters": 2, "river_iters": 10}', "--deals", "24", "--chunk", "12"]
    a, b = _arms()

    def main(*extra, out):
        monkeypatch.setattr(sys, "argv", argv + list(extra) + ["--out", str(tmp_path / out)])
        ab.main()
        return json.loads((tmp_path / out).read_text(encoding="utf-8"))

    def paired(got, blocks_a, blocks_b, key="a_minus_b"):
        assert got["summary"][key] == ab.sc.paired_change(blocks_a, blocks_b)

    for row, kind, seed, kw in (("baseline_sizes_x1.5", "perturbed", 9,{"perturb": 1.5}),
                                ("baseline_realistic_sizes", "realistic", 7, {})):
        loaded.clear()
        got = main("--row", row, out=f"{kind}.json")
        assert loaded == ["bp.bin", "base.bin"] and got["config"]["measure"] == "bot"
        whole = {side: dict(arm.match_against(kind, 24, 50, 100, 10_000, other=bot, seed=seed, **kw))
                 for side, arm in (("a", a), ("b", b))}
        assert whole["a"]["per_deal_bb"] != whole["b"]["per_deal_bb"]
        blk = {}
        for side in "ab":
            assert got["run"][side]["per_deal_bb"] == whole[side]["per_deal_bb"]
            blk[side] = ab.sc.to_blocks(whole[side]["per_deal_bb"], 2, 2)
            assert got["summary"][side] == ab.sc.summarize(blk[side])
        paired(got, blk["a"], blk["b"])
    got = main("--lbr", out="lbr.json")
    assert got["config"]["measure"] == "lbr" and got["config"]["row"] is None
    whole = {side: dict(arm.lbr(24, 50, 100, 10_000, seed=11)) for side, arm in (("a", a), ("b", b))}
    post = {}
    for side in "ab":
        assert got["run"][side]["per_deal_preflop_bb"] == whole[side]["per_deal_preflop_bb"]
        post[side] = ab.sc.lbr_parts(whole[side]["per_deal_bb"], whole[side]["per_deal_preflop_bb"], 2, 2)["postflop"]
    paired(got, *(ab.sc.to_blocks(whole[side]["per_deal_bb"], 2, 2) for side in "ab"))
    paired(got, post["a"], post["b"], "postflop_a_minus_b")
    # The arms playing the baseline blueprint itself: it is loaded once.
    loaded.clear()
    monkeypatch.setattr(sys, "argv", argv[:4] + [str(tmp_path / "base.bin")] + argv[5:] +
                        ["--row", "baseline_realistic_sizes", "--out", str(tmp_path / "self.json")])
    ab.main()
    assert loaded == ["base.bin"]

    def refused(match, *extra, out="lbr.json"):
        monkeypatch.setattr(sys, "argv", argv + list(extra) + ["--out", str(tmp_path / out)])
        with pytest.raises(SystemExit, match=match):
            ab.main()

    refused(r"\(measure, row differ\)", "--row", "baseline_realistic_sizes")
    refused("not a row's", "--lbr", "--row", "baseline")
    refused("neither a strength row", "--row", "nit(0.20)")
    refused("at most 100", "--lbr", "--deals", "102")
    (tmp_path / "base.bin").write_bytes(b"retrained in place")
    refused("baseline blueprint .* changed", "--row", "baseline_sizes_x1.5", out="perturbed.json")
    main("--lbr", out="lbr.json")  # LBR and the strength rows play no baseline
    state = json.loads((tmp_path / "lbr.json").read_text(encoding="utf-8"))
    state["run"]["b"]["per_deal_preflop_bb"].pop()
    (tmp_path / "lbr.json").write_text(json.dumps(state), encoding="utf-8")
    refused("differ in length", "--lbr")


def test_the_identity_keys_a_net_by_its_contents(ab, tmp_path, monkeypatch):
    bench, m = _bench(tmp_path, ab)
    monkeypatch.setattr(ab, "sources", lambda arms: {})  # (imports no torch; the CLI test checks sources)
    monkeypatch.setattr(ab, "gpu_env", lambda arms: None)
    net = tmp_path / "net.pt"
    net.write_bytes(b"trained")
    args = types.SimpleNamespace(row="baseline", blueprint=str(tmp_path / "bp.bin"), b=ab.arm('{"solver": "gpu"}'),
                                 a=ab.arm(json.dumps({"from_flop": True, "solver": "flop", "net": str(net)})))
    before = ab.identity(m, args, 50)
    net.write_bytes(b"retrained in place")
    after = ab.identity(m, args, 50)
    assert set(before["nets"]) == {"a"} and [k for k in before if before[k] != after[k]] == ["nets"]


def test_the_gpu_environment_names_torch_cuda_the_gpu_triton_and_numpy(ab, monkeypatch):
    import numpy as np

    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "a GPU")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="9.9", version=types.SimpleNamespace(cuda="13.2"),
                                                                    cuda=cuda))  # stand-ins: no GPU is touched
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(__version__="3.8"))
    assert ab.gpu_env([ab.arm("{}"), ab.arm('{"from_turn": true}')]) is None
    assert ab.gpu_env([ab.arm("{}"), ab.arm('{"solver": "flop"}')]) == {
        "torch": "9.9", "cuda": "13.2", "gpu": "a GPU", "triton": "3.8", "numpy": np.__version__}


def test_sources_refuses_a_parciver_loaded_from_elsewhere(ab, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "parciver", types.SimpleNamespace(__file__=str(tmp_path / "parciver" / "__init__.py")))
    with pytest.raises(SystemExit, match="its sources would key nothing"):
        ab.sources([ab.arm("{}")])
