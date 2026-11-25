# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Scorecard: statistics, validation, identities, and a real-engine integration
run on the exact small game (no blueprint artifacts needed)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

from parciver import scorecard as sc

ROOT = Path(__file__).resolve().parents[1]


def test_summary_statistics_known_values():
    s = sc.summarize([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5
    assert s["se"] == pytest.approx(math.sqrt(5 / 3) / 2)  # sd (ddof=1) / sqrt(4)
    assert s["lo"] == pytest.approx(2.5 - 3.182446305 * s["se"], rel=1e-9)  # t, 3 degrees of freedom
    assert s["hi"] == pytest.approx(2.5 + 3.182446305 * s["se"], rel=1e-9)
    # The two-sided 95% t quantiles (tables): 50 blocks is 2.0096, not z's 1.96.
    for df, q in ((1, 12.706204736), (19, 2.093024054), (49, 2.009575237), (1000, 1.962339081)):
        assert sc.t95(df) == pytest.approx(q, abs=1e-8), df
    assert sc.summarize([1.0, 2.0] * 25)["hi"] == pytest.approx(1.5 + 2.009575237 * math.sqrt(0.25 * 50 / 49 / 50))


@pytest.mark.parametrize("lo, hi, better, expected", [
    (0.1, 2.0, True, "UP ON THIS BENCHMARK"),
    (-2.0, -0.1, True, "DOWN ON THIS BENCHMARK"),
    (-1.0, 1.0, True, "UNCLEAR / NO RESOLVED CHANGE"),
    (0.0, 0.0, True, "UNCLEAR / NO RESOLVED CHANGE"),
    (0.1, 2.0, False, "MORE EXPLOITABLE ON THIS BENCHMARK"),
    (-2.0, -0.1, False, "LESS EXPLOITABLE ON THIS BENCHMARK"),
])
def test_status_words(lo, hi, better, expected):
    assert sc.status(lo, hi, better) == expected


def test_blocks_and_measurement_validation():
    per_deal = [0.1 * i for i in range(100)]
    blocks = sc.to_blocks(per_deal, 10, 2)
    mean = sum(per_deal) / 100 * 100 / 2
    ok = {"per_deal_bb": per_deal, "bb_per_100": mean}
    assert sc.validate_measurement(ok, 100, 10, 2) == blocks
    with pytest.raises(sc.ScorecardError):
        sc.validate_measurement({**ok, "per_deal_bb": per_deal[:-1]}, 100, 10, 2)
    with pytest.raises(sc.ScorecardError):
        sc.validate_measurement({**ok, "per_deal_bb": per_deal[:-1] + [float("nan")]}, 100, 10, 2)
    with pytest.raises(sc.ScorecardError):
        sc.validate_measurement({**ok, "bb_per_100": mean + 1.0}, 100, 10, 2)
    with pytest.raises(sc.ScorecardError):
        sc.to_blocks(per_deal, 7, 2)


def _manifest(**kw):
    return sc.make_manifest("test", {"path": "b", "sha256": "B"}, {"dir": "a", "sha256": {}}, **kw)


def test_manifest_tamper_and_cross_benchmark_comparison_rejected():
    m = _manifest()
    sc.check_manifest(m)
    with pytest.raises(sc.ScorecardError):
        sc.check_manifest({**m, "blocks": 10})
    other = _manifest(blocks=25)
    fake = {"manifest_hash": m["manifest_hash"], "scores": {}}
    with pytest.raises(sc.ScorecardError):
        sc.compare(fake, {"state": "complete", "manifest_hash": other["manifest_hash"]})
    with pytest.raises(sc.ScorecardError):
        sc.compare(fake, {"state": "incomplete", "manifest_hash": m["manifest_hash"]})


def test_missing_row_gives_no_score():
    m = _manifest()
    with pytest.raises(sc.ScorecardError, match="missing"):
        sc.score_receipt(m, {}, {}, [1, 0, 0])


def _lbr_receipt(total, pre=None, blocks=10, label="r"):
    """A minimal complete receipt holding only an LBR result (and its split)."""
    lbr = {"per_deal_bb": total, **({} if pre is None else {"per_deal_preflop_bb": pre})}
    b = sc.to_blocks(total, blocks, 2)
    scores = {"strength": {"mean": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0},
              "exploitability_lbr": {"blocks": b, **{k: v for k, v in sc.summarize(b).items() if k != "blocks"}},
              "off_tree_share": None, "rows": {sc.HEADLINE: {"blocks": [0.0] * blocks, "mean": 0.0, "lo": 0.0,
                                                             "hi": 0.0, "se": 0.0}}}
    for p, pb in (sc.lbr_parts(total, pre, blocks, 2) or {}).items():
        scores[f"exploitability_lbr_{p}"] = {"blocks": pb, **{k: v for k, v in sc.summarize(pb).items()
                                                              if k != "blocks"}}
    return {"state": "complete", "manifest_hash": "M", "candidate": {"label": label}, "started_utc": "t",
            "benchmark_id": "test", "raw": {"lbr": lbr}, "scores": scores}


def test_lbr_split_parts_merge_sidecar_and_postflop_change():
    total = [((i * 7) % 11 - 5) * 0.5 for i in range(100)]
    pre = [x if i % 3 == 0 else 0.0 for i, x in enumerate(total)]
    parts = sc.lbr_parts(total, pre, 10, 2)
    whole = sc.to_blocks(total, 10, 2)
    assert [a + b for a, b in zip(parts["postflop"], parts["preflop"])] == pytest.approx(whole, abs=1e-12)
    assert sc.lbr_parts(total, None, 10, 2) is None
    with pytest.raises(sc.ScorecardError):
        sc.lbr_parts(total, pre[:-1], 10, 2)

    # Chunks keep the split (and the luck-corrected list) only when every chunk has it.
    chunks = [{"per_deal_bb": total[:50], "per_deal_preflop_bb": pre[:50]},
              {"per_deal_bb": total[50:], "per_deal_preflop_bb": pre[50:]}]
    assert sc.merge_chunks(chunks, 2)["per_deal_preflop_bb"] == pre
    assert "per_deal_preflop_bb" not in sc.merge_chunks([chunks[0], {"per_deal_bb": total[50:]}], 2)
    cv = [dict(c, per_deal_bb_cv=c["per_deal_bb"][::-1]) for c in chunks]
    assert sc.merge_chunks(cv, 2)["per_deal_bb_cv"] == total[:50][::-1] + total[50:][::-1]
    assert "per_deal_bb_cv" not in sc.merge_chunks([cv[0], chunks[1]], 2)
    # So do the failed-search counts (a match's two, LBR's two).
    keys = ("a_search_failed", "b_search_failed", "search_failed", "model_search_failed")
    counted = [dict(chunks[0], a_search_failed=[1, 2], b_search_failed=[0, 3], search_failed=[4, 0], model_search_failed=[0, 5]),
               dict(chunks[1], a_search_failed=[3, 0], b_search_failed=[1, 1], search_failed=[0, 2], model_search_failed=[6, 0])]
    merged = sc.merge_chunks(counted, 2)
    assert [merged[k] for k in keys] == [[4, 2], [1, 4], [4, 2], [6, 5]]
    merged = sc.merge_chunks([counted[0], chunks[1]], 2)
    assert not set(keys) & merged.keys()

    # An old receipt (no split) gets it only from a sidecar made for exactly it.
    old = _lbr_receipt(total, label="old")
    assert sc.lbr_split_of(old) is None and sc.lbr_part_scores(old, None) == {}
    good = {"receipt_hash": sc.canonical_hash(old), "per_deal_preflop_bb": pre}
    assert sc.lbr_split_of(old, good) == pre
    with pytest.raises(sc.ScorecardError, match="another receipt"):
        sc.lbr_split_of(old, {**good, "receipt_hash": "X"})
    with pytest.raises(sc.ScorecardError, match="wrong number"):
        sc.lbr_split_of(old, {**good, "per_deal_preflop_bb": pre[:-1]})
    with pytest.raises(sc.ScorecardError, match="do not match"):
        sc.lbr_part_scores(old, pre, hands_per_deal=1)
    new = _lbr_receipt(total, pre, label="new")
    assert sc.lbr_split_of(new) == pre
    assert sc.lbr_part_scores(old, pre) == {k: new["scores"][k] for k in
                                            ("exploitability_lbr_postflop", "exploitability_lbr_preflop")}
    assert sc.summary_of(old, pre)["lbr_postflop"] == sc.summary_of(new)["lbr_postflop"]
    assert "lbr_postflop" not in sc.summary_of(old)

    # compare: a postflop change only when both sides have the split; the
    # previous receipt's hash is unchanged by using a sidecar.
    shifted = _lbr_receipt([x + (1.0 if i % 3 else 0.0) for i, x in enumerate(total)], pre, label="cur")
    ch = sc.compare(shifted, old, pre)
    assert ch["previous_receipt_hash"] == good["receipt_hash"]
    # +1 bb on the 66 of 100 deals that went postflop: +0.66 bb/deal = +33 bb/100 hands.
    assert ch["exploitability_lbr_postflop"]["mean"] == pytest.approx(33.0, abs=1e-9)
    assert ch["exploitability_lbr_postflop"]["status"] == "MORE EXPLOITABLE ON THIS BENCHMARK"
    assert "exploitability_lbr_postflop" not in sc.compare(shifted, old)
    assert sc.compare(new, new)["exploitability_lbr_postflop"]["mean"] == 0.0
    text = sc.render({**shifted, "change": ch})
    assert "of which postflop" in text and "postflop LBR" in text


# ------------------------------------------------------------- real engine

pe = pytest.importorskip("parciver_engine")


def _script():
    spec = importlib.util.spec_from_file_location("scorecard_cli", ROOT / "scripts" / "scorecard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SMALL_ROWS = (
    sc.Row(sc.HEADLINE, "agent", 2_000, 11),
    sc.Row("station", "builtin", 1_000, 13, builtin="station"),
    sc.Row("sizes_x1.5", "perturbed", 1_000, 17, perturb=1.5),
)


@pytest.fixture(scope="module")
def small_setup():
    if not hasattr(pe.PlayAgent, "lbr"):
        pytest.skip("engine without the play layer")
    manifest = sc.make_manifest("small-test", {"path": "-", "sha256": "-"}, {"dir": "-", "sha256": {}},
                                rows=SMALL_ROWS, lbr={"deals": 1_000, "seed": 19}, blocks=20,
                                stakes={"sb": 1, "bb": 2, "stack": 4})
    eq = pe.solve_nlhe_small(1000)
    return manifest, eq


def test_identical_candidate_scores_exactly_zero_and_is_reproducible(small_setup):
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    cand = pe.PlayAgent.from_strategy(eq, "small")
    runs = []
    for threads in (1, 4):
        meas, lbr = cli.measure(manifest, cand, base, None, threads, log=lambda *_: None)
        runs.append((meas, lbr))
        scores = sc.score_receipt(manifest, meas, lbr, meas[sc.HEADLINE]["a_sources"])
        assert scores["strength"]["mean"] == 0.0 and scores["strength"]["se"] == 0.0
        assert scores["off_tree_share"] == 0.0
    # Worker-count invariance: identical numbers at 1 and 4 threads.
    assert runs[0] == runs[1]
    # A receipt compared with itself: zero change, unresolved status.
    receipt = {"manifest_hash": manifest["manifest_hash"], "scores": scores,
               "candidate": {"label": "x"}, "state": "complete"}
    ch = sc.compare(receipt, receipt)
    assert ch["strength"]["mean"] == 0.0
    assert ch["strength"]["status"] == "UNCLEAR / NO RESOLVED CHANGE"


def test_worse_candidate_scores_negative(small_setup):
    manifest, eq = small_setup
    cli = _script()
    uniform = {k: [1.0 / len(v)] * len(v) for k, v in eq.items()}
    base = pe.PlayAgent.from_strategy(eq, "small")
    cand = pe.PlayAgent.from_strategy(uniform, "small")
    meas, lbr = cli.measure(manifest, cand, base, None, 0, log=lambda *_: None)
    scores = sc.score_receipt(manifest, meas, lbr, meas[sc.HEADLINE]["a_sources"])
    assert scores["strength"]["hi"] < 0  # a uniform-random bot loses to equilibrium
    text = sc.render({"benchmark_id": "small-test", "scores": scores})
    assert "Strength vs frozen baseline bot" in text and "NO PREVIOUS COMPARISON" in text


def test_live_evaluation_streams_valid_events_and_matches_one_call(small_setup, tmp_path):
    from parciver import telemetry
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    uniform = {k: [1.0 / len(v)] * len(v) for k, v in eq.items()}
    cand = pe.PlayAgent.from_strategy(uniform, "small")
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest))
    # Reference: every row in ONE engine call.
    one_call, one_lbr = cli.measure(manifest, cand, base, None, 0, log=lambda *_: None, chunks=1)

    run_dir = tmp_path / "live-test"
    live = cli.Live(run_dir, manifest, 2)
    live.start(cli.live_config(manifest, manifest_path, candidate={"label": "u"}, baseline_file="b",
                               previous=None, threads=2))
    receipt, path = cli.run_evaluation(manifest, manifest_path, cand, base, None, label="u", candidate={},
                                       previous=None, threads=0, live=live, log=lambda *_: None)
    live.end("completed", cli.end_summary(receipt, path))

    for name, res in receipt["raw"]["rows"].items():
        assert res["per_deal_bb"] == one_call[name]["per_deal_bb"], name
        assert res["a_sources"] == one_call[name]["a_sources"], name
    assert receipt["raw"]["lbr"]["per_deal_bb"] == one_lbr["per_deal_bb"]

    events = telemetry.read_events(run_dir, validate=True)
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    finals = {e["sc_row"]["name"]: e["sc_row"] for e in events if e.get("sc_row")}
    assert set(finals) == {r.name for r in sc.rows_of(manifest)} | {"LBR"}
    s = receipt["scores"]
    assert finals[sc.HEADLINE]["mean"] == pytest.approx(s["strength"]["mean"], abs=1e-9)
    assert finals["LBR"]["hi"] == pytest.approx(s["exploitability_lbr"]["hi"], abs=1e-9)
    progress = [e["sc_progress"] for e in events if e.get("sc_progress")]
    assert len(progress) == 10 * (len(finals))  # 10 chunks per row
    assert progress[-1]["units_done"] == progress[-1]["units_total"]
    assert events[-1]["summary"]["scores"]["strength"]["mean"] == s["strength"]["mean"]

    # Against itself as the previous receipt: every paired change is exactly 0.
    ch = sc.compare(receipt, receipt)
    assert all(c["mean"] == 0.0 for c in ch["rows"].values())
    assert ch["exploitability_lbr_postflop"]["mean"] == 0.0
    assert receipt["stakes"] == manifest["stakes"]
    assert "1/2, 2bb heads-up" in sc.render(receipt)

    # LBR's split: recorded, chunk-invariant, and the parts add up to the total.
    assert receipt["raw"]["lbr"]["per_deal_preflop_bb"] == one_lbr["per_deal_preflop_bb"]
    po, pr = s["exploitability_lbr_postflop"], s["exploitability_lbr_preflop"]
    assert po["mean"] + pr["mean"] == pytest.approx(s["exploitability_lbr"]["mean"], abs=1e-9)
    assert finals["LBR"]["postflop"]["mean"] == pytest.approx(po["mean"], abs=1e-9)
    assert finals["LBR"]["preflop"]["hi"] == pytest.approx(pr["hi"], abs=1e-9)
    assert sc.summary_of(receipt)["lbr_postflop"]["mean"] == po["mean"]


class _Reporting:
    """A real agent whose `at`-th engine call (rows then LBR, 10 chunks each)
    reports `counts` (None: the key is absent, as from an engine before 1 Oct)."""

    def __init__(self, agent, at, **counts):
        self.agent, self.at, self.counts, self.calls = agent, at, counts, 0

    def _done(self, res):
        res = dict(res)
        if self.calls == self.at:
            for k, v in self.counts.items():
                res.pop(k) if v is None else res.__setitem__(k, v)
        self.calls += 1
        return res

    def match_against(self, *a, **kw):
        return self._done(self.agent.match_against(*a, **kw))

    def lbr(self, *a, **kw):
        return self._done(self.agent.lbr(*a, **kw))


def test_a_failed_search_stops_the_evaluation(small_setup, tmp_path, monkeypatch, capsys):
    """A decision that reached a failed search (redone by the built-in solver, or
    decided by the blueprint) means the agent measured is not the one named: the
    run stops at that chunk and writes nothing, as search_h2h and search_lbr do.
    A search-enabled candidate's results must carry the counts."""
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    cand = pe.PlayAgent.from_strategy(eq, "small")
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest))
    blob = tmp_path / "cand.bin"
    blob.write_bytes(b"candidate")
    info = {"path": str(blob), "sha256": sc.sha256_file(blob)}
    search = {**info, "search": {"streets": "river"}}

    labels = iter(range(100))

    def run(agent, candidate):  # a new label each: two runs can start in the same second
        return cli.run_evaluation(manifest, manifest_path, agent, base, None, label=f"u{next(labels)}",
                                  candidate=candidate, previous=None, threads=0, live=None, log=lambda *_: None)

    # Calls 0-9 the headline (200 deals each), 10-19 station, 20-29 sizes_x1.5, 30-39 LBR (100 each).
    for at, counts, match in ((13, {"a_search_failed": [0, 1]}, r"station deals 300\.\.399: .*'a_search_failed': \[0, 1\]"),
                              (25, {"b_search_failed": [2, 0]}, r"sizes_x1\.5 deals 500\.\.599: .*'b_search_failed'"),
                              (31, {"model_search_failed": [1, 0]}, r"LBR deals 100\.\.199: .*'model_search_failed'"),
                              (39, {"search_failed": [0, 4]}, r"LBR deals 900\.\.999: .*'search_failed': \[0, 4\]")):
        for candidate in (info, search):
            with pytest.raises(sc.ScorecardError, match=match):
                run(_Reporting(cand, at, **counts), candidate)
    for at, key, match in ((2, "b_search_failed", r"baseline deals 400\.\.599: no b_search_failed"),
                           (37, "model_search_failed", r"LBR deals 700\.\.799: no model_search_failed")):
        with pytest.raises(sc.ScorecardError, match=match + r" \(the engine predates"):
            run(_Reporting(cand, at, **{key: None}), search)
        run(_Reporting(cand, at, **{key: None}), info)  # without search the counts are not required
    assert len(list(tmp_path.glob("receipt-*"))) == 2
    receipt, path = run(_Reporting(cand, -1), search)
    assert receipt["raw"]["rows"][sc.HEADLINE]["a_search_failed"] == [0, 0]
    assert receipt["raw"]["lbr"]["model_search_failed"] == [0, 0]

    # lbr-split reruns LBR under the same rule.
    receipt["raw"]["lbr"].pop("per_deal_preflop_bb")
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(cli, "_checked_manifest", lambda p: json.loads(p.read_text()))
    failing = _Reporting(cand, 4, search_failed=[1, 0])
    monkeypatch.setattr(cli, "_candidate_agent", lambda m, p, h, s, tr, op: (None, failing, "x"))
    assert cli.main(["lbr-split", "--receipt", str(path)]) == 2
    assert "LBR deals 400..499: decisions reached a failed search {'search_failed': [1, 0]}" in capsys.readouterr().err
    failing = _Reporting(cand, 6, model_search_failed=None)  # a search receipt: the counts are required
    assert cli.main(["lbr-split", "--receipt", str(path)]) == 2
    assert "LBR deals 600..699: no model_search_failed (the engine predates" in capsys.readouterr().err
    assert not cli._sidecar_path(path).exists()


def test_evaluate_ends_its_run_cleanly_on_a_failed_search(small_setup, tmp_path, monkeypatch, capsys):
    """Through the command: exit 2 with the reason, the run's events end with an
    error, and the dashboard never showed the chunk that failed."""
    from parciver import telemetry
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest))
    blob = tmp_path / "cand.bin"
    blob.write_bytes(b"candidate")

    class Baseline:
        def play_agent(self):
            return base

        def s32_policy(self):
            return None

    agent = _Reporting(pe.PlayAgent.from_strategy(eq, "small"), 13, a_search_failed=[0, 1])
    monkeypatch.setattr(cli, "_checked_manifest", lambda p: json.loads(p.read_text()))
    monkeypatch.setattr(cli, "_candidate_agent", lambda *a: (Baseline(), agent, "x"))
    assert cli.main(["evaluate", "--benchmark", str(manifest_path), "--candidate", str(blob), "--label", "u",
                     "--search", "river", "--no-dashboard"]) == 2
    assert "station deals 300..399: decisions reached a failed search" in capsys.readouterr().err
    assert not list(tmp_path.glob("receipt-*"))
    (run,) = tmp_path.glob("live-*")
    events = telemetry.read_events(run, validate=True)
    assert events[-1]["type"] == "run_end" and events[-1]["reason"] == "error"
    assert "station deals 300..399" in events[-1]["summary"]["error"]
    shown = [(e["sc_progress"]["row"], e["sc_progress"]["done"]) for e in events if e.get("sc_progress")]
    assert shown[-1] == ("station", 300)


def test_evaluate_passes_and_records_the_open_reading(small_setup, tmp_path, monkeypatch):
    """`--open-reading` reaches the candidate agent and is recorded in the
    receipt's candidate when it is not snap (receipts from before it read as snap)."""
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest))
    blob = tmp_path / "cand.bin"
    blob.write_bytes(b"candidate")

    class Baseline:
        def play_agent(self):
            return base

        def s32_policy(self):
            return None

    seen = []
    monkeypatch.setattr(cli, "_checked_manifest", lambda p: json.loads(p.read_text()))
    monkeypatch.setattr(cli, "_candidate_agent", lambda m, p, h, s, tr, op: seen.append(op) or (Baseline(), base, "x"))
    for reading in ("limp", "snap"):
        assert cli.main(["evaluate", "--benchmark", str(manifest_path), "--candidate", str(blob), "--label", reading,
                         "--open-reading", reading, "--no-dashboard"]) == 0
    assert seen == ["limp", "snap"]
    cands = {c["label"]: c for c in (json.loads(p.read_text())["candidate"] for p in tmp_path.glob("receipt-*.json"))}
    assert cands["limp"]["open_reading"] == "limp" and "open_reading" not in cands["snap"]


def test_lbr_split_backfill_writes_a_verified_sidecar(small_setup, tmp_path, monkeypatch, capsys):
    """`lbr-split` on a receipt from before the split: reruns LBR, requires an
    exact reproduction, writes a sidecar; history and --previous then use it."""
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    uniform = {k: [1.0 / len(v)] * len(v) for k, v in eq.items()}
    cand = pe.PlayAgent.from_strategy(uniform, "small")
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest))
    blob = tmp_path / "cand.bin"
    blob.write_bytes(b"candidate")
    receipt, path = cli.run_evaluation(manifest, manifest_path, cand, base, None, label="u",
                                       candidate={"path": str(blob), "sha256": sc.sha256_file(blob)},
                                       previous=None, threads=0, live=None, log=lambda *_: None)
    split = receipt["raw"]["lbr"].pop("per_deal_preflop_bb")
    for p in sc.LBR_PARTS:
        receipt["scores"].pop(f"exploitability_lbr_{p}")
    old = path.with_name("receipt-20000101T000000Z-old.json")
    old.write_text(json.dumps(receipt))
    path.unlink()
    monkeypatch.setattr(cli, "_checked_manifest", lambda p: json.loads(p.read_text()))
    translations = []
    monkeypatch.setattr(cli, "_candidate_agent",
                        lambda m, p, h, s, tr, op: translations.append((tr, op)) or (None, cand, "uniform"))

    assert cli.main(["lbr-split", "--receipt", str(old)]) == 0
    side = cli._sidecar_path(old)
    assert json.loads(side.read_text())["per_deal_preflop_bb"] == split
    assert translations == [("snap", "snap")]  # receipts from before the options
    assert cli._load_receipt(old) == (receipt, split)
    assert cli.main(["lbr-split", "--receipt", str(old)]) == 0
    assert "already split" in capsys.readouterr().out
    hist = cli._history(manifest_path, manifest)
    assert hist[0]["lbr_postflop"]["mean"] == pytest.approx(
        sc.summarize(sc.lbr_parts(receipt["raw"]["lbr"]["per_deal_bb"], split, manifest["blocks"], 2)["postflop"])
        ["mean"], abs=1e-12)
    assert cli.main(["history", "--benchmark", str(manifest_path)]) == 0
    assert "postflop LBR" in capsys.readouterr().out
    # The postflop change needs the split on both sides (here only the previous has it).
    assert "exploitability_lbr_postflop" not in sc.compare(receipt, receipt, split)

    # A receipt the engine does not reproduce gets no sidecar.
    side.unlink()
    first = receipt["raw"]["lbr"]["per_deal_bb"][0]
    receipt["raw"]["lbr"]["per_deal_bb"][0] += 1.0
    old.write_text(json.dumps(receipt))
    assert cli.main(["lbr-split", "--receipt", str(old)]) == 2
    assert "did not reproduce" in capsys.readouterr().err and not side.exists()
    # The rerun plays the receipt's translation option.
    receipt["raw"]["lbr"]["per_deal_bb"][0] = first
    receipt["candidate"]["translation"] = "standin"
    old.write_text(json.dumps(receipt))
    assert cli.main(["lbr-split", "--receipt", str(old)]) == 0
    assert translations[-1] == ("standin", "snap")


def test_init_at_another_stack_depth_is_a_separate_benchmark(tmp_path):
    cli = _script()
    abs_dir = tmp_path / "abstraction"
    abs_dir.mkdir()
    for name in cli.ABSTRACTION_FILES:
        (abs_dir / name).write_bytes(name.encode())
    baseline = tmp_path / "blueprint.bin"
    baseline.write_bytes(b"bp")
    out = tmp_path / "scorecard-v2"

    def init(*extra):
        return cli.main(["init", "--baseline", str(baseline), "--abstraction-dir", str(abs_dir),
                         "--out", str(out), *extra])

    manifests = {}
    for extra, directory, bid, stack in [
        ((), "scorecard-v2", "scorecard-v2", 10_000),
        (("--stack", "200"), "scorecard-v2-200bb", "scorecard-v2-200bb", 20_000),
        (("--stack", "200", "--quick"), "scorecard-v2-200bb-quick", "scorecard-v2-200bb-quick", 20_000),
    ]:
        assert init(*extra) == 0
        m = json.loads((tmp_path / directory / "benchmark.json").read_text())
        sc.check_manifest(m)
        assert m["benchmark_id"] == bid
        assert m["stakes"] == {"sb": 50, "bb": 100, "stack": stack}
        manifests[directory] = m
    # Depth is part of the definition: a 100bb receipt can never be a --previous here.
    assert manifests["scorecard-v2"]["manifest_hash"] != manifests["scorecard-v2-200bb"]["manifest_hash"]
    assert manifests["scorecard-v2-200bb"]["rows"] == manifests["scorecard-v2"]["rows"]
    # Explicit --stack 100 is the default benchmark (already initialized, so refused).
    assert init("--stack", "100") == 2
    assert init("--stack", "0") == 2


def test_realistic_row_is_reproducible_and_worker_invariant(small_setup):
    manifest, eq = small_setup
    cli = _script()
    base = pe.PlayAgent.from_strategy(eq, "small")
    try:
        base.match_against("realistic", 2, 1, 2, 4, other=base)
    except ValueError:
        pytest.skip("the installed engine predates the realistic-sizing opponent")
    rows = (sc.Row(sc.HEADLINE, "agent", 1_000, 11), sc.Row("realistic", "realistic", 1_000, 23))
    m = sc.make_manifest("realistic-test", {"path": "-", "sha256": "-"}, {"dir": "-", "sha256": {}},
                         rows=rows, lbr={"deals": 1_000, "seed": 19}, blocks=20, stakes=manifest["stakes"])
    runs = [cli.measure(m, base, base, None, threads, log=lambda *_: None) for threads in (1, 4)]
    assert runs[0] == runs[1]
    scores = sc.score_receipt(m, runs[0][0], runs[0][1], runs[0][0][sc.HEADLINE]["a_sources"])
    assert "realistic" in sc.render({"benchmark_id": "realistic-test", "scores": scores})
    assert any(r.kind == "realistic" for r in sc.DEFAULT_ROWS)

