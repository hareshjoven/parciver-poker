# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D3-SWEEP's driver (scripts/d3_sweep.py): the registered rules as pure functions, and every refusal. No GPU, no data."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import d3_sweep as sw  # noqa: E402
import turn_net as tn  # noqa: E402

CONTROL = dict(all=0.0118, real=0.0022, random=0.0214)
PREREG = Path(__file__).resolve().parents[1] / "records" / "d3-sweep" / "PREREG.md"


def sc(loss, real=0.0021, random=0.02):
    return dict(all=loss, real=real, random=random)


GOOD = 0.98 * CONTROL["all"] - 1e-9  # inside the bar


def test_arm_table_is_the_registered_one():
    text = PREREG.read_text()
    for t in sw.ARM_TAGS + (sw.COMBO_TAG, sw.CONFIRM_TAG):
        assert t in text, t
    assert sw.ARM_TAGS == ("d3s-drop05", "d3s-drop10", "d3s-ema999", "d3s-ema9999", "d3s-rw")
    got = {x["tag"]: (x["family"], {k: x[k] for k in sw.OFF if k in x}) for x in sw.ARMS}
    assert got == {"d3s-drop05": ("dropout", dict(dropout=0.05)), "d3s-drop10": ("dropout", dict(dropout=0.10)),
                   "d3s-ema999": ("ema", dict(ema=0.999)), "d3s-ema9999": ("ema", dict(ema=0.9999)), "d3s-rw": ("loss", dict(loss="rw"))}
    assert (sw.BAR, sw.COMBO_BAR, sw.CONFIRM_BAR) == (0.98, 0.99, 0.99)


def test_table_arms_are_n1s_recipe_plus_one_option():
    for x in sw.ARMS:
        a = sw.table_arm(x["tag"])
        assert (a["width"], a["wd"], a["data"]) == (2048, 0.1, "union")
        assert sum(a[k] != sw.OFF[k] for k in sw.OFF) == 1


def test_wins_needs_both_clauses_and_the_bar_is_inclusive():
    assert sw.wins(sc(0.98 * CONTROL["all"], real=CONTROL["real"]), CONTROL, 0.98)  # exactly on both bars
    assert not sw.wins(sc(0.98 * CONTROL["all"] + 1e-12), CONTROL, 0.98)
    assert not sw.wins(sc(0.01, real=CONTROL["real"] + 1e-12), CONTROL, 0.98)  # the real-row clause


def test_round1_no_winner_and_failed_arm():
    s = {t: sc(0.0118) for t in sw.ARM_TAGS}
    s["d3s-rw"] = None
    r = sw.round1(s, CONTROL)
    assert r["single"] is None and r["combo"] is None and not any(r["wins"].values())


def test_round1_one_family_is_not_a_combo_and_best_is_lowest_loss():
    s = {t: sc(0.0118) for t in sw.ARM_TAGS}
    s["d3s-drop05"], s["d3s-drop10"] = sc(GOOD), sc(GOOD - 0.0001)
    r = sw.round1(s, CONTROL)
    assert r["family_best"] == {"dropout": "d3s-drop10"} and r["single"] == "d3s-drop10" and r["combo"] is None


def test_round1_two_families_call_the_combo_from_each_best():
    s = {t: sc(0.0118) for t in sw.ARM_TAGS}
    s["d3s-drop05"], s["d3s-ema9999"], s["d3s-ema999"] = sc(GOOD - 0.0002), sc(GOOD - 0.0003), sc(GOOD)
    r = sw.round1(s, CONTROL)
    assert r["single"] == "d3s-ema9999"
    assert r["combo"]["options"] == dict(dropout=0.05, ema=0.9999, loss="huber") and r["combo"]["tag"] == sw.COMBO_TAG


def test_round1_exact_tie_goes_to_the_arm_lower_in_the_table():
    s = {t: sc(0.0118) for t in sw.ARM_TAGS}
    s["d3s-drop05"] = s["d3s-drop10"] = sc(GOOD)
    assert sw.round1(s, CONTROL)["single"] == "d3s-drop10"
    s = {t: sc(0.0118) for t in sw.ARM_TAGS}
    s["d3s-ema999"] = s["d3s-rw"] = sc(GOOD)
    assert sw.round1(s, CONTROL)["single"] == "d3s-rw"


def test_a_real_row_regression_blocks_a_family():
    s = {t: sc(0.0118) for t in sw.ARM_TAGS}
    s["d3s-drop05"] = sc(GOOD, real=CONTROL["real"] * 1.5)
    s["d3s-rw"] = sc(GOOD)
    r = sw.round1(s, CONTROL)
    assert r["family_best"] == {"loss": "d3s-rw"} and r["combo"] is None


def test_combo_decision():
    one = sc(GOOD)
    assert sw.combo_decision("d3s-rw", one, sc(one["all"] * 0.99), CONTROL)["candidate"] == sw.COMBO_TAG  # on the 0.99 bar
    d = sw.combo_decision("d3s-rw", one, sc(one["all"] * 0.99 + 1e-9), CONTROL)
    assert d["candidate"] == "d3s-rw" and d["combo_passes_rule2"] and not d["combo_beats_single"]
    d = sw.combo_decision("d3s-rw", one, sc(0.0001, real=CONTROL["real"] + 1e-6), CONTROL)  # best loss, fails the real clause
    assert d["candidate"] == "d3s-rw" and not d["combo_passes_rule2"]
    assert sw.combo_decision("d3s-rw", one, None, CONTROL)["candidate"] == "d3s-rw"  # the combo failed (NaN)


def test_confirmed():
    c = dict(all=0.0119, real=0.0022)
    assert sw.confirmed(sc(0.99 * c["all"], real=c["real"]), c)
    assert not sw.confirmed(sc(0.99 * c["all"] + 1e-12, real=c["real"]), c)
    assert not sw.confirmed(sc(0.001, real=c["real"] + 1e-12), c)


def test_candidate_and_verdict_branches():
    none = dict(single=None, combo=None)
    assert sw.candidate_of(none, None)[0] is None
    assert sw.verdict_of(none, None, None)["pin"] == "N1"
    one = dict(single="d3s-rw", combo=None)
    assert sw.candidate_of(one, None)[0] == "d3s-rw"
    with pytest.raises(SystemExit, match="`confirm` first"):
        sw.verdict_of(one, None, None)
    v = sw.verdict_of(one, None, dict(confirmed=True, options=dict(dropout=0.0, ema=0.0, loss="rw")))
    assert v["pin"] == "d3s-rw" and v["recipe"]["loss"] == "rw"
    v = sw.verdict_of(one, None, dict(confirmed=False, options={}))
    assert v["pin"] == "N1" and v["recipe"] == sw.OFF and "no second place" in v["why"]
    two = dict(single="d3s-rw", combo=dict(options=dict(dropout=0.05, ema=0.0, loss="rw")))
    with pytest.raises(SystemExit, match="combo is called for"):
        sw.candidate_of(two, None)
    assert sw.candidate_of(two, dict(candidate=sw.COMBO_TAG))[0] == sw.COMBO_TAG
    assert sw.candidate_of(two, dict(candidate="d3s-rw"))[0] == "d3s-rw"


def test_candidate_arm_options_follow_the_candidate():
    sel = dict(combo=dict(options=dict(dropout=0.05, ema=0.999, loss="huber")))
    a = sw.candidate_arm(sel, sw.COMBO_TAG, sw.CONFIRM_TAG)
    assert (a["tag"], a["dropout"], a["ema"], a["loss"]) == (sw.CONFIRM_TAG, 0.05, 0.999, "huber")
    a = sw.candidate_arm(sel, "d3s-rw", sw.CONFIRM_TAG)
    assert (a["tag"], a["loss"], a["dropout"], a["ema"]) == (sw.CONFIRM_TAG, "rw", 0.0, 0.0)


def test_ema_horizon_round_trip():
    assert sw.ema_horizon_epochs(0.999, 464) == pytest.approx(1 / 0.464) and sw.ema_horizon_epochs(0.9999, 464) == pytest.approx(21.55, abs=0.01)
    h = sw.ema_horizon_epochs(0.9999, 464)
    assert sw.ema_d_for(h, 464) == pytest.approx(0.9999)
    assert sw.ema_d_for(h, 928) == pytest.approx(1 - 0.0001 / 2)  # twice the steps an epoch: half the per-step weight


TREES = dict(src="a", scripts="b")


def test_check_refused():
    assert "run `d3_sweep check`" in sw.check_refused(None, TREES, "u")
    assert "did not pass" in sw.check_refused(dict(passed=False), TREES, "u")
    assert "changed since" in sw.check_refused(dict(passed=True, trees=dict(src="a", scripts="c"), union="u"), TREES, "u")
    assert "changed since" in sw.check_refused(dict(passed=True, trees={}, union="u"), TREES, "u")
    assert "data" in sw.check_refused(dict(passed=True, trees=TREES, union="v"), TREES, "u")
    assert sw.check_refused(dict(passed=True, trees=TREES, union="u"), TREES, "u") is None


def test_train_plan():
    A = set(sw.ARM_TAGS)
    assert sw.train_plan(None, set(), set(), None, None, False) == list(sw.ARM_TAGS)
    assert sw.train_plan(None, {"d3s-drop05", "d3s-rw"}, {"d3s-ema999"}, None, None, False) == ["d3s-drop10", "d3s-ema9999"]
    assert sw.train_plan("d3s-rw", set(), set(), None, None, False) == ["d3s-rw"]
    with pytest.raises(SystemExit, match="not in the table"):
        sw.train_plan("d3s-drop20", set(), set(), None, None, False)
    with pytest.raises(SystemExit, match="not in the table"):
        sw.train_plan(sw.CONFIRM_TAG, set(), set(), None, None, False)  # the confirmation is `confirm`'s
    with pytest.raises(SystemExit, match="is finished"):
        sw.train_plan("d3s-rw", {"d3s-rw"}, set(), None, None, False)
    assert sw.train_plan("d3s-rw", {"d3s-rw"}, set(), None, None, True) == ["d3s-rw"]
    with pytest.raises(SystemExit, match="--redo needs --arm"):
        sw.train_plan(None, {"d3s-rw"}, set(), None, None, True)
    with pytest.raises(SystemExit, match="FAILED"):
        sw.train_plan("d3s-rw", set(), {"d3s-rw"}, None, None, True)  # failed: not retried, not even by --redo
    with pytest.raises(SystemExit, match="every round-1 arm"):
        sw.train_plan(None, A, set(), None, None, False)
    sel = dict(combo=None)
    with pytest.raises(SystemExit, match="round 1 is decided"):
        sw.train_plan("d3s-rw", A, set(), sel, None, True)
    with pytest.raises(SystemExit, match="does not call for d3s-combo"):
        sw.train_plan(sw.COMBO_TAG, A, set(), None, None, False)
    with pytest.raises(SystemExit, match="does not call for d3s-combo"):
        sw.train_plan(sw.COMBO_TAG, A, set(), sel, None, False)
    sel = dict(combo=dict(options={}))
    assert sw.train_plan(sw.COMBO_TAG, A, set(), sel, None, False) == [sw.COMBO_TAG]
    with pytest.raises(SystemExit, match="written once"):
        sw.train_plan(sw.COMBO_TAG, A, set(), sel, dict(candidate="x"), False)


def hist(*losses):
    return [dict(epoch=i + 1, train_loss=l, val_loss=l / 2, val_mae=1.0, seconds=i) for i, l in enumerate(losses)]


def rec(h, **kw):
    return dict(dict(width=2048, depth=5, wd=0.1, seed=0, data="d", train_samples=1, val_samples=2, pools=2, epochs=200, batch=2048, lr=1e-3,
                     canonical=False, history=h), **kw)


def test_compare_history_is_bit_exact_and_checks_options_and_schedule():
    n1 = rec(hist(1.0, 0.5, 0.25, 0.1, 0.05))
    assert sw.compare_history(rec(hist(1.0, 0.5, 0.25)), n1, 3)["equal"]
    assert not sw.compare_history(rec(hist(1.0, 0.5, 0.25 + 1e-15)), n1, 3)["equal"]
    assert not sw.compare_history(rec(hist(1.0, 0.5)), n1, 3)["equal"]  # too short
    assert not sw.compare_history(rec(hist(1.0, 0.5, 0.25), epochs=3), n1, 3)["equal"]  # not the 200-epoch schedule
    assert not sw.compare_history(rec(hist(1.0, 0.5, 0.25), dropout=0.05), n1, 3)["equal"]  # an option is on


def test_compare_scores_per_source():
    want = {lab: dict(rows=10, loss=0.5, mae=1.5) for lab in sw.LABELS}
    assert all(sw.compare_scores({lab: dict(want[lab], rw_loss=9.0) for lab in sw.LABELS}, want).values())
    got = {lab: dict(want[lab]) for lab in sw.LABELS}
    got["real"]["loss"] = 0.5 + 1e-16
    assert sw.compare_scores(got, want) == dict(all=True, real=False, random=True)


class FakeLs:
    def __init__(self, peers=None):
        self.peers = peers or []

    def gpu_state(self):
        return dict(peers=self.peers, util=5, apps=["dwm.exe"])

    def busy(self, g):
        return bool(g["peers"])

    def utc(self):
        return "2026-11-09T00:00:00Z"


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path / "nets")
    monkeypatch.setattr(sw, "RECORDS", tmp_path / "rec")
    (tmp_path / "nets").mkdir()
    (tmp_path / "rec").mkdir()
    return tmp_path


def test_run_arm_records_a_nan_as_failed_and_does_not_void(box, monkeypatch):
    def boom(*a, **k):
        raise FloatingPointError("[d3s-rw] non-finite loss at epoch 7: stopped")

    monkeypatch.setattr(tn, "d2b_train_arm", boom)
    arm = sw.table_arm("d3s-rw")
    out = sw.run_arm({}, {}, arm, 0, "abc", FakeLs(), set(), {})
    assert out is None
    f = json.loads((sw.RECORDS / "failed.json").read_text())
    assert f == [dict(tag="d3s-rw", error="[d3s-rw] non-finite loss at epoch 7: stopped", epochs=0, commit="abc",
                      options=dict(dropout=0.0, ema=0.0, loss="rw"))]
    assert sw.void_net.__name__ == "void_net" and not list(tn.NETS.glob("*-void-*"))
    with pytest.raises(SystemExit, match="not VOID"):
        sw.void_net("d3s-rw", "cause", FakeLs())


def test_run_arm_refuses_a_peer_before_anything_trains(box, monkeypatch):
    def fake(views, parts, arm, held, gpu, seed, commit=None, **k):
        gpu()
        raise AssertionError("trained")

    monkeypatch.setattr(tn, "d2b_train_arm", fake)
    with pytest.raises(SystemExit, match="nothing trained"):
        sw.run_arm({}, {}, sw.table_arm("d3s-rw"), 0, "abc", FakeLs(peers=[123]), set(), {})
    assert not (tn.NETS / "d3s-rw").exists()


def test_void_net_refusals(box):
    ls = FakeLs()
    with pytest.raises(SystemExit, match="needs a cause"):
        sw.void_net("d3s-rw", "  ", ls)
    with pytest.raises(SystemExit, match="needs a cause"):
        sw.void_net("d2b-seed1", "cause", ls)  # not a sweep net
    with pytest.raises(SystemExit, match="no net to void"):
        sw.void_net("d3s-rw", "cause", ls)
    (tn.NETS / "d3s-rw").mkdir()
    (sw.RECORDS / "selection.json").write_text("{}")
    with pytest.raises(SystemExit, match="needs an amendment first"):
        sw.void_net("d3s-rw", "cause", ls)  # the result exists
    (sw.RECORDS / "selection.json").unlink()
    with pytest.raises(SystemExit, match="another process"):
        sw.void_net("d3s-rw", "cause", FakeLs(peers=[9]))
    assert sw.void_net("d3s-rw", "a compute job", ls) == 0
    assert not (tn.NETS / "d3s-rw").exists()
    v = json.loads((tn.NETS / "d3s-rw-void-1" / "void.json").read_text())
    assert v["cause"] == "a compute job" and v["epochs"] == 0
    assert sw.voided() == [v]


def test_void_of_the_combo_and_confirm_use_their_own_result_files(box):
    assert sw.result_file(sw.COMBO_TAG).name == "combo.json" and sw.result_file(sw.CONFIRM_TAG).name == "confirm.json"
    assert sw.result_file("d3s-drop05").name == "selection.json"
    (tn.NETS / sw.COMBO_TAG).mkdir()
    (sw.RECORDS / "selection.json").write_text("{}")  # round 1 is read: the combo may still be voided (its own result is combo.json)
    assert sw.void_net(sw.COMBO_TAG, "a compute job", FakeLs()) == 0


def test_nets_refused_counts_data_and_the_void_list(box):
    ok = dict(nets=[], failed=[], voided=[], union="u")
    assert "not the registered" in sw.nets_refused(ok, "u", sw.ARM_TAGS)
    full = dict(ok, failed=[dict(tag=t) for t in sw.ARM_TAGS])
    assert "data" in sw.nets_refused(dict(full, union="v"), "u", sw.ARM_TAGS)
    assert "voided" in sw.nets_refused(dict(full, voided=[dict(tag="x")]), "u", sw.ARM_TAGS)
    assert sw.nets_refused(full, "u", sw.ARM_TAGS) is None  # five failed arms: nothing to read, nothing wrong


def test_main_pairs_void_with_tag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["d3_sweep.py", "train", "--void", "cause"])
    with pytest.raises(SystemExit, match="go together"):
        sw.main()


def test_code_trees_are_two_hashes_and_a_git_failure_refuses(monkeypatch):
    t = sw.code_trees()
    assert set(t) == {"src", "scripts"} and all(len(v) == 40 for v in t.values())

    class Bad:
        returncode, stdout = 128, ""

    monkeypatch.setattr(sw.subprocess, "run", lambda *a, **k: Bad())
    with pytest.raises(SystemExit, match="git could not say"):
        sw.code_trees()


def test_rules_text_matches_the_prereg_numbers():
    text = PREREG.read_text()
    for needle in ("0.98", "0.99", "d3s-combo", "d3s-confirm", "0.9999", "0.999"):
        assert needle in text, needle


def test_guard_compares_the_peak_with_the_free_commit_from_before_the_data_is_loaded(monkeypatch):
    """The measured peak includes the data the process is about to hold, so the free commit it is compared with must be read first
    (read after the load, 46 GiB free looked like 29 and refused a run that fits)."""
    free = [46.0]
    monkeypatch.setattr(sw.d3, "need_commit", lambda: "c")
    monkeypatch.setattr(sw.d3, "free_commit_gib", lambda: free[0])

    def load(a):
        free[0] = 29.0  # the loaded data now counts against the machine's free commit
        return {}, {"union": sw.np.array([tn.TRAIN] * 10)}

    monkeypatch.setattr(sw.tn, "d2b_data", load)
    monkeypatch.setattr(sw.tn, "d2b_refuse_other_data", lambda v: "u")
    monkeypatch.setattr(sw, "read_json", lambda p: dict(train_rows=10, peak_commit=31.8))
    monkeypatch.setattr(sw, "check_refused", lambda *a: None)
    monkeypatch.setattr(sw, "code_trees", lambda: {})
    monkeypatch.setattr(sw, "gpu_refused", lambda ls: None)

    class A:
        told = 31.8

    assert sw.guard(A(), None)[0] == "c"
    free[0] = 30.0  # really too little before the load: refused
    monkeypatch.setattr(sw.tn, "d2b_data", lambda a: ({}, {"union": sw.np.array([tn.TRAIN] * 10)}))
    with pytest.raises(SystemExit, match="not below the free commit"):
        sw.guard(A(), None)
