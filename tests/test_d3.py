# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D3's code stage (records/d3/PREREG.md section 5.0): the speed gate mode (5.0.6). More pieces are added below as they land."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_d3_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:  # the GPU solver needs torch/triton; CPU-only installs skip these tests
        if e.name in ("torch", "triton"):
            pytest.skip(f"needs {e.name} (GPU solver)")
        raise
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 5.0.6: flop_sym_speed.py --cand d3


def _speed(tmp_path, monkeypatch, n1_is_registered=True):
    sp = _script("flop_sym_speed")
    for t in ("d2b-w2048-wd0.1", "d3-w2048-wd0.1", "d3-other"):
        (tmp_path / t).mkdir()
        (tmp_path / t / "net.pt").write_bytes(t.encode())
    n1 = tmp_path / "d2b-w2048-wd0.1" / "net.pt"
    if n1_is_registered:
        monkeypatch.setattr(sp, "N1_SHA256", _sha(n1))
    monkeypatch.setattr(sp, "D3_RECORDS", tmp_path / "rec")
    monkeypatch.setattr(sp, "D3_OUT", tmp_path / "out")
    monkeypatch.setattr(sp, "D3_JOBS_SHA256", "j" * 64)
    monkeypatch.setattr(sp, "D2B_RECORDS", tmp_path / "d2b-must-stay-empty")
    monkeypatch.setattr(sp, "D2B_OUT", tmp_path / "d2b-out-must-stay-empty")
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "selection.json").write_text(json.dumps(dict(n2="d3-w2048-wd0.1", sha256={
        "d3-w2048-wd0.1": _sha(tmp_path / "d3-w2048-wd0.1" / "net.pt")})))
    return sp, tmp_path


def test_the_d3_speed_mode_exists_and_demands_net_b():
    src = (ROOT / "scripts" / "flop_sym_speed.py").read_text(encoding="utf-8")
    assert '"d3"' in src and 'choices=("sym", "orbit", "leafnet", "d2b", "d3")' in src
    assert '(a.cand in ("leafnet", "d2b", "d3")) != (a.net_b is not None)' in src


def test_d3_times_n1_against_the_selected_n2_only(tmp_path, monkeypatch):
    sp, t = _speed(tmp_path, monkeypatch)
    n1, n2, other = (str(t / x / "net.pt") for x in ("d2b-w2048-wd0.1", "d3-w2048-wd0.1", "d3-other"))
    sp.d3_check_nets(n1, n2)
    with pytest.raises(SystemExit, match="--net must be N1"):
        sp.d3_check_nets(n2, n2)
    with pytest.raises(SystemExit, match="--net-b must be N2"):
        sp.d3_check_nets(n1, other)  # not the selected net
    with pytest.raises(SystemExit, match="not N1"):
        sp.d3_check_nets(n1, n1)
    (t / "d3-w2048-wd0.1" / "net.pt").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="changed after the selection"):
        sp.d3_check_nets(n1, n2)


def test_the_registered_n1_sha_is_d2bs_selection_n1():
    sp = _script("flop_sym_speed")
    sel = json.loads((ROOT / "records" / "d2b" / "selection.json").read_text())
    assert sp.N1_SHA256 == sel["sha256"][sel["n1"]] == "54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72"


def _fake_rows(ratio_of_job):
    def two_net_rows(jobs, na, nb):
        rows = [dict(job=i, flop=[int(c) for c in j["board"]], group=1, cls="1/1", a=[1.0, 1.0], b=[ratio_of_job(i)] * 2,
                     ratio=ratio_of_job(i)) for i, j in enumerate(jobs)]
        return rows, {"A": type("S", (), {"symmetric": 0, "solves": 2 * len(jobs)})(),
                      "B": type("S", (), {"symmetric": 0, "solves": 2 * len(jobs)})()}
    return two_net_rows


def _jobs(n=479, flops=40):
    return [dict(board=(i % flops, 60, 61)) for i in range(n)]


def _run(sp, tmp_path, monkeypatch, ratio, jobs=None):
    monkeypatch.setattr(sp.vn, "load", lambda p: (None, None))
    monkeypatch.setattr(sp, "two_net_rows", _fake_rows(ratio))
    n1, n2 = (str(tmp_path / x / "net.pt") for x in ("d2b-w2048-wd0.1", "d3-w2048-wd0.1"))
    return sp.d3_speed(_jobs() if jobs is None else jobs, n1, n2, "j" * 64)


def test_d3_passes_iff_the_upper_end_of_the_median_ratio_is_at_most_1_02(tmp_path, monkeypatch):
    sp, t = _speed(tmp_path, monkeypatch)
    assert _run(sp, t, monkeypatch, lambda i: 1.0) == 0
    rep = json.loads((t / "rec" / "speed.json").read_text())
    assert rep["passed"] is True and rep["bar"] == 1.02 and rep["ci"] == [1.0, 1.0] and rep["jobs"] == 479 and rep["flops"] == 40
    assert rep["jobs_sha256"] == "j" * 64 and len(json.loads((t / "out" / "d3-speed.json").read_text())["rows"]) == 479
    assert not (t / "d2b-must-stay-empty").exists() and not (t / "d2b-out-must-stay-empty").exists()  # never records/d2b


def test_d3_fails_above_the_bar_and_the_bar_is_inclusive(tmp_path, monkeypatch):
    sp, t = _speed(tmp_path, monkeypatch)
    assert _run(sp, t, monkeypatch, lambda i: 1.021) == 5
    assert json.loads((t / "rec" / "speed.json").read_text())["passed"] is False
    (t / "rec" / "speed.json").unlink()
    (t / "out" / "d3-speed.json").unlink()
    assert _run(sp, t, monkeypatch, lambda i: 1.02) == 0  # upper end exactly 1.02 passes ("<= 1.02")


def test_d3_refuses_another_jobs_file_and_an_existing_result(tmp_path, monkeypatch):
    sp, t = _speed(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="registered 479 over 40"):
        _run(sp, t, monkeypatch, lambda i: 1.0, jobs=_jobs(478, 40))
    with pytest.raises(SystemExit, match="registered 479 over 40"):
        _run(sp, t, monkeypatch, lambda i: 1.0, jobs=_jobs(479, 39))
    assert _run(sp, t, monkeypatch, lambda i: 1.0) == 0
    before = (t / "rec" / "speed.json").read_text()
    with pytest.raises(SystemExit, match="exists"):
        _run(sp, t, monkeypatch, lambda i: 5.0)
    assert (t / "rec" / "speed.json").read_text() == before
    (t / "rec" / "speed.json").unlink()  # the rows file alone also blocks a second timing
    with pytest.raises(SystemExit, match="d3-speed.json exists"):
        _run(sp, t, monkeypatch, lambda i: 5.0)
    assert not (t / "rec" / "speed.json").exists()
    with pytest.raises(SystemExit, match="not the registered"):  # another jobs file, whatever its shape
        sp.d3_speed(_jobs(), "x", "y", "k" * 64)


def test_the_gate_reads_the_upper_end_of_the_flop_clustered_ci_not_the_median(tmp_path, monkeypatch):
    sp, t = _speed(tmp_path, monkeypatch)
    # 18 of 40 flops 30% slower, 22 faster: the median ratio is 0.9 (<= 1.02) but flop resamples often have a bad majority,
    # so the CI's upper end is 1.3. A median or lower-end rule would pass it; an unclustered bootstrap would see ~0.9.
    assert _run(sp, t, monkeypatch, lambda i: 1.3 if i % 40 < 18 else 0.9) == 5
    rep = json.loads((t / "rec" / "speed.json").read_text())
    assert rep["median_ratio"] <= 1.02 and rep["ci"][0] <= 1.02 < rep["ci"][1] and rep["passed"] is False


def test_the_ratio_is_n2_over_n1_and_a_is_the_net_arg(monkeypatch):
    sp = _script("flop_sym_speed")

    class Solver:
        def __init__(self, net, gamma, leaves):
            self.net, self.seconds, self.symmetric, self._tables = net, [], 0, {}

        def tables(self, flop):
            self._tables[flop] = 1

        def __call__(self, job):
            self.seconds.append(self.net.t)

    fake_fs = type("FS", (), dict(FlopSearch=Solver, flop_group=staticmethod(lambda f: [0]), TURNS=49))
    monkeypatch.setattr(sp, "fs", fake_fs)
    slow, fast = type("N", (), dict(t=2.0))(), type("N", (), dict(t=1.0))()
    rows, _ = sp.two_net_rows(_jobs(3), fast, slow)  # A = fast, B = slow: B/A = 2
    assert [r["ratio"] for r in rows] == [2.0] * 3 and rows[0]["a"] == [1.0, 1.0] and rows[0]["b"] == [2.0, 2.0]


def test_main_wires_n1_as_a_n2_as_b_checks_the_nets_and_pins_the_jobs_file(tmp_path, monkeypatch):
    import pickle
    import sys

    sp, t = _speed(tmp_path, monkeypatch)
    out = tmp_path / "evalout"
    out.mkdir()
    raw = pickle.dumps(_jobs())
    (out / "m1-jobs-sym.pkl").write_bytes(raw)
    n1, n2, other = (str(t / x / "net.pt") for x in ("d2b-w2048-wd0.1", "d3-w2048-wd0.1", "d3-other"))
    seen = {}
    monkeypatch.setattr(sp.vn, "load", lambda p: (("net", p), None))

    def rows(jobs, na, nb):
        seen["nets"] = (na, nb)
        return _fake_rows(lambda i: 1.0)(jobs, na, nb)

    monkeypatch.setattr(sp, "two_net_rows", rows)
    monkeypatch.setattr(sp, "D3_JOBS_SHA256", hashlib.sha256(raw).hexdigest())
    argv = ["x", "--cand", "d3", "--net", n1, "--net-b", n2, "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert sp.main() == 0
    assert seen["nets"] == (("net", n1), ("net", n2))
    assert json.loads((t / "rec" / "speed.json").read_text())["jobs_sha256"] == hashlib.sha256(raw).hexdigest()
    (t / "rec" / "speed.json").unlink()
    (t / "out" / "d3-speed.json").unlink()
    monkeypatch.setattr(sys, "argv", ["x", "--cand", "d3", "--net", n1, "--net-b", other, "--out", str(out)])
    with pytest.raises(SystemExit, match="--net-b must be N2"):  # main() runs the net check
        sp.main()
    monkeypatch.setattr(sys, "argv", ["x", "--cand", "d3", "--net", n2, "--net-b", n1, "--out", str(out)])
    with pytest.raises(SystemExit, match="--net must be N1"):  # and not swapped
        sp.main()
    (out / "m1-jobs-sym.pkl").write_bytes(raw + b" ")  # a different file
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="not the registered"):
        sp.main()


# ---------------------------------------------------------------- 5.0.7: flop_m1.py, Gate B (PREREG 5.5)

import types  # noqa: E402

X_SETTINGS = dict(seeds=[0], flop_iters=500, gamma=5.0)  # X_N1's, in runs/flop-eval-d2b/m1-x-d2b1.json
RULE = [5.0, 100, 0.2, 1000]  # ORACLE_GAMMA, CHECK_EVERY, TOLERANCE, ITERS
MENUS = None
ROW = dict(pot=200, hero=0, frozen=3, isomorphism=True, iters=500, br=[1.0, 2.0], expl=0.1, lo=-0.01, hi=0.01,
           value=0.0, profile=0.0, curve=[[100, 0.5, 0.0]], sec=10.0)


def _edit(path: Path, change) -> None:
    f = json.loads(path.read_text())
    change(f)
    path.write_text(json.dumps(f))


def _tree(path: Path) -> dict:
    return {p.name: _sha(p) for p in path.iterdir() if p.is_file()}


def _head3(out: Path, c: str, mode: str = "carry") -> dict:
    inputs = dict(spots=_sha(out / "m1-spots.json"), x=_sha(out / f"m1-x-{c}.npz"), engine="E", threads=12)
    if c == "d3n1":
        inputs.update(mode=mode, carry=_sha(out / "d3-carry.json"),
                      old_files=json.loads((out / "d3-carry.json").read_text())["files"])
        if mode == "resolve":
            inputs["failed"] = _sha(out / "d3-remake-failed.json")
    return dict(cand=c, menus=MENUS, rule=RULE, precision="f32", inputs=inputs)


def _gate_b3(tmp_path, monkeypatch, gap=0.0, solved=160, resolve=False):
    """runs/flop-eval-d2b (X_N1's 160 spots, rows and strategies), runs/flop-eval (the 40 old spots and V*), and the
    new folder with the solves done: X_d3n1 from the carried rows (or made again, `resolve`), X_d3n2 new."""
    m1 = _script("flop_m1")
    global MENUS
    MENUS = json.loads(json.dumps(m1.MENUS))  # the headers the solve writes
    repo, d3old, d2bold, rec2, rec3, out = (tmp_path / x for x in ("repo", "d3old", "d2bold", "rec2", "rec3", "eval"))
    for p in (repo, d3old, d2bold, rec2, rec3, out):
        p.mkdir(parents=True)
    for n in ("n0", "n1", "n2"):
        (repo / f"{n}.pt").write_bytes(n.upper().encode())
    for k, v in dict(REPO=repo, D3_OLD=d3old, D2B_OLD=d2bold, D2B_RECORDS=rec2, D3_RECORDS=rec3, N0_SHA256=_sha(repo / "n0.pt"),
                     N1_SHA256=_sha(repo / "n1.pt"), engine_sha=lambda pe: "E").items():
        monkeypatch.setattr(m1, k, v)
    (rec2 / "gate-a.json").write_text(json.dumps(dict(proceed=True, sha256=dict(N0=_sha(repo / "n0.pt"), N1=_sha(repo / "n1.pt")))))
    (rec3 / "gate-a.json").write_text(json.dumps(dict(proceed=True, n2="d3-w2048-wd0.1",
                                                      sha256=dict(N1=_sha(repo / "n1.pt"), N2=_sha(repo / "n2.pt")))))
    rng = np.random.default_rng(0)
    spots = [dict(board=[0, 5, 10], pot=200 + i, hero=i % 2, ranges=[[1.0] * 3, [1.0] * 3], actions=[]) for i in range(160)]
    head = dict(seed=550000, stakes=[50, 100, 20000], deals=300)
    (d3old / "m1-spots.json").write_text(json.dumps(dict(head, spots=spots)))
    (out / "m1-spots.json").write_text((d3old / "m1-spots.json").read_text())
    (d2bold / "m1-spots.json").write_text(json.dumps(dict(head, spots=spots[:40])))
    (d3old / "d2b-carry.json").write_text(json.dumps(dict(engine="E")))
    arrays1 = {f"{i}/": rng.random((2, 3)) for i in range(160)} | {"0/1": rng.random((2, 3))}
    np.savez(d3old / "m1-x-d2b1.npz", **arrays1)
    (d3old / "m1-x-d2b1.json").write_text(json.dumps(dict(cand="d2b1", net="n1.pt", net_sha256=_sha(repo / "n1.pt"), **X_SETTINGS)))
    v1 = rng.normal(0, 0.01, 160)
    rows1 = [dict(ROW, spot=i, value=float(v1[i])) for i in range(160)]
    (d3old / "m1-v-d2b1.json").write_text(json.dumps(dict(
        cand="d2b1", menus=MENUS, rule=RULE, precision="f32", rows=rows1,
        inputs=dict(spots=_sha(d3old / "m1-spots.json"), x=_sha(d3old / "m1-x-d2b1.npz"), engine="E", threads=12))))
    (d2bold / "m1-v-star.json").write_text(json.dumps(dict(
        cand="star", rule=RULE, inputs=dict(spots=_sha(d2bold / "m1-spots.json"), x=None),
        rows=[dict(ROW, spot=i, value=float(v1[i]) + 0.02) for i in range(40)])))
    assert m1.cmd_d3_init(types.SimpleNamespace(out=str(out))) == 0
    mode = "resolve" if resolve else "carry"
    if resolve:
        (out / "d3-remake-failed.json").write_text(json.dumps(dict(spot=0, engine="E")))
        np.savez(out / "m1-x-d3n1.npz", **{f"{i}/": rng.random((2, 3)) for i in range(160)})
        spots1 = [dict(spot=i, nodes=1) for i in range(160)]
    else:
        np.savez(out / "m1-x-d3n1.npz", **arrays1)
        spots1 = [dict(spot=i, remade_equal=True) if i < 2 else dict(spot=i, carried="m1-x-d2b1.npz") for i in range(160)]
    np.savez(out / "m1-x-d3n2.npz", **{f"{i}/": rng.random((2, 3)) for i in range(160)})
    for c, net, sp in (("d3n1", "n1.pt", spots1), ("d3n2", "n2.pt", [dict(spot=i, nodes=1) for i in range(160)])):
        (out / f"m1-x-{c}.json").write_text(json.dumps(dict(cand=c, net=net, net_sha256=_sha(repo / net), solves=1, mode=mode,
                                                           spots=sp, **X_SETTINGS)))
    h1 = _head3(out, "d3n1", mode)
    rows_n1 = [dict(ROW, spot=i, value=float(v1[i])) for i in range(160)] if resolve else m1.d3_carried_rows(
        out, np.load(out / "m1-x-d3n1.npz"), h1)
    (out / "m1-v-d3n1.json").write_text(json.dumps(dict(h1, rows=rows_n1)))
    v2 = v1 - gap + rng.normal(0, 0.0005, 160)
    (out / "m1-v-d3n2.json").write_text(json.dumps(dict(_head3(out, "d3n2"), rows=[dict(ROW, spot=i, value=float(v2[i])) for i in range(solved)])))
    return m1, out, rec3


def test_d3_gate_b_report_carry_mode(tmp_path, monkeypatch):
    m1, out, rec = _gate_b3(tmp_path, monkeypatch)
    assert m1.cmd_report_d3(types.SimpleNamespace(out=str(out))) == 0
    got = json.loads((rec / "m1-report.json").read_text())
    d = np.array([got["per_spot"][str(i)] for i in range(160)])
    n1 = {r["spot"]: r["value"] for r in json.loads((out / "m1-v-d3n1.json").read_text())["rows"]}
    n2 = {r["spot"]: r["value"] for r in json.loads((out / "m1-v-d3n2.json").read_text())["rows"]}
    assert d == pytest.approx([n1[i] - n2[i] for i in range(160)])  # D = V(N1) - V(N2) = Loss(X_N2) - Loss(X_N1)
    assert got["final"] and got["spots"] == 160 and got["z"] == 1.96 and got["margin"] == 0.0025 and got["mode"] == "carry"
    assert got["mean"] == pytest.approx(d.mean()) and got["ci"][1] == pytest.approx(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(160))
    assert got["non_inferior"] is bool(got["ci"][1] + m1.ORACLE_ERROR <= 0.0025)
    assert got["old40"]["spots"] == 40 and got["new120"]["spots"] == 120
    assert len(got["largest"]) == 5 and got["largest"][0]["d"] == max(d) and got["largest"][0]["board"] == "rainbow"
    assert got["loss_old40"]["d3n1"]["spots"] == 40 and got["loss_old40"]["d3n1"]["mean"] == pytest.approx(0.02)
    assert got["hours"]["d3n1"] == 0 and got["hours"]["d3n2"] == pytest.approx(160 * 10 / 3600)  # N1's rows are carried
    assert got["sha256"] == dict(N1=_sha(tmp_path / "repo" / "n1.pt"), N2=_sha(tmp_path / "repo" / "n2.pt")) and got["n2"] == "d3-w2048-wd0.1"


def test_d3_gate_b_orientation_a_worse_n2_makes_d_positive_and_its_largest_spot_first(tmp_path, monkeypatch):
    m1, out, rec = _gate_b3(tmp_path, monkeypatch)
    _edit(out / "m1-v-d3n2.json", lambda f: f["rows"][100].update(value=f["rows"][100]["value"] - 0.05))  # N2 lost 5% pot at spot 100
    m1.cmd_report_d3(types.SimpleNamespace(out=str(out)))
    got = json.loads((rec / "m1-report.json").read_text())
    assert got["per_spot"]["100"] > 0.04 and got["largest"][0]["spot"] == 100 and got["largest"][0]["old"] is False


def test_d3_gate_b_a_worse_net_fails_and_an_interim_look_has_no_verdict(tmp_path, monkeypatch):
    m1, out, rec = _gate_b3(tmp_path / "worse", monkeypatch, gap=0.01)
    assert m1.cmd_report_d3(types.SimpleNamespace(out=str(out))) == 1
    assert json.loads((rec / "m1-report.json").read_text())["non_inferior"] is False
    m1, out, rec = _gate_b3(tmp_path / "better", monkeypatch, gap=-0.01)
    assert m1.cmd_report_d3(types.SimpleNamespace(out=str(out))) == 0
    got = json.loads((rec / "m1-report.json").read_text())
    assert got["non_inferior"] is True and got["mean"] < 0
    m1, out, rec = _gate_b3(tmp_path / "interim", monkeypatch, solved=150)
    assert m1.cmd_report_d3(types.SimpleNamespace(out=str(out))) == 1
    got = json.loads((rec / "m1-report.json").read_text())
    assert got["final"] is False and got["non_inferior"] is None and got["spots"] == 150
    (out / "m1-v-d3n2.json").unlink()
    assert m1.cmd_report_d3(types.SimpleNamespace(out=str(out))) == 2  # not ready: no verdict


def test_d3_gate_b_resolve_mode_reports_with_every_spot_solved(tmp_path, monkeypatch):
    m1, out, rec = _gate_b3(tmp_path, monkeypatch, resolve=True)
    assert m1.cmd_report_d3(types.SimpleNamespace(out=str(out))) == 0
    got = json.loads((rec / "m1-report.json").read_text())
    assert got["mode"] == "resolve" and got["hours"]["d3n1"] == pytest.approx(160 * 10 / 3600) and got["final"]


def test_a_carried_x_that_differs_is_refused(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    x = dict(np.load(out / "m1-x-d3n1.npz"))
    x["3/"] = x["3/"] + 1
    np.savez(out / "m1-x-d3n1.npz", **x)
    with pytest.raises(SystemExit):
        m1.d3_carried_rows(out, np.load(out / "m1-x-d3n1.npz"), _head3(out, "d3n1"))


def test_the_registered_n1_sha_is_d2bs_selection_n1_in_flop_m1():
    m1 = _script("flop_m1")
    sel = json.loads((ROOT / "records" / "d2b" / "selection.json").read_text())
    assert m1.N1_SHA256 == sel["sha256"][sel["n1"]] == _script("flop_sym_speed").N1_SHA256
    assert (m1.D3_SPOTS, m1.D3_REMADE) == (160, 2) and m1.D3_OLD.name == "flop-eval-d2b" and m1.D3_RECORDS.name == "d3"


def test_the_command_line_takes_d3_init_report_d3_and_resolve_all(monkeypatch):
    import sys

    m1 = _script("flop_m1")
    seen = []
    for name in ("cmd_d3_init", "cmd_report_d3", "cmd_strategy", "cmd_solve"):
        monkeypatch.setattr(m1, name, lambda a, name=name: seen.append((name, a)) or 0)
    for argv in (["d3-init"], ["report-d3"], ["strategy", "--cand", "d3n1", "--net", "n", "--resolve-all"],
                 ["strategy", "--cand", "d3n2", "--net", "n"], ["solve", "--cand", "d3n1"], ["solve", "--cand", "d3n2"]):
        monkeypatch.setattr(sys, "argv", ["flop_m1.py", "--out", "o"] + argv)
        assert m1.main() == 0
    assert [n for n, _ in seen] == ["cmd_d3_init", "cmd_report_d3", "cmd_strategy", "cmd_strategy", "cmd_solve", "cmd_solve"]
    assert seen[2][1].resolve_all is True and seen[3][1].resolve_all is False


@pytest.mark.parametrize("bad, match", [
    (lambda t, o: _edit(o / "m1-v-d3n1.json", lambda f: f["rows"][3].update(value=9.0)), "no longer"),
    (lambda t, o: _edit(o / "m1-v-d3n1.json", lambda f: f["rows"][3].pop("carried")), "carried"),
    (lambda t, o: _edit(o / "m1-v-d3n2.json", lambda f: f["rows"][3].update(carried="m1-v-d2b1")), "solved"),
    (lambda t, o: _edit(o / "m1-v-d3n2.json", lambda f: f["inputs"].update(engine="F")), "engine"),
    (lambda t, o: _edit(o / "m1-v-d3n1.json", lambda f: f["inputs"].update(engine="F")), "engine"),
    (lambda t, o: _edit(o / "m1-v-d3n2.json", lambda f: f.update(rule=[1])), "paired"),
    (lambda t, o: _edit(o / "m1-v-d3n1.json", lambda f: f["inputs"].update(mode="resolve")), "mode"),
    (lambda t, o: _edit(o / "m1-x-d3n2.json", lambda f: f.update(mode="resolve")), "mode"),
    (lambda t, o: _edit(o / "m1-x-d3n1.json", lambda f: f["spots"][0].update(remade_equal=False)), "equal"),
    (lambda t, o: _edit(o / "m1-x-d3n1.json", lambda f: f["spots"][5].pop("carried")), "equal"),
    (lambda t, o: (t / "d3old" / "m1-v-d2b1.json").write_text("{}"), "d3-init"),
    (lambda t, o: (t / "repo" / "n2.pt").write_bytes(b"other"), "registered net"),
    (lambda t, o: (t / "repo" / "n1.pt").write_bytes(b"other"), "registered net"),
    (lambda t, o: _edit(t / "rec3" / "gate-a.json", lambda f: f.update(proceed=False)), "Gate A"),
    (lambda t, o: _edit(t / "d2bold" / "m1-v-star.json", lambda f: f["inputs"].update(spots="other")), "m1-v-star"),
    (lambda t, o: _edit(o / "m1-spots.json", lambda f: f["spots"][100].update(pot=1)), "d3-init"),
    (lambda t, o: _edit(o / "m1-x-d3n2.json", lambda f: f.update(seeds=[0, 1])), "seeds"),
    (lambda t, o: _edit(o / "m1-x-d3n1.json", lambda f: f.update(flop_iters=400)), "seeds"),
    (lambda t, o: (o / "d3-remake-failed.json").write_text(json.dumps(dict(spot=0, engine="E"))), "mode"),
])
def test_d3_gate_b_report_refuses(tmp_path, monkeypatch, bad, match):
    m1, out, rec = _gate_b3(tmp_path, monkeypatch)
    bad(tmp_path, out)
    with pytest.raises(SystemExit, match=match):
        m1.cmd_report_d3(types.SimpleNamespace(out=str(out)))
    assert not (rec / "m1-report.json").exists()


@pytest.mark.parametrize("bad", [
    lambda t, o: _edit(o / "m1-v-d3n1.json", lambda f: f["rows"][3].update(carried="m1-v-d2b1")),
    lambda t, o: _edit(o / "m1-x-d3n1.json", lambda f: f["spots"][3].update(carried="m1-x-d2b1.npz")),
    lambda t, o: _edit(o / "m1-x-d3n1.json", lambda f: f["spots"][1].update(remade_equal=True)),
    lambda t, o: _edit(o / "m1-v-d3n1.json", lambda f: f["inputs"].update(failed="other")),
    lambda t, o: (o / "d3-remake-failed.json").write_text(json.dumps(dict(spot=1, engine="E"))),
    lambda t, o: (o / "d3-remake-failed.json").unlink(),
])
def test_d3_gate_b_report_refuses_in_resolve_mode(tmp_path, monkeypatch, bad):
    m1, out, rec = _gate_b3(tmp_path, monkeypatch, resolve=True)
    bad(tmp_path, out)
    with pytest.raises(SystemExit):
        m1.cmd_report_d3(types.SimpleNamespace(out=str(out)))
    assert not (rec / "m1-report.json").exists()


def test_the_init_refuses_other_spots_engine_or_x(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    a = types.SimpleNamespace(out=str(out))
    good = (out / "m1-spots.json").read_text()
    for change in (lambda f: f["spots"][159].update(pot=1), lambda f: f["spots"].pop(), lambda f: f.update(seed=1)):
        _edit(out / "m1-spots.json", change)
        with pytest.raises(SystemExit, match="registered 160 spots"):
            m1.cmd_d3_init(a)
        (out / "m1-spots.json").write_text(good)
    assert m1.cmd_d3_init(a) == 0
    carry = json.loads((out / "d3-carry.json").read_text())
    assert carry["engine"] == "E" and set(carry["files"]) == {"m1-v-d2b1.json", "m1-x-d2b1.npz", "m1-x-d2b1.json"}
    old = tmp_path / "d3old"
    monkeypatch.setattr(m1, "engine_sha", lambda pe: "F")
    with pytest.raises(SystemExit, match="engine"):
        m1.cmd_d3_init(a)
    monkeypatch.setattr(m1, "engine_sha", lambda pe: "E")
    for name, change, match in (("m1-x-d2b1.json", lambda f: f.update(net_sha256="other"), "N1"),
                                ("m1-v-d2b1.json", lambda f: f["inputs"].update(x="other"), "solved from"),
                                ("m1-v-d2b1.json", lambda f: f["rows"].pop(), "160 solved"),
                                ("m1-v-d2b1.json", lambda f: f["rows"][5].update(carried="m1-v-sym"), "160 solved")):
        keep = (old / name).read_text()
        _edit(old / name, change)
        with pytest.raises(SystemExit, match=match):
            m1.cmd_d3_init(a)
        (old / name).write_text(keep)
    _edit(tmp_path / "rec2" / "gate-a.json", lambda f: f.update(proceed=False))
    with pytest.raises(SystemExit, match="Gate A"):
        m1.cmd_d3_init(a)


def test_a_strategy_needs_gate_a_and_the_right_net(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    n1, n2 = str(tmp_path / "repo" / "n1.pt"), str(tmp_path / "repo" / "n2.pt")
    m1.d3_strategy_guard("d3n1", n1, out, False)
    m1.d3_strategy_guard("d3n2", n2, out, False)
    with pytest.raises(SystemExit):
        m1.d3_strategy_guard("d3n1", n2, out, False)
    with pytest.raises(SystemExit):
        m1.d3_strategy_guard("d3n2", n1, out, False)  # N1 is not the net Gate A passed for N2
    with pytest.raises(SystemExit):
        m1.d3_strategy_guard("d3n2", None, out, False)
    with pytest.raises(SystemExit, match="none did"):
        m1.d3_strategy_guard("d3n1", n1, out, True)
    with pytest.raises(SystemExit, match="X_N1's alone"):
        m1.d3_strategy_guard("d3n2", n2, out, True)
    _edit(tmp_path / "rec3" / "gate-a.json", lambda f: f.update(proceed=False))
    for c, net in (("d3n1", n1), ("d3n2", n2)):
        with pytest.raises(SystemExit, match="Gate A"):
            m1.d3_strategy_guard(c, net, out, False)
    _edit(tmp_path / "rec3" / "gate-a.json", lambda f: f.update(proceed=True))
    (out / "d3-carry.json").unlink()
    with pytest.raises(SystemExit, match="d3-init"):
        m1.d3_strategy_guard("d3n1", n1, out, False)
    assert not set(m1.D3) & set(m1.CANDIDATES) and not set(m1.D3) & set(m1.D2B)
    assert m1.LEAVES["d3n1"] == m1.LEAVES["d3n2"] == "sym" and m1.N1_SHA256 != m1.N0_SHA256


def test_the_solve_of_x_d3n1_starts_from_the_carried_rows(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    (out / "m1-v-d3n1.json").unlink()
    before = _tree(tmp_path / "d3old")
    monkeypatch.setattr(m1, "below_normal", lambda: None)
    monkeypatch.setattr(m1, "frozen_oracle", lambda *a: pytest.fail("solved a carried spot"))
    a = types.SimpleNamespace(out=str(out), cand="d3n1", threads=12, spots=None)
    assert m1.cmd_solve(a) == 0  # all 160 carried: nothing is solved
    got = json.loads((out / "m1-v-d3n1.json").read_text())
    assert got["inputs"]["engine"] == "E" and got["inputs"]["mode"] == "carry" and "failed" not in got["inputs"]
    assert got["inputs"]["carry"] == _sha(out / "d3-carry.json")
    old = json.loads((tmp_path / "d3old" / "m1-v-d2b1.json").read_text())["rows"]
    assert got["rows"] == [dict(r, carried="m1-v-d2b1") for r in old]
    assert m1.cmd_solve(a) == 0  # a resume reads them back
    monkeypatch.setattr(m1, "engine_sha", lambda pe: "F")
    with pytest.raises(SystemExit, match="not the one X_N1's carried rows were solved under"):
        m1.cmd_solve(a)
    assert _tree(tmp_path / "d3old") == before  # runs/flop-eval-d2b is only read


def test_the_solve_of_x_d3n2_solves_from_spot_0_under_the_carried_rows_build(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    (out / "m1-v-d3n2.json").unlink()

    class Reached(Exception):
        pass

    def stop(pe, spot, threads, fixed, iso):
        raise Reached(spot["pot"])

    monkeypatch.setattr(m1, "below_normal", lambda: None)
    monkeypatch.setattr(m1, "frozen_oracle", stop)
    a = types.SimpleNamespace(out=str(out), cand="d3n2", threads=12, spots=None)
    with pytest.raises(Reached) as e:
        m1.cmd_solve(a)
    assert str(e.value) == "200"
    monkeypatch.setattr(m1, "engine_sha", lambda pe: "F")
    with pytest.raises(SystemExit, match="not the one X_N1's carried rows were solved under"):
        m1.cmd_solve(a)


@pytest.mark.parametrize("cand", ["d3n1", "d3n2"])
@pytest.mark.parametrize("changed", ["d3old/m1-spots.json", "eval/m1-spots.json", "d3old/m1-v-d2b1.json", "d3old/m1-x-d2b1.npz"])
def test_each_d3_solve_re_hashes_the_files_the_spots_rest_on(tmp_path, monkeypatch, cand, changed):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    (out / f"m1-v-{cand}.json").unlink()
    monkeypatch.setattr(m1, "below_normal", lambda: None)
    monkeypatch.setattr(m1, "frozen_oracle", lambda *a: pytest.fail("solved"))
    path = tmp_path / changed
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="d3-init"):
        m1.cmd_solve(types.SimpleNamespace(out=str(out), cand=cand, threads=12, spots=None))


def _strategy3(tmp_path, monkeypatch, wrong_at=None, resolve=False):
    """cmd_strategy's path with the search faked: each spot's nodes and what the "agent" plays there (X_N1 as
    m1-x-d2b1.npz holds it, except `wrong_at`, for d3n1; anything else for d3n2)."""
    m1, out, _ = _gate_b3(tmp_path, monkeypatch, resolve=resolve)
    for c in ("d3n1", "d3n2"):
        for ext in ("npz", "json"):
            (out / f"m1-x-{c}.{ext}").unlink()
    old_x = np.load(tmp_path / "d3old" / "m1-x-d2b1.npz")
    built, now = [], {}

    def oracle(pe, spot):
        now["spot"] = spot["pot"] - 200
        built.append(now["spot"])
        nodes = [((), spot["hero"], [0, 1])] + ([((1,), spot["hero"], [0, 1])] if now["spot"] == 0 else [])
        return types.SimpleNamespace(holdings=[1, 2, 3], flop_nodes=lambda: nodes)

    def agent_for(cand):
        def options(hero, board, actions, sb, bb, stacks, holdings, seed=0):
            arr = old_x[f"{now['spot']}/{','.join(map(str, actions))}"] if cand == "d3n1" else np.full((2, 3), 0.5)
            return list((arr + 1.0 if now["spot"] == wrong_at else arr).T)
        return types.SimpleNamespace(options=options)

    search = types.SimpleNamespace(errors=[], jobs=[], solves=1, solver=types.SimpleNamespace(symmetric=0, drift=0.0))
    for k, v in dict(oracle=oracle, make_agent=lambda pe, cand, net: (agent_for(cand), search),
                     to_oracle=lambda r, acts: (r, 0.0), history=lambda nodes, path: list(path)).items():
        monkeypatch.setattr(m1, k, v)
    return m1, out, built


def _a3(out, cand, net, seeds=None, resolve_all=False):
    return types.SimpleNamespace(out=str(out), cand=cand, net=str(net), seeds=seeds, resolve_all=resolve_all)


def test_x_d3n1_carries_all_160_spots_and_remakes_2(tmp_path, monkeypatch):
    m1, out, built = _strategy3(tmp_path, monkeypatch)
    before = _tree(tmp_path / "d3old")
    assert m1.cmd_strategy(_a3(out, "d3n1", tmp_path / "repo" / "n1.pt")) == 0
    assert built == [0, 1]  # spots 2-159 are copied, 0-1 made again
    x, old = np.load(out / "m1-x-d3n1.npz"), np.load(tmp_path / "d3old" / "m1-x-d2b1.npz")
    assert sorted(x.files) == sorted(old.files) and all(m1.same_x(x, old, i) for i in range(160))
    meta = json.loads((out / "m1-x-d3n1.json").read_text())
    assert meta["net_sha256"] == m1.N1_SHA256 and meta["mode"] == "carry" and [r["spot"] for r in meta["spots"]] == list(range(160))
    assert meta["spots"][0]["remade_equal"] and meta["spots"][1]["remade_equal"]
    assert meta["spots"][2] == dict(spot=2, carried="m1-x-d2b1.npz")
    assert not (out / "d3-remake-failed.json").exists() and _tree(tmp_path / "d3old") == before


def test_x_d3n2_makes_every_spot_and_neither_takes_seeds(tmp_path, monkeypatch):
    m1, out, built = _strategy3(tmp_path, monkeypatch)
    n1, n2 = tmp_path / "repo" / "n1.pt", tmp_path / "repo" / "n2.pt"
    for cand, net in (("d3n1", n1), ("d3n2", n2)):
        with pytest.raises(SystemExit, match="no --seeds"):
            m1.cmd_strategy(_a3(out, cand, net, seeds="0,1"))
    assert m1.cmd_strategy(_a3(out, "d3n1", n1)) == 0  # X_N1's re-made spots are decided first
    built.clear()
    assert m1.cmd_strategy(_a3(out, "d3n2", n2)) == 0
    assert built == list(range(160))
    meta = json.loads((out / "m1-x-d3n2.json").read_text())
    assert not any("carried" in r or "remade_equal" in r for r in meta["spots"]) and meta["net_sha256"] == _sha(n2)


@pytest.mark.parametrize("wrong", [0, 1])
def test_x_d3n1_saves_nothing_records_the_failure_and_is_remade_in_full_only_with_resolve_all(tmp_path, monkeypatch, wrong):
    m1, out, built = _strategy3(tmp_path, monkeypatch, wrong_at=wrong)
    n1 = tmp_path / "repo" / "n1.pt"
    with pytest.raises(SystemExit, match="differs from m1-x-d2b1"):
        m1.cmd_strategy(_a3(out, "d3n1", n1))
    assert built[-1] == wrong and not (out / "m1-x-d3n1.npz").exists() and not (out / "m1-x-d3n1.json").exists()
    assert json.loads((out / "d3-remake-failed.json").read_text()) == dict(spot=wrong, engine="E")
    built.clear()
    assert m1.cmd_strategy(_a3(out, "d3n1", n1, resolve_all=True)) == 0
    assert built == list(range(160))  # all of X_N1, none carried
    meta = json.loads((out / "m1-x-d3n1.json").read_text())
    assert meta["mode"] == "resolve" and not any("carried" in r or "remade_equal" in r for r in meta["spots"])


def test_resolve_all_is_refused_without_the_record_and_outside_x_d3n1(tmp_path, monkeypatch):
    m1, out, built = _strategy3(tmp_path, monkeypatch)
    n1, n2 = tmp_path / "repo" / "n1.pt", tmp_path / "repo" / "n2.pt"
    with pytest.raises(SystemExit, match="none did"):
        m1.cmd_strategy(_a3(out, "d3n1", n1, resolve_all=True))
    with pytest.raises(SystemExit, match="X_N1's alone"):
        m1.cmd_strategy(_a3(out, "d3n2", n2, resolve_all=True))
    with pytest.raises(SystemExit, match="X_d3n1's alone"):
        m1.cmd_strategy(_a3(out, "bp", n1, resolve_all=True))
    assert built == []


def test_x_d3n2_waits_for_x_d3n1s_remake_to_be_decided(tmp_path, monkeypatch):
    m1, out, built = _strategy3(tmp_path, monkeypatch)
    n1, n2 = tmp_path / "repo" / "n1.pt", tmp_path / "repo" / "n2.pt"
    with pytest.raises(SystemExit, match="not decided yet"):
        m1.cmd_strategy(_a3(out, "d3n2", n2))
    assert built == [] and not (out / "m1-x-d3n2.npz").exists()
    assert m1.cmd_strategy(_a3(out, "d3n1", n1)) == 0
    assert m1.cmd_strategy(_a3(out, "d3n2", n2)) == 0 and json.loads((out / "m1-x-d3n2.json").read_text())["mode"] == "carry"


def test_a_failure_record_must_name_a_remade_spot_and_the_carried_rows_build(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    rec = out / "d3-remake-failed.json"
    assert m1.d3_mode(out) == "carry"
    for bad in ("{}", "[]", "not json", json.dumps(dict(spot=2, engine="E")), json.dumps(dict(spot=0, engine="F")),
                json.dumps(dict(spot=1)), json.dumps(dict(spot=True, engine="E")), json.dumps(dict(spot="0", engine="E"))):
        rec.write_text(bad)
        with pytest.raises(SystemExit):
            m1.d3_mode(out)
    for spot in (0, 1):
        rec.write_text(json.dumps(dict(spot=spot, engine="E")))
        assert m1.d3_mode(out) == "resolve"
    (out / "d3-carry.json").unlink()
    with pytest.raises(SystemExit, match="d3-init"):
        m1.d3_mode(out)


def test_a_stale_failure_record_is_not_a_way_into_resolve_mode(tmp_path, monkeypatch):
    m1, out, built = _strategy3(tmp_path, monkeypatch, wrong_at=0)
    n1 = tmp_path / "repo" / "n1.pt"
    monkeypatch.setattr(m1, "engine_sha", lambda pe: "F")  # rebuilt after d3-init
    with pytest.raises(SystemExit, match="not the one X_N1's carried rows were solved under"):
        m1.cmd_strategy(_a3(out, "d3n1", n1))
    assert built == [] and not (out / "d3-remake-failed.json").exists()  # nothing made, so no failure recorded
    (out / "d3-remake-failed.json").write_text(json.dumps(dict(spot=0, engine="F")))  # or one written by hand
    with pytest.raises(SystemExit, match="under the carried rows' build"):
        m1.cmd_strategy(_a3(out, "d3n1", n1, resolve_all=True))
    assert built == []


def test_d3_n1_without_resolve_all_is_refused_after_a_failure(tmp_path, monkeypatch):
    m1, out, built = _strategy3(tmp_path, monkeypatch, wrong_at=0)
    n1 = tmp_path / "repo" / "n1.pt"
    with pytest.raises(SystemExit, match="differs from m1-x-d2b1"):
        m1.cmd_strategy(_a3(out, "d3n1", n1))
    built.clear()
    with pytest.raises(SystemExit, match="--resolve-all"):
        m1.cmd_strategy(_a3(out, "d3n1", n1))
    assert built == []


def test_d3_never_writes_runs_flop_eval_d2b(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch)
    old = tmp_path / "d3old"
    before = _tree(old)
    a = types.SimpleNamespace(out=str(old))
    for call in (lambda: m1.cmd_d3_init(a), lambda: m1.d3_carry(old), lambda: m1.cmd_report_d3(a),
                 lambda: m1.d3_strategy_guard("d3n1", str(tmp_path / "repo" / "n1.pt"), old, False)):
        with pytest.raises(SystemExit, match="runs/flop-eval-d2b itself"):
            call()
    assert _tree(old) == before


def test_the_solve_of_x_d3n1_in_resolve_mode_writes_the_failure_into_its_header(tmp_path, monkeypatch):
    m1, out, _ = _gate_b3(tmp_path, monkeypatch, resolve=True)
    (out / "m1-v-d3n1.json").unlink()

    class Reached(Exception):
        pass

    def stop(pe, spot, threads, fixed, iso):
        raise Reached(spot["pot"])

    monkeypatch.setattr(m1, "below_normal", lambda: None)
    monkeypatch.setattr(m1, "frozen_oracle", stop)
    monkeypatch.setattr(m1, "engine_sha", lambda pe: "E")
    with pytest.raises(Reached) as e:
        m1.cmd_solve(types.SimpleNamespace(out=str(out), cand="d3n1", threads=12, spots=None))
    assert str(e.value) == "200"  # spot 0: nothing carried
