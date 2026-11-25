"""FLOPSEARCH.md D4 M1 tooling: the `FlopOracle` binding and `scripts/flop_m1.py`'s
translation helpers. The Rust tests check the oracle's solve itself."""
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

pe = pytest.importorskip("parciver_engine")
ROOT = Path(__file__).resolve().parents[1]
BOARD = [0, 13, 26]
# Stacks of one pot behind: every bet is all-in, so the tree is small.
MENUS = [([0.75], [], 1)] * 3


def _m1():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("flop_m1_cli", ROOT / "scripts" / "flop_m1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _oracle(precision="f64", **kw):
    ranges = ([1.0] * 1176, [0.5 + (i % 7) / 7 for i in range(1176)])
    return pe.FlopOracle(BOARD, (200, 200), (400, 400), 1, 100, MENUS, ranges, threads=2, precision=precision, **kw)


def test_oracle_binding_solves_and_brackets_in_both_precisions():
    o, h = _oracle(), _oracle("f32")
    assert (o.precision, h.precision) == ("f64", "f32")
    assert len(o.holdings) == 1176 and not any(c in BOARD for pair in o.holdings for c in pair)
    nodes = o.flop_nodes()
    assert nodes[0][0] == [] and nodes[0][1] == 1  # the root, the first actor's
    assert o.size == h.size and o.iterations == 0
    for x in (o, h):
        x.run(10, 5.0)
        x.run(10, 5.0)
    assert o.iterations == 20
    b, c = o.best_responses(), h.best_responses()
    assert b[0] + b[1] >= -1e-9
    assert -b[1] - 1e-9 <= o.value(0) <= b[0] + 1e-9
    assert all(abs(x - y) < 1e-6 * 400 for x, y in zip(b, c)), (b, c)


def test_oracle_binding_freezes_and_validates():
    o = _oracle()
    path, _, acts = o.flop_nodes()[0]
    k = len(acts)
    fixed = [1.0 if a == k - 1 else 0.0 for a in range(k) for _ in range(1176)]  # always the last action
    o.freeze(path, fixed)
    o.run(5)
    assert o.average(path) == fixed
    with pytest.raises(ValueError):
        o.freeze(path, [0.5] * (k * 1176 - 1))
    with pytest.raises(ValueError):
        o.run(1, -1.0)
    with pytest.raises(ValueError):
        o.value(2)
    with pytest.raises(ValueError):
        _oracle("f16")
    with pytest.raises(ValueError):
        pe.FlopOracle([0, 0, 1], (200, 200), (400, 400), 1, 100, MENUS, ([1.0] * 1176, [1.0] * 1176))


@pytest.mark.skipif(sys.platform != "win32", reason="reads the process's commit through the Windows API")
def test_dropping_the_oracle_returns_its_store():
    # flop_m1 waits for free commit before each spot; on 29 Sep it stalled on the
    # last spot's store, freed but still committed (mimalloc purges on a later
    # call only). `run` allocates the store on the pool's threads, `freeze` on
    # the caller's: both must come back.
    import ctypes
    import ctypes.wintypes as w

    class Counters(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
            (f, ctypes.c_size_t) for f in ("peak_ws", "ws", "a", "b", "c", "d", "private", "peak_private")]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = w.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]

    def private():
        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.private

    ranges = ([1.0] * 1176, [1.0] * 1176)
    for allocate in ("run", "freeze"):
        o = pe.FlopOracle(BOARD, (200, 200), (200, 200), 1, 100, MENUS, ranges, threads=2)  # a 0.3 GiB store
        path, _, acts = o.flop_nodes()[0]
        store = 16 * o.size[2] * 1176
        before = private()
        if allocate == "run":
            o.run(1)
        else:
            o.freeze(path, [1.0 / len(acts)] * (len(acts) * 1176))
        assert private() - before > 0.8 * store, allocate
        del o
        assert private() - before < 0.2 * store, allocate


def test_to_oracle_maps_exact_sizes_and_the_nearest_in_log_size():
    m = _m1()
    acts = [("f", 0), ("c", 0), ("b", 150), ("b", 300), ("b", 1000)]
    opts = [("fold", 0, 0.2), ("call", 0, 0.3), ("bet", 300, 0.1), ("bet", 200, 0.4)]
    row, moved = m.to_oracle(opts, acts)
    assert row == pytest.approx([0.2, 0.3, 0.4, 0.1, 0.0]) and moved == pytest.approx(0.4)
    # Equal distance in log size: the smaller bet.
    row, moved = m.to_oracle([("bet", 200, 1.0)], [("c", 0), ("b", 100), ("b", 400)])
    assert row == [0.0, 1.0, 0.0] and moved == 1.0
    assert math.isclose(abs(math.log(200 / 100)), abs(math.log(200 / 400)))


def test_history_writes_checks_and_calls_as_the_agent_takes_them():
    m = _m1()
    nodes = {
        (): (1, [("c", 0), ("b", 100)]),
        (0,): (0, [("c", 0), ("b", 100)]),
        (1,): (0, [("f", 0), ("c", 0), ("b", 300)]),
        (0, 1): (1, [("f", 0), ("c", 0), ("b", 300)]),
    }
    assert m.history(nodes, (0, 0)) == [("x", 0), ("x", 0)]
    assert m.history(nodes, (1, 1)) == [("b", 100), ("c", 0)]
    assert m.history(nodes, (0, 1, 2)) == [("x", 0), ("b", 100), ("b", 300)]
    with pytest.raises(ValueError):
        m.history(nodes, (1, 0))
    # Every path of a real oracle converts.
    o = _oracle()
    real = {tuple(p): (player, acts) for p, player, acts in o.flop_nodes()}
    for p in real:
        assert len(m.history(real, p)) == len(p)


def test_oracle_isomorphism_solves_the_same_game_and_falls_back(monkeypatch):
    # A two-tone flop (ace and king of suit 0, a seven of suit 1): suits 2 and 3 alike.
    board = [48, 44, 21]
    even = ([1.0] * 1176, [1.0] * 1176)
    args = (board, (200, 200), (200, 200), 1, 100, MENUS)  # every bet all-in: a small tree
    o = pe.FlopOracle(*args, even, threads=2)
    h = pe.FlopOracle(*args, even, threads=2, isomorphism=True)
    assert (o.isomorphism, h.isomorphism) == (False, True) and h.size[2] < o.size[2]
    # The same iterates summed in another order: equal up to rounding, which
    # regret matching amplifies over many iterations, so compare early.
    for x in (o, h):
        x.run(5, 5.0)
    assert all(abs(a - b) < 1e-9 * 400 for a, b in zip(o.best_responses(), h.best_responses()))
    # A range the swap changes is refused; flop_m1 then solves the full game.
    i = h.holdings.index((2, 4))  # suit 2 to 3 moves it to (3, 4)
    skew = (even[0], [0.5 if j == i else 1.0 for j in range(1176)])
    with pytest.raises(ValueError):
        pe.FlopOracle(*args, skew, threads=2, isomorphism=True)
    m = _m1()
    monkeypatch.setattr(m, "wait_for_memory", lambda o: None)
    spot = dict(pot=400, board=board, total=[400, 400], first=1, ranges=list(even))
    assert m.frozen_oracle(pe, spot, 2, [], True).isomorphism
    assert not m.frozen_oracle(pe, dict(spot, ranges=list(skew)), 2, [], True).isomorphism
    # So is a frozen strategy the swap changes.
    k = len(m.oracle(pe, spot, 2).flop_nodes()[0][2])
    lop = [1.0 if a == 0 else 0.0 for a in range(k) for _ in range(1176)]
    lop[i], lop[1176 + i] = 0.0, 1.0
    assert not m.frozen_oracle(pe, spot, 2, [([], lop)], True).isomorphism
    assert m.frozen_oracle(pe, spot, 2, [([], [1.0 if a == 0 else 0.0 for a in range(k) for _ in range(1176)])], True).isomorphism


def test_solve_writes_rows_whole_and_resumes_only_under_the_same_settings(tmp_path, monkeypatch):
    m = _m1()
    for name, v in (("wait_for_memory", lambda o: None), ("below_normal", lambda: None), ("CHECK_EVERY", 5), ("ITERS", 5)):
        monkeypatch.setattr(m, name, v)
    spot = dict(pot=400, board=[48, 44, 21], total=[400, 400], first=1, hero=1, ranges=[[1.0] * 1176] * 2)
    (tmp_path / "m1-spots.json").write_text(json.dumps(dict(spots=[spot, spot])))
    args = types.SimpleNamespace(out=str(tmp_path), cand="star", threads=2, spots={0})
    assert m.cmd_solve(args) == 0
    path = tmp_path / "m1-v-star.json"
    first = json.loads(path.read_text())
    assert [r["spot"] for r in first["rows"]] == [0] and first["rule"] == [m.ORACLE_GAMMA, 5, m.TOLERANCE, 5]
    assert not list(tmp_path.glob("*.tmp"))
    args.spots = None
    assert m.cmd_solve(args) == 0  # spot 0 is kept, spot 1 added
    rows = json.loads(path.read_text())["rows"]
    assert [r["spot"] for r in rows] == [0, 1] and rows[0] == first["rows"][0]
    monkeypatch.setattr(m, "ITERS", 6)
    with pytest.raises(SystemExit):  # rows solved under another rule are not mixed
        m.cmd_solve(args)
    assert json.loads(path.read_text())["rows"] == rows
    monkeypatch.setattr(m, "ITERS", 5)
    # Nor rows of other spots (the spots file rewritten) ...
    assert first["inputs"] == dict(spots=m.sha256(tmp_path / "m1-spots.json"), x=None)
    spots = (tmp_path / "m1-spots.json").read_text()
    (tmp_path / "m1-spots.json").write_text(spots + "\n")
    with pytest.raises(SystemExit, match="inputs"):
        m.cmd_solve(args)
    (tmp_path / "m1-spots.json").write_text(spots)
    # ... but a file from before the inputs were recorded resumes under its own header.
    old = {k: v for k, v in json.loads(path.read_text()).items() if k != "inputs"}
    old["rows"] = old["rows"][:1]
    path.write_text(json.dumps(old))
    assert m.cmd_solve(args) == 0
    again = json.loads(path.read_text())
    assert "inputs" not in again and [r["spot"] for r in again["rows"]] == [0, 1]


def test_report_pairs_only_rows_of_the_same_game(tmp_path):
    m = _m1()
    (tmp_path / "m1-spots.json").write_text(json.dumps(dict(spots=[{}] * 3)))  # the report reads the spot count
    (tmp_path / "m1-x-net.npz").write_bytes(b"X")
    head = dict(menus=[1], rule=[5.0, 100, 0.2, 1000], precision="f32")
    files = dict(bp=dict(head, cand="bp"),  # an older file: no inputs
                 net=dict(head, cand="net", inputs=dict(spots="s", x=m.sha256(tmp_path / "m1-x-net.npz"))),
                 star=dict(head, cand="star", inputs=dict(spots="s", x=None)))
    rows = [dict(spot=i, value=0.01 * i, lo=0.01 * i - 1e-4, hi=0.01 * i + 1e-4) for i in range(3)]

    def report(**change):
        for c, h in files.items():
            (tmp_path / f"m1-v-{c}.json").write_text(json.dumps(dict(change.get(c, h), rows=rows)))
        return m.cmd_report(types.SimpleNamespace(out=str(tmp_path)))

    assert report() == 0
    for c, h, what in (("bp", dict(files["bp"], precision="f64"), "precision"),
                       ("star", dict(files["star"], rule=[2.0, 100, 0.2, 1000]), "rule"),
                       ("star", dict(files["star"], inputs=dict(spots="t", x=None)), "spots")):
        with pytest.raises(SystemExit, match=what):
            report(**{c: h})
    (tmp_path / "m1-x-net.npz").write_bytes(b"Y")  # X regenerated after its rows were solved
    with pytest.raises(SystemExit, match="m1-x-net.npz"):
        report()


def test_report_adds_the_oracle_error_to_the_upper_end(tmp_path):
    m = _m1()
    (tmp_path / "m1-spots.json").write_text(json.dumps(dict(spots=[{}] * 4)))

    def write(cand, values):
        rows = [dict(spot=i, value=v, lo=v - 0.0005, hi=v + 0.0005) for i, v in enumerate(values)]
        (tmp_path / f"m1-v-{cand}.json").write_text(json.dumps(dict(rows=rows)))

    net = [0.010, 0.012, 0.011, 0.013]
    # V(bp) - V(net) is about -gap at every spot, so the CI's upper end is just above -gap.
    for gap, want in ((0.0005, False), (0.0030, True)):
        write("net", net)
        write("bp", [v - gap + 1e-5 * (i % 2) for i, v in enumerate(net)])
        m.cmd_report(types.SimpleNamespace(out=str(tmp_path)))
        got = json.loads((tmp_path / "m1-report.json").read_text())["primary"]
        assert -gap < got["ci"][1] < 0 and got["oracle_error"] == m.ORACLE_ERROR == 0.001
        assert got["below_zero"] is want


def test_a_failed_search_is_kept():
    m = _m1()

    class Boom:
        solves = 3

        def __call__(self, job):
            raise RuntimeError("out of memory")

    s = m.Strict(Boom())
    with pytest.raises(RuntimeError):
        s({"board": [0, 1, 2, 3]})
    assert s.errors == ["RuntimeError('out of memory')"] and s.solves == 3
    # One slot of 2 actions, 2 holdings. An answer the engine refuses is kept too.
    job = {"board": [0, 1, 2, 3], "slot": [0, -1, -1], "child_start": [0, 2, 2, 2], "ranges": ([1.0, 1.0], [1.0, 1.0]), "want": [0]}
    ok = m.Strict(lambda job: [[0.5] * 4])
    assert ok(job) == [[0.5] * 4] and ok.errors == []
    nan = m.Strict(lambda job: [[0.5, 0.5, 0.5, float("nan")]])
    assert len(nan(job)[0]) == 4 and len(nan.errors) == 1 and "non-finite" in nan.errors[0]
