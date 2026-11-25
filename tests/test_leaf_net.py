# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""LEAF-NET (records/leaf-net/PREREG.md, "Tests before any data"): the canonical
suit frame (`value_net.ValueNet(..., canonical=True)`), the grid's training
changes and Gate A (`scripts/turn_net.py`), Gate B's harness
(`scripts/flop_m1.py`: symcheck, enginecheck, the leafnet report) and the
two-net speed mode (`scripts/flop_sym_speed.py`)."""
import hashlib
import importlib.util
import itertools
import json
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parciver import value_net as vn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = "c097586"  # the commit before LEAF-NET's code: "today's" value net and training
CUDA = torch.cuda.is_available()
N0 = next((p for p in (ROOT / "runs" / "turn-net" / "final-w2048-d5-e200" / "net.pt",
                       ROOT.parent / "parciver" / "runs" / "turn-net" / "final-w2048-d5-e200" / "net.pt") if p.exists()),
          None)


def card(rank: int, suit: int) -> int:
    return rank * 4 + suit


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:  # the GPU solver needs triton; CPU-only installs skip these tests
        if e.name == "triton":
            pytest.skip("needs triton (GPU solver)")
        raise
    return mod


def _script(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    return _module(f"{name}_leafnet_test", ROOT / "scripts" / f"{name}.py")


def _at_base(rel: str, tmp_path: Path, name: str):
    """`rel` as it was at BASE, imported as `name`."""
    try:
        src = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git cannot show {BASE}:{rel}")
    p = tmp_path / f"{name}.py"
    p.write_bytes(src)
    return _module(name, p)


def _stabilizer(board) -> list[int]:
    s = set(int(c) for c in board)
    return [k for k in range(24) if set(vn.CARD_PERM[k, list(board)].tolist()) == s]


def _boards(n: int, seed: int, trivial: bool | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        b = rng.choice(52, 4, replace=False)
        if trivial is None or (len(_stabilizer(b)) == 1) == trivial:
            out.append(b)
    return np.array(out, dtype=np.int64)


def _situations(boards: np.ndarray, seed: int, dtype=torch.float32):
    """Random valid inputs in the net's 1,326 form: ranges of mass 1 on the
    live holdings, the one-hot board, scalars, the live mask."""
    rng = np.random.default_rng(seed)
    live = vn.live_mask(boards)
    r = rng.random((len(boards), 2, vn.HOLDINGS)) * live[:, None, :]
    r /= r.sum(-1, keepdims=True)
    onehot = np.zeros((len(boards), 52))
    np.put_along_axis(onehot, boards, 1.0, 1)
    scal = vn.scalars(rng.uniform(100, 2000, len(boards)), rng.uniform(0, 10000, len(boards)))
    return (torch.as_tensor(r, dtype=dtype), torch.as_tensor(onehot, dtype=dtype), torch.as_tensor(scal, dtype=dtype),
            torch.as_tensor(live))


def _net(canonical: bool, seed: int = 0, width: int = 64, depth: int = 2):
    torch.manual_seed(seed)
    net = vn.ValueNet(width, depth, canonical).eval()
    with torch.no_grad():
        net.scal_mean.copy_(torch.tensor([0.3, 1.5]))
        net.scal_std.copy_(torch.tensor([0.2, 1.1]))
    return net


def _inverse(k: int) -> int:
    p = vn.SUIT_PERMS
    return int(np.flatnonzero((p[:, p[k]] == np.arange(4)).all(1))[0])


# ---------------------------------------------------------------- canonical=False


def test_plain_net_is_bit_identical_to_the_base_code(tmp_path):
    old = _at_base("src/parciver/value_net.py", tmp_path, "value_net_base")
    boards = _boards(16, 1)
    x = _situations(boards, 2)
    if N0 is not None:  # N0's weights, as the live search runs them
        new, _ = vn.load(N0, device="cpu")
        before, _ = old.load(N0, device="cpu")
    else:
        new = _net(False)
        before = old.ValueNet(64, 2).eval()
        before.load_state_dict(new.state_dict())
    assert new.canonical is False
    with torch.no_grad():
        assert torch.equal(new(*x), before(*x))
    assert list(new.state_dict()) == list(before.state_dict())


def test_blob_without_the_flag_loads_plain_and_save_keeps_the_flag(tmp_path):
    plain, canon = _net(False), _net(True)
    assert list(canon.state_dict()) == list(plain.state_dict())  # the frame's tables are not in the state dict
    vn.save(plain, tmp_path / "plain.pt")
    vn.save(canon, tmp_path / "canon.pt")
    assert "canonical" not in torch.load(tmp_path / "plain.pt", weights_only=False)  # a plain blob as before
    a, _ = vn.load(tmp_path / "plain.pt", device="cpu")
    b, _ = vn.load(tmp_path / "canon.pt", device="cpu")
    assert a.canonical is False and b.canonical is True
    x = _situations(_boards(8, 3), 4)
    with torch.no_grad():
        assert torch.equal(b(*x), canon(*x)) and torch.equal(a(*x), plain(*x))


# ---------------------------------------------------------------- canonical=True


def _python_first_min(board) -> int:
    best, at = None, None
    for k in range(24):
        key = tuple(sorted(vn.CARD_PERM[k, list(board)].tolist()))
        if best is None or key < best:
            best, at = key, k
    return at


def test_frame_index_is_the_first_minimum():
    boards = _boards(10_000, 5)
    want = np.array([_python_first_min(b) for b in boards])
    assert np.array_equal(vn.frame_index(boards), want)
    # Every board class: each canonical board and one relabelling of it.
    every = np.array(list(itertools.combinations(range(52), 4)), dtype=np.int64)
    rel = np.sort(vn.CARD_PERM[:, every], -1)  # [24, N, 4]
    key = ((rel[..., 0] * 52 + rel[..., 1]) * 52 + rel[..., 2]) * 52 + rel[..., 3]
    classes = np.unique(key.min(0))
    reps = np.stack([classes // 52 ** 3, classes // 52 ** 2 % 52, classes // 52 % 52, classes % 52], 1)
    moved = vn.CARD_PERM[np.random.default_rng(6).integers(0, 24, len(reps))[:, None], reps]
    for b in np.concatenate([reps, moved]):
        assert vn.frame_index(b) == _python_first_min(b)
        assert tuple(sorted(vn.CARD_PERM[vn.frame_index(b), b].tolist())) == vn.canonical_board(b)
    # The torch frame equals the reference.
    net = _net(True)
    for bs in (boards, reps, moved):
        onehot = torch.zeros(len(bs), 52)
        onehot.scatter_(1, torch.as_tensor(bs), 1.0)
        assert np.array_equal(net.frame(onehot).numpy(), vn.frame_index(bs))


def test_equivariance_is_bit_exact_where_the_stabilizer_is_trivial():
    net = _net(True)
    boards = _boards(64, 7, trivial=True)
    r, onehot, scal, live = _situations(boards, 8)
    rng = np.random.default_rng(9)
    with torch.no_grad():
        v = net(r, onehot, scal, live)
        for _ in range(5):
            k = torch.as_tensor(rng.integers(0, 24, len(boards)))
            r2, sv, oh2, live2 = vn.permute(k, r, v, onehot, live)
            assert torch.equal(net(r2, oh2, scal, live2), sv)  # N(s.x) == s.N(x), bit for bit


def _inner_inputs(net, x):
    got = []
    inner = net.inner

    def spy(*a):
        got.append(a)
        return inner(*a)

    net.inner = spy
    try:
        with torch.no_grad():
            v = net(*x)
    finally:
        del net.inner
    return got[0], v


def test_inner_input_is_a_valid_canonical_situation():
    net = _net(True)
    boards = np.concatenate([_boards(48, 10), [[card(12, 0), card(11, 1), card(10, 2), card(12, 3)]]])
    (r, oh, scal, live), _ = _inner_inputs(net, _situations(boards, 11))
    for i, b in enumerate(boards):
        inner_board = np.flatnonzero(oh[i].numpy())
        assert tuple(inner_board.tolist()) == vn.canonical_board(b)
        assert np.array_equal(live[i].numpy(), vn.live_mask(inner_board))
        assert (r[i][:, ~live[i]] == 0).all()
        np.testing.assert_allclose(r[i].sum(-1).numpy(), 1.0, rtol=1e-5)


def test_forward_is_the_explicit_permute_reference():
    net = _net(True)
    boards = _boards(64, 12)
    r, onehot, scal, live = _situations(boards, 13)
    k = torch.as_tensor(vn.frame_index(boards))
    back = torch.as_tensor([_inverse(int(i)) for i in k])
    with torch.no_grad():
        r2, _, oh2, live2 = vn.permute(k, r, r, onehot, live)
        v = net.inner(r2, oh2, scal, live2)
        _, ref, _, _ = vn.permute(back, r2, v, oh2, live2)
        assert torch.equal(net(r, onehot, scal, live), ref)


def test_larger_stabilizer_inner_inputs_differ_by_a_member_of_it():
    net = _net(True)
    boards = np.array([[card(12, 0), card(11, 1), card(10, 2), card(12, 3)],  # As Kh Qd Ac: |H_b| = 2
                       [card(12, 0), card(12, 1), card(3, 2), card(3, 3)],  # two pairs
                       [card(12, 0), card(11, 0), card(3, 0), card(7, 0)]])  # monotone: |H_b| = 6
    assert [len(_stabilizer(b)) for b in boards] == [2, 4, 6]
    x = _situations(boards, 14)
    (r, oh, _, live), _ = _inner_inputs(net, x)
    rng = np.random.default_rng(15)
    for _ in range(4):
        k = torch.as_tensor(rng.integers(0, 24, len(boards)))
        r2, _, oh2, live2 = vn.permute(k, x[0], x[0], x[1], x[3])
        (rs, ohs, _, lives), _ = _inner_inputs(net, (r2, oh2, x[2], live2))
        for i in range(len(boards)):
            canon = vn.canonical_board(boards[i])
            assert tuple(np.flatnonzero(ohs[i].numpy()).tolist()) == canon
            ok = False
            for h in _stabilizer(canon):
                hh = torch.tensor([h])
                a, _, _, _ = vn.permute(hh, r[i:i + 1], r[i:i + 1], oh[i:i + 1], live[i:i + 1])
                ok |= torch.equal(a, rs[i:i + 1])
            assert ok


def test_canonical_values_are_zero_sum():
    net = _net(True)
    r, onehot, scal, live = _situations(_boards(64, 16), 17)
    with torch.no_grad():
        v = net(r, onehot, scal, live)
    assert ((r * v).sum((1, 2)).abs() <= 1e-5 * v.abs().amax((1, 2))).all()
    assert (v[~live[:, None, :].expand(-1, 2, -1)] == 0).all()


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_cuda_graph_replay_equals_eager():
    net = _net(True).cuda()
    x = [t.cuda() for t in _situations(_boards(32, 18), 19)]
    with torch.no_grad():
        eager = net(*x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            net(*x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = net(*x)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
def test_flop_search_with_a_canonical_net_graph_equals_eager():
    pytest.importorskip("parciver_engine")
    from parciver import flop_search as fs

    sym = _module("test_flop_sym_leafnet", ROOT / "tests" / "test_flop_sym.py")
    net = _net(True).cuda()
    job = dict(sym._job(sym.FLOPS["two-tone"], True), iters=20)
    out = fs.FlopSearch(net, leaves="sym")(job)
    assert fs.FlopSearch(net, leaves="sym", graph=False)(job) == out


@pytest.mark.parametrize("cosets", [True, False])
def test_flop_sym_brute_force_with_a_canonical_net(cosets):
    pytest.importorskip("parciver_engine")
    pytest.importorskip("triton")
    from parciver import flop_search as fs

    sym_t = _module("test_flop_sym_brute", ROOT / "tests" / "test_flop_sym.py")
    net = _net(True, 3, width=32).double()

    def ev(ranges, board, live, half, stack):
        L, R = ranges.shape[:2]
        onehot = torch.zeros(R, 52, dtype=torch.float64).scatter_(1, board.cpu(), 1.0)
        scal = torch.as_tensor(vn.scalars(half.numpy(), stack.numpy()), dtype=torch.float64)
        with torch.no_grad():
            v = net(ranges.reshape(L * R, 2, -1), onehot.repeat(L, 1), scal.repeat_interleave(R, 0), live.repeat(L, 1))
        return v.view(L, R, 2, -1)

    for flop in sym_t.FLOPS.values():
        sym = fs.SymLeaves(flop, ev, device="cpu", dtype=torch.float64, cosets=cosets)
        dense = fs.NetLeaves(flop, sym_t._brute(ev, flop, cosets), device="cpu", dtype=torch.float64)
        reach = sym_t._symmetric_reach(sym, 3, 1)
        half = torch.tensor([150.0, 400.0, 1000.0], dtype=torch.float64)
        stack = torch.tensor([2000.0, 1000.0, 500.0], dtype=torch.float64)
        a, b = sym(reach, half, stack), dense(reach, half, stack)
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-10, atol=1e-10 * float(b.abs().max()))


def test_symmetry_defect_is_zero_for_a_canonical_net_where_hb_is_trivial():
    tn = _script("turn_net")
    d = _toy_data(48, 5)
    d["board"][:3] = [[card(12, 0), card(11, 1), card(10, 2), card(12, 3)], [card(12, 0), card(12, 1), card(3, 2), card(3, 3)],
                      [card(12, 0), card(11, 0), card(3, 0), card(7, 0)]]
    b = tn.Batches(d, np.arange(48), torch.device("cpu"))
    trivial = np.array([len(h) == 1 for h in tn.hb_members(d["board"])])
    canon, plain = tn.symmetry_defect(_net(True, 6), b), tn.symmetry_defect(_net(False, 6), b)
    assert trivial.sum() >= 10 and (canon[trivial] == 0).all() and (plain[trivial] > 0).all()
    assert (canon[~trivial] > 0).any()  # a member of H_b relabels the inner input
    assert np.array_equal(tn.symmetry_defect(_net(False, 6), b), plain)  # the same s every time (seed 0)


def test_gate_a_result_from_per_sample_arrays():
    tn = _script("turn_net")
    rng = np.random.default_rng(0)
    n = 400
    cluster, source = rng.integers(0, 80, n), rng.integers(0, 2, n)
    trivial = rng.random(n) < 0.9
    n0 = rng.uniform(3, 7, n)
    defect = {"P": np.where(trivial, 0.0, 0.3), "N0": rng.uniform(0.1, 1, n)}

    def run(p, bar=9.0):
        return tn.gate_a_result({"P": p, "N0": n0}, {"P": p + 0.1, "N0": n0 + 0.1}, defect, cluster, trivial, source, bar)

    got = run(n0 - 0.2 + rng.normal(0, 0.05, n))
    assert got["proceed"] and got["averaged"]["ci"][1] < 0 and got["samples"] == n and got["trivial_hb"] == trivial.sum()
    assert got["averaged"]["relative"] == pytest.approx(got["averaged"]["P"] / got["averaged"]["N0"] - 1)
    assert got["bare"]["P"] == pytest.approx(got["averaged"]["P"] + 0.1)
    assert got["averaged"]["real"]["P"] == pytest.approx((n0 - 0.2)[source == 0].mean(), abs=0.02)
    assert got["symmetry_defect"]["P"]["trivial"] == dict(mean=0.0, max=0.0)
    assert not run(n0 - 0.2 + rng.normal(0, 0.05, n), bar=1.0)["proceed"]  # the G2 bar
    assert not run(n0 + rng.normal(0, 0.05, n))["proceed"]  # not superior
    json.dumps(got)  # all plain JSON


def test_factored_leaves_refuse_a_canonical_net():
    pytest.importorskip("triton")
    from parciver import flop_leaves as fl

    with pytest.raises(ValueError, match="canonical"):
        fl.FactoredLeaves((0, 13, 26), _net(True), device="cpu")


# ---------------------------------------------------------------- training and Gate A


def test_optimizer_decays_exactly_the_linear_weights():
    tn = _script("turn_net")
    net = _net(False)
    assert type(tn.optimizer(net, 1e-3, 0.0)) is torch.optim.Adam
    opt = tn.optimizer(net, 1e-3, 0.1)
    assert type(opt) is torch.optim.AdamW
    decay, rest = opt.param_groups
    weights = [m.weight for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert decay["weight_decay"] == 0.1 and rest["weight_decay"] == 0.0
    assert {id(p) for p in decay["params"]} == {id(p) for p in weights}
    assert {id(p) for p in rest["params"]} == {id(p) for p in net.parameters()} - {id(p) for p in weights}


def _toy_data(n: int = 96, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return dict(values=rng.normal(size=(n, 2, vn.LIVE)).astype(np.float32),
                ranges=rng.random((n, 2, vn.LIVE)).astype(np.float32),
                board=_boards(n, seed + 1), half=rng.uniform(100, 2000, n), stack=rng.uniform(0, 10000, n),
                source=rng.integers(0, 2, n))


def test_wd0_trains_bit_identically_to_the_base_code(tmp_path, monkeypatch):
    tn = _script("turn_net")
    old = _at_base("scripts/turn_net.py", tmp_path, "turn_net_base")
    d = _toy_data()
    part = np.full(len(d["board"]), tn.TRAIN)
    part[::6] = tn.SELECT
    for m, sub in ((tn, "new"), (old, "old")):
        monkeypatch.setattr(m, "NETS", tmp_path / sub)
    args = (d, part, 16, 1, 1.0, 2, 8, 1e-3, 0, "t")
    new = tn.train(*args, log=lambda *_: None, device="cpu", canonical=False, wd=0.0)
    was = old.train(*args, log=lambda *_: None, device="cpu")
    strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731
    assert strip(new["history"]) == strip(was["history"])
    a = torch.load(tmp_path / "new" / "t" / "net.pt", weights_only=False)["state"]
    b = torch.load(tmp_path / "old" / "t" / "net.pt", weights_only=False)["state"]
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert new["canonical"] is False and new["wd"] == 0.0
    # A record from before LEAF-NET (no canonical / wd keys) is reused as a plain wd-0 net.
    old_rec = json.loads((tmp_path / "new" / "t" / "record.json").read_text())
    for k in ("canonical", "wd"):
        old_rec.pop(k)
    (tmp_path / "new" / "t" / "record.json").write_text(json.dumps(old_rec))
    assert tn.train(*args, log=lambda *_: None, device="cpu", reuse=True) == old_rec
    assert tn.train(*args, log=lambda *_: None, device="cpu", reuse=True, wd=0.1)["wd"] == 0.1  # trained, not reused


def test_the_grid_and_gate_a_refuse_other_data(tmp_path, monkeypatch, capsys):
    tn = _script("turn_net")
    monkeypatch.setattr(tn, "NETS", tmp_path)
    monkeypatch.setattr(tn, "_data", lambda a: (_toy_data(8), np.zeros(8, int)))
    assert tn.fingerprint(_toy_data(8)) != tn.LEAFNET_DATA
    assert tn.cmd_leafnet(types.SimpleNamespace()) == 1
    (tmp_path / "leafnet.json").write_text(json.dumps(dict(reproduced=True, pick="leafnet-c1-wd0.1", stop=False)))
    assert tn.cmd_gate_a(types.SimpleNamespace()) == 1
    assert capsys.readouterr().out.count("refused") == 2


def test_the_grid_stops_unless_n0_is_reproduced(tmp_path, monkeypatch):
    tn = _script("turn_net")
    monkeypatch.setattr(tn, "NETS", tmp_path)
    monkeypatch.setattr(tn, "_data", lambda a: ({}, None))
    monkeypatch.setattr(tn, "fingerprint", lambda d: tn.LEAFNET_DATA)
    (tmp_path / tn.LEAFNET_N0).mkdir()
    n0 = dict(epoch=158, val_loss=0.014464557576207186, val_mae=5.209)
    (tmp_path / tn.LEAFNET_N0 / "record.json").write_text(json.dumps(dict(best=n0)))
    losses = {tn.leafnet_tag(c, w): 0.0145 - 0.0001 * i for i, (c, w) in
              enumerate(itertools.product((False, True), tn.LEAFNET_WD))}
    trained = []

    def fake(d, part, width, depth, fraction, epochs, batch, lr, seed, tag, canonical, wd, **kw):
        assert (width, depth, epochs, batch, lr, seed, fraction) == (2048, 5, 200, 2048, 1e-3, 0, 1.0)
        assert tag == tn.leafnet_tag(canonical, wd) and kw["reuse"] is True
        trained.append(tag)
        best = dict(n0) if tag == tn.leafnet_tag(False, 0.0) else dict(epoch=100, val_loss=losses[tag], val_mae=5.0)
        return dict(tag=tag, canonical=canonical, wd=wd, best=best)

    monkeypatch.setattr(tn, "train", lambda *a, **kw: fake(*a, **kw))
    assert tn.cmd_leafnet(types.SimpleNamespace()) == 0
    got = json.loads((tmp_path / "leafnet.json").read_text())
    assert got["reproduced"] and got["pick"] == "leafnet-c1-wd1" and not got["stop"] and len(trained) == 8
    assert trained[0] == "leafnet-c0-wd0"
    n0["val_loss"] += 1e-12  # not reproduced to the last bit: nothing after D2's recipe is trained
    trained.clear()
    (tmp_path / tn.LEAFNET_N0 / "record.json").write_text(json.dumps(dict(best=n0)))
    monkeypatch.setattr(tn, "train", lambda *a, **kw: dict(fake(*a, **kw), best=dict(epoch=158, val_loss=0.014464557576207186, val_mae=5.209)))
    assert tn.cmd_leafnet(types.SimpleNamespace()) == 1
    assert trained == ["leafnet-c0-wd0"] and json.loads((tmp_path / "leafnet.json").read_text())["reproduced"] is False


def test_stabilizers_and_the_rainbow_flop_claim():
    tn = _script("turn_net")
    flop = [card(12, 0), card(11, 1), card(10, 2)]  # As Kh Qd: unpaired rainbow
    turns = np.array([flop + [c] for c in range(52) if c not in flop])
    hb = tn.hb_members(turns)
    assert sum(len(h) == 1 for h in hb) == 46 and sorted(len(h) for h in hb)[-3:] == [2, 2, 2]
    for b, h in zip(turns, hb):
        assert h[0] == 0 and h == _stabilizer(b)
    for k in range(24):
        assert (vn.SUIT_PERMS[tn.inverse_perm(k)][vn.SUIT_PERMS[k]] == np.arange(4)).all()


def test_board_bootstrap_resamples_whole_boards():
    tn = _script("turn_net")
    diff = np.array([1.0, 1.0, 1.0, -2.0])
    cluster = np.array([5, 5, 5, 9])
    got = tn.board_bootstrap(diff, cluster, resamples=2000)
    assert got["boards"] == 2 and got["samples"] == 4 and got["mean"] == pytest.approx(0.25)
    # Resampled boards carry all their samples: a mean is one of (3 + 3) / 6, (3 - 2) / 4, -4 / 2.
    assert got["ci"][0] == pytest.approx(-2.0) and got["ci"][1] == pytest.approx(1.0)
    again = tn.board_bootstrap(diff, cluster, resamples=2000)
    assert again == got


def test_hb_average_and_weighted_mae_on_cpu():
    tn = _script("turn_net")
    d = _toy_data(40, 3)
    d["board"][0] = [card(12, 0), card(11, 1), card(10, 2), card(12, 3)]  # |H_b| = 2
    rows = np.arange(40)
    b = tn.Batches(d, rows, torch.device("cpu"))
    hb = tn.hb_members(d["board"])
    net = _net(False, seed=4)
    avg = tn.hb_average(net, b, hb)
    r, v, onehot, scal, live = b.get(torch.arange(40), False)
    with torch.no_grad():
        bare = net(r, onehot, scal, live).double().numpy()
    one = np.array([len(h) == 1 for h in hb])
    assert np.array_equal(avg[one], bare[one])  # the identity term alone
    h = hb[0][1]
    r2, _, oh2, l2 = vn.permute(torch.tensor([h]), r[:1], r[:1], onehot[:1], live[:1])
    with torch.no_grad():
        _, w, _, _ = vn.permute(torch.tensor([_inverse(h)]), r2, net(r2, oh2, scal[:1], l2), oh2, l2)
    # Another batch shape (the rows with |H_b| > 1) rounds float32 differently.
    np.testing.assert_allclose(avg[0], (bare[0] + w[0].double().numpy()) / 2, rtol=1e-4, atol=1e-7)
    mae = tn.weighted_mae(bare, b)
    np.testing.assert_allclose(mae, tn.evaluate(net, b)["mae_per_sample"].double().numpy() * 100, rtol=1e-5)


# ---------------------------------------------------------------- Gate B (flop_m1.py)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


ROW = dict(pot=200, hero=0, frozen=3, isomorphism=True, iters=500, br=[1.0, 2.0], expl=0.1, lo=-0.01, hi=0.01,
           value=0.0, profile=0.0, curve=[[100, 0.5, 0.0]])


def test_symcheck_and_the_engine_check(tmp_path, monkeypatch):
    m1 = _script("flop_m1")
    monkeypatch.setattr(m1, "LEAF_RECORDS", tmp_path / "rec")
    a = types.SimpleNamespace(out=str(tmp_path))
    (tmp_path / "m1-x-sym.npz").write_bytes(b"X_sym")
    (tmp_path / "m1-x-sym.json").write_text(json.dumps(dict(cand="sym", net_sha256="n0")))
    rows = [dict(ROW, spot=i, value=0.01 * i, sec=[50, 30, 40, 9, 20][i]) for i in range(5)]
    rows[3]["copied"] = rows[4]["copied"] = "net"
    (tmp_path / "m1-v-sym.json").write_text(json.dumps(dict(cand="sym", rows=rows)))
    assert m1.check_spots(rows) == (1, 3)  # the fastest solved row, the fastest copied row
    assert m1.cmd_symcheck_init(a) == 0 and m1.cmd_symcheck_init(a) == 1
    assert (tmp_path / "m1-x-symcheck.npz").read_bytes() == b"X_sym"
    assert json.loads((tmp_path / "m1-x-symcheck.json").read_text())["cand"] == "symcheck"
    assert m1.cmd_enginecheck(a) == 2  # the check spots are not solved yet

    def solved(change):
        again = [dict({k: v for k, v in rows[i].items() if k != "copied"}, sec=1.0, **change) for i in (1, 3)]
        (tmp_path / "m1-v-symcheck.json").write_text(json.dumps(dict(cand="symcheck", inputs=dict(engine="E"), rows=again)))
        return m1.cmd_enginecheck(a), json.loads((tmp_path / "rec" / "enginecheck.json").read_text())

    code, got = solved({})
    assert code == 0 and got["passed"] and got["baseline"] == "sym" and got["spots"] == [1, 3] and got["engine"] == "E"
    assert got["files"] == {f: _sha(tmp_path / f) for f in ("m1-v-sym.json", "m1-x-sym.npz")}
    code, got = solved(dict(curve=[[100, 0.5, 1e-17]]))
    assert code == 1 and got["baseline"] == "symcheck" and got["differ"] == {"1": ["curve"], "3": ["curve"]}
    with pytest.raises(SystemExit, match="symcheck"):
        m1.cmd_strategy(types.SimpleNamespace(out=str(tmp_path), cand="symcheck"))
    (tmp_path / "p.pt").write_bytes(b"P")
    with pytest.raises(SystemExit, match="needs --net"):
        m1.cmd_strategy(types.SimpleNamespace(out=str(tmp_path), cand="leafnet", net=None))
    with pytest.raises(SystemExit, match="Gate A"):  # no gate-a.json
        m1.cmd_strategy(types.SimpleNamespace(out=str(tmp_path), cand="leafnet", net=str(tmp_path / "p.pt")))
    (tmp_path / "rec" / "gate-a.json").write_text(json.dumps(dict(proceed=True, sha256=dict(P="another net"))))
    with pytest.raises(SystemExit, match="Gate A"):
        m1.cmd_strategy(types.SimpleNamespace(out=str(tmp_path), cand="leafnet", net=str(tmp_path / "p.pt")))
    with pytest.raises(SystemExit, match="one of each"):
        m1.check_spots([dict(spot=0, sec=1.0)])


def _gate_b_files(tmp_path, monkeypatch, n=4, solved=4, base="sym", gap=0.0):
    m1 = _script("flop_m1")
    repo = tmp_path / "repo"
    (repo / "runs" / "turn-net" / "pick").mkdir(parents=True)
    (repo / "runs" / "turn-net" / "pick" / "net.pt").write_bytes(b"P")
    (repo / "runs" / "turn-net" / "leafnet.json").write_text(json.dumps(dict(reproduced=True, pick="pick", stop=False)))
    (repo / "n0.pt").write_bytes(b"N0")
    monkeypatch.setattr(m1, "REPO", repo)
    monkeypatch.setattr(m1, "LEAF_RECORDS", tmp_path / "rec")
    monkeypatch.setattr(m1, "N0_SHA256", _sha(repo / "n0.pt"))
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "gate-a.json").write_text(json.dumps(dict(proceed=True, sha256=dict(P=_sha(repo / "runs" / "turn-net" / "pick" / "net.pt")))))
    out = tmp_path / "eval"
    out.mkdir()
    (out / "m1-spots.json").write_text(json.dumps(dict(spots=[dict(board=[0, 5, 10])] * n)))
    head = dict(menus=[1], rule=[5.0, 100, 0.2, 1000], precision="f32")
    rng = np.random.default_rng(0)
    base_v = rng.normal(0, 0.01, n)
    for c, v, nets in ((base, base_v, "n0.pt"), ("leafnet", base_v - gap + rng.normal(0, 0.0005, n), "runs/turn-net/pick/net.pt")):
        (out / f"m1-x-{c}.npz").write_bytes(c.encode())
        (out / f"m1-x-{c}.json").write_text(json.dumps(dict(cand=c, net=nets, net_sha256=_sha(repo / nets))))
        k = n if c != "leafnet" else solved
        rows = [dict(ROW, spot=i, value=float(v[i]), sec=10.0) for i in range(k)]
        (out / f"m1-v-{c}.json").write_text(json.dumps(dict(head, cand=c, rows=rows, inputs=dict(
            spots="s", x=_sha(out / f"m1-x-{c}.npz"), **({} if c == "sym" else dict(engine="E", threads=12))))))
    files = {f: _sha(out / f) for f in ("m1-v-sym.json", "m1-x-sym.npz")} if base == "sym" else {}
    (tmp_path / "rec" / "enginecheck.json").write_text(json.dumps(dict(passed=base == "sym", baseline=base, engine="E",
                                                                       files=files)))
    return m1, out


def test_gate_b_report_pairs_leafnet_with_the_engine_checked_baseline(tmp_path, monkeypatch):
    m1, out = _gate_b_files(tmp_path, monkeypatch)
    assert m1.cmd_report(types.SimpleNamespace(out=str(out))) == 0
    got = json.loads((tmp_path / "rec" / "m1-report-leafnet.json").read_text())
    d = np.array([got["per_spot"][str(i)] for i in range(4)])
    assert got["baseline"] == "sym" and got["final"] and got["z"] == 1.96 and got["margin"] == 0.0025
    assert got["mean"] == pytest.approx(d.mean()) and got["ci"][1] == pytest.approx(d.mean() + 1.96 * d.std(ddof=1) / 2)
    assert got["non_inferior"] is (got["ci"][1] + m1.ORACLE_ERROR <= 0.0025)
    assert not (out / "m1-report-flopsym.json").exists()
    # A worse leafnet fails; an interim look gives no verdict.
    m1, out = _gate_b_files(tmp_path / "worse", monkeypatch, gap=0.01)
    m1.cmd_report(types.SimpleNamespace(out=str(out)))
    assert json.loads((tmp_path / "worse" / "rec" / "m1-report-leafnet.json").read_text())["non_inferior"] is False
    m1, out = _gate_b_files(tmp_path / "interim", monkeypatch, solved=3)
    m1.cmd_report(types.SimpleNamespace(out=str(out)))
    got = json.loads((tmp_path / "interim" / "rec" / "m1-report-leafnet.json").read_text())
    assert got["final"] is False and got["non_inferior"] is None and got["spots"] == 3
    # A failed engine check pairs with symcheck.
    m1, out = _gate_b_files(tmp_path / "check", monkeypatch, base="symcheck")
    m1.cmd_report(types.SimpleNamespace(out=str(out)))
    assert json.loads((tmp_path / "check" / "rec" / "m1-report-leafnet.json").read_text())["baseline"] == "symcheck"


def test_gate_b_report_refuses_another_net_or_a_copied_row(tmp_path, monkeypatch):
    m1, out = _gate_b_files(tmp_path, monkeypatch)
    (m1.REPO / "runs" / "turn-net" / "pick" / "net.pt").write_bytes(b"P changed")
    with pytest.raises(SystemExit, match="pick P"):
        m1.cmd_report(types.SimpleNamespace(out=str(out)))
    m1, out = _gate_b_files(tmp_path / "n0", monkeypatch)
    monkeypatch.setattr(m1, "N0_SHA256", "other")
    with pytest.raises(SystemExit, match="N0"):
        m1.cmd_report(types.SimpleNamespace(out=str(out)))
    m1, out = _gate_b_files(tmp_path / "gate", monkeypatch)
    (tmp_path / "gate" / "rec" / "gate-a.json").write_text(json.dumps(dict(proceed=False, sha256=dict(P="x"))))
    with pytest.raises(SystemExit, match="Gate A"):
        m1.cmd_report(types.SimpleNamespace(out=str(out)))
    m1, out = _gate_b_files(tmp_path / "engine", monkeypatch)
    f = json.loads((out / "m1-v-leafnet.json").read_text())
    f["inputs"]["engine"] = "another build"
    (out / "m1-v-leafnet.json").write_text(json.dumps(f))
    with pytest.raises(SystemExit, match="engine build"):
        m1.cmd_report(types.SimpleNamespace(out=str(out)))
    m1, out = _gate_b_files(tmp_path / "moved", monkeypatch)
    f = json.loads((out / "m1-v-sym.json").read_text())
    f["rows"][0]["value"] += 1e-9  # X_sym's rows changed after the engine check
    (out / "m1-v-sym.json").write_text(json.dumps(f))
    with pytest.raises(SystemExit, match="after the engine check"):
        m1.cmd_report(types.SimpleNamespace(out=str(out)))
    m1, out = _gate_b_files(tmp_path / "copied", monkeypatch)
    f = json.loads((out / "m1-v-leafnet.json").read_text())
    f["rows"][0]["copied"] = "sym"
    (out / "m1-v-leafnet.json").write_text(json.dumps(f))
    with pytest.raises(SystemExit, match="copied"):
        m1.cmd_report(types.SimpleNamespace(out=str(out)))


# ---------------------------------------------------------------- speed (flop_sym_speed.py)


def test_flop_bootstrap_is_the_old_inline_code():
    sp = _script("flop_sym_speed")
    rng = np.random.default_rng(1)
    rows = [dict(flop=[int(f), 1, 2], ratio=float(r)) for f, r in zip(rng.integers(0, 9, 60), rng.normal(1, 0.05, 60))]
    flops = sorted({tuple(r["flop"]) for r in rows})
    by_flop = [np.array([r["ratio"] for r in rows if tuple(r["flop"]) == f]) for f in flops]
    g = np.random.default_rng(0)
    boot = [np.median(np.concatenate([by_flop[j] for j in g.integers(0, len(flops), len(flops))])) for _ in range(10_000)]
    assert sp.flop_bootstrap(rows) == tuple(float(x) for x in np.percentile(boot, [2.5, 97.5]))


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
def test_leafnet_speed_runs_its_jobs(tmp_path, monkeypatch):
    pytest.importorskip("parciver_engine")
    sp = _script("flop_sym_speed")
    sym = _module("test_flop_sym_speed_jobs", ROOT / "tests" / "test_flop_sym.py")
    monkeypatch.setattr(sp, "LEAF_RECORDS", tmp_path / "rec")
    monkeypatch.setattr(sp, "ITERS", 5)
    jobs = [sym._job(sym.FLOPS["two-tone"], True), sym._job(sym.RAINBOW, False), sym._job(sym.FLOPS["monotone"], True)]
    vn.save(_net(False, 1), tmp_path / "a.pt")
    vn.save(_net(True, 2), tmp_path / "b.pt")
    with pytest.raises(SystemExit, match="not canonical"):
        sp.leafnet_speed(tmp_path, jobs, str(tmp_path / "a.pt"), str(tmp_path / "a.pt"), "a", "a")
    code = sp.leafnet_speed(tmp_path, jobs, str(tmp_path / "a.pt"), str(tmp_path / "b.pt"), "a", "b")
    got = json.loads((tmp_path / "rec" / "speed.json").read_text())
    assert code in (0, 5) and got["passed"] is (code == 0) and got["jobs"] == 3 and got["flops"] == 3
    assert got["bar"] == 1.02 and got["solves"]["A"] == got["solves"]["B"]
    assert got["symmetric"]["A"] == got["symmetric"]["B"] >= 2
    assert len(json.loads((tmp_path / "leafnet-speed.json").read_text())["rows"]) == 3


def test_speed_two_net_mode_checks_both_nets(tmp_path, monkeypatch):
    sp = _script("flop_sym_speed")
    (tmp_path / "a.pt").write_bytes(b"N0")
    (tmp_path / "b.pt").write_bytes(b"P")
    (tmp_path / "m1-jobs-leafnet.pkl").write_bytes(pickle.dumps([]))
    (tmp_path / "m1-x-leafnet.json").write_text(json.dumps(dict(net_sha256=_sha(tmp_path / "b.pt"))))
    monkeypatch.setattr(sp, "N0_SHA256", _sha(tmp_path / "a.pt"))
    monkeypatch.setattr(sp, "leafnet_speed", lambda out, jobs, a, b, sa, sb: 7)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["flop_sym_speed.py", "--out", str(tmp_path), *argv])
        return sp.main()

    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    assert run("--cand", "leafnet", "--net", a, "--net-b", b) == 7
    for argv in (("--cand", "leafnet", "--net", b, "--net-b", b), ("--cand", "leafnet", "--net", a, "--net-b", a),
                 ("--cand", "leafnet", "--net", a), ("--cand", "sym", "--net", a, "--net-b", b)):
        with pytest.raises(SystemExit):
            run(*argv)


def test_engine_sha_is_the_loaded_binary():
    pe = pytest.importorskip("parciver_engine")
    m1 = _script("flop_m1")
    f = Path(pe.parciver_engine.__file__)
    assert f.suffix in (".pyd", ".so") and m1.engine_sha(pe) == _sha(f)
    with pytest.raises(SystemExit, match="engine binary"):
        m1.engine_sha(types.SimpleNamespace(__file__="x.py"))
