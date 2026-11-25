"""FLOP-SYM (records/flop-sym/PREREG.md): suit-symmetric flop leaves
(`flop_search.SymLeaves`, `FlopSearch(leaves="sym" | "orbit")`).

- The row counts the registration states, over all 22,100 flops.
- With any evaluator (a random, non-symmetric one, float64), C2 equals the
  dense leaves run on its brute-force symmetrization: each turn card's input
  mapped to its orbit's canonical turn, the evaluator averaged over the turn
  board's whole stabilizer H_b (C1: over G_t), mapped back. This checks the
  orbit, coset and relabelling bookkeeping against the definition.
- With the exact (symmetric) check-down evaluator, through a turn-indexed
  adapter, C1 and C2 equal dense.
- Outputs are bit-equal across each holding orbit of G.
- The search: a rainbow flop or a job G changes takes the dense path
  (bit-identical); a symmetric job returns exactly symmetric strategies;
  CUDA graph replay equals eager."""
import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pe = pytest.importorskip("parciver_engine")
pytest.importorskip("triton")

from parciver import flop_search as fs  # noqa: E402
from parciver import value_net as vn  # noqa: E402

CUDA = torch.cuda.is_available()


def card(rank: int, suit: int) -> int:
    return rank * 4 + suit


FLOPS = {
    "two-tone": (card(12, 0), card(11, 0), card(3, 1)),
    "paired": (card(12, 0), card(12, 1), card(3, 2)),
    "paired two-tone": (card(12, 0), card(12, 1), card(3, 0)),
    "monotone": (card(12, 0), card(11, 0), card(3, 0)),
    "trips": (card(12, 0), card(12, 1), card(12, 2)),
}
RAINBOW = (3, 22, 41)


def test_row_counts_over_all_flops():
    c1 = c2 = 0
    classes = {}
    for f in itertools.combinations(range(52), 3):
        g, turns, size, rows2, _, _ = fs.sym_plan(f, cosets=True)
        r1 = len(turns)
        r2 = len(rows2) if len(g) > 1 else fs.TURNS  # rainbow flops stay dense
        c1, c2 = c1 + r1, c2 + r2
        classes[(len(g), r1, len(rows2))] = classes.get((len(g), r1, len(rows2)), 0) + 1
    assert c1 / 22100 == pytest.approx(39.4235, abs=1e-4)
    assert c2 / 22100 == pytest.approx(40.32, abs=1e-9)
    assert classes == {(1, 49, 52): 6864, (2, 36, 37): 10296, (2, 36, 38): 1872, (2, 37, 40): 1872,
                       (6, 23, 23): 1144, (6, 25, 28): 52}


def _random_evaluator(seed: int):
    """A smooth evaluator with no symmetry, float64, zero at dead holdings."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(2 * vn.HOLDINGS, 16, generator=g, dtype=torch.float64) * 0.05
    b = torch.randn(52, 16, generator=g, dtype=torch.float64)
    c = torch.randn(16, 2 * vn.HOLDINGS, generator=g, dtype=torch.float64)

    def f(ranges, board, live, half, stack):
        L, R = ranges.shape[:2]
        onehot = torch.zeros(R, 52, dtype=torch.float64).scatter_(1, board.cpu(), 1.0)
        h = torch.tanh(ranges.reshape(L, R, -1) @ a + onehot @ b + (half / (half + stack))[:, None, None])
        return (h @ c).view(L, R, 2, -1) * live[None, :, None, :]

    return f


def _brute(evaluate, flop, cosets: bool):
    """`evaluate` symmetrized by brute force, for dense `NetLeaves` (49 turns)."""
    group, turns, *_ = fs.sym_plan(flop)
    rep = {}
    for t in turns:
        for k in group:
            rep.setdefault(int(vn.CARD_PERM[k, t]), (t, k))
    P, PI = torch.as_tensor(vn.PAIR_PERM), torch.as_tensor(vn.PAIR_PERM_INV)

    def f(ranges, board, live, half, stack):
        out = torch.zeros_like(ranges)
        for i, t2 in enumerate(board[:, 3].tolist()):
            t, k = rep[t2]
            b = list(flop) + [t]
            if cosets:
                h = [e for e in range(24) if set(vn.CARD_PERM[e, b].tolist()) == set(b)]
            else:
                h = [e for e in group if vn.CARD_PERM[e, t] == t]
            bt = torch.as_tensor([b])
            lt = torch.as_tensor(vn.live_mask(np.array([b])))
            xs = ranges[:, i][..., P[k]]  # sigma^-1 x: the canonical turn's input
            acc = 0
            for e in h:
                v = evaluate(xs[..., PI[e]][:, None], bt, lt, half, stack)[:, 0]
                acc = acc + v[..., P[e]]
            out[:, i] = (acc / len(h))[..., PI[k]]
        return out

    return f


def _symmetric_reach(sym, L: int, seed: int):
    rng = np.random.default_rng(seed)
    r = rng.random((L, 2, fs.HOLDINGS)) * (rng.random((L, 2, fs.HOLDINGS)) < 0.7)
    return torch.as_tensor(sym.average(r))


@pytest.mark.parametrize("name", list(FLOPS))
@pytest.mark.parametrize("cosets", [True, False])
def test_sym_equals_dense_on_its_brute_force_symmetrization(name, cosets):
    flop = FLOPS[name]
    ev = _random_evaluator(7)
    sym = fs.SymLeaves(flop, ev, device="cpu", dtype=torch.float64, cosets=cosets)
    dense = fs.NetLeaves(flop, _brute(ev, flop, cosets), device="cpu", dtype=torch.float64)
    reach = _symmetric_reach(sym, 3, 1)
    half = torch.tensor([150.0, 400.0, 1000.0], dtype=torch.float64)
    stack = torch.tensor([3000.0, 2000.0, 500.0], dtype=torch.float64)
    a, b = sym(reach, half, stack), dense(reach, half, stack)
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-10, atol=1e-10 * float(b.abs().max()))
    for p in torch.as_tensor(sym.perm_host):
        assert torch.equal(a[..., p], a)  # bit-equal across each holding orbit


def _rows_of(cv):
    """`CheckdownValues` for any rows of its flop's turn boards."""
    pos = {t: i for i, t in enumerate(cv.turns)}

    def f(ranges, board, live, half, stack):
        ti = torch.as_tensor([pos[int(t)] for t in board[:, 3].tolist()], device=cv.w.device)
        r = ranges[..., cv.idx].to(cv.w.dtype)
        v = torch.einsum("lrqj,rij->lrqi", r.flip(2), cv.w[ti]) * (0.5 / fs.RIVERS)
        out = ranges.new_zeros(ranges.shape)
        out[..., cv.idx] = v.to(ranges.dtype)
        return out

    return f


@pytest.mark.skipif(not CUDA, reason="the check-down tables are built on the GPU")
@pytest.mark.parametrize("name", ["two-tone", "paired", "monotone"])
def test_sym_equals_dense_with_the_exact_symmetric_evaluator(name):
    flop = FLOPS[name]
    cv = fs.CheckdownValues(flop, dtype=torch.float64)
    dense = fs.NetLeaves(flop, cv, dtype=torch.float64)
    half = torch.tensor([150.0, 400.0], dtype=torch.float64, device="cuda")
    stack = torch.tensor([3000.0, 2000.0], dtype=torch.float64, device="cuda")
    for cosets in (True, False):
        sym = fs.SymLeaves(flop, _rows_of(cv), dtype=torch.float64, cosets=cosets)
        reach = _symmetric_reach(sym, 2, 2).cuda()
        b = dense(reach, half, stack)
        np.testing.assert_allclose(sym(reach, half, stack).cpu().numpy(), b.cpu().numpy(), rtol=1e-9,
                                   atol=1e-9 * float(b.abs().max()))
    del cv


def test_rainbow_flop_has_no_sym_leaves():
    with pytest.raises(ValueError, match="identity"):
        fs.SymLeaves(RAINBOW, _random_evaluator(0), device="cpu")


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(1)
    return vn.ValueNet(64, 2).cuda().eval()


_PRE = [("c", 0), ("x", 0)]


def _job(flop, symmetric: bool, seed: int = 3):
    captured = []

    def grab(job):
        captured.append(job)
        return fs.FlopSearch(net=None, dtype=torch.float64)(job)

    hole = [c for c in range(52) if c not in flop][:2]
    pe.PlayAgent.uniform("100bb").with_search(from_flop=True, flop_iters=1, solver=grab).decide(
        hero=1, hole=tuple(hole), board=list(flop), actions=_PRE, sb=50, bb=100, stacks=(3000, 3000), seed=0)
    rng = np.random.default_rng(seed)
    r = rng.random((2, fs.HOLDINGS))
    if symmetric and len(fs.flop_group(flop)) > 1:
        r = fs.SymLeaves(flop, _random_evaluator(0), device="cpu").average(r)
    return dict(captured[0], ranges=r.tolist())


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
@pytest.mark.parametrize("leaves", ["sym", "orbit"])
def test_search_dense_path_is_bit_identical(net, leaves):
    for flop, symmetric in ((RAINBOW, True), (FLOPS["two-tone"], False)):
        job = dict(_job(flop, symmetric), iters=20)
        s, d = fs.FlopSearch(net, leaves=leaves), fs.FlopSearch(net, leaves="dense")
        assert s(job) == d(job)
        assert s.symmetric == 0


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
@pytest.mark.parametrize("name", ["two-tone", "monotone"])
def test_search_symmetric_job(net, name):
    flop = FLOPS[name]
    job = dict(_job(flop, True), iters=20)
    s = fs.FlopSearch(net, leaves="sym")
    out = s(job)
    assert s.symmetric == 1
    sym = s.tables(flop)[2]
    for x in out:
        x = np.reshape(x, (-1, fs.HOLDINGS))
        assert sym.symmetric(x)
        np.testing.assert_allclose(x.sum(0), 1.0, atol=1e-6)
    eager = fs.FlopSearch(net, leaves="sym", graph=False)(job)
    assert eager == out  # graph replay is bit-identical to eager steps
    assert fs.FlopSearch(net, leaves="sym")(job) == out  # deterministic
