"""The factored net leaves (src/parciver/flop_leaves.py; KERNELS.md N1) against
the dense path `NetLeaves(board, net_values(net))`, on the CPU: in float64
the two agree to rounding on random, sparse, empty and concentrated reaches;
in float32 the factored path is no further from the float64 dense values than
the dense float32 path's own scale of error, also where nearly all of a
range's mass holds one turn card (the case a plain difference of sums loses);
with the last stage in float64 (N1b) it is closer than the dense path at
every leaf."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")  # flop_search imports gpu_fused

from parciver import flop_leaves as fl  # noqa: E402
from parciver import flop_search as fs  # noqa: E402
from parciver import value_net as vn  # noqa: E402

FLOP = (3, 22, 41)
HALF = [150.0, 400.0, 1000.0, 75.0, 2600.0]
STACK = [9850.0, 9600.0, 500.0, 9925.0, 7400.0]


def _dense(net):
    """`flop_search.net_values` in the net's own dtype (that one feeds float32)."""

    def f(ranges, board, live, half, stack):
        L, T = ranges.shape[:2]
        wd = net.mlp[0].weight.dtype
        onehot = torch.zeros(T, 52, dtype=wd)
        onehot.scatter_(1, board, 1.0)
        h, s = half.double(), stack.double()
        scal = torch.stack([h / (h + s), s / (2 * h)], -1).to(wd)
        with torch.no_grad():
            v = net(ranges.reshape(L * T, 2, -1).to(wd), onehot.repeat(L, 1), scal.repeat_interleave(T, 0),
                    live.repeat(L, 1))
        return v.view(L, T, 2, -1).to(ranges.dtype)

    return f


def _net(width=48, depth=3, seed=0):
    torch.manual_seed(seed)
    net = vn.ValueNet(width, depth).eval()
    with torch.no_grad():
        net.scal_mean.copy_(torch.tensor([0.1, 3.0]))
        net.scal_std.copy_(torch.tensor([0.2, 5.0]))
        for m in net.mlp:
            if isinstance(m, torch.nn.PReLU):
                m.weight.fill_(0.2)
    return net


def _reaches(seed=0) -> np.ndarray:
    """[5, 2, 1176] float64: dense random; sparse; one player empty at a leaf;
    nearly everything on the holdings of one card (both players, two cards);
    nearly everything on two holdings that share a card."""
    rng = np.random.default_rng(seed)
    hold = fs.combos(FLOP)
    n = len(hold)
    r = np.zeros((5, 2, n))
    r[0] = rng.random((2, n))
    r[1] = rng.random((2, n)) * (rng.random((2, n)) < 0.05)
    r[2, 0] = rng.random(n)
    for q, card in ((0, 50), (1, 7)):
        has = (hold == card).any(1)
        r[3, q] = np.where(has, rng.random(n), 1e-7 * rng.random(n))
    r[4, 0] = 1e-9 * rng.random(n)
    r[4, 0, [np.flatnonzero((hold == 30).any(1))[k] for k in (2, 40)]] = [0.6, 0.4]
    r[4, 1] = rng.random(n) * (rng.random(n) < 0.3)
    return r


def _both(net, reach, dtype):
    half, stack = torch.tensor(HALF, dtype=dtype), torch.tensor(STACK, dtype=dtype)
    dense = fs.NetLeaves(FLOP, _dense(net), device="cpu", dtype=dtype)(reach.to(dtype), half, stack)
    fact = fl.FactoredLeaves(FLOP, net, device="cpu", dtype=dtype)
    return dense, fact(reach.to(dtype), half, stack), fact, half, stack


def test_factored_leaves_equal_the_dense_forward_in_float64():
    net = _net().double()
    reach = torch.as_tensor(_reaches())
    dense, got, fact, half, stack = _both(net, reach, torch.float64)
    assert dense.abs().max() > 0
    scale = dense.abs().amax((1, 2), keepdim=True)
    assert ((got - dense).abs() / scale).max() < 1e-10
    for p in (0, 1):  # one player's values alone are the same numbers
        assert torch.equal(fact(reach, half, stack, players=(p,))[:, 0], got[:, p])


def test_net_values_is_the_dense_reference():
    # The test's float64-capable dense evaluator is `net_values` in float32.
    net = _net()
    reach = torch.as_tensor(_reaches(), dtype=torch.float32)
    half, stack = torch.tensor(HALF), torch.tensor(STACK)
    a = fs.NetLeaves(FLOP, fs.net_values(net), device="cpu")(reach, half, stack)
    b = fs.NetLeaves(FLOP, _dense(net), device="cpu")(reach, half, stack)
    assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [0, 1])
def test_float32_error_is_no_worse_than_the_dense_path(seed):
    net = _net(seed=seed)
    reach64 = torch.as_tensor(_reaches(seed))
    reach32 = reach64.float()
    # The oracle sees the float32 inputs exactly, in float64.
    want, _, _, _, _ = _both(_net(seed=seed).double(), reach32.double(), torch.float64)
    dense, got, _, _, _ = _both(net, reach32, torch.float32)
    scale = want.abs().amax((1, 2))
    err_dense = ((dense.double() - want).abs().amax((1, 2)) / scale)
    err_fact = ((got.double() - want).abs().amax((1, 2)) / scale)
    # Per leaf: within a small multiple of the dense path's float32 error
    # (floor: a few float32 ulps of the leaf's largest value).
    assert (err_fact <= 4 * err_dense + 2e-6).all(), (err_fact.tolist(), err_dense.tolist())


def _errors(seed, **kw):
    """Per leaf, against the float64 dense values of the float32 inputs: the
    dense float32 path's error and `FactoredLeaves(**kw)`'s, relative to the
    leaf's largest value."""
    net = _net(seed=seed)
    reach = torch.as_tensor(_reaches(seed)).float()
    half, stack = torch.tensor(HALF), torch.tensor(STACK)
    want, _, _, _, _ = _both(_net(seed=seed).double(), reach.double(), torch.float64)
    dense = fs.NetLeaves(FLOP, fs.net_values(net), device="cpu")(reach, half, stack)
    got = fl.FactoredLeaves(FLOP, net, device="cpu", **kw)(reach, half, stack)
    assert got.dtype == torch.float32
    scale = want.abs().amax((1, 2))
    return [((x.double() - want).abs().amax((1, 2)) / scale) for x in (dense, got)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_float64_last_stage_is_closer_than_the_dense_path(seed):
    err_dense, err_last = _errors(seed, last=torch.float64)
    assert (err_last < err_dense).all(), (err_last.tolist(), err_dense.tolist())
    assert err_last.mean() < 0.6 * err_dense.mean()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eps_in_float64_as_well_is_closer_still(seed):
    # N1c: the zero-sum term's range side in float64 too.
    err_dense, err_b = _errors(seed, last=torch.float64)
    _, err_c = _errors(seed, last=torch.float64, last_eps=True)
    assert (err_c < err_dense).all(), (err_c.tolist(), err_dense.tolist())
    assert err_c.mean() < err_b.mean(), (err_c.tolist(), err_b.tolist())


def test_the_last_stage_option_changes_nothing_in_the_nets_own_dtype():
    reach = torch.as_tensor(_reaches())
    for net, dtype in ((_net().double(), torch.float64), (_net(), torch.float32)):
        half, stack = torch.tensor(HALF, dtype=dtype), torch.tensor(STACK, dtype=dtype)
        a = fl.FactoredLeaves(FLOP, net, device="cpu")(reach.to(dtype), half, stack)
        for kw in (dict(last=dtype), dict(last=dtype, last_eps=True), dict(last_eps=True)):
            assert torch.equal(a, fl.FactoredLeaves(FLOP, net, device="cpu", **kw)(reach.to(dtype), half, stack))
