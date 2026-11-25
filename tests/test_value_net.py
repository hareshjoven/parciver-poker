# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The turn-start value net's plumbing (src/parciver/value_net.py; FLOPSEARCH.md
D2): the 1,326-holding layout against the solver's 1,128-holding order, suit
relabelling against `turn_values.holding_map` (itself checked against solves in
G1), the canonical board of the train/validation split, the zero-sum layer and
the error metric."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parciver import turn_values as tv  # noqa: E402
from parciver import value_net as vn  # noqa: E402

BOARDS = [(0, 5, 22, 51), (48, 49, 50, 51), (3, 7, 11, 15), (12, 13, 26, 40)]


def test_expand_places_each_holding_at_its_pair():
    live = vn.live_mask(BOARDS)
    assert live.shape == (4, 1326) and (live.sum(1) == 1128).all()
    x = np.stack([np.stack([np.arange(1128.0) + 1, -np.arange(1128.0) - 1]) for _ in BOARDS])
    y = vn.expand(x, live)
    for b, board in enumerate(BOARDS):
        for h, (c1, c2) in enumerate(tv.combos(board)):
            assert y[b, 0, vn.PAIR_INDEX[c1, c2]] == h + 1 and y[b, 1, vn.PAIR_INDEX[c1, c2]] == -h - 1
        assert (y[b][:, ~live[b]] == 0).all()
    assert np.array_equal(vn.contract(y, live), x)
    t = vn.expand(torch.as_tensor(x), torch.as_tensor(live))
    assert np.array_equal(t.numpy(), y)
    assert np.array_equal(vn.contract(t, torch.as_tensor(live)).numpy(), x)


def test_suit_relabelling_matches_turn_values():
    rng = np.random.default_rng(3)
    for board in BOARDS:
        data = rng.random((1, 2, 1128))
        live = vn.live_mask([board])
        big = torch.as_tensor(vn.expand(data, live))
        onehot = torch.zeros(1, 52, dtype=torch.float64)
        onehot[0, list(board)] = 1
        for k in range(24):
            new_board, idx = tv.holding_map(board, vn.SUIT_PERMS[k])
            r, v, b, lv = vn.permute(torch.tensor([k]), big, 2 * big, onehot, torch.as_tensor(live))
            assert sorted(np.flatnonzero(b[0].numpy())) == sorted(new_board)
            assert np.array_equal(lv.numpy(), vn.live_mask([new_board]))
            assert np.array_equal(vn.contract(r, lv).numpy(), data[..., idx])
            assert np.array_equal(vn.contract(v, lv).numpy(), 2 * data[..., idx])


def test_canonical_board_is_the_suit_class():
    for board in BOARDS:
        c = vn.canonical_board(board)
        for k in range(24):
            assert vn.canonical_board(vn.CARD_PERM[k, list(board)]) == c
    # A monotone board and a rainbow one with the same ranks are different classes.
    assert vn.canonical_board((0, 4, 8, 12)) != vn.canonical_board((0, 5, 10, 15))
    assert vn.canonical_board((0, 5, 10, 15)) == vn.canonical_board((1, 6, 11, 12))


def test_zero_sum_layer_is_exact_and_masks_dead_holdings():
    torch.manual_seed(0)
    net = vn.ValueNet(64, 2).double()
    live = torch.as_tensor(vn.live_mask(BOARDS))
    r = torch.rand(4, 2, 1128, dtype=torch.float64) ** 4
    r = vn.expand(r / r.sum(-1, keepdim=True), live)
    onehot = torch.zeros(4, 52, dtype=torch.float64)
    onehot.scatter_(1, torch.tensor(BOARDS), 1.0)
    scal = torch.as_tensor(vn.scalars([100, 500, 2000, 8000], [9900, 5000, 18000, 1900]))
    v = net(r, onehot, scal, live)
    assert (v[~live[:, None, :].expand(-1, 2, -1)] == 0).all()
    assert torch.allclose((r * v).sum((1, 2)), torch.zeros(4, dtype=torch.float64), atol=1e-13)


def test_output_layer_is_float32_under_autocast():
    """Under bfloat16 autocast the hidden layers run in bfloat16 and the output
    layer in float32 (FLOPSEARCH.md D2 amendment); without autocast the forward
    is the plain MLP's, bit for bit."""
    torch.manual_seed(2)
    net = vn.ValueNet(64, 2)
    live = torch.as_tensor(vn.live_mask(BOARDS))
    r = torch.rand(4, 2, 1128) ** 4
    r = vn.expand(r / r.sum(-1, keepdim=True), live)
    onehot = torch.zeros(4, 52)
    onehot.scatter_(1, torch.tensor(BOARDS), 1.0)
    scal = torch.as_tensor(vn.scalars([100, 500, 2000, 8000], [9900, 5000, 18000, 1900]), dtype=torch.float32)
    x = torch.cat([r.flatten(1), onehot, (scal - net.scal_mean) / net.scal_std], 1)
    lv = live[:, None, :].float()
    v = net.mlp(x).view(-1, 2, vn.HOLDINGS) * lv
    assert torch.equal(net(r, onehot, scal, live), v - 0.5 * (r * v).sum((1, 2))[:, None, None] * lv)
    seen = []
    for layer in (net.mlp[0], net.mlp[-1]):
        layer.register_forward_hook(lambda m, i, o: seen.append((i[0].dtype, o.dtype)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(r, onehot, scal, live)
    assert seen == [(torch.float32, torch.bfloat16), (torch.float32, torch.float32)]
    assert out.dtype == torch.float32


def test_predict_is_scale_free_and_in_the_solver_order():
    torch.manual_seed(1)
    net = vn.ValueNet(32, 1)
    rng = np.random.default_rng(0)
    ranges = rng.random((4, 2, 1128))
    a = vn.predict(net, BOARDS, ranges, [300, 300, 1000, 5000], [9700, 1000, 3000, 5000])
    b = vn.predict(net, BOARDS, 7.0 * ranges, [300, 300, 1000, 5000], [9700, 1000, 3000, 5000])
    assert a.shape == (4, 2, 1128)
    assert np.allclose(a, b, atol=1e-5)
    w = ranges / ranges.sum(-1, keepdims=True)
    assert np.allclose((w * a).sum((1, 2)), 0, atol=1e-5)


def test_range_mae():
    rng = np.random.default_rng(2)
    p, t, r = rng.normal(size=(3, 5, 2, 1128))
    r = np.abs(r)
    want = np.array([np.mean([np.sum(r[i, q] * np.abs(p[i, q] - t[i, q])) / np.sum(r[i, q]) for q in (0, 1)])
                     for i in range(5)])
    assert np.allclose(vn.range_mae(p, t, r), want)
    assert np.allclose(vn.range_mae(torch.as_tensor(p), torch.as_tensor(t), torch.as_tensor(r)).numpy(), want)


def test_scalars():
    s = vn.scalars([100, 1000], [19900, 1000])
    assert np.allclose(s, [[100 / 20000, 19900 / 200], [0.5, 0.5]])
