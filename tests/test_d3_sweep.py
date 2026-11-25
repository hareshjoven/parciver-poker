# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D3-SWEEP (records/d3-sweep/PREREG.md): the recipe's three options (dropout, EMA, range-weighted loss) and the driver.

The golden test comes first: it was written and passed on the code as it stood BEFORE the options existed (commit 7d7dde2),
and it must pass unchanged after them (options off = the old path to the bit). The pins are this machine's CPU and torch
build (torch 2.14.0+cu132); a torch change rebuilds them by amendment, not by editing this file."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import turn_net as tn  # noqa: E402

from parciver import value_net as vn  # noqa: E402,F401


def _boards(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(52, 4, replace=False) for _ in range(n)])


def _tiny(n=48, seed=11):
    rng = np.random.default_rng(seed)
    d = dict(ranges=rng.random((n, 2, 1128)).astype(np.float32),
             values=rng.normal(size=(n, 2, 1128)).astype(np.float32), board=_boards(n, seed),
             half=rng.integers(50, 400, n), stack=rng.integers(1000, 9000, n))
    part = np.where(np.arange(n) % 6 == 0, tn.SELECT, tn.TRAIN).astype(np.int8)
    return d, part


def _digest(state) -> str:
    """sha256 over the state dict: each key, then its tensor's bytes, in sorted key order."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


PINNED_TORCH = "2.14.0+cu132"

# width 32, depth 2, 3 steps (40 train rows, batch 13), CPU, seed 0, options off: pinned on the pre-sweep code.
GOLDEN = {
    0.0: dict(sha="a5ce6b8c09f1b6da7a48388546a7254dcb6c7f044c198a3bc573575e205dc13c",
              train_loss=0.4274941285451253, val_loss=0.4262999892234802, val_mae=79.60484027862549),
    0.1: dict(sha="060ac7afe1fa682b0e6423054d6b6b16e3dc43dc0ee7d0618dc1c90e249fe6f1",
              train_loss=0.4274940490722656, val_loss=0.4262997806072235, val_mae=79.6048104763031),
}


@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_golden_options_off_is_the_pre_sweep_path_to_the_bit(tmp_path, monkeypatch, wd):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    rec = tn.train(d, part, 32, 2, 1.0, 1, 13, 1e-3, 0, "g", log=lambda _: None, device="cpu", wd=wd)
    state = torch.load(tmp_path / "g" / "net.pt", weights_only=False)["state"]
    want, h = GOLDEN[wd], rec["history"][0]
    assert sorted(state) == ["mlp.0.bias", "mlp.0.weight", "mlp.1.weight", "mlp.2.bias", "mlp.2.weight", "mlp.3.weight",
                             "mlp.4.bias", "mlp.4.weight", "scal_mean", "scal_std"]  # no module added to net.mlp
    got = (h["train_loss"], h["val_loss"], h["val_mae"])
    exp = (want["train_loss"], want["val_loss"], want["val_mae"])
    if torch.__version__ == PINNED_TORCH and sys.platform == "win32":
        assert _digest(state) == want["sha"]  # the bit-exact pin, on the machine and build it was taken on
        assert got == exp
    else:  # another torch build or OS rounds CPU float32 differently (CI: 1e-7 relative): same path, loose numbers
        assert got == pytest.approx(exp, rel=1e-4)


# ---------------------------------------------------------------- the options: dropout, EMA, the range-weighted loss

def _batch(n=6, seed=3):
    d, _ = _tiny(n, seed)
    return tn.Batches(d, np.arange(n), torch.device("cpu"))


def _args(b):
    r, v, onehot, scal, live = b.get(torch.arange(b.n), False)
    return (r, onehot, scal, live), v


def _net(p=0.0, width=256, depth=3, seed=0):
    torch.manual_seed(seed)
    net = vn.ValueNet(width, depth)
    net.p = p
    return net


def _count_dropouts(monkeypatch):
    calls = []
    real = torch.nn.functional.dropout

    def spy(x, p=0.5, training=True, inplace=False):
        out = real(x, p, training, inplace)
        calls.append((p, training, float((out == 0).float().mean())))
        return out

    monkeypatch.setattr(torch.nn.functional, "dropout", spy)
    return calls


def test_dropout_adds_no_module_and_no_state():
    plain, dropped = _net(0.0), _net(0.3)
    assert not any(isinstance(m, torch.nn.Dropout) for m in dropped.modules())
    assert len(dropped.mlp) == len(plain.mlp) and sorted(dropped.state_dict()) == sorted(plain.state_dict())
    assert "p" not in dropped.state_dict() and plain.p == 0.0


def test_dropout_is_zero_in_eval_and_inactive_at_p0(monkeypatch):
    calls = _count_dropouts(monkeypatch)
    x, _ = _args(_batch())
    want = _net(0.0).eval()(*x)
    assert torch.equal(_net(0.5).eval()(*x), want) and calls == []  # eval: the old path, no dropout call
    assert torch.equal(_net(0.0).train()(*x), want) and calls == []  # p = 0 in training: the old path too
    assert not torch.equal(_net(0.5).train()(*x), want) and len(calls) == 3  # one per hidden PReLU, none on the input or output


def test_dropout_active_fraction_is_about_p(monkeypatch):
    calls = _count_dropouts(monkeypatch)
    x, _ = _args(_batch(n=24))
    _net(0.3, width=512).train()(*x)
    assert [c[:2] for c in calls] == [(0.3, True)] * 3
    assert all(0.25 < c[2] < 0.35 for c in calls)  # PReLU outputs are almost never exactly 0: the zeros are the mask


def test_dropout_with_the_mask_removed_is_the_old_forward(monkeypatch):
    monkeypatch.setattr(torch.nn.functional, "dropout", lambda x, p=0.5, training=True, inplace=False: x)
    x, _ = _args(_batch())
    assert torch.equal(_net(0.4).train()(*x), _net(0.0).train()(*x))  # the explicit loop applies the same layers


def test_dropout_is_one_rng_stream_with_the_seed():
    x, _ = _args(_batch())
    net = _net(0.3).train()
    torch.manual_seed(5)
    a = net(*x)
    torch.manual_seed(5)
    b = net(*x)
    torch.manual_seed(6)
    c = net(*x)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_a_dropout_net_saved_loads_in_value_net_unchanged(tmp_path):
    net = _net(0.3)
    vn.save(net, tmp_path / "n.pt", meta=dict(dropout=0.3))
    loaded, meta = vn.load(tmp_path / "n.pt", "cpu")
    assert loaded.p == 0.0 and meta["dropout"] == 0.3 and not loaded.training
    x, _ = _args(_batch())
    assert torch.equal(loaded(*x), net.eval()(*x))


def _huber(x):
    return 0.5 * x * x if abs(x) <= 1 else abs(x) - 0.5


def test_loss_rw_is_the_hand_sum():
    rng = np.random.default_rng(0)
    pred, true = (torch.tensor(rng.normal(0, 1.5, (2, 2, 3)), dtype=torch.float32) for _ in range(2))
    r = torch.tensor(rng.random((2, 2, 3)), dtype=torch.float32)
    r = r / r.sum(-1, keepdim=True)
    want = np.mean([sum(float(r[b, p, i]) * _huber(float(pred[b, p, i] - true[b, p, i])) for i in range(3))
                    for b in range(2) for p in range(2)])
    assert float(tn._loss_rw(pred, true, r)) == pytest.approx(want, rel=1e-6)


def test_training_loss_is_plain_or_the_half_half_mix():
    rng = np.random.default_rng(1)
    pred, true = (torch.tensor(rng.normal(0, 1, (3, 2, 7)), dtype=torch.float32) for _ in range(2))
    r = torch.tensor(rng.random((3, 2, 7)), dtype=torch.float32)
    r = r / r.sum(-1, keepdim=True)
    assert torch.equal(tn.training_loss("huber", pred, true, r), tn._loss(pred, true))
    mix = 0.5 * float(tn._loss(pred, true)) + 0.5 * float(tn._loss_rw(pred, true, r))
    assert float(tn.training_loss("rw", pred, true, r)) == pytest.approx(mix, rel=1e-6)


def test_loss_rw_dead_holdings_carry_nothing():
    (r, onehot, scal, live), v = _args(_batch())
    dead = ~live[:, None, :].expand_as(r)
    assert dead.any() and (r[dead] == 0).all()  # the ranges are 0 at the dead holdings
    dirty = v.clone()
    dirty[dead] = 123.0  # garbage where the holding is dead
    assert float(tn._loss_rw(dirty, v, r)) == 0.0
    flat = float(tn._loss_rw(v + 0.5, v, r))  # a flat 0.5 error (Huber 0.125 each) at the live holdings, weighted by r
    assert flat == pytest.approx(0.125 * float(r.sum(-1).mean()), rel=1e-5)


def _spy(monkeypatch):
    """The live net, its weights at the start and after each optimiser step, and the net and weights each evaluation saw."""
    seen = dict(steps=[], scored=[], scored_w=[])
    real_opt, real_eval = tn.optimizer, tn.evaluate

    def optimizer(net, lr, wd):
        opt = real_opt(net, lr, wd)
        seen["net"], seen["w0"] = net, [p.detach().clone() for p in net.parameters()]
        opt.register_step_post_hook(lambda o, a, k: seen["steps"].append([p.detach().clone() for p in net.parameters()]))
        return opt

    def evaluate(net, b, *a, **k):
        seen["scored"].append(net)
        seen["scored_w"].append([p.detach().clone() for p in net.parameters()])
        return real_eval(net, b, *a, **k)

    monkeypatch.setattr(tn, "optimizer", optimizer)
    monkeypatch.setattr(tn, "evaluate", evaluate)
    return seen


def _run(tmp_path, monkeypatch, tag, epochs=1, **opts):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    seen = _spy(monkeypatch)
    rec = tn.train(d, part, 32, 2, 1.0, epochs, 13, 1e-3, 0, tag, log=lambda _: None, device="cpu", **opts)
    return rec, seen, (d, part)


def test_ema_is_the_formula_after_k_steps_and_the_live_weights_keep_training(tmp_path, monkeypatch):
    rec, seen, _ = _run(tmp_path, monkeypatch, "e", ema=0.9)
    assert len(seen["steps"]) == 3  # 40 rows, batch 13
    e = [w.clone() for w in seen["w0"]]  # starts at the initial weights: no bias correction, no warm-up
    for ws in seen["steps"]:
        for ei, wi in zip(e, ws):
            ei.mul_(0.9).add_(wi, alpha=1 - 0.9)
    assert all(torch.equal(a, b) for a, b in zip(seen["scored_w"][0], e))  # the evaluated net is e_3, to the bit
    live = [p.detach() for p in seen["net"].parameters()]
    assert all(torch.equal(a, b) for a, b in zip(live, seen["steps"][-1])) and not all(torch.equal(a, b) for a, b in zip(live, e))
    assert seen["scored"][0] is not seen["net"]
    # EMA changes neither the live trajectory nor the RNG: the live weights are those of the run without it.
    _, plain, _ = _run(tmp_path, monkeypatch, "p")
    assert all(torch.equal(a, b) for s, t in zip(seen["steps"], plain["steps"]) for a, b in zip(s, t))
    # What is saved is the EMA net, buffers as the live net's.
    state = torch.load(tmp_path / "e" / "net.pt", weights_only=False)["state"]
    assert all(torch.equal(state[k], p) for k, p in zip([k for k, _ in seen["net"].named_parameters()], e))
    assert torch.equal(state["scal_mean"], seen["net"].scal_mean) and torch.equal(state["scal_std"], seen["net"].scal_std)
    assert rec["ema"] == 0.9 and rec["dropout"] == 0.0 and rec["loss"] == "huber"


def test_ema_net_is_evaluated_each_epoch_and_dropout_never_reaches_it(tmp_path, monkeypatch):
    _, seen, _ = _run(tmp_path, monkeypatch, "d", epochs=2, ema=0.99, dropout=0.2)
    assert len(seen["scored"]) == 2 and all(n is seen["scored"][0] and n is not seen["net"] for n in seen["scored"])
    assert seen["scored"][0].p == 0.0 and seen["net"].p == 0.2


def test_a_saved_option_net_loads_and_reproduces_the_training_eval_number(tmp_path, monkeypatch):
    rec, _, (d, part) = _run(tmp_path, monkeypatch, "o", epochs=2, ema=0.99, dropout=0.2, loss="rw")
    net, meta = vn.load(tmp_path / "o" / "net.pt", "cpu")
    assert (meta["dropout"], meta["ema"], meta["loss"]) == (0.2, 0.99, "rw") and meta["epoch"] == rec["best"]["epoch"]
    assert net.p == 0.0 and len(net.mlp) == 5
    va = tn.Batches(d, np.flatnonzero(part == tn.SELECT), torch.device("cpu"))
    assert tn.evaluate(net, va)["loss"] == rec["best"]["val_loss"]  # float32, the plain Huber, to the bit on CPU
    out = vn.predict(net, np.array([[0, 1, 2, 3]]), np.ones((1, 2, 1128)), np.array([100]), np.array([2000]))
    assert out.shape == (1, 2, 1128) and np.isfinite(out).all()


def test_rw_changes_training_not_selection(tmp_path, monkeypatch):
    plain, _, _ = _run(tmp_path, monkeypatch, "h")
    rw, _, _ = _run(tmp_path, monkeypatch, "w", loss="rw")
    assert rw["history"][0]["train_loss"] != plain["history"][0]["train_loss"] and rw["loss"] == "rw"
    assert rw["history"][0]["val_loss"] != plain["history"][0]["val_loss"]  # a different net, scored by the same plain Huber


def test_cfg_defaults_keep_old_records_and_options_make_new_ones(tmp_path, monkeypatch):
    assert tn.CFG_DEFAULTS["dropout"] == 0.0 and tn.CFG_DEFAULTS["ema"] == 0.0 and tn.CFG_DEFAULTS["loss"] == "huber"
    rec, _, (d, part) = _run(tmp_path, monkeypatch, "r")
    path = tmp_path / "r" / "record.json"
    old = {k: v for k, v in rec.items() if k not in ("dropout", "ema", "loss")}  # a record from before D3-SWEEP
    path.write_text(json.dumps(old))
    args = (d, part, 32, 2, 1.0, 1, 13, 1e-3, 0, "r")
    assert tn.train(*args, log=lambda _: None, device="cpu", reuse=True) == old  # reused as it stands
    monkeypatch.setattr(tn, "Batches", None)
    for opts in (dict(dropout=0.1), dict(ema=0.9), dict(loss="rw")):
        path.write_text(json.dumps(old))
        with pytest.raises(TypeError):  # not this recipe: the old record goes, the net is trained again (reached sample loading)
            tn.train(*args, log=lambda _: None, device="cpu", reuse=True, **opts)
        assert not path.exists()


def test_bad_options_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    for opts in (dict(dropout=1.0), dict(dropout=-0.1), dict(ema=1.0), dict(ema=-0.5), dict(loss="mse")):
        with pytest.raises(ValueError):
            tn.train(d, part, 16, 1, 1.0, 1, 8, 1e-3, 0, "x", log=lambda _: None, device="cpu", **opts)


def test_stop_after_is_the_first_epochs_of_the_full_run_to_the_bit(tmp_path, monkeypatch):
    full, _, _ = _run(tmp_path, monkeypatch, "full", epochs=4)
    cut, _, _ = _run(tmp_path, monkeypatch, "cut", epochs=4, stop_after=2)
    assert len(full["history"]) == 4 and len(cut["history"]) == 2
    for a, b in zip(full["history"], cut["history"]):  # the schedule is the 4-epoch run's, only stopped early
        assert (a["train_loss"], a["val_loss"], a["val_mae"]) == (b["train_loss"], b["val_loss"], b["val_mae"])
    assert json.loads((tmp_path / "cut" / "record.json").read_text())["history"] == cut["history"]


def test_a_record_cut_by_stop_after_is_never_reused(tmp_path, monkeypatch):
    rec, _, (d, part) = _run(tmp_path, monkeypatch, "c", epochs=3, stop_after=1)
    assert len(rec["history"]) == 1
    monkeypatch.setattr(tn, "Batches", None)
    with pytest.raises(TypeError):  # not reused: it went on to load the samples (the cut record is not the 3-epoch run)
        tn.train(d, part, 32, 2, 1.0, 3, 13, 1e-3, 0, "c", log=lambda _: None, device="cpu", reuse=True)


def test_bad_stop_after_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    for k in (0, 3, -1):
        with pytest.raises(ValueError):
            tn.train(d, part, 16, 1, 1.0, 2, 8, 1e-3, 0, "x", log=lambda _: None, device="cpu", stop_after=k)
