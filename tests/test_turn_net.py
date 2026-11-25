# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The value net's training script (scripts/turn_net.py; FLOPSEARCH.md D2): the
board split behind the selection / gate protocol, and the loss."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parciver import value_net as vn  # noqa: E402

_spec = importlib.util.spec_from_file_location("turn_net", Path(__file__).resolve().parents[1] / "scripts" / "turn_net.py")
tn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tn)


def _boards(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(52, 4, replace=False) for _ in range(n)])


def test_canonical_codes_are_the_suit_classes():
    boards = _boards(300, 0)
    codes = tn.canonical_codes(boards)
    classes = [vn.canonical_board(b) for b in boards]
    for i in range(len(boards)):
        for j in range(i + 1, len(boards)):
            assert (codes[i] == codes[j]) == (classes[i] == classes[j])


def test_split_is_a_property_of_the_suit_class():
    boards = _boards(200, 1)
    part = tn.split(boards)
    for k in range(24):
        assert np.array_equal(tn.split(vn.CARD_PERM[k][boards]), part)
    # Row-wise: any subset gets the same parts.
    assert np.array_equal(tn.split(boards[::3]), part[::3])


def test_split_fractions():
    part = tn.split(_boards(200_000, 2))
    n = len(part)
    for p, want in ((tn.SELECT, 0.025), (tn.GATE, 0.025)):
        # Boards repeat within a class, so allow well over binomial noise.
        assert abs((part == p).mean() - want) < 6 * np.sqrt(want / n) + 0.003
    assert set(np.unique(part)) == {tn.TRAIN, tn.SELECT, tn.GATE}


def test_batches_store_ranges_scaled_in_float16():
    """Ranges come back as float16(r * 2^14) / 2^14: within float16's relative
    step at every entry, the small ones included (unscaled, entries under 2^-14
    would be subnormal, in absolute steps of 2^-24)."""
    rng = np.random.default_rng(5)
    boards = _boards(6, 6)
    r = 10.0 ** rng.uniform(-4.5, 0, (6, 2, 1128))  # normalized: ~3e-7 to ~0.01, half under 2^-14
    d = dict(ranges=r, values=np.zeros((6, 2, 1128), np.float32), board=boards,
             half=np.full(6, 100.0), stack=np.full(6, 1000.0))
    b = tn.Batches(d, np.arange(6), "cpu")
    assert b.ranges.dtype == torch.float16
    got, _, _, _, live = b.get(torch.arange(6), False)
    want = vn.expand(torch.as_tensor(r / r.sum(-1, keepdims=True)), live)
    exact = torch.as_tensor(((r / r.sum(-1, keepdims=True)) * tn.RANGE_SCALE).astype(np.float16).astype(np.float64))
    assert torch.equal(got.double(), vn.expand(exact, live) / tn.RANGE_SCALE)
    small = (want > 0) & (want < 2.0 ** -14)
    assert small.sum() > 1000
    assert ((got.double() - want).abs() <= 2.0 ** -11 * want).all()


def test_loss_is_the_huber_mean_over_live_holdings():
    rng = np.random.default_rng(3)
    boards = _boards(3, 4)
    live = torch.as_tensor(vn.live_mask(boards))
    pred = vn.expand(torch.as_tensor(rng.normal(size=(3, 2, 1128)) * 2), live)
    true = vn.expand(torch.as_tensor(rng.normal(size=(3, 2, 1128))), live)
    m = live[:, None, :].expand(-1, 2, -1)
    want = torch.nn.functional.huber_loss(pred[m], true[m])
    assert torch.allclose(tn._loss(pred, true), want, rtol=1e-12)


def test_batches_normalize_in_chunks_bit_for_bit(monkeypatch):
    rng = np.random.default_rng(7)
    r = (10.0 ** rng.uniform(-4.5, 0, (11, 2, 1128))).astype(np.float32)  # as the shards store them
    d = dict(ranges=r, values=np.zeros((11, 2, 1128), np.float32), board=_boards(11, 8),
             half=np.full(11, 100.0), stack=np.full(11, 1000.0))
    rows = np.array([9, 0, 4, 10, 2, 7, 1])
    x = r[rows].astype(np.float64)
    want = ((x / x.sum(-1, keepdims=True)) * tn.RANGE_SCALE).astype(np.float16)
    for chunk in (3, 7, 16384):
        monkeypatch.setattr(tn, "CHUNK", chunk)
        assert np.array_equal(tn.Batches(d, rows, "cpu").ranges.numpy(), want)


def _shards(path, n_shards=3, m=5, seed=0):
    """`turn_data.py generate`'s files, small: shards of `m` samples and their situations."""
    rng = np.random.default_rng(seed)
    n = n_shards * m
    np.savez(path / "situations.npz", board=_boards(n, seed).astype(np.int8), half=rng.integers(50, 400, n),
             stack=rng.integers(1000, 9000, n), cell=rng.integers(0, 9, n))
    cols = []
    for s in range(n_shards):
        c = dict(index=rng.permutation(n)[:m], source=rng.integers(0, 2, m),
                 values=rng.normal(size=(m, 2, 1128)).astype(np.float32),
                 ranges=rng.random((m, 2, 1128)).astype(np.float32), expl=rng.random(m), game=rng.random(m))
        np.savez(path / f"shard-{s:05d}.npz", **c)
        cols.append(c)
    return cols


def test_load_is_the_shards_in_order_and_refuses_a_bad_one(tmp_path):
    cols = _shards(tmp_path)
    np.savez(tmp_path / "shard-00009.tmp.npz", index=np.zeros(1))  # an unfinished shard is not read
    d = tn.load(tmp_path)
    sit = np.load(tmp_path / "situations.npz")
    for k in ("index", "source", "values", "ranges", "expl"):
        want = np.concatenate([c[k] for c in cols])
        assert d[k].dtype == want.dtype and np.array_equal(d[k], want)
    assert np.array_equal(d["board"], sit["board"][d["index"]]) and d["board"].dtype == np.int64
    assert np.array_equal(d["half"], sit["half"][d["index"]])
    assert len(tn.load(tmp_path, max_shards=2)["index"]) == 10

    bad = dict(cols[1])
    for k, spoil in (("values", np.nan), ("values", np.inf), ("ranges", np.nan), ("ranges", -1.0)):
        x = bad[k].copy()
        x[3, 1, 17] = spoil
        np.savez(tmp_path / "shard-00001.npz", **dict(bad, **{k: x}))
        with pytest.raises(ValueError, match="shard-00001.npz"):
            tn.load(tmp_path)
    x = bad["ranges"].copy()
    x[2, 0] = 0.0  # a player with no mass: 0 / 0 in the normalization
    np.savez(tmp_path / "shard-00001.npz", **dict(bad, ranges=x))
    with pytest.raises(ValueError, match="shard-00001.npz"):
        tn.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        tn.load(tmp_path / "nothing-here")


def test_load_patches_the_d1_resolves_over_the_rows_they_resolved(tmp_path):
    """records/d1-tail/PREREG15.md: the kept re-solves replace their rows' values
    and expl, and only theirs; a patch that does not fit the shards stops the load."""
    cols = _shards(tmp_path)
    perm = np.random.default_rng(5).permutation(15)
    for s, c in enumerate(cols):  # (index, source) unique, as in D1, and not in the rows' order
        c["index"] = perm[5 * s:5 * s + 5]
        np.savez(tmp_path / f"shard-{s:05d}.npz", **c)
    plain = tn.load(tmp_path)
    assert not np.array_equal(np.argsort(plain["index"] * 2 + plain["source"]), np.arange(15))
    ix = {r: int(plain["index"][r]) for r in (2, 7, 13)}  # the index of row r
    rng = np.random.default_rng(1)
    pdir = tmp_path / "resolve-q3"
    pdir.mkdir()

    def write(name, rows, n=None, **spoil):
        rows = np.asarray(rows, np.int64)
        p = dict(index=plain["index"][rows], source=plain["source"][rows],
                 values=rng.normal(size=(len(rows), 2, 1128)).astype(np.float32), game=rng.random(len(rows)),
                 expl=plain["expl"][rows] / 2, ranges=plain["ranges"][rows], e_stored=plain["expl"][rows])
        p.update(spoil)
        np.savez(pdir / f"{name}.npz", **p)
        (pdir / f"{name}.json").write_text(json.dumps({"rows": len(rows) if n is None else n, "head": "abc"}))
        return p

    def load(**kw):
        lines = []
        return tn.load(tmp_path, log=lines.append, **kw), lines

    main, ext = write("patch", [7, 2, 13]), write("patch-extend", [0])
    d, lines = load()
    rows = [7, 2, 13, 0]
    for k in ("values", "expl"):
        want = plain[k].copy()
        want[rows] = np.concatenate([main[k], ext[k]])
        assert np.array_equal(d[k], want) and d[k].dtype == plain[k].dtype, k
    for k in ("index", "source", "ranges", "board", "half", "stack"):
        assert np.array_equal(d[k], plain[k]), k
    assert lines == ["patch-extend.npz: 1 of its 1 re-solves patched over the shards (commit abc)",
                     "patch.npz: 3 of its 3 re-solves patched over the shards (commit abc)"]
    # Two shards loaded: row 13 (shard 2) is not there, and is skipped.
    d, lines = load(max_shards=2)
    assert np.array_equal(d["values"][[7, 2, 0]], np.concatenate([main["values"][:2], ext["values"]]))
    assert "patch.npz: 2 of its 3" in lines[1]
    assert len(load(max_shards=3)[1]) == 2  # all the shards: complete

    other = {"index": main["index"].copy(), "source": main["source"].copy()}
    other["source"][1] = 1 - other["source"][1]  # (2, other source): not in the shards
    spoiled_e, spoiled_r, not_better, nan = (main["e_stored"].copy(), main["ranges"].copy(),
                                             main["expl"].copy(), main["values"].copy())
    spoiled_e[2] = np.nextafter(spoiled_e[2], 1.0)
    spoiled_r[0, 1, 5] += 1.0
    not_better[1] = main["e_stored"][1]
    nan[2, 0, 9] = np.nan
    for spoil, match in (({"index": main["index"][[0, 0, 2]], "source": main["source"][[0, 0, 2]]},
                          rf"row \({ix[7]}, \d\) is patched twice"),
                         (other, rf"row \({ix[2]}, \d\) is not in the shards"),
                         ({"e_stored": spoiled_e}, rf"row \({ix[13]}, \d\) is not the row it re-solved"),
                         ({"ranges": spoiled_r}, rf"row \({ix[7]}, \d\) is not the row it re-solved"),
                         ({"expl": not_better}, rf"row \({ix[2]}, \d\): e .* does not beat"),
                         ({"values": nan}, rf"row \({ix[13]}, \d\): .* non-finite"),
                         ({"values": main["values"].astype(np.float64)}, "values float64")):
        write("patch", [7, 2, 13], **{k: v for k, v in main.items() if k not in spoil}, **spoil)
        with pytest.raises(ValueError, match=match):
            load()
    write("patch", [7, 2, 13], **main)
    write("patch-extend", [7])  # a row in both patches
    with pytest.raises(ValueError, match=rf"patch.npz: row \({ix[7]}, \d\) is patched twice"):
        load()
    write("patch", [7, 2, 13], n=4)
    with pytest.raises(ValueError, match="3 rows, its record says 4"):
        load()
    write("patch", [7, 2, 13], **main)
    write("patch-extend", [0], **ext)
    np.savez(pdir / "patch-more.npz", **main)  # no record: torn, or not a patch of this protocol
    with pytest.raises(ValueError, match="patch-more.npz: no patch-more.json"):
        load()
    (pdir / "patch-more.npz").unlink()
    write("patch-extend", [], values=np.zeros((0, 2, 1128), np.float32))  # a patch with no kept rows
    assert load()[1][0] == "patch-extend.npz: 0 of its 0 re-solves patched over the shards (commit abc)"


def test_g2_pairs_cached_checkdown_values_by_their_inputs(tmp_path):
    _shards(tmp_path)
    d = tn.load(tmp_path)
    d["source"] = np.arange(15)  # unique (index, source) keys, as turn_data writes them
    rows = np.arange(0, 15, 2)
    vals = np.random.default_rng(1).random((len(rows), 2, 1128)).astype(np.float32)
    p = tmp_path / "checkdown-val.npz"  # as `baseline` writes it

    def cached(d, rows, **cols):
        np.savez(p, **(cols or dict(index=d["index"][rows], source=d["source"][rows], values=vals,
                                     inputs=tn.row_digests(d, rows))))
        with np.load(p) as cd:
            return tn.cached_checkdown(cd, d, rows)

    cached(d, rows)
    with np.load(p) as cd:
        base, why = tn.cached_checkdown(cd, d, rows[::-1])  # any order, any subset
        assert np.array_equal(base, vals[::-1]) and why == ""
        # More shards move the rows: the same values (and new values are not inputs).
        more = {k: np.concatenate([x[:3] + 100 if k == "index" else x[:3], x]) for k, x in d.items()}
        base, _ = tn.cached_checkdown(cd, dict(more, values=more["values"] + 1), rows + 3)
        assert np.array_equal(base, vals)
        # A row generated again under its (index, source), with any input changed: refused.
        for k in ("ranges", "board", "half", "stack"):
            x = d[k].copy()
            x[rows[1]] = (x[rows[1]] + 1) % 52 if k == "board" else x[rows[1]] + 1
            base, why = tn.cached_checkdown(cd, dict(d, **{k: x}), rows)
            assert base is None and why.startswith("1 gate samples have other inputs"), k
        base, why = tn.cached_checkdown(cd, d, np.append(rows, 1))  # a row not cached
        assert base is None and why.startswith("1 gate samples have no cached")
    # A cache written before the digests: refused, not trusted.
    base, why = cached(d, rows, index=d["index"][rows], source=d["source"][rows], values=vals)
    assert base is None and "predate" in why


def _tiny(n=48, seed=11):
    rng = np.random.default_rng(seed)
    d = dict(ranges=rng.random((n, 2, 1128)).astype(np.float32),
             values=rng.normal(size=(n, 2, 1128)).astype(np.float32), board=_boards(n, seed),
             half=rng.integers(50, 400, n), stack=rng.integers(1000, 9000, n))
    part = np.where(np.arange(n) % 6 == 0, tn.SELECT, tn.TRAIN).astype(np.int8)
    return d, part


def test_train_leaves_a_record_every_epoch_and_reuses_a_finished_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    out, seen = tmp_path / "t", []

    def log(line):
        if "epoch" in line and (out / "record.json").exists():  # called before this epoch's record is written
            seen.append(len(json.loads((out / "record.json").read_text())["history"]))

    args = (d, part, 16, 1, 1.0, 3, 8, 1e-3, 0, "t")
    rec = tn.train(*args, log=log, device="cpu")
    assert seen == [1, 2]  # the record on disk during epochs 2 and 3
    assert sorted(p.name for p in out.iterdir()) == ["net.pt", "record.json"]  # no temporary file left
    assert json.loads((out / "record.json").read_text()) == rec and len(rec["history"]) == 3
    assert rec["train_samples"] == 40 and rec["val_samples"] == 8
    net, meta = vn.load(out / "net.pt", "cpu")
    assert meta["epoch"] == rec["best"]["epoch"] and (net.width, net.depth) == (16, 1)

    # A finished run under the same settings is reused: no training, the files untouched.
    before = (out / "net.pt").read_bytes()
    monkeypatch.setattr(tn, "Batches", None)
    assert tn.train(*args, log=lambda _: None, device="cpu", reuse=True) == rec
    assert (out / "net.pt").read_bytes() == before
    # Other settings, or other data (other counts, or the same counts of other
    # values), are not: the old record goes and the net is trained again.
    changed = dict(d, values=d["values"] + 1.0)
    for other in ((d, part, 16, 1, 1.0, 2, 8, 1e-3, 0, "t"), (d, np.where(part == tn.SELECT, tn.GATE, part), *args[2:]),
                  (changed, *args[1:])):
        (out / "record.json").write_text(json.dumps(rec))
        with pytest.raises(TypeError):  # reached the (patched out) sample loading
            tn.train(*other, log=lambda _: None, device="cpu", reuse=True)
        assert not (out / "record.json").exists()
    assert tn.fingerprint(changed) != tn.fingerprint(d) == rec["data"]


def test_a_sweeps_nets_share_the_samples_on_the_device(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    built, real = [], tn.Batches

    def counting(*a):
        built.append(1)
        return real(*a)

    monkeypatch.setattr(tn, "Batches", counting)
    held = {}
    tn.train(d, part, 16, 1, 1.0, 1, 8, 1e-3, 0, "a", log=lambda _: None, device="cpu", held=held)
    tn.train(d, part, 8, 2, 1.0, 1, 8, 1e-3, 0, "b", log=lambda _: None, device="cpu", held=held)
    assert len(built) == 2  # the training and the selection samples, once
    # Sharing them changes nothing: the same nets as ones trained on their own
    # (b is the one that was handed a's samples).
    tn.train(d, part, 16, 1, 1.0, 1, 8, 1e-3, 0, "c", log=lambda _: None, device="cpu")
    tn.train(d, part, 8, 2, 1.0, 1, 8, 1e-3, 0, "d", log=lambda _: None, device="cpu")
    assert len(built) == 6
    for shared, alone in (("a", "c"), ("b", "d")):
        s, t = (torch.load(tmp_path / x / "net.pt", weights_only=False)["state"] for x in (shared, alone))
        assert all(torch.equal(s[k], t[k]) for k in s)
        r, q = (json.loads((tmp_path / x / "record.json").read_text())["history"] for x in (shared, alone))
        assert [{k: v for k, v in e.items() if k != "seconds"} for e in r] == [{k: v for k, v in e.items() if k != "seconds"} for e in q]
    tn.train(d, part, 16, 1, 0.5, 1, 8, 1e-3, 0, "e", log=lambda _: None, device="cpu", held=held)
    assert len(built) == 8 and held["tr"].n == 20  # another fraction: its own samples, the old ones dropped


def test_train_stops_on_a_non_finite_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    real, calls = tn.evaluate, []

    def evaluate(net, b):
        calls.append(1)
        return dict(real(net, b), loss=float("nan")) if len(calls) == 2 else real(net, b)

    monkeypatch.setattr(tn, "evaluate", evaluate)
    with pytest.raises(FloatingPointError, match="epoch 2"):
        tn.train(d, part, 16, 1, 1.0, 3, 8, 1e-3, 0, "n", log=lambda _: None, device="cpu")
    rec = json.loads((tmp_path / "n" / "record.json").read_text())
    assert len(rec["history"]) == 1 and rec["best"]["epoch"] == 1  # epoch 1's net and record are what is left


def test_the_sweeps_choice_is_refused_on_other_data(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    d, part = _tiny()
    best = dict(epoch=1, val_loss=0.5, val_mae=1.0)
    sweep = dict(tag="s", width=16, depth=1, train_samples=40, val_samples=8, data=tn.fingerprint(d), epochs=1, best=best)
    (tmp_path / "sweep.json").write_text(json.dumps([sweep]))
    assert tn._chosen(d, part)["tag"] == "s"
    with pytest.raises(SystemExit, match="other data"):  # other counts
        tn._chosen(d, part[:24])
    with pytest.raises(SystemExit, match="other data"):  # the same counts of other samples
        tn._chosen(dict(d, ranges=d["ranges"][::-1].copy()), part)
    # A sweep.json that did not record its data (a preview's) is refused too.
    (tmp_path / "sweep.json").write_text(json.dumps([{k: v for k, v in sweep.items() if k != "data"}]))
    with pytest.raises(SystemExit, match="other data"):
        tn._chosen(d, part)
