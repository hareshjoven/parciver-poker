# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""scripts/d3_load.py (records/d3/PREREG.md 5.0.3): the memory-mapped loader gives `turn_net.load_many`'s data, bit for
bit, on toys: the dict, its patches, its fingerprint, its views, the nets trained on it (CPU and GPU), and a mixed D3
folder's harvested rows. A mapped column is never turned into one array."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parciver import value_net as vn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import d3_load as dl  # noqa: E402
import turn_data as td  # noqa: E402
import turn_net as tn  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="needs the GPU"))]
quiet = lambda *_: None  # noqa: E731


def _boards(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(52, 4, replace=False) for _ in range(n)], dtype=np.int64)


def _toy(n: int = 96, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return dict(values=rng.normal(size=(n, 2, vn.LIVE)).astype(np.float32),
                ranges=rng.random((n, 2, vn.LIVE)).astype(np.float32),
                board=_boards(n, seed + 1), half=rng.uniform(100, 2000, n), stack=rng.uniform(0, 10000, n),
                source=rng.integers(0, 2, n), expl=rng.uniform(0, 1, n).astype(np.float32),
                index=np.arange(n).astype(np.int32))


def _folder(path: Path, d: dict, shard_rows: int = 32, save=np.savez) -> Path:
    path.mkdir(parents=True)
    n = len(d["index"])
    np.savez(path / "situations.npz", board=d["board"].astype(np.int8), half=d["half"], stack=d["stack"], cell=np.zeros(n, np.int32))
    for k, s in enumerate(range(0, n, shard_rows)):
        save(path / f"shard-{k:04d}.npz", **{x: d[x][s:s + shard_rows] for x in ("index", "source", "values", "ranges", "expl")})
    return path


def _patch(path: Path, d: dict, rows: list[int]) -> None:
    (path / "resolve-q3").mkdir()
    np.savez(path / "resolve-q3" / "patch.npz", index=d["index"][rows], source=d["source"][rows], ranges=d["ranges"][rows],
             values=d["values"][rows] + 1, expl=d["expl"][rows] / 2, e_stored=d["expl"][rows])
    (path / "resolve-q3" / "patch.json").write_text(json.dumps(dict(rows=len(rows), head="h")))


def _two(tmp_path, n1=70, n2=45):
    a, b = _toy(n1, 0), _toy(n2, 7)
    pa, pb = _folder(tmp_path / "a", a), _folder(tmp_path / "b", b, shard_rows=20)
    _patch(pa, a, [3, 40, 69])
    return [pa, pb]


def _same(mapped: dict, eager: dict) -> None:
    assert sorted(mapped) == sorted(eager)
    for k, want in eager.items():
        got = mapped[k]
        if hasattr(got, "chunks"):
            assert got.dtype == want.dtype and got.shape == want.shape, k
            assert np.array_equal(got[np.arange(len(want))], want), k
        else:
            assert got.dtype == want.dtype and np.array_equal(got, want), k


# ---------------------------------------------------------------- the same data


def test_mapped_equals_eager_for_one_and_two_folders_with_patches(tmp_path):
    dirs = _two(tmp_path)
    for use in (dirs[:1], dirs):
        eager, mapped = tn.load_many(use, log=quiet), dl.load_mapped(use, log=quiet)
        _same(mapped, eager)
        assert tn.fingerprint(mapped) == tn.fingerprint(eager)
        assert ("dataset" in mapped) == (len(use) > 1)
    assert np.array_equal(mapped["values"][3], np.load(dirs[0] / "resolve-q3" / "patch.npz")["values"][0])


def test_the_patch_lines_and_max_shards_are_the_eager_ones(tmp_path):
    dirs = _two(tmp_path)
    lines_e, lines_m = [], []
    tn.load_many(dirs, log=lines_e.append)
    dl.load_mapped(dirs, log=lines_m.append)
    assert lines_e == lines_m and len(lines_m) == 1
    few_e, few_m = tn.load_many(dirs, max_shards=1, log=quiet), dl.load_mapped(dirs, max_shards=1, log=quiet)
    _same(few_m, few_e)  # the patch's rows 40 and 69 are in shards not loaded: skipped, as eager skips them
    assert tn.fingerprint(few_m) == tn.fingerprint(few_e)


def test_max_shards_may_be_one_cap_per_folder(tmp_path):
    dirs = _two(tmp_path)  # a: 3 shards (70 rows), b: 3 shards (45 rows)
    all_a = dl.load_mapped(dirs[:1], log=quiet)
    d = dl.load_mapped(dirs, max_shards=[None, 1], log=quiet)
    assert len(d["index"]) == 70 + 20 and d["dataset"].tolist() == [0] * 70 + [1] * 20
    assert np.array_equal(d["values"][np.arange(70)], all_a["values"][np.arange(70)])
    d = dl.load_mapped(dirs, max_shards=(2, None), log=quiet)
    assert len(d["index"]) == 64 + 45
    assert len(dl.load_mapped(dirs, max_shards=2, log=quiet)["index"]) == 64 + 40
    with pytest.raises(ValueError, match="2 entries for 1 folders"):
        dl.load_mapped(dirs[:1], max_shards=[1, 2], log=quiet)


def test_views_slice_a_mapped_dict_like_an_eager_one_and_share_its_patches(tmp_path):
    dirs = _two(tmp_path)
    eager, mapped = tn.load_many(dirs, log=quiet), dl.load_mapped(dirs, log=quiet)
    ve, vm = tn.d2b_views(eager), tn.d2b_views(mapped)
    for name in ("d1", "d1b"):
        _same(vm[name], ve[name])
        assert tn.fingerprint(vm[name]) == tn.fingerprint(ve[name])
    assert np.array_equal(vm["d1"]["values"][3], eager["values"][3])


def test_a_refused_row_stops_the_load_as_it_stops_the_eager_one(tmp_path):
    d = _toy(40)
    d["ranges"][5, 1] = 0.0  # a range of mass 0
    path = _folder(tmp_path / "bad", d)
    with pytest.raises(ValueError, match="mass 0"):
        tn.load_many([path], log=quiet)
    with pytest.raises(ValueError, match="mass 0"):
        dl.load_mapped([path], log=quiet)
    d = _toy(40)
    d["values"][33, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dl.load_mapped([_folder(tmp_path / "nan", d)], log=quiet)


def test_a_compressed_shard_is_read_not_mapped(tmp_path):
    d = _toy(40)
    path = _folder(tmp_path / "z", d, save=np.savez_compressed)
    z = dl.npz_members(path / "shard-0000.npz", ("values",))
    assert not isinstance(z["values"], np.memmap)
    _same(dl.load_mapped([path], log=quiet), tn.load_many([path], log=quiet))
    path = _folder(tmp_path / "s", d)
    assert isinstance(dl.npz_members(path / "shard-0000.npz", ("values",))["values"], np.memmap)


# ---------------------------------------------------------------- the column


def _col(tmp_path, n=100, shard=32):
    d = _toy(n)
    path = _folder(tmp_path / "c", d, shard_rows=shard)
    mapped = dl.load_mapped([path], log=quiet)
    return mapped["values"], d["values"], mapped


def test_a_column_indexes_like_its_array(tmp_path):
    col, want, _ = _col(tmp_path)
    rng = np.random.default_rng(3)
    for rows in (np.arange(100), rng.permutation(100)[:37], np.array([99, 0, 31, 32, 32, 63, 64]), np.array([], np.int64), np.array([-1, -100])):
        assert np.array_equal(col[rows], want[rows]) and col[rows].dtype == want.dtype
    mask = rng.random(100) < 0.3
    assert np.array_equal(col[mask], want[mask])
    assert np.array_equal(col[5], want[5]) and np.array_equal(col[-1], want[-1]) and np.array_equal(col[np.int64(40)], want[40])
    assert len(col) == 100 and col.shape == want.shape and col.ndim == want.ndim and col.dtype == want.dtype
    for bad in (100, -101):
        with pytest.raises(IndexError):
            col[bad]
    with pytest.raises(IndexError):
        col[np.array([0, 100])]
    with pytest.raises(TypeError):
        col[np.array([0.5])]


def test_slices_are_views_with_the_same_rows_and_overlay(tmp_path):
    col, want, _ = _col(tmp_path)
    sub = col[30:70]
    assert sub.shape == want[30:70].shape and np.array_equal(sub[np.arange(40)], want[30:70])
    assert np.array_equal(sub[np.array([2, 33, 39])], want[30:70][[2, 33, 39]])
    assert np.array_equal(np.concatenate(list(sub.chunks())), want[30:70])
    new = np.full(want.shape[1:], 7.0, np.float32)
    col[40] = new  # the overlay is shared: the slice sees it at 40 - 30
    assert np.array_equal(sub[10], new) and np.array_equal(col[40], new) and np.array_equal(col[np.array([39, 40, 41])][1], new)
    assert np.array_equal(np.concatenate(list(sub.chunks()))[10], new) and np.array_equal(np.concatenate(list(col.chunks()))[40], new)
    assert len(col[100:]) == 0 and len(col[50:10]) == 0 and list(col[100:].chunks()) == []
    with pytest.raises(IndexError):
        col[::2]


def test_a_column_is_never_one_array_and_replaces_one_row_at_a_time(tmp_path):
    col, _, _ = _col(tmp_path)
    with pytest.raises(TypeError, match="never turned into one array"):
        np.asarray(col)
    with pytest.raises(TypeError):
        col[np.array([1, 2])] = 0
    with pytest.raises(ValueError, match="shape"):
        col[3] = np.zeros(5, np.float32)
    with pytest.raises(IndexError):
        col[100] = np.zeros(col.shape[1:], np.float32)


def test_chunks_are_the_arrays_bytes_in_order(tmp_path):
    col, want, _ = _col(tmp_path, n=100, shard=32)
    assert [len(c) for c in col.chunks()] == [32, 32, 32, 4]
    assert np.concatenate(list(col.chunks())).tobytes() == want.tobytes()


# ---------------------------------------------------------------- training on it


def _part(n: int) -> np.ndarray:
    part = np.full(n, tn.TRAIN)
    part[::6] = tn.SELECT
    return part


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pools", [1, 2, 3])
def test_a_net_trained_on_the_mapped_data_is_the_same_net_bit_for_bit(tmp_path, monkeypatch, pools, device):
    monkeypatch.setattr(tn, "NETS", tmp_path)
    dirs = _two(tmp_path, 70, 45)
    eager, mapped = tn.load_many(dirs, log=quiet), dl.load_mapped(dirs, log=quiet)
    part = _part(len(eager["index"]))
    a = tn.train(eager, part, 16, 1, 1.0, 3, 8, 1e-3, 0, "eager", log=quiet, device=device, pools=pools, wd=0.1)
    b = tn.train(mapped, part, 16, 1, 1.0, 3, 8, 1e-3, 0, "mapped", log=quiet, device=device, pools=pools, wd=0.1)
    strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731
    assert a["data"] == b["data"] and strip(a["history"]) == strip(b["history"]) and a["best"]["epoch"] == b["best"]["epoch"]
    sa = torch.load(tmp_path / "eager" / "net.pt", weights_only=False)["state"]
    sb = torch.load(tmp_path / "mapped" / "net.pt", weights_only=False)["state"]
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert (tmp_path / "eager" / "net.pt").read_bytes() == (tmp_path / "mapped" / "net.pt").read_bytes()


@pytest.mark.parametrize("pools", [2, 3])
def test_pools_prepare_a_mapped_columns_ranges_once_and_as_the_eager_ones_do(tmp_path, pools, monkeypatch):
    dirs = _two(tmp_path, 70, 45)
    eager, mapped = tn.load_many(dirs, log=quiet), dl.load_mapped(dirs, log=quiet)
    rows = np.flatnonzero(_part(len(eager["index"])) == tn.TRAIN)
    calls = []
    real = tn.Batches.prepared_ranges
    monkeypatch.setattr(tn.Batches, "prepared_ranges", staticmethod(lambda d, r: calls.append(len(r)) or real(d, r)))
    pe = tn.Pools(eager, rows, "cpu", pools, 8)
    pm = tn.Pools(mapped, rows, "cpu", pools, 8)
    assert calls == [len(rows), len(rows)]  # once per Pools, at construction, never per upload
    assert isinstance(pm.host_ranges, np.ndarray) and pm.host_ranges.dtype == np.float16
    assert np.array_equal(pm.host_ranges, pe.host_ranges)


# ---------------------------------------------------------------- a mixed D3 folder


def _mixed(tmp_path, pairs=10, harvested=8, shard=7):
    rng = np.random.default_rng(11)
    src, idx = td.mixed_layout(pairs, harvested)
    n = len(src)
    d = tmp_path / "d1c"
    d.mkdir()
    sit = dict(board=_boards(pairs, 2).astype(np.int8), half=rng.integers(100, 900, pairs), stack=rng.integers(5000, 9000, pairs),
               cell=rng.integers(0, 9, pairs).astype(np.int32))
    np.savez(d / "situations.npz", **sit)
    hv = tmp_path / "queries"
    hv.mkdir()
    har = dict(board=_boards(harvested, 3).astype(np.int8), half=rng.integers(100, 900, harvested),
               stack=rng.integers(5000, 9000, harvested), cell=rng.integers(0, 9, harvested).astype(np.int64),
               ranges=rng.random((harvested, 2, vn.LIVE)).astype(np.float32))
    for k, v in har.items():
        np.save(hv / f"{k}.npy", v)
    (d / "generate.json").write_text(json.dumps(dict(layout=dl.MIXED, harvest=td.sha256_tree(hv), harvest_rows=harvested, samples=n)))
    vals = rng.normal(size=(n, 2, vn.LIVE)).astype(np.float32)
    rngs = rng.random((n, 2, vn.LIVE)).astype(np.float32)
    expl = rng.uniform(0, 1, n).astype(np.float32)
    for k, s in enumerate(range(0, n, shard)):
        e = slice(s, s + shard)
        np.savez(d / f"shard-{k:05d}.npz", index=idx[e], source=src[e], values=vals[e], ranges=rngs[e], expl=expl[e])
    return d, hv, src, idx, sit, har, vals


def test_a_mixed_folder_takes_a_harvested_rows_situation_from_the_harvest(tmp_path):
    d, hv, src, idx, sit, har, vals = _mixed(tmp_path)
    got = dl.load_mapped([d], log=quiet, harvest=hv)
    assert np.array_equal(got["source"], src) and np.array_equal(got["index"], idx)
    for k in ("board", "half", "stack", "cell"):
        want = np.where((src == 2)[(slice(None),) + (None,) * (got[k].ndim - 1)], har[k][np.where(src == 2, idx, 0)], sit[k][np.where(src == 2, 0, idx)])
        assert np.array_equal(got[k], want.astype(got[k].dtype)), k
    assert got["board"].dtype == np.int64 and np.array_equal(got["values"][np.arange(len(src))], vals)
    assert (got["board"][src == 2] == har["board"][idx[src == 2]]).all()
    assert tn.fingerprint(got) == tn.fingerprint(got)  # streams without a harvest copy


def test_a_mixed_folder_needs_its_own_harvest(tmp_path):
    d, hv, *_ = _mixed(tmp_path)
    with pytest.raises(SystemExit, match="pass its harvest"):
        dl.load_mapped([d], log=quiet)
    other = tmp_path / "other"
    other.mkdir()
    for f in hv.glob("*.npy"):
        x = np.load(f)
        np.save(other / f.name, x + 1 if f.stem == "stack" else x)
    with pytest.raises(SystemExit, match="not this run's harvest"):
        dl.load_mapped([d], log=quiet, harvest=other)
    dl.load_mapped([d], log=quiet, harvest=other, verify_harvest=False)  # (the check is the only guard: a caller opts out knowingly)


def test_source_2_rows_in_a_folder_that_is_not_mixed_are_refused(tmp_path):
    d, hv, *_ = _mixed(tmp_path)
    (d / "generate.json").write_text(json.dumps(dict(samples=26)))
    with pytest.raises(ValueError, match="no 'interleave-1:1' layout"):
        dl.load_mapped([d], log=quiet, harvest=hv)


# ---------------------------------------------------------------- measurements


def test_memory_reports_positive_gibibytes():
    m = dl.memory()
    assert set(m) == {"peak_working_set", "peak_commit", "private"} and all(v > 0 for v in m.values())
    assert m["peak_commit"] >= m["private"] * 0.99  # (the working set is no bound: torch's CUDA context commits more than it touches)


def test_measure_runs_on_a_toy_folder(tmp_path, capsys):
    import argparse

    dirs = _two(tmp_path)
    out = tmp_path / "m.json"
    dl.cmd_measure(argparse.Namespace(data=[str(x) for x in dirs], harvest=None, pool=20, out=str(out)))
    res = json.loads(out.read_text())
    assert res["rows"] == 115 and res["fingerprint"] == tn.fingerprint(tn.load_many(dirs, log=quiet)) and res["pool_rows"] == 20


# ---------------------------------------------------------------- d3-check


def _net_file(path, w=1.0, **meta):
    import torch

    torch.save(dict(width=2, depth=1, state=dict(a=torch.full((3,), w), b=torch.arange(4.0)), meta=dict(width=2, epoch=5, wd=0.1, **meta)), path)
    return path


def test_net_files_equal_is_the_net_not_the_bytes(tmp_path):
    import torch

    base = _net_file(tmp_path / "n1.pt")
    new = _net_file(tmp_path / "new.pt", dropout=0.0, ema=0.0, loss="huber")  # the three option keys at their off defaults: the same net
    r = dl.net_files_equal(base, new)
    assert r["ok"] and r["tensor_count"] == 2 and set(r["extra_meta"]) == {"dropout", "ema", "loss"} and base.read_bytes() != new.read_bytes()
    assert dl.net_files_equal(base, _net_file(tmp_path / "same.pt"))["ok"]
    assert not dl.net_files_equal(base, _net_file(tmp_path / "w.pt", w=1.0 + 1e-7))["ok"]  # one tensor differs
    assert not dl.net_files_equal(base, _net_file(tmp_path / "rw.pt", dropout=0.0, ema=0.0, loss="rw"))["ok"]  # an option that is on
    assert not dl.net_files_equal(base, _net_file(tmp_path / "d.pt", dropout=0.1))["ok"]
    other = tmp_path / "e.pt"
    torch.save(dict(width=2, depth=1, state=dict(a=torch.full((3,), 1.0)), meta=dict(width=2, epoch=5, wd=0.1)), other)
    assert not dl.net_files_equal(base, other)["ok"]  # a tensor missing
    for k, v in (("pools", 1), ("canonical", False), ("zz", None), ("ema", False), ("dropout", 0), ("loss", "huber ")):  # not one of the three at its default
        extra = dict(dropout=0.0, ema=0.0, loss="huber")
        extra.update({k: v})
        assert not dl.net_files_equal(base, _net_file(tmp_path / f"x{k}.pt", **extra))["ok"], k
    for name, change in (("width", lambda b: b.update(width=3)), ("dtype", lambda b: b["state"].update(a=b["state"]["a"].double())),
                         ("shared", lambda b: b["meta"].update(epoch=6))):
        blob = torch.load(base, map_location="cpu", weights_only=False)
        change(blob)
        torch.save(blob, tmp_path / f"{name}.pt")
        assert not dl.net_files_equal(base, tmp_path / f"{name}.pt")["ok"], name  # width; equal values in another dtype; a shared meta key


def test_check_compare_demands_every_field_and_the_net():
    hist = [dict(epoch=e, train_loss=0.1 / e, val_loss=0.2 / e, val_mae=1.0, seconds=s) for e, s in ((1, 5.0), (2, 9.0))]
    n1 = dict(best=hist[1], history=hist, width=1, depth=2, wd=0.1, seed=0, data="x", train_samples=3, val_samples=1, pools=2, epochs=2,
              batch=4, lr=1e-3)
    rec = json.loads(json.dumps(dict(n1, history=[dict(h, seconds=h["seconds"] + 1) for h in hist])))
    rec["best"] = dict(hist[1], seconds=99.0)
    ok, bad_net = dict(ok=True), dict(ok=False)
    assert dl.check_compare(rec, n1, dl.N1_SHA, ok)["reproduced"]
    r = dl.check_compare(rec, n1, "0" * 64, ok)  # the file's bytes are reported (Amendment 2), the net decides
    assert r["reproduced"] and not r["file_sha256"] and dl.check_compare(rec, n1, dl.N1_SHA, ok)["file_sha256"]
    assert not dl.check_compare(rec, n1, dl.N1_SHA, bad_net)["reproduced"] and not dl.check_compare(rec, n1, dl.N1_SHA, bad_net)["net"]
    for bad in (lambda r: r["history"][0].update(train_loss=0.1 + 1e-12), lambda r: r["best"].update(val_loss=0.3),
                lambda r: r["best"].update(epoch=1), lambda r: r.update(pools=1), lambda r: r.update(data="y")):
        x = json.loads(json.dumps(rec))
        bad(x)
        assert not dl.check_compare(x, n1, dl.N1_SHA, ok)["reproduced"]


def test_check_reproduces_a_net_trained_eager_on_the_same_toy_data(tmp_path, monkeypatch):
    import argparse
    import hashlib

    import livesearch as ls

    runs = tmp_path / "runs"
    a, b = _toy(1200, 0), _toy(800, 7)
    _folder(runs / "d1", a, shard_rows=256)
    _folder(runs / "d1b", b, shard_rows=300)
    (runs / "turn-net").mkdir()
    monkeypatch.setattr(tn, "NETS", runs / "turn-net")
    monkeypatch.setattr(tn, "D2B_DIRS", ("d1", "d1b"))
    monkeypatch.setattr(tn, "D2B_CANDIDATES", (dict(tag=dl.N1_TAG, width=16, data="union", wd=0.1),))
    monkeypatch.setattr(tn, "LEAFNET_SETTINGS", dict(width=16, depth=1, epochs=3, batch=64, lr=1e-3, seed=0))
    monkeypatch.setattr(tn, "d2b_check_files", lambda *x: None)
    monkeypatch.setattr(tn, "_commit", lambda: "a" * 40)
    monkeypatch.setattr(dl, "CHECK_JSON", tmp_path / "check.json")
    monkeypatch.setattr(ls, "gpu_state", lambda: dict(utc="t", util=0.0, mem_mib=1.0, apps=["1, x.exe"], peers=[]))
    eager = tn.d2b_views(tn.load_many([runs / "d1", runs / "d1b"], log=quiet))
    monkeypatch.setattr(tn, "LEAFNET_DATA", tn.fingerprint(eager["d1"]))
    monkeypatch.setattr(tn, "D2B_UNION", tn.fingerprint(eager["union"]))
    parts = {k: tn.split(v["board"]) for k, v in eager.items()}
    assert (parts["union"] == tn.SELECT).sum() > 0
    tn.train(eager["union"], parts["union"], 16, 1, 1.0, 3, 64, 1e-3, 0, dl.N1_TAG, log=quiet, device="cpu", wd=0.1, pools=tn.D2B_POOLS,
             commit="a" * 40)
    monkeypatch.setattr(dl, "N1_SHA", hashlib.sha256((runs / "turn-net" / dl.N1_TAG / "net.pt").read_bytes()).hexdigest())
    assert dl.cmd_check(argparse.Namespace(data=str(runs / "d1"), device="cpu")) == 0
    res = json.loads((tmp_path / "check.json").read_text())
    assert res["reproduced"] and res["sha256_check"] == dl.N1_SHA and res["train_rows"] > 0 and res["memory_end_gib"]["peak_working_set"] > 0
    real_sha = dl.N1_SHA
    monkeypatch.setattr(dl, "N1_SHA", "0" * 64)  # a net that is not the registered N1's file is refused before anything trains
    with pytest.raises(SystemExit, match="is not N1"):
        dl.cmd_check(argparse.Namespace(data=str(runs / "d1"), device="cpu"))
    monkeypatch.setattr(dl, "N1_SHA", real_sha)
    # a rerun never reuses the finished d3-check (train would "reproduce" without training): refused, or moved aside by --redo
    with pytest.raises(SystemExit, match="exists"):
        dl.cmd_check(argparse.Namespace(data=str(runs / "d1"), device="cpu"))
    first = (runs / "turn-net" / dl.CHECK_TAG / "net.pt").read_bytes()
    assert dl.cmd_check(argparse.Namespace(data=str(runs / "d1"), device="cpu", redo=True)) == 0
    assert (runs / "turn-net" / "d3-check-void-1" / "net.pt").read_bytes() == first
    assert (tmp_path / "check.json-void-1").exists() and (runs / "turn-net" / dl.CHECK_TAG / "net.pt").read_bytes() == first
    # Amendment 2: a finished check that failed only on the file's bytes (the new meta keys) is judged again by the net, nothing trained
    import torch

    net_path = runs / "turn-net" / dl.CHECK_TAG / "net.pt"
    blob = torch.load(net_path, map_location="cpu", weights_only=False)
    blob["meta"].update(dropout=0.0, ema=0.0, loss="huber")
    torch.save(blob, net_path)
    old = json.loads((tmp_path / "check.json").read_text())
    old.update(reproduced=False, file_sha256=False, sha256_check=hashlib.sha256(net_path.read_bytes()).hexdigest())
    old["net"] = False
    (tmp_path / "check.json").write_text(json.dumps(old))
    assert dl.cmd_recompare(argparse.Namespace(data=str(runs / "d1"))) == 0
    new = json.loads((tmp_path / "check.json").read_text())
    assert new["reproduced"] and new["recompared"] and new["run_reproduced"] is False and new["net_detail"]["extra_at_defaults"]
    assert new["file_sha256"] is False and (tmp_path / "check.json-void-2").exists() and new["run_commit"] == "a" * 40 and "sha256" not in new
    with pytest.raises(SystemExit, match="already a recompare"):
        dl.cmd_recompare(argparse.Namespace(data=str(runs / "d1")))
    blob["state"][next(iter(blob["state"]))] += 1e-7  # one weight differs: not N1, whatever the history says
    torch.save(blob, net_path)
    new2 = dict(old, sha256_check=hashlib.sha256(net_path.read_bytes()).hexdigest())
    (tmp_path / "check.json").write_text(json.dumps(new2))
    before = (tmp_path / "check.json").read_text()
    assert dl.cmd_recompare(argparse.Namespace(data=str(runs / "d1"))) == 1
    assert (tmp_path / "check.json").read_text() == before and not (tmp_path / "check.json.new").exists()  # a failed judgement changes nothing
    (tmp_path / "check.json").write_text(json.dumps(dict(new2, sha256_check="0" * 64)))
    with pytest.raises(SystemExit, match="is not the net the check recorded"):
        dl.cmd_recompare(argparse.Namespace(data=str(runs / "d1")))


def test_a_first_gpu_look_that_cannot_say_refuses_before_anything_trains(tmp_path, monkeypatch):
    import argparse
    import hashlib

    import livesearch as ls

    runs = tmp_path / "runs"
    _folder(runs / "d1", _toy(1200, 0), shard_rows=256)
    _folder(runs / "d1b", _toy(800, 7), shard_rows=300)
    (runs / "turn-net").mkdir()
    monkeypatch.setattr(tn, "NETS", runs / "turn-net")
    monkeypatch.setattr(tn, "D2B_DIRS", ("d1", "d1b"))
    monkeypatch.setattr(tn, "D2B_CANDIDATES", (dict(tag=dl.N1_TAG, width=16, data="union", wd=0.1),))
    monkeypatch.setattr(tn, "LEAFNET_SETTINGS", dict(width=16, depth=1, epochs=3, batch=64, lr=1e-3, seed=0))
    monkeypatch.setattr(tn, "d2b_check_files", lambda *x: None)
    monkeypatch.setattr(tn, "_commit", lambda: "a" * 40)
    monkeypatch.setattr(dl, "CHECK_JSON", tmp_path / "check.json")
    eager = tn.d2b_views(tn.load_many([runs / "d1", runs / "d1b"], log=quiet))
    monkeypatch.setattr(tn, "LEAFNET_DATA", tn.fingerprint(eager["d1"]))
    monkeypatch.setattr(tn, "D2B_UNION", tn.fingerprint(eager["union"]))
    parts = {k: tn.split(v["board"]) for k, v in eager.items()}
    tn.train(eager["union"], parts["union"], 16, 1, 1.0, 3, 64, 1e-3, 0, dl.N1_TAG, log=quiet, device="cpu", wd=0.1, pools=tn.D2B_POOLS,
             commit="a" * 40)
    monkeypatch.setattr(dl, "N1_SHA", hashlib.sha256((runs / "turn-net" / dl.N1_TAG / "net.pt").read_bytes()).hexdigest())
    for state in (dict(util=None, mem_mib=None, apps=[], peers=None), dict(util=0.0, mem_mib=1.0, apps=[], peers=["pid 9"])):
        monkeypatch.setattr(ls, "gpu_state", lambda s=state: dict(utc="t", **s))
        with pytest.raises(SystemExit, match="nothing trained"):
            dl.cmd_check(argparse.Namespace(data=str(runs / "d1"), device="cpu"))
        assert not (runs / "turn-net" / dl.CHECK_TAG / "record.json").exists() and not (tmp_path / "check.json").exists()


def test_a_column_takes_an_empty_list_and_a_zero_d_index_like_numpy(tmp_path):
    d = _toy(40, 0)
    col = dl.load_mapped([_folder(tmp_path / "a", d, shard_rows=16)], log=quiet)["values"]
    assert col[[]].shape == (0,) + d["values"].shape[1:] and np.array_equal(col[np.array(3)], d["values"][3])


def test_shards_that_disagree_on_a_columns_dtype_are_refused(tmp_path):
    d = _toy(40, 0)
    f = _folder(tmp_path / "a", d, shard_rows=20)
    p = sorted(f.glob("shard-*.npz"))[1]
    z = dict(np.load(p))
    z["expl"] = z["expl"].astype(np.float64)
    np.savez(p, **z)
    with pytest.raises(AssertionError, match="dtype differs"):
        dl.load_mapped([f], log=quiet)


def test_check_refuses_uncommitted_code(tmp_path, monkeypatch):
    import argparse

    monkeypatch.setattr(tn, "_commit", lambda: "a" * 40 + "+dirty")
    with pytest.raises(SystemExit, match="commit first"):
        dl.cmd_check(argparse.Namespace(data=str(tmp_path / "d1"), device="cpu"))
