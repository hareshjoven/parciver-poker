"""scripts/d1b_accept.py: D1b's A1-A3 as before, and D1c's (pairs, or D3's mixed layout with a harvest), on tiny runs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import d1b_accept as acc  # noqa: E402
import turn_data as td  # noqa: E402

H = 1128


def _meta(samples, shard, **extra):
    return dict(iters=500, gamma=5, plus=False, samples=samples, shard=shard, seed=700007, dtype="float32",
                terminals="q4", **extra)


def _run(tmp_path, samples, shard, harvested=0, expl=0.001, make=True):
    """A tiny finished run: situations.npz (the recorded ones), a harvest directory, generate.json and every shard."""
    rng = np.random.default_rng(5)
    d = tmp_path / "run"
    d.mkdir()
    pairs = (samples - harvested) // 2
    sit = rng.random((pairs, 2, H)).astype(np.float32)
    np.savez(d / "situations.npz", ranges=sit, half=np.full(pairs, 300), stack=np.full(pairs, 9700))
    meta = _meta(samples, shard)
    hv = None
    if harvested:
        hv = tmp_path / "queries"
        hv.mkdir()
        np.save(hv / "ranges.npy", rng.random((harvested, 2, H)).astype(np.float32))
        np.save(hv / "half.npy", np.full(harvested, 300))
        meta.update(layout=acc.MIXED, harvest=td.sha256_tree(hv), harvest_rows=harvested)
    (d / "generate.json").write_text(json.dumps(meta))
    src, idx, shards = acc._layout(meta)
    ranges = {0: sit, 2: np.load(hv / "ranges.npy") if hv else None}
    if make:
        for k in range(shards):
            s, i = src[k * shard:(k + 1) * shard], idx[k * shard:(k + 1) * shard]
            r = np.stack([sit[j] if x == 0 else sit[j][::-1] if x == 1 else ranges[2][j] for x, j in zip(s, i)])
            np.savez(d / f"shard-{k:05d}.npz", index=i, source=s, values=np.zeros((len(i), 2, H), np.float32),
                     game=np.zeros(len(i), np.float32), expl=np.full(len(i), expl, np.float32), ranges=r.astype(np.float32))
    return d, hv, meta


@pytest.fixture
def tiny(monkeypatch):
    def use(samples, shard, d1c_cloud=0):
        monkeypatch.setitem(acc.PROFILES, "d1c", dict(acc.PROFILES["d1c"], settings=_meta(samples, shard), cloud=d1c_cloud, harvest_rows=None))
    return use


def _args(d, hv=None, out=None, partial=False):
    return argparse.Namespace(profile="d1c", dir=str(d), harvest=str(hv) if hv else None, out=str(out), partial=partial)


def _check(tmp_path, d, hv=None, partial=False):
    out = tmp_path / "out.json"
    code = acc.cmd_check(_args(d, hv, out, partial))
    return code, json.loads(out.read_text())


def test_layout_pairs_matches_the_d1b_layout():
    src, idx, shards = acc._layout(dict(samples=500004, shard=1000))
    assert shards == 501 and len(src) == 500004
    for k in (0, 1, 500):  # D1b's _expected_index, kept here as the definition
        per, pairs = 500, 250002
        want = np.repeat(np.arange(k * per, min((k + 1) * per, pairs)), 2)
        assert np.array_equal(idx[k * 1000:(k + 1) * 1000], want)
        assert np.array_equal(src[k * 1000:(k + 1) * 1000], np.tile([0, 1], len(want) // 2))


def test_layout_mixed_at_d3_size_has_1001_shards_and_the_last_has_8_rows():
    src, idx, shards = acc._layout(_meta(1000008, 1000, layout=acc.MIXED, harvest_rows=499968))
    assert shards == 1001 and len(src) == 1000008 and len(src) - 1000 * 1000 == 8
    assert np.bincount(src).tolist() == [250020, 250020, 499968]


def test_a_complete_pairs_run_passes(tmp_path, tiny):
    tiny(40, 10)
    d, _, _ = _run(tmp_path, 40, 10)
    code, out = _check(tmp_path, d)
    assert code == 0 and out["verdict"] == "PASS" and out["A1"] and out["A2"] and out["A3"]
    assert out["shards_expected"] == 4 and out["sources"]["0"]["rows"] == 20 == out["sources"]["1"]["rows"]
    assert "2" not in out["sources"]


def test_a_complete_mixed_run_passes_and_reports_each_source(tmp_path, tiny):
    tiny(22, 10)
    d, hv, _ = _run(tmp_path, 22, 10, harvested=6)
    code, out = _check(tmp_path, d, hv)
    assert code == 0 and out["verdict"] == "PASS"
    assert out["shards_expected"] == 3 and out["whole"]["rows"] == 22
    assert {k: v["rows"] for k, v in out["sources"].items()} == {"0": 8, "1": 8, "2": 6}


def test_a_mixed_run_needs_its_own_harvest(tmp_path, tiny):
    tiny(22, 10)
    d, hv, _ = _run(tmp_path, 22, 10, harvested=6)
    with pytest.raises(SystemExit, match="pass --harvest"):
        _check(tmp_path, d, None)
    other = tmp_path / "other"
    other.mkdir()
    np.save(other / "ranges.npy", np.zeros((6, 2, H), np.float32))
    np.save(other / "half.npy", np.full(6, 300))
    with pytest.raises(SystemExit, match="not the harvest"):
        _check(tmp_path, d, other)


def test_a_wrong_source_a_wrong_index_or_wrong_ranges_fail_a1(tmp_path, tiny):
    tiny(22, 10)
    for what in ("source", "index", "ranges2", "ranges0"):
        t = tmp_path / what
        t.mkdir()
        d, hv, _ = _run(t, 22, 10, harvested=6)
        p = d / "shard-00000.npz"
        s = dict(np.load(p))
        if what == "source":
            s["source"] = s["source"][::-1].copy()
        elif what == "index":
            s["index"] = s["index"] + 1
        elif what == "ranges2":
            s["ranges"][np.flatnonzero(s["source"] == 2)[0]] += 1.0
        else:
            s["ranges"][0] += 1.0
        np.savez(p, **s)
        code, out = _check(t, d, hv)
        assert code == 2 and out["verdict"] == "FAIL" and not out["A1"] and out["A1_bad"][0]["shard"] == 0, what


def test_nonfinite_and_a_bad_zero_sum_and_a_high_mean_fail(tmp_path, tiny):
    tiny(40, 10)
    cases = {"nan": ("values", np.nan), "zerosum": ("values", 1.0), "expl": ("expl", 0.1)}
    for what, (field, val) in cases.items():
        t = tmp_path / what
        t.mkdir()
        d, _, _ = _run(t, 40, 10)
        p = d / "shard-00001.npz"
        s = dict(np.load(p))
        s[field] = np.full_like(s[field], val) if field == "expl" else s[field] + np.float32(val)
        np.savez(p, **s)
        code, out = _check(t, d)
        assert code == 2 and out["verdict"] == "FAIL", what
        assert {"nan": not out["A2"], "zerosum": not out["A2"], "expl": not out["A3"]}[what], what


def test_a_missing_shard_fails_but_partial_is_a_dry_run(tmp_path, tiny):
    tiny(40, 10)
    d, _, _ = _run(tmp_path, 40, 10)
    (d / "shard-00003.npz").unlink()
    code, out = _check(tmp_path, d)
    assert code == 2 and out["shards_missing"] == 1 and not out["A1"]
    code, out = _check(tmp_path, d, partial=True)
    assert code == 0 and out["verdict"] == "PARTIAL" and out["A1"] and out["whole"]["rows"] == 30


def test_settings_that_differ_from_the_profile_fail(tmp_path, tiny):
    tiny(40, 10)
    d, _, meta = _run(tmp_path, 40, 10)
    (d / "generate.json").write_text(json.dumps(dict(meta, seed=1)))
    code, out = _check(tmp_path, d)
    assert code == 2 and out["settings_clash"] == {"seed": [1, 700007]}


def test_a_bad_harvest_row_count_is_refused(tmp_path, tiny):
    tiny(22, 10)
    d, hv, meta = _run(tmp_path, 22, 10, harvested=6)
    (d / "generate.json").write_text(json.dumps(dict(meta, harvest_rows=0)))
    with pytest.raises(SystemExit, match="generate.json says"):
        _check(tmp_path, d, hv)


def test_the_registered_harvest_size_is_pinned_when_the_profile_has_one(tmp_path, monkeypatch, tiny):
    tiny(22, 10)
    d, hv, _ = _run(tmp_path, 22, 10, harvested=6)
    monkeypatch.setitem(acc.PROFILES["d1c"], "harvest_rows", 8)
    code, out = _check(tmp_path, d, hv)
    assert code == 2 and out["settings_clash"] == {"harvest_rows": [6, 8]}
    assert acc.PROFILES["d1c"]["harvest_rows"] == 8 and acc.PROFILES["d1b"].get("harvest_rows") is None


def test_d1b_profile_keeps_its_registered_settings():
    s = acc.PROFILES["d1b"]["settings"]
    assert (s["samples"], s["seed"], s["workers"], s["code"], acc.PROFILES["d1b"]["cloud"]) == (500004, 600007, 4, "7839aa6", 372)
    assert acc._draw(372000).tolist()[:3] == np.sort(np.random.default_rng(2026).choice(372000, 200, replace=False)).tolist()[:3]
    c = acc.PROFILES["d1c"]
    assert (c["settings"]["samples"], c["settings"]["seed"], c["cloud"]) == (1000008, 700007, 0)
