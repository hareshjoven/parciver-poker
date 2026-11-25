# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The turn-data generator's resume rule (scripts/turn_data.py; FLOPSEARCH.md D1):
a run writing into a directory with shards must use their settings, and the
metadata keeps the code version that wrote the first shards."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("turn_data", Path(__file__).resolve().parents[1] / "scripts" / "turn_data.py")
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)

META = dict(iters=500, gamma=5, plus=False, samples=500004, shard=1000, seed=520007, dtype="float32", units="u",
            code="new")


def test_a_new_directory_takes_the_run_as_is():
    assert td.merge_meta(None, META) == (META, [])


def test_a_resume_keeps_the_first_code_and_appends_its_own():
    first = dict(META, code="12ad87f-dirty")
    first.pop("plus")  # written before `plus` was recorded: it was False
    meta, clash = td.merge_meta(first, dict(META, samples=600000))
    assert clash == []
    assert meta["code"] == "12ad87f-dirty" and meta["resumed"] == ["new"] and meta["samples"] == 600000
    again, clash = td.merge_meta(meta, dict(META, code="newer"))
    assert clash == [] and again["code"] == "12ad87f-dirty" and again["resumed"] == ["new", "newer"]


def test_other_settings_clash():
    for k, v in (("iters", 400), ("gamma", 2), ("plus", True), ("shard", 500), ("seed", 1), ("dtype", "float64")):
        _, clash = td.merge_meta(META, dict(META, **{k: v}))
        assert clash == [f"{k}={META[k]}"]
    old = dict(META)
    old.pop("gamma")  # the first runs had no gamma: it was 2
    assert td.merge_meta(old, META)[1] == ["gamma=2"]


def test_terminals_clash_and_d1_defaults_to_float32():
    q4 = dict(META, terminals="q4")
    assert td.merge_meta(META, q4)[1] == ["terminals=float32"]  # D1's file has no `terminals`: float32
    assert td.merge_meta(q4, META)[1] == ["terminals=q4"]
    meta, clash = td.merge_meta(q4, dict(q4, code="newer"))
    assert clash == [] and meta["terminals"] == "q4"


def test_workers_clash_and_d1_defaults_to_one():
    w2 = dict(META, workers=2)
    assert td.merge_meta(META, w2)[1] == ["workers=1"]  # a split run never resumes a 1-worker directory
    assert td.merge_meta(w2, dict(META, workers=3))[1] == ["workers=2"]
    assert td.merge_meta(w2, dict(w2, code="newer"))[1] == []


def test_exclusion_key_matches_the_stored_float32_ranges():
    import numpy as np

    rng = np.random.default_rng(1)
    r64 = rng.random((2, 1326))
    k = td._xkey((1, 2, 3, 4), 1, 19000, 1, r64)
    assert k == td._xkey(np.array([1, 2, 3, 4], dtype=np.int8), 1, 19000, 1, r64.astype(np.float32))
    assert k != td._xkey((1, 2, 3, 5), 1, 19000, 1, r64)
    assert k != td._xkey((1, 2, 3, 4), 0, 19000, 1, r64)
    r2 = r64.astype(np.float32).copy()
    r2[0, 7] = np.nextafter(r2[0, 7], np.float32(2))
    assert k != td._xkey((1, 2, 3, 4), 1, 19000, 1, r2)


# ---- D3 5.0.2: record --exclude takes two files; the seed-block guard; the chunk cap -----------------------------

import sys
import types

import numpy as np


def test_seed_clash_blocks_are_18000_wide_and_touching_counts():
    assert td.BLOCK_WIDTH == 18000
    for seed, hit in ((520000, "D1"), (502001, "D1"), (537999, "D1"), (600000, "D1b"), (582001, "D1b"), (617999, "D1b"),
                      (780000, "S0a"), (800000, "harvest"), (790000, "S0a")):
        assert any(c.startswith(hit + " ") for c in td.seed_clash(seed)), (seed, td.seed_clash(seed))
    for seed in (700000, 502000, 538000, 582000, 618000, 762000, 818000, 410001, 550000):
        assert td.seed_clash(seed) == [], seed  # D1c's 700000 is free; the edges just touching are free


def test_seed_clash_names_every_block_touched():
    assert [c.split(" ")[0] for c in td.seed_clash(790000)] == ["S0a", "harvest"]


def _sit(tag: int, cell: int):
    """A fake situation, distinct per (tag, cell)."""
    rng = np.random.default_rng([tag, cell])
    return types.SimpleNamespace(board=(1, 2, 3, 4 + tag % 40), half=100 + cell, stack=19000, first=tag % 2, bb=100,
                                 ranges=rng.random((2, 1128)))


class _FakeWorld:
    """Fakes `parciver_engine` and `parciver.turn_values` for cmd_record. A match call at match seed `s` records
    `per_chunk` situations: the first `pool` come from a pool shared by every chunk of that cell (so they repeat), the
    rest have tags unique to `s`."""

    def __init__(self, per_chunk=6, pool=3):
        self.seeds, self.per_chunk, self.pool, self.base = [], per_chunk, pool, 0
        world = self

        class Rec:
            def __init__(self, stack, bb):
                self.stack, self.situations, self._seen = stack, [], set()

        class Agent:
            def __init__(self, solver=None):
                self.rec = solver

            def with_search(self, from_turn, solver):
                return Agent(solver)

            def match_against(self, opp, deals, sb, bb, stack, seed=0, threads=1, **kw):
                world.seeds.append(seed)
                cell = (seed - world.base) // 1000
                for i in range(world.per_chunk):
                    tag = (seed * 100 + i) if i >= world.pool else (cell * 1000 + i)
                    s = _sit(tag, cell)
                    key = (s.board, s.half, s.first, s.ranges.tobytes())
                    if key not in self.rec._seen:
                        self.rec._seen.add(key)
                        self.rec.situations.append(types.SimpleNamespace(
                            board=s.board, half=s.half, stack=s.stack, first=s.first, bb=s.bb, ranges=s.ranges))

        class Bp:
            def __init__(self, *a):
                pass

            def play_agent(self, *a, **k):
                return Agent()

        self.pe = types.SimpleNamespace(SavedBlueprint=Bp)
        self.tv = types.SimpleNamespace(Recorder=Rec)

    def install(self, monkeypatch, base):
        import parciver

        self.base = base
        monkeypatch.setitem(sys.modules, "parciver_engine", self.pe)
        monkeypatch.setitem(sys.modules, "parciver.turn_values", self.tv)
        monkeypatch.setattr(parciver, "turn_values", self.tv, raising=False)
        monkeypatch.setattr(td, "OPPONENTS", ["self", "baseline"])  # 2 stakes x 2 = 4 cells, for speed
        monkeypatch.setattr(td, "BLOCK_WIDTH", 1000 * 4)
        monkeypatch.setattr(td, "SEED_BLOCKS", {"D1": 520000, "D1b": 600000})
        monkeypatch.setattr(td, "_match", lambda pe, agent, v1, opp, deals, stakes, seed, threads: agent.match_against(
            "agent", deals, 0, 0, 0, seed=seed, threads=threads))


def _args(out, seed, exclude, situations=8, chunk=10):
    return types.SimpleNamespace(out=str(out), situations=situations, chunk=chunk, threads=1, seed=seed, exclude=exclude)


def _stored(path, sits):
    np.savez(path, board=np.array([s.board for s in sits], dtype=np.int8), half=np.array([s.half for s in sits]),
             stack=np.array([s.stack for s in sits]), first=np.array([s.first for s in sits]),
             ranges=np.array([s.ranges.astype(np.float32) for s in sits]))


def test_excluded_takes_none_one_path_or_many(tmp_path):
    a, b = [_sit(t, 0) for t in range(3)], [_sit(t, 1) for t in range(2)]
    _stored(tmp_path / "a.npz", a)
    _stored(tmp_path / "b.npz", b)
    assert td._excluded(None) == set()
    assert len(td._excluded(str(tmp_path / "a.npz"))) == 3
    both = td._excluded([str(tmp_path / "a.npz"), tmp_path / "b.npz"])
    assert len(both) == 5 and both == td._excluded(tmp_path / "a.npz") | td._excluded(tmp_path / "b.npz")


def test_record_refuses_a_seed_touching_d1_or_d1b_before_any_match(tmp_path, monkeypatch, capsys):
    w = _FakeWorld()
    w.install(monkeypatch, 700000)
    _stored(tmp_path / "x.npz", [_sit(0, 0)])
    for seed in (520000, 517000, 601000, 597000):
        assert td.cmd_record(_args(tmp_path / f"o{seed}", seed, [str(tmp_path / "x.npz")])) == 1
        assert "overlap" in capsys.readouterr().out
    assert w.seeds == [] and not (tmp_path / "o520000" / "situations.npz").exists()


def test_record_with_two_files_drops_both_and_stays_inside_each_cells_seeds(tmp_path, monkeypatch, capsys):
    w = _FakeWorld(per_chunk=6, pool=3)
    w.install(monkeypatch, 700000)
    # chunk i of cell 0 records tags (700000 + i) * 100 + j for j in 3..5 (fresh per chunk). Store all of chunks 0 and 1
    # of cell 0, one chunk in each file: neither may come back.
    s0 = [_sit(700000 * 100 + j, 0) for j in range(3, 6)]
    s1 = [_sit(700001 * 100 + j, 0) for j in range(3, 6)]
    _stored(tmp_path / "d1.npz", s0)
    _stored(tmp_path / "d1b.npz", s1)
    out = tmp_path / "o"
    ex_files = [str(tmp_path / "d1.npz"), str(tmp_path / "d1b.npz")]
    assert td.cmd_record(_args(out, 700000, ex_files, situations=8)) == 0
    d = np.load(out / "situations.npz")
    assert len(d["half"]) == 8
    ex = td._excluded(ex_files)
    got = {td._xkey(d["board"][i], d["half"][i], d["stack"][i], d["first"][i], d["ranges"][i]) for i in range(8)}
    assert not (got & ex) and len(got) == 8
    assert np.bincount(d["cell"], minlength=4).tolist() == [2] * 4
    assert d["excluded_hits"][0] >= 1  # cell 0 met stored ones
    for s in w.seeds:  # every match seed lies inside its own cell's 999 chunks
        assert 700000 <= s < 704000 and (s - 700000) % 1000 < td.MAX_CHUNKS


def test_record_fails_when_a_cell_cannot_fill_under_the_chunk_cap(tmp_path, monkeypatch, capsys):
    w = _FakeWorld(per_chunk=3, pool=3)  # every chunk repeats the same 3 situations: a cell never reaches 4
    w.install(monkeypatch, 700000)
    _stored(tmp_path / "x.npz", [_sit(0, 9)])
    assert td.cmd_record(_args(tmp_path / "o", 700000, [str(tmp_path / "x.npz")], situations=16)) == 1
    assert "short of 4" in capsys.readouterr().out
    assert not (tmp_path / "o" / "situations.npz").exists()
    assert len(w.seeds) <= td.MAX_CHUNKS and max(w.seeds) - 700000 < td.MAX_CHUNKS  # never chunk 1000 (cell 1's seeds)


def test_record_cap_holds_without_exclude_too(tmp_path, monkeypatch, capsys):
    w = _FakeWorld(per_chunk=3, pool=3)
    w.install(monkeypatch, 520000)
    assert td.cmd_record(_args(tmp_path / "o", 520000, None, situations=16)) == 1
    assert len(w.seeds) <= td.MAX_CHUNKS and "short of 4" in capsys.readouterr().out

def test_the_chunk_cap_is_999():
    assert td.MAX_CHUNKS == 999  # 1000 would run into the next cell's seeds


def test_the_second_excluded_file_is_honoured_and_the_top_up_runs(tmp_path, monkeypatch):
    w = _FakeWorld(per_chunk=3, pool=0)  # every chunk: 3 situations nobody else has
    w.install(monkeypatch, 700000)
    # Cell 0's chunks 0 and 1 are stored, one in each file. per_cell is 4: chunk 0-1 are run by the first loop (6
    # situations, all excluded), so chunks 2 and 3 must be run by the top-up. With D1b's file ignored chunk 3 is not needed.
    _stored(tmp_path / "d1.npz", [_sit((700000) * 100 + i, 0) for i in range(3)])
    _stored(tmp_path / "d1b.npz", [_sit((700001) * 100 + i, 0) for i in range(3)])
    ex = [str(tmp_path / "d1.npz"), str(tmp_path / "d1b.npz")]
    assert td.cmd_record(_args(tmp_path / "o", 700000, ex, situations=16)) == 0
    cell0 = [s for s in w.seeds if (s - 700000) // 1000 == 0]
    assert cell0 == [700000, 700001, 700002, 700003]  # chunk 3 only because both files hold their chunks
    d = np.load(tmp_path / "o" / "situations.npz")
    assert np.bincount(d["cell"], minlength=4).tolist() == [4] * 4 and d["excluded_hits"][0] == 6
    got = {td._xkey(d["board"][i], d["half"][i], d["stack"][i], d["first"][i], d["ranges"][i]) for i in range(16)}
    assert not (got & td._excluded(ex))
    # the same run with only D1's file needs one chunk fewer
    w2 = _FakeWorld(per_chunk=3, pool=0)
    w2.install(monkeypatch, 700000)
    assert td.cmd_record(_args(tmp_path / "o2", 700000, ex[:1], situations=16)) == 0
    assert [s for s in w2.seeds if (s - 700000) // 1000 == 0] == [700000, 700001, 700002]
