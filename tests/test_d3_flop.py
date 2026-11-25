"""D3's flop-spot recorder, leaf-query hook and harvest layout (scripts/d3_flop.py, scripts/turn_data.py;
records/d3/PREREG.md 5.0.1 and 5.0.5)."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import d3_flop as d3  # noqa: E402
import turn_data as td  # noqa: E402


def _text(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


# ---- seed blocks ---------------------------------------------------------------------------------------------------


def test_registered_blocks_are_18000_wide_and_where_the_prereg_puts_them():
    assert d3.BLOCK == 18000 == td.BLOCK_WIDTH
    assert (d3.BLOCKS["D1"], d3.BLOCKS["D1b"], d3.BLOCKS["D1c"], d3.BLOCKS["S0a"], d3.BLOCKS["harvest"]) == (
        520000, 600000, 700000, 780000, 800000)
    assert d3.STREAMS["s0a"]["seed"] == 780000 and d3.STREAMS["harvest"]["seed"] == 800000


def test_s0a_and_the_harvest_touch_no_other_seed_in_the_repo():
    assert d3.clashes("S0a") == [] and d3.clashes("harvest") == []
    for name in ("D1", "D1b", "D1c", "S0a", "harvest"):
        for other in ("D1", "D1b", "D1c", "S0a", "harvest"):
            if other != name:
                assert frozenset((name, other)) not in d3.overlaps(), (name, other)


def test_the_only_overlaps_are_the_two_known_benign_ones():
    # D1c [700000, 718000) holds the rating's anchor seeds 700000 + i; D1b [600000, 618000) its calibration seeds.
    assert d3.overlaps() == d3.KNOWN_OVERLAPS


def test_a_block_moved_onto_any_other_span_is_refused(monkeypatch):
    for name, (lo, hi) in d3.spans().items():
        for seed in (lo - d3.BLOCK + 1, lo, hi - 1):
            moved = dict(d3.BLOCKS, harvest=seed)
            monkeypatch.setattr(d3, "BLOCKS", moved)
            if name != "harvest":
                assert name in d3.clashes("harvest") or frozenset(("harvest", name)) in d3.KNOWN_OVERLAPS, (name, seed)
    monkeypatch.setattr(d3, "BLOCKS", dict(d3.BLOCKS, harvest=800000 + d3.BLOCK))
    assert d3.clashes("harvest") == []  # just touching is free


def test_turn_datas_table_has_the_new_blocks_and_refuses_them():
    assert td.SEED_BLOCKS["S0a"] == 780000 and td.SEED_BLOCKS["harvest"] == 800000
    assert [c.split(" ")[0] for c in td.seed_clash(790000)] == ["S0a", "harvest"]


# ---- quota arithmetic ---------------------------------------------------------------------------------------------


def test_stream_is_round_robin_over_the_18_cells():
    assert d3.N_CELLS == 18 and d3.CELLS[0] == ("200bb", "self") and d3.CELLS[9] == ("100bb", "self")
    assert [d3.stream_cell(s) for s in (0, 1, 17, 18, 19, 35, 36)] == [(0, 0), (1, 0), (17, 0), (0, 1), (1, 1), (17, 1), (0, 2)]


def test_harvest_quota_is_567_per_cell_and_10204_spots():
    h = d3.STREAMS["harvest"]
    assert d3.quota("harvest") == 567 and 18 * 567 == 10206
    assert d3.per_cell(10206) == [567] * 18
    got = d3.per_cell(h["spots"])
    assert got == [567] * 16 + [566] * 2 and sum(got) == 10204


def test_harvest_keeps_exactly_499968_queries_the_last_spot_truncated_to_21():
    h = d3.STREAMS["harvest"]
    assert 10204 * 49 == 499996 and h["rows"] == 499968
    takes = [d3.take("harvest", s) for s in range(h["spots"])]
    assert sum(takes) == 499968 and set(takes[:-1]) == {49} and takes[-1] == 21
    assert d3.take("harvest", h["spots"]) == 0


def test_s0a_is_200_spots_of_10_queries_12_recorded_per_cell():
    s = d3.STREAMS["s0a"]
    assert sum(d3.take("s0a", i) for i in range(s["spots"])) == 2000 == s["rows"]
    assert d3.quota("s0a") == 12 and d3.per_cell(200) == [12] * 2 + [11] * 16


def test_the_draw_is_seeded_unique_and_the_last_spot_is_a_prefix():
    pool = 2 * 441
    a = d3.draw("harvest", 5, pool)
    assert len(a) == 49 and len(set(a.tolist())) == 49 and a.min() >= 0 and a.max() < pool
    assert (a == d3.draw("harvest", 5, pool)).all() and not (a == d3.draw("harvest", 6, pool)).all()
    assert not (a == d3.draw("s0a", 5, pool)[:10].tolist() + [0] * 39).all()
    last = d3.draw("harvest", 10203, pool)
    full = np.random.default_rng([800000, 10203]).choice(pool, 49, replace=False)
    assert len(last) == 21 and (last == full[:21]).all()
    with pytest.raises(RuntimeError, match="rows in the last iteration"):
        d3.draw("harvest", 0, 48)


# ---- the mixed layout --------------------------------------------------------------------------------------------


def test_mixed_layout_small():
    src, idx = td.mixed_layout(5, 4)  # 10 recorded rows, 4 harvested
    # row 2k is recorded row k (situation k // 2 or its twin), row 2k + 1 harvested row k, the rest recorded in order
    want = [(0, 0), (2, 0), (1, 0), (2, 1), (0, 1), (2, 2), (1, 1), (2, 3), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]
    assert list(zip(src.tolist(), idx.tolist())) == want


def test_mixed_layout_at_d3_size():
    pairs, harvested = 250020, 499968
    src, idx = td.mixed_layout(pairs, harvested)
    assert len(src) == 1000008 == 2 * pairs + harvested
    assert (src[1:2 * harvested:2] == 2).all() and (idx[1:2 * harvested:2] == np.arange(harvested)).all()
    assert (src[2 * harvested:] != 2).all() and len(src) - 2 * harvested == 72  # the 72 leftover recorded rows, at the end
    rec = np.concatenate([src[0:2 * harvested:2], src[2 * harvested:]])
    ri = np.concatenate([idx[0:2 * harvested:2], idx[2 * harvested:]])
    assert len(rec) == 500040
    assert ((2 * ri + rec) == np.arange(500040)).all()  # every recorded row once, in order
    assert (np.bincount(src) == [250020, 250020, 499968]).all()


def test_mixed_layout_refuses_more_harvest_than_recorded_rows():
    with pytest.raises(ValueError, match="recorded rows must cover"):
        td.mixed_layout(2, 5)
    with pytest.raises(ValueError):
        td.mixed_layout(2, -1)
    src, idx = td.mixed_layout(3, 0)  # no harvest: the plain pair layout
    assert src.tolist() == [0, 1] * 3 and idx.tolist() == [0, 0, 1, 1, 2, 2]


def test_harvest_only_layout_is_every_row_source_2_in_order():
    src, idx = td.harvest_only_layout(5)
    assert src.tolist() == [2] * 5 and idx.tolist() == [0, 1, 2, 3, 4] and src.dtype == np.int8


def _harvest(path, n):
    import numpy as np
    path.mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.save(path / "board.npy", np.array([[0, 5, 9, 14]] * n, dtype=np.int8))
    np.save(path / "half.npy", np.full(n, 300.0))
    np.save(path / "stack.npy", np.full(n, 9700.0))
    np.save(path / "cell.npy", np.zeros(n, np.int32))
    np.save(path / "ranges.npy", rng.random((n, 2, 1128)).astype(np.float32))
    return path


def test_generate_harvest_only_writes_an_empty_situations_file_and_a_harvest_only_meta(tmp_path, monkeypatch):
    import argparse
    import json

    hv = _harvest(tmp_path / "queries", 6)
    out = tmp_path / "s0a"
    out.mkdir()
    seen = {}
    monkeypatch.setattr(td, "_generate_mixed", lambda a, o, data, harvest, meta, log, pairs: seen.update(
        pairs=pairs, data=data, n=len(harvest["half"]), meta=meta) or 0)
    a = argparse.Namespace(out=str(out), iters=500, gamma=5, samples=6, shard=4, seed=780007, terminals64=True,
                           harvest=str(hv), harvest_only=True, worker=0, workers=1)
    assert td.cmd_generate(a) == 0
    m = json.loads((out / "generate.json").read_text())
    assert m["layout"] == td.HARVEST_ONLY and m["samples"] == 6 and m["harvest_rows"] == 6 and m["harvest"] == td.sha256_tree(hv)
    assert seen["pairs"] == 0 and seen["n"] == 6 and len(seen["data"]["half"]) == 0
    sit = np.load(out / "situations.npz")
    assert sit["board"].shape == (0, 4) and sit["board"].dtype == np.int8 and len(sit["half"]) == 0
    assert td.cmd_generate(argparse.Namespace(**dict(vars(a), samples=5))) == 1  # the harvest's rows, no fewer
    assert td.cmd_generate(argparse.Namespace(**dict(vars(a), harvest=None))) == 1


def test_merge_meta_clashes_on_the_layout_the_harvest_and_the_total():
    base = dict(iters=500, gamma=5, plus=False, samples=1000008, shard=1000, seed=700007, dtype="float32", code="x",
                layout="interleave-1:1", harvest="h1")
    assert td.merge_meta(base, dict(base, code="y"))[1] == []
    assert td.merge_meta(base, dict(base, harvest="h2"))[1] == ["harvest=h1"]
    assert td.merge_meta(base, dict(base, samples=1000010))[1] == ["samples=1000008"]
    plain = {k: v for k, v in base.items() if k not in ("layout", "harvest")}
    assert td.merge_meta(plain, base)[1] == ["layout=None", "harvest=None"]  # a pair directory never resumes as mixed
    assert td.merge_meta(base, plain)[1] == ["layout=interleave-1:1", "harvest=h1"]
    assert td.merge_meta(plain, dict(plain, samples=600000))[1] == []  # plain runs still extend freely


def test_harvest_loader_and_hash(tmp_path):
    cols = {"half": np.arange(5, dtype=np.int64), "ranges": np.ones((5, 2, 3), dtype=np.float32)}
    d = tmp_path / "queries"
    d.mkdir()
    for k, v in cols.items():
        np.save(d / f"{k}.npy", v)
    got = td.load_harvest(d)
    assert set(got) == {"half", "ranges"} and isinstance(got["ranges"], np.memmap) and got["ranges"][3].sum() == 6
    h = td.sha256_tree(d)
    assert h == td.sha256_tree(d) and len(h) == 64
    del got  # Windows will not overwrite a file that is still mapped
    np.save(d / "half.npy", cols["half"] + 1)
    assert td.sha256_tree(d) != h
    np.savez(tmp_path / "q.npz", **cols)
    assert set(td.load_harvest(tmp_path / "q.npz")) == {"half", "ranges"}
    assert td.sha256_tree(tmp_path / "q.npz") == td.sha256_file(tmp_path / "q.npz")


# ---- the constants are the live path's -----------------------------------------------------------------------------


def test_constants_are_the_live_search_s_and_the_registered_hashes():
    live = _text("slumbot_run.py")
    assert re.search(r"^FLOP_GAMMA = 5\.0", live, re.M) and re.search(r'^LEAVES = "sym"', live, re.M)
    assert "flop_iters=500" in live
    assert (d3.GAMMA, d3.LEAVES, d3.ITERS) == (5.0, "sym", 500)
    assert f'N1_SHA256 = "{d3.N1_SHA256}"' in _text("flop_sym_speed.py")
    assert f'V2_8B_SHA = "{d3.BP_8B_SHA256}"' in _text("bp8b_live.py")
    assert d3.BP_8B.endswith("runs\\bp-8b-stage2\\blueprint.bin") or d3.BP_8B.endswith("runs/bp-8b-stage2/blueprint.bin")


# ---- _prepare_one is _prepare, one situation at a time ----------------------------------------------------------------


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if hasattr(a, "__dict__"):
        return type(a) is type(b) and _same(vars(a), vars(b))
    return a == b


def test_prepare_one_makes_what_prepare_makes():
    pytest.importorskip("torch")
    pytest.importorskip("parciver_engine")
    pytest.importorskip("triton")
    from parciver.gpu_search import Hands

    board = (50, 40, 30, 9)
    n = len(Hands(list(board)).combos)
    r = np.random.default_rng(3).random((2, n)).astype(np.float32)
    row = dict(board=np.array(board, dtype=np.int8), half=300, stack=9700, first=1, bb=100, ranges=r)
    both, plans = td._prepare(row, 7, 520007)
    for src in (0, 1):
        sits, plan = td._prepare_one(row, 7, 520007, src)
        assert len(sits) == 1 and np.array_equal(sits[0].ranges, both[src].ranges)
        assert (sits[0].board, sits[0].half, sits[0].stack, sits[0].first, sits[0].bb) == \
               (both[src].board, both[src].half, both[src].stack, both[src].first, both[src].bb)
        assert _same(plan[0], plans[src])
    with pytest.raises(ValueError):
        td._prepare_one(row, 7, 520007, 2)


# ---- the leaf-query hook (GPU) -------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()


def _card(rank: int, suit: int) -> int:
    return rank * 4 + suit


FLOPS = {"two-tone": (_card(12, 0), _card(11, 0), _card(3, 1)), "monotone": (_card(12, 0), _card(11, 0), _card(3, 0))}
RAINBOW = (3, 22, 41)


@pytest.fixture(scope="module")
def net():
    from parciver import value_net as vn

    torch.manual_seed(1)
    return vn.ValueNet(64, 2).cuda().eval()


def _job(flop, symmetric: bool, seed: int = 3):
    pe = pytest.importorskip("parciver_engine")
    from parciver import flop_search as fs

    captured = []

    def grab(job):
        captured.append(job)
        return fs.FlopSearch(net=None, dtype=torch.float64)(job)

    hole = [c for c in range(52) if c not in flop][:2]
    pe.PlayAgent.uniform("100bb").with_search(from_flop=True, flop_iters=1, solver=grab).decide(
        hero=1, hole=tuple(hole), board=list(flop), actions=[("c", 0), ("x", 0)], sb=50, bb=100, stacks=(3000, 3000), seed=0)
    r = np.random.default_rng(seed).random((2, fs.HOLDINGS))
    if symmetric and len(fs.flop_group(flop)) > 1:
        def ev(ranges, board, live, half, stack):  # any evaluator will do: only the group average is used
            return torch.zeros_like(ranges)

        r = fs.SymLeaves(flop, ev, device="cpu").average(r)
    return dict(captured[0], ranges=r.tolist())


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
@pytest.mark.parametrize("name", ["two-tone", "monotone", "rainbow"])
def test_hook_sees_two_calls_per_iteration_and_changes_nothing(net, monkeypatch, name):
    monkeypatch.setattr(d3, "ITERS", 4)
    flop = RAINBOW if name == "rainbow" else FLOPS[name]
    job = _job(flop, name != "rainbow")
    hooked, plain = d3.make_search(net), d3.make_search(net, hook=False)
    last = d3.last_iteration(hooked, job)
    assert hooked.log.calls == 8 and len(last) == 2
    # a second solve of the same flop hits the table cache: still two calls per iteration, not wrapped twice
    d3.last_iteration(hooked, job)
    assert hooked.log.calls == 8
    # the hook is a pass-through: same strategies as the unhooked graph=False search
    assert hooked(d3.replay_job(job)) == plain(d3.replay_job(job))


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
@pytest.mark.parametrize("name", ["two-tone", "monotone", "rainbow"])
def test_fast_search_logs_the_same_last_iteration_bit_for_bit(net, monkeypatch, name):
    monkeypatch.setattr(d3, "ITERS", 5)
    flop = RAINBOW if name == "rainbow" else FLOPS[name]
    job = _job(flop, name != "rainbow")
    slow, fast, plain = d3.make_search(net), d3.make_search(net, fast=True), d3.make_search(net, hook=False)
    a, b = d3.last_iteration(slow, job), d3.last_iteration(fast, job)
    assert slow.log.calls == 10 and fast.log.calls == 2 and len(a) == len(b) == 2
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            assert torch.equal(u, v)
    # a second job on the same flop (table cache hit, a solver graph rebuilt) and the strategies are unchanged too
    assert fast(d3.replay_job(job)) == slow(d3.replay_job(job)) == plain(d3.replay_job(job))
    c = d3.last_iteration(fast, job)
    for x, y in zip(a, c):
        for u, v in zip(x, y):
            assert torch.equal(u, v)
    import parciver.flop_search as fsmod
    assert fsmod.FlopSolver.__name__ == "FlopSolver"  # the patch is undone


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
def test_fast_search_restores_the_log_and_the_solver_when_a_solve_raises(net, monkeypatch):
    import parciver.flop_search as fsmod

    monkeypatch.setattr(d3, "ITERS", 4)
    job = _job(FLOPS["two-tone"], True)
    fast = d3.make_search(net, fast=True)
    real = fsmod.FlopSolver
    monkeypatch.setattr(real, "step", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        fast(d3.replay_job(job))
    assert fast.log.on and fsmod.FlopSolver is real
    monkeypatch.undo()
    monkeypatch.setattr(d3, "ITERS", 4)
    assert len(d3.last_iteration(fast, job)) == 2 and fast.log.calls == 2  # still usable


def test_last_iteration_refuses_a_log_that_is_not_two_calls_per_iteration(monkeypatch):
    class Stub:
        log = d3.QueryLog()

        def __call__(self, job):
            for _ in range(7):  # 7 calls where 2 * 4 are expected
                self.log.add(torch.zeros(1, 1, 2, 1326), torch.zeros(1, 4), torch.zeros(1, 1326), torch.zeros(1), torch.zeros(1))

    monkeypatch.setattr(d3, "ITERS", 4)
    with pytest.raises(RuntimeError, match="not two per iteration"):
        d3.last_iteration(Stub(), {})
    s = Stub()
    s.log.reset()
    assert s.log.calls == 0 and s.log.last == []


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
@pytest.mark.parametrize("name,pool", [("two-tone", [333, 333]), ("monotone", [207, 207]), ("rainbow", [441, 441])])
def test_rows_are_the_turn_situations_in_the_turn_solvers_holding_order(net, monkeypatch, name, pool):
    from parciver import value_net as vn
    from parciver.gpu_search import Hands

    monkeypatch.setattr(d3, "ITERS", 3)
    flop = RAINBOW if name == "rainbow" else FLOPS[name]
    job = _job(flop, name != "rainbow")
    last = d3.last_iteration(d3.make_search(net), job)
    # 9 leaves x 37 / 23 / 49 turn rows per call on this fixture's job (symmetric flops use the orbit rows)
    assert d3.pool_sizes(last) == pool
    picks = d3.draw("harvest", 3, sum(pool))
    cols = d3.rows_to_situations(last, picks)
    n = len(picks)
    assert n == 49 and len(cols["ranges"]) == n and cols["ranges"][0].shape == (2, d3.td_holdings()) == (2, 1128)
    for i in range(n):
        x, board, live, half, stack = last[cols["call"][i]]
        l, t = divmod(cols["row"][i] - (0 if cols["call"][i] == 0 else pool[0]), x.shape[1])
        b = [int(c) for c in cols["board"][i]]
        assert len(b) == 4 and b == [int(c) for c in board[t].cpu()] and len(set(b)) == 4
        combos = Hands(b).combos
        assert (vn.PAIR_INDEX[combos[:, 0], combos[:, 1]] == np.flatnonzero(live[t].cpu().numpy() > 0)).all()
        assert np.array_equal(cols["ranges"][i], x[l, t].cpu().numpy()[:, vn.PAIR_INDEX[combos[:, 0], combos[:, 1]]])
        assert (cols["half"][i], cols["stack"][i]) == (int(half[l]), int(stack[l])) and cols["half"][i] > 0


@pytest.mark.skipif(not CUDA, reason="the flop search runs on the GPU")
def test_a_row_with_mass_on_a_dead_holding_or_odd_chips_is_refused(net, monkeypatch):
    monkeypatch.setattr(d3, "ITERS", 2)
    last = d3.last_iteration(d3.make_search(net), _job(FLOPS["two-tone"], True))
    x, board, live, half, stack = last[0]
    bad = x.clone()
    dead = int(torch.nonzero(live[0] == 0)[0])
    bad[0, 0, 0, dead] = 1.0
    with pytest.raises(RuntimeError, match="dead one"):
        d3.rows_to_situations([(bad, board, live, half, stack), last[1]], [0])
    odd = half.clone()
    odd[0] = 12.5
    with pytest.raises(RuntimeError, match="whole chips"):
        d3.rows_to_situations([(x, board, live, odd, stack), last[1]], [0])


def test_an_empty_or_wrong_harvest_is_refused_not_ignored(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="not a harvest"):
        td.load_harvest(tmp_path / "empty")
