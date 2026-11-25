# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""LIVESEARCH B1 (LIVESEARCH.md, Amendment 3): `SubgameGpu(turn_schedule=...)` and scripts/livesearch.py's
harness: the schedule reaches turn jobs only, the default path is unchanged, the schedules with Q4's terminals,
frozen slots and kept sums match the NumPy reference; the Stage 1 rule, the chunk proofs, the resume refusal and
P1's statistic."""
import importlib.util
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("livesearch_cli", ROOT / "scripts" / "livesearch.py")
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    CUDA = False
gpu = pytest.mark.skipif(not CUDA, reason="the fused solver needs CUDA")


# ---------------------------------------------------------------- the solver option (GPU)


def _setup(cards, seed):
    """A tree on `cards` board cards with two frozen slots (the root and a node of the other player), as a
    re-solve freezes the searcher's earlier nodes (test_gpu_fused's)."""
    from parciver import gpu_search as gs

    hands = gs.Hands(gs.board(seed)[:cards])
    tree = gs.Tree((gs.small_turn_spec if cards == 4 else gs.spec)(), hands)
    dec = [x for x in tree.nodes if x.kind == "D"]
    other = next(x for x in dec if x.player != dec[0].player)
    rng = np.random.default_rng(seed)
    frozen = []
    for x in (dec[0], other):
        st = rng.random((len(x.acts), hands.n)) + 0.05
        frozen.append((x.slot, st / st.sum(0)))
    ranges = [gs.range_(hands.n, seed + 1), gs.range_(hands.n, seed + 2)]
    want = [dec[0].slot, dec[3].slot, dec[-1].slot]
    return hands, tree, dec, frozen, ranges, want


def _job(cards, seed, iters):
    from parciver import gpu_fused as gf

    hands, tree, dec, frozen, ranges, want = _setup(cards, seed)
    job = gf.job_from_tree(hands, tree, ranges, iters=iters, want=want, frozen=frozen)
    return dict(job, strength=hands.strength.astype(np.int64).reshape(-1).tolist()), hands, want


def _direct(job, hands, want, **kw):
    """`SubgameGpu.__call__`'s solve as it was before the option: the answers, action-major lists."""
    from parciver import gpu_fused as gf

    plan = gf.host_plan(hands, job, job["ranges"])
    s = gf.FusedSolver(hands, job, job["ranges"], dtype=torch.float32, plan=plan, sums=want, **kw)
    for slot, st in job["frozen"]:
        s.freeze(slot, np.asarray(st))
    for _ in range(int(job["iters"])):
        s.step()
    avg = s.average(want).to(torch.float64).cpu().numpy()
    return [avg[i, :plan.dk[w]].reshape(-1).tolist() for i, w in enumerate(want)]


@gpu
def test_turn_schedule_reaches_turn_jobs_only(monkeypatch):
    from parciver import gpu_fused as gf

    turn, _, _ = _job(4, 21, 6)
    river, _, _ = _job(5, 22, 6)
    seen, real = [], gf.FusedSolver

    def spy(*a, **kw):
        seen.append({k: kw[k] for k in ("gamma", "plus", "shift", "restart") if k in kw})
        return real(*a, **kw)

    monkeypatch.setattr(gf, "FusedSolver", spy)
    sub = gf.SubgameGpu(terminals64=True, turn_schedule=dict(gamma=5, plus=True))
    sub(turn)
    sub(river)
    sub(turn)
    assert seen == [dict(gamma=5, plus=True), {}, dict(gamma=5, plus=True)]
    # The schedules as the solver objects ran them (DCFR+'s shift defaults to 1).
    assert sub.solves == 3 and sub.schedules == {(4, 5, True, 1, False): 2, (5, 2, False, 0, False): 1}
    assert sorted(sub.times) == [4, 5] and len(sub.times[4]) == 2 and len(sub.times[5]) == 1
    assert sub.seconds == pytest.approx(sum(sub.times[4]) + sum(sub.times[5]))
    plain = gf.SubgameGpu(terminals64=True)
    plain(turn)
    assert plain.schedules == {(4, 2, False, 0, False): 1} and plain.turn_schedule is None and seen[-1] == {}


@gpu
@pytest.mark.parametrize("terminals64", [False, True], ids=["default", "q4"])
def test_the_default_path_is_unchanged_bit_for_bit(terminals64):
    from parciver import gpu_fused as gf

    for cards, seed in ((4, 23), (5, 24)):
        job, hands, want = _job(cards, seed, 12)
        want_ans = _direct(job, hands, want, terminals64=terminals64)
        for sub in (gf.SubgameGpu(terminals64=terminals64), gf.SubgameGpu(terminals64=terminals64, turn_schedule=None)):
            assert sub(job) == want_ans
            assert sub.schedules == {(cards, 2, False, 0, False): 1}


@gpu
@pytest.mark.parametrize("arm", list(ls.ARMS))
def test_a_turn_schedule_solves_as_the_solver_with_it(arm):
    """On a turn job the answer is `FusedSolver`'s with the schedule, bit for bit; on a river job it is the
    default solve's."""
    from parciver import gpu_fused as gf

    kw = ls.ARMS[arm]
    sub = gf.SubgameGpu(terminals64=True, turn_schedule=kw)
    turn, th, tw = _job(4, 25, 20)
    river, rh, rw = _job(5, 26, 20)
    assert sub(turn) == _direct(turn, th, tw, terminals64=True, **kw)
    assert sub(river) == _direct(river, rh, rw, terminals64=True) == gf.SubgameGpu(terminals64=True)(river)
    # The proof's expectation (`resolved`) is what the solver reports it ran.
    assert sub.schedules == {(4, *ls.resolved(kw)): 1, (5, 2, False, 0, False): 1}
    assert ls.resolved(None) == ls.resolved({}) == ls.resolved(dict(gamma=2)) == (2, False, 0, False)


def test_a_bad_turn_schedule_is_refused_before_any_solve():
    gf = pytest.importorskip("parciver.gpu_fused")
    for bad in (dict(alpha=1.5), dict(gamma=2.5), dict(gamma=0), dict(shift=-1), dict(shift=0.5)):
        with pytest.raises(ValueError):
            gf.SubgameGpu(turn_schedule=bad)
    for good in ls.ARMS.values():
        assert gf.SubgameGpu(turn_schedule=good).turn_schedule == good


def _synced(ref, gpu_solver, dec, frozen, kept, iters, rtol, f32=False):
    """Each traversal of `gpu_solver` against `ref` from the same state (both re-synced to the reference after
    every half-iteration); frozen slots skipped, sums compared on the kept slots only."""
    fz = {slot for slot, _ in frozen}
    dt = np.float32 if f32 else np.float64
    for half in range(2 * iters):
        if half % 2 == 0:
            ref.begin()
            gpu_solver.begin()
        ref.traverse(half % 2)
        gpu_solver.traverse(half % 2)
        regret, sums = gpu_solver.regret.cpu().numpy(), gpu_solver.sum.cpu().numpy()
        for x in dec:
            if x.slot in fz:
                continue
            k = len(x.acts)
            pairs = [(ref.regret[x.slot], regret[x.slot, :k])] + ([(ref.sum[x.slot], sums[x.slot, :k])] if x.slot in kept else [])
            for want, got in pairs:
                assert np.allclose(got, want, rtol=rtol, atol=rtol * max(1.0, float(np.abs(want).max()))), (x.slot, half)
            assert not ref.plus or (ref.regret[x.slot] >= 0).all()
            regret[x.slot, :k] = ref.regret[x.slot].astype(dt)
            if x.slot in kept:
                sums[x.slot, :k] = ref.sum[x.slot].astype(dt)
        gpu_solver.regret.copy_(torch.as_tensor(regret))
        gpu_solver.sum.copy_(torch.as_tensor(sums))


@gpu
@pytest.mark.parametrize("arm", list(ls.ARMS))
@pytest.mark.parametrize("cards,iters", [(4, 6), (5, 18)], ids=["turn", "river"])
def test_schedules_with_q4_frozen_slots_and_kept_sums_match_the_reference(cards, iters, arm):
    """Stage 1's arms as the live search runs them (float32 with Q4's terminals, frozen slots, `sums=want`)
    against the NumPy reference per traversal from a synced state, past the restarts at t - 1 = 1 and 4 (and 16 on
    the river); the same in float64 (Q3's kernels) to 1e-11; float32 graph steps equal eager ones bit for bit."""
    from parciver import gpu_fused as gf
    from parciver import gpu_search as gs

    kw = ls.ARMS[arm]
    hands, tree, dec, frozen, ranges, want = _setup(cards, 27)
    kept = set(want)

    def pair(dtype, **more):
        ref = gs.RefSolver(hands, tree, ranges, **kw)
        s = gf.FusedSolver(hands, tree, ranges, dtype=dtype, terminals64=True, sums=want, **kw, **more)
        for slot, st in frozen:
            ref.freeze(slot, st)
            s.freeze(slot, st)
        return ref, s

    ref, s64 = pair(torch.float64)
    _synced(ref, s64, dec, frozen, kept, iters, 1e-11)
    ref, s32 = pair(torch.float32)
    assert s32.ff  # Q4's float-float terminals
    _synced(ref, s32, dec, frozen, kept, iters, 2e-5, f32=True)
    _, a = pair(torch.float32)
    _, b = pair(torch.float32, graph=False)
    for _ in range(20):
        a.step()
        b.begin()
        b.traverse(0)
        b.traverse(1)
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)


def test_pot_of_is_the_round_start_pot():
    gf = pytest.importorskip("parciver.gpu_fused")
    from parciver import gpu_search as gs
    from parciver import turn_values as tv

    hands = gs.Hands(gs.board(31)[:4])
    s = tv.Situation(tuple(hands.board), 750, 20000 - 750, 1, 100, np.stack([gs.range_(hands.n, 1), gs.range_(hands.n, 2)]))
    tree = gs.Tree(s.spec(), hands)
    job = gf.job_from_tree(hands, tree, list(s.ranges), iters=1, want=[])
    assert ls.pot_of(job) == s.pot == 1500
    assert tv.root_of(job, 20000, 100).pot == 1500


# ---------------------------------------------------------------- the harness (no GPU)


def _table(**means):
    """A `tables()` subset with the means at 100 and 400 iterations given per arm (others 1.0)."""
    out = {}
    for nm in ls.ARMS:
        m = [1.0] * len(ls.CHECK)
        for b in ls.BUDGETS:
            m[ls.CHECK.index(b)] = means.get(nm, 1.0)
        out[nm] = dict(mean=m, median=m)
    return out


def test_the_stage1_rule():
    # dcfr_g2 lowest: kept.
    assert ls.pick(_table(dcfr_g2=0.9), 400)["pick"] == "dcfr_g2"
    # The lowest arm only 4% below dcfr_g2: kept.
    assert ls.pick(_table(plus_g4=0.96), 400)["pick"] == "dcfr_g2"
    # 10% below: replaces it.
    r = ls.pick(_table(plus_g4=0.90), 400)
    assert (r["pick"], r["lowest"], r["tied"], r["first"]) == ("plus_g4", "plus_g4", ["plus_g4"], "plus_g4")
    # Within 3% of the lowest: the order decides (plus_g5 before plus_g4), and the first is >= 5% below.
    assert ls.pick(_table(plus_g4=0.90, plus_g5=0.92), 400)["pick"] == "plus_g5"
    # The rule as written: the first within 3% is chosen, and only it is held to the 5% margin. dcfr_g5 comes
    # first and is 4% below dcfr_g2, so dcfr_g2 stays, though plus_g4 is 6% below.
    r = ls.pick(_table(plus_g4=0.94, dcfr_g5=0.96), 400)
    assert (r["tied"], r["first"], r["pick"]) == (["dcfr_g5", "plus_g4"], "dcfr_g5", "dcfr_g2")
    # dcfr_g2 itself within 3% of the lowest: no arm can be 5% below it.
    assert ls.pick(_table(dcfr_g5=0.98), 400)["pick"] == "dcfr_g2"
    # More than 3% above the lowest: not tied, whatever the order.
    assert ls.pick(_table(plus_g2=0.80, dcfr_g5=0.90), 400)["pick"] == "plus_g2"
    # Exactly at the bounds: included.
    assert ls.pick(_table(dcfr_g5=0.95), 400)["pick"] == "dcfr_g5"
    assert ls.pick(_table(postflop_solver=0.90, dcfr_g5=0.927), 400)["pick"] == "dcfr_g5"
    # Each budget reads its own column.
    t = _table(plus_g4=0.90)
    t["plus_g4"]["mean"][ls.CHECK.index(100)] = 1.0
    assert ls.pick(t, 100)["pick"] == "dcfr_g2" and ls.pick(t, 400)["pick"] == "plus_g4"


def test_tables_count_the_registered_and_distinct_starts():
    rows = []
    for start, same in enumerate([0, 1, 0]):  # start 2 re-solves start 0's root
        for kind in ("real", "random"):
            r = dict(kind=kind, start=start, expl={nm: [float(start)] * len(ls.CHECK) for nm in ls.ARMS})
            if kind == "real":
                r["same_as"] = same
            rows.append(r)
    t = ls.tables(rows)
    assert (t["real"]["n"], t["real_distinct"]["n"], t["random"]["n"], t["all"]["n"]) == (3, 2, 3, 6)
    assert t["real"]["dcfr_g2"]["mean"][0] == 1.0 and t["real_distinct"]["dcfr_g2"]["mean"][0] == 0.5


SCHED = dict(gamma=5, plus=True)
T1, T0, R = "4:gamma5:plus1:shift1:restart0", "4:gamma2:plus0:shift0:restart0", "5:gamma2:plus0:shift0:restart0"


def _counts(**over):
    c = dict(calls=10, failures=0, last_error=None, seen={"3:500": 2, "4:400": 5, "5:200": 3}, flop=2, turn=5, river=3,
             ran={T1: 5, R: 3}, symmetric=0)
    c.update(over)
    return c


def test_a_chunk_proves_its_arm():
    it1, it0 = {3: 500, 4: 400, 5: 200}, {3: 500, 4: 100, 5: 200}
    assert ls.run_key(4, ls.resolved(SCHED)) == T1 and ls.run_key(5, ls.resolved(None)) == R
    assert ls.prove("B1", _counts(), it1, SCHED) == []
    b0 = _counts(seen={"3:500": 2, "4:100": 5, "5:200": 3}, ran={T0: 5, R: 3})
    assert ls.prove("B0", b0, it0, None) == []
    assert ls.prove("B0", _counts(), it0, None)  # B1's turn iterations and schedule on B0
    assert ls.prove("B1", b0, it1, SCHED)  # B0's on B1
    assert ls.prove("B1", _counts(ran={T1: 4, T0: 1, R: 3}), it1, SCHED)  # one turn solve ran the default
    assert ls.prove("B1", _counts(ran={T1: 5, R: 2, "5:gamma5:plus1:shift1:restart0": 1}), it1, SCHED)  # on a river
    assert ls.prove("B1", _counts(ran={T1: 5, R: 3, "3:gamma2:plus0:shift0:restart0": 1}), it1, SCHED)  # a flop
    assert ls.prove("B1", _counts(ran={T1: 4, R: 3}), it1, SCHED)  # a turn solve without a recorded schedule
    assert ls.prove("B1", _counts(river=2), it1, SCHED)  # a job the solver did not solve
    assert ls.prove("B1", _counts(calls=11), it1, SCHED)
    assert ls.prove("B1", _counts(seen={"5:200": 3}, flop=0, turn=0, ran={R: 3}, calls=3), it1, SCHED)  # no flop/turn
    assert ls.prove("B1", _counts(failures=1, last_error="boom"), it1, SCHED)
    # dcfr_g2 as B1's schedule is the default's run.
    assert ls.prove("B1", dict(b0, seen={"3:500": 2, "4:400": 5, "5:200": 3}), it1, dict(gamma=2)) == []


def test_p2_at_4_threads_needs_a_smoke_run_on_this_code(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "RUNS", tmp_path)
    prov = dict(play_code={"src": "a", "engine": "b"})
    assert not ls.smoke_passed(prov, "s1", "e1")  # none
    good = dict(identical=True, threads=[1, ls.THREADS], play_code=prov["play_code"], engine="e1", stage1_sha256="s1")
    (tmp_path / "smoke.json").write_text(json.dumps(good))
    assert ls.smoke_passed(prov, "s1", "e1")
    assert not ls.smoke_passed(prov, "s2", "e1") and not ls.smoke_passed(prov, "s1", "e2")
    assert not ls.smoke_passed(dict(play_code={"src": "x", "engine": "b"}), "s1", "e1")
    for bad in (dict(identical=False), dict(threads=[1, 2]), dict(engine=None)):
        (tmp_path / "smoke.json").write_text(json.dumps(dict(good, **bad)))
        assert not ls.smoke_passed(prov, "s1", "e1")


def test_each_smoke_run_has_fresh_arms(tmp_path, monkeypatch):
    """An agent's search cache answers a replayed deal from its first solve, so the run at 4 threads must not
    reuse the 1-thread run's arms."""
    engine = tmp_path / "engine.pyd"
    engine.write_bytes(b"engine")
    pe = type("pe", (), dict(parciver_engine=type("m", (), dict(__file__=str(engine)))))

    class Fake:
        pass

    built, played, alive = [], [], []

    def build(threads, schedule):
        # The first run's arms are released before the second run's are built (one blueprint in memory at a time).
        alive.append(sum(r() is not None for r in played))
        built.append((threads, schedule))
        return None, pe, Fake(), Fake()

    def play(b0, b1, stakes, seed, first, deals, threads):
        played.extend([weakref.ref(b0), weakref.ref(b1)])
        args.append((id(b0), id(b1), seed, first, deals, threads))
        c = dict(flop=1, turn=1, river=1)
        return dict(per_deal_bb=[0.5] * deals, per_deal_bb_aivat=[0.25] * deals), dict(seconds=1.0, peak_mib=1.0, b0=c, b1=c)

    args = []

    bench = tmp_path / "benchmark.json"
    bench.write_text(json.dumps(dict(stakes=dict(sb=50, bb=100, stack=20000))))
    for k, v in dict(RUNS=tmp_path, BENCHMARK=bench, build_arms=build, play=play, provenance=lambda: dict(commit="c"),
                     b1_settings=lambda: ("shift1_g5", dict(gamma=5, shift=1), "s1")).items():
        monkeypatch.setattr(ls, k, v)
    assert ls.cmd_smoke(None) == 0
    assert built == [(ls.THREADS, dict(gamma=5, shift=1))] * 2 and alive == [0, 0]
    assert [p[5] for p in args] == [1, ls.THREADS] and all(p[2:5] == (ls.SMOKE_SEED, 0, ls.SMOKE_DEALS) for p in args)
    got = json.loads((tmp_path / "smoke.json").read_text())
    assert got["identical"] and got["threads"] == [1, ls.THREADS] and got["stage1_sha256"] == "s1"
    assert got["same_solves"] == dict(b0=True, b1=True)


def test_busy_counts_an_unknown_gpu_as_shared():
    assert not ls.busy(dict(peers=[], util=3.0))
    assert ls.busy(dict(peers=None, util=3.0)) and ls.busy(dict(peers=[123], util=3.0))
    assert ls.busy(dict(peers=[], util=None))  # nvidia-smi ran and failed: it lists no processes


def test_the_tap_counts_and_keeps_turn_jobs_only_while_asked():
    got = []
    tap = ls.Tap(lambda job: got.append(job) or "answer")
    turn = dict(board=[1, 2, 3, 4], iters=100, ranges=[[0.5], [0.5]])
    river = dict(board=[1, 2, 3, 4, 5], iters=200)
    assert tap(turn) == "answer" and tap(river) == "answer"
    tap.keep = True
    tap(turn)
    tap(river)
    turn["ranges"][0][0] = 9.0  # the engine's job changes later: the kept copy does not
    seen, jobs = tap.take()
    assert seen == {"4:100": 2, "5:200": 2} and len(jobs) == 1 and jobs[0]["ranges"][0][0] == 0.5
    assert tap.take() == ({}, []) and len(got) == 4


def test_resume_refuses_another_configuration(tmp_path):
    path = tmp_path / "p2-state.json"
    cfg = dict(seed=410001, threads=4, b1=dict(turn_schedule=dict(gamma=2)))
    assert ls.resume(path, cfg) == {"config": cfg}
    path.write_text(json.dumps(dict(config=cfg, run=dict(per_deal_bb=[1.0]))))
    assert ls.resume(path, dict(cfg))["run"]["per_deal_bb"] == [1.0]
    for other in (dict(cfg, threads=1), dict(cfg, b1=dict(turn_schedule=dict(gamma=5))), dict(cfg, extra=1)):
        with pytest.raises(SystemExit, match="another configuration"):
            ls.resume(path, other)


def test_the_board_bootstrap():
    rng = np.random.default_rng(5)
    boards = [(i % 7, 0, 0, 0) for i in range(70)]
    d = -1.0 + 0.1 * rng.standard_normal(70)
    b = ls.board_bootstrap(d, boards)
    assert b["hi"] < 0 and b["lo"] < b["mean"] < b["hi"] and b["boards"] == 7 and b["jobs"] == 70
    assert b == ls.board_bootstrap(d, boards)  # seeded
    assert b["mean"] == pytest.approx(d.mean())
    sym = np.array([1.0, -1.0] * 35)
    s = ls.board_bootstrap(sym, [(i // 2, 0, 0, 0) for i in range(70)])
    assert s["lo"] <= 0 <= s["hi"]
    # One board carries every negative job: resampling boards, not jobs, keeps the interval wide.
    one = np.array([-10.0] * 10 + [0.0] * 60)
    wide = ls.board_bootstrap(one, [(0, 0, 0, 0)] * 10 + [(i, 1, 0, 0) for i in range(60)])
    assert wide["lo"] < -0.5 and wide["hi"] == 0.0


def test_job_digest_and_board_key():
    job = dict(board=[5, 1, 3, 9], kind=[1, 0, 0], ranges=([0.25, 0.75], [0.5, 0.5]), frozen=[(0, [0.1, 0.9])], iters=100)
    same = {k: job[k] for k in reversed(list(job))}
    assert ls.job_digest(job) == ls.job_digest(same)
    assert ls.job_digest(job) != ls.job_digest(dict(job, frozen=[(0, [0.2, 0.8])]))
    assert ls.job_digest(job) != ls.job_digest(dict(job, ranges=([0.75, 0.25], [0.5, 0.5])))
    assert ls.board_key([5, 1, 3, 9]) == ls.board_key([3, 5, 1, 9]) == (1, 3, 5, 9) != ls.board_key([5, 1, 9, 3])
