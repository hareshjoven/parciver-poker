"""Fused GPU solver (src/parciver/gpu_fused.py): its O(n) fold and showdown
terminals against the dense card-removal matrices, and CUDA-graph steps
against eager ones. Its DCFR against the reference runs in test_gpu_search.py
(parametrized over both GPU solvers)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("parciver_engine")
if not torch.cuda.is_available():
    pytest.skip("the fused solver needs CUDA", allow_module_level=True)

from parciver import gpu_fused as gf  # noqa: E402
from parciver import gpu_search as gs  # noqa: E402


@pytest.mark.parametrize("terminals64", [False, True], ids=["default", "terminals64"])
@pytest.mark.parametrize("cards,sp", [(5, gs.spec), (4, gs.small_turn_spec)], ids=["river", "turn"])
def test_fold_and_showdown_terminals_match_the_dense_matrices(cards, sp, terminals64):
    hands = gs.Hands(gs.board(3)[:cards])
    tree = gs.Tree(sp(), hands)
    ranges = [gs.range_(hands.n, 1), gs.range_(hands.n, 2)]
    s = gf.FusedSolver(hands, tree, ranges, dtype=torch.float64, terminals64=terminals64)
    rng = np.random.default_rng(0)
    T = len(s.term_nodes)
    rq = rng.random((T, hands.n)) * (rng.random((T, hands.n)) < 0.7)  # sparse, like real reaches
    for j, i in enumerate(s.term_nodes):  # reaches below a turn card are dead on it
        rq[j] *= hands.alive[tree.nodes[i].view]
    W = [hands.showdown_matrix(v) for v in range(hands.views)]
    for p in (0, 1):
        got = s.terminal_values(p, torch.as_tensor(rq, device="cuda")).cpu().numpy()
        for j, i in enumerate(s.term_nodes):
            x = tree.nodes[i]
            if x.folded is not None:
                amount = -float(x.stake[p]) if x.folded == p else float(x.stake[1 - p])
                want = (hands.compat @ rq[j]) * amount
            else:
                want = (W[x.view] @ rq[j]) * float(min(x.stake))
            np.testing.assert_allclose(got[j], want, rtol=0, atol=1e-12 * max(1.0, np.abs(want).max()))  # prefix-sum differences cancel


@pytest.mark.parametrize("q3", [False, True], ids=["q4", "q3"])
@pytest.mark.parametrize("cards,sp", [(5, gs.spec), (4, gs.small_turn_spec)], ids=["river", "turn"])
def test_terminals64_float32_values_are_the_dense_values_rounded_once(cards, sp, q3):
    """KERNELS Q3: with `terminals64` the kernels sum in float64 (Q4: in pairs
    of float32; `q3=True` keeps Q3's), so a float32 value is the dense float64
    value on the same float32 inputs up to one rounding, at every holding. The
    default's float32 sums leave ~1e-7 of the row's scale on holdings whose
    value is far smaller, and regret matching follows it (records/d1-tail
    PREREG14)."""
    hands = gs.Hands(gs.board(3)[:cards])
    tree = gs.Tree(sp(), hands)
    ranges = [gs.range_(hands.n, 1), gs.range_(hands.n, 2)]
    s = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, terminals64=True, q3=q3)
    assert s.ff is not q3
    rng = np.random.default_rng(1)
    T = len(s.term_nodes)
    rq = (rng.random((T, hands.n)) ** 8 * (rng.random((T, hands.n)) < 0.3)).astype(np.float32)  # skewed, sparse
    for j, i in enumerate(s.term_nodes):
        rq[j] *= hands.alive[tree.nodes[i].view]
    r64 = rq.astype(np.float64)
    W = [hands.showdown_matrix(v) for v in range(hands.views)]
    for p in (0, 1):
        got = s.terminal_values(p, torch.as_tensor(rq, device="cuda")).double().cpu().numpy()
        for j, i in enumerate(s.term_nodes):
            x = tree.nodes[i]
            if x.folded is not None:
                amount = -float(x.stake[p]) if x.folded == p else float(x.stake[1 - p])
                dense = hands.compat @ r64[j]
            else:
                amount = float(min(x.stake))
                dense = W[x.view] @ r64[j]
            amount = float(np.float32(amount))  # the solver's own float32 amount
            want = dense * amount
            bound = 2.0 ** -24 * np.abs(want) + 1e-12 * r64[j].sum() * abs(amount)
            assert np.all(np.abs(got[j] - want) <= bound), (p, j, np.abs(got[j] - want).max())


def test_graph_steps_equal_eager_steps():
    hands = gs.Hands(gs.board(2)[:4])
    tree = gs.Tree(gs.small_turn_spec(), hands)
    ranges = [gs.range_(hands.n, 3), gs.range_(hands.n, 4)]
    a = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, graph=True)
    b = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, graph=False)
    for _ in range(6):
        a.step()
        b.step()
    assert a.iter == b.iter == 6 and int(a.iter_t) == 6
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)


@pytest.mark.parametrize("dtype,regret_dtype", [(torch.float32, None), (torch.float64, None), (torch.float32, torch.float16)],
                         ids=["f32", "f64", "f16-regrets"])
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_step_defers_the_discount_bit_identically(dtype, regret_dtype, graph):
    """`step()` discounts each regret at its first read of the iteration; the
    public `begin()` + `traverse(0)` + `traverse(1)` discounts them all first.
    Same bits, frozen slots included."""
    for cards, sp in ((4, gs.small_turn_spec), (5, gs.spec)):
        hands, tree, dec, frozen = _frozen_setup(cards, sp, 17)
        ranges = [gs.range_(hands.n, 8), gs.range_(hands.n, 9)]
        a = gf.FusedSolver(hands, tree, ranges, dtype=dtype, graph=graph, regret_dtype=regret_dtype)
        b = gf.FusedSolver(hands, tree, ranges, dtype=dtype, graph=False, regret_dtype=regret_dtype)
        for s in (a, b):
            for slot, st in frozen:
                s.freeze(slot, st)
        for _ in range(40):
            a.step()
            b.begin()
            b.traverse(0)
            b.traverse(1)
        assert a.iter == b.iter == 40 and int(a.iter_t) == int(b.iter_t) == 40
        assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)
        assert a.value(0) == b.value(0) and a.exploitability() == b.exploitability()


def _random_strategy(k, n, seed):
    s = np.random.default_rng(seed).random((k, n)) + 0.05
    return s / s.sum(0)


def _frozen_setup(cards, sp, seed):
    hands = gs.Hands(gs.board(seed)[:cards])
    tree = gs.Tree(sp(), hands)
    dec = [x for x in tree.nodes if x.kind == "D"]
    # The root and a node of the other player below it, as a solved round freezes
    # the searcher's own earlier nodes.
    other = next(x for x in dec if x.player != dec[0].player)
    frozen = [(x.slot, _random_strategy(len(x.acts), hands.n, x.slot)) for x in (dec[0], other)]
    return hands, tree, dec, frozen


@pytest.mark.parametrize("cards,sp,iters", [(5, gs.spec, 8), (4, gs.small_turn_spec, 2)], ids=["river", "turn"])
def test_frozen_nodes_match_the_reference(cards, sp, iters):
    hands, tree, dec, frozen = _frozen_setup(cards, sp, 11)
    ranges = [gs.range_(hands.n, 5), gs.range_(hands.n, 6)]
    ref = gs.RefSolver(hands, tree, ranges)
    gpu = gf.FusedSolver(hands, tree, ranges, dtype=torch.float64)
    for slot, st in frozen:
        ref.freeze(slot, st)
        gpu.freeze(slot, st)
    fz = {slot for slot, _ in frozen}
    for half in range(2 * iters):  # per traversal from a synced state (see test_gpu_search)
        if half % 2 == 0:
            ref.begin()
            gpu.begin()
        ref.traverse(half % 2)
        gpu.traverse(half % 2)
        regret, sums = gpu.regret.cpu().numpy(), gpu.sum.cpu().numpy()
        for x in dec:
            k = len(x.acts)
            if x.slot in fz:
                np.testing.assert_array_equal(regret[x.slot, :k], sums[x.slot, :k])  # untouched
                continue
            for want, got in ((ref.regret[x.slot], regret[x.slot, :k]), (ref.sum[x.slot], sums[x.slot, :k])):
                assert np.allclose(got, want, rtol=1e-11, atol=1e-11 * max(1.0, float(np.abs(want).max())))
            regret[x.slot, :k] = ref.regret[x.slot]
            sums[x.slot, :k] = ref.sum[x.slot]
        gpu.regret.copy_(torch.as_tensor(regret))
        gpu.sum.copy_(torch.as_tensor(sums))
    avg = gpu.average().cpu().numpy()
    for slot, st in frozen:
        np.testing.assert_allclose(avg[slot, :st.shape[0]], st, rtol=1e-12)


def _synced_against_the_reference(ref, gpu, dec, iters: int) -> None:
    """Each traversal of a float64 `gpu` against `ref` from the same state
    (both re-synced to the reference after every half-iteration)."""
    for half in range(2 * iters):
        if half % 2 == 0:
            ref.begin()
            gpu.begin()
        ref.traverse(half % 2)
        gpu.traverse(half % 2)
        regret, sums = gpu.regret.cpu().numpy(), gpu.sum.cpu().numpy()
        for x in dec:
            k = len(x.acts)
            for want, got in ((ref.regret[x.slot], regret[x.slot, :k]), (ref.sum[x.slot], sums[x.slot, :k])):
                assert np.allclose(got, want, rtol=1e-11, atol=1e-11 * max(1.0, float(np.abs(want).max())))
            regret[x.slot, :k] = ref.regret[x.slot]
            sums[x.slot, :k] = ref.sum[x.slot]
        gpu.regret.copy_(torch.as_tensor(regret))
        gpu.sum.copy_(torch.as_tensor(sums))


def test_q3_switch_selects_the_terminal_kernels():
    """KERNELS Q4: float32 `terminals64` solves run Q4's float-float kernels
    unless `q3=True`; float64 solves run Q3's either way; without
    `terminals64`, `q3` changes nothing."""
    hands = gs.Hands(gs.board(3)[:5])
    tree = gs.Tree(gs.spec(), hands)
    ranges = [gs.range_(hands.n, 1), gs.range_(hands.n, 2)]

    def solver(**kw):
        return gf.FusedSolver(hands, tree, ranges, graph=False, **kw)

    q4, q3 = solver(dtype=torch.float32, terminals64=True), solver(dtype=torch.float32, terminals64=True, q3=True)
    assert q4.ff and q4.scr.dtype == torch.float32 and q4.scr.shape[1] == 2 * q4.SW
    assert not q3.ff and q3.scr.dtype == torch.float64 and q3.scr.shape[1] == q3.SW
    for kw in (dict(dtype=torch.float64, terminals64=True), dict(dtype=torch.float32), dict(dtype=torch.float32, q3=True)):
        assert not solver(**kw).ff, kw


@pytest.mark.parametrize("q3", [False, True], ids=["q4", "q3"])
@pytest.mark.parametrize("cards,sp,iters", [(5, gs.spec, 8), (4, gs.small_turn_spec, 2)], ids=["river", "turn"])
def test_terminals64_solves_match_the_reference(cards, sp, iters, q3):
    """`terminals64` in a solve: float64 traversals against the oracle from a
    synced state, and float32 graph steps against eager ones bit for bit."""
    hands = gs.Hands(gs.board(19)[:cards])
    tree = gs.Tree(sp(), hands)
    dec = [x for x in tree.nodes if x.kind == "D"]
    ranges = [gs.range_(hands.n, 12), gs.range_(hands.n, 13)]
    if not q3:  # float64 solves run Q3's kernels either way: once is enough
        _synced_against_the_reference(gs.RefSolver(hands, tree, ranges),
                                      gf.FusedSolver(hands, tree, ranges, dtype=torch.float64, terminals64=True), dec,
                                      iters)
    a = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, terminals64=True, q3=q3)
    b = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, terminals64=True, q3=q3, graph=False)
    assert a.ff is not q3 and b.ff is not q3
    for _ in range(20):
        a.step()
        b.begin()
        b.traverse(0)
        b.traverse(1)
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)


@pytest.mark.parametrize("cards,sp,iters", [(5, gs.spec, 8), (4, gs.small_turn_spec, 2)], ids=["river", "turn"])
def test_gamma_matches_the_reference(cards, sp, iters):
    """The strategy-sum exponent option (default 2 = search.rs) against the
    oracle with the same exponent, per traversal from a synced state; and
    `step()` (graph, deferred discount) against begin/traverse bit for bit."""
    hands = gs.Hands(gs.board(19)[:cards])
    tree = gs.Tree(sp(), hands)
    dec = [x for x in tree.nodes if x.kind == "D"]
    ranges = [gs.range_(hands.n, 12), gs.range_(hands.n, 13)]
    _synced_against_the_reference(gs.RefSolver(hands, tree, ranges, gamma=5),
                                  gf.FusedSolver(hands, tree, ranges, dtype=torch.float64, gamma=5), dec, iters)
    a = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, gamma=5)
    b = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, gamma=5, graph=False)
    two = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, graph=False)
    for _ in range(20):
        a.step()
        b.begin()
        b.traverse(0)
        b.traverse(1)
        two.step()
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)
    assert torch.equal(a.regret, two.regret) and not torch.equal(a.sum, two.sum)  # only the sums weigh differently
    with pytest.raises(ValueError):
        gf.FusedSolver(hands, tree, ranges, gamma=2.5)


SCHEDULES = {"dcfr+": dict(gamma=4, plus=True), "postflop-solver": dict(gamma=3, shift=2, restart=True)}


@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("cards,sp,iters", [(5, gs.spec, 18), (4, gs.small_turn_spec, 5)], ids=["river", "turn"])
def test_schedules_match_the_reference(cards, sp, iters, schedule):
    """The other schedules (off by default) against the oracle's, per traversal
    from a synced state, past the restarts at t - 1 = 1, 4 and 16; `step()`
    against begin/traverse bit for bit; DCFR+'s regrets never below 0."""
    kw = SCHEDULES[schedule]
    hands = gs.Hands(gs.board(19)[:cards])
    tree = gs.Tree(sp(), hands)
    dec = [x for x in tree.nodes if x.kind == "D"]
    ranges = [gs.range_(hands.n, 12), gs.range_(hands.n, 13)]
    ref = gs.RefSolver(hands, tree, ranges, **kw)
    gpu = gf.FusedSolver(hands, tree, ranges, dtype=torch.float64, **kw)
    for half in range(2 * iters):
        if half % 2 == 0:
            ref.begin()
            gpu.begin()
        ref.traverse(half % 2)
        gpu.traverse(half % 2)
        regret, sums = gpu.regret.cpu().numpy(), gpu.sum.cpu().numpy()
        for x in dec:
            k = len(x.acts)
            for want, got in ((ref.regret[x.slot], regret[x.slot, :k]), (ref.sum[x.slot], sums[x.slot, :k])):
                assert np.allclose(got, want, rtol=1e-11, atol=1e-11 * max(1.0, float(np.abs(want).max())))
            assert not ref.plus or (ref.regret[x.slot] >= 0).all()
            regret[x.slot, :k] = ref.regret[x.slot]
            sums[x.slot, :k] = ref.sum[x.slot]
        gpu.regret.copy_(torch.as_tensor(regret))
        gpu.sum.copy_(torch.as_tensor(sums))
    a = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, **kw)
    b = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, graph=False, **kw)
    dcfr = gf.FusedSolver(hands, tree, ranges, dtype=torch.float32, gamma=kw["gamma"], graph=False)
    for _ in range(20):
        a.step()
        b.begin()
        b.traverse(0)
        b.traverse(1)
        dcfr.step()
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)
    assert not torch.equal(a.regret, dcfr.regret)
    assert not a.plus or (a.regret >= 0).all()
    with pytest.raises(ValueError):
        gf.FusedSolver(hands, tree, ranges, shift=-1)


def test_restart_weights():
    """The restart schedule's sum weights: 0 (a restart) at t - 1 = 0, 1, 4,
    16, 64; else (tg / (tg + 1)) ** gamma, tg counted from the restart."""
    hands = gs.Hands(gs.board(19)[:5])
    ref = gs.RefSolver(hands, gs.Tree(gs.spec(), hands), [gs.range_(hands.n, 1)] * 2, gamma=3, shift=2, restart=True)
    w = []
    for _ in range(70):
        ref.begin()
        w.append(ref.sum_w)
    zero = [t for t, x in enumerate(w, 1) if x == 0.0]
    assert zero == [1, 2, 5, 17, 65]
    assert w[2] == pytest.approx((1 / 2) ** 3) and w[69] == pytest.approx((5 / 6) ** 3)  # t - 1 = 2, 69


def test_a_frozen_node_is_not_a_best_response_choice():
    hands, tree, dec, frozen = _frozen_setup(5, gs.spec, 11)
    ranges = [gs.range_(hands.n, 5), gs.range_(hands.n, 6)]
    s = gf.FusedSolver(hands, tree, ranges, dtype=torch.float64)
    for slot, st in frozen:
        s.freeze(slot, st)
    for _ in range(200):
        s.step()
    # The frozen player's best response cannot beat the profile at the frozen
    # root, so BR values stay close to the profile values: exploitability small.
    e = s.exploitability()
    assert -1e-6 < e / 2000.0 < 0.01, e
    assert abs(s.value(0) + s.value(1)) < 1e-6


def test_subgame_jobs_solve_like_the_reference():
    hands, tree, dec, frozen = _frozen_setup(5, gs.spec, 13)
    ranges = [gs.range_(hands.n, 7), gs.range_(hands.n, 8)]
    want = [dec[0].slot, dec[3].slot, dec[-1].slot]
    job = gf.job_from_tree(hands, tree, ranges, iters=4, want=want, frozen=frozen)
    # Through the same form the Rust search sends (and without eval7 here).
    job = dict(job, strength=hands.strength.astype(np.int64).reshape(-1).tolist())
    got = gf.SubgameGpu(dtype=torch.float64)(job)
    ref = gs.RefSolver(hands, tree, ranges)
    for slot, st in frozen:
        ref.freeze(slot, st)
    for _ in range(4):
        ref.step()
    fz = dict(frozen)
    for w, g in zip(want, got):
        if w in fz:
            want_avg = fz[w]
        else:
            s = ref.sum[w]
            tot = s.sum(0)
            want_avg = np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / s.shape[0])
        np.testing.assert_allclose(np.asarray(g).reshape(want_avg.shape), want_avg, rtol=1e-9, atol=1e-12)


def test_subgame_gpu_passes_terminals64_on(monkeypatch):
    # VERIFY V5's arm F: `SubgameGpu(terminals64=True, q3=True)` solves with
    # Q3's kernels, `terminals64` alone with Q4's (their own tests are above);
    # the default stays the default kernels.
    hands, tree, dec, frozen = _frozen_setup(5, gs.spec, 13)
    ranges = [gs.range_(hands.n, 7), gs.range_(hands.n, 8)]
    want = [dec[0].slot, dec[3].slot, dec[-1].slot]
    job = gf.job_from_tree(hands, tree, ranges, iters=20, want=want, frozen=frozen)
    job = dict(job, strength=hands.strength.astype(np.int64).reshape(-1).tolist())
    seen, real = [], gf.FusedSolver

    def spy(*a, **kw):
        s = real(*a, **kw)
        seen.append((kw["terminals64"], kw["q3"], s.ff))
        return s

    monkeypatch.setattr(gf, "FusedSolver", spy)
    for kw in (dict(terminals64=True, q3=True), dict(terminals64=True), {}):
        got = gf.SubgameGpu(**kw)(job)
        assert len(got) == len(want)
        for w, g in zip(want, got):
            g = np.asarray(g).reshape(-1, hands.n)
            assert np.isfinite(g).all() and (g >= 0).all()
            np.testing.assert_allclose(g.sum(0), 1.0, atol=1e-5)
    assert seen == [(True, True, False), (True, False, True), (False, False, False)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_kept_sums_average_like_a_full_solve(dtype, graph):
    """`sums=want` (the live search's path) skips the other slots' sum
    updates; the wanted averages and every regret are unchanged bit for bit."""
    for cards, sp in ((4, gs.small_turn_spec), (5, gs.spec)):
        hands, tree, dec, frozen = _frozen_setup(cards, sp, 19)
        ranges = [gs.range_(hands.n, 10), gs.range_(hands.n, 11)]
        want = [dec[0].slot, dec[3].slot, dec[-1].slot]  # the frozen root among them
        full = gf.FusedSolver(hands, tree, ranges, dtype=dtype, graph=graph)
        kept = gf.FusedSolver(hands, tree, ranges, dtype=dtype, graph=graph, sums=want)
        assert kept.sum_slots == set(want)
        for s in (full, kept):
            for slot, st in frozen:
                s.freeze(slot, st)
            for _ in range(30):
                s.step()
        assert torch.equal(kept.average(want), full.average(want))
        assert torch.equal(kept.regret, full.regret) and torch.equal(kept.sum[want], full.sum[want])
        others = [x.slot for x in dec if x.slot not in want and x.slot not in dict(frozen)]
        assert others and not kept.sum[others].any()


def test_kept_sums_refuse_the_rest():
    hands = gs.Hands(gs.board(5)[:5])
    tree = gs.Tree(gs.spec(), hands)
    ranges = [gs.range_(hands.n, 1), gs.range_(hands.n, 2)]
    D = sum(x.kind == "D" for x in tree.nodes)
    for bad in ([D], [-1], [0, 1.5]):
        with pytest.raises(ValueError):
            gf.FusedSolver(hands, tree, ranges, sums=bad)
    assert gf.FusedSolver(hands, tree, ranges, sums=range(D)).sum_slots is None  # all kept: the default kernels
    s = gf.FusedSolver(hands, tree, ranges, sums=[0, 2])
    s.step()
    s.average([2, 0])
    assert torch.equal(s.average(torch.tensor([2 - D, 0], device=s.sum.device)), s.average([2, 0]))  # as torch indexes
    for read in (lambda: s.average(), lambda: s.average([0, 1]), lambda: s.average([1 - D]), s.exploitability,
                 lambda: s.value(0)):
        with pytest.raises(ValueError):
            read()


def test_kept_sums_guard_reads_slots_as_torch_indexes():
    """The read guard checks the slots `sum[slots]` reads, however they are
    given: a bool mask or a slice is not a list of slot numbers."""
    hands = gs.Hands(gs.board(5)[:5])
    tree = gs.Tree(gs.spec(), hands)
    ranges = [gs.range_(hands.n, 1), gs.range_(hands.n, 2)]
    D = sum(x.kind == "D" for x in tree.nodes)
    s = gf.FusedSolver(hands, tree, ranges, sums=[0, 1])
    s.step()
    kept = torch.zeros(D, dtype=torch.bool)
    kept[[0, 1]] = True
    assert torch.equal(s.average(kept), s.average([0, 1])) and torch.equal(s.average(slice(0, 2)), s.average([0, 1]))
    other = torch.zeros(D, dtype=torch.bool)
    other[2] = True  # as numbers {0, 1}, both kept: the old guard let it read slot 2's empty sum
    for read in (lambda: s.average(other), lambda: s.average(other.to(s.sum.device)), lambda: s.average(slice(1, 3)),
                 lambda: s.average((Ellipsis, [0])), lambda: s.average((Ellipsis, 0)),  # `...`: every slot
                 lambda: s.average((slice(None), slice(1, None))), lambda: s.average((2, slice(1, None)))):
        with pytest.raises(ValueError):
            read()


def test_concurrent_subgame_solves_equal_serial_ones():
    # The search's match threads call the solver at once (torch drops the GIL);
    # solves must not break each other's CUDA-graph capture.
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for seed, (cards, sp) in enumerate([(5, gs.spec), (4, gs.small_turn_spec)] * 3):
        hands = gs.Hands(gs.board(seed + 40)[:cards])
        tree = gs.Tree(sp(), hands)
        slots = [x.slot for x in tree.nodes if x.kind == "D"]
        ranges = [gs.range_(hands.n, seed), gs.range_(hands.n, seed + 9)]
        jobs.append(gf.job_from_tree(hands, tree, ranges, iters=20, want=slots[:2]))
    gpu = gf.SubgameGpu()
    serial = [gpu(j) for j in jobs]
    with ThreadPoolExecutor(6) as ex:
        concurrent = list(ex.map(gpu, jobs * 2))
    for got, want in zip(concurrent, serial * 2):
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)


def _search_agents(from_turn, iters, gpu):
    pe = pytest.importorskip("parciver_engine")
    base = pe.PlayAgent.uniform("100bb")
    try:
        # rescue=False: a failed GPU solve must fail the test, not fall back to
        # the built-in solver (which would match the CPU agent trivially).
        with_gpu = base.with_search(from_turn=from_turn, river_iters=iters, turn_iters=iters, solver=gpu,
                                    rescue=False)
    except TypeError:
        pytest.skip("the installed engine predates external subgame solvers (rebuild it)")
    return base.with_search(from_turn=from_turn, river_iters=iters, turn_iters=iters), with_gpu


# (hero, actions): seat 0 is the button/SB; postflop seat 1 acts first.
_TO_TURN = [("c", 0), ("x", 0), ("x", 0), ("x", 0)]
_TO_RIVER = _TO_TURN + [("x", 0), ("x", 0)]
RIVER_SPOTS = [
    (1, _TO_RIVER),
    (0, _TO_RIVER + [("b", 150)]),
    (1, _TO_RIVER + [("x", 0), ("b", 300)]),
    (1, _TO_RIVER + [("b", 100), ("b", 400)]),  # its own bet this round is frozen
    (0, _TO_RIVER + [("b", 100), ("b", 400), ("b", 1200)]),
]
TURN_SPOTS = [(1, _TO_TURN), (0, _TO_TURN + [("b", 200)]), (1, _TO_TURN + [("b", 150), ("b", 500)])]


@pytest.mark.parametrize("street", ["river", "turn"])
def test_agent_decisions_with_the_gpu_solver_equal_the_cpu_search(street):
    # The uniform blueprint ties every range value, so these turn subgames are
    # ill-conditioned: past two iterations a 1e-13 change to the ranges moves
    # the answer by up to 1e-2, and two summation orders drift apart. Two turn
    # iterations still run every part of the path (chance nodes, both
    # updates, the averages) and must agree exactly.
    gpu = gf.SubgameGpu(dtype=torch.float64)
    cpu_agent, gpu_agent = _search_agents(street == "turn", 25 if street == "river" else 2, gpu)
    spots = RIVER_SPOTS if street == "river" else TURN_SPOTS
    checked = 0
    for seed in range(3):
        cards = [int(c) for c in np.random.default_rng(seed).permutation(52)[:9]]
        holes = [(cards[0], cards[1]), (cards[2], cards[3])]
        board = cards[4:9] if street == "river" else cards[4:8]
        for hero, actions in spots:
            kw = dict(hero=hero, hole=holes[hero], board=board, actions=actions, sb=50, bb=100,
                      stacks=(10_000, 10_000), seed=seed)
            a, b = cpu_agent.decide(**kw), gpu_agent.decide(**kw)
            assert a["source"] == b["source"] == "search", (a["off_tree_reason"], b["off_tree_reason"])
            assert a["labels"] == b["labels"]
            np.testing.assert_allclose(b["probs"], a["probs"], atol=1e-7 if street == "river" else 1e-12)
            checked += 1
    # At least one solve per decision; more where the searcher's own earlier
    # node this round was searched too (its frozen strategy).
    assert gpu.solves >= checked

def test_float16_regrets_converge_like_float32():
    # 16-bit regrets run on unit-mass ranges and pot-unit chips; the reported
    # values come back in chips per dealt pair.
    hands = gs.Hands(gs.board(7))
    tree = gs.Tree(gs.spec(), hands)
    ranges = [gs.range_(hands.n, 0), gs.range_(hands.n, 50)]
    full = gf.FusedSolver(hands, tree, ranges)
    half = gf.FusedSolver(hands, tree, ranges, regret_dtype=torch.float16)
    assert half.regret.dtype == torch.float16 and half.chip == 2000.0
    for _ in range(300):
        full.step()
        half.step()
    e_full, e_half = full.exploitability() / 2000.0, half.exploitability() / 2000.0
    assert -1e-3 < e_half < 0.005 and abs(e_half - e_full) < 0.001, (e_full, e_half)
    assert half.value(0) == pytest.approx(full.value(0), abs=2.0)


@pytest.mark.parametrize("fail", [False, True])
def test_the_subgame_solver_is_freed_inside_the_gpu_lock(monkeypatch, fail):
    # RIVER R1 Amendment 2 (as tests/test_flop_search.py's): the solver and its CUDA graph go before the lock is
    # released, also when the solve fails after its capture.
    import weakref

    events = []

    class Lock:
        def __enter__(self):
            events.append("lock")

        def __exit__(self, *exc):
            events.append("unlock")

    class Probe(gf.FusedSolver):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            weakref.finalize(self, events.append, "freed")

        def step(self):
            had = self.graph is not None
            super().step()
            if not had and self.graph is not None:
                weakref.finalize(self.graph, events.append, "graph")

        def average(self, *a, **k):
            if fail:
                raise RuntimeError("after the capture")
            return super().average(*a, **k)

    hands, tree, dec, frozen = _frozen_setup(5, gs.spec, 13)
    ranges = [gs.range_(hands.n, 7), gs.range_(hands.n, 8)]
    job = gf.job_from_tree(hands, tree, ranges, iters=4, want=[dec[0].slot], frozen=frozen)
    job = dict(job, strength=hands.strength.astype(np.int64).reshape(-1).tolist())
    monkeypatch.setattr(gf, "FusedSolver", Probe)
    monkeypatch.setattr(gf, "GPU_LOCK", Lock())
    sub = gf.SubgameGpu(dtype=torch.float64)
    if fail:
        with pytest.raises(RuntimeError, match="after the capture"):
            sub(job)
    else:
        sub(job)
    assert sub.solves == (0 if fail else 1)
    assert sorted(events[-3:-1]) == ["freed", "graph"] and events[-1] == "unlock"
