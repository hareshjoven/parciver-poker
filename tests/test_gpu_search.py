"""GPU subgame solver prototype (src/parciver/gpu_search.py): its showdown
matrices, its level-synchronous DCFR against the recursive reference (the
Rust solver's algorithm) in float64, convergence, and the turn chance node
against brute-force equity. Runs on CUDA when present, else on CPU torch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pe = pytest.importorskip("parciver_engine")

from parciver import gpu_search as gs  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _fused_available():
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return DEV == "cuda"


SOLVERS = [pytest.param("torch", id="torch"),
           pytest.param("fused", id="fused", marks=pytest.mark.skipif(not _fused_available(), reason="needs CUDA + triton"))]


def _solver(kind, hands, tree, ranges, dtype):
    if kind == "fused":
        from parciver import gpu_fused
        return gpu_fused.FusedSolver(hands, tree, ranges, dtype=dtype)
    return gs.GpuSolver(hands, tree, ranges, device=DEV, dtype=dtype)


def test_showdown_matrix_matches_the_naive_definition():
    hands = gs.Hands(gs.board(0)[:4])  # a turn: 48 views with dead holdings
    reach = gs.range_(hands.n, 5)
    for v in (0, 17, 47):
        fast = hands.showdown_matrix(v) @ reach
        live = hands.alive[v]
        for h in range(0, hands.n, 37):
            a, b = hands.combos[h]
            want = 0.0
            if live[h]:
                for j, (c, d) in enumerate(hands.combos):
                    if live[j] and len({a, b, c, d}) == 4:
                        want += reach[j] * np.sign(hands.strength[v, h] - hands.strength[v, j])
            assert fast[h] == pytest.approx(want, abs=1e-9)


def _check_against_reference(kind, hands, tree, iters):
    """Every player's traversal from an identical state: run it in both
    solvers, compare, then copy the reference's state into the GPU solver.
    (Free-running, the two drift apart: regret matching is discontinuous where
    a regret that is exactly 0 in real arithmetic crosses zero by rounding, and
    one player's regret updates feed the other's traversal.)"""
    ranges = [gs.range_(hands.n, 31), gs.range_(hands.n, 32)]
    ref = gs.RefSolver(hands, tree, ranges)
    gpu = _solver(kind, hands, tree, ranges, torch.float64)
    dec = [x for x in tree.nodes if x.kind == "D"]
    for half in range(2 * iters):
        if half % 2 == 0:
            ref.begin()
            gpu.begin()
        ref.traverse(half % 2)
        gpu.traverse(half % 2)
        regret = gpu.regret.cpu().numpy()
        sums = gpu.sum.cpu().numpy()
        for x in dec:
            k = len(x.acts)
            for name, want, got in (("regret", ref.regret[x.slot], regret[x.slot, :k]),
                                    ("sum", ref.sum[x.slot], sums[x.slot, :k])):
                scale = max(1.0, float(np.abs(want).max()))
                assert np.allclose(got, want, rtol=1e-11, atol=1e-11 * scale), f"iter {ref.iter} slot {x.slot} {name}"
            regret[x.slot, :k] = ref.regret[x.slot]
            sums[x.slot, :k] = ref.sum[x.slot]
        gpu.regret.copy_(torch.as_tensor(regret))
        gpu.sum.copy_(torch.as_tensor(sums))


@pytest.mark.parametrize("kind", SOLVERS)
def test_gpu_solver_reproduces_the_reference_on_a_river(kind):
    hands = gs.Hands(gs.board(4))
    tree = gs.Tree(gs.spec(), hands)
    assert tree.slots > 20
    _check_against_reference(kind, hands, tree, 12)


@pytest.mark.parametrize("kind", SOLVERS)
def test_gpu_solver_reproduces_the_reference_on_a_turn(kind):
    b = gs.board(9)
    hands = gs.Hands(b[:4])
    tree = gs.Tree(gs.small_turn_spec(), hands)
    _check_against_reference(kind, hands, tree, 2)


@pytest.mark.parametrize("kind", SOLVERS)
def test_river_solve_converges_in_float32(kind):
    hands = gs.Hands(gs.board(7))
    tree = gs.Tree(gs.spec(), hands)
    ranges = [gs.range_(hands.n, 0), gs.range_(hands.n, 50)]
    s = _solver(kind, hands, tree, ranges, torch.float32)
    for _ in range(300):
        s.step()
    e = s.exploitability()
    assert -1e-3 < e / 2000.0 < 0.005, f"exploitability {100 * e / 2000:.3f}% of pot"
    assert abs(s.value(0) + s.value(1)) < 1e-2


@pytest.mark.parametrize("kind", SOLVERS)
def test_turn_all_in_value_matches_brute_force_equity(kind):
    b = gs.board(5)
    hands = gs.Hands(b[:4])
    sp = gs.small_turn_spec()
    sp.stacks = (0, 0)
    tree = gs.Tree(sp, hands)

    def sparse(seed, keep=25):
        r = np.zeros(hands.n)
        s = seed
        for _ in range(keep):
            s = gs.splitmix(s)
            r[s % hands.n] = 0.1 + gs.unit(gs.splitmix(s ^ 77))
        return r

    ranges = [sparse(1), sparse(2)]
    s = _solver(kind, hands, tree, ranges, torch.float64)
    num = den = 0.0
    for i, h in enumerate(hands.combos):
        for j, q in enumerate(hands.combos):
            if ranges[0][i] == 0 or ranges[1][j] == 0 or len({*h, *q}) < 4:
                continue
            ev = []
            for c in range(52):
                if c in b[:4] or c in (*h, *q):
                    continue
                full = [*b[:4], c]
                ev.append(np.sign(pe.eval7([int(h[0]), int(h[1]), *full]) - pe.eval7([int(q[0]), int(q[1]), *full])))
            assert len(ev) == 44
            w = ranges[0][i] * ranges[1][j]
            num += w * 1000.0 * float(np.mean(ev))
            den += w
    brute = num / den
    assert abs(brute) > 1.0, "degenerate test"
    assert s.value(0) == pytest.approx(brute, abs=1e-9)
    assert s.value(0) + s.value(1) == pytest.approx(0.0, abs=1e-9)
