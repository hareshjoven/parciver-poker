"""Turn-start values (src/parciver/turn_values.py; FLOPSEARCH.md D1, gate G1):
the situation the recorder reads from the search's own subgame jobs, root
values against a NumPy recursion, the zero-sum identity, suit symmetry, the
check-down baseline and the random range generator."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pe = pytest.importorskip("parciver_engine")
if not torch.cuda.is_available():
    pytest.skip("turn values are solved on the fused GPU solver", allow_module_level=True)

from parciver import gpu_fused as gf  # noqa: E402
from parciver import gpu_search as gs  # noqa: E402
from parciver import turn_values as tv  # noqa: E402


def _situation(board_seed=5, half=1000, stack=1500, first=1, seed=1):
    board = tuple(gs.board(board_seed)[:4])
    n = len(tv.combos(board))
    return tv.Situation(board, half, stack, first, 100, np.stack([gs.range_(n, seed), gs.range_(n, seed + 1)]))


def _ref_values(tree, hands, ranges, avg, p):
    """Per-holding counterfactual values for `p` at the root under `avg`
    ([slot, action, hand]), by plain recursion with the dense matrices."""
    W = [hands.showdown_matrix(v) for v in range(hands.views)]

    def go(i, rq):
        x = tree.nodes[i]
        if x.kind == "T":
            if x.folded is not None:
                amount = -float(x.stake[p]) if x.folded == p else float(x.stake[1 - p])
                return (hands.compat @ rq) * amount
            return (W[x.view] @ rq) * float(min(x.stake))
        if x.kind == "C":
            out = np.zeros(hands.n)
            for kid, v in zip(x.children, x.kid_views):
                live = hands.alive[v]
                out += go(kid, rq * live) * live
            return out / x.norm
        sig = avg[x.slot, :len(x.acts)]
        if x.player == p:
            return sum(sig[a] * go(c, rq) for a, c in enumerate(x.children))
        return sum(go(c, rq * sig[a]) for a, c in enumerate(x.children))

    return go(0, np.asarray(ranges[1 - p], dtype=np.float64))


def test_root_values_match_a_numpy_recursion_and_are_zero_sum():
    sit = _situation()
    hands = gs.Hands(list(sit.board))
    tree = gs.Tree(sit.spec(), hands)
    s = gf.FusedSolver(hands, tree, list(sit.ranges), dtype=torch.float64)
    for _ in range(30):
        s.step()
    values, game = tv.values_of(s, sit)
    avg = s.average().cpu().numpy()
    for p in (0, 1):
        want = _ref_values(tree, hands, sit.ranges, avg, p) / (sit.ranges[1 - p].sum() * sit.pot)
        np.testing.assert_allclose(values[p], want, rtol=0, atol=1e-12)
    # Zero-sum: with each range scaled to mass 1, r0 . v0 + r1 . v1 = 0 (pots per unit of joint mass).
    r = [sit.ranges[p] / sit.ranges[p].sum() for p in (0, 1)]
    assert abs(r[0] @ values[0] + r[1] @ values[1]) < 1e-12
    # `game` is player 0's value per card-disjoint dealt pair.
    joint = sit.ranges[0] @ hands.compatible_mass(sit.ranges[1])
    assert game == pytest.approx(sit.ranges[0] @ values[0] * sit.ranges[1].sum() / joint, abs=1e-12)


def test_suit_permutation_maps_solved_values():
    sit = _situation(board_seed=11, seed=3)
    perm = [2, 0, 3, 1]
    board2, idx = tv.holding_map(sit.board, perm)
    sit2 = tv.Situation(tuple(board2), sit.half, sit.stack, sit.first, sit.bb, sit.ranges[:, idx])
    a = tv.solve(sit, gs.Hands(list(sit.board)), 40, dtype=torch.float64)
    b = tv.solve(sit2, gs.Hands(list(sit2.board)), 40, dtype=torch.float64)
    np.testing.assert_allclose(b["values"], a["values"][:, idx], rtol=0, atol=1e-9)
    assert b["exploitability"] == pytest.approx(a["exploitability"], abs=1e-9)


def test_holding_map_relabels_suits():
    board = (0, 5, 22, 51)  # 2c 3d 7h As
    new, idx = tv.holding_map(board, [3, 2, 1, 0])
    assert new == [3, 6, 21, 48]  # 2s 3h 7d Ac
    old = tv.combos(board)
    for i, (a, b) in enumerate(tv.combos(new)):
        assert sorted(tv.permute_cards(old[idx[i]], [3, 2, 1, 0])) == [a, b]


def test_checkdown_values_are_the_all_in_solve():
    """With no chips behind, the turn subgame is the check-down."""
    sit = _situation(half=10000, stack=0, seed=7)
    hands = gs.Hands(list(sit.board))
    got = tv.checkdown_values(sit, hands)
    want = tv.solve(sit, hands, 2, dtype=torch.float64)["values"]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_random_range_is_a_distribution_over_the_order():
    order = np.random.default_rng(0).permutation(tv.HOLDINGS)
    a = tv.random_range(order, np.random.default_rng(1))
    b = tv.random_range(order, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.min() >= 0 and a.sum() == pytest.approx(1.0, abs=1e-12)
    # DeepStack's split halves the order recursively: the weaker half's share is uniform in expectation.
    shares = [tv.random_range(order, np.random.default_rng(s))[order[:564]].sum() for s in range(400)]
    assert abs(np.mean(shares) - 0.5) < 0.05


def test_strength_order_puts_the_nuts_last():
    hands = gs.Hands([48, 49, 50, 44])  # Ac Ad Ah Kc
    order = tv.strength_order(hands)
    assert sorted(order) == list(range(hands.n))
    # The case ace makes quads on every river; nothing else comes close on average.
    top = {int(c) for h in order[-47:] for c in hands.combos[h]}
    assert all(51 in hands.combos[h] for h in order[-47:]) and 51 in top


def test_recorder_reads_the_search_subgame():
    """The recorded situation is the search's own: its tree (Python port) is
    the job's tree node for node, its cards views are the job's, and its
    ranges are the job's ranges (`subgame::ranges_at` at the round's start)."""
    agent = pe.PlayAgent.uniform("200bb")
    rec = tv.Recorder(20000, 100)
    rec.keep_jobs = True
    searched = agent.with_search(from_turn=True, solver=rec)
    searched.match_against("agent", 24, 50, 100, 20000, seed=5, threads=1)
    assert rec.first_decisions and rec.situations
    for job in rec.first_decisions[:4]:
        sit = tv.root_of(job, 20000, 100)
        hands = gs.Hands(list(sit.board))
        assert np.array_equal(np.asarray(job["strength"]), hands.strength.reshape(-1))
        assert np.array_equal(np.asarray(job["alive"]), hands.alive.reshape(-1))
        flat = gf.flatten(gs.Tree(sit.spec(), hands))
        for k in ("kind", "player", "slot", "view", "child_start", "children"):
            assert list(job[k]) == list(flat[k]), k
        assert [tuple(x) for x in job["stake"]] == [tuple(x) for x in flat["stake"]]
        assert np.array_equal(sit.ranges, np.asarray(job["ranges"], dtype=np.float64))
        key = (sit.board, sit.half, sit.first, sit.ranges.tobytes())
        assert key in rec._seen
