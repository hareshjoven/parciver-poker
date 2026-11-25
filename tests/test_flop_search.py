"""Depth-limited flop search (src/parciver/flop_search.py; FLOPSEARCH.md D3):
runout equity against brute force, the check-down evaluator against the D1
baseline, the net adapter against `value_net.predict`, and the leaf plumbing:
with exact check-down leaves the GPU depth-limited solve equals the Rust
solver's full solve of the game with no betting after the flop (the agent's
decisions, float64). Also: the searched agent's decisions with a net, and
the flop search off leaves every decision unchanged."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pe = pytest.importorskip("parciver_engine")
if not torch.cuda.is_available():
    pytest.skip("the flop search runs on the GPU", allow_module_level=True)
if not hasattr(pe, "flop_runouts"):
    pytest.skip("the installed engine predates the flop search (rebuild it)", allow_module_level=True)

from parciver import flop_search as fs  # noqa: E402
from parciver import gpu_fused as gf  # noqa: E402
from parciver import gpu_search as gs  # noqa: E402
from parciver import turn_values as tv  # noqa: E402
from parciver import value_net as vn  # noqa: E402

FLOP = (3, 22, 41)


def _strength(cards) -> int:
    return pe.eval7([int(c) for c in cards])


def test_runout_equity_matches_brute_force():
    e = fs.runout_equity(FLOP, dtype=torch.float64).cpu().numpy()
    hold = fs.combos(FLOP)
    assert e.shape == (1176, 1176)
    np.testing.assert_array_equal(e, -e.T)
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 6:
        h, q = rng.integers(0, 1176, 2)
        a, b = hold[h], hold[q]
        if len({*a, *b}) < 4:
            assert e[h, q] == 0.0
            continue
        rest = [c for c in range(52) if c not in (*FLOP, *a, *b)]
        wins = sum(np.sign(_strength([*a, *FLOP, t, r]) - _strength([*b, *FLOP, t, r]))
                   for i, t in enumerate(rest) for r in rest[i + 1:])
        assert len(rest) == 45
        assert e[h, q] == pytest.approx(wins / 990, abs=1e-15)
        checked += 1


def test_checkdown_values_are_the_d1_baseline():
    # The exact evaluator in the net's interface, on each turn board of the
    # flop, equals turn_values' check-down baseline (the G2 baseline's units).
    ev = fs.CheckdownValues(FLOP, dtype=torch.float64)
    leaves = fs.NetLeaves(FLOP, ev, dtype=torch.float64)
    rng = np.random.default_rng(1)
    raw = torch.as_tensor(rng.random((2, 2, 1176)))  # 2 leaves' reaches
    r = raw[:, None] * leaves.mask[None, :, None, :].cpu()
    x = torch.zeros(2, 49, 2, 1326, dtype=torch.float64)
    x[..., leaves.idx.cpu()] = r / r.sum(-1, keepdim=True)
    got = ev(x.cuda(), leaves.board, leaves.live, torch.ones(2).cuda(), torch.ones(2).cuda()).cpu().numpy()
    for t in (0, 17, 48):
        board = (*FLOP, leaves.turns[t])
        hands = gs.Hands(list(board))
        idx = vn.PAIR_INDEX[hands.combos[:, 0], hands.combos[:, 1]]
        for leaf in (0, 1):
            ranges = x[leaf, t][:, idx].numpy()
            sit = tv.Situation(board, 500, 1000, 1, 100, ranges)
            np.testing.assert_allclose(got[leaf, t][:, idx], tv.checkdown_values(sit, hands), rtol=1e-12, atol=1e-14)


def test_net_values_match_predict():
    torch.manual_seed(0)
    net = vn.ValueNet(64, 2).cuda().eval()
    leaves = fs.NetLeaves(FLOP, fs.net_values(net))
    rng = np.random.default_rng(2)
    r = torch.as_tensor(rng.random((3, 49, 2, 1176)), dtype=torch.float32).cuda() * leaves.mask[None, :, None, :]
    r = r / r.sum(-1, keepdim=True)
    x = torch.zeros(3, 49, 2, 1326, device="cuda")
    x[..., leaves.idx] = r
    half, stack = torch.tensor([150.0, 400.0, 1000.0]).cuda(), torch.tensor([9850.0, 9600.0, 500.0]).cuda()
    got = fs.net_values(net)(x, leaves.board, leaves.live, half, stack).cpu().numpy()
    for leaf in range(3):
        for t in (0, 30):
            board = np.array([[*FLOP, leaves.turns[t]]])
            live = vn.live_mask(board)[0]
            want = vn.predict(net, board, x[leaf, t][:, torch.as_tensor(live).cuda()].cpu().numpy()[None],
                              [half[leaf].item()], [stack[leaf].item()])[0]
            np.testing.assert_allclose(got[leaf, t][:, live], want, rtol=1e-5, atol=1e-6)
            assert (got[leaf, t][:, ~live] == 0).all()


# (hero, actions): seat 0 is the button/SB; postflop seat 1 acts first.
_PRE = [("c", 0), ("x", 0)]
FLOP_SPOTS = [
    (1, _PRE),
    (0, _PRE + [("x", 0)]),
    (1, _PRE + [("x", 0), ("b", 150)]),  # its own check this round is frozen
    (0, _PRE + [("b", 200)]),
    (1, _PRE + [("b", 100), ("b", 400)]),
]


def _decisions(agent, seeds=(0, 1), spots=FLOP_SPOTS, cards=3):
    out = []
    for seed in seeds:
        deal = [int(c) for c in np.random.default_rng(seed).permutation(52)[:4 + cards]]
        holes = [(deal[0], deal[1]), (deal[2], deal[3])]
        for hero, actions in spots:
            out.append(agent.decide(hero=hero, hole=holes[hero], board=deal[4:], actions=actions, sb=50, bb=100,
                                    stacks=(10_000, 10_000), seed=seed))
    return out


@pytest.mark.parametrize("iters", [2, 12])
def test_checkdown_leaves_solve_like_the_full_game(iters):
    # The leaf plumbing (card removal per turn card, the 1/45 average, the
    # masses and the pot) against the Rust full solve of the same game.
    base = pe.PlayAgent.uniform("100bb")
    gpu = fs.FlopSearch(net=None, dtype=torch.float64)
    cpu = base.with_search(from_flop=True, flop_iters=iters, flop_checkdown=True)
    for a, b in zip(_decisions(cpu), _decisions(base.with_search(from_flop=True, flop_iters=iters, solver=gpu))):
        assert a["source"] == b["source"] == "search", (a["off_tree_reason"], b["off_tree_reason"])
        assert a["labels"] == b["labels"]
        np.testing.assert_allclose(b["probs"], a["probs"], atol=1e-12)
    assert gpu.solves >= len(FLOP_SPOTS) * 2


def _root_job():
    """The flop round of FLOP from its first action, 30bb behind, random ranges."""
    captured = []

    def grab(job):
        captured.append(job)
        return fs.FlopSearch(net=None, dtype=torch.float64)(job)

    pe.PlayAgent.uniform("100bb").with_search(from_flop=True, flop_iters=1, solver=grab).decide(
        hero=1, hole=(0, 1), board=list(FLOP), actions=_PRE, sb=50, bb=100, stacks=(3000, 3000), seed=0)
    rng = np.random.default_rng(3)
    return dict(captured[0], ranges=(rng.random(1176).tolist(), rng.random(1176).tolist()))


def test_flop_solver_converges_on_the_checkdown_game():
    job = _root_job()
    leaves = fs.NetLeaves(FLOP, fs.CheckdownValues(FLOP, dtype=torch.float64), dtype=torch.float64)
    s = fs.FlopSolver(job, leaves, fs.runout_equity(FLOP, dtype=torch.float64), dtype=torch.float64)
    assert (np.asarray(job["kind"]) == 3).any() and (np.asarray(job["kind"]) == 4).any()
    for _ in range(3):
        s.step()
    early = s.exploitability()
    for _ in range(197):
        s.step()
    e, pot = s.exploitability(), 200.0
    assert 0 <= e < early and e / pot < 0.01, (early, e)
    assert abs(s.value(0) + s.value(1)) < 1e-9


def test_graph_replay_is_bit_identical():
    # The search's default: one captured CUDA graph per iteration, replayed.
    # Same kernels on the same tensors as eager steps, across solve() calls,
    # a frozen slot, and a re-capture when the discount tables grow.
    torch.manual_seed(2)
    net = vn.ValueNet(64, 2).cuda().eval()
    job = _root_job()
    leaves, equity = fs.NetLeaves(FLOP, fs.net_values(net)), fs.runout_equity(FLOP)
    runs = []
    for graph in (False, True):
        s = fs.FlopSolver(job, leaves, equity, gamma=5.0)
        s.freeze(0, np.full((s.dk_host[0], s.n), 1.0 / s.dk_host[0]))
        s._grow(4)
        s.solve(3, graph=graph)
        s.solve(6, graph=graph)  # past the tables: they grow (and the graph is captured again)
        runs.append(s)
    a, b = runs
    assert b.graph is not None and a.iter == b.iter == 9
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum)


def test_factored_leaves_give_the_traverser_alone_the_same_bits():
    # FlopSolver asks an evaluator that takes `players` (FactoredLeaves,
    # KERNELS.md N1 + N1b) for the traverser's values alone; it computes each
    # player's last stage on its own, so the solve is the one that reads the
    # traverser's row of both players' values, bit for bit.
    from parciver import flop_leaves as fl

    torch.manual_seed(2)
    net = vn.ValueNet(64, 2).cuda().eval()
    job = _root_job()
    equity = fs.runout_equity(FLOP)
    f = fl.FactoredLeaves(FLOP, net, last=torch.float64)
    runs = []
    for leaves in (f, lambda b, h, st: f(b, h, st)):
        s = fs.FlopSolver(job, leaves, equity, gamma=5.0)
        s.solve(9)
        runs.append(s)
    a, b = runs
    assert a.one and not b.one and len(a.leaf) > 0
    assert torch.equal(a.regret, b.regret) and torch.equal(a.sum, b.sum) and torch.equal(a.val, b.val)
    assert not fs.FlopSolver(job, fs.NetLeaves(FLOP, fs.net_values(net)), equity).one
    with pytest.raises(ValueError, match="leaves"):
        fs.FlopSearch(net, leaves="sparse")


@pytest.mark.parametrize("leaves", ["dense", "factored"])
def test_searched_agent_with_a_net_decides_legally(leaves):
    torch.manual_seed(1)
    net = vn.ValueNet(64, 2).cuda().eval()
    search = fs.FlopSearch(net, leaves=leaves)
    base = pe.PlayAgent.uniform("100bb")
    agent = base.with_search(from_flop=True, from_turn=True, flop_iters=20, turn_iters=10, solver=search)
    for d in _decisions(agent, seeds=range(3)):
        assert d["source"] == "search", d["off_tree_reason"]
        assert len(d["probs"]) == len(d["labels"]) and abs(sum(d["probs"]) - 1) < 1e-6
        assert min(d["probs"]) >= 0
    assert search.solves >= 3 * len(FLOP_SPOTS)
    with pytest.raises(ValueError, match="solver"):
        base.with_search(from_flop=True)


TURN_SPOTS = [(1, _PRE + [("x", 0), ("x", 0)]), (0, _PRE + [("x", 0), ("x", 0), ("b", 200)])]


def test_flop_search_off_changes_nothing():
    base = pe.PlayAgent.uniform("100bb")
    turn = gf.SubgameGpu(dtype=torch.float64)
    # rescue=False: a failed GPU solve must not pass as the built-in solver's.
    plain = base.with_search(from_turn=True, turn_iters=3, solver=turn, rescue=False)
    routed = base.with_search(from_turn=True, turn_iters=3, solver=fs.FlopSearch(net=None, turn=turn), rescue=False)
    for spots, cards in ((FLOP_SPOTS, 3), (TURN_SPOTS, 4)):
        for a, b in zip(_decisions(plain, spots=spots, cards=cards), _decisions(routed, spots=spots, cards=cards)):
            assert a == b
            assert (a["source"] == "search") == (cards == 4)



class _Lock:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lock")

    def __exit__(self, *exc):
        self.events.append("unlock")


@pytest.mark.parametrize("fail", [False, True])
def test_the_flop_solver_is_freed_inside_the_gpu_lock(monkeypatch, fail):
    # RIVER R1 Amendment 2: a FlopSolver freed after the lock resets its CUDA graph while another match thread
    # may be capturing one, which invalidates that capture (cudaErrorStreamCaptureInvalidated). Also when the
    # solve fails after its capture: the traceback must not carry the solver out of the lock.
    import weakref

    events = []

    class Probe(fs.FlopSolver):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            weakref.finalize(self, events.append, "freed")

        def solve(self, *a, **k):
            super().solve(*a, **k)
            weakref.finalize(self.graph, events.append, "graph")

        def average(self, *a, **k):
            if fail:
                raise RuntimeError("after the capture")
            return super().average(*a, **k)

    job = _root_job()
    monkeypatch.setattr(fs, "FlopSolver", Probe)
    monkeypatch.setattr(fs, "GPU_LOCK", _Lock(events))
    search = fs.FlopSearch(net=None, dtype=torch.float64)
    if fail:
        with pytest.raises(RuntimeError, match="after the capture"):
            search(job)
    else:
        search(job)
    assert search.solves == (0 if fail else 1)
    assert sorted(events[-3:-1]) == ["freed", "graph"] and events[-1] == "unlock"  # before the outer unlock
    assert events.count("lock") == events.count("unlock") >= 2  # the capture's lock nests inside
