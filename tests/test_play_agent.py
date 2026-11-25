"""Play-time agent (`parciver_engine.PlayAgent`): real hand in, legal action out.

The Rust unit tests (engine/src/play.rs) cover translation exactness on random
grid lines, arbitrary-size fuzzing and off-tree handling; these pin the Python
boundary and the blueprint lookup against a solved strategy.
"""
import pytest

pe = pytest.importorskip("parciver_engine")
if not hasattr(pe, "PlayAgent"):
    pytest.skip("engine built without PlayAgent", allow_module_level=True)

HERO_HOLE = (40, 44)  # Qc Kc in the small game's deck


@pytest.fixture(scope="module")
def small():
    strategy = pe.solve_nlhe_small(300)
    return strategy, pe.PlayAgent.from_strategy(strategy, "small")


@pytest.mark.parametrize("hero, actions", [
    (0, []),                      # SB first to act preflop
    (1, [("c", 0)]),              # BB after an SB limp
    (1, [("a", 0)]),              # BB facing an SB shove
])
def test_decision_uses_the_solved_row(small, hero, actions):
    strategy, agent = small
    d = agent.decide(hero, HERO_HOLE, [], actions, 1, 2, (4, 4), seed=5)
    assert d["source"] == "blueprint"
    row = strategy[d["key"]]
    assert d["probs"] == pytest.approx([p / sum(row) for p in row], abs=1e-6)
    assert len(d["labels"]) == len(row)


def test_rescaled_blinds_translate_to_the_same_infoset(small):
    _, agent = small
    base = agent.decide(1, HERO_HOLE, [], [("c", 0)], 1, 2, (4, 4), seed=5)
    scaled = agent.decide(1, HERO_HOLE, [], [("c", 0)], 5, 10, (20, 20), seed=5)
    assert scaled["key"] == base["key"]
    assert scaled["probs"] == base["probs"]


def test_same_inputs_same_decision(small):
    _, agent = small
    a = agent.decide(0, HERO_HOLE, [], [], 1, 2, (4, 4), seed=11)
    b = agent.decide(0, HERO_HOLE, [], [], 1, 2, (4, 4), seed=11)
    assert a == b


def test_chip_accounting(small):
    _, agent = small
    for seed in range(40):
        d = agent.decide(1, HERO_HOLE, [], [("a", 0)], 1, 2, (4, 4), seed=seed)
        if d["action"] == "call":
            assert d["add"] == 2 and d["all_in"]  # BB owes 2 of its last 2 chips
        else:
            assert d["action"] == "fold" and d["add"] == 0


@pytest.mark.parametrize("hero, hole, board, actions", [
    (1, HERO_HOLE, [], []),                 # not BB's turn preflop
    (0, (40, 40), [], []),                  # duplicate card
    (0, (40, 44), [], [("x", 0)]),          # SB cannot check facing the big blind
    (0, (40, 44), [], [("q", 0)]),          # unknown action kind
])
def test_impossible_inputs_raise(small, hero, hole, board, actions):
    _, agent = small
    with pytest.raises(ValueError):
        agent.decide(hero, hole, board, actions, 1, 2, (4, 4))


def test_full_grid_off_tree_and_missing_key_fallbacks():
    agent = pe.PlayAgent.from_strategy({}, "100bb_exact")
    # Empty strategy: an on-tree node has no row -> missing_key fallback. 32o has
    # ~0.32 equity vs a random hand; completing the SB costs 50 into 200 (0.25).
    d = agent.decide(0, (0, 5), [], [], 50, 100, (10_000, 10_000))
    assert d["source"] == "missing_key" and d["key"].startswith("P|")
    assert (d["action"], d["add"]) == ("call", 50)
    # SB opens to 3bb, BB 3-bets to 9bb (not all-in): the grid only has all-in
    # there, so after the call the flop is off-tree and the fallback plays.
    acts = [("b", 250), ("b", 800), ("c", 0)]
    d = agent.decide(1, (48, 49), [10, 20, 30], acts, 50, 100, (10_000, 10_000))
    assert d["source"] == "off_tree" and d["off_tree_reason"]
    assert d["action"] == "check"  # BB acts first postflop; free -> check


@pytest.mark.parametrize("stack", [20_000, 10_000])
def test_deep_grid_keeps_three_bet_pots_on_the_tree(stack):
    """The 200bb grid has sized 3-/4-/5-bets: facing a 10bb 3-bet the agent can
    call or 4-bet (the 100bb grid only folds or shoves), and the called 3-bet
    pot is still played from the blueprint on the flop (not the fallback) --
    also at 100bb real stacks, where translation maps onto the deep tree."""
    agent = pe.PlayAgent.uniform("200bb")
    three_bet = [("b", 250), ("b", 900)]  # SB opens to 3bb, BB 3-bets to 10bb
    d = agent.decide(0, (48, 49), [], three_bet, 50, 100, (stack, stack), seed=1)
    assert d["source"] == "blueprint", d["off_tree_reason"]
    assert set(d["labels"]) >= {"fold", "call", "allin"}
    assert any(label not in ("fold", "call", "allin") for label in d["labels"]), d["labels"]
    flop = agent.decide(0, (48, 49), [0, 17, 34], three_bet + [("c", 0), ("b", 667)], 50, 100,
                        (stack, stack), seed=1)
    assert flop["source"] == "blueprint", flop["off_tree_reason"]
    # The 100bb grid in the same spot: fold or shove, then off the tree.
    shallow = pe.PlayAgent.uniform()
    d = shallow.decide(0, (48, 49), [], three_bet, 50, 100, (stack, stack), seed=1)
    assert d["labels"] == ["fold", "allin"]
    flop = shallow.decide(0, (48, 49), [0, 17, 34], three_bet + [("c", 0), ("b", 667)], 50, 100,
                          (stack, stack), seed=1)
    assert flop["source"] == "off_tree"


def test_unknown_grid_is_rejected():
    with pytest.raises(ValueError, match="unknown bet grid"):
        pe.PlayAgent.uniform("300bb")


def test_river_search_decides_with_real_sizes():
    """With search, a river decision comes from the exact-card subgame (source
    "search") and an off-grid river bet is faced at its real size."""
    agent = pe.PlayAgent.uniform().with_search(river_iters=50)
    board = [0, 9, 22, 31, 44]
    # Limped, checked down to the river; BB bets an off-grid 137 into 200; SB decides.
    acts = [("c", 0), ("x", 0), ("x", 0), ("x", 0), ("x", 0), ("x", 0), ("b", 137)]
    d = agent.decide(0, (50, 51), board, acts, 50, 100, (10_000, 10_000), seed=3)
    assert d["source"] == "search", d["off_tree_reason"]
    assert d["action"] in ("fold", "call", "bet")
    assert abs(sum(d["probs"]) - 1.0) < 1e-9
    # Preflop and flop stay on the policy.
    d = agent.decide(0, (50, 51), [], [], 50, 100, (10_000, 10_000), seed=3)
    assert d["source"] == "blueprint"
    # Same public state, same seed: identical decision (deterministic, cached).
    again = agent.decide(0, (50, 51), board, acts, 50, 100, (10_000, 10_000), seed=3)
    first = pe.PlayAgent.uniform().with_search(river_iters=50).decide(0, (50, 51), board, acts, 50, 100,
                                                                       (10_000, 10_000), seed=3)
    assert again["probs"] == first["probs"]


def test_options_report_a_failed_search():
    """Finding 16: `options` raises where a failed search decided (the options
    are then the built-in solver's or the blueprint's), and `options_failure`
    says which: 0 the built-in solver in place of the external one, 1 the
    blueprint, None no failure."""
    if not hasattr(pe.PlayAgent, "options_failure"):
        pytest.skip("engine built before options_failure")

    def broken(job):
        raise RuntimeError("no GPU")

    board = [0, 9, 22, 31, 44]
    acts = [("c", 0), ("x", 0), ("x", 0), ("x", 0), ("x", 0), ("x", 0), ("b", 137)]
    args = (0, board, acts, 50, 100, (10_000, 10_000), [(50, 51), (48, 49), (1, 2)])
    plain = pe.PlayAgent.uniform()
    built_in = plain.with_search(river_iters=50)
    rescued = plain.with_search(river_iters=50, solver=broken)
    failed = plain.with_search(river_iters=50, solver=broken, rescue=False)
    for agent in (plain, built_in):
        opts, k = agent.options_failure(*args, seed=3)
        assert k is None and agent.options(*args, seed=3) == opts
    assert rescued.options_failure(*args, seed=3) == (built_in.options(*args, seed=3), 0)
    assert failed.options_failure(*args, seed=3) == (plain.options(*args, seed=3), 1)
    with pytest.raises(RuntimeError, match="built-in solver used"):
        rescued.options(*args, seed=3)
    with pytest.raises(RuntimeError, match="blueprint used"):
        failed.options(*args, seed=3)
    # Preflop is never searched; bad input stays a ValueError.
    pre = (0, [], [], 50, 100, (10_000, 10_000), [(50, 51)])
    assert failed.options_failure(*pre) == (plain.options(*pre), None)
    with pytest.raises(ValueError):
        failed.options(0, board, [("q", 0)], 50, 100, (10_000, 10_000), [(50, 51)])
