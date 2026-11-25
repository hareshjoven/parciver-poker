"""N-player Kuhn foundation (RMAPSS.md §33, Step 0.2).

Pure-Python structural + invariant checks (always run) plus the Rust<->Python
parity gate (skipped until the engine is rebuilt with the N-player API). The
independent BR-vs-brute-force ground truth (§28) lives in the Rust suite.
"""
from __future__ import annotations

import pytest

from parciver.games.kuhn import KuhnPoker
from parciver.games.kuhn_multi import KuhnPokerN
from parciver.eval.exploitability import (
    best_response_value,
    expected_value,
    nash_conv_multiplayer,
    nash_conv_per_player,
    on_policy_value,
)

try:
    import parciver_engine as pe
except Exception:  # pragma: no cover - extension optional
    pe = None

HAS_NPLAYER = pe is not None and hasattr(pe, "kuhn_n_nashconv")
_skip_engine = pytest.mark.skipif(
    not HAS_NPLAYER, reason="parciver_engine lacks the N-player API; rebuild with maturin develop"
)


def _keys(game):
    keys = set()

    def walk(s):
        if game.is_terminal(s):
            return
        if game.is_chance(s):
            for a, _ in game.chance_outcomes(s):
                walk(game.next_state(s, a))
            return
        keys.add(game.infoset_key(s))
        for a in game.legal_actions(s):
            walk(game.next_state(s, a))

    walk(game.initial_state())
    return keys


def test_infoset_counts():
    assert len(_keys(KuhnPokerN(2))) == 12
    assert len(_keys(KuhnPokerN(3))) == 48  # 12 public decision histories x 4 cards


def test_two_player_reduces_to_certified_kuhn():
    # Byte-identical infoset structure to the certified 2p engine.
    assert _keys(KuhnPokerN(2)) == _keys(KuhnPoker())


def test_zero_sum_at_every_terminal():
    for n in (2, 3, 4):
        game = KuhnPokerN(n)

        def walk(s):
            if game.is_terminal(s):
                tot = sum(game.terminal_utility_for(s, i) for i in range(n))
                assert abs(tot) < 1e-12
                return
            if game.is_chance(s):
                for a, _ in game.chance_outcomes(s):
                    walk(game.next_state(s, a))
                return
            for a in game.legal_actions(s):
                walk(game.next_state(s, a))

        walk(game.initial_state())


def test_two_player_kuhn_value_is_minus_one_eighteenth():
    # Solve KuhnN(2) with the existing 2p Python CFR solver; it must reach -1/18.
    from parciver.solvers.cfr import DCFR, CFRSolver

    game = KuhnPokerN(2)
    solver = CFRSolver(game=game, variant=DCFR, alternating=False)
    solver.run(3000)
    avg = solver.average_strategy()
    assert abs(expected_value(game, avg) + 1.0 / 18.0) < 3e-3


def test_best_response_never_worse_than_on_policy():
    # A pure-Python invariant that needs no solver: BR >= following the profile.
    game = KuhnPokerN(3)
    uniform = {k: [0.5, 0.5] for k in _keys(game)}
    for p in range(3):
        assert best_response_value(game, uniform, p) >= on_policy_value(game, uniform, p) - 1e-12


@_skip_engine
def test_rust_python_parity_3p():
    game = KuhnPokerN(3)
    strat = pe.solve_kuhn_n(3, 4000, "cfr")

    assert pe.kuhn_n_infoset_count(2) == 12
    assert pe.kuhn_n_infoset_count(3) == 48

    assert abs(pe.kuhn_n_nashconv(3, strat) - nash_conv_multiplayer(game, strat)) < 1e-9
    for p in range(3):
        assert abs(pe.kuhn_n_best_response_value(3, strat, p) - best_response_value(game, strat, p)) < 1e-9
        assert abs(pe.kuhn_n_on_policy_value(3, strat, p) - on_policy_value(game, strat, p)) < 1e-9

    rs_pp = pe.kuhn_n_nashconv_per_player(3, strat)
    py_pp = nash_conv_per_player(game, strat)
    assert all(abs(a - b) < 1e-9 for a, b in zip(rs_pp, py_pp))
    # Empirical (no >=3p guarantee): CFR drives NashConv low in 3p Kuhn.
    assert pe.kuhn_n_nashconv(3, strat) < 0.05
