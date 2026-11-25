# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""N-player Leduc hold'em — correctness gate (RMAPSS.md §33/§34, Step 2/3).

The decisive game for the interaction kernel. Two anchors:
* N=2 reduces BYTE-IDENTICALLY to the certified ``parciver.games.leduc`` (same
  tree, same 288 infoset keys, same utilities, exploitability -> 0, value -0.086)
  — the "recompute known Leduc structure" gate;
* N=3 is finite, chip-conserving / total zero-sum at every terminal, and its exact
  best response (the §28 control) dominates on-policy — with no reliance on a CFR
  fixpoint (no >=3p guarantee, R5).
"""
from __future__ import annotations

import pytest

from parciver.games.leduc import LeducHoldem
from parciver.games.leduc_multi import LeducHoldemN
from parciver.solvers.cfr_multi import MultiCFR
from parciver.eval.exploitability import (
    best_response_value, on_policy_value, nash_conv, exploitability,
)


def _walk_collect(game, state, path, acc):
    if game.is_terminal(state):
        acc["term"][path] = tuple(round(game.terminal_utility_for(state, p), 9)
                                  for p in range(game.num_players))
        return
    if game.is_chance(state):
        for a, p in game.chance_outcomes(state):
            acc["chance"].setdefault(path, []).append((a, round(p, 9)))
            _walk_collect(game, game.next_state(state, a), path + (("C", a),), acc)
        return
    acc["dec"][path] = (game.current_player(state), game.infoset_key(state),
                        tuple(game.legal_actions(state)))
    for a in game.legal_actions(state):
        _walk_collect(game, game.next_state(state, a), path + (a,), acc)


def test_n2_reduces_byte_identically_to_certified_leduc():
    a2 = {"term": {}, "chance": {}, "dec": {}}
    an = {"term": {}, "chance": {}, "dec": {}}
    _walk_collect(LeducHoldem(), LeducHoldem().initial_state(), (), a2)
    _walk_collect(LeducHoldemN(2), LeducHoldemN(2).initial_state(), (), an)

    assert set(a2["dec"]) == set(an["dec"])
    assert set(a2["term"]) == set(an["term"])
    assert set(a2["chance"]) == set(an["chance"])
    # decision nodes: same infoset key + same legal-action order at every path
    assert all(a2["dec"][p][1:] == an["dec"][p][1:] for p in a2["dec"])
    # terminals: identical per-player utilities
    assert all(a2["term"][p] == an["term"][p] for p in a2["term"])
    assert all(a2["chance"][p] == an["chance"][p] for p in a2["chance"])
    # the canonical 288 infosets
    assert len({an["dec"][p][1] for p in an["dec"]}) == 288


@pytest.fixture(scope="module")
def leduc2_solution():
    g = LeducHoldemN(2)
    s = MultiCFR(g, variant="cfr+")
    s.run(2500)
    return g, s.average_strategy()


def test_n2_converges_to_known_leduc_value_and_zero_exploitability(leduc2_solution):
    g, strat = leduc2_solution
    # CFR+ on the byte-identical 2p game drives exploitability toward 0 and the
    # on-policy value toward the known Leduc figure (-0.086), same as the certified
    # engine. (This IS a 2p zero-sum game, so the fixpoint is meaningful here.)
    assert exploitability(g, strat) < 2e-2
    assert nash_conv(g, strat) >= -1e-12
    assert abs(on_policy_value(g, strat, 0) - (-0.086)) < 0.03


def test_n3_finite_zero_sum_chip_conserving():
    g = LeducHoldemN(3)
    stats = {"term": 0, "dec": 0, "bad": 0}

    def walk(s):
        if g.is_terminal(s):
            stats["term"] += 1
            if abs(sum(g.terminal_utility_for(s, p) for p in range(3))) > 1e-9:
                stats["bad"] += 1
            return
        if g.is_chance(s):
            for a, _ in g.chance_outcomes(s):
                walk(g.next_state(s, a))
            return
        stats["dec"] += 1
        for a in g.legal_actions(s):
            walk(g.next_state(s, a))

    walk(g.initial_state())
    assert stats["bad"] == 0            # total zero-sum at every terminal
    assert stats["term"] == 33714       # structural regression
    assert stats["dec"] == 25416


def test_n3_exact_br_dominates_on_policy():
    # §28 control: a seat's exact best response can never do worse than following
    # the profile. No CFR fixpoint assumed (R5) — a raw invariant.
    g = LeducHoldemN(3)
    s = MultiCFR(g, variant="cfr+")
    s.run(60)
    strat = s.average_strategy()
    for p in range(3):
        assert best_response_value(g, strat, p) >= on_policy_value(g, strat, p) - 1e-9
