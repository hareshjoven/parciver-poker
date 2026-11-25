"""R-MAPSS §35.B — the NO-REGRET-LEARNER decider: exactness + §28 controls.

§35.B replaces the dense lagged-BR ρ with principled no-regret learners (RM / RM+ /
Hedge) adapting to the JOINT field. Because a positive ``A5 − A4`` here would be the H2
research claim, every number the decider produces must be reproducible by the CERTIFIED
game-driven oracle (``parciver.eval.exploitability``), not just the fast tree.

Anchors:
* the learner UPDATE (``cf_regrets``) equals an INDEPENDENT game-driven counterfactual-regret
  walk, bit-for-bit — so the successor opponent state cannot be a FastTree artifact;
* the composed g1 pipeline (block EV + the learner step + the myopic BR value) equals the
  certified oracle's on-policy / best-response values;
* ``cf_regrets`` is a valid no-regret update (RM self-play reaches the certified 3p-Kuhn
  NashConv regime);
* the star-vs-dense transition genuinely diverges (opp→opp coupling present ⇒ ``A5 ≠ A4`` is
  structurally possible, not forced to zero);
* the §28 exact controls (realised ≤ trajectory-BR ceiling, A5 predicted==realised, A5 ≥ A2.5).
"""
from __future__ import annotations

import pytest

from parciver.games.kuhn_multi import KuhnPokerN
from parciver.eval import exploitability as E
from parciver.rmapss.fast_tree import FastTree
from parciver.rmapss.step2 import build_shared
from parciver.rmapss.step2_bait import BaitConfig
from parciver.rmapss.step2_learn import (
    cf_regrets, rm_policy, hedge_policy, link_policy, LearnLifted,
    init_regret_for, default_learn_families, run_learn_decider,
)


# --------------------------------------------------------------------------
# the anti-hallucination gate: cf_regrets == an independent game-driven walk
# --------------------------------------------------------------------------

def _game_cf_regrets(game, profile, targets):
    """Independent counterfactual-regret computation over the CERTIFIED game object
    (game.next_state / is_terminal / chance_outcomes / current_player / infoset_key /
    legal_actions / terminal_utility_for), mirroring the standard CFR regret increment.
    A different tree walker than FastTree — agreement certifies cf_regrets against the
    game semantics, not just against itself."""
    n = game.num_players
    reg = {p: {} for p in targets}

    def traverse(state, reach, reach_c):
        if game.is_terminal(state):
            return [game.terminal_utility_for(state, i) for i in range(n)]
        if game.is_chance(state):
            val = [0.0] * n
            for a, p in game.chance_outcomes(state):
                cv = traverse(game.next_state(state, a), reach, reach_c * p)
                for i in range(n):
                    val[i] += p * cv[i]
            return val
        player = game.current_player(state)
        key = game.infoset_key(state)
        acts = game.legal_actions(state)
        na = len(acts)
        probs = profile.get(key)
        if probs is None:
            probs = [1.0 / na] * na
        avals, val = [], [0.0] * n
        for idx, a in enumerate(acts):
            cr = list(reach)
            cr[player] *= probs[idx]
            cv = traverse(game.next_state(state, a), cr, reach_c)
            avals.append(cv)
            for i in range(n):
                val[i] += probs[idx] * cv[i]
        if player in targets:
            cf = reach_c
            for j in range(n):
                if j != player:
                    cf *= reach[j]
            d = reg[player].setdefault(key, [0.0] * na)
            vp = val[player]
            for idx in range(na):
                d[idx] += cf * (avals[idx][player] - vp)
        return val

    traverse(game.initial_state(), [1.0] * n, 1.0)
    return reg


def _make_profile(ft, rng_seed):
    """A reproducible mixed profile over all decision infosets."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    prof = {}
    for i in range(len(ft.kind)):
        if ft.kind[i] == 2:
            prof[ft.dkey[i]] = [float(x) for x in rng.dirichlet([1.0] * ft.nactions[i])]
    return prof


def test_cf_regrets_matches_game_oracle():
    game = KuhnPokerN(3)
    ft = FastTree(game)
    for seed in (0, 1, 2):
        prof = _make_profile(ft, seed)
        targets = set(range(game.num_players))
        fast = cf_regrets(ft, prof, targets)
        cert = _game_cf_regrets(game, prof, targets)
        for p in targets:
            assert set(fast[p]) == set(cert[p])
            for key in fast[p]:
                assert fast[p][key] == pytest.approx(cert[p][key], abs=1e-12)


def test_g1_immediate_and_future_match_certified_oracle():
    """The composed g1 (immediate on-policy EV + BR value against the learner successor)
    equals the certified game-driven oracle."""
    from parciver.rmapss.step2_learn import LearnLifted
    game = KuhnPokerN(3)
    sh = build_shared(game, 0, cfr_iters=1500)
    for fam in default_learn_families()[:4]:
        L = LearnLifted(sh, rule=fam.rule, eta=fam.eta)
        init = init_regret_for(L, fam)
        opp = L.state_policy(init)
        # immediate EV of the myopic BR seed == certified on-policy value
        br_id, br_val = L.br_now(init)
        h = L.pol(br_id)
        assert L.block_ev(br_id, init) == pytest.approx(
            E.on_policy_value(game, {**h, **opp}, 0), abs=1e-12)
        # BR value against the (learner) successor == certified BR value
        succ = L.step(init, br_id, "dense", None)
        succ_pol = L.state_policy(succ)
        assert L.br_now(succ)[1] == pytest.approx(
            E.best_response_value(game, {**succ_pol, **h}, 0), abs=1e-12)


# --------------------------------------------------------------------------
# cf_regrets is a valid no-regret learner (convergence, not solver byte-identity)
# --------------------------------------------------------------------------

def _selfplay_avg(ft, iters, rule="rm", eta=1.0):
    n = ft.n
    regret, ssum = {}, {}

    def strat_of(key, na):
        return link_policy(rule, regret.get(key, [0.0] * na), eta)

    def acc(node, reach, profile):
        k = ft.kind[node]
        if k == 0:
            return
        if k == 1:
            for c, p in ft.cchildren[node]:
                acc(c, reach, profile)
            return
        pl, key, na, ch = ft.dplayer[node], ft.dkey[node], ft.nactions[node], ft.dchildren[node]
        s = profile.get(key) or [1.0 / na] * na
        d = ssum.setdefault(key, [0.0] * na)
        for a in range(na):
            d[a] += reach[pl] * s[a]
            cr = list(reach)
            cr[pl] *= s[a]
            acc(ch[a], cr, profile)

    for _ in range(iters):
        profile = {key: strat_of(key, len(r)) for key, r in regret.items()}
        acc(ft.root, [1.0] * n, profile)
        inc = cf_regrets(ft, profile, set(range(n)))
        for p in range(n):
            for key, r in inc[p].items():
                dd = regret.setdefault(key, [0.0] * len(r))
                for a in range(len(r)):
                    dd[a] = max(dd[a] + r[a], 0.0) if rule == "rm+" else dd[a] + r[a]
    out = {}
    for key, s in ssum.items():
        tot = sum(s)
        out[key] = [x / tot for x in s] if tot > 0 else [1.0 / len(s)] * len(s)
    return out


def test_rm_selfplay_reaches_certified_nashconv_regime():
    game = KuhnPokerN(3)
    ft = FastTree(game)
    avg = _selfplay_avg(ft, 3000, "rm")
    total_nc = sum(ft.best_response_value(avg, p) - ft.on_policy_value(avg, p)
                   for p in range(ft.n))
    # phase_audit records empirical 3p-Kuhn CFR NashConv ~3.2e-3; RM self-play lands there.
    assert total_nc < 1e-2, total_nc


def test_links_produce_valid_distributions():
    assert rm_policy([0.0, 0.0, 0.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert rm_policy([2.0, -1.0, 6.0]) == pytest.approx([0.25, 0.0, 0.75])
    h = hedge_policy([1.0, 1.0, 1.0], 1.0)
    assert h == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    for rule in ("rm", "rm+", "hedge"):
        d = link_policy(rule, [0.3, -0.2, 0.9], 1.0)
        assert d == pytest.approx(d) and abs(sum(d) - 1.0) < 1e-12 and all(x >= 0 for x in d)


# --------------------------------------------------------------------------
# structural: the opp->opp coupling exists (A5 != A4 is possible, not forced 0)
# --------------------------------------------------------------------------

def test_star_dense_transition_diverges():
    game = KuhnPokerN(3)
    sh = build_shared(game, 0, 1500)
    L = LearnLifted(sh, "rm")
    fam = [f for f in default_learn_families() if f.label == "rm:aggro|nit"][0]
    s0 = init_regret_for(L, fam)
    aggro = sh.menu["aggro"]
    hid = L.intern({k: aggro[k] for k in sh.keys_by_seat[0]})
    s1 = L.step(s0, hid, "dense", None)
    dense2 = L.step(s1, hid, "dense", None)
    star2 = L.step(s1, hid, "star", s0)      # co-opps FROZEN at s0 (an earlier ref)
    diff = max(abs(a - b) for k in L.opp_keys
               for a, b in zip(L.state_policy(dense2)[k], L.state_policy(star2)[k]))
    assert diff > 1e-6, "no opp->opp coupling: A5 would equal A4 by construction"


def test_block1_dense_equals_star_at_plan_start():
    """At the plan-start state the co-opponents' current == frozen, so the first dense and
    star steps must coincide (the cascade only appears from block>=2)."""
    game = KuhnPokerN(3)
    sh = build_shared(game, 0, 1500)
    L = LearnLifted(sh, "rm")
    fam = [f for f in default_learn_families() if f.label == "rm:aggro|nit"][0]
    s0 = init_regret_for(L, fam)
    hid = L.br_now(s0)[0]
    dense1 = L.step(s0, hid, "dense", None)
    star1 = L.step(s0, hid, "star", s0)
    assert dense1 == star1


# --------------------------------------------------------------------------
# the decider end-to-end (small, fast) + §28 controls + invariants
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kuhn_learn():
    cfg = BaitConfig(n_pure=30, n_mixed=30, coord=True, coord_budget=250, seed=7)
    return run_learn_decider(game_name="3p-kuhn", m=4, depths=(1, 2, 3), delta=0.02,
                             cfr_iters=1500, cfg=cfg, max_baits=6, do_ceiling=True,
                             ctl_horizon=4)


def test_decider_controls_all_pass(kuhn_learn):
    ctl = {k: v for k, v in kuhn_learn.gates.items() if k.startswith("ctl:")}
    assert all(ctl.values()), [k for k, v in ctl.items() if not v]


def test_decider_verdict_valid(kuhn_learn):
    assert kuhn_learn.verdict in ("FILTER-GO", "FILTER-FLAT")


def test_a5_ge_a4_and_a25_invariants(kuhn_learn):
    for f in kuhn_learn.families:
        assert f.a5 >= f.a25 - 1e-7          # correct-model MPC never below myopic
        assert f.a5_minus_a4 >= -1e-7        # correct dense model never below the star model
        assert f.ctl_traj_ok                 # realised EV <= trajectory-BR ceiling
        assert f.ctl_a5_selfconsistent       # A5 predicted-window == realised-window


def test_decider_deterministic():
    cfg = lambda: BaitConfig(n_pure=15, n_mixed=15, coord=True, coord_budget=120, seed=42)
    a = run_learn_decider(game_name="3p-kuhn", m=3, depths=(1, 2), cfr_iters=1200,
                          cfg=cfg(), max_baits=4, do_ceiling=False, ctl_horizon=3,
                          families=default_learn_families()[:4])
    b = run_learn_decider(game_name="3p-kuhn", m=3, depths=(1, 2), cfr_iters=1200,
                          cfg=cfg(), max_baits=4, do_ceiling=False, ctl_horizon=3,
                          families=default_learn_families()[:4])
    va = [f.a5_minus_a4 for f in a.families]
    vb = [f.a5_minus_a4 for f in b.families]
    assert va == pytest.approx(vb, abs=1e-12)
