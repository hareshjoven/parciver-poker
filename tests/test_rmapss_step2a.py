"""R-MAPSS §35.A — the RICH-BAIT DECIDER: exactness + §28 anti-hallucination gates.

The decider replaces §34.5's 5-archetype ρ-ceiling with a rich bait search on the
1-step-ahead Stackelberg objective. Because a positive headroom / A5−A4 here would
be a research claim, the discipline is: every number the decider produces must be
reproducible by the CERTIFIED game-driven oracle (``parciver.eval.exploitability``),
not just by the fast tree. These tests pin exactly that.

Anchors:
* the composed g1 pipeline (block EV + the dense-ρ adaptation step + the myopic BR
  value) equals an INDEPENDENT recomputation through the certified oracle — so the
  headroom cannot be a FastTree artifact;
* the structural invariants that make the decider trustworthy: BR-seed ∈ candidates
  ⇒ headroom ≥ 0; the myopic move maximises immediate EV (no candidate beats it);
  ``headroom ≤ screen_gap`` (the future gain must exceed the sacrifice); and, on
  escalation, the exact controls (A5 ≥ A2.5, ceiling ≥ A2.5, A5 ≥ A4).
"""
from __future__ import annotations

import pytest

from parciver.games.kuhn_multi import KuhnPokerN
from parciver.eval import exploitability as E
from parciver.rmapss.step2 import build_shared, lifted_for, default_families, _init_opp_profile
from parciver.rmapss.step2_bait import (
    run_decider, BaitConfig, eval_g1, random_pure, random_mixed,
)
import numpy as np


# --------------------------------------------------------------------------
# the anti-hallucination gate: the WHOLE g1 pipeline == the certified oracle
# --------------------------------------------------------------------------

def _certified_successor(game, L, hero_pol, opp_pol, eps):
    """Independently recompute the dense-ρ successor through the certified
    game-driven best response (NOT FastTree), replicating ``Lifted.step`` semantics."""
    succ = {}
    for j in L.opp_seats:
        others = dict(hero_pol)
        for k in L.opp_seats:
            if k == j:
                continue
            for key in L.keys_by_seat[k]:
                others[key] = opp_pol[key]
        br = E.best_response_strategy(game, others, j)
        for key in L.keys_by_seat[j]:
            na = len(L.actions_by_key[key])
            oh = br.get(key)
            succ[key] = ([1.0 / na] * na if oh is None
                         else [(1.0 - eps) * oh[i] + eps / na for i in range(na)])
    return succ


def test_g1_pipeline_matches_certified_oracle():
    game = KuhnPokerN(3)
    hero, eps = 0, 0.1
    sh = build_shared(game, hero, cfr_iters=1500)
    L = lifted_for(sh, eps)
    fams = default_families()
    rng = np.random.default_rng(0)
    hero_keys = L.keys_by_seat[hero]
    for fam in fams[:3]:
        opp_id = _init_opp_profile(L, fam)
        opp = L.pol(opp_id)
        # a handful of baits: the myopic BR seed + random pure + random mixed
        baits = [L.br_now(opp_id)[0],
                 L.intern(random_pure(hero_keys, L.actions_by_key, rng)),
                 L.intern(random_mixed(hero_keys, L.actions_by_key, rng, 0.5))]
        for hid in baits:
            h = L.pol(hid)
            g = eval_g1(L, hid, opp_id, "pure")
            # (a) immediate EV == certified on-policy value
            imm_cert = E.on_policy_value(game, {**h, **opp}, hero)
            assert g.immediate == pytest.approx(imm_cert, abs=1e-12)
            # (b) the dense-ρ successor profile == certified recomputation, key by key
            succ = L.pol(g.succ_id)
            succ_cert = _certified_successor(game, L, h, opp, eps)
            assert set(succ) == set(succ_cert)
            for key in succ:
                assert succ[key] == pytest.approx(succ_cert[key], abs=1e-12)
            # (c) future = BR value against that successor == certified BR value
            fut_cert = E.best_response_value(game, {**succ, **h}, hero)
            assert g.future == pytest.approx(fut_cert, abs=1e-12)


def test_myopic_seed_is_consistent_and_optimal_immediate():
    """The BR seed's immediate EV equals the exact BR value (it IS the BR), and no
    bait beats it on immediate EV — the §28 'BR maximises immediate' tripwire."""
    game = KuhnPokerN(3)
    sh = build_shared(game, 0, cfr_iters=1500)
    L = lifted_for(sh, 0.1)
    rng = np.random.default_rng(1)
    hero_keys = L.keys_by_seat[0]
    for fam in default_families()[:3]:
        opp_id = _init_opp_profile(L, fam)
        br_id, br_val = L.br_now(opp_id)
        assert L.block_ev(br_id, opp_id) == pytest.approx(br_val, abs=1e-9)
        for _ in range(40):
            hid = L.intern(random_pure(hero_keys, L.actions_by_key, rng))
            assert L.block_ev(hid, opp_id) <= br_val + 1e-9


# --------------------------------------------------------------------------
# the decider end-to-end (small, fast) + its structural invariants
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kuhn_decider():
    cfg = BaitConfig(n_pure=40, n_mixed=40, coord=True, coord_budget=300, seed=7)
    return run_decider(game_name="3p-kuhn", m=3, delta=0.02, cfr_iters=1500,
                       depth=3, n_states=3, cfg=cfg)


def test_decider_controls_all_pass(kuhn_decider):
    ctl = {k: v for k, v in kuhn_decider.gates.items() if k.startswith("ctl")}
    assert all(ctl.values()), [k for k, v in ctl.items() if not v]


def test_decider_verdict_is_valid(kuhn_decider):
    assert kuhn_decider.verdict in ("NULL-REAL", "MENU-ARTIFACT")


def test_headroom_nonneg_and_bounded_by_screen(kuhn_decider):
    """BR seed ∈ candidates ⇒ headroom ≥ 0; and headroom ≤ screen_gap because the
    immediate sacrifice is ≥ 0, so the future gain must exceed the whole headroom."""
    for r in kuhn_decider.results:
        for s in r.per_state:
            assert s.headroom >= -1e-9
            assert s.headroom <= s.screen_gap + 1e-9
            assert s.lb_pure <= s.lb_mixed + 1e-9 <= s.lb_coord + 2e-9


def test_escalation_exact_controls(kuhn_decider):
    """If headroom appeared, the escalation ran the rich-menu A2.5/A4/A5/ceiling; its
    exact controls must hold (A5 ≥ A2.5, ceiling ≥ A2.5, and the misspecified star
    model A4 never beats the correct dense model A5)."""
    if kuhn_decider.verdict != "MENU-ARTIFACT":
        pytest.skip("no headroom -> no escalation")
    e = kuhn_decider.escalation
    assert e is not None and e.controls_ok
    for r in e.per_family:
        assert r["A5"] >= r["A2.5"] - 1e-9
        assert r["ceiling"] >= r["A2.5"] - 1e-9
        assert r["a5_minus_a4"] >= -1e-9          # correct model never below the star model


def test_experiment_frontend_ids_consistent():
    """Every DOM id the experiment client reads must exist in experiment.html (the
    most common way a from-scratch dashboard silently breaks). Pure-Python so it runs
    in the audit; the node harness (scripts/check_experiment_dom.mjs) covers rendering."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "parciver" / "dashboard" / "static"
    html = (root / "experiment.html").read_text(encoding="utf-8")
    js = (root / "experiment.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    referenced = set(re.findall(r'\$\("([^"]+)"\)', js)) | set(re.findall(r'getElementById\("([^"]+)"\)', js))
    missing = referenced - html_ids
    assert not missing, f"experiment.js reads ids absent from experiment.html: {sorted(missing)}"
    # the client must load a stylesheet and target the same SSE endpoint the server exposes
    assert "/static/style.css" in html and "/events" in js


def test_decider_is_deterministic():
    cfg = BaitConfig(n_pure=25, n_mixed=25, coord=True, coord_budget=150, seed=99)
    a = run_decider(game_name="3p-kuhn", m=2, delta=0.02, cfr_iters=1200, n_states=2,
                    cfg=cfg, do_escalate=False)
    b = run_decider(game_name="3p-kuhn", m=2, delta=0.02, cfr_iters=1200, n_states=2,
                    cfg=BaitConfig(n_pure=25, n_mixed=25, coord=True, coord_budget=150, seed=99),
                    do_escalate=False)
    ha = [s.headroom for r in a.results for s in r.per_state]
    hb = [s.headroom for r in b.results for s in r.per_state]
    assert ha == pytest.approx(hb, abs=1e-12)
