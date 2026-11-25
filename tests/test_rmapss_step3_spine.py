"""R-MAPSS Step 3 (S3.1) — the HU exploitation SPINE on `nlhe_small`.

Sub-step-1 primitives are certified in `test_rmapss_step3.py`; this covers the
spine (`parciver.rmapss.step3`): the opponent population, the confidence- and
safety-gated deviation, and the exploratory regression gates (RMAPSS §41),
including the §28 anti-hallucination controls and the §41.DEV1 observability
boundary. One (coarse-grid, modest-anchor) run is shared across the assertions.
"""
from copy import deepcopy

import pytest

pytest.importorskip("parciver_engine")
import parciver_engine as pe  # noqa: E402

from parciver.rmapss import step3  # noqa: E402

# A modest, seeded anchor + coarse α grid keep the exact run fast while still
# clearing every gate with margin (the headline 2M-anchor run lives in the runner).
_COARSE = (1.0, 0.7, 0.45, 0.25, 0.1, 0.0)


@pytest.fixture(scope="module")
def anchor():
    return pe.solve_nlhe_small_mccfr(120_000, "linear", 0)


@pytest.fixture(scope="module")
def report(anchor):
    return step3.run_step3(anchor=anchor, alpha_grid=_COARSE)


# ---- construction (cheap) --------------------------------------------------

def test_infoset_split_is_two_seats():
    struct = step3.infoset_structure()
    p0, p1 = step3.split_keys(struct)
    assert len(p0) == 228 and len(p1) == 228
    assert not (set(p0) & set(p1)), "seats must partition the keys"


def test_population_classes_and_valid_distributions(anchor):
    struct = step3.infoset_structure()
    _p0, p1 = step3.split_keys(struct)
    bb = pe.nlhe_small_observed_marginal(anchor, anchor, 0)
    pop = step3.build_population(struct, p1, anchor, bb)
    klasses = {o.klass for o in pop}
    assert klasses == {"equilibrium", "suboptimal", "hidden"}
    assert any(o.klass == "hidden" for o in pop), "the board-blind boundary case must be present"
    for o in pop:
        for k, row in o.strat.items():
            assert abs(sum(row) - 1.0) < 1e-9, (o.label, k)
            assert all(x >= -1e-12 for x in row)


def test_archetypes_behave(anchor):
    """nit folds facing a bet, station never folds, maniac always jams — the
    behavioural distortions that make the opponents observably exploitable."""
    struct = step3.infoset_structure()
    _p0, p1 = step3.split_keys(struct)
    bb = pe.nlhe_small_observed_marginal(anchor, anchor, 0)
    nit = step3.build_opponent(struct, p1, anchor, "nit", board_blind_flat=bb)
    station = step3.build_opponent(struct, p1, anchor, "station", board_blind_flat=bb)
    maniac = step3.build_opponent(struct, p1, anchor, "maniac", board_blind_flat=bb)
    for k in p1:
        cats = struct[k][1]
        if "fold" in cats:
            fi = cats.index("fold")
            assert nit[k][fi] == 1.0, f"nit must fold facing a bet at {k}"
            assert station[k][fi] == 0.0, f"station must never fold at {k}"
            assert maniac[k][fi] == 0.0, f"maniac must never fold at {k}"
        if "allin" in cats:
            ai = cats.index("allin")
            assert maniac[k][ai] == 1.0, f"maniac must jam at {k}"


# ---- §28 controls (load-bearing) -------------------------------------------

def test_no_hallucination_marginal_br_under_ceiling(report):
    """Every observed-model best response is bounded by the EXACT best-response
    ceiling — the spine never claims an exploit the exact control forbids (§28)."""
    for r in report.results:
        assert r.ev_a2 <= r.ev_ceiling + 1e-9, r.label
    assert report.gates["observed-model BR <= exact BR ceiling (no hallucination, §28)"]


def test_safety_exact_exploitability_within_budget(report):
    """Hero's deviation keeps the EXACT worst-case exploitability within the
    S-scaled budget on EVERY opponent (the load-bearing safety gate)."""
    for r in report.results:
        assert r.expl_final <= report.base_expl + report.budget + 1e-9, r.label
    assert report.gates["safety: exact exploitability within budget (all opponents)"]


# ---- exploratory outcome regressions --------------------------------------

def test_beats_suboptimal_heldout(report):
    held = [r for r in report.results if r.klass == "suboptimal" and r.split == "held"]
    assert held, "need held-out suboptimal opponents"
    for r in held:
        assert r.exploit_gain > report.delta_sub, (r.label, r.exploit_gain, report.delta_sub)
    assert report.detail["mean exploit gain (suboptimal HELD-OUT)"] > 0.0


def test_matches_near_eq(report):
    for r in report.results:
        if r.klass == "equilibrium":
            assert abs(r.exploit_gain) <= report.tol_eq, r.label


def test_declines_hidden_boundary(report):
    """Historical hole-blind opponent has a small, nonzero model signal.
    This is an exploratory regression, not an indistinguishability proof."""
    hidden = [r for r in report.results if r.klass == "hidden"]
    assert hidden, "the board_blind boundary case must be present"
    # its Hero-observable signal is far below any OBSERVABLY-distorted opponent's
    # (those are ~0.6–1.0); the tiny residual is reach re-weighting, not a leak.
    max_sub_signal = max(r.signal for r in report.results if r.klass == "suboptimal")
    for r in hidden:
        assert r.signal < 0.1, f"hidden case must look ~ eq (signal {r.signal})"
        assert r.signal < max_sub_signal / 5.0
        assert abs(r.exploit_gain) <= report.tol_eq
        # it forgoes real EV precisely because Hero cannot safely see the distortion
        assert r.ceiling_gain > report.delta_sub
    assert report.gates["matches hole-blind model boundary (|gain| <= tol)"]


def test_all_gates_pass_and_verdict(report):
    assert report.all_pass, [k for k, v in report.gates.items() if not v]
    assert report.verdict.startswith("PASS (exploratory")
    # thresholds carry step1's shape: tolerance strictly below the beat margin
    assert report.tol_eq < report.delta_sub
    assert report.scale_S > 0.0


def test_deterministic(anchor, report):
    """Seeded MCCFR anchor => byte-identical gate outcomes (exact, no sampling in
    the spine itself)."""
    r1 = report
    r2 = step3.run_step3(anchor=anchor, alpha_grid=_COARSE)
    assert r1.gates == r2.gates
    assert [round(x.ev_final, 9) for x in r1.results] == [round(x.ev_final, 9) for x in r2.results]


def test_empty_report_fails():
    rep = step3.Step3Report()
    step3._finalize(rep)
    assert not rep.all_pass
    assert rep.verdict.startswith("FAIL")


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nan", "unsafe"])
def test_incomplete_or_invalid_results_cannot_pass(report, damage):
    rep = deepcopy(report)
    if damage == "missing":
        rep.results.pop()
    elif damage == "duplicate":
        rep.results[-1] = deepcopy(rep.results[0])
    elif damage == "nan":
        rep.results[0].ev_final = float("nan")
    else:
        # Cached safety_ok must not be trusted over the measured number.
        rep.results[0].expl_final = rep.base_expl + rep.budget + 1
        rep.results[0].safety_ok = True
    step3._finalize(rep)
    assert not rep.all_pass


def test_heldout_results_do_not_calibrate_budget(report):
    rows = [{"opp": step3.Opp(r.label, {}, r.klass, r.split),
             "ev_ceiling": r.ev_ceiling, "ev_a0": r.ev_a0} for r in report.results]
    expected = step3._fit_scale(rows)
    for row in rows:
        if row["opp"].split == "held":
            row["ev_ceiling"] = 1e9
    assert step3._fit_scale(rows) == expected == report.scale_S


def test_cached_safety_scan_matches_full_oracle(anchor, report):
    struct = step3.infoset_structure()
    p0, _p1 = step3.split_keys(struct)
    base = {k: anchor.get(k, [1 / len(struct[k][1])] * len(struct[k][1])) for k in p0}
    target = pe.nlhe_small_best_response_strategy({}, 0)
    budget = report.budget
    # Independently run the original full two-seat oracle at every candidate.
    original = []
    for a in _COARSE:
        pi = step3.blend(base, target, a, p0)
        expl = pe.nlhe_small_exploitability({**anchor, **pi})
        original.append((a, expl))
    expected = next(pair for pair in original if pair[1] <= report.base_expl + budget + 1e-12)
    got = step3.safety_gated_alpha(anchor, base, target, p0,
                                 base_expl=report.base_expl, budget=budget, grid=_COARSE)
    assert got[0] == expected[0]
    assert got[1] == pytest.approx(expected[1], abs=1e-12)
