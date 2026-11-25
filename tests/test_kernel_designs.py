# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The kernel candidates' algebra on the CPU (src/parciver/kernel_designs.py;
KERNELS.md K1, K2, K3b, K4), before any of them is written in Triton: the
aliased reach rows hold the same bits as today's down pass; the fold and
showdown forms match the dense card-removal matrices in float64 to rule 5's
tolerance; a showdown row's reach is already 0 on the hands dead in its view."""
import numpy as np
import pytest

pytest.importorskip("parciver_engine")
pytest.importorskip("triton")  # gpu_fused (host_plan) imports it

from parciver import gpu_fused as gf  # noqa: E402
from parciver import gpu_search as gs  # noqa: E402
from parciver import kernel_designs as kd  # noqa: E402


@pytest.fixture(scope="module", params=["river", "turn"])
def game(request):
    board = gs.board(7)
    cards, sp = (board, gs.spec()) if request.param == "river" else (board[:4], gs.small_turn_spec())
    hands = gs.Hands(cards)
    tree = gs.Tree(sp, hands)
    plan = gf.host_plan(hands, tree, [gs.range_(hands.n, 31), gs.range_(hands.n, 32)])
    return hands, plan


def _strategies(plan, dtype=np.float32):
    return [kd.uniform(plan, dtype), kd.skewed(plan, 1, dtype=dtype), kd.skewed(plan, 2, power=30.0, dtype=dtype)]


def _close(got, want):
    """Rule 5: 1e-12 x max(1, |want|max)."""
    return np.abs(got - want).max() <= 1e-12 * max(1.0, np.abs(want).max())


def test_k1_aliased_rows_hold_todays_reach_bit_for_bit(game):
    _, plan = game
    for p in (0, 1):
        rrow, levels = kd.reach_rows(plan, p)
        dpl = np.asarray(plan.dpl)
        # The rows that hold a reach: the root, and the children of opponent and chance nodes.
        holders = {0}
        for ld, lc in levels:
            assert (dpl[ld] != p).all()
            for d in ld:
                holders.update(plan.dkid[d, :plan.dk[d]].tolist())
            for c in lc:
                holders.update(np.asarray(plan.ckid)[c].tolist())
        assert set(rrow.tolist()) <= holders and all(rrow[h] == h for h in holders)
        for strat in _strategies(plan):
            rq, _, written = kd.down(plan, p, strat)
            ra, rrow2, fewer = kd.down(plan, p, strat, alias=True)
            assert np.array_equal(rrow, rrow2)
            assert np.array_equal(ra[rrow], rq)  # every reader finds the same bits through the table
            assert fewer < written
            untouched = np.setdiff1d(np.arange(plan.N), list(holders))
            assert not ra[untouched].any()  # aliased rows are never written


def test_k2_card_sums_match_the_dense_matrix_in_float64(game):
    hands, plan = game
    for p in (0, 1):
        for strat in _strategies(plan, np.float64):
            r = kd.down(plan, p, strat, np.float64)[0][plan.f_lo:plan.f_lo + plan.F]
            want = kd.dense_fold(hands, plan, p, r)
            assert np.abs(want).max() > 0
            assert _close(kd.fold_today(plan, p, r, np.float64), want)
            assert _close(kd.fold_cards(plan, p, r, np.float64), want)


def test_k3b_per_card_prefix_sums_match_the_dense_matrix_in_float64(game):
    hands, plan = game
    for p in (0, 1):
        for strat in _strategies(plan, np.float64):
            r = kd.down(plan, p, strat, np.float64)[0][plan.s_lo:]
            want = kd.dense_showdown(hands, plan, r)
            assert np.abs(want).max() > 0
            assert _close(kd.showdown(plan, r, np.float64), want)
            assert _close(kd.showdown(plan, r, np.float64, cards=True), want)


def test_k4_showdown_reach_is_zero_on_dead_hands(game):
    _, plan = game
    view = np.asarray(plan.sd_view)[:plan.S]
    dead = np.asarray(plan.alive)[view] == 0  # [S, n]
    for p in (0, 1):
        for strat in _strategies(plan):
            r = kd.down(plan, p, strat)[0][plan.s_lo:]
            assert not r[dead].any()
            # so masking the input again changes no bit, in either form
            for cards in (False, True):
                assert np.array_equal(kd.showdown(plan, r, cards=cards), kd.showdown(plan, r, cards=cards, mask=False))


def test_float32_previews_are_no_worse_than_todays_forms(game):
    # NumPy's float32 sums stand in for the GPU's: a preview of rule 2, which is judged on the GPU.
    hands, plan = game
    today, k2, sd, k3b, wide = [], [], [], [], []
    for p in (0, 1):
        for strat in _strategies(plan):
            rq = kd.down(plan, p, strat)[0]
            rf, rs = rq[plan.f_lo:plan.f_lo + plan.F], rq[plan.s_lo:]
            sf, ss = kd.scales(plan, p, rf, rs)
            wf, ws = kd.dense_fold(hands, plan, p, rf), kd.dense_showdown(hands, plan, rs)
            today.append(kd.errors(kd.fold_today(plan, p, rf), wf, sf))
            k2.append(kd.errors(kd.fold_cards(plan, p, rf), wf, sf))
            sd.append(kd.errors(kd.showdown(plan, rs), ws, ss))
            k3b.append(kd.errors(kd.showdown(plan, rs, cards=True), ws, ss))
            for got, want, scale in ((kd.fold_cards(plan, p, rf, sums=np.float64), wf, sf),
                                     (kd.showdown(plan, rs, cards=True, sums=np.float64), ws, ss)):
                assert got.dtype == np.float32
                wide.append(kd.errors(got, want, scale))
    today, k2, sd, k3b, wide = (np.concatenate(x) for x in (today, k2, sd, k3b, wide))
    assert max(today.max(), k2.max(), sd.max(), k3b.max()) < 1e-5  # float32 rounding, not an algebra error
    assert k3b.mean() <= sd.mean()
    # Q3: float64 sums leave the one rounding of each float32 result (half an ulp of a value no larger than the scale).
    assert wide.max() <= 2.0 ** -24 * 1.01 and wide.mean() < min(today.mean(), sd.mean())
