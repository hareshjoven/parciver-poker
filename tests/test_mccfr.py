# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Cross-validation of the Rust external-sampling MCCFR solver.

MCCFR is the sampled hot loop that will scale to the abstracted hold'em tree,
where the full-traversal CFR of `test_rust_engine.py` is intractable. Sampling
means we cannot demand bit-parity with the full solver (the two no longer compute
the same iterate), so the guarantees pinned here are:

  * structural parity  -- the sampled solver reaches the SAME infoset partition
    (keys and action counts) as the full-traversal engine and the Python oracle;
  * equilibrium quality -- a Rust-MCCFR-solved strategy, scored by the INDEPENDENT
    Python best-response oracle, drives exploitability down to the same standard
    the full solver meets (Kuhn value -1/18; Leduc exploitability < 4e-2), and
    exploitability demonstrably DECREASES with more iterations (it converges, it
    did not merely land well once);
  * determinism -- a fixed seed reproduces the average strategy bit-for-bit, the
    property that makes a sampled run debuggable and its gates non-flaky.

Thresholds carry wide margin over measured behaviour (Kuhn 200k expl <= 3e-3 all
variants; Leduc 500k linear expl ~1.9e-2, ~2.4e-2 worst across seeds).

Skips cleanly if the extension is not built (`maturin develop -m engine/Cargo.toml`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

engine = pytest.importorskip(
    "parciver_engine",
    reason="Rust extension not built; run: maturin develop -m engine/Cargo.toml",
)

from parciver.eval.exploitability import exploitability, expected_value
from parciver.games.kuhn import GAME_VALUE, KuhnPoker
from parciver.games.leduc import LeducHoldem

VARIANTS = ("mccfr", "mccfr+", "linear")


# --- structural parity ----------------------------------------------------

def test_mccfr_infoset_partition_matches_full_engine():
    """The sampled solver must reach the exact infoset partition of the full one.

    Coverage is not automatic for a sampler, so this doubles as proof that every
    infoset is visited with positive probability: at 100k iters all 288 Leduc /
    12 Kuhn keys appear, matching the full-traversal solver key-for-key with equal
    action-vector lengths.
    """
    for solve, full in (
        (engine.solve_leduc_mccfr, engine.solve_leduc),
        (engine.solve_kuhn_mccfr, engine.solve_kuhn),
    ):
        mc = solve(100_000, "linear", 0)
        ref = full(200, "dcfr")
        assert set(mc.keys()) == set(ref.keys())
        for key, probs in ref.items():
            assert len(mc[key]) == len(probs), f"action count mismatch at {key!r}"


# --- equilibrium quality (Rust-solved, Python-scored) ---------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_kuhn_mccfr_reaches_game_value(variant):
    strat = engine.solve_kuhn_mccfr(200_000, variant, 0)
    kuhn = KuhnPoker()
    assert expected_value(kuhn, strat) == pytest.approx(GAME_VALUE, abs=5e-3)
    assert exploitability(kuhn, strat) < 8e-3


def test_leduc_mccfr_converges_to_equilibrium():
    """Leduc MCCFR clears the same 4e-2 gate the full DCFR solver clears, and
    exploitability falls sharply from a small budget to a large one."""
    leduc = LeducHoldem()
    coarse = exploitability(leduc, engine.solve_leduc_mccfr(20_000, "linear", 0))
    fine_strat = engine.solve_leduc_mccfr(500_000, "linear", 0)
    fine = exploitability(leduc, fine_strat)
    assert fine < 4e-2, f"Leduc MCCFR exploitability {fine:.4e} exceeds gate"
    assert fine < 0.5 * coarse, f"did not converge: {fine:.4e} vs coarse {coarse:.4e}"
    assert -0.15 < expected_value(leduc, fine_strat) < 0.0


# --- determinism ----------------------------------------------------------

def test_mccfr_is_deterministic_for_a_fixed_seed():
    a = engine.solve_leduc_mccfr(50_000, "linear", 123)
    b = engine.solve_leduc_mccfr(50_000, "linear", 123)
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], f"strategy for {key!r} differs run-to-run"


def test_mccfr_seed_changes_the_sample_path():
    """Different seeds give different sampled iterates (same equilibrium, not the
    same bits) -- confirms the seed actually drives the sampler."""
    a = engine.solve_leduc_mccfr(50_000, "linear", 1)
    b = engine.solve_leduc_mccfr(50_000, "linear", 2)
    assert any(a[key] != b[key] for key in a)


def test_unknown_mccfr_variant_raises():
    with pytest.raises(ValueError):
        engine.solve_kuhn_mccfr(10, "not-a-variant", 0)
