"""Tests for the read-only blueprint analysis (`parciver_engine.BlueprintAnalyzer`)
that feeds the dashboard's action-frequency, bet-sizing, and head-to-head panels.

The analysis Monte-Carlo self-plays a saved blueprint through the real 100bb game
(so action legality / bet sizing come from the certified engine) and plays it
head-to-head against a calling-station baseline. This gates the Python surface:
the per-street mixes are valid distributions, and head-to-head is a finite bb/100
that is positive (a real blueprint beats a station that never folds).

Skipped unless the *current* extension exposes ``BlueprintAnalyzer`` (a fresh build
that a running, file-locking trainer can prevent installing until it stops — the
Rust `analysis` unit test always covers the underlying logic) and the baked bucket
tables are present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _abstraction_dir():
    d = ROOT / "data" / "abstraction"
    if all((d / f"{s}_buckets.npy").exists() for s in ("flop", "turn", "river")):
        return d
    return None


def _require(engine):
    if not hasattr(engine, "BlueprintAnalyzer"):
        pytest.skip("installed parciver_engine has no BlueprintAnalyzer (build pending)")


def test_blueprint_analyzer_panels_and_head_to_head(tmp_path):
    engine = pytest.importorskip("parciver_engine")
    _require(engine)
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")

    # Solve a short blueprint and save it (the PLAY artifact the analyzer reads).
    bp = engine.Blueprint(str(d), "linear", 0, 4)
    bp.run(60000)
    model = tmp_path / "blueprint.bin"
    bp.save(str(model))

    az = engine.BlueprintAnalyzer(str(d))
    res = az.analyze(str(model), 8000, 8000, 0)

    assert set(res.keys()) == {"action_freq", "bet_size_freq", "hh_bb_per_100"}

    # Preflop is always reached; every reported street's action mix is a proper
    # distribution over fold/check/call/raise/all-in.
    af = res["action_freq"]
    assert "P" in af
    for st, mix in af.items():
        assert set(mix.keys()) == {"f", "x", "c", "r", "A"}
        assert abs(sum(mix.values()) - 1.0) < 1e-6
        assert all(v >= 0.0 for v in mix.values())

    # Bet-size mixes (over aggressive actions) are distributions where present.
    for st, mix in res["bet_size_freq"].items():
        assert set(mix.keys()) <= {"0.5pot", "1.0pot", "allin"}
        assert abs(sum(mix.values()) - 1.0) < 1e-6

    # A real blueprint beats a calling station (which never folds) by a clear margin.
    hh = res["hh_bb_per_100"]
    assert isinstance(hh, float)
    assert hh == hh and abs(hh) != float("inf")  # finite
    assert hh > 0.0, f"blueprint should beat a calling station, got {hh:+.2f} bb/100"
