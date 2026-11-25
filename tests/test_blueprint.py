"""Tests for the B4 blueprint solver (`parciver_engine.Blueprint`).

Exact exploitability is intractable on the full 100bb game (that is why B3 built the
small-game bridge), so this gates what CAN be checked cheaply from a short real
solve: the solver runs and accumulates infosets, the variant-normalized average
regret is a positive number that TRENDS DOWN with more iterations (the convergence
proxy the dashboard plots), the preflop range grid is a valid per-hand distribution
that is SANE poker (trash folds more than premiums), and the blueprint saves to a
non-empty file. Requires the baked bucket tables (gitignored); skipped without them.
"""
from __future__ import annotations

import os
import shutil
from math import isfinite
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _abstraction_dir():
    d = ROOT / "data" / "abstraction"
    if all((d / f"{s}_buckets.npy").exists() for s in ("flop", "turn", "river")):
        return d
    return None


def test_blueprint_runs_and_range_is_sane(tmp_path):
    engine = pytest.importorskip("parciver_engine")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent; run scripts/build_buckets.py")
    from parciver import telemetry as T

    names = T.preflop_hand_names()
    bp = engine.Blueprint(str(d), "linear", 0)

    bp.run(12000)
    assert bp.iterations() == 12000
    assert bp.num_infosets() > 10000  # the full game discovers many infosets fast

    grid = bp.preflop_range_grid()
    assert len(grid) == 169
    for row in grid:
        assert len(row) == 3
        s = sum(row)
        assert abs(s - 1.0) < 1e-6 or s == 0.0  # a valid mix, or an unvisited cell

    # Sane poker: trash folds strictly more than premiums.
    fold = lambda h: grid[names.index(h)][0]  # noqa: E731
    assert fold("72o") > fold("AA")
    assert fold("32o") > fold("KK")

    # The convergence proxy is positive and trends down with more iterations.
    r1 = bp.avg_regret()
    bp.run(12000)
    r2 = bp.avg_regret()
    assert r1 > 0 and r2 > 0
    assert r2 < r1

    # Coverage-by-street is consistent with the total.
    p, f, t, r = bp.infosets_by_street()
    assert p > 0 and f > 0
    assert p + f + t + r <= bp.num_infosets()

    # The blueprint persists to a non-empty file.
    out = tmp_path / "blueprint.bin"
    bp.save(str(out))
    assert out.exists() and out.stat().st_size > 1000

    # The durable average strategy reloads through the strict, read-only player
    # and can be evaluated without recovering mutable solver state.
    saved = engine.SavedBlueprint(str(d), str(out))
    assert saved.num_infosets() == bp.num_infosets()
    assert saved.source_path() == str(out)
    match = saved.evaluate_against(saved, 2000, 17)
    assert match["hands"] == 2000
    assert isfinite(match["bb_per_100"])
    assert isfinite(match["stddev_bb_per_hand"]) and match["stddev_bb_per_hand"] >= 0.0

    # Fixed-opponent response is additive and returns responder rows only; blending
    # restores a complete policy with exact anchor fallback for unvisited rows.
    response = saved.fixed_response(0, 1000, "linear", 23, 1)
    assert 0 < response.num_infosets() <= saved.num_infosets()
    candidate = saved.blend_response(response, 0.2)
    assert candidate.num_infosets() == saved.num_infosets()
    with pytest.raises(ValueError, match="alpha"):
        saved.blend_response(response, 1.1)

    # Protocol policy views keep leak/model composition in Rust, so no 13M-row
    # strategy is copied through Python and sampled observations expose public keys.
    policy = saved.s32_policy()
    station = policy.leak("station", 0.6)
    train = policy.observe(station, 5000, 31)
    validation = policy.observe(station, 2000, 37)
    reference = policy.observe(policy, 5000, 41)
    assert train.decisions() > 5000
    assert train.public_keys_valid() and validation.public_keys_valid()
    assert all("|B[" in key and "|P" in key for key in train.exact_key_sample())
    model = train.fit_model(reference, 5, 20.0)
    score = validation.score_model(model)
    assert score["decisions"] == validation.decisions()
    assert 0.0 <= score["coverage"] <= 1.0
    assert isfinite(score["model_log_loss"])
    assert isfinite(score["reference_log_loss"])
    model_response = model.fixed_response(0, 1000, "linear", 43, 1)
    assert model_response.label().startswith("response(")


def test_blueprint_rejects_bad_variant(tmp_path):
    engine = pytest.importorskip("parciver_engine")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")
    with pytest.raises(Exception):
        engine.Blueprint(str(d), "not-a-variant", 0)


def test_blueprint_parallel_runs_and_is_sane():
    """The parallel path (threads > 1) runs the real solve and still produces sane
    poker. Not a bit-identity check — concurrent float additions reorder — but the
    range must be a valid distribution with trash folding more than premiums, the
    same equilibrium shape the serial path reaches."""
    engine = pytest.importorskip("parciver_engine")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")
    from parciver import telemetry as T

    names = T.preflop_hand_names()
    bp = engine.Blueprint(str(d), "linear", 0, 4)  # 4 worker threads
    assert bp.threads() == 4
    bp.run(40000)
    assert bp.iterations() == 40000
    grid = bp.preflop_range_grid()
    for row in grid:
        s = sum(row)
        assert abs(s - 1.0) < 1e-6 or s == 0.0
    fold = lambda h: grid[names.index(h)][0]  # noqa: E731
    assert fold("72o") > fold("AA")
    assert fold("32o") > fold("KK")


def test_blueprint_checkpoint_resume_roundtrip(tmp_path):
    """A full-state checkpoint saved and reloaded resumes EXACTLY: same iteration
    count, same infoset coverage, the same average-strategy range grid (accumulators
    are f64, stored losslessly), and the reloaded solver keeps solving. This is what
    makes a multi-day solve restart-safe."""
    engine = pytest.importorskip("parciver_engine")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")

    bp = engine.Blueprint(str(d), "linear", 7, 4)
    bp.run(15000)
    iters0, ninf0 = bp.iterations(), bp.num_infosets()
    grid0 = bp.preflop_range_grid()

    ckpt = tmp_path / "checkpoint.bin"
    bp.save_checkpoint(str(ckpt))
    assert ckpt.exists() and ckpt.stat().st_size > 1000

    bp2 = engine.Blueprint.resume(str(d), str(ckpt), 4)
    assert bp2.iterations() == iters0
    assert bp2.num_infosets() == ninf0
    grid1 = bp2.preflop_range_grid()
    for r0, r1 in zip(grid0, grid1):
        for a, b in zip(r0, r1):
            assert abs(a - b) < 1e-9  # lossless f64 round-trip

    # The resumed solver continues from exactly where the checkpoint stopped.
    bp2.run(10000)
    assert bp2.iterations() == iters0 + 10000

    # Replacement publishes the new generation and keeps the old one resumable.
    bp2.save_checkpoint(str(ckpt))
    latest = engine.Blueprint.resume(str(d), str(ckpt), 1)
    previous = engine.Blueprint.resume(str(d), str(ckpt) + ".previous", 1)
    assert latest.iterations() == iters0 + 10000
    assert previous.iterations() == iters0
    assert previous.preflop_range_grid() == grid0


def test_a_resumed_blueprint_runs_and_reports_its_checkpoints_variant(tmp_path):
    """The variant is solve state: resume takes it from the checkpoint (so the
    training script can record it rather than its --variant flag)."""
    engine = pytest.importorskip("parciver_engine")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")
    bp = engine.Blueprint(str(d), "mccfr+", 5, 2)
    assert bp.variant == "mccfr+" and engine.Blueprint(str(d), "mccfr", 5, 1).variant == "mccfr"
    bp.run(1000)
    ckpt = tmp_path / "checkpoint.bin"
    bp.save_checkpoint(str(ckpt))
    assert engine.Blueprint.resume(str(d), str(ckpt), 1).variant == "mccfr+"


def test_blueprint_grid_is_stored_in_its_files_and_restored(tmp_path):
    """A 200bb-grid solve writes its grid into blueprint.bin and checkpoint.bin;
    resume and SavedBlueprint rebuild that grid from the file (never a guess), and
    the loaded agent keeps a 10bb 3-bet pot on the tree."""
    engine = pytest.importorskip("parciver_engine")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")
    with pytest.raises(ValueError, match="unknown bet grid"):
        engine.Blueprint(str(d), "linear", 3, 2, grid="300bb")
    bp = engine.Blueprint(str(d), "linear", 3, 2, grid="200bb")
    assert bp.grid == "200bb"
    bp.run(3000)
    model, ckpt = tmp_path / "blueprint.bin", tmp_path / "checkpoint.bin"
    bp.save(str(model))
    bp.save_checkpoint(str(ckpt))
    assert engine.Blueprint.resume(str(d), str(ckpt), 1).grid == "200bb"
    saved = engine.SavedBlueprint(str(d), str(model))
    assert saved.grid == "200bb"
    agent = saved.play_agent()
    d3 = agent.decide(0, (48, 49), [], [("b", 250), ("b", 900)], 50, 100, (20_000, 20_000), seed=1)
    assert d3["source"] in ("blueprint", "missing_key"), d3["off_tree_reason"]
    assert "call" in d3["labels"]
    # The default grid is unchanged.
    assert engine.Blueprint(str(d), "linear", 3, 1).grid == "100bb"


def test_resume_refuses_a_checkpoint_of_other_bucket_tables_or_a_corrupt_one(tmp_path):
    """A checkpoint records the abstraction's fingerprint and a checksum: resuming
    it over other bucket tables, or after a byte of it changed, is refused."""
    engine = pytest.importorskip("parciver_engine")
    np = pytest.importorskip("numpy")
    d = _abstraction_dir()
    if d is None:
        pytest.skip("baked abstraction tables absent")
    bp = engine.Blueprint(str(d), "linear", 4, 2)
    bp.run(1000)
    ckpt = tmp_path / "checkpoint.bin"
    bp.save_checkpoint(str(ckpt))
    assert engine.Blueprint.resume(str(d), str(ckpt), 1).iterations() == 1000

    # The same tables but one flop bucket (the engine only reads these files).
    other = tmp_path / "abstraction"
    other.mkdir()
    flop = np.load(d / "flop_buckets.npy")
    flop[0] ^= 1
    np.save(other / "flop_buckets.npy", flop)
    for s in ("turn", "river"):
        try:
            os.link(d / f"{s}_buckets.npy", other / f"{s}_buckets.npy")
        except OSError:
            shutil.copyfile(d / f"{s}_buckets.npy", other / f"{s}_buckets.npy")
    with pytest.raises(ValueError, match="different abstraction"):
        engine.Blueprint.resume(str(other), str(ckpt), 1)

    data = bytearray(ckpt.read_bytes())
    data[-1] ^= 1
    ckpt.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        engine.Blueprint.resume(str(d), str(ckpt), 1)
