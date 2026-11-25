from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from parciver.rmapss import step32
from parciver import telemetry as T

spec = importlib.util.spec_from_file_location(
    "step32_runner", Path(__file__).resolve().parents[1] / "scripts" / "rmapss_step32.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _measurement(gain: float) -> dict:
    return {
        "hands": step32.EVAL_HANDS,
        "block_size": step32.BLOCK_SIZE,
        "candidate_bb_per_100": gain,
        "anchor_bb_per_100": 0.0,
        "gain_bb_per_100": gain,
        "difference_stddev_bb_per_hand": 0.0,
        "block_gains_bb_per_100": [gain] * step32.BLOCKS,
    }


def test_proxy_search_is_descending_and_attacks_each_candidate(monkeypatch):
    class Anchor:
        def blend(self, _response, alpha):
            return SimpleNamespace(alpha=alpha)

    calls = []

    def fake_attack(candidate, _anchor, **kwargs):
        calls.append((candidate.alpha, kwargs["replicate"]))
        # Incremental loss is -gain: 2*alpha. Budget=1 first passes at alpha=.4.
        return {"replicate": kwargs["replicate"],
                "measurement": _measurement(-2.0 * candidate.alpha)}

    monkeypatch.setattr(runner, "_attack_measurement", fake_attack)
    alpha, rows = runner._search_alpha_proxy(
        Anchor(), object(), seed=103, case_index=0, proxy_budget=1.0, threads=1
    )
    assert alpha == 0.40
    assert [row["alpha"] for row in rows] == [1.0, 0.70, 0.40]
    assert len(calls) == 3 * step32.ATTACK_REPLICATES
    assert {rep for a, rep in calls if a == alpha} == {0, 1, 2}


def test_locked_cli_has_no_research_setting_overrides():
    with pytest.raises(SystemExit) as exc:
        runner.build_parser().parse_args(["--eval-hands", "1"])
    assert exc.value.code == 2


def test_runner_refuses_missing_full_audit_before_creating_a_run(tmp_path):
    run_dir = tmp_path / "must-not-exist"
    args = SimpleNamespace(
        protocol=step32.PROTOCOL,
        audit_receipt=tmp_path / "missing.json",
        run_dir=run_dir,
    )
    with pytest.raises(ValueError, match="missing S3.2 engineering-audit receipt"):
        runner.run(args)
    assert not run_dir.exists()


def test_resume_state_requires_exact_locked_metadata_and_no_duplicates(tmp_path):
    hashes = {str(path): digest for path, digest in step32.FROZEN_ARTIFACTS.items()}
    state = {
        "protocol": step32.PROTOCOL,
        "metadata": step32.frozen_metadata(),
        "artifact_hashes": hashes,
        "calibration_rows": [{"seed": 103, "label": "nit(.60)"}],
        "case_rows": [],
    }
    path = tmp_path / "s3.2-results.json"
    runner._write_json_atomic(path, state)
    assert runner._load_resume_state(path, hashes)["protocol"] == step32.PROTOCOL

    state["metadata"]["response_iters"] = 1
    runner._write_json_atomic(path, state)
    with pytest.raises(ValueError, match="locked protocol"):
        runner._load_resume_state(path, hashes)


def test_peak_memory_is_reportable():
    assert runner._peak_memory_gb() >= 0.0


def test_s32_dashboard_renders_locked_telemetry(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    dashboard_row = {
        "seed": 103, "label": "station(.20)", "split": "held", "klass": "suboptimal",
        "alpha": 0.4, "alpha_conf": 0.6, "alpha_proxy": 0.4,
        "gain": 1.5, "lower": 1.1, "upper": 1.9, "delta_sub": 1.0, "tol": 0.5,
        "proxy_budget": 1.0, "proxy_loss_upper": 0.7, "coverage": 0.98,
        "model_log_loss": 0.2, "reference_log_loss": 0.3,
        "compute_s": 12.0, "peak_memory_gb": 3.5,
    }
    report = {
        "protocol": step32.PROTOCOL,
        "all_pass": True,
        "verdict": "PASS (S3.2 synthetic-field product + ABR-proxy non-regression)",
        "gates": {"artifact/completeness": True, "small-game control": True},
        "safety_language": "lower-bound sampled attacks; not a safety certificate",
    }
    with T.RunWriter(tmp_path) as writer:
        writer.start(variant="s3.2", game="nlhe_pluribus_100bb",
                     config={"kind": "experiment", "protocol": step32.PROTOCOL})
        writer.checkpoint(iteration=1, step32_row=dashboard_row,
                          step32_progress={"completed": 1, "total": 45})
        writer.checkpoint(iteration=2, step32=report)
        writer.end(iteration=2, summary=report)
    check = subprocess.run(
        [node, str(Path(__file__).resolve().parents[1] / "scripts" / "check_experiment32_dom.mjs"),
         str(tmp_path / "events.jsonl")],
        capture_output=True, text=True, timeout=30,
    )
    assert check.returncode == 0, check.stdout + check.stderr
