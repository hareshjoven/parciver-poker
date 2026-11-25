from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from parciver.rmapss import step33
from parciver import telemetry as T

spec = importlib.util.spec_from_file_location(
    "step33_runner", Path(__file__).resolve().parents[1] / "scripts" / "rmapss_step33.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_cli_has_no_research_setting_overrides():
    with pytest.raises(SystemExit) as exc:
        runner.build_parser().parse_args(["--fit-hands", "1"])
    assert exc.value.code == 2


def test_runner_refuses_missing_audit_before_creating_run(tmp_path):
    run_dir = tmp_path / "must-not-exist"
    args = SimpleNamespace(
        protocol=step33.PROTOCOL,
        threads=16,
        audit_receipt=tmp_path / "missing.json",
        run_dir=run_dir,
    )
    with pytest.raises(ValueError, match="missing S3.3 engineering-audit receipt"):
        runner.run(args)
    assert not run_dir.exists()


def test_resume_rejects_terminal_gate_failure(tmp_path):
    hashes = {str(path): digest for path, digest in step33.FROZEN_ARTIFACTS.items()}
    state = runner._initial_state(
        hashes, {"small_game_control": {"passed": True},
                 "parity_control": {"passed": True}})
    state["terminal_failure"] = True
    state["failure"] = "locked gate breach"
    path = tmp_path / "s3.3-results.json"
    runner._write_json_atomic(path, state)
    with pytest.raises(ValueError, match="terminal"):
        runner._load_resume_state(path, hashes)


def test_worker_cap_is_locked_before_audit_lookup(tmp_path):
    args = SimpleNamespace(
        protocol=step33.PROTOCOL,
        threads=17,
        audit_receipt=tmp_path / "missing.json",
    )
    with pytest.raises(ValueError, match="1..=16"):
        runner.run(args)


def test_s33_dashboard_renders_observation_and_gates(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    row = {
        "seed": 311, "label": "nit(0.20)", "effective_coverage": 0.995,
        "legacy_coverage": 0.949, "model_nll": 0.8, "reference_nll": 2.5,
        "legacy_nll": 0.81, "ece": 0.02, "compute_s": 10.0,
        "peak_memory_gb": 2.0,
    }
    report = {
        "protocol": step33.PROTOCOL, "all_pass": True,
        "verdict": "PASS (S3.3 locked observation recovery)",
        "gates": {"completeness/leakage": True, "continuous coverage": True},
    }
    with T.RunWriter(tmp_path) as writer:
        writer.start(variant="s3.3", game="nlhe_pluribus_100bb",
                     config={"kind": "experiment", "protocol": step33.PROTOCOL,
                             "threads": 16})
        writer.checkpoint(iteration=1, step33_observation_row=row,
                          step33_progress={"stage": "observation",
                                           "completed": 1, "total": 27})
        writer.checkpoint(iteration=2, step33_observation=report)
        writer.end(iteration=2, summary=report)
    check = subprocess.run(
        [node, str(Path(__file__).resolve().parents[1]
                   / "scripts" / "check_experiment33_dom.mjs"),
         str(tmp_path / "events.jsonl")],
        capture_output=True, text=True, timeout=30,
    )
    assert check.returncode == 0, check.stdout + check.stderr
