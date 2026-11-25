# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Confirmation orchestration must not hide a failed seed or silently retune."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("parciver_engine")

from parciver import telemetry as T
from parciver.rmapss import step3

spec = importlib.util.spec_from_file_location(
    "step3_runner", Path(__file__).resolve().parents[1] / "scripts" / "rmapss_step3.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_confirmation_runs_all_seeds_and_keeps_individual_failure(tmp_path, monkeypatch):
    visited = []

    def fake_run(args):
        visited.append(args.seed)
        ok = args.seed != 7
        return SimpleNamespace(all_pass=ok, verdict="PASS" if ok else "FAIL",
                               scale_S=1.0, gates={"test gate": ok}, detail={})

    monkeypatch.setattr(runner, "run", fake_run)
    args = SimpleNamespace(run_dir=tmp_path / "confirmation", port=8774)
    assert runner.run_confirmation(args) == 1
    assert visited == [7, 19, 41]
    events = T.read_events(args.run_dir)
    rows = [e["confirmation_seed"] for e in events if "confirmation_seed" in e]
    assert [r["all_pass"] for r in rows] == [False, True, True]
    assert events[-1]["summary"]["all_pass"] is False


@pytest.mark.parametrize("override", [["--coarse"], ["--seed", "7"], ["--anchor-iters", "1"]])
def test_confirmation_cli_rejects_protocol_overrides(override):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--protocol", step3.CONFIRMATION_PROTOCOL, *override])
    assert exc.value.code == 2


def test_confirmation_library_rejects_unregistered_anchor():
    with pytest.raises(ValueError, match="locked anchor"):
        step3.run_step3(protocol=step3.CONFIRMATION_PROTOCOL, anchor={})
