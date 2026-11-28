# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""`parciver doctor`: reports what is installed, never changes anything, and names the fix for each gap."""
from __future__ import annotations

import sys

import pytest

from parciver import assets, cli, doctor as D


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PARCIVER_HOME", str(tmp_path))
    monkeypatch.delenv("PARCIVER_SLUMBOT_SAMPLE", raising=False)
    # a tiny stand-in manifest: the real tables are gigabytes and the doctor only compares sizes
    fake = {"files": [{"name": "a.gz", "dest": "blueprint.bin", "bytes": 64, "sha256": "0" * 64},
                      {"name": "b.gz", "dest": "abstraction/flop_buckets.npy", "bytes": 32, "sha256": "0" * 64}],
            "nets": {"files": [{"name": "n.pt", "dest": "nets/N0.pt", "bytes": 16, "sha256": "0" * 64}]}}
    monkeypatch.setattr(assets, "load_manifest", lambda: fake)
    return tmp_path


def _sparse(p, n):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)


def _install(root, entry):
    _sparse(root / entry["dest"], entry["bytes"])


def test_an_empty_home_points_at_fetch_blueprint_first(tmp_home, monkeypatch):
    monkeypatch.setattr(D, "engine_present", lambda: True)
    checks = D.run_checks()
    can, steps = D.summary(checks)
    assert can == ["kuhn", "leduc"]
    assert steps == ["parciver fetch-blueprint"]  # one line, however many tables are missing
    assert not any(c.status == D.OK for c in checks if c.name.startswith("table "))


def test_a_missing_engine_comes_before_the_download(tmp_home, monkeypatch):
    monkeypatch.setattr(D, "engine_present", lambda: False)
    _, steps = D.summary(D.run_checks())
    assert steps[0].startswith("pip install maturin")
    assert steps[1] == "parciver fetch-blueprint"


def test_everything_installed_means_nlhe_is_playable(tmp_home, monkeypatch):
    monkeypatch.setattr(D, "engine_present", lambda: True)
    for e in assets.load_manifest()["files"]:
        _install(tmp_home, e)
    can, steps = D.summary(D.run_checks())
    assert can == ["kuhn", "leduc", "nlhe"]
    assert steps == []


def test_a_truncated_table_is_reported_as_damaged_not_ok(tmp_home, monkeypatch):
    monkeypatch.setattr(D, "engine_present", lambda: True)
    files = assets.load_manifest()["files"]
    for e in files:
        _install(tmp_home, e)
    _sparse(tmp_home / files[0]["dest"], 1234)
    checks = {c.name: c for c in D.run_checks()}
    c = checks[f"table {files[0]['dest']}"]
    assert c.status == D.MISSING and "damaged or partial" in c.detail and c.fix == "parciver fetch-blueprint"
    assert "nlhe" not in D.summary(list(checks.values()))[0]


def test_the_gpu_tier_never_blocks_the_playable_games(tmp_home, monkeypatch):
    monkeypatch.setattr(D, "engine_present", lambda: True)
    for e in assets.load_manifest()["files"]:
        _install(tmp_home, e)
    monkeypatch.setattr(D, "gpu_state", lambda: (D.SKIP, "PyTorch is not installed"))
    checks = D.run_checks()
    assert any(c.tier == 3 and c.status != D.OK for c in checks)
    can, steps = D.summary(checks)
    assert "nlhe" in can and steps == []


def test_missing_flop_nets_are_optional_not_missing(tmp_home):
    nets = [c for c in D.run_checks() if c.name.startswith("flop net ")]
    assert nets and all(c.status == D.SKIP and c.fix == "parciver fetch-nets" for c in nets)


def test_the_report_is_readable_and_changes_nothing(tmp_home, monkeypatch):
    monkeypatch.setattr(D, "engine_present", lambda: True)
    out: list[str] = []
    assert D.run(say=out.append) == 0
    text = "\n".join(out)
    assert "Kuhn and Leduc poker" in text and "No-limit hold'em" in text and "Full GPU configuration" in text
    assert "You can play now: kuhn, leduc" in text
    assert "Next, in order:" in text and "1. parciver fetch-blueprint" in text
    assert list(tmp_home.iterdir()) == []  # a read-only check: it created nothing


def test_a_broken_torch_does_not_crash_the_doctor(monkeypatch):
    monkeypatch.setattr(D, "_find", lambda mod: True)  # pretend torch is installed ...
    monkeypatch.setitem(sys.modules, "torch", None)  # ... but importing it raises ImportError
    st, detail = D.gpu_state()
    assert st == D.MISSING and "fails to import" in detail


def test_tables_found_by_play_nlhe_elsewhere_count_as_installed(tmp_home, monkeypatch):
    """`play nlhe` also reads a source checkout's data/ folder or --blueprint; the doctor must agree with it."""
    monkeypatch.setattr(D, "engine_present", lambda: True)
    elsewhere = tmp_home / "checkout-data" / "blueprint.bin"
    monkeypatch.setattr(D, "locate_assets", lambda: (elsewhere.parent / "abstraction", elsewhere))
    can, steps = D.summary(D.run_checks())
    assert "nlhe" in can and steps == []


def test_cli_doctor_and_the_no_argument_guide(tmp_home, monkeypatch, capsys):
    monkeypatch.setattr(D, "engine_present", lambda: True)
    assert cli.main(["doctor"]) == 0
    assert "parciver doctor" in capsys.readouterr().out
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "parciver play kuhn" in out and "parciver doctor" in out and "BEGINNERS" in out
