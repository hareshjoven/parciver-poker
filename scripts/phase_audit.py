#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Phase-end audit for parciver -- a hallucination guard.

The single biggest risk in this project is a claim ("Phase X is done and
validated", "288 infosets", "~20x faster", "exploitability < 4e-2") that is true
in a narrative but NOT true in the repository. This script exists
to make that impossible to get away with: it re-derives every headline claim
directly from the code -- compiling the Rust engine, running both test suites,
and independently recomputing the ground-truth numbers (infoset counts,
exploitability, evaluator ordering, Rust-vs-Python parity) -- and prints
PASS/FAIL for each. It verifies the CODE, not the repository: git/commit state is
deliberately out of scope. Nothing here trusts the session's memory of what was
done.

Ritual: run this at the END OF EVERY PHASE, and only recommend starting a fresh
session when it is all green. If a check is red, the phase is not actually done.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/phase_audit.py [--rebuild] [--full]

    --rebuild  Rebuild the Rust extension first (maturin develop --release).
               Use after changing engine/src to prove the committed source
               actually compiles and the tests run against a fresh artifact.
    --full     Also run the exhaustive evaluator certification (all
               133,784,560 seven-card hands via the #[ignore]d cargo test;
               ~2 min). Run this whenever a phase touched engine/src/hand_eval.

Exit code is 0 only if every non-informational check passed.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
PYEXE = sys.executable
_S32_SMALL_GAME_CONTROL: dict = {"passed": False, "note": "not run"}
_S33_PARITY_CONTROL: dict = {"passed": False, "note": "not run"}
_S34_PARITY_CONTROL: dict = {"passed": False, "note": "not run"}
_S34_PERFORMANCE_CONTROL: dict = {"passed": False, "note": "not run"}


@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    informational: bool = False  # reported, but never fails the audit


def _run(cmd: list[str], cwd: Path, env_extra: dict | None = None, timeout: int = 900):
    import os

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    # Force UTF-8 with replacement: cargo/maturin emit UTF-8 (emoji, box glyphs)
    # that the Windows locale (cp1252) cannot decode, which otherwise crashes the
    # reader thread. An audit must never fall over on tool output encoding.
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# --- individual checks ----------------------------------------------------
# Each returns a Result. Any exception is caught by the runner and turned into a
# FAIL, so a broken check can never masquerade as a pass.


def check_environment() -> Result:
    import torch

    ok = sys.version_info[:2] == (3, 12)
    detail = (
        f"python {sys.version.split()[0]}, torch {torch.__version__}, "
        f"cuda={torch.cuda.is_available()}"
    )
    return Result("environment/toolchain", ok and torch.cuda.is_available(), detail)


def check_files_exist() -> Result:
    expected = [
        "engine/Cargo.toml",
        "engine/pyproject.toml",
        "engine/src/lib.rs",
        "engine/src/game.rs",
        "engine/src/cfr.rs",
        "engine/src/kuhn.rs",
        "engine/src/leduc.rs",
        "engine/src/cards.rs",
        "engine/src/hand_eval.rs",
        "engine/src/abstraction.rs",
        "engine/src/equity.rs",
        "engine/src/features.rs",
        "engine/src/isomorphism.rs",
        "engine/src/mccfr.rs",
        "engine/src/nlhe.rs",
        "engine/src/s32.rs",
        "engine/src/bucketing.rs",
        "engine/src/exploit.rs",
        "src/parciver/games/leduc.py",
        "src/parciver/games/leduc_multi.py",
        "src/parciver/rmapss/step2.py",
        "src/parciver/rmapss/step2_bait.py",
        "src/parciver/rmapss/step2_learn.py",
        "src/parciver/rmapss/stepQ.py",
        "src/parciver/rmapss/step3.py",
        "src/parciver/rmapss/step32.py",
        "src/parciver/rmapss/step33.py",
        "src/parciver/rmapss/step34.py",
        "src/parciver/rmapss/fast_tree.py",
        "scripts/rmapss_step2.py",
        "scripts/rmapss_step2a.py",
        "scripts/rmapss_step2b.py",
        "scripts/rmapss_stepQ.py",
        "scripts/rmapss_step3.py",
        "scripts/rmapss_step32.py",
        "scripts/rmapss_step33.py",
        "scripts/rmapss_step34.py",
        "scripts/check_experiment32_dom.mjs",
        "scripts/check_experiment33_dom.mjs",
        "scripts/check_experiment34_dom.mjs",
        "src/parciver/dashboard/static/experiment.html",
        "src/parciver/dashboard/static/experiment.js",
        "src/parciver/dashboard/static/experimentQ.html",
        "src/parciver/dashboard/static/experimentQ.js",
        "src/parciver/dashboard/static/experiment3.html",
        "src/parciver/dashboard/static/experiment3.js",
        "src/parciver/dashboard/static/experiment32.html",
        "src/parciver/dashboard/static/experiment32.js",
        "src/parciver/dashboard/static/experiment33.html",
        "src/parciver/dashboard/static/experiment33.js",
        "src/parciver/dashboard/static/experiment34.html",
        "src/parciver/dashboard/static/experiment34.js",
        "src/parciver/blueprint_artifact.py",
        "src/parciver/solvers/cfr.py",
        "src/parciver/eval/exploitability.py",
        "src/parciver/abstraction.py",
        "src/parciver/clustering.py",
        "src/parciver/bucketing.py",
        "src/parciver/telemetry.py",
        "src/parciver/resources.py",
        "src/parciver/dashboard/server.py",
        "src/parciver/dashboard/static/index.html",
        "src/parciver/dashboard/static/app.js",
        "src/parciver/dashboard/static/style.css",
        "scripts/build_buckets.py",
        "scripts/dashboard.py",
        "scripts/train_small.py",
        "scripts/replay_demo.py",
        "scripts/train_blueprint.py",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    detail = "all present" if not missing else f"MISSING: {missing}"
    return Result("expected source files exist", not missing, detail)


def check_extension_api() -> Result:
    import importlib

    e = importlib.import_module("parciver_engine")
    expected = {
        "solve_leduc",
        "solve_kuhn",
        "solve_leduc_mccfr",
        "solve_kuhn_mccfr",
        "leduc_infoset_count",
        "kuhn_infoset_count",
        "solve_kuhn_n",
        "kuhn_n_infoset_count",
        "kuhn_n_nashconv",
        "kuhn_n_nashconv_per_player",
        "kuhn_n_best_response_value",
        "kuhn_n_on_policy_value",
        "eval7",
        "eval5",
        "hand_category",
        "hand_category_name",
        "parse_card",
        "preflop_index",
        "preflop_name",
        "canonical_cards",
        "equity_vs_random",
        "equity_vs_hand",
        "equity_vs_random_mc",
        "flop_equity_histogram",
        "turn_equity_histogram",
        "river_equity",
        "flop_histograms_batch",
        "turn_histograms_batch",
        "river_equities_batch",
        "canonical_situation",
        "enumerate_canonical_flops",
        "canonical_index_size",
        "street_card_count",
        "canonical_index",
        "canonical_unindex_batch",
        "Abstraction",
        "kuhn_exploitability",
        "leduc_exploitability",
        "solve_nlhe_small",
        "solve_nlhe_small_mccfr",
        "nlhe_small_fixed_response",
        "nlhe_small_exploitability",
        "nlhe_small_infoset_count",
        "nlhe_small_best_response_value",
        "nlhe_small_best_response_strategy",
        "nlhe_small_on_policy_value",
        "nlhe_small_observed_marginal",
        "nlhe_small_observed_marginal_nodes",
        "nlhe_small_observed_model",
        "nlhe_small_infoset_structure",
        "Blueprint",
        "SavedBlueprint",
        "S32Policy",
        "S32Observation",
    }
    missing = sorted(n for n in expected if not hasattr(e, n))
    detail = "all functions present" if not missing else f"MISSING: {missing}"
    return Result("parciver_engine API surface", not missing, detail)


def check_rust_tests(full: bool) -> Result:
    env = {"PYO3_PYTHON": PYEXE}
    rc, out, err = _run(
        ["cargo", "test", "--no-default-features", "--release"],
        ENGINE,
        env,
        timeout=600,
    )
    m = re.findall(r"test result: ok\. (\d+) passed", out + err)
    passed = sum(int(x) for x in m)
    ok = rc == 0
    detail = f"cargo test: {passed} passed, rc={rc}"
    if not ok:
        detail += " | " + (err.strip().splitlines()[-1] if err.strip() else "see output")
    if full and ok:
        # All #[ignore]d exhaustive certs: the 7-card category histogram AND the
        # eval7_fast==eval7 bit-equality over all 133,784,560 hands.
        rc2, out2, err2 = _run(
            [
                "cargo", "test", "--no-default-features", "--release",
                "--", "--ignored",
            ],
            ENGINE, env, timeout=1200,
        )
        m2 = re.findall(r"test result: ok\. (\d+) passed", out2 + err2)
        n_ignored = sum(int(x) for x in m2)
        ok = rc2 == 0 and n_ignored >= 2
        detail += (
            f" | exhaustive certs (7-card counts + eval7_fast parity): "
            f"{'PASS' if ok else 'FAIL'} ({n_ignored} ran, rc={rc2})"
        )
    elif not full:
        detail += " | (exhaustive certs skipped; run with --full)"
    return Result("rust unit tests", ok, detail)


def check_python_suite() -> Result:
    """Run every Python test file in four isolated, balanced-enough shards.

    The serial suite exceeded ten minutes once the R-MAPSS decider and full NLHE
    controls accumulated.  Files share no pytest fixture scope, so independent
    subprocesses preserve the coverage/order contract while bounding wall time by
    the slowest shard.  Each shard retains the historical 600-second hard cap.
    """
    from concurrent.futures import ThreadPoolExecutor

    files = sorted((ROOT / "tests").glob("test_*.py"))
    workers = min(4, len(files))
    shards: list[list[Path]] = [[] for _ in range(workers)]
    # Distribute adjacent heavy research modules across workers deterministically.
    for index, path in enumerate(files):
        shards[index % workers].append(path)

    def run_shard(index: int, paths: list[Path]):
        command = [PYEXE, "-m", "pytest", "-q", "-o", "addopts="]
        command.extend(str(path.relative_to(ROOT)) for path in paths)
        rc, out, err = _run(command, ROOT, timeout=600)
        return index, rc, out, err

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda pair: run_shard(*pair), enumerate(shards)))
    rows.sort(key=lambda row: row[0])
    passed = 0
    failures = []
    for index, rc, out, err in rows:
        matches = re.findall(r"(\d+) passed", out + err)
        passed += sum(int(value) for value in matches)
        if rc != 0:
            tail = (out + "\n" + err).strip().splitlines()
            failures.append(f"shard {index}: rc={rc}, {tail[-1] if tail else 'no output'}")
    elapsed = time.perf_counter() - started
    detail = f"pytest: {passed} passed across {workers} shards in {elapsed:.1f}s"
    if failures:
        detail += " | " + "; ".join(failures)
    return Result("python test suite", not failures and passed > 0, detail)


def check_infoset_parity() -> Result:
    """Rust and Python must agree on the exact infoset partition, independently."""
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.games.kuhn import KuhnPoker
    from parciver.games.leduc import LeducHoldem
    from parciver.solvers.cfr import DCFR, CFRSolver

    def py_keys(game):
        s = CFRSolver(game=game, variant=DCFR, alternating=False)
        s.run(60)
        return set(s.nodes.keys())

    leduc_rust = set(e.solve_leduc(60, "dcfr").keys())
    kuhn_rust = set(e.solve_kuhn(60, "dcfr").keys())
    leduc_py = py_keys(LeducHoldem())
    kuhn_py = py_keys(KuhnPoker())

    ok = (
        e.leduc_infoset_count() == 288
        and e.kuhn_infoset_count() == 12
        and leduc_rust == leduc_py
        and kuhn_rust == kuhn_py
    )
    detail = (
        f"leduc: rust={e.leduc_infoset_count()} py={len(leduc_py)} "
        f"keys_equal={leduc_rust == leduc_py}; "
        f"kuhn: rust={e.kuhn_infoset_count()} py={len(kuhn_py)} "
        f"keys_equal={kuhn_rust == kuhn_py}"
    )
    return Result("infoset parity (rust==python)", ok, detail)


def check_kuhn_n_gates() -> Result:
    """N-player Kuhn foundation (RMAPSS.md §33, Step 0.2).

    Re-derives from ground truth: infoset counts (2p->12, 3p->48, 4p Rust==Python),
    the KuhnN(2)-reduces-to-certified-2p-Kuhn key identity, the zero-sum invariant,
    the Rust<->Python parity of exact BR / on-policy EV / per-player + total NashConv,
    and the EMPIRICAL 3p-Kuhn low-NashConv property (CFR has no >=3p guarantee, so
    this is a measured regression bound, not a theorem). The independent BR-vs-brute-
    force pure-strategy ground truth (§28) runs in the Rust suite (~60s)."""
    import parciver_engine as e

    needed = [
        "solve_kuhn_n", "kuhn_n_infoset_count", "kuhn_n_nashconv",
        "kuhn_n_nashconv_per_player", "kuhn_n_best_response_value",
        "kuhn_n_on_policy_value",
    ]
    missing = [n for n in needed if not hasattr(e, n)]
    if missing:
        return Result(
            "n-player kuhn foundation", False,
            f"engine missing {missing} -- rebuild: maturin develop --release -m engine/Cargo.toml",
        )

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.games.kuhn_multi import KuhnPokerN
    from parciver.games.kuhn import KuhnPoker
    from parciver.eval.exploitability import (
        best_response_value, nash_conv_multiplayer, on_policy_value,
    )

    def py_keys(game):
        keys = set()

        def walk(s):
            if game.is_terminal(s):
                return
            if game.is_chance(s):
                for a, _ in game.chance_outcomes(s):
                    walk(game.next_state(s, a))
                return
            keys.add(game.infoset_key(s))
            for a in game.legal_actions(s):
                walk(game.next_state(s, a))

        walk(game.initial_state())
        return keys

    c2, c3, c4 = (e.kuhn_n_infoset_count(n) for n in (2, 3, 4))
    counts_ok = (
        c2 == 12 == len(py_keys(KuhnPokerN(2)))
        and c3 == 48 == len(py_keys(KuhnPokerN(3)))
        and c4 == len(py_keys(KuhnPokerN(4)))
    )
    reduce_ok = py_keys(KuhnPokerN(2)) == py_keys(KuhnPoker())

    g3 = KuhnPokerN(3)
    strat = e.solve_kuhn_n(3, 4000, "cfr")
    zs_ok = abs(sum(on_policy_value(g3, strat, i) for i in range(3))) < 1e-9
    nc_rs, nc_py = e.kuhn_n_nashconv(3, strat), nash_conv_multiplayer(g3, strat)
    parity_ok = abs(nc_rs - nc_py) < 1e-9
    for p in range(3):
        parity_ok = parity_ok and abs(
            e.kuhn_n_best_response_value(3, strat, p) - best_response_value(g3, strat, p)
        ) < 1e-9
        parity_ok = parity_ok and abs(
            e.kuhn_n_on_policy_value(3, strat, p) - on_policy_value(g3, strat, p)
        ) < 1e-9

    best_nc = min(e.kuhn_n_nashconv(3, e.solve_kuhn_n(3, 5000, v)) for v in ("cfr", "cfr+", "dcfr"))
    emp_ok = 0.0 <= best_nc < 0.05

    ok = counts_ok and reduce_ok and zs_ok and parity_ok and emp_ok
    detail = (
        f"counts 2p/3p/4p={c2}/{c3}/{c4}; 2p-reduces-to-Kuhn={reduce_ok}; "
        f"zero-sum={zs_ok}; NashConv rust==py ({nc_rs:.3e}); BR/on-policy parity={parity_ok}; "
        f"empirical best 3p CFR NashConv={best_nc:.3e}"
    )
    return Result("n-player kuhn foundation", ok, detail)


def check_rmapss_step1_gates() -> Result:
    """R-MAPSS Step 1 — multiplayer exploitation spine (RMAPSS.md §33, 3p Kuhn).

    Re-runs the exact, tabular exploitation experiment and re-derives its gates:
    the confidence-scheduled safety-gated deviation stays within the exploitability
    budget on every opponent pair, beats CFR on suboptimal opponents (including
    HELD-OUT ones, R4), matches CFR on (near-)equilibrium opponents, and the
    independent (marginal) model never exceeds the exact BR ceiling (R2). All EVs
    are exact (full traversal); the exact N-player BR is the §28 control."""
    sys.path.insert(0, str(ROOT / "src"))
    from parciver.rmapss.step1 import run_step1

    rep = run_step1()
    ok = rep.all_pass
    failed = [k for k, v in rep.gates.items() if not v]
    detail = (
        f"gates {sum(rep.gates.values())}/{len(rep.gates)} pass; "
        f"eq NashConv={rep.detail['eq NashConv (exact)']:.4f}; "
        f"mean suboptimal gain={rep.detail['mean exploit gain (suboptimal)']:.4f} "
        f"(held-out {rep.detail['mean exploit gain (suboptimal HELD-OUT)']:.4f}); "
        f"max |eq gain|={rep.detail['max |exploit gain| (equilibrium)']:.4f}"
    )
    if failed:
        detail += " | FAILED: " + ", ".join(failed)
    return Result("rmapss step1 exploitation spine", ok, detail)


def check_rmapss_step3_gates() -> Result:
    """R-MAPSS Step 3 / S3.1 — the HU exploitation spine on REAL reduced NLHE
    (`nlhe_small`, 456 infosets; RMAPSS.md §40/§41). Re-runs the exact spine on a
    seeded MCCFR anchor and re-derives its exploratory regression gates from ground
    truth: it beats the equilibrium anchor vs HELD-OUT observably-distorted
    opponents (> δ_sub), matches the anchor vs near-eq (|gain| ≤ tol), DECLINES the
    hole-blind model boundary (§41.DEV1), keeps
    the EXACT worst-case exploitability within the S-scaled budget on every
    opponent, and never lets the observed-marginal BR exceed the exact BR ceiling
    (§28 anti-hallucination). Scale S uses FIT only. Historical threshold revisions
    make this an engineering regression, NOT a pre-registered confirmation.
    A coarse α grid keeps it tractable; exact controls remain the ground truth."""
    sys.path.insert(0, str(ROOT / "src"))
    import parciver_engine as pe
    from parciver.rmapss.step3 import run_step3

    anchor = pe.solve_nlhe_small_mccfr(120_000, "linear", 0)
    rep = run_step3(anchor=anchor, alpha_grid=(1.0, 0.7, 0.45, 0.25, 0.1, 0.0))
    ok = rep.all_pass
    failed = [k for k, v in rep.gates.items() if not v]
    # independent §28 restatements (belt-and-braces on the report's own gates)
    no_halluc = all(r.ev_a2 <= r.ev_ceiling + 1e-9 for r in rep.results)
    safe = all(r.expl_final <= rep.base_expl + rep.budget + 1e-9 for r in rep.results)
    ok = ok and no_halluc and safe
    detail = (
        f"gates {sum(rep.gates.values())}/{len(rep.gates)} pass; "
        f"protocol={rep.protocol}; S(FIT ceiling-gain scale)={rep.scale_S:.3f}; anchor expl={rep.base_expl:.4f}; "
        f"delta_sub={rep.delta_sub:.4f}/tol={rep.tol_eq:.4f}/budget={rep.budget:.4f}; "
        f"mean held-out gain={rep.detail['mean exploit gain (suboptimal HELD-OUT)']:.4f}; "
        f"max eq |gain|={rep.detail['max |exploit gain| (equilibrium)']:.4f}; "
        f"hidden foregoes={rep.detail.get('hidden ceiling gain (foregone, R2 boundary)', 0.0):.3f} "
        f"(realizes {rep.detail.get('hidden realized gain (near 0 = correct decline)', 0.0):.3f}); "
        f"sec28 no-hallucination={no_halluc}, within-budget={safe}"
    )
    if failed:
        detail += " | FAILED: " + ", ".join(failed)
    return Result("rmapss step3 exploitation spine (nlhe_small)", ok, detail)


def check_rmapss_step32_gates() -> Result:
    """S3.2 pre-outcome engineering gate (RMAPSS §43).

    Verify frozen full-game artifacts without evaluating product outcomes, then
    run the sampled response path at its locked 5M budget on exact ``nlhe_small``
    FIT controls. The control record is sealed into the production audit receipt.
    """
    global _S32_SMALL_GAME_CONTROL
    sys.path.insert(0, str(ROOT / "src"))
    import parciver_engine as pe
    from parciver.rmapss import step32, step3

    hashes = step32.verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(step32.ABSTRACTION_DIR), str(step32.BLUEPRINT))
    artifact_ok = saved.num_infosets() == 13_469_060

    anchor = pe.solve_nlhe_small_mccfr(120_000, "linear", 73)
    structure = step3.infoset_structure()
    p0, p1 = step3.split_keys(structure)
    boundary = pe.nlhe_small_observed_marginal(anchor, anchor, 0)
    captures = {}
    rows = []
    for index, kind in enumerate(("nit", "station", "maniac")):
        opponent = step3.build_opponent(
            structure, p1, anchor, kind, board_blind_flat=boundary
        )
        anchor_hero = {
            key: list(anchor.get(
                key, [1.0 / len(structure[key][1])] * len(structure[key][1])
            ))
            for key in p0
        }
        fixed_profile = {**anchor, **opponent}
        baseline = pe.nlhe_small_on_policy_value({**anchor_hero, **opponent}, 0)
        exact = pe.nlhe_small_best_response_value(fixed_profile, 0)
        response = pe.nlhe_small_fixed_response(
            fixed_profile, 0, step32.RESPONSE_ITERS, "linear", 32_000 + index, 1
        )
        realized = pe.nlhe_small_on_policy_value({**opponent, **response}, 0)
        available = exact - baseline
        captured = ((realized - baseline) / available) if available > 0 else float("nan")
        captures[kind] = captured
        rows.append({
            "family": kind, "baseline": baseline, "realized": realized,
            "exact_br": exact, "captured_fraction": captured,
            "under_exact_br": realized <= exact + 1e-9,
        })

    passed = (
        artifact_ok
        and len(hashes) == len(step32.FROZEN_ARTIFACTS)
        and all(row["under_exact_br"] and row["captured_fraction"] >= 0.95 for row in rows)
    )
    _S32_SMALL_GAME_CONTROL = {
        "passed": passed,
        "protocol": step32.PROTOCOL,
        "iterations": step32.RESPONSE_ITERS,
        "anchor_iters": 120_000,
        "anchor_seed": 73,
        "rows": rows,
    }
    detail = (
        f"artifacts={len(hashes)}/4, infosets={saved.num_infosets():,}; "
        + ", ".join(f"{kind}={value:.3%}" for kind, value in captures.items())
        + " exact-BR gain captured"
    )
    return Result("rmapss S3.2 locked pre-outcome controls", passed, detail)


def check_rmapss_step33_parity() -> Result:
    """S3.3 worker-count invariance on the frozen full-game engine path."""
    global _S33_PARITY_CONTROL
    sys.path.insert(0, str(ROOT / "src"))
    import parciver_engine as pe
    from parciver.rmapss import step33

    step33.verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(step33.ABSTRACTION_DIR), str(step33.BLUEPRINT))
    anchor = saved.s32_policy()
    opponent = anchor.leak("nit", 0.35)
    seeds = [91_001, 91_003, 91_007, 91_009]
    observations = {}
    scores = {}
    paired = {}
    reference = anchor.observe_blocks(anchor, 100, [92_001, 92_003, 92_007, 92_009], 1)
    ref = reference[0]
    for block in reference[1:]:
        ref = ref.merged(block)
    for threads in (1, 2, 16):
        blocks = anchor.observe_blocks(opponent, 100, seeds, threads)
        observations[threads] = [block.deterministic_fingerprint() for block in blocks]
        train = blocks[0]
        for block in blocks[1:]:
            train = train.merged(block)
        model = train.fit_hierarchical_model(ref, 200.0, 50.0, 20.0)
        scores[threads] = [dict(block.score_hierarchical_model(model)) for block in blocks]
        paired[threads] = dict(anchor.paired_evaluate_blocks(
            anchor, opponent, 4_000, 93_001, 1_000, threads
        ))
    passed = (
        observations[1] == observations[2] == observations[16]
        and scores[1] == scores[2] == scores[16]
        and paired[1] == paired[2] == paired[16]
    )
    _S33_PARITY_CONTROL = {
        "passed": passed,
        "protocol": step33.PROTOCOL,
        "threads": [1, 2, 16],
        "observation_fingerprints": observations[1],
        "paired_block_gains_bb_per_100": paired[1]["block_gains_bb_per_100"],
    }
    return Result(
        "rmapss S3.3 fixed-block 1/2/16-thread parity",
        passed,
        (f"observation_blocks={len(observations[1])}, "
         f"score_blocks={len(scores[1])}, paired_blocks="
         f"{len(paired[1]['block_gains_bb_per_100'])}; byte/value-identical={passed}"),
    )


def check_rmapss_step34_controls() -> Result:
    """S3.4 five-level parity and locked engineering performance fixture."""
    global _S34_PARITY_CONTROL, _S34_PERFORMANCE_CONTROL
    sys.path.insert(0, str(ROOT / "src"))
    import parciver_engine as pe
    from parciver.rmapss import step34

    step34.verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(step34.ABSTRACTION_DIR), str(step34.BLUEPRINT))
    anchor = saved.s32_policy()
    opponent = anchor.leak("nit", 0.35)
    fit_seeds = [94_001, 94_003, 94_007, 94_009]
    confirm_seeds = [95_001, 95_003, 95_007, 95_009]
    reference_blocks = anchor.observe_blocks(
        anchor, 100, [96_001, 96_003, 96_007, 96_009], 1)

    def merge(blocks):
        out = blocks[0]
        for block in blocks[1:]:
            out = out.merged(block)
        return out

    reference = merge(reference_blocks)
    fingerprints = {}
    scores = {}
    paired = {}
    for threads in (1, 2, 16):
        fit_blocks = anchor.observe_blocks(opponent, 100, fit_seeds, threads)
        confirmation = anchor.observe_blocks(opponent, 100, confirm_seeds, threads)
        fingerprints[threads] = [
            block.support_fingerprint() for block in fit_blocks + confirmation
        ]
        model = merge(fit_blocks).fit_five_level_model(
            reference, 200.0, 50.0, 20.0, 10.0, 5.0)
        scores[threads] = [
            dict(block.score_five_level_model(model)) for block in confirmation
        ]
        paired[threads] = dict(anchor.paired_evaluate_blocks(
            anchor, opponent, 4_000, 97_001, 1_000, threads))
    parity_passed = (
        fingerprints[1] == fingerprints[2] == fingerprints[16]
        and scores[1] == scores[2] == scores[16]
        and paired[1] == paired[2] == paired[16]
    )
    _S34_PARITY_CONTROL = {
        "passed": parity_passed,
        "protocol": step34.PROTOCOL,
        "threads": [1, 2, 16],
        "observation_fingerprints": fingerprints[1],
        "paired_block_gains_bb_per_100": paired[1]["block_gains_bb_per_100"],
    }

    def timed(version: int) -> tuple[float, int]:
        started = time.perf_counter()
        fit = merge(anchor.observe_blocks(opponent, 100, fit_seeds, 1))
        confirmation = anchor.observe_blocks(opponent, 100, confirm_seeds, 1)
        if version == 3:
            model = fit.fit_hierarchical_model(reference, 200.0, 50.0, 20.0)
            [block.score_hierarchical_model(model) for block in confirmation]
            retained = fit.count_table_bytes(False)
        else:
            model = fit.fit_five_level_model(reference, 200.0, 50.0, 20.0, 10.0, 5.0)
            [block.score_five_level_model(model) for block in confirmation]
            retained = fit.count_table_bytes(True)
        return time.perf_counter() - started, retained

    # One pipeline pass is only a few ms and ~90% of it is the observation stage
    # both versions share, so machine drift in that shared stage (+/-20% between
    # separately-timed runs) used to decide the verdict: the 1.10 bound flaked at
    # 1.134-1.155 while the version-specific fit+score costs ~0.3 vs ~0.45 ms.
    # PAIRED design: each pair times one S3.3 and one S3.4 pass back-to-back
    # (order alternating to cancel order bias), and the gate uses the median of
    # the per-pair ratios, so drift cancels within each pair. The locked
    # thresholds (step34.PERFORMANCE_*) are unchanged. S3.4 is terminal (RMAPSS
    # §49), so this is now an engine-regression gate; the exact §48.5 fixture
    # (1 warm-up, median of 5 each) is still computed and recorded alongside.
    WARMUPS, PAIRS = 2, 61

    timed(3)
    timed(4)
    locked33 = [timed(3)[0] for _ in range(5)]
    locked34 = [timed(4)[0] for _ in range(5)]
    locked_ratio = statistics.median(locked34) / statistics.median(locked33)

    for _ in range(WARMUPS):
        timed(3)
        timed(4)
    s33_times = []
    s34_times = []
    ratios = []
    s33_bytes = s34_bytes = 0
    for i in range(PAIRS):
        if i % 2 == 0:
            t33, s33_bytes = timed(3)
            t34, s34_bytes = timed(4)
        else:
            t34, s34_bytes = timed(4)
            t33, s33_bytes = timed(3)
        s33_times.append(t33)
        s34_times.append(t34)
        ratios.append(t34 / t33)
    time_ratio = statistics.median(ratios)
    bytes_ratio = s34_bytes / s33_bytes
    performance_passed = (
        time_ratio <= step34.PERFORMANCE_TIME_RATIO
        and bytes_ratio <= step34.PERFORMANCE_BYTES_RATIO
    )
    _S34_PERFORMANCE_CONTROL = {
        "passed": performance_passed,
        "protocol": step34.PROTOCOL,
        "warmups": WARMUPS,
        "timed_pairs": PAIRS,
        "estimator": "median of per-pair s3.4/s3.3 ratios (order alternated)",
        "locked_fixture_48_5_time_ratio": locked_ratio,
        "s3.3_median_s": statistics.median(s33_times),
        "s3.4_median_s": statistics.median(s34_times),
        "time_ratio": time_ratio,
        "s3.3_count_table_bytes": s33_bytes,
        "s3.4_count_table_bytes": s34_bytes,
        "bytes_ratio": bytes_ratio,
    }
    passed = parity_passed and performance_passed
    return Result(
        "rmapss S3.4 five-level parity/performance controls",
        passed,
        (f"1/2/16 parity={parity_passed}; paired time ratio={time_ratio:.3f} "
         f"<= {step34.PERFORMANCE_TIME_RATIO:.2f} (locked 48.5 fixture {locked_ratio:.3f}, "
         f"recorded only); count bytes ratio={bytes_ratio:.3f} "
         f"<= {step34.PERFORMANCE_BYTES_RATIO:.2f}"),
    )


def check_leduc_n_gates() -> Result:
    """N-player Leduc foundation (RMAPSS.md §33/§34, Step 2/3). Re-derives the
    correctness anchors from ground truth: N=2 reduces BYTE-IDENTICALLY to the
    certified 2p Leduc (same tree, 288 infoset keys, same per-player utilities), and
    3p Leduc is finite + total zero-sum (chip-conserving) at every terminal, with the
    exact BR (§28 control) dominating on-policy. No CFR fixpoint assumed (R5)."""
    sys.path.insert(0, str(ROOT / "src"))
    from parciver.games.leduc import LeducHoldem
    from parciver.games.leduc_multi import LeducHoldemN
    from parciver.solvers.cfr_multi import MultiCFR
    from parciver.eval.exploitability import best_response_value, on_policy_value

    # N=2 byte-identical reduction (tree lockstep by action path).
    def walk(game, s, path, acc):
        if game.is_terminal(s):
            acc["t"][path] = tuple(round(game.terminal_utility_for(s, p), 9)
                                   for p in range(game.num_players))
            return
        if game.is_chance(s):
            for a, _ in game.chance_outcomes(s):
                walk(game, game.next_state(s, a), path + (("C", a),), acc)
            return
        acc["d"][path] = (game.infoset_key(s), tuple(game.legal_actions(s)))
        for a in game.legal_actions(s):
            walk(game, game.next_state(s, a), path + (a,), acc)

    a2 = {"t": {}, "d": {}}
    an = {"t": {}, "d": {}}
    walk(LeducHoldem(), LeducHoldem().initial_state(), (), a2)
    walk(LeducHoldemN(2), LeducHoldemN(2).initial_state(), (), an)
    identical = (a2["t"] == an["t"] and a2["d"] == an["d"])
    n_infosets = len({an["d"][p][0] for p in an["d"]})

    # 3p finite + total zero-sum at every terminal.
    g3 = LeducHoldemN(3)
    bad = [0]
    nt = [0]

    def walk3(s):
        if g3.is_terminal(s):
            nt[0] += 1
            if abs(sum(g3.terminal_utility_for(s, p) for p in range(3))) > 1e-9:
                bad[0] += 1
            return
        if g3.is_chance(s):
            for a, _ in g3.chance_outcomes(s):
                walk3(g3.next_state(s, a))
            return
        for a in g3.legal_actions(s):
            walk3(g3.next_state(s, a))

    walk3(g3.initial_state())

    # §28: exact BR dominates on-policy on a short CFR profile (no fixpoint assumed).
    s = MultiCFR(g3, variant="cfr+")
    s.run(40)
    strat = s.average_strategy()
    br_ok = all(best_response_value(g3, strat, p) >= on_policy_value(g3, strat, p) - 1e-9
                for p in range(3))

    ok = identical and n_infosets == 288 and bad[0] == 0 and nt[0] == 33714 and br_ok
    detail = (f"N=2 byte-identical to certified Leduc: {identical} ({n_infosets} infosets); "
              f"3p terminals={nt[0]} zero-sum-violations={bad[0]}; exact BR>=on-policy: {br_ok}")
    return Result("N-player Leduc foundation", ok, detail)


def check_rmapss_step2_gates() -> Result:
    """R-MAPSS Step 2 — the interaction kernel machinery + its exact controls
    (RMAPSS.md §34). Runs the FAST 3p-Kuhn experiment (the within-round filter) and
    re-derives: FastTree is byte-for-byte the certified exact BR (speed costs no
    correctness); the correct-model MPC (A5) reproduces the exact ρ-ceiling at full
    horizon and never falls below the myopic baseline; realised EV stays under the
    valid upper bound; depth-1 == A2.5 (fairness). The DECISIVE Leduc verdict is a
    one-off run (scripts/rmapss_step2.py), recorded in HANDOFF — not re-run here."""
    sys.path.insert(0, str(ROOT / "src"))
    from parciver.games.kuhn_multi import KuhnPokerN
    from parciver.eval import exploitability as E
    from parciver.rmapss.fast_tree import FastTree
    from parciver.rmapss.step2 import run_step2, collect_seat_infosets, _uniform

    # FastTree byte-for-byte vs the certified game-driven exact BR / on-policy value.
    game = KuhnPokerN(3)
    ft = FastTree(game)
    eq = ft.solve_cfr(2000, "cfr+")
    _, abk = collect_seat_infosets(game)
    full = {k: eq.get(k, _uniform(len(abk[k]))) for k in abk}
    parity = all(
        ft.on_policy_value(full, p) == E.on_policy_value(game, full, p)
        and ft.best_response_value(full, p) == E.best_response_value(game, full, p)
        and ft.best_response_strategy(full, p) == E.best_response_strategy(game, full, p)
        for p in range(3)
    )

    rep = run_step2(game_name="3p-kuhn", m=6, depths=(1, 2, 3), cfr_iters=2500)
    ctl = {k: v for k, v in rep.gates.items() if k.startswith("ctl")}
    fairness = all(abs(dict(r.depths["A5"])[1] - r.ev["A2.5"]) < 1e-9 for r in rep.results)
    verdict_ok = rep.verdict in ("FILTER-PASS", "FILTER-FLAT", "INCONCLUSIVE")

    ok = parity and all(ctl.values()) and fairness and verdict_ok
    failed = [k for k, v in ctl.items() if not v]
    detail = (f"FastTree byte-parity: {parity}; controls {sum(ctl.values())}/{len(ctl)} pass; "
              f"depth1==A2.5: {fairness}; Kuhn verdict={rep.verdict} "
              f"(held A5-A2.5={rep.detail['mean A5-A2.5 (held)']:+.4f}, "
              f"A5-A4={rep.detail['mean A5-A4 (held)']:+.4f})")
    if failed:
        detail += " | FAILED: " + ", ".join(failed)
    return Result("rmapss step2 interaction kernel", ok, detail)


def check_rmapss_step2a_gates() -> Result:
    """R-MAPSS §35.A — the RICH-BAIT DECIDER. Runs the FAST 3p-Kuhn decider and
    re-derives the anti-hallucination gate: the composed g1 pipeline (block EV + the
    dense-ρ adaptation step + the myopic BR value) is byte-identical to the CERTIFIED
    game-driven oracle (so a headroom / A5−A4 finding cannot be a FastTree artifact);
    the structural invariants hold (headroom ≥ 0, headroom ≤ screen_gap, immediate ≤
    exact BR); and on escalation the exact controls hold (A5 ≥ A2.5, ceiling ≥ A2.5,
    A5 ≥ A4). The DECISIVE Leduc decider is a one-off (scripts/rmapss_step2a.py),
    recorded in HANDOFF/RMAPSS §35.A — not re-run every audit."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from parciver.games.kuhn_multi import KuhnPokerN
    from parciver.eval import exploitability as E
    from parciver.rmapss.step2 import build_shared, lifted_for, default_families, _init_opp_profile
    from parciver.rmapss.step2_bait import run_decider, BaitConfig, eval_g1, random_pure

    # -- composed-pipeline parity vs the certified oracle (the anti-hallucination gate) --
    game = KuhnPokerN(3)
    hero, eps = 0, 0.1
    sh = build_shared(game, hero, cfr_iters=1500)
    L = lifted_for(sh, eps)
    rng = np.random.default_rng(0)
    hero_keys = L.keys_by_seat[hero]
    parity = True
    for fam in default_families()[:3]:
        opp_id = _init_opp_profile(L, fam)
        opp = L.pol(opp_id)
        for hid in (L.br_now(opp_id)[0], L.intern(random_pure(hero_keys, L.actions_by_key, rng))):
            h = L.pol(hid)
            g = eval_g1(L, hid, opp_id, "pure")
            imm_cert = E.on_policy_value(game, {**h, **opp}, hero)
            succ = L.pol(g.succ_id)
            fut_cert = E.best_response_value(game, {**succ, **h}, hero)
            if abs(g.immediate - imm_cert) > 1e-12 or abs(g.future - fut_cert) > 1e-12:
                parity = False

    # -- run the fast decider + structural invariants --
    cfg = BaitConfig(n_pure=40, n_mixed=40, coord=True, coord_budget=300, seed=7)
    rep = run_decider(game_name="3p-kuhn", m=3, delta=0.02, cfr_iters=1500,
                      depth=3, n_states=3, cfg=cfg)
    ctl = {k: v for k, v in rep.gates.items() if k.startswith("ctl")}
    invariants = all(
        s.headroom >= -1e-9 and s.headroom <= s.screen_gap + 1e-9 and s.control_immediate_ok
        for r in rep.results for s in r.per_state)
    verdict_ok = rep.verdict in ("NULL-REAL", "MENU-ARTIFACT")
    esc_ok = True
    if rep.escalation is not None:
        esc_ok = rep.escalation.controls_ok and all(
            r["A5"] >= r["A2.5"] - 1e-9 and r["ceiling"] >= r["A2.5"] - 1e-9
            and r["a5_minus_a4"] >= -1e-9 for r in rep.escalation.per_family)

    ok = parity and all(ctl.values()) and invariants and verdict_ok and esc_ok
    a5a4 = (f", rich A5-A4(held)={rep.escalation.mean_a5_minus_a4_held:+.4f}"
            if rep.escalation is not None else "")
    detail = (f"g1==certified-oracle: {parity}; controls {sum(ctl.values())}/{len(ctl)} pass; "
              f"invariants(headroom>=0, <=screen, imm<=BR): {invariants}; esc_controls: {esc_ok}; "
              f"Kuhn verdict={rep.verdict} (max headroom held={rep.detail['max headroom (held)']:+.4f}{a5a4})")
    return Result("rmapss step2a rich-bait decider", ok, detail)


def check_rmapss_step2b_gates() -> Result:
    """R-MAPSS §35.B — the NO-REGRET-LEARNER decider. Re-derives the exact controls from
    ground truth on the FAST 3p-Kuhn learner decider: (a) the learner UPDATE cf_regrets is
    byte-identical to an INDEPENDENT game-driven counterfactual-regret walk (so the
    successor opp state cannot be a FastTree artifact); (b) the composed g1 (block EV +
    learner step + myopic BR value) matches the certified oracle's on-policy / BR values;
    (c) RM self-play reaches the certified 3p-Kuhn NashConv regime (cf_regrets is a valid
    no-regret learner); (d) the star-vs-dense transition diverges (opp→opp coupling present
    ⇒ A5≠A4 is possible, not forced 0); (e) the §28 controls hold and A5 ≥ A4 ≥/… A2.5. The
    DECISIVE Leduc run is a one-off (scripts/rmapss_step2b.py), recorded in HANDOFF/RMAPSS
    §37 — not re-run every audit."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from parciver.games.kuhn_multi import KuhnPokerN
    from parciver.eval import exploitability as E
    from parciver.rmapss.fast_tree import FastTree
    from parciver.rmapss.step2 import build_shared
    from parciver.rmapss.step2_bait import BaitConfig
    from parciver.rmapss.step2_learn import (
        cf_regrets, link_policy, LearnLifted, init_regret_for, default_learn_families,
        run_learn_decider,
    )

    game = KuhnPokerN(3)
    ft = FastTree(game)
    n = game.num_players

    # -- (a) cf_regrets == independent game-driven counterfactual-regret walk --
    def game_cf(profile, targets):
        reg = {p: {} for p in targets}

        def tr(state, reach, rc):
            if game.is_terminal(state):
                return [game.terminal_utility_for(state, i) for i in range(n)]
            if game.is_chance(state):
                v = [0.0] * n
                for a, p in game.chance_outcomes(state):
                    cv = tr(game.next_state(state, a), reach, rc * p)
                    for i in range(n):
                        v[i] += p * cv[i]
                return v
            pl = game.current_player(state); key = game.infoset_key(state)
            acts = game.legal_actions(state); na = len(acts)
            probs = profile.get(key) or [1.0 / na] * na
            avals, v = [], [0.0] * n
            for idx, a in enumerate(acts):
                cr = list(reach); cr[pl] *= probs[idx]
                cv = tr(game.next_state(state, a), cr, rc)
                avals.append(cv)
                for i in range(n):
                    v[i] += probs[idx] * cv[i]
            if pl in targets:
                cf = rc
                for j in range(n):
                    if j != pl:
                        cf *= reach[j]
                d = reg[pl].setdefault(key, [0.0] * na)
                for idx in range(na):
                    d[idx] += cf * (avals[idx][pl] - v[pl])
            return v
        tr(game.initial_state(), [1.0] * n, 1.0)
        return reg

    rng = np.random.default_rng(0)
    parity = True
    for _ in range(3):
        prof = {ft.dkey[i]: [float(x) for x in rng.dirichlet([1.0] * ft.nactions[i])]
                for i in range(len(ft.kind)) if ft.kind[i] == 2}
        fast = cf_regrets(ft, prof, set(range(n)))
        cert = game_cf(prof, set(range(n)))
        for p in range(n):
            for key in fast[p]:
                if any(abs(a - b) > 1e-12 for a, b in zip(fast[p][key], cert[p].get(key, []))):
                    parity = False

    # -- (b) g1 immediate/future == certified oracle; (d) coupling present --
    sh = build_shared(game, 0, cfr_iters=1500)
    g1_ok = True
    for fam in default_learn_families()[:3]:
        L = LearnLifted(sh, rule=fam.rule, eta=fam.eta)
        init = init_regret_for(L, fam)
        opp = L.state_policy(init)
        br_id, _ = L.br_now(init)
        h = L.pol(br_id)
        succ = L.step(init, br_id, "dense", None)
        if (abs(L.block_ev(br_id, init) - E.on_policy_value(game, {**h, **opp}, 0)) > 1e-12
                or abs(L.br_now(succ)[1]
                       - E.best_response_value(game, {**L.state_policy(succ), **h}, 0)) > 1e-12):
            g1_ok = False
    L = LearnLifted(sh, "rm")
    fam = [f for f in default_learn_families() if f.label == "rm:aggro|nit"][0]
    s0 = init_regret_for(L, fam)
    hid = L.intern({k: sh.menu["aggro"][k] for k in sh.keys_by_seat[0]})
    s1 = L.step(s0, hid, "dense", None)
    coupling = max(abs(a - b) for k in L.opp_keys
                   for a, b in zip(L.state_policy(L.step(s1, hid, "dense", None))[k],
                                   L.state_policy(L.step(s1, hid, "star", s0))[k])) > 1e-6

    # -- (c) RM self-play reaches the certified NashConv regime --
    regret, ssum = {}, {}

    def strat_of(key, na):
        return link_policy("rm", regret.get(key, [0.0] * na), 1.0)

    def acc(node, reach, profile):
        k = ft.kind[node]
        if k == 0:
            return
        if k == 1:
            for c, p in ft.cchildren[node]:
                acc(c, reach, profile)
            return
        pl, key, na, ch = ft.dplayer[node], ft.dkey[node], ft.nactions[node], ft.dchildren[node]
        s = profile.get(key) or [1.0 / na] * na
        d = ssum.setdefault(key, [0.0] * na)
        for a in range(na):
            d[a] += reach[pl] * s[a]
            cr = list(reach); cr[pl] *= s[a]
            acc(ch[a], cr, profile)

    for _ in range(2500):
        profile = {key: strat_of(key, len(r)) for key, r in regret.items()}
        acc(ft.root, [1.0] * n, profile)
        inc = cf_regrets(ft, profile, set(range(n)))
        for p in range(n):
            for key, r in inc[p].items():
                dd = regret.setdefault(key, [0.0] * len(r))
                for a in range(len(r)):
                    dd[a] += r[a]
    avg = {key: [x / sum(s) for x in s] if sum(s) > 0 else [1.0 / len(s)] * len(s)
           for key, s in ssum.items()}
    nashconv = sum(ft.best_response_value(avg, p) - ft.on_policy_value(avg, p) for p in range(n))

    # -- (e) run the fast decider + §28 controls + invariants --
    cfg = BaitConfig(n_pure=30, n_mixed=30, coord=True, coord_budget=250, seed=7)
    rep = run_learn_decider(game_name="3p-kuhn", m=4, depths=(1, 2, 3), delta=0.02,
                            cfr_iters=1500, cfg=cfg, max_baits=6, do_ceiling=True, ctl_horizon=4)
    ctl = {k: v for k, v in rep.gates.items() if k.startswith("ctl:")}
    invariants = all(f.a5 >= f.a25 - 1e-7 and f.a5_minus_a4 >= -1e-7
                     and f.ctl_traj_ok and f.ctl_a5_selfconsistent for f in rep.families)
    verdict_ok = rep.verdict in ("FILTER-GO", "FILTER-FLAT")

    ok = (parity and g1_ok and coupling and nashconv < 1e-2
          and all(ctl.values()) and invariants and verdict_ok)
    detail = (f"cf_regrets==game-oracle: {parity}; g1==certified: {g1_ok}; "
              f"coupling(star!=dense): {coupling}; RM self-play NashConv={nashconv:.2e}; "
              f"controls {sum(ctl.values())}/{len(ctl)} pass; A5>=A4>=A2.5 & self-consistent: {invariants}; "
              f"Kuhn verdict={rep.verdict} (A5-A4 held mean={rep.detail['mean A5-A4 (held)']:+.4f}, "
              f"max={rep.detail['max A5-A4 (held)']:+.4f})")
    return Result("rmapss step2b no-regret-learner decider", ok, detail)


def check_stepQ_gates() -> Result:
    """R-MAPSS §38 — the H-Q FRAGILITY→EV PREDICTOR. Re-derives the correctness machinery
    on the FAST 3p-Kuhn filter (deterministic CFR+ eq, so no jitter): (a) THE anti-circularity
    guard — the cheap, BR-free metrics make ZERO Hero best-response calls (so no metric can
    launder the exact BR it is meant to predict); (b) the §28 invariants (Y_cap ≤ Y_ceil;
    Y_ceil ~ eq-residual vs the equilibrium anchors; the α=1 exploiter within safety; the
    perfect-predictor pipeline sanity ρ≈1); (c) the finding reproduces — the cheap
    `mistake_mass` clears ρ*=0.70 AND beats the `dist_eq` baseline by ≥ Δρ=0.15, sign-stable.
    The DECISIVE Leduc run is a one-off (scripts/rmapss_stepQ.py), recorded in HANDOFF/RMAPSS
    §38 — not re-run every audit."""
    sys.path.insert(0, str(ROOT / "src"))
    import parciver.rmapss.stepQ as Q
    from parciver.games.kuhn_multi import KuhnPokerN
    from parciver.rmapss.stepQ import (
        run_stepQ, build_pop_q, collect_infosets, observe_marginal, opp_keys_by_seat,
        solve_eq, CHEAP_METRICS, BASELINE_METRIC,
    )

    game = KuhnPokerN(3)
    eq = solve_eq(game, 4000)                       # deterministic CFR+ eq (fast on Kuhn)
    sk = collect_infosets(game)
    he = {k: eq[k] for k in sk[0]}
    okf = [k for _, ks in opp_keys_by_seat(sk).items() for k in ks]
    _, em, _ = observe_marginal(game, he, {k: eq[k] for k in okf}, sk)
    ctx = {"hero_eq": he, "hero_keys": sk[0], "eq_marg": em}
    pop = build_pop_q(game, eq, sk)

    # -- (a) circularity guard: cheap metrics make ZERO Hero best-response calls --
    calls = {"n": 0}
    ov, os = Q.best_response_value, Q.best_response_strategy
    Q.best_response_value = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), ov(*a, **k))[1]
    Q.best_response_strategy = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), os(*a, **k))[1]
    try:
        for name, fn in CHEAP_METRICS.items():
            for o in pop[:6]:
                fn(game, o.profile, sk, ctx)
    finally:
        Q.best_response_value, Q.best_response_strategy = ov, os
    br_free = (calls["n"] == 0)

    # -- (b)/(c) full pipeline gates + the reproduced finding --
    rep = run_stepQ(game, "3p-kuhn", cfr_iters=4000, eq=eq, pert_k=6)
    invariants = all(rep.gates[k] for k in rep.gates if not k.startswith("H-Q GO"))
    base = abs(rep.metric_rho[BASELINE_METRIC].rho_held)
    mm = rep.metric_rho["mistake_mass"]
    import math
    mm_signstable = (math.copysign(1, mm.rho_fit) == math.copysign(1, mm.rho_held))
    mm_wins = (abs(mm.rho_held) >= 0.70 and abs(mm.rho_held) - base >= 0.15 and mm_signstable)

    ok = br_free and invariants and mm_wins
    detail = (f"cheap metrics Hero-BR-free (0 BR calls): {br_free}; "
              f"§28 invariants {sum(rep.gates[k] for k in rep.gates if not k.startswith('H-Q GO'))}"
              f"/{sum(1 for k in rep.gates if not k.startswith('H-Q GO'))} pass; "
              f"perfect-pred rho={rep.control_rho_perfect_held:.2f}; "
              f"mistake_mass rho_held={mm.rho_held:+.3f} vs baseline dist_eq {rep.metric_rho[BASELINE_METRIC].rho_held:+.3f} "
              f"(beats by {abs(mm.rho_held)-base:+.3f}); Kuhn verdict={rep.verdict}")
    return Result("rmapss stepQ fragility->EV predictor", ok, detail)


def check_cfr_gates() -> Result:
    """Re-derive the CFR headline numbers, independent of the test thresholds."""
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.eval.exploitability import exploitability, expected_value
    from parciver.games.kuhn import GAME_VALUE, KuhnPoker
    from parciver.games.leduc import LeducHoldem
    from parciver.solvers.cfr import DCFR, CFRSolver

    kuhn, leduc = KuhnPoker(), LeducHoldem()
    kuhn_val = expected_value(kuhn, e.solve_kuhn(5000, "dcfr"))
    leduc_expl = exploitability(leduc, e.solve_leduc(1000, "dcfr"))

    # Rust vs Python solver, matched iterations -> same iterate.
    n = 200
    py = CFRSolver(game=leduc, variant=DCFR, alternating=False)
    py.run(n)
    parity = abs(
        exploitability(leduc, py.average_strategy())
        - exploitability(leduc, e.solve_leduc(n, "dcfr"))
    )

    ok = (
        abs(kuhn_val - GAME_VALUE) < 2e-3
        and leduc_expl < 4e-2
        and parity < 1e-6
    )
    detail = (
        f"kuhn value={kuhn_val:+.5f} (known {GAME_VALUE:+.5f}); "
        f"leduc DCFR expl={leduc_expl:.4e} (<4e-2); "
        f"rust-vs-python parity={parity:.2e} (<1e-6)"
    )
    return Result("CFR correctness gates", ok, detail)


def check_mccfr_gates() -> Result:
    """Re-derive the external-sampling MCCFR claims from ground truth.

    The sampled hot loop must (1) solve Kuhn to its known value, (2) drive Leduc
    exploitability below the same 4e-2 bar the full solver meets AND visibly
    converge (a large budget beats a small one), (3) be seed-reproducible, and (4)
    reach the exact 288/12-infoset partition of the full-traversal engine. All
    scored by the independent Python oracle, not by MCCFR's own internals.
    """
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.eval.exploitability import exploitability, expected_value
    from parciver.games.kuhn import GAME_VALUE, KuhnPoker
    from parciver.games.leduc import LeducHoldem

    kuhn, leduc = KuhnPoker(), LeducHoldem()

    kuhn_strat = e.solve_kuhn_mccfr(200_000, "linear", 0)
    kuhn_val = expected_value(kuhn, kuhn_strat)
    kuhn_expl = exploitability(kuhn, kuhn_strat)

    coarse = exploitability(leduc, e.solve_leduc_mccfr(20_000, "linear", 0))
    fine_strat = e.solve_leduc_mccfr(500_000, "linear", 0)
    leduc_expl = exploitability(leduc, fine_strat)
    leduc_val = expected_value(leduc, fine_strat)
    converges = leduc_expl < 0.5 * coarse

    # Determinism: a fixed seed reproduces the average strategy bit-for-bit.
    a = e.solve_leduc_mccfr(50_000, "linear", 123)
    b = e.solve_leduc_mccfr(50_000, "linear", 123)
    deterministic = a.keys() == b.keys() and all(a[k] == b[k] for k in a)

    # Infoset partition parity with the full-traversal engine.
    full = e.solve_leduc(200, "dcfr")
    parity = set(fine_strat.keys()) == set(full.keys()) and all(
        len(fine_strat[k]) == len(v) for k, v in full.items()
    )

    ok = (
        abs(kuhn_val - GAME_VALUE) < 5e-3
        and kuhn_expl < 8e-3
        and leduc_expl < 4e-2
        and converges
        and -0.15 < leduc_val < 0.0
        and deterministic
        and parity
    )
    detail = (
        f"kuhn value={kuhn_val:+.5f} expl={kuhn_expl:.2e}; "
        f"leduc expl {coarse:.2e}@20k->{leduc_expl:.2e}@500k (converges={converges}); "
        f"deterministic={deterministic}; partition parity={parity}"
    )
    return Result("MCCFR (external sampling) gates", ok, detail)


def check_evaluator() -> Result:
    """Sanity-rank hands via the exposed API (exhaustive proof is the cargo test)."""
    import parciver_engine as e

    def h(*cs):
        return [e.parse_card(c) for c in cs]

    royal = e.eval7(h("Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"))
    boat = e.eval7(h("Ah", "Ac", "Ad", "Kh", "Kc", "2d", "3s"))
    wheel = e.eval5(h("Ac", "2d", "3h", "4s", "5c"))
    six_high = e.eval5(h("2c", "3d", "4h", "5s", "6c"))
    ok = (
        e.hand_category(royal) == 8
        and e.hand_category(boat) == 6
        and royal > boat
        and e.hand_category(wheel) == 4
        and wheel < six_high
    )
    detail = (
        f"royal(cat={e.hand_category(royal)}) > boat(cat={e.hand_category(boat)}): "
        f"{royal > boat}; wheel<six_high: {wheel < six_high}"
    )
    return Result("hand evaluator ordering", ok, detail)


def check_abstraction_gates() -> Result:
    """Re-derive the Phase 3a card-abstraction claims from scratch in Python.

    Independent of the Rust unit tests: recompute the suit-isomorphism class
    counts by iterating every holding/flop through the exposed API, and check the
    equity engine on exact invariants plus the textbook AA anchor.
    """
    import itertools

    import parciver_engine as e

    # 1326 two-card holdings collapse to exactly 169 classes, with the correct
    # multiplicities (pairs 6, suited 4, offsuit 12).
    from collections import Counter

    classes = Counter(
        e.preflop_index(a, b) for a, b in itertools.combinations(range(52), 2)
    )
    mult_ok = all(
        classes[i] == (6 if i < 13 else 4 if i < 91 else 12) for i in range(169)
    )
    n_pf = len(classes)

    # C(52,3) flops collapse to exactly 1755 suit-isomorphic classes.
    flops = {
        tuple(e.canonical_cards(list(f)))
        for f in itertools.combinations(range(52), 3)
    }
    n_flop = len(flops)

    # Equity: hero+villain complementary on a turn (exact), MC↔exact on a flop,
    # and AA vs a random hand ~85.2% preflop.
    pc = e.parse_card
    turn = [pc(c) for c in ("2h", "7h", "Ts", "Jd")]
    a = e.equity_vs_hand([pc("Ah"), pc("Kh")], [pc("Qs"), pc("Qd")], turn)
    b = e.equity_vs_hand([pc("Qs"), pc("Qd")], [pc("Ah"), pc("Kh")], turn)
    compl = abs(a + b - 1.0)

    flop = [pc(c) for c in ("Ad", "7h", "2s")]
    exact = e.equity_vs_random([pc("Ac"), pc("Ks")], flop)
    mc = e.equity_vs_random_mc([pc("Ac"), pc("Ks")], flop, 400_000, 0xC0FFEE)
    mc_gap = abs(exact - mc)

    aa = e.equity_vs_random_mc([pc("Ac"), pc("Ad")], [], 300_000, 12345)

    ok = (
        n_pf == 169
        and mult_ok
        and n_flop == 1755
        and compl < 1e-12
        and mc_gap < 5e-3
        and abs(aa - 0.852) < 0.01
    )
    detail = (
        f"preflop classes={n_pf} (mult_ok={mult_ok}); flop classes={n_flop}; "
        f"complementarity gap={compl:.1e}; mc-vs-exact={mc_gap:.1e}; "
        f"AA vs random={aa:.4f} (~0.852)"
    )
    return Result("card abstraction + equity gates", ok, detail)


def check_clustering_gates() -> Result:
    """Re-derive the Phase 3b GPU-clustering claims independently.

    EMD is checked against a from-scratch numpy CDF-L1 reference; k-means is
    checked for exact reproducibility (same seed -> identical labels, the flaw a
    CUDA atomic-add centroid update would introduce) and for recovering
    well-separated synthetic clusters.
    """
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from parciver.clustering import emd_distance_matrix, kmeans_emd

    # EMD vs an independent numpy reference (L1 between CDFs).
    rng = np.random.default_rng(7)
    P = rng.random((11, 16)).astype(np.float32)
    Q = rng.random((6, 16)).astype(np.float32)
    Pn = P / P.sum(1, keepdims=True)
    Qn = Q / Q.sum(1, keepdims=True)
    got = (
        emd_distance_matrix(torch.tensor(Pn), torch.tensor(Qn)).cpu().numpy()
    )
    ref = np.array(
        [[np.abs(np.cumsum(p) - np.cumsum(q)).sum() for q in Qn] for p in Pn]
    )
    emd_gap = float(np.abs(got - ref).max())

    # Synthetic clusters: three separated gaussian-bump histograms + noise.
    bins = np.arange(24)
    archs = [np.exp(-0.5 * ((bins - c) / 1.5) ** 2) for c in (3, 12, 20)]
    X, truth = [], []
    gnr = np.random.default_rng(1)
    for g, a in enumerate(archs):
        a = a / a.sum()
        for _ in range(40):
            noisy = a + 0.01 * gnr.random(24)
            X.append(noisy / noisy.sum())
            truth.append(g)
    X = np.array(X, dtype=np.float32)
    truth = np.array(truth)
    res = kmeans_emd(X, k=3, seed=0, n_init=3)
    labels = res.labels.cpu().numpy()
    pure = all(len(set(labels[truth == g].tolist())) == 1 for g in (0, 1, 2))
    distinct = len({labels[truth == g][0] for g in (0, 1, 2)}) == 3

    # Reproducibility: identical labels for a fixed seed (not just same partition).
    a = kmeans_emd(X, k=3, seed=13).labels.cpu().numpy()
    b = kmeans_emd(X, k=3, seed=13).labels.cpu().numpy()
    deterministic = bool(np.array_equal(a, b))

    ok = emd_gap < 1e-5 and pure and distinct and deterministic
    detail = (
        f"emd-vs-numpy max gap={emd_gap:.1e}; synthetic recovery pure={pure and distinct}; "
        f"seed-reproducible labels={deterministic}; device={res.centroids.device.type}"
    )
    return Result("GPU clustering (k-means/EMD) gates", ok, detail)


def check_feature_gates() -> Result:
    """Re-derive the Phase 3b feature-extraction claims independently.

    The fast evaluator (eval7_fast) driving these is certified bit-identical to
    the reference eval7 in Rust (exhaustively under --full). Here we check the
    Python-facing feature surface: histograms are valid distributions that order
    made hands above trash, the rayon batch equals the single-situation function,
    the river feature agrees with the general equity engine, and the role-aware
    canonical key separates hole from board.
    """
    import parciver_engine as e

    def h(*cs):
        return [e.parse_card(c) for c in cs]

    def mean(hist):
        n = len(hist)
        return sum(((b + 0.5) / n) * m for b, m in enumerate(hist))

    bins = 50
    made = e.flop_equity_histogram(h("Ac", "Ad"), h("Ah", "7d", "2c"), bins)
    trash = e.flop_equity_histogram(h("7c", "2d"), h("Ah", "Kd", "Qs"), bins)
    flop_ok = (
        len(made) == bins
        and abs(sum(made) - 1.0) < 1e-5
        and mean(made) > 0.85
        and mean(trash) < 0.35
    )

    nuts = e.turn_equity_histogram(h("Ah", "Kh"), h("Qh", "Jh", "2h", "3d"), bins)
    weak = e.turn_equity_histogram(h("7c", "2d"), h("Ah", "Kd", "Qs", "Jc"), bins)
    turn_ok = abs(sum(nuts) - 1.0) < 1e-5 and mean(nuts) > mean(weak) and mean(nuts) > 0.9

    # Batch must reproduce the single-situation function exactly.
    holes = h("Ac", "Ad", "7c", "2d")
    flops = h("Ah", "7d", "2s", "Ah", "Kd", "Qs")
    batch = e.flop_histograms_batch(holes, flops, bins)
    r0 = e.flop_equity_histogram(holes[0:2], flops[0:3], bins)
    r1 = e.flop_equity_histogram(holes[2:4], flops[3:6], bins)
    batch_ok = batch[0:bins] == r0 and batch[bins:2 * bins] == r1

    # River feature (uses eval7_fast) must agree with the general equity engine
    # on the same complete board.
    rb = h("Ah", "7d", "2c", "Ts", "Js")
    river_ok = abs(
        e.river_equity(h("Ac", "Ad"), rb) - e.equity_vs_random(h("Ac", "Ad"), rb)
    ) < 1e-12

    # Role-aware canonical key: distinguishes hole from board, permutation-invariant.
    canon_ok = (
        e.canonical_situation(h("Ac", "Kc"), h("Qc", "Jd", "2h"))
        != e.canonical_situation(h("Qc", "Jd"), h("Ac", "Kc", "2h"))
        and e.canonical_situation(h("As", "Ks"), h("Qs", "Js", "2c"))
        == e.canonical_situation(h("Ah", "Kh"), h("Qh", "Jh", "2d"))
    )

    ok = flop_ok and turn_ok and batch_ok and river_ok and canon_ok
    detail = (
        f"flop made/trash means={mean(made):.3f}/{mean(trash):.3f} (ok={flop_ok}); "
        f"turn nut mean={mean(nuts):.3f} (ok={turn_ok}); batch==single={batch_ok}; "
        f"river==equity_engine={river_ok}; canonical role-aware={canon_ok}"
    )
    return Result("feature extraction (Phase 3b) gates", ok, detail)


def check_indexer_gates() -> Result:
    """Re-derive the dense hand-isomorphism indexer claims (Phase 3b) from the API.

    The exhaustive flop set-agreement is the Rust cargo test; here we confirm the
    literature sizes, that index/unindex round-trips, that the index is
    suit-permutation invariant, and that its equivalence classes agree with the
    independent role-aware ``canonical_situation`` — over random samples of every
    street, including all 123,156,254 river classes' index space.
    """
    import random

    import parciver_engine as e

    streets = [(0, 1_286_792, 5), (1, 13_960_050, 6), (2, 123_156_254, 7)]
    sizes_ok = all(e.canonical_index_size(s) == exp for s, exp, _ in streets)
    cards_ok = all(e.street_card_count(s) == c for s, _, c in streets)

    rng = random.Random(0)
    rt_ok = inv_ok = canon_ok = True
    for s, _, _ in streets:
        n = e.canonical_index_size(s)
        for _ in range(400):
            i = rng.randrange(n)
            hand = list(e.canonical_unindex_batch(s, i, 1))
            if e.canonical_index(s, hand) != i:
                rt_ok = False
            perm = [0, 1, 2, 3]
            rng.shuffle(perm)
            ph = [(x >> 2) * 4 + perm[x & 3] for x in hand]
            if e.canonical_index(s, ph) != i:
                inv_ok = False
            if e.canonical_situation(ph[:2], ph[2:]) != e.canonical_situation(hand[:2], hand[2:]):
                canon_ok = False

    ok = sizes_ok and cards_ok and rt_ok and inv_ok and canon_ok
    detail = (
        f"sizes {tuple(e.canonical_index_size(s) for s, _, _ in streets)} "
        f"(match={sizes_ok}); round-trip={rt_ok}; suit-invariant={inv_ok}; "
        f"agrees w/ canonical_situation={canon_ok}"
    )
    return Result("hand-isomorphism indexer gates", ok, detail)


def check_bucketing_gates() -> Result:
    """Re-derive the end-to-end bucketing claims (Phase 3b) on small slices.

    Runs the real driver (enumerate/index -> extract -> cluster -> persist ->
    gate) for the river (1-D k-means, fast) and a tiny flop slice (histogram +
    EMD), and confirms the bucket-count + monotone-equity gate passes, river
    buckets are disjoint equity intervals, and the play-time ``bucket_of`` lookup
    round-trips through the persisted table.
    """
    import shutil
    import tempfile

    import numpy as np

    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver import bucketing as B

    d = Path(tempfile.mkdtemp(prefix="parciver_buckets_"))
    try:
        gr = B.build_street("river", k=8, out_dir=d, chunk=20000, limit=40000, log=False)
        gf = B.build_street("flop", k=6, bins=20, out_dir=d, chunk=40, limit=120, log=False)
        B._TABLE_CACHE.clear()
        table = np.load(B._paths(d, "flop")["buckets"])
        hand = list(e.canonical_unindex_batch(0, 50, 1))
        lookup_ok = B.bucket_of("flop", hand, out_dir=d) == int(table[50])
        ok = bool(gr.ok and gf.ok and gr.river_intervals_disjoint and lookup_ok)
        detail = (
            f"river gate ok={gr.ok} (intervals disjoint={gr.river_intervals_disjoint}); "
            f"flop gate ok={gf.ok} ({gf.buckets_nonempty}/6 non-empty, monotone={gf.monotone}); "
            f"bucket_of round-trip={lookup_ok}"
        )
    finally:
        B._TABLE_CACHE.clear()
        shutil.rmtree(d, ignore_errors=True)
    return Result("bucketing pipeline (Phase 3b) gates", ok, detail)


def check_nlhe_bucket_key() -> Result:
    """Re-derive the Phase-3c B2 claim: the NLHE infoset card key is the Phase-3b
    bucket abstraction, loaded in Rust.

    The gate is *parity*: the Rust ``Abstraction.bucket_of`` (which the NLHE
    ``infoset_key`` calls) must equal the Python reference
    ``parciver.bucketing.bucket_of`` for sampled situations on every postflop
    street, and the preflop class must equal ``preflop_index``. Both sides use the
    same ``canonical_index``, so agreement certifies the Rust ``.npy`` reader
    loaded the persisted tables faithfully and mapped each street correctly.
    Requires the baked bucket tables (gitignored) to be present on disk.
    """
    import json
    import random

    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver import bucketing as B

    streets = ("flop", "turn", "river")
    d = B.default_out_dir()
    missing = [s for s in streets if not B._paths(d, s)["buckets"].exists()]
    if missing:
        return Result(
            "NLHE bucket infoset key (B2) gates",
            False,
            f"baked bucket tables absent for {missing}; run scripts/build_buckets.py",
        )

    abs_ = e.Abstraction(str(d))
    rng = random.Random(20260911)

    # Postflop parity, per street, against the Python reference lookup.
    postflop_ok = True
    checked = 0
    max_bucket = {}
    for street in streets:
        sid = B.STREET_IDS[street]
        n = e.canonical_index_size(sid)
        k = json.loads(B._paths(d, street)["meta"].read_text())["k"]
        mx = -1
        for _ in range(200):
            i = rng.randrange(n)
            hand = list(e.canonical_unindex_batch(sid, i, 1))
            rust = abs_.bucket_of(hand[:2], hand[2:])
            if rust != B.bucket_of(street, hand) or not (0 <= rust < k):
                postflop_ok = False
            mx = max(mx, rust)
            checked += 1
        max_bucket[street] = mx

    # Preflop = the 169 exact classes (preflop_index), and suit-invariant.
    preflop_ok = True
    for _ in range(300):
        a, b = rng.sample(range(52), 2)
        if abs_.bucket_of([a, b], []) != e.preflop_index(a, b):
            preflop_ok = False

    # Suit-permutation invariance on a postflop sample.
    perm = [2, 3, 0, 1]
    suit_ok = True
    for _ in range(100):
        hand = list(e.canonical_unindex_batch(2, rng.randrange(e.canonical_index_size(2)), 1))
        ph = [(c >> 2) * 4 + perm[c & 3] for c in hand]
        if abs_.bucket_of(hand[:2], hand[2:]) != abs_.bucket_of(ph[:2], ph[2:]):
            suit_ok = False

    ok = postflop_ok and preflop_ok and suit_ok
    detail = (
        f"rust==python bucket_of on {checked} postflop situations={postflop_ok}; "
        f"preflop==preflop_index={preflop_ok}; suit-invariant={suit_ok}; "
        f"max bucket flop/turn/river={max_bucket['flop']}/{max_bucket['turn']}/{max_bucket['river']}"
    )
    return Result("NLHE bucket infoset key (B2) gates", ok, detail)


def check_nlhe_small_gate() -> Result:
    """Re-derive the Phase-3c B3 claim: the NLHE solvers are cross-validated on a
    game small enough for full-traversal CFR + exact best-response exploitability.

    Two independent legs:
      1. META-GATE — the Rust exact best response (`exploit.rs`) must equal the
         independent Python exploitability oracle on Kuhn AND Leduc (~1e-9). This
         is what licenses trusting the Rust BR on the small NLHE game, where there
         is no Python game to check against.
      2. CROSS-VALIDATION — on `NlheConfig::small` (a tiny flop-only NLHE), both
         full-traversal CFR and external-sampling MCCFR must drive exact
         exploitability DOWN monotonically and land in the same low neighborhood.
    """
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.eval.exploitability import exploitability as py_expl
    from parciver.games.kuhn import KuhnPoker
    from parciver.games.leduc import LeducHoldem

    # 1. Meta-gate: Rust best response == Python oracle, exactly.
    ks = e.solve_kuhn(3000, "dcfr")
    ls = e.solve_leduc(500, "dcfr")
    kuhn_gap = abs(e.kuhn_exploitability(ks) - py_expl(KuhnPoker(), ks))
    leduc_gap = abs(e.leduc_exploitability(ls) - py_expl(LeducHoldem(), ls))
    meta_ok = kuhn_gap < 1e-6 and leduc_gap < 1e-6

    # 2. Small NLHE cross-validation.
    n_infosets = e.nlhe_small_infoset_count()
    cfr_lo = e.nlhe_small_exploitability(e.solve_nlhe_small(80, "dcfr"))
    cfr_hi = e.nlhe_small_exploitability(e.solve_nlhe_small(300, "dcfr"))
    mc_lo = e.nlhe_small_exploitability(e.solve_nlhe_small_mccfr(50_000, "linear", 0))
    mc_hi = e.nlhe_small_exploitability(e.solve_nlhe_small_mccfr(500_000, "linear", 0))

    cfr_converges = cfr_hi < cfr_lo and cfr_hi < 0.008
    mccfr_converges = mc_hi < mc_lo and mc_hi < 0.006
    agree = abs(cfr_hi - mc_hi) < 0.004  # both solvers reach the same neighborhood

    ok = meta_ok and cfr_converges and mccfr_converges and agree
    detail = (
        f"meta BR==oracle (kuhn {kuhn_gap:.1e}, leduc {leduc_gap:.1e})={meta_ok}; "
        f"small NLHE infosets={n_infosets}; "
        f"CFR expl {cfr_lo:.4f}@80->{cfr_hi:.4f}@300; "
        f"MCCFR expl {mc_lo:.4f}@50k->{mc_hi:.4f}@500k; agree={agree}"
    )
    return Result("NLHE small cross-validation (B3) gates", ok, detail)


def check_telemetry_gate() -> Result:
    """Re-derive the Phase-3c telemetry-contract claim from ground truth.

    The training dashboard decouples the trainer from the viewer through an
    append-only versioned event log, and B4 emits to it, so the contract is
    load-bearing. Independently of the pytest thresholds, this: emits one of every
    event type through ``RunWriter``, reads them back through ``read_events``, and
    asserts an exact round-trip with monotone ``seq`` and the right schema version;
    confirms the validator REJECTS a malformed event (so garbage cannot reach the
    log); and confirms a live ``ResourceSampler`` reading satisfies the contract's
    ``resources`` shape.
    """
    import tempfile
    from pathlib import Path as _Path

    sys.path.insert(0, str(ROOT / "src"))
    from parciver import telemetry as tel
    from parciver.resources import ResourceSampler

    d = _Path(tempfile.mkdtemp(prefix="parciver_tel_"))
    try:
        cfg = {
            "hand_names": [f"H{i}" for i in range(tel.NUM_PREFLOP_HANDS)],
            "abstraction": {"preflop": 169, "flop": 10000, "turn": 10000, "river": 1950},
        }
        emitted = []
        with tel.RunWriter(d) as w:
            emitted.append(w.start(variant="linear-mccfr", game="nlhe-small", config=cfg,
                                   target_iters=1_000_000))
            emitted.append(w.metrics(iteration=1, it_per_s=1234.5, exploitability=0.0115,
                                     infosets=456, resources=ResourceSampler(threads=16).sample()))
            grid = {"actions": ["fold", "call", "raise"],
                    "cells": [tel.normalize_mix([1, 1, 1]) for _ in range(tel.NUM_PREFLOP_HANDS)]}
            emitted.append(w.checkpoint(iteration=1, range_grid=grid,
                                        action_freq={"P": {"f": 0.1, "c": 0.6, "r": 0.3}},
                                        infosets_by_street={"P": 100}))
            emitted.append(w.milestone("checkpoint_saved", "snapshot @1", iteration=1))
            emitted.append(w.end(iteration=1_000_000, reason="completed",
                                 summary={"final_exploitability": 0.001}))

        read = tel.read_events(d)
        types_ok = [e["type"] for e in read] == list(tel.EVENT_TYPES)
        round_trip = read == emitted
        seq_ok = [e["seq"] for e in read] == list(range(len(read)))
        versioned = all(e["v"] == tel.SCHEMA_VERSION for e in read)

        # The validator must reject garbage (here: a bad run_end reason).
        rejected = False
        try:
            tel.validate_event({"v": tel.SCHEMA_VERSION, "type": "run_end", "seq": 0,
                                "ts": 1.0, "iteration": 1, "reason": "exploded"})
        except tel.SchemaError:
            rejected = True

        # A real sampler reading must satisfy the resources shape.
        res = ResourceSampler(threads=16).sample()
        tel.validate_event({"v": tel.SCHEMA_VERSION, "type": "metrics", "seq": 0,
                            "ts": 1.0, "iteration": 1, "resources": res})

        ok = types_ok and round_trip and seq_ok and versioned and rejected
        detail = (
            f"5 event types round-trip={round_trip}; seq monotone={seq_ok}; "
            f"v=={tel.SCHEMA_VERSION} on all={versioned}; malformed rejected={rejected}; "
            f"live resources gpu={res.get('gpu_pct')} ram_total={res.get('ram_total_gb')}"
        )
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    return Result("telemetry contract (dashboard) gates", ok, detail)


def check_blueprint_gate() -> Result:
    """Re-derive the Phase-3c B4 claim: the blueprint solver runs on the REAL 100bb
    game and produces a sane, converging strategy.

    Exact exploitability is intractable here, so this checks what a short real solve
    must show: infosets accumulate, the variant-normalized average regret is positive
    and TRENDS DOWN with more iterations (the convergence proxy), and the preflop
    range grid is a valid per-hand distribution that is sane poker — trash folds more
    than premiums. Requires the baked bucket tables (gitignored).
    """
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver import telemetry as tel

    d = ROOT / "data" / "abstraction"
    if not all((d / f"{s}_buckets.npy").exists() for s in ("flop", "turn", "river")):
        return Result("blueprint solver (B4) gates", False,
                      "baked bucket tables absent; run scripts/build_buckets.py")

    names = tel.preflop_hand_names()
    bp = e.Blueprint(str(d), "linear", 0)
    bp.run(12000)
    r1 = bp.avg_regret()
    grid = bp.preflop_range_grid()
    bp.run(12000)
    r2 = bp.avg_regret()

    grid_valid = len(grid) == 169 and all(
        abs(sum(row) - 1.0) < 1e-6 or sum(row) == 0.0 for row in grid
    )
    fold = lambda h: grid[names.index(h)][0]  # noqa: E731
    sane = fold("72o") > fold("AA") and fold("32o") > fold("KK")
    converging = r1 > 0 and r2 > 0 and r2 < r1
    coverage = bp.num_infosets() > 10000

    ok = grid_valid and sane and converging and coverage
    detail = (
        f"throughput OK ({bp.num_infosets():,} infosets @ 24k iters); "
        f"avg_regret {r1:.3e}->{r2:.3e} (down={converging}); "
        f"range sane (72o fold {fold('72o'):.2f} > AA fold {fold('AA'):.2f})={sane}; "
        f"grid valid={grid_valid}"
    )
    return Result("blueprint solver (B4) gates", ok, detail)


def check_parallel_mccfr_gates() -> Result:
    """Re-derive the Phase-3c Step C claim: the PARALLEL MCCFR path converges to the
    same equilibrium as the serial one, scored by the independent Python oracle.

    Not a bit-identity check (concurrent float additions reorder) — a convergence
    check: with multiple worker threads, Kuhn reaches its known value and Leduc
    exploitability falls below the same 4e-2 bar the serial/full solvers meet AND a
    larger budget beats a smaller one. This is what makes parallelization safe: if
    thread staleness were corrupting the update, exploitability would not fall.
    """
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.eval.exploitability import exploitability, expected_value
    from parciver.games.kuhn import GAME_VALUE, KuhnPoker
    from parciver.games.leduc import LeducHoldem

    kuhn, leduc = KuhnPoker(), LeducHoldem()
    threads = 8

    kuhn_strat = e.solve_kuhn_mccfr(300_000, "linear", 0, threads)
    kuhn_val = expected_value(kuhn, kuhn_strat)
    kuhn_expl = exploitability(kuhn, kuhn_strat)

    coarse = exploitability(leduc, e.solve_leduc_mccfr(20_000, "linear", 0, threads))
    fine = e.solve_leduc_mccfr(500_000, "linear", 0, threads)
    leduc_expl = exploitability(leduc, fine)
    converges = leduc_expl < 0.5 * coarse

    # Infoset partition is thread-order-independent: same set as the full solver.
    full = e.solve_leduc(200, "dcfr")
    parity = set(fine.keys()) == set(full.keys())

    # Parallel is higher-variance than serial, so the value/expl bars are a touch
    # looser than check_mccfr_gates — still tight enough to prove equilibrium.
    ok = (
        abs(kuhn_val - GAME_VALUE) < 1e-2
        and kuhn_expl < 1.5e-2
        and leduc_expl < 4e-2
        and converges
        and parity
    )
    detail = (
        f"[{threads} threads] kuhn value={kuhn_val:+.5f} expl={kuhn_expl:.2e}; "
        f"leduc expl {coarse:.2e}@20k->{leduc_expl:.2e}@500k (converges={converges}); "
        f"partition parity={parity}"
    )
    return Result("parallel MCCFR equilibrium agreement", ok, detail)


def check_checkpoint_gate() -> Result:
    """Re-derive the resumable-checkpoint claim: a full-state checkpoint saved and
    reloaded resumes the solve EXACTLY — same iteration count, same infoset coverage,
    the same average-strategy range grid (accumulators are stored losslessly as f64) —
    and the reloaded solver keeps solving. This is what lets a multi-day 100bb solve
    survive a restart. Requires the baked bucket tables (gitignored).
    """
    import os
    import tempfile

    import parciver_engine as e

    d = ROOT / "data" / "abstraction"
    if not all((d / f"{s}_buckets.npy").exists() for s in ("flop", "turn", "river")):
        return Result("resumable checkpoint (B4) gates", False, "baked bucket tables absent")

    bp = e.Blueprint(str(d), "linear", 7, 4)
    bp.run(15_000)
    iters0, ninf0, grid0 = bp.iterations(), bp.num_infosets(), bp.preflop_range_grid()

    tmp = Path(tempfile.gettempdir()) / f"parciver_audit_ckpt_{os.getpid()}.bin"
    bp.save_checkpoint(str(tmp))
    bp2 = e.Blueprint.resume(str(d), str(tmp), 4)
    try:
        os.remove(tmp)
    except OSError:
        pass

    iters_match = bp2.iterations() == iters0
    ninf_match = bp2.num_infosets() == ninf0
    grid1 = bp2.preflop_range_grid()
    grid_match = all(
        abs(a - b) < 1e-9 for r0, r1 in zip(grid0, grid1) for a, b in zip(r0, r1)
    )
    bp2.run(10_000)
    advanced = bp2.iterations() == iters0 + 10_000

    ok = iters_match and ninf_match and grid_match and advanced
    detail = (
        f"resume iters match={iters_match} ({iters0:,}); infosets match={ninf_match} "
        f"({ninf0:,}); range grid exact={grid_match}; resumed solve advanced={advanced}"
    )
    return Result("resumable checkpoint (B4) gates", ok, detail)


def check_blueprint_analysis_gate() -> Result:
    """Re-derive the blueprint-analysis panels claim (action-frequency, bet-sizing,
    head-to-head): Monte-Carlo self-play yields valid per-street action / bet-size
    distributions, and the blueprint BEATS a calling-station baseline (bb/100 > 0 —
    a station that never folds is maximally exploitable). Skipped-as-informational if
    the installed engine predates BlueprintAnalyzer (a fresh build cannot overwrite a
    running trainer's file-locked .pyd until it stops); the Rust `analysis` unit test
    always covers the core logic. Requires the baked bucket tables.
    """
    import os
    import tempfile

    import parciver_engine as e

    if not hasattr(e, "BlueprintAnalyzer"):
        return Result(
            "blueprint analysis (panels) gates", True,
            "BlueprintAnalyzer not in installed engine yet (run `maturin develop` once "
            "the trainer stops); Rust analysis unit test covers the logic",
            informational=True,
        )

    d = ROOT / "data" / "abstraction"
    if not all((d / f"{s}_buckets.npy").exists() for s in ("flop", "turn", "river")):
        return Result("blueprint analysis (panels) gates", False, "baked bucket tables absent")

    bp = e.Blueprint(str(d), "linear", 0, 4)
    bp.run(60_000)
    model = Path(tempfile.gettempdir()) / f"parciver_audit_bp_{os.getpid()}.bin"
    bp.save(str(model))
    az = e.BlueprintAnalyzer(str(d))
    res = az.analyze(str(model), 8_000, 8_000, 0)
    try:
        os.remove(model)
    except OSError:
        pass

    af = res["action_freq"]
    streets_ok = "P" in af and all(abs(sum(m.values()) - 1.0) < 1e-6 for m in af.values())
    bet_ok = all(abs(sum(m.values()) - 1.0) < 1e-6 for m in res["bet_size_freq"].values())
    hh = res["hh_bb_per_100"]
    beats_station = hh > 0.0

    ok = streets_ok and bet_ok and beats_station
    detail = (
        f"action mixes valid (streets={sorted(af)})={streets_ok}; "
        f"bet mixes valid={bet_ok}; blueprint vs calling station={hh:+.1f} bb/100 (>0)={beats_station}"
    )
    return Result("blueprint analysis (panels) gates", ok, detail)


def check_dashboard_gate() -> Result:
    """Re-derive the Phase-3c dashboard-producer claims from ground truth.

    The viewer only reads the telemetry log, so what must be guarded is that BOTH
    producers emit schema-valid telemetry the viewer can render: the full-scale
    SIMULATOR (`replay_demo`) must produce a 169-hand range grid and a completed
    run, and the REAL small-game driver (`train_small`) must drive exact
    exploitability DOWN (its whole purpose) and finish. Also confirms the viewer's
    static SPA assets are present and the server module imports.
    """
    import contextlib
    import importlib.util
    import io
    import tempfile
    from pathlib import Path as _Path

    sys.path.insert(0, str(ROOT / "src"))
    from parciver import telemetry as tel

    def load_script(name: str):
        spec = importlib.util.spec_from_file_location(f"_audit_{name}", ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Viewer assets + server import.
    from parciver.dashboard import server as dash
    assets_ok = all((dash.STATIC_DIR / f).exists() for f in ("index.html", "app.js", "style.css"))

    d = _Path(tempfile.mkdtemp(prefix="parciver_dash_"))
    try:
        # Simulator: a tiny full-scale run must carry the 169-cell range grid.
        # (Silence the producers' progress prints so the audit output stays clean.)
        replay = load_script("replay_demo")
        with contextlib.redirect_stdout(io.StringIO()):
            replay.run(d / "demo", target_iters=1_000_000, duration=0.05, tick=0.05,
                       checkpoint_every=1, fast=False, seed=3)
        sim = tel.read_events(d / "demo")
        grids = [e for e in sim if e["type"] == "checkpoint" and e.get("range_grid")]
        sim_ok = (sim[0]["type"] == "run_start" and sim[-1]["type"] == "run_end"
                  and sim[-1]["reason"] == "completed"
                  and bool(grids) and len(grids[-1]["range_grid"]["cells"]) == tel.NUM_PREFLOP_HANDS)

        # Real driver: exact exploitability must fall as the MCCFR budget grows.
        train = load_script("train_small")
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(d / "small", variant="linear", seed=0, iters_max=8000, points=4,
                      min_step_seconds=0.0)
        real = tel.read_events(d / "small")
        expl = [e["exploitability"] for e in real
                if e["type"] == "metrics" and e.get("exploitability") is not None]
        real_ok = (len(expl) >= 3 and expl[-1] < expl[0]
                   and real[-1]["type"] == "run_end" and real[-1]["reason"] == "completed")

        ok = assets_ok and sim_ok and real_ok
        detail = (
            f"viewer assets present={assets_ok}; simulator grid+complete={sim_ok} "
            f"({len(grids)} grids); real MCCFR exploitability {expl[0]:.4f}->{expl[-1]:.4f} "
            f"(down={real_ok})"
        )
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    return Result("dashboard producers + viewer gates", ok, detail)


def check_rating_gate() -> Result:
    """The rating of record (RATING.md) must be re-derivable now: every receipt is replayed under the
    current engine + rating code if it is stale (`rating.py verify`), then `rating.py audit` must pass
    (no INVALID receipt, an accepted calibration, a VERIFIED headline for the current bot, the
    reference at exactly 1500). Pool of record: runs/rating/pool-v2 of the main checkout (RATING.md Amendment 2;
    pool-v1 is history)."""
    pool = next((p for p in (ROOT / "runs" / "rating" / "pool-v2",
                             ROOT.parents[2] / "runs" / "rating" / "pool-v2") if (p / "pool.json").exists()), None)
    if pool is None:
        return Result("rating of record (RATING.md)", False, "no pool-v2 found (runs/rating/pool-v2)")
    script = str(ROOT / "scripts" / "rating.py")
    env = {"PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    rc_v, out_v, err_v = _run([PYEXE, script, "verify", "--stale-only", "--dir", str(pool)], ROOT, env, timeout=7200)
    rc_a, out_a, err_a = _run([PYEXE, script, "audit", "--dir", str(pool)], ROOT, env, timeout=600)
    ok = rc_v == 0 and rc_a == 0 and "AUDIT OK" in out_a and "MISMATCH" not in out_v
    tail = (out_a.strip().splitlines() or [err_a.strip()])[-1]
    return Result("rating of record (RATING.md)", ok,
                  f"{pool}: verify exit {rc_v}, audit exit {rc_a}: {tail[:160]}")


def check_speedup() -> Result:
    """Informational: report the real Leduc DCFR speedup, so summaries can't inflate it."""
    import parciver_engine as e

    sys.path.insert(0, str(ROOT / "src"))
    from parciver.games.leduc import LeducHoldem
    from parciver.solvers.cfr import DCFR, CFRSolver

    leduc = LeducHoldem()
    n = 200
    py = []
    ru = []
    for _ in range(3):
        t = time.perf_counter()
        CFRSolver(game=leduc, variant=DCFR, alternating=False).run(n)
        py.append(time.perf_counter() - t)
        t = time.perf_counter()
        e.solve_leduc(n, "dcfr")
        ru.append(time.perf_counter() - t)
    pt, rt = min(py), min(ru)
    detail = (
        f"leduc DCFR: python {n/pt:.0f} it/s, rust {n/rt:.0f} it/s, "
        f"speedup {pt/rt:.1f}x"
    )
    return Result("speedup (informational)", True, detail, informational=True)


CHECKS_NO_ARG = [
    check_environment,
    check_files_exist,
    check_extension_api,
    check_python_suite,
    check_infoset_parity,
    check_kuhn_n_gates,
    check_rmapss_step1_gates,
    check_rmapss_step3_gates,
    check_rmapss_step32_gates,
    check_rmapss_step33_parity,
    check_rmapss_step34_controls,
    check_leduc_n_gates,
    check_rmapss_step2_gates,
    check_rmapss_step2a_gates,
    check_rmapss_step2b_gates,
    check_stepQ_gates,
    check_cfr_gates,
    check_mccfr_gates,
    check_evaluator,
    check_abstraction_gates,
    check_clustering_gates,
    check_feature_gates,
    check_indexer_gates,
    check_bucketing_gates,
    check_nlhe_bucket_key,
    check_nlhe_small_gate,
    check_telemetry_gate,
    check_dashboard_gate,
    check_blueprint_gate,
    check_parallel_mccfr_gates,
    check_checkpoint_gate,
    check_blueprint_analysis_gate,
    check_rating_gate,
    check_speedup,
]


def main() -> int:
    # Audit output is often piped by automation.  Force line buffering so the
    # currently running gate and its elapsed time remain visible instead of
    # making a healthy long check look hung.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="parciver phase-end audit")
    ap.add_argument("--rebuild", action="store_true", help="maturin develop first")
    ap.add_argument("--full", action="store_true", help="run exhaustive 7-card cert")
    args = ap.parse_args()

    print("=" * 70)
    print("parciver phase-end audit -- re-deriving every claim from ground truth")
    print("=" * 70)

    if args.rebuild:
        print("\n[rebuild] maturin develop --release -m engine/Cargo.toml ...")
        rc, out, err = _run(
            [PYEXE, "-m", "maturin", "develop", "--release", "-m", "engine/Cargo.toml"],
            ROOT,
            timeout=900,
        )
        print("  " + ("ok" if rc == 0 else "FAILED:\n" + err))
        if rc != 0:
            print("\nAUDIT ABORTED: extension failed to build.")
            return 1

    results: list[Result] = []
    for fn in CHECKS_NO_ARG:
        started = time.perf_counter()
        print(f"[audit] START {fn.__name__}", flush=True)
        try:
            result = fn()
        except Exception as exc:  # a broken check is a failed check
            result = Result(fn.__name__, False, f"EXCEPTION: {exc!r}")
        results.append(result)
        tag = "INFO" if result.informational else ("PASS" if result.passed else "FAIL")
        print(f"[audit] {tag}  {fn.__name__}  {time.perf_counter() - started:.2f}s",
              flush=True)
    # rust tests take an arg
    started = time.perf_counter()
    print("[audit] START check_rust_tests", flush=True)
    try:
        result = check_rust_tests(args.full)
    except Exception as exc:
        result = Result("rust unit tests", False, f"EXCEPTION: {exc!r}")
    results.append(result)
    print(f"[audit] {'PASS' if result.passed else 'FAIL'}  check_rust_tests  "
          f"{time.perf_counter() - started:.2f}s", flush=True)

    print()
    width = max(len(r.name) for r in results)
    hard_fail = False
    for r in results:
        tag = "INFO" if r.informational else ("PASS" if r.passed else "FAIL")
        print(f"  [{tag}] {r.name.ljust(width)}  {r.detail}")
        if not r.passed and not r.informational:
            hard_fail = True

    print("\n" + "=" * 70)
    if hard_fail:
        print("RESULT: FAIL -- the phase is NOT done. Do not start a fresh session.")
    else:
        print("RESULT: PASS -- every claim verified against ground truth.")
        if args.rebuild and args.full:
            sys.path.insert(0, str(ROOT / "src"))
            from parciver.rmapss import step34
            print("S3.2 AUDIT RECEIPT: not refreshed (terminal protocol).")
            print("S3.3 AUDIT RECEIPT: not refreshed (terminal protocol).")
            receipt34 = step34.write_audit_receipt(
                small_game_control=_S32_SMALL_GAME_CONTROL,
                parity_control=_S34_PARITY_CONTROL,
                performance_control=_S34_PERFORMANCE_CONTROL,
            )
            print(f"S3.4 AUDIT RECEIPT: {receipt34}")
        else:
            print("S3.2 AUDIT RECEIPT: not refreshed (terminal protocol).")
            print("S3.3 AUDIT RECEIPT: not refreshed (terminal protocol).")
            print("S3.4 AUDIT RECEIPT: not written (requires --rebuild --full).")
    print("=" * 70)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
