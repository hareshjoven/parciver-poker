#!/usr/bin/env python
"""Locked R-MAPSS S3.2 full-100bb runner (`s3.2-full-v1`, RMAPSS §43).

The command exposes only operational controls (run directory, worker count,
dashboard port). Every research setting comes from ``rmapss.step32``. It refuses
to load the blueprint unless the frozen artifacts match and refuses to execute
product outcomes unless a green rebuild+full phase audit covers the current
source hashes.

This experiment reports sampled lower-bound attacks as ``ABR-proxy
non-regression``. It never calls them exploitability bounds or a safety proof.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import parciver_engine as pe  # noqa: E402

from parciver import telemetry as T  # noqa: E402
from parciver.resources import ResourceSampler  # noqa: E402
from parciver.rmapss import step32  # noqa: E402


class ProtocolFailure(RuntimeError):
    """A locked gate failed; do not continue reading additional outcomes."""


def _spawn_dashboard(run_dir: Path, port: int):
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"),
           "--run", str(run_dir), "--port", str(port), "--no-browser"]
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(cmd, cwd=str(ROOT), **kwargs)


def _peak_memory_gb() -> float:
    """Peak process working set, including Rust allocations (Windows stdlib FFI)."""
    if sys.platform != "win32":
        try:
            import resource
            scale = 1024.0 if sys.platform == "darwin" else 1.0
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / (1 << 30)
        except Exception:  # noqa: BLE001
            return 0.0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    ok = get_memory(
        ctypes.c_void_p(-1),
        ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize / (1 << 30)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_resume_state(path: Path, artifact_hashes: dict[str, str]) -> dict:
    source = path / "s3.2-results.json" if path.is_dir() else path
    if not source.is_file():
        raise ValueError(f"missing S3.2 resume state: {source}")
    state = json.loads(source.read_text(encoding="utf-8"))
    if (state.get("protocol") != step32.PROTOCOL
            or state.get("metadata") != step32.frozen_metadata()
            or state.get("artifact_hashes") != artifact_hashes):
        raise ValueError("S3.2 resume state does not match the locked protocol/artifacts")
    calibration = state.get("calibration_rows")
    cases = state.get("case_rows")
    if not isinstance(calibration, list) or not isinstance(cases, list):
        raise ValueError("S3.2 resume state is incomplete")
    calibration_ids = [(row.get("seed"), row.get("label")) for row in calibration]
    case_ids = [(row.get("seed"), row.get("label")) for row in cases]
    if len(calibration_ids) != len(set(calibration_ids)) or len(case_ids) != len(set(case_ids)):
        raise ValueError("S3.2 resume state contains duplicate records")
    state.pop("report", None)
    state.pop("failure", None)
    return state


def _compact_measurement(measurement: dict, *, threshold: float = 0.0,
                         family_size: int = 1) -> dict:
    estimate = step32.estimate_payload(
        measurement["block_gains_bb_per_100"], threshold=threshold,
        family_size=family_size
    )
    return {
        "mean": measurement["gain_bb_per_100"],
        "lower": estimate["lower"],
        "upper": estimate["upper"],
        "standard_error": estimate["standard_error"],
        "difference_stddev_bb_per_hand": measurement["difference_stddev_bb_per_hand"],
    }


def _measure(candidate, anchor, opponent, seed: int) -> dict:
    return step32.paired_measure(candidate, anchor, opponent, seed=seed)


def _known_response(anchor, opponent, *, seed: int, case_index: int,
                    threads: int) -> tuple[object, dict]:
    policy = step32.build_two_seat_response(
        opponent, anchor, iterations=step32.RESPONSE_ITERS, seed=seed,
        threads=threads, stage="known_response", case_index=case_index
    )
    measurement = _measure(
        policy, anchor, opponent,
        step32.stream_seed(seed, "known_evaluation", case_index=case_index),
    )
    return policy, measurement


def _attack_measurement(candidate, anchor, *, seed: int, case_index: int,
                        alpha_index: int, replicate: int, threads: int,
                        final: bool) -> dict:
    response_stage = "final_attack_response" if final else "proxy_search_response"
    evaluation_stage = "final_attack_evaluation" if final else "proxy_search_evaluation"
    attacker = step32.build_two_seat_response(
        candidate, anchor, iterations=step32.ATTACK_ITERS, seed=seed,
        threads=threads, replicate=replicate, stage=response_stage,
        case_index=case_index, alpha_index=alpha_index,
    )
    measurement = _measure(
        candidate, anchor, attacker,
        step32.stream_seed(
            seed, evaluation_stage, case_index=case_index,
            alpha_index=alpha_index, replicate=replicate,
        ),
    )
    return {"replicate": replicate, "measurement": measurement}


def _search_alpha_proxy(anchor, response, *, seed: int, case_index: int,
                        proxy_budget: float, threads: int,
                        on_alpha=None) -> tuple[float, list[dict]]:
    """Descending locked-grid search with candidate-specific sampled attackers."""
    search: list[dict] = []
    for alpha_index, alpha in enumerate(step32.ALPHA_GRID):
        candidate = anchor.blend(response, alpha)
        attacks = [
            _attack_measurement(
                candidate, anchor, seed=seed, case_index=case_index,
                alpha_index=alpha_index, replicate=replicate,
                threads=threads, final=False,
            )
            for replicate in range(step32.ATTACK_REPLICATES)
        ]
        estimates = [
            step32.block_estimate(
                [-value for value in attack["measurement"]["block_gains_bb_per_100"]],
                threshold=proxy_budget,
                family_size=step32.PROXY_SEARCH_FAMILY_SIZE,
            )
            for attack in attacks
        ]
        passed = all(estimate.upper <= proxy_budget for estimate in estimates)
        row = {
            "alpha": alpha,
            "passes_proxy": passed,
            "attacks": attacks,
            "loss_intervals": [
                step32.estimate_payload(
                    [-value for value in attack["measurement"]["block_gains_bb_per_100"]],
                    threshold=proxy_budget,
                    family_size=step32.PROXY_SEARCH_FAMILY_SIZE,
                )
                for attack in attacks
            ],
        }
        search.append(row)
        if on_alpha is not None:
            on_alpha(row)
        if passed:
            return alpha, search
    raise AssertionError("alpha=0 is identical to anchor and must pass the paired proxy")


def _case_row(*, seed: int, spec: step32.OpponentSpec, case_index: int,
              anchor, opponent, reference, known_measurement: dict,
              thresholds: dict, threads: int, on_alpha=None) -> dict:
    started = time.perf_counter()
    train, validation = anchor.observe_split(
        opponent, step32.OBS_FIT_HANDS, step32.OBS_VALIDATION_HANDS,
        step32.stream_seed(seed, "observation", case_index=case_index),
    )
    model = train.fit_model(reference, step32.MIN_PUBLIC_COUNT, step32.PRIOR_STRENGTH)
    model_score = dict(validation.score_model(model))
    public_only = bool(
        train.public_keys_valid() and validation.public_keys_valid()
        and reference.public_keys_valid()
    )
    if spec in step32.HELD_SPEC and not (
        model_score["coverage"] >= 0.95
        and model_score["model_log_loss"] < model_score["reference_log_loss"]
        and public_only
    ):
        raise ProtocolFailure(
            f"observation gate failed at seed={seed}, opponent={spec.label}: {model_score}"
        )

    response = step32.build_two_seat_response(
        model, anchor, iterations=step32.RESPONSE_ITERS, seed=seed,
        threads=threads, stage="model_response", case_index=case_index,
    )
    alpha_conf = min(1.0, float(model_score["deviation_l1"]) / step32.KAPPA)
    alpha_proxy, search = _search_alpha_proxy(
        anchor, response, seed=seed, case_index=case_index,
        proxy_budget=thresholds["proxy_budget"], threads=threads,
        on_alpha=on_alpha,
    )
    alpha = min(alpha_conf, alpha_proxy)
    candidate = anchor.blend(response, alpha)
    product = _measure(
        candidate, anchor, opponent,
        step32.stream_seed(seed, "product_evaluation", case_index=case_index),
    )

    selected = next(row for row in search if row["alpha"] == alpha_proxy)
    if alpha == alpha_proxy:
        final_attacks = selected["attacks"]
    else:
        final_attacks = [
            _attack_measurement(
                candidate, anchor, seed=seed, case_index=case_index,
                alpha_index=len(step32.ALPHA_GRID), replicate=replicate,
                threads=threads, final=True,
            )
            for replicate in range(step32.ATTACK_REPLICATES)
        ]

    return {
        "protocol": step32.PROTOCOL,
        "seed": seed,
        "label": spec.label,
        "family": spec.family,
        "strength": spec.strength,
        "split": spec.split,
        "klass": spec.klass,
        "model": model_score,
        "public_only_keys": public_only,
        "alpha_conf": alpha_conf,
        "alpha_proxy": alpha_proxy,
        "alpha": alpha,
        "product": product,
        "known_response": known_measurement,
        "proxy_search": search,
        "final_attacks": final_attacks,
        "compute_s": time.perf_counter() - started,
        "peak_memory_gb": _peak_memory_gb(),
    }


def _dashboard_row(row: dict, thresholds: dict) -> dict:
    product = _compact_measurement(
        row["product"], threshold=thresholds["delta_sub"] if row["klass"] == "suboptimal" else 0.0,
        family_size=(step32.PRODUCT_FAMILY_SIZE if row["split"] == "held"
                     and row["klass"] == "suboptimal" else step32.NULL_FAMILY_SIZE),
    )
    proxy_upper = max(
        step32.block_estimate(
            [-value for value in attack["measurement"]["block_gains_bb_per_100"]],
            threshold=thresholds["proxy_budget"],
            family_size=step32.PROXY_FINAL_FAMILY_SIZE,
        ).upper
        for attack in row["final_attacks"]
    )
    return {
        "seed": row["seed"], "label": row["label"], "split": row["split"],
        "klass": row["klass"], "alpha": row["alpha"],
        "alpha_conf": row["alpha_conf"], "alpha_proxy": row["alpha_proxy"],
        "gain": product["mean"], "lower": product["lower"], "upper": product["upper"],
        "delta_sub": thresholds["delta_sub"], "tol": thresholds["tol"],
        "proxy_budget": thresholds["proxy_budget"], "proxy_loss_upper": proxy_upper,
        "coverage": row["model"]["coverage"],
        "model_log_loss": row["model"]["model_log_loss"],
        "reference_log_loss": row["model"]["reference_log_loss"],
        "compute_s": row["compute_s"], "peak_memory_gb": row["peak_memory_gb"],
    }


def run(args) -> dict:
    if args.protocol != step32.PROTOCOL:
        raise ValueError(f"only locked protocol {step32.PROTOCOL!r} may run")
    receipt = step32.require_audit_receipt(args.audit_receipt)
    artifact_hashes = step32.verify_frozen_artifacts()

    prefix = "rmapss-step32-resume" if args.resume else "rmapss-step32"
    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id(prefix)))
    events = run_dir / "events.jsonl"
    if events.exists():
        raise ValueError(f"refusing to append a second experiment to {events}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "s3.2-results.json"

    dashboard = _spawn_dashboard(run_dir, args.port)
    url = f"http://127.0.0.1:{args.port}/static/experiment32.html"
    print(f"[dashboard] {url}  (watching {run_dir.name})", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    saved = pe.SavedBlueprint(str(step32.ABSTRACTION_DIR), str(step32.BLUEPRINT))
    if saved.num_infosets() != 13_469_060:
        raise ProtocolFailure("frozen anchor infoset count changed")
    anchor = saved.s32_policy()
    population = step32.build_population(anchor)
    sampler = ResourceSampler(threads=args.threads)
    state = (_load_resume_state(args.resume, artifact_hashes) if args.resume else {
        "protocol": step32.PROTOCOL,
        "metadata": step32.frozen_metadata(),
        "artifact_hashes": artifact_hashes,
        "small_game_control": receipt["small_game_control"],
        "calibration_rows": [],
        "case_rows": [],
    })
    state["small_game_control"] = receipt["small_game_control"]

    total_cases = len(step32.SEEDS) * len(step32.POPULATION_SPEC)
    with T.RunWriter(run_dir, fsync=True) as writer:
        writer.start(
            run_id=run_dir.name, variant="s3.2-full-100bb", game="nlhe_pluribus_100bb",
            config={
                "kind": "experiment", "experiment": "rmapss-s3.2",
                **step32.frozen_metadata(),
                "artifact_hashes": artifact_hashes,
                "audit_receipt": str(Path(args.audit_receipt)),
                "threads": args.threads,
            },
        )
        iteration = 0
        try:
            for seed in step32.SEEDS:
                reference = anchor.observe(
                    anchor, step32.REFERENCE_HANDS,
                    step32.stream_seed(seed, "reference"),
                )
                if not reference.public_keys_valid():
                    raise ProtocolFailure("anchor reference contains a non-public key")

                seed_calibration = [
                    row for row in state["calibration_rows"] if row.get("seed") == seed
                ]
                for case_index, spec in enumerate(step32.FIT_SPEC):
                    if any(row.get("label") == spec.label for row in seed_calibration):
                        continue
                    _policy, measurement = _known_response(
                        anchor, population[spec.label], seed=seed,
                        case_index=case_index, threads=args.threads,
                    )
                    calibration = {"seed": seed, "label": spec.label,
                                   "measurement": measurement}
                    state["calibration_rows"].append(calibration)
                    seed_calibration.append(calibration)
                    writer.milestone(
                        "s3.2_calibration", f"seed {seed}: calibrated {spec.label}",
                        iteration=iteration,
                        s3_2_calibration={"seed": seed, "label": spec.label,
                                          **_compact_measurement(measurement)},
                    )
                    _write_json_atomic(state_path, state)

                thresholds = step32.calibrate_thresholds(seed_calibration, seed)
                writer.checkpoint(iteration=iteration, step32_thresholds=thresholds)

                for case_index, spec in enumerate(step32.POPULATION_SPEC):
                    existing = next((
                        row for row in state["case_rows"]
                        if row.get("seed") == seed and row.get("label") == spec.label
                    ), None)
                    if existing is not None:
                        iteration += 1
                        writer.checkpoint(
                            iteration=iteration,
                            step32_row=_dashboard_row(existing, thresholds),
                            step32_progress={"completed": iteration, "total": total_cases},
                            resumed=True,
                        )
                        continue
                    opponent = population[spec.label]
                    if spec in step32.FIT_SPEC:
                        known = next(row["measurement"] for row in seed_calibration
                                     if row["label"] == spec.label)
                    else:
                        _policy, known = _known_response(
                            anchor, opponent, seed=seed, case_index=case_index,
                            threads=args.threads,
                        )

                    def on_alpha(alpha_row):
                        writer.milestone(
                            "s3.2_proxy_search",
                            f"seed {seed} {spec.label}: alpha={alpha_row['alpha']:.2f} "
                            f"{'pass' if alpha_row['passes_proxy'] else 'reject'}",
                            iteration=iteration,
                            s3_2_proxy={"seed": seed, "label": spec.label,
                                        "alpha": alpha_row["alpha"],
                                        "passed": alpha_row["passes_proxy"]},
                        )

                    row = _case_row(
                        seed=seed, spec=spec, case_index=case_index,
                        anchor=anchor, opponent=opponent, reference=reference,
                        known_measurement=known, thresholds=thresholds,
                        threads=args.threads, on_alpha=on_alpha,
                    )
                    state["case_rows"].append(row)
                    iteration += 1
                    writer.metrics(iteration=iteration, resources=sampler.sample())
                    writer.checkpoint(
                        iteration=iteration,
                        step32_row=_dashboard_row(row, thresholds),
                        step32_progress={"completed": iteration, "total": total_cases},
                    )
                    _write_json_atomic(state_path, state)

            report = step32.finalize_report(
                artifact_hashes=artifact_hashes,
                small_game_control=receipt["small_game_control"],
                calibration_rows=state["calibration_rows"],
                case_rows=state["case_rows"],
            )
            state["report"] = report
            _write_json_atomic(state_path, state)
            writer.checkpoint(iteration=iteration + 1, step32=report)
            # The telemetry contract intentionally has only lifecycle reasons
            # (completed/interrupted/error).  A protocol gate failure is an error
            # outcome, with the locked verdict carried in the summary payload.
            writer.end(iteration + 1, reason="completed" if report["all_pass"] else "error",
                       summary=report)
            return report
        except ProtocolFailure as exc:
            state["failure"] = str(exc)
            _write_json_atomic(state_path, state)
            # Preserve the protocol breach in the summary while emitting a
            # schema-valid terminal lifecycle event.
            writer.end(iteration, reason="error",
                       summary={"protocol": step32.PROTOCOL, "all_pass": False,
                                "verdict": "FAIL (S3.2 locked gate breach)",
                                "failure": str(exc)})
            raise
        finally:
            _ = dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="locked R-MAPSS S3.2 full-100bb run")
    parser.add_argument("--protocol", default=step32.PROTOCOL, choices=[step32.PROTOCOL])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path,
                        help="resume completed calibration/cases from a prior s3.2-results.json")
    parser.add_argument("--threads", type=int, default=1,
                        help="fixed-response worker threads (operational, not a research setting)")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--audit-receipt", type=Path, default=step32.AUDIT_RECEIPT,
                        help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    try:
        report = run(args)
    except (ProtocolFailure, ValueError) as exc:
        print(f"[rmapss-step32] FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    print(report["verdict"], flush=True)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
