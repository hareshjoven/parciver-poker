#!/usr/bin/env python
"""Locked S3.3 public-observation recovery runner (RMAPSS §46)."""
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
from parciver.rmapss import step32, step33  # noqa: E402


class ProtocolFailure(RuntimeError):
    """A locked gate failed; this protocol may not continue or resume."""


def _spawn_dashboard(run_dir: Path, port: int):
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"),
           "--run", str(run_dir), "--port", str(port), "--no-browser"]
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(cmd, cwd=str(ROOT), **kwargs)


def _peak_memory_gb() -> float:
    if sys.platform != "win32":
        try:
            import resource
            scale = 1024.0 if sys.platform == "darwin" else 1.0
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / (1 << 30)
        except Exception:  # noqa: BLE001
            return 0.0
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    if not get_memory(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize / (1 << 30)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _initial_state(artifact_hashes: dict[str, str], receipt: dict) -> dict:
    return {
        "protocol": step33.PROTOCOL,
        "metadata": step33.frozen_metadata(),
        "artifact_hashes": artifact_hashes,
        "small_game_control": receipt["small_game_control"],
        "parity_control": receipt["parity_control"],
        "observation_rows": [],
        "calibration_rows": [],
        "case_rows": [],
    }


def _load_resume_state(path: Path, artifact_hashes: dict[str, str]) -> dict:
    source = path / "s3.3-results.json" if path.is_dir() else path
    if not source.is_file():
        raise ValueError(f"missing S3.3 resume state: {source}")
    state = json.loads(source.read_text(encoding="utf-8"))
    if state.get("terminal_failure") or state.get("failure"):
        raise ValueError("a gate-failed S3.3 run is terminal and cannot resume")
    if (state.get("protocol") != step33.PROTOCOL
            or state.get("metadata") != step33.frozen_metadata()
            or state.get("artifact_hashes") != artifact_hashes):
        raise ValueError("S3.3 resume state does not match locked metadata/artifacts")
    for key in ("observation_rows", "calibration_rows", "case_rows"):
        if not isinstance(state.get(key), list):
            raise ValueError(f"S3.3 resume state lacks {key}")
        ids = [(row.get("seed"), row.get("label")) for row in state[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"S3.3 resume state has duplicate {key}")
    state.pop("report", None)
    return state


def _observe_blocks(anchor, opponent, *, seed: int, stage: str,
                    case_index: int, blocks: int, threads: int):
    seeds = step33.block_seeds(seed, stage, case_index=case_index, blocks=blocks)
    return anchor.observe_blocks(opponent, step33.OBS_BLOCK_SIZE, seeds, threads)


def _fit_model(anchor, opponent, reference, *, seed: int,
               case_index: int, threads: int):
    fit_blocks = _observe_blocks(
        anchor, opponent, seed=seed, stage="observation_fit",
        case_index=case_index, blocks=step33.FIT_BLOCKS, threads=threads)
    train = step33.merge_observations(fit_blocks)
    confirmation = _observe_blocks(
        anchor, opponent, seed=seed, stage="observation_confirmation",
        case_index=case_index, blocks=step33.CONFIRM_BLOCKS, threads=threads)
    model = step33.fit_hierarchical_model(train, reference)
    block_scores = step33.score_blocks(model, confirmation)
    summary = step33.summarize_model_blocks(block_scores)
    public_only = bool(train.public_keys_valid() and reference.public_keys_valid()
                       and all(block.public_keys_valid() for block in confirmation))
    return model, summary, block_scores, public_only


def _observation_row(*, seed: int, spec, case_index: int, anchor,
                     opponent, reference, threads: int) -> tuple[object, dict]:
    started = time.perf_counter()
    model, summary, block_scores, public_only = _fit_model(
        anchor, opponent, reference, seed=seed,
        case_index=case_index, threads=threads)
    return model, {
        "protocol": step33.PROTOCOL,
        "seed": seed,
        "label": spec.label,
        "family": spec.family,
        "strength": spec.strength,
        "split": spec.split,
        "model": summary,
        "block_scores": block_scores,
        "public_only_keys": public_only,
        "compute_s": time.perf_counter() - started,
        "peak_memory_gb": _peak_memory_gb(),
    }


def _measure(candidate, anchor, opponent, seed: int, threads: int) -> dict:
    return step33.paired_measure(candidate, anchor, opponent, seed=seed, threads=threads)


def _known_response(anchor, opponent, *, seed: int, case_index: int,
                    threads: int) -> tuple[object, dict]:
    policy = step33.build_two_seat_response(
        opponent, anchor, iterations=step33.RESPONSE_ITERS, seed=seed,
        threads=threads, stage="known_response", case_index=case_index)
    measurement = _measure(
        policy, anchor, opponent,
        step33.stream_seed(seed, "known_evaluation", case_index=case_index), threads)
    return policy, measurement


def _attack_measurement(candidate, anchor, *, seed: int, case_index: int,
                        alpha_index: int, replicate: int, threads: int,
                        final: bool) -> dict:
    response_stage = "final_attack_response" if final else "proxy_search_response"
    evaluation_stage = "final_attack_evaluation" if final else "proxy_search_evaluation"
    attacker = step33.build_two_seat_response(
        candidate, anchor, iterations=step33.ATTACK_ITERS, seed=seed,
        threads=threads, replicate=replicate, stage=response_stage,
        case_index=case_index, alpha_index=alpha_index)
    measurement = _measure(
        candidate, anchor, attacker,
        step33.stream_seed(seed, evaluation_stage, case_index=case_index,
                           alpha_index=alpha_index, replicate=replicate), threads)
    return {"replicate": replicate, "measurement": measurement}


def _search_alpha_proxy(anchor, response, *, seed: int, case_index: int,
                        proxy_budget: float, threads: int,
                        on_alpha=None) -> tuple[float, list[dict]]:
    search = []
    for alpha_index, alpha in enumerate(step33.ALPHA_GRID):
        candidate = anchor.blend(response, alpha)
        attacks = [_attack_measurement(
            candidate, anchor, seed=seed, case_index=case_index,
            alpha_index=alpha_index, replicate=replicate,
            threads=threads, final=False)
            for replicate in range(step33.ATTACK_REPLICATES)]
        estimates = [step32.block_estimate(
            [-v for v in attack["measurement"]["block_gains_bb_per_100"]],
            threshold=proxy_budget, family_size=step33.PROXY_SEARCH_FAMILY_SIZE)
            for attack in attacks]
        passed = all(est.upper <= proxy_budget for est in estimates)
        row = {"alpha": alpha, "passes_proxy": passed, "attacks": attacks,
               "loss_intervals": [step32.estimate_payload(
                   [-v for v in attack["measurement"]["block_gains_bb_per_100"]],
                   threshold=proxy_budget,
                   family_size=step33.PROXY_SEARCH_FAMILY_SIZE)
                   for attack in attacks]}
        search.append(row)
        if on_alpha:
            on_alpha(row)
        if passed:
            return alpha, search
    raise AssertionError("alpha=0 must pass the paired proxy")


def _case_row(*, seed: int, spec, case_index: int, anchor, opponent,
              reference, known_measurement: dict, thresholds: dict,
              threads: int, locked_observation: dict | None,
              on_alpha=None) -> dict:
    started = time.perf_counter()
    model, model_score, _blocks, public_only = _fit_model(
        anchor, opponent, reference, seed=seed,
        case_index=case_index, threads=threads)
    if locked_observation is not None:
        if (model_score != locked_observation["model"]
                or public_only != locked_observation["public_only_keys"]):
            raise ProtocolFailure(
                f"deterministic observation replay drift at seed={seed}, {spec.label}")
    response = step33.build_two_seat_response(
        model, anchor, iterations=step33.RESPONSE_ITERS, seed=seed,
        threads=threads, stage="model_response", case_index=case_index)
    alpha_conf = min(1.0, float(model_score["deviation_l1"]) / step33.KAPPA)
    alpha_proxy, search = _search_alpha_proxy(
        anchor, response, seed=seed, case_index=case_index,
        proxy_budget=thresholds["proxy_budget"], threads=threads,
        on_alpha=on_alpha)
    alpha = min(alpha_conf, alpha_proxy)
    candidate = anchor.blend(response, alpha)
    product = _measure(
        candidate, anchor, opponent,
        step33.stream_seed(seed, "product_evaluation", case_index=case_index), threads)
    selected = next(row for row in search if row["alpha"] == alpha_proxy)
    final_attacks = selected["attacks"] if alpha == alpha_proxy else [
        _attack_measurement(
            candidate, anchor, seed=seed, case_index=case_index,
            alpha_index=len(step33.ALPHA_GRID), replicate=replicate,
            threads=threads, final=True)
        for replicate in range(step33.ATTACK_REPLICATES)]
    return {
        "protocol": step33.PROTOCOL, "seed": seed, "label": spec.label,
        "family": spec.family, "strength": spec.strength,
        "split": spec.split, "klass": spec.klass,
        "model": model_score, "public_only_keys": public_only,
        "alpha_conf": alpha_conf, "alpha_proxy": alpha_proxy, "alpha": alpha,
        "product": product, "known_response": known_measurement,
        "proxy_search": search, "final_attacks": final_attacks,
        "compute_s": time.perf_counter() - started,
        "peak_memory_gb": _peak_memory_gb(),
    }


def _dashboard_observation(row: dict) -> dict:
    model = row["model"]
    return {"seed": row["seed"], "label": row["label"],
            "effective_coverage": model["effective_coverage"],
            "legacy_coverage": model["legacy_coverage"],
            "model_nll": model["model_nll"],
            "reference_nll": model["reference_nll"],
            "legacy_nll": model["legacy_nll"], "ece": model["ece"],
            "compute_s": row["compute_s"],
            "peak_memory_gb": row["peak_memory_gb"]}


def run(args) -> dict:
    if args.protocol != step33.PROTOCOL:
        raise ValueError(f"only locked protocol {step33.PROTOCOL!r} may run")
    if not 1 <= args.threads <= step33.MAX_WORKERS:
        raise ValueError(f"threads must be within 1..={step33.MAX_WORKERS}")
    receipt = step33.require_audit_receipt(args.audit_receipt)
    artifact_hashes = step33.verify_frozen_artifacts()
    prefix = "rmapss-step33-resume" if args.resume else "rmapss-step33"
    run_dir = Path(args.run_dir or (ROOT / "runs" / T.make_run_id(prefix)))
    events = run_dir / "events.jsonl"
    if events.exists():
        raise ValueError(f"refusing to append a second experiment to {events}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "s3.3-results.json"
    state = (_load_resume_state(args.resume, artifact_hashes) if args.resume
             else _initial_state(artifact_hashes, receipt))
    dashboard = _spawn_dashboard(run_dir, args.port)
    url = f"http://127.0.0.1:{args.port}/static/experiment33.html"
    print(f"[dashboard] {url}  (watching {run_dir.name})", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    saved = pe.SavedBlueprint(str(step33.ABSTRACTION_DIR), str(step33.BLUEPRINT))
    if saved.num_infosets() != 13_469_060:
        raise ProtocolFailure("frozen anchor infoset count changed")
    anchor = saved.s32_policy()
    population = step33.build_population(anchor)
    sampler = ResourceSampler(threads=args.threads)
    total_observation = step33.OBSERVATION_FAMILY_SIZE
    total_cases = len(step33.SEEDS) * len(step33.POPULATION_SPEC)

    with T.RunWriter(run_dir, fsync=True) as writer:
        writer.start(run_id=run_dir.name, variant="s3.3-public-recovery",
                     game="nlhe_pluribus_100bb",
                     config={"kind": "experiment", "experiment": "rmapss-s3.3",
                             **step33.frozen_metadata(),
                             "artifact_hashes": artifact_hashes,
                             "audit_receipt": str(Path(args.audit_receipt)),
                             "threads": args.threads})
        iteration = 0
        try:
            # Complete and score the full untouched observation family first.
            for seed in step33.SEEDS:
                reference_blocks = _observe_blocks(
                    anchor, anchor, seed=seed, stage="reference", case_index=0,
                    blocks=step33.REFERENCE_BLOCKS, threads=args.threads)
                reference = step33.merge_observations(reference_blocks)
                if not reference.public_keys_valid():
                    raise ProtocolFailure("anchor reference contains a non-public key")
                for spec in step33.HELD_SPEC:
                    case_index = step33.POPULATION_SPEC.index(spec)
                    existing = next((row for row in state["observation_rows"]
                                     if row["seed"] == seed and row["label"] == spec.label), None)
                    if existing is None:
                        _model, row = _observation_row(
                            seed=seed, spec=spec, case_index=case_index,
                            anchor=anchor, opponent=population[spec.label],
                            reference=reference, threads=args.threads)
                        state["observation_rows"].append(row)
                        _write_json_atomic(state_path, state)
                    else:
                        row = existing
                    iteration += 1
                    writer.metrics(iteration=iteration, resources=sampler.sample())
                    writer.checkpoint(
                        iteration=iteration,
                        step33_observation_row=_dashboard_observation(row),
                        step33_progress={"stage": "observation", "completed": iteration,
                                         "total": total_observation})

            observation = step33.observation_report(state["observation_rows"])
            state["observation_report"] = observation
            _write_json_atomic(state_path, state)
            writer.checkpoint(iteration=iteration, step33_observation=observation)
            if not observation["all_pass"]:
                raise ProtocolFailure(observation["verdict"])

            # Observation PASS unlocks the unchanged downstream H1 pipeline.
            observation_by_id = {(row["seed"], row["label"]): row
                                 for row in state["observation_rows"]}
            case_done = len(state["case_rows"])
            for seed in step33.SEEDS:
                reference = step33.merge_observations(_observe_blocks(
                    anchor, anchor, seed=seed, stage="reference", case_index=0,
                    blocks=step33.REFERENCE_BLOCKS, threads=args.threads))
                seed_calibration = [row for row in state["calibration_rows"]
                                    if row["seed"] == seed]
                for spec in step33.FIT_SPEC:
                    case_index = step33.POPULATION_SPEC.index(spec)
                    if any(row["label"] == spec.label for row in seed_calibration):
                        continue
                    _policy, measurement = _known_response(
                        anchor, population[spec.label], seed=seed,
                        case_index=case_index, threads=args.threads)
                    calibration = {"seed": seed, "label": spec.label,
                                   "measurement": measurement}
                    state["calibration_rows"].append(calibration)
                    seed_calibration.append(calibration)
                    _write_json_atomic(state_path, state)
                thresholds = step33.calibrate_thresholds(seed_calibration, seed)
                writer.checkpoint(iteration=iteration, step33_thresholds=thresholds)
                for spec in step33.POPULATION_SPEC:
                    case_index = step33.POPULATION_SPEC.index(spec)
                    existing = next((row for row in state["case_rows"]
                                     if row["seed"] == seed and row["label"] == spec.label), None)
                    if existing is not None:
                        continue
                    opponent = population[spec.label]
                    if spec in step33.FIT_SPEC:
                        known = next(row["measurement"] for row in seed_calibration
                                     if row["label"] == spec.label)
                    else:
                        _policy, known = _known_response(
                            anchor, opponent, seed=seed, case_index=case_index,
                            threads=args.threads)

                    def on_alpha(alpha_row):
                        writer.milestone(
                            "s3.3_proxy_search",
                            f"seed {seed} {spec.label}: alpha={alpha_row['alpha']:.2f} "
                            f"{'pass' if alpha_row['passes_proxy'] else 'reject'}",
                            iteration=iteration,
                            s3_3_proxy={"seed": seed, "label": spec.label,
                                        "alpha": alpha_row["alpha"],
                                        "passed": alpha_row["passes_proxy"]})

                    row = _case_row(
                        seed=seed, spec=spec, case_index=case_index,
                        anchor=anchor, opponent=opponent, reference=reference,
                        known_measurement=known, thresholds=thresholds,
                        threads=args.threads,
                        locked_observation=observation_by_id.get((seed, spec.label)),
                        on_alpha=on_alpha)
                    state["case_rows"].append(row)
                    case_done += 1
                    iteration += 1
                    _write_json_atomic(state_path, state)
                    writer.metrics(iteration=iteration, resources=sampler.sample())
                    writer.checkpoint(
                        iteration=iteration,
                        step33_case={"seed": seed, "label": spec.label,
                                     "alpha": row["alpha"],
                                     "gain": row["product"]["gain_bb_per_100"],
                                     "effective_coverage": row["model"]["effective_coverage"],
                                     "compute_s": row["compute_s"],
                                     "peak_memory_gb": row["peak_memory_gb"]},
                        step33_progress={"stage": "downstream", "completed": case_done,
                                         "total": total_cases})

            report = step33.finalize_report(
                artifact_hashes=artifact_hashes,
                small_game_control=receipt["small_game_control"],
                parity_control=receipt["parity_control"], observation=observation,
                calibration_rows=state["calibration_rows"],
                case_rows=state["case_rows"])
            state["report"] = report
            _write_json_atomic(state_path, state)
            writer.checkpoint(iteration=iteration + 1, step33=report)
            writer.end(iteration + 1,
                       reason="completed" if report["all_pass"] else "error",
                       summary=report)
            return report
        except ProtocolFailure as exc:
            state["failure"] = str(exc)
            state["terminal_failure"] = True
            _write_json_atomic(state_path, state)
            writer.end(iteration, reason="error",
                       summary={"protocol": step33.PROTOCOL, "all_pass": False,
                                "verdict": "FAIL (S3.3 locked gate breach)",
                                "failure": str(exc)})
            raise
        finally:
            _ = dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="locked R-MAPSS S3.3 run")
    parser.add_argument("--protocol", default=step33.PROTOCOL,
                        choices=[step33.PROTOCOL])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path,
                        help="resume an interrupted, non-failed S3.3 logical run")
    parser.add_argument("--threads", type=int, default=16,
                        help="operational worker count, locked to 1..16")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--audit-receipt", type=Path, default=step33.AUDIT_RECEIPT,
                        help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (ProtocolFailure, ValueError) as exc:
        print(f"[rmapss-step33] FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    print(report["verdict"], flush=True)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
