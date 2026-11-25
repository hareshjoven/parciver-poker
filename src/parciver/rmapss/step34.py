"""Locked R-MAPSS S3.4 five-level public-support protocol (§48).

The failed S3.2 and S3.3 implementations remain frozen.  This module owns the
new protocol constants, stream namespace, observation summaries and gates.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from math import isfinite
import os
from pathlib import Path
from typing import Sequence

from parciver.blueprint_artifact import file_sha256
from parciver.rmapss import step32, step33

ROOT = step33.ROOT
PROTOCOL = "s3.4-public-support-v1"

BLUEPRINT = step33.BLUEPRINT
ABSTRACTION_DIR = step33.ABSTRACTION_DIR
FROZEN_ARTIFACTS = step33.FROZEN_ARTIFACTS

SEEDS = (401, 409, 419)
KAPPA = step33.KAPPA
ALPHA_GRID = step33.ALPHA_GRID
C_SUB, C_TOL, C_PROXY = step33.C_SUB, step33.C_TOL, step33.C_PROXY

REFERENCE_HANDS = step33.REFERENCE_HANDS
OBS_FIT_HANDS = step33.OBS_FIT_HANDS
OBS_CONFIRM_HANDS = step33.OBS_CONFIRM_HANDS
OBS_BLOCK_SIZE = step33.OBS_BLOCK_SIZE
REFERENCE_BLOCKS = step33.REFERENCE_BLOCKS
FIT_BLOCKS = step33.FIT_BLOCKS
CONFIRM_BLOCKS = step33.CONFIRM_BLOCKS

EXACT_STRENGTH = 200.0
BACKOFF_STRENGTH = 50.0
CONTEXT_STRENGTH = 20.0
POSITION_STRENGTH = 10.0
GLOBAL_STRENGTH = 5.0
EFFECTIVE_COVERAGE = 0.99
SHADOW_NLL_MARGIN = 0.005
ECE_LIMIT = step33.ECE_LIMIT

RESPONSE_ITERS = step33.RESPONSE_ITERS
ATTACK_ITERS = step33.ATTACK_ITERS
ATTACK_REPLICATES = step33.ATTACK_REPLICATES
EVAL_HANDS = step33.EVAL_HANDS
BLOCK_SIZE = step33.BLOCK_SIZE
BLOCKS = step33.BLOCKS
FWER_ALPHA = step33.FWER_ALPHA
MAX_WORKERS = 16
PERFORMANCE_TIME_RATIO = 1.10
PERFORMANCE_BYTES_RATIO = 1.25

OpponentSpec = step32.OpponentSpec
FIT_SPEC = step33.FIT_SPEC
HELD_SPEC = step33.HELD_SPEC
NULL_SPEC = step33.NULL_SPEC
POPULATION_SPEC = step33.POPULATION_SPEC
FINAL_POLICY_SPEC = step33.FINAL_POLICY_SPEC

OBSERVATION_FAMILY_SIZE = len(SEEDS) * len(HELD_SPEC)
PRODUCT_FAMILY_SIZE = len(SEEDS) * len(HELD_SPEC)
NULL_FAMILY_SIZE = len(SEEDS) * len(NULL_SPEC)
PROXY_FINAL_FAMILY_SIZE = len(SEEDS) * len(FINAL_POLICY_SPEC) * ATTACK_REPLICATES
PROXY_SEARCH_FAMILY_SIZE = PROXY_FINAL_FAMILY_SIZE * len(ALPHA_GRID)
KNOWN_SANITY_FAMILY_SIZE = len(SEEDS) * len(FINAL_POLICY_SPEC)

_STREAM_STAGE = {
    "reference": 51,
    "observation_fit": 52,
    "observation_confirmation": 53,
    "known_response": 54,
    "known_evaluation": 55,
    "model_response": 56,
    "proxy_search_response": 57,
    "proxy_search_evaluation": 58,
    "product_evaluation": 59,
    "final_attack_response": 60,
    "final_attack_evaluation": 61,
}

AUDIT_RECEIPT = ROOT / "runs" / f"{PROTOCOL}-engineering-audit.json"
AUDIT_SOURCES = (
    ROOT / "RMAPSS.md",
    ROOT / "engine" / "src" / "analysis.rs",
    ROOT / "engine" / "src" / "lib.rs",
    ROOT / "engine" / "src" / "mccfr.rs",
    ROOT / "engine" / "src" / "nlhe.rs",
    ROOT / "engine" / "src" / "s32.rs",
    ROOT / "src" / "parciver" / "blueprint_artifact.py",
    ROOT / "src" / "parciver" / "rmapss" / "step32.py",
    ROOT / "src" / "parciver" / "rmapss" / "step33.py",
    ROOT / "src" / "parciver" / "rmapss" / "step34.py",
    ROOT / "scripts" / "rmapss_step32.py",
    ROOT / "scripts" / "rmapss_step33.py",
    ROOT / "scripts" / "rmapss_step34.py",
    ROOT / "scripts" / "phase_audit.py",
    ROOT / "scripts" / "check_experiment34_dom.mjs",
    ROOT / "src" / "parciver" / "dashboard" / "static" / "style.css",
    ROOT / "src" / "parciver" / "dashboard" / "static" / "experiment34.html",
    ROOT / "src" / "parciver" / "dashboard" / "static" / "experiment34.js",
)


def verify_frozen_artifacts() -> dict[str, str]:
    return step33.verify_frozen_artifacts()


def load_anchor():
    return step33.load_anchor()


def build_population(anchor) -> dict[str, object]:
    return step33.build_population(anchor)


def stream_seed(seed: int, stage: str, *, case_index: int = 0,
                alpha_index: int = 0, replicate: int = 0,
                responder: int = 0, block: int = 0) -> int:
    if stage not in _STREAM_STAGE:
        raise ValueError(f"unknown S3.4 stream stage {stage!r}")
    if (seed not in SEEDS or case_index < 0 or alpha_index < 0
            or replicate < 0 or responder not in (0, 1) or not 0 <= block < 500):
        raise ValueError("invalid S3.4 stream coordinate")
    return (
        _STREAM_STAGE[stage] * 1_000_000_000_000
        + seed * 1_000_000_000
        + case_index * 1_000_000
        + alpha_index * 10_000
        + replicate * 1_000
        + responder * 500
        + block
    )


def block_seeds(seed: int, stage: str, *, case_index: int, blocks: int) -> list[int]:
    if blocks < 1 or blocks > 500:
        raise ValueError("S3.4 block count must be within 1..=500")
    return [stream_seed(seed, stage, case_index=case_index, block=index)
            for index in range(blocks)]


def response_seed(seed: int, responder: int, replicate: int = 0, *,
                  stage: str = "model_response", case_index: int = 0,
                  alpha_index: int = 0) -> int:
    return stream_seed(seed, stage, case_index=case_index,
                       alpha_index=alpha_index, replicate=replicate,
                       responder=responder)


merge_observations = step33.merge_observations


def fit_five_level_model(train, reference):
    return train.fit_five_level_model(
        reference, EXACT_STRENGTH, BACKOFF_STRENGTH, CONTEXT_STRENGTH,
        POSITION_STRENGTH, GLOBAL_STRENGTH,
    )


# The S3.4 runner uses the same observation-stage interface as S3.3 while this
# alias selects the separately implemented five-level policy.
fit_hierarchical_model = fit_five_level_model


def score_blocks(model, blocks: Sequence[object]) -> list[dict]:
    if len(blocks) != CONFIRM_BLOCKS:
        raise ValueError(f"need exactly {CONFIRM_BLOCKS} confirmation blocks")
    return [dict(block.score_five_level_model(model)) for block in blocks]


def _sum(rows: Sequence[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _sum_int(rows: Sequence[dict], key: str) -> int:
    return sum(int(row[key]) for row in rows)


def _block_ratio(row: dict, numerator: str, denominator: str = "decisions") -> float:
    den = int(row[denominator])
    if den <= 0:
        raise ValueError(f"zero denominator {denominator!r} in confirmation block")
    return float(row[numerator]) / den


def _paired_blocks(rows: Sequence[dict], left: str, right: str,
                   denominator: str = "decisions") -> list[float]:
    return [(_block_ratio(row, left, denominator)
             - _block_ratio(row, right, denominator)) for row in rows]


def summarize_model_blocks(rows: Sequence[dict]) -> dict:
    if len(rows) != CONFIRM_BLOCKS:
        raise ValueError(f"need exactly {CONFIRM_BLOCKS} confirmation score blocks")
    decisions = _sum_int(rows, "decisions")
    if decisions <= 0:
        raise ValueError("confirmation produced no observed decisions")
    sparse_decisions = _sum_int(rows, "sparse_decisions")
    ece_count = [sum(int(row["ece_count"][i]) for row in rows) for i in range(10)]
    ece_conf = [sum(float(row["ece_confidence_sum"][i]) for row in rows)
                for i in range(10)]
    ece_acc = [sum(int(row["ece_accuracy_sum"][i]) for row in rows)
               for i in range(10)]
    ece = sum(abs(ece_acc[i] / count - ece_conf[i] / count) * count / decisions
              for i, count in enumerate(ece_count) if count)
    sparse_blocks = [
        (float(row["sparse_model_nll_sum"])
         - float(row["sparse_reference_nll_sum"])) / OBS_BLOCK_SIZE
        for row in rows
    ]
    return {
        "decisions": decisions,
        "effective_coverage": _sum(rows, "effective_weight_sum") / decisions,
        "zero_weight_decisions": _sum_int(rows, "zero_weight_decisions"),
        "three_level_zero_decisions": _sum_int(rows, "three_level_zero_decisions"),
        "recovered_three_level_zero_decisions": _sum_int(
            rows, "recovered_three_level_zero_decisions"),
        "legacy_coverage": _sum_int(rows, "legacy_supported_decisions") / decisions,
        "model_nll": _sum(rows, "model_nll_sum") / decisions,
        "reference_nll": _sum(rows, "reference_nll_sum") / decisions,
        "shadow_nll": _sum(rows, "shadow_nll_sum") / decisions,
        "legacy_nll": _sum(rows, "legacy_nll_sum") / decisions,
        "sparse_decisions": sparse_decisions,
        "sparse_model_nll": (_sum(rows, "sparse_model_nll_sum") / sparse_decisions
                             if sparse_decisions else None),
        "sparse_reference_nll": (_sum(rows, "sparse_reference_nll_sum") / sparse_decisions
                                 if sparse_decisions else None),
        "model_brier": _sum(rows, "model_brier_sum") / decisions,
        "reference_brier": _sum(rows, "reference_brier_sum") / decisions,
        "deviation_l1": _sum(rows, "deviation_l1_sum") / decisions,
        "ece": ece,
        "nll_anchor_blocks": _paired_blocks(rows, "model_nll_sum", "reference_nll_sum"),
        "sparse_nll_anchor_blocks": sparse_blocks,
        "nll_shadow_blocks": _paired_blocks(rows, "model_nll_sum", "shadow_nll_sum"),
        "brier_anchor_blocks": _paired_blocks(rows, "model_brier_sum", "reference_brier_sum"),
    }


def _finite_numbers(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    return False


def observation_report(rows: Sequence[dict]) -> dict:
    expected = {(seed, spec.label) for seed in SEEDS for spec in HELD_SPEC}
    by_id = {(int(row.get("seed", -1)), row.get("label")): row for row in rows}
    gate_names = ("completeness/leakage", "five-level continuous support",
                  "anchor NLL gain", "sparse-row recovery",
                  "S3.3-shadow non-inferiority", "calibration")
    complete = (
        len(rows) == len(expected)
        and set(by_id) == expected
        and all(row.get("protocol") == PROTOCOL for row in rows)
        and all(row.get("public_only_keys") is True for row in rows)
        and _finite_numbers(list(rows))
    )
    if not complete:
        return {"protocol": PROTOCOL, "complete": False, "all_pass": False,
                "verdict": "FAIL (S3.4 observation family incomplete or malformed)",
                "gates": {name: False for name in gate_names}}

    ordered = [by_id[(seed, spec.label)] for seed in SEEDS for spec in HELD_SPEC]
    estimates = {
        "anchor": [step32.block_estimate(row["model"]["nll_anchor_blocks"],
                                          family_size=OBSERVATION_FAMILY_SIZE)
                   for row in ordered],
        "sparse": [step32.block_estimate(row["model"]["sparse_nll_anchor_blocks"],
                                          family_size=OBSERVATION_FAMILY_SIZE)
                   for row in ordered],
        "shadow": [step32.block_estimate(row["model"]["nll_shadow_blocks"],
                                          threshold=SHADOW_NLL_MARGIN,
                                          family_size=OBSERVATION_FAMILY_SIZE)
                   for row in ordered],
        "brier": [step32.block_estimate(row["model"]["brier_anchor_blocks"],
                                         family_size=OBSERVATION_FAMILY_SIZE)
                  for row in ordered],
    }
    adjusted = {name: step32.holm_adjust(est.raw_p_less for est in family)
                for name, family in estimates.items()}
    coverage_ok = all(
        float(row["model"]["effective_coverage"]) >= EFFECTIVE_COVERAGE
        and int(row["model"]["zero_weight_decisions"]) == 0
        and (int(row["model"]["recovered_three_level_zero_decisions"])
             + int(row["model"]["zero_weight_decisions"])
             == int(row["model"]["three_level_zero_decisions"]))
        for row in ordered
    )
    anchor_ok = all(est.upper < 0.0 and p < FWER_ALPHA
                    for est, p in zip(estimates["anchor"], adjusted["anchor"], strict=True))
    sparse_ok = all(
        ((int(row["model"]["sparse_decisions"]) == 0
          and float(row["model"]["legacy_coverage"]) == 1.0)
         or (est.upper < 0.0 and p < FWER_ALPHA))
        for row, est, p in zip(ordered, estimates["sparse"], adjusted["sparse"], strict=True)
    )
    shadow_ok = all(est.upper <= SHADOW_NLL_MARGIN and p < FWER_ALPHA
                    for est, p in zip(estimates["shadow"], adjusted["shadow"], strict=True))
    brier_ok = all(est.upper < 0.0 and p < FWER_ALPHA
                   for est, p in zip(estimates["brier"], adjusted["brier"], strict=True))
    ece_ok = all(float(row["model"]["ece"]) <= ECE_LIMIT for row in ordered)
    gates = {
        "completeness/leakage": True,
        "five-level continuous support": coverage_ok,
        "anchor NLL gain": anchor_ok,
        "sparse-row recovery": sparse_ok,
        "S3.3-shadow non-inferiority": shadow_ok,
        "calibration": brier_ok and ece_ok,
    }
    details = [{
        "seed": row["seed"], "label": row["label"],
        "effective_coverage": row["model"]["effective_coverage"],
        "three_level_zero_decisions": row["model"]["three_level_zero_decisions"],
        "recovered_three_level_zero_decisions": row["model"]["recovered_three_level_zero_decisions"],
        "zero_weight_decisions": row["model"]["zero_weight_decisions"],
        "legacy_coverage": row["model"]["legacy_coverage"],
        "ece": row["model"]["ece"],
        "adjusted_p": {name: adjusted[name][index] for name in adjusted},
        "upper": {name: asdict(estimates[name][index])["upper"] for name in estimates},
    } for index, row in enumerate(ordered)]
    passed = all(gates.values())
    return {
        "protocol": PROTOCOL, "complete": True, "all_pass": passed,
        "verdict": ("PASS (S3.4 locked public-support recovery)" if passed
                    else "FAIL (S3.4 locked observation gate breach)"),
        "gates": gates, "details": details,
        "correction": {"method": "Holm", "fwer_alpha": FWER_ALPHA,
                       "family_size": OBSERVATION_FAMILY_SIZE},
    }


def build_two_seat_response(fixed, anchor, *, iterations: int, seed: int,
                            threads: int = 1, replicate: int = 0,
                            stage: str = "model_response", case_index: int = 0,
                            alpha_index: int = 0):
    response0 = fixed.fixed_response(
        0, iterations, "linear",
        response_seed(seed, 0, replicate, stage=stage,
                      case_index=case_index, alpha_index=alpha_index), threads,
    )
    response1 = fixed.fixed_response(
        1, iterations, "linear",
        response_seed(seed, 1, replicate, stage=stage,
                      case_index=case_index, alpha_index=alpha_index), threads,
    )
    return anchor.blend(response0, 1.0).blend(response1, 1.0)


def paired_measure(candidate, anchor, opponent, *, seed: int, threads: int) -> dict:
    result = dict(candidate.paired_evaluate_blocks(
        anchor, opponent, EVAL_HANDS, seed, BLOCK_SIZE, threads
    ))
    step32.validate_measurement(result)
    return result


def calibrate_thresholds(rows: Sequence[dict], seed: int) -> dict:
    expected = {spec.label for spec in FIT_SPEC}
    selected = [row for row in rows if row.get("seed") == seed]
    if len(selected) != len(expected) or {row.get("label") for row in selected} != expected:
        raise ValueError(f"seed {seed} needs exactly the three locked FIT measurements")
    gains = [float(row["measurement"]["gain_bb_per_100"]) for row in selected]
    if any(not isfinite(value) for value in gains):
        raise ValueError("FIT calibration gains must be finite")
    scale = sum(gains) / len(gains)
    if scale <= 0.0:
        raise ValueError("FIT best-known-response scale S must be positive")
    return {"seed": seed, "scale_S": scale, "delta_sub": C_SUB * scale,
            "tol": C_TOL * scale, "proxy_budget": C_PROXY * scale}


def _expected_case_ids() -> set[tuple[int, str]]:
    return {(seed, spec.label) for seed in SEEDS for spec in FINAL_POLICY_SPEC}


def finalize_report(*, artifact_hashes: dict[str, str], small_game_control: dict,
                    parity_control: dict, performance_control: dict,
                    observation: dict,
                    calibration_rows: Sequence[dict], case_rows: Sequence[dict]) -> dict:
    """Re-derive the observation and downstream §48 gates from stored blocks."""
    thresholds = {seed: calibrate_thresholds(calibration_rows, seed) for seed in SEEDS}
    rows_by_id = {(int(row.get("seed", -1)), row.get("label")): row for row in case_rows}
    expected_ids = _expected_case_ids()
    expected_hashes = {str(path): digest for path, digest in FROZEN_ARTIFACTS.items()}
    complete = (
        set(rows_by_id) == expected_ids
        and len(case_rows) == len(expected_ids)
        and artifact_hashes == expected_hashes
        and _finite_numbers(list(case_rows))
        and _finite_numbers(list(calibration_rows))
        and all(row.get("protocol") == PROTOCOL for row in case_rows)
    )

    product_tests = []
    null_tests = []
    proxy_tests = []
    known_tests = []
    if complete:
        for (seed, label), row in rows_by_id.items():
            spec = next(spec for spec in FINAL_POLICY_SPEC if spec.label == label)
            threshold = thresholds[seed]
            product_blocks = step32.measurement_blocks(row["product"])
            known_blocks = step32.measurement_blocks(row["known_response"])
            if spec in HELD_SPEC:
                product_tests.append((row, step32.block_estimate(
                    product_blocks, threshold=threshold["delta_sub"],
                    family_size=PRODUCT_FAMILY_SIZE)))
            elif spec in NULL_SPEC:
                center = step32.block_estimate(product_blocks, family_size=NULL_FAMILY_SIZE)
                lower = step32.block_estimate(
                    product_blocks, threshold=-threshold["tol"],
                    family_size=NULL_FAMILY_SIZE)
                upper = step32.block_estimate(
                    product_blocks, threshold=threshold["tol"],
                    family_size=NULL_FAMILY_SIZE)
                null_tests.append((row, center, lower, upper))
            excess = [a - b for a, b in zip(product_blocks, known_blocks, strict=True)]
            known_tests.append((row, step32.block_estimate(
                excess, family_size=KNOWN_SANITY_FAMILY_SIZE)))
            attacks = row.get("final_attacks", [])
            if len(attacks) != ATTACK_REPLICATES:
                complete = False
                continue
            for attack in attacks:
                loss = [-value for value in step32.measurement_blocks(attack["measurement"])]
                proxy_tests.append((row, attack, step32.block_estimate(
                    loss, threshold=threshold["proxy_budget"],
                    family_size=PROXY_FINAL_FAMILY_SIZE)))

    product_adj = step32.holm_adjust(est.raw_p_greater for _, est in product_tests) \
        if product_tests else []
    null_ps = ([est.raw_p_greater for _, _, est, _ in null_tests]
               + [est.raw_p_less for _, _, _, est in null_tests])
    null_adj = step32.holm_adjust(null_ps) if null_ps else []
    proxy_adj = step32.holm_adjust(est.raw_p_less for _, _, est in proxy_tests) \
        if proxy_tests else []
    known_adj = step32.holm_adjust(est.raw_p_greater for _, est in known_tests) \
        if known_tests else []
    product_ok = len(product_tests) == PRODUCT_FAMILY_SIZE and all(
        est.lower > thresholds[int(row["seed"])]["delta_sub"] and p < FWER_ALPHA
        for (row, est), p in zip(product_tests, product_adj, strict=True))
    n_null = len(null_tests)
    null_ok = n_null == NULL_FAMILY_SIZE and all(
        center.lower >= -thresholds[int(row["seed"])]["tol"]
        and center.upper <= thresholds[int(row["seed"])]["tol"]
        and null_adj[i] < FWER_ALPHA and null_adj[n_null + i] < FWER_ALPHA
        for i, (row, center, _lower, _upper) in enumerate(null_tests))
    proxy_ok = len(proxy_tests) == PROXY_FINAL_FAMILY_SIZE and all(
        est.upper <= thresholds[int(row["seed"])]["proxy_budget"] and p < FWER_ALPHA
        for (row, _attack, est), p in zip(proxy_tests, proxy_adj, strict=True))
    known_ok = len(known_tests) == KNOWN_SANITY_FAMILY_SIZE and all(
        not (est.lower > 0.0 and p < FWER_ALPHA)
        for (_row, est), p in zip(known_tests, known_adj, strict=True))
    gates = {
        "artifact/completeness": complete,
        "small-game control": bool(small_game_control.get("passed")),
        "parallel parity": bool(parity_control.get("passed")),
        "performance bounds": bool(performance_control.get("passed")),
        "observation recovery": bool(observation.get("all_pass")),
        "product": complete and product_ok,
        "null discipline": complete and null_ok,
        "ABR-proxy non-regression": complete and proxy_ok,
        "known-response sanity": complete and known_ok,
    }
    passed = all(gates.values())
    return {
        "protocol": PROTOCOL,
        "thresholds": thresholds,
        "observation": observation,
        "gates": gates,
        "all_pass": passed,
        "verdict": ("PASS (S3.4 five-level public-support recovery + "
                    "synthetic-field product + ABR-proxy non-regression)"
                    if passed else "FAIL (S3.4 locked gate breach)"),
        "correction": {"method": "Holm", "fwer_alpha": FWER_ALPHA},
        "safety_language": "lower-bound sampled attacks; not a safety certificate",
    }



def frozen_metadata() -> dict:
    return {
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "alpha_grid": list(ALPHA_GRID),
        "fractions": {"sub": C_SUB, "tol": C_TOL, "proxy": C_PROXY},
        "observation": {
            "reference_hands": REFERENCE_HANDS,
            "fit_hands": OBS_FIT_HANDS,
            "confirmation_hands": OBS_CONFIRM_HANDS,
            "block_size": OBS_BLOCK_SIZE,
            "hierarchy_strengths": [EXACT_STRENGTH, BACKOFF_STRENGTH,
                                    CONTEXT_STRENGTH, POSITION_STRENGTH,
                                    GLOBAL_STRENGTH],
            "effective_coverage": EFFECTIVE_COVERAGE,
            "shadow_nll_margin": SHADOW_NLL_MARGIN,
            "ece_limit": ECE_LIMIT,
        },
        "response_iters": RESPONSE_ITERS,
        "attack_iters": ATTACK_ITERS,
        "attack_replicates": ATTACK_REPLICATES,
        "evaluation": {"hands": EVAL_HANDS, "block_size": BLOCK_SIZE,
                       "blocks": BLOCKS},
        "performance": {"time_ratio": PERFORMANCE_TIME_RATIO,
                        "bytes_ratio": PERFORMANCE_BYTES_RATIO,
                        "workers": [1, 2, 16]},
        "max_workers": MAX_WORKERS,
        "fwer_alpha": FWER_ALPHA,
        "population": [spec.__dict__ for spec in POPULATION_SPEC],
        "stream_stages": dict(_STREAM_STAGE),
        "safety_language": "ABR-proxy non-regression; lower-bound attacks, not safety",
    }


def source_hashes() -> dict[str, str]:
    missing = [path for path in AUDIT_SOURCES if not path.is_file()]
    if missing:
        raise ValueError(f"S3.4 audit source set incomplete: {missing}")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in AUDIT_SOURCES}


def write_audit_receipt(path: str | Path = AUDIT_RECEIPT, *,
                        small_game_control: dict | None = None,
                        parity_control: dict | None = None,
                        performance_control: dict | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL, "full_audit_passed": True,
        "source_hashes": source_hashes(),
        "artifact_hashes": verify_frozen_artifacts(),
        "small_game_control": small_game_control or {"passed": False},
        "parity_control": parity_control or {"passed": False},
        "performance_control": performance_control or {"passed": False},
        "pid": os.getpid(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, target)
    return target


def require_audit_receipt(path: str | Path = AUDIT_RECEIPT) -> dict:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"missing S3.4 engineering-audit receipt: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (payload.get("protocol") != PROTOCOL
            or payload.get("full_audit_passed") is not True
            or payload.get("small_game_control", {}).get("passed") is not True
            or payload.get("parity_control", {}).get("passed") is not True
            or payload.get("performance_control", {}).get("passed") is not True
            or payload.get("source_hashes") != source_hashes()
            or payload.get("artifact_hashes") != verify_frozen_artifacts()):
        raise ValueError("S3.4 engineering-audit receipt is stale or invalid")
    return payload
