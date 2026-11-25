"""Locked R-MAPSS S3.2 protocol (`s3.2-full-v1`, RMAPSS §43).

This module keeps the frozen design visible on the Python side and composes the
thin Rust primitives. Full-game best responses are sampled lower-bound attacks;
nothing here calls them an exploitability bound or a safety certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import erf, isfinite, sqrt
import os
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Sequence

import parciver_engine as pe

from parciver.blueprint_artifact import file_sha256, require_artifacts

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "s3.2-full-v1"

BLUEPRINT = ROOT / "runs" / "blueprint-20260914-144924" / "blueprint.bin"
ABSTRACTION_DIR = ROOT / "data" / "abstraction"
FROZEN_ARTIFACTS = {
    BLUEPRINT: "9AB151368DA3B01A1A02C6FBC94414022FA3D017426170AAC3B154037CA9EECA",
    ABSTRACTION_DIR / "flop_buckets.npy":
        "0F578E1EE3A693B7405AABED021D49102DDC40190C19E8DB2ADAC2443A469A76",
    ABSTRACTION_DIR / "turn_buckets.npy":
        "D8F0A30BDCE200B591B60D38C6EE3434F33AA89B55D8BB71B3DFBBB188A47C5D",
    ABSTRACTION_DIR / "river_buckets.npy":
        "7F15923BBCD88B30ABCA7A811B80A2623117772958043556EC5D577715B0BD09",
}

SEEDS = (103, 211, 307)
KAPPA = 0.35
ALPHA_GRID = (1.0, 0.70, 0.40, 0.20, 0.10, 0.0)
C_SUB, C_TOL, C_PROXY = 0.10, 0.05, 0.10
OBS_FIT_HANDS = 200_000
OBS_VALIDATION_HANDS = 50_000
# The separately sampled prior reference uses the same locked fit-stream budget.
# Section 43 fixes the opponent observation split and requires a separate anchor
# sample; matching the 200k fit budget avoids introducing a tunable sample size.
REFERENCE_HANDS = OBS_FIT_HANDS
MIN_PUBLIC_COUNT = 200
PRIOR_STRENGTH = 20.0
RESPONSE_ITERS = 5_000_000
ATTACK_ITERS = 2_000_000
ATTACK_REPLICATES = 3
EVAL_HANDS = 2_000_000
BLOCK_SIZE = 10_000
BLOCKS = 200
FWER_ALPHA = 0.01

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
    ROOT / "scripts" / "rmapss_step32.py",
    ROOT / "scripts" / "phase_audit.py",
    ROOT / "scripts" / "check_experiment32_dom.mjs",
    ROOT / "src" / "parciver" / "dashboard" / "static" / "style.css",
    ROOT / "src" / "parciver" / "dashboard" / "static" / "experiment32.html",
    ROOT / "src" / "parciver" / "dashboard" / "static" / "experiment32.js",
)


@dataclass(frozen=True)
class OpponentSpec:
    label: str
    family: str
    strength: float
    split: str
    klass: str


FIT_SPEC = tuple(
    OpponentSpec(f"{family}(.60)", family, 0.60, "fit", "suboptimal")
    for family in ("nit", "station", "maniac")
)
HELD_SPEC = tuple(
    OpponentSpec(f"{family}({strength:.2f})", family, strength, "held", "suboptimal")
    for family in ("nit", "station", "maniac")
    for strength in (0.20, 0.35, 0.50)
)
NULL_SPEC = (
    OpponentSpec("anchor", "anchor", 0.0, "held", "null"),
    OpponentSpec("near-anchor(.02)", "uniform", 0.02, "held", "null"),
    OpponentSpec("near-anchor(.05)", "uniform", 0.05, "held", "null"),
)
POPULATION_SPEC = FIT_SPEC + HELD_SPEC + NULL_SPEC
FINAL_POLICY_SPEC = POPULATION_SPEC

PRODUCT_FAMILY_SIZE = len(SEEDS) * len(HELD_SPEC)
NULL_FAMILY_SIZE = len(SEEDS) * len(NULL_SPEC)
PROXY_FINAL_FAMILY_SIZE = len(SEEDS) * len(FINAL_POLICY_SPEC) * ATTACK_REPLICATES
PROXY_SEARCH_FAMILY_SIZE = PROXY_FINAL_FAMILY_SIZE * len(ALPHA_GRID)
KNOWN_SANITY_FAMILY_SIZE = len(SEEDS) * len(FINAL_POLICY_SPEC)

_STREAM_STAGE = {
    "reference": 1,
    "observation": 2,
    "known_response": 3,
    "known_evaluation": 4,
    "model_response": 5,
    "proxy_search_response": 6,
    "proxy_search_evaluation": 7,
    "product_evaluation": 8,
    "final_attack_response": 9,
    "final_attack_evaluation": 10,
}


def verify_frozen_artifacts() -> dict[str, str]:
    """Fail before engine loading if any locked artifact identity changed."""
    return require_artifacts(FROZEN_ARTIFACTS)


def load_anchor():
    verify_frozen_artifacts()
    saved = pe.SavedBlueprint(str(ABSTRACTION_DIR), str(BLUEPRINT))
    if saved.num_infosets() != 13_469_060:
        raise ValueError(f"frozen anchor infoset count changed: {saved.num_infosets()}")
    return saved, saved.s32_policy()


def build_population(anchor) -> dict[str, object]:
    """Build exactly the §43.3 field without evaluating any policy."""
    uniform = pe.S32Policy.uniform_like(anchor)
    population: dict[str, object] = {}
    for spec in FIT_SPEC + HELD_SPEC:
        population[spec.label] = anchor.leak(spec.family, spec.strength)
    population["anchor"] = anchor
    population["near-anchor(.02)"] = anchor.blend(uniform, 0.02)
    population["near-anchor(.05)"] = anchor.blend(uniform, 0.05)
    if tuple(population) != tuple(spec.label for spec in POPULATION_SPEC):
        raise AssertionError("population construction drifted from locked order")
    return population


def stream_seed(seed: int, stage: str, *, case_index: int = 0,
                alpha_index: int = 0, replicate: int = 0,
                responder: int = 0) -> int:
    """Derive disjoint deterministic streams for every locked sampling stage."""
    if stage not in _STREAM_STAGE:
        raise ValueError(f"unknown S3.2 stream stage {stage!r}")
    if (seed < 0 or case_index < 0 or alpha_index < 0 or replicate < 0
            or responder not in (0, 1)):
        raise ValueError("invalid S3.2 stream coordinate")
    return (
        _STREAM_STAGE[stage] * 1_000_000_000
        + seed * 1_000_000
        + case_index * 10_000
        + alpha_index * 1_000
        + replicate * 10
        + responder
    )


def response_seed(seed: int, responder: int, replicate: int = 0,
                  *, stage: str = "model_response", case_index: int = 0,
                  alpha_index: int = 0) -> int:
    """Compatibility wrapper for deterministic response-solver streams."""
    return stream_seed(seed, stage, case_index=case_index,
                       alpha_index=alpha_index, replicate=replicate,
                       responder=responder)


def build_two_seat_response(fixed, anchor, *, iterations: int, seed: int,
                            threads: int = 1, replicate: int = 0,
                            stage: str = "model_response", case_index: int = 0,
                            alpha_index: int = 0):
    """Train both seat responses and overlay sparse rows on the frozen anchor."""
    response0 = fixed.fixed_response(
        0, iterations, "linear",
        response_seed(seed, 0, replicate, stage=stage, case_index=case_index,
                      alpha_index=alpha_index), threads
    )
    response1 = fixed.fixed_response(
        1, iterations, "linear",
        response_seed(seed, 1, replicate, stage=stage, case_index=case_index,
                      alpha_index=alpha_index), threads
    )
    return anchor.blend(response0, 1.0).blend(response1, 1.0)


def fit_public_response(anchor, opponent, reference_fit, *, seed: int,
                        threads: int = 1):
    """Observe one opponent, validate its public model, then build pi_response."""
    train, validation = anchor.observe_split(
        opponent, OBS_FIT_HANDS, OBS_VALIDATION_HANDS, seed
    )
    model = train.fit_model(reference_fit, MIN_PUBLIC_COUNT, PRIOR_STRENGTH)
    model_score = dict(validation.score_model(model))
    response = build_two_seat_response(
        model, anchor, iterations=RESPONSE_ITERS, seed=seed, threads=threads
    )
    alpha_conf = min(1.0, model_score["deviation_l1"] / KAPPA)
    return model, response, model_score, alpha_conf


def paired_measure(candidate, anchor, opponent, *, seed: int) -> dict:
    result = dict(candidate.paired_evaluate(
        anchor, opponent, EVAL_HANDS, seed, BLOCK_SIZE
    ))
    validate_measurement(result)
    return result


def validate_measurement(result: dict) -> None:
    if result.get("hands") != EVAL_HANDS or result.get("block_size") != BLOCK_SIZE:
        raise ValueError("paired measurement does not use locked hand/block counts")
    blocks = result.get("block_gains_bb_per_100", [])
    if len(blocks) != BLOCKS or any(not isfinite(float(value)) for value in blocks):
        raise ValueError("paired measurement has incomplete/non-finite blocks")
    for key in (
        "candidate_bb_per_100", "anchor_bb_per_100", "gain_bb_per_100",
        "difference_stddev_bb_per_hand",
    ):
        if not isfinite(float(result[key])):
            raise ValueError(f"non-finite paired metric: {key}")


@dataclass(frozen=True)
class BlockEstimate:
    mean: float
    standard_error: float
    lower: float
    upper: float
    raw_p_greater: float
    raw_p_less: float


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normal_quantile(probability: float) -> float:
    """Monotone bisection avoids a scipy dependency for fixed block-normal CIs."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be within (0, 1)")
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _normal_cdf(mid) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def block_estimate(blocks: Sequence[float], *, threshold: float = 0.0,
                   family_size: int = 1) -> BlockEstimate:
    """Conservative familywise CI plus one-sided p-values from fixed blocks.

    Gate families additionally use Holm-adjusted p-values via `holm_adjust`; the
    interval uses the no-weaker Bonferroni alpha so every displayed bound is
    simultaneously valid for its family.
    """
    values = [float(value) for value in blocks]
    if len(values) < 2 or family_size < 1 or any(not isfinite(value) for value in values):
        raise ValueError("need at least two finite blocks and a positive family size")
    mean = fmean(values)
    se = stdev(values) / sqrt(len(values))
    tail_alpha = FWER_ALPHA / (2.0 * family_size)
    critical = _normal_quantile(1.0 - tail_alpha)
    lower, upper = mean - critical * se, mean + critical * se
    if se == 0.0:
        p_greater = 0.0 if mean > threshold else 1.0
        p_less = 0.0 if mean < threshold else 1.0
    else:
        z = (mean - threshold) / se
        p_greater = 1.0 - _normal_cdf(z)
        p_less = _normal_cdf(z)
    return BlockEstimate(mean, se, lower, upper, p_greater, p_less)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm step-down adjusted p-values in original order."""
    values = [float(value) for value in p_values]
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be a non-empty sequence within [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def frozen_metadata() -> dict:
    return {
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "alpha_grid": list(ALPHA_GRID),
        "fractions": {"sub": C_SUB, "tol": C_TOL, "proxy": C_PROXY},
        "observation_hands": [OBS_FIT_HANDS, OBS_VALIDATION_HANDS],
        "reference_hands": REFERENCE_HANDS,
        "model": {"min_count": MIN_PUBLIC_COUNT, "prior_strength": PRIOR_STRENGTH},
        "response_iters": RESPONSE_ITERS,
        "attack_iters": ATTACK_ITERS,
        "attack_replicates": ATTACK_REPLICATES,
        "evaluation": {"hands": EVAL_HANDS, "block_size": BLOCK_SIZE, "blocks": BLOCKS},
        "fwer_alpha": FWER_ALPHA,
        "population": [spec.__dict__ for spec in POPULATION_SPEC],
        "safety_language": "ABR-proxy non-regression; lower-bound attacks, not safety",
        "family_sizes": {
            "product": PRODUCT_FAMILY_SIZE,
            "null": NULL_FAMILY_SIZE,
            "proxy_final": PROXY_FINAL_FAMILY_SIZE,
            "proxy_search": PROXY_SEARCH_FAMILY_SIZE,
            "known_sanity": KNOWN_SANITY_FAMILY_SIZE,
        },
        "stream_stages": dict(_STREAM_STAGE),
    }


def estimate_payload(blocks: Sequence[float], *, threshold: float = 0.0,
                     family_size: int = 1) -> dict:
    """JSON-ready corrected interval and one-sided test summary."""
    return asdict(block_estimate(blocks, threshold=threshold,
                                 family_size=family_size))


def calibrate_thresholds(rows: Sequence[dict], seed: int) -> dict:
    """Freeze one seed's S and derived thresholds from exactly the FIT field."""
    expected = {spec.label for spec in FIT_SPEC}
    selected = [row for row in rows if row.get("seed") == seed]
    labels = {row.get("label") for row in selected}
    if labels != expected or len(selected) != len(expected):
        raise ValueError(f"seed {seed} needs exactly the three locked FIT measurements")
    gains = [float(row["measurement"]["gain_bb_per_100"]) for row in selected]
    if any(not isfinite(value) for value in gains):
        raise ValueError("FIT calibration gains must be finite")
    scale = fmean(gains)
    if scale <= 0.0:
        raise ValueError("FIT best-known-response scale S must be positive")
    return {
        "seed": seed,
        "scale_S": scale,
        "delta_sub": C_SUB * scale,
        "tol": C_TOL * scale,
        "proxy_budget": C_PROXY * scale,
    }


def measurement_blocks(measurement: dict) -> list[float]:
    validate_measurement(measurement)
    return [float(value) for value in measurement["block_gains_bb_per_100"]]


def _expected_case_ids() -> set[tuple[int, str]]:
    return {(seed, spec.label) for seed in SEEDS for spec in FINAL_POLICY_SPEC}


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


def finalize_report(*, artifact_hashes: dict[str, str], small_game_control: dict,
                    calibration_rows: Sequence[dict], case_rows: Sequence[dict]) -> dict:
    """Re-derive all seven §43.6 gates from complete immutable run records.

    This is deliberately pure: runner telemetry and dashboard labels cannot turn a
    missing or malformed case into a PASS. Holm corrections span all three seeds.
    """
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

    product_tests: list[tuple[dict, BlockEstimate]] = []
    null_tests: list[tuple[dict, BlockEstimate, BlockEstimate, BlockEstimate]] = []
    proxy_tests: list[tuple[dict, dict, BlockEstimate]] = []
    known_tests: list[tuple[dict, BlockEstimate]] = []
    observation_ok = True

    if complete:
        for (seed, label), row in rows_by_id.items():
            spec = next(spec for spec in FINAL_POLICY_SPEC if spec.label == label)
            threshold = thresholds[seed]
            product_blocks = measurement_blocks(row["product"])
            known_blocks = measurement_blocks(row["known_response"])

            if spec in HELD_SPEC:
                product_tests.append((row, block_estimate(
                    product_blocks, threshold=threshold["delta_sub"],
                    family_size=PRODUCT_FAMILY_SIZE)))
                model = row["model"]
                observation_ok = observation_ok and (
                    float(model["coverage"]) >= 0.95
                    and float(model["model_log_loss"]) < float(model["reference_log_loss"])
                    and bool(row.get("public_only_keys"))
                )
            elif spec in NULL_SPEC:
                center = block_estimate(product_blocks, family_size=NULL_FAMILY_SIZE)
                lower = block_estimate(
                    product_blocks, threshold=-threshold["tol"],
                    family_size=NULL_FAMILY_SIZE)
                upper = block_estimate(
                    product_blocks, threshold=threshold["tol"],
                    family_size=NULL_FAMILY_SIZE)
                null_tests.append((row, center, lower, upper))

            excess = [a - b for a, b in zip(product_blocks, known_blocks, strict=True)]
            known_tests.append((row, block_estimate(
                excess, family_size=KNOWN_SANITY_FAMILY_SIZE)))

            attacks = row.get("final_attacks", [])
            if len(attacks) != ATTACK_REPLICATES:
                complete = False
                continue
            for attack in attacks:
                loss_blocks = [-value for value in measurement_blocks(attack["measurement"])]
                proxy_tests.append((row, attack, block_estimate(
                    loss_blocks, threshold=threshold["proxy_budget"],
                    family_size=PROXY_FINAL_FAMILY_SIZE)))

    product_adj = holm_adjust(est.raw_p_greater for _, est in product_tests) if product_tests else []
    null_ps = ([est.raw_p_greater for _, _, est, _ in null_tests]
               + [est.raw_p_less for _, _, _, est in null_tests])
    null_adj = holm_adjust(null_ps) if null_ps else []
    proxy_adj = holm_adjust(est.raw_p_less for _, _, est in proxy_tests) if proxy_tests else []
    known_adj = holm_adjust(est.raw_p_greater for _, est in known_tests) if known_tests else []

    product_ok = len(product_tests) == PRODUCT_FAMILY_SIZE and all(
        est.lower > thresholds[int(row["seed"])]["delta_sub"]
        and adjusted < FWER_ALPHA
        for (row, est), adjusted in zip(product_tests, product_adj, strict=True)
    )
    n_null = len(null_tests)
    null_ok = n_null == NULL_FAMILY_SIZE and all(
        center.lower >= -thresholds[int(row["seed"])]["tol"]
        and center.upper <= thresholds[int(row["seed"])]["tol"]
        and null_adj[index] < FWER_ALPHA
        and null_adj[n_null + index] < FWER_ALPHA
        for index, (row, center, _lower, _upper) in enumerate(null_tests)
    )
    proxy_ok = len(proxy_tests) == PROXY_FINAL_FAMILY_SIZE and all(
        est.upper <= thresholds[int(row["seed"])]["proxy_budget"]
        and adjusted < FWER_ALPHA
        for (row, _attack, est), adjusted in zip(proxy_tests, proxy_adj, strict=True)
    )
    known_ok = len(known_tests) == KNOWN_SANITY_FAMILY_SIZE and all(
        not (est.lower > 0.0 and adjusted < FWER_ALPHA)
        for (_row, est), adjusted in zip(known_tests, known_adj, strict=True)
    )

    gates = {
        "artifact/completeness": complete,
        "small-game control": bool(small_game_control.get("passed")),
        "observation": complete and observation_ok,
        "product": complete and product_ok,
        "null discipline": complete and null_ok,
        "ABR-proxy non-regression": complete and proxy_ok,
        "known-response sanity": complete and known_ok,
    }
    passed = all(gates.values())
    return {
        "protocol": PROTOCOL,
        "thresholds": thresholds,
        "gates": gates,
        "all_pass": passed,
        "verdict": ("PASS (S3.2 synthetic-field product + ABR-proxy non-regression)"
                    if passed else "FAIL (S3.2 locked gate breach)"),
        "correction": {"method": "Holm", "fwer_alpha": FWER_ALPHA},
        "safety_language": "lower-bound sampled attacks; not a safety certificate",
    }


def source_hashes() -> dict[str, str]:
    missing = [path for path in AUDIT_SOURCES if not path.is_file()]
    if missing:
        raise ValueError(f"S3.2 audit source set incomplete: {missing}")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in AUDIT_SOURCES}


def write_audit_receipt(path: str | Path = AUDIT_RECEIPT, *,
                        small_game_control: dict | None = None) -> Path:
    """Write the full-audit receipt consumed by the locked production runner."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "full_audit_passed": True,
        "source_hashes": source_hashes(),
        "small_game_control": small_game_control or {"passed": False},
        "pid": os.getpid(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def require_audit_receipt(path: str | Path = AUDIT_RECEIPT) -> dict:
    """Refuse product execution if code changed since the green full audit."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"missing S3.2 engineering-audit receipt: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (payload.get("protocol") != PROTOCOL
            or payload.get("full_audit_passed") is not True
            or payload.get("small_game_control", {}).get("passed") is not True
            or payload.get("source_hashes") != source_hashes()):
        raise ValueError("S3.2 engineering-audit receipt is stale or invalid")
    return payload
