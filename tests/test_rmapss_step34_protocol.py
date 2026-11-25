# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
from __future__ import annotations

from copy import deepcopy

import parciver_engine as pe

from parciver.rmapss import step34


def _score_block() -> dict:
    decisions = 100
    return {
        "decisions": decisions,
        "effective_weight_sum": 99.5,
        "zero_weight_decisions": 0,
        "three_level_zero_decisions": 1,
        "recovered_three_level_zero_decisions": 1,
        "legacy_supported_decisions": 95,
        "model_nll_sum": 10.0,
        "reference_nll_sum": 20.0,
        "shadow_nll_sum": 10.1,
        "legacy_nll_sum": 10.2,
        "sparse_decisions": 5,
        "sparse_model_nll_sum": 0.5,
        "sparse_reference_nll_sum": 1.0,
        "model_brier_sum": 5.0,
        "reference_brier_sum": 10.0,
        "deviation_l1_sum": 1.0,
        "ece_count": [decisions] + [0] * 9,
        "ece_confidence_sum": [0.0] * 10,
        "ece_accuracy_sum": [0] * 10,
    }


def _observation_rows() -> list[dict]:
    rows = []
    for seed in step34.SEEDS:
        for spec in step34.HELD_SPEC:
            blocks = [_score_block() for _ in range(step34.CONFIRM_BLOCKS)]
            rows.append({
                "protocol": step34.PROTOCOL,
                "seed": seed,
                "label": spec.label,
                "public_only_keys": True,
                "model": step34.summarize_model_blocks(blocks),
            })
    return rows


def test_locked_constants_seed_registry_and_engine_surface():
    assert step34.PROTOCOL == "s3.4-public-support-v1"
    assert step34.SEEDS == (401, 409, 419)
    assert (step34.EXACT_STRENGTH, step34.BACKOFF_STRENGTH,
            step34.CONTEXT_STRENGTH, step34.POSITION_STRENGTH,
            step34.GLOBAL_STRENGTH) == (200.0, 50.0, 20.0, 10.0, 5.0)
    assert step34.EFFECTIVE_COVERAGE == 0.99
    assert step34.SHADOW_NLL_MARGIN == 0.005
    assert step34.OBSERVATION_FAMILY_SIZE == 27
    assert step34.PERFORMANCE_TIME_RATIO == 1.10
    assert step34.PERFORMANCE_BYTES_RATIO == 1.25
    assert hasattr(pe.S32Observation, "support_fingerprint")
    assert hasattr(pe.S32Observation, "count_table_bytes")
    assert hasattr(pe.S32Observation, "fit_five_level_model")
    assert hasattr(pe.S32Observation, "score_five_level_model")


def test_all_stream_coordinates_are_disjoint_and_new_namespace():
    streams = {
        step34.stream_seed(seed, stage, case_index=case, alpha_index=alpha,
                           replicate=replicate, responder=responder, block=block)
        for seed in step34.SEEDS
        for stage in step34.frozen_metadata()["stream_stages"]
        for case in range(len(step34.POPULATION_SPEC))
        for alpha in range(len(step34.ALPHA_GRID))
        for replicate in range(step34.ATTACK_REPLICATES)
        for responder in (0, 1)
        for block in (0, 199)
    }
    expected = (len(step34.SEEDS) * len(step34.frozen_metadata()["stream_stages"])
                * len(step34.POPULATION_SPEC) * len(step34.ALPHA_GRID)
                * step34.ATTACK_REPLICATES * 2 * 2)
    assert len(streams) == expected
    assert min(streams) // 1_000_000_000_000 >= 51


def test_summary_and_complete_observation_report_pass():
    rows = _observation_rows()
    report = step34.observation_report(rows)
    assert report["all_pass"] is True
    assert all(report["gates"].values())
    assert rows[0]["model"]["effective_coverage"] == 0.995
    assert rows[0]["model"]["three_level_zero_decisions"] == 50
    assert rows[0]["model"]["recovered_three_level_zero_decisions"] == 50
    assert rows[0]["model"]["zero_weight_decisions"] == 0
    assert rows[0]["model"]["ece"] == 0.0


def test_observation_report_fails_closed_on_support_accounting_or_missing_case():
    rows = _observation_rows()
    failed = deepcopy(rows)
    failed[0]["model"]["zero_weight_decisions"] = 1
    report = step34.observation_report(failed)
    assert report["gates"]["five-level continuous support"] is False
    assert report["all_pass"] is False

    malformed = deepcopy(rows)
    malformed[0]["model"]["recovered_three_level_zero_decisions"] = 49
    report = step34.observation_report(malformed)
    assert report["gates"]["five-level continuous support"] is False

    incomplete = step34.observation_report(rows[:-1])
    assert incomplete["complete"] is False
    assert incomplete["all_pass"] is False
    assert not any(incomplete["gates"].values())


def test_confirmation_score_requires_all_fixed_blocks():
    try:
        step34.summarize_model_blocks([_score_block()] * 49)
    except ValueError as exc:
        assert "exactly 50" in str(exc)
    else:
        raise AssertionError("incomplete confirmation blocks must fail")


def _measurement(gain: float) -> dict:
    return {
        "hands": step34.EVAL_HANDS,
        "block_size": step34.BLOCK_SIZE,
        "candidate_bb_per_100": gain,
        "anchor_bb_per_100": 0.0,
        "gain_bb_per_100": gain,
        "difference_stddev_bb_per_hand": 0.0,
        "block_gains_bb_per_100": [gain] * step34.BLOCKS,
    }


def test_final_report_rederives_complete_downstream_gate_family():
    observation = step34.observation_report(_observation_rows())
    calibration = [
        {"seed": seed, "label": spec.label, "measurement": _measurement(10.0)}
        for seed in step34.SEEDS for spec in step34.FIT_SPEC
    ]
    cases = []
    for seed in step34.SEEDS:
        for spec in step34.POPULATION_SPEC:
            gain = 2.0 if spec in step34.HELD_SPEC else (0.0 if spec in step34.NULL_SPEC else 1.0)
            cases.append({
                "protocol": step34.PROTOCOL,
                "seed": seed,
                "label": spec.label,
                "product": _measurement(gain),
                "known_response": _measurement(3.0),
                "final_attacks": [
                    {"replicate": replicate, "measurement": _measurement(0.0)}
                    for replicate in range(step34.ATTACK_REPLICATES)
                ],
            })
    hashes = {str(path): digest for path, digest in step34.FROZEN_ARTIFACTS.items()}
    controls = {"passed": True}
    report = step34.finalize_report(
        artifact_hashes=hashes, small_game_control=controls,
        parity_control=controls, performance_control=controls,
        observation=observation, calibration_rows=calibration, case_rows=cases,
    )
    assert report["all_pass"] is True
    assert all(report["gates"].values())

    incomplete = step34.finalize_report(
        artifact_hashes=hashes, small_game_control=controls,
        parity_control=controls, performance_control=controls,
        observation=observation, calibration_rows=calibration,
        case_rows=cases[:-1],
    )
    assert incomplete["all_pass"] is False
    assert incomplete["gates"]["artifact/completeness"] is False
