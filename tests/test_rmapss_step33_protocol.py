from __future__ import annotations

from copy import deepcopy

from parciver.rmapss import step33


def _score_block() -> dict:
    decisions = 100
    return {
        "decisions": decisions,
        "effective_weight_sum": 99.5,
        "zero_weight_decisions": 0,
        "legacy_supported_decisions": 95,
        "model_nll_sum": 10.0,
        "reference_nll_sum": 20.0,
        "legacy_nll_sum": 10.1,
        "sparse_decisions": 5,
        "sparse_model_nll_sum": 0.5,
        "sparse_reference_nll_sum": 1.0,
        "model_brier_sum": 5.0,
        "reference_brier_sum": 6.0,
        "deviation_l1_sum": 30.0,
        "ece_count": [0, 0, 0, 0, 0, 100, 0, 0, 0, 0],
        "ece_confidence_sum": [0, 0, 0, 0, 0, 50, 0, 0, 0, 0],
        "ece_accuracy_sum": [0, 0, 0, 0, 0, 50, 0, 0, 0, 0],
    }


def _observation_rows() -> list[dict]:
    rows = []
    for seed in step33.SEEDS:
        for spec in step33.HELD_SPEC:
            blocks = [_score_block() for _ in range(step33.CONFIRM_BLOCKS)]
            rows.append({
                "protocol": step33.PROTOCOL,
                "seed": seed,
                "label": spec.label,
                "public_only_keys": True,
                "model": step33.summarize_model_blocks(blocks),
            })
    return rows


def test_locked_constants_and_seed_registry():
    assert step33.PROTOCOL == "s3.3-public-recovery-v1"
    assert step33.SEEDS == (311, 313, 317)
    assert (step33.EXACT_STRENGTH, step33.BACKOFF_STRENGTH,
            step33.CONTEXT_STRENGTH) == (200.0, 50.0, 20.0)
    assert step33.EFFECTIVE_COVERAGE == 0.99
    assert step33.OBSERVATION_FAMILY_SIZE == 27
    assert step33.MAX_WORKERS == 16


def test_all_stream_coordinates_are_disjoint():
    streams = {
        step33.stream_seed(seed, stage, case_index=case, alpha_index=alpha,
                           replicate=replicate, responder=responder, block=block)
        for seed in step33.SEEDS
        for stage in step33.frozen_metadata()["stream_stages"]
        for case in range(len(step33.POPULATION_SPEC))
        for alpha in range(len(step33.ALPHA_GRID))
        for replicate in range(step33.ATTACK_REPLICATES)
        for responder in (0, 1)
        for block in (0, 199)
    }
    expected = (len(step33.SEEDS) * len(step33.frozen_metadata()["stream_stages"])
                * len(step33.POPULATION_SPEC) * len(step33.ALPHA_GRID)
                * step33.ATTACK_REPLICATES * 2 * 2)
    assert len(streams) == expected


def test_model_block_summary_and_complete_observation_report_pass():
    rows = _observation_rows()
    report = step33.observation_report(rows)
    assert report["all_pass"] is True
    assert all(report["gates"].values())
    assert rows[0]["model"]["effective_coverage"] == 0.995
    assert rows[0]["model"]["ece"] == 0.0


def test_observation_report_fails_closed_on_coverage_or_missing_case():
    rows = _observation_rows()
    failed = deepcopy(rows)
    failed[0]["model"]["effective_coverage"] = 0.989
    report = step33.observation_report(failed)
    assert report["gates"]["continuous coverage"] is False
    assert report["all_pass"] is False

    incomplete = step33.observation_report(rows[:-1])
    assert incomplete["complete"] is False
    assert incomplete["all_pass"] is False


def test_confirmation_score_requires_all_fixed_blocks():
    try:
        step33.summarize_model_blocks([_score_block()] * 49)
    except ValueError as exc:
        assert "exactly 50" in str(exc)
    else:
        raise AssertionError("incomplete confirmation blocks must fail")


def _measurement(gain: float) -> dict:
    return {
        "hands": step33.EVAL_HANDS,
        "block_size": step33.BLOCK_SIZE,
        "candidate_bb_per_100": gain,
        "anchor_bb_per_100": 0.0,
        "gain_bb_per_100": gain,
        "difference_stddev_bb_per_hand": 0.0,
        "block_gains_bb_per_100": [gain] * step33.BLOCKS,
    }


def test_final_report_rederives_complete_downstream_gate_family():
    observation = step33.observation_report(_observation_rows())
    calibration = [
        {"seed": seed, "label": spec.label, "measurement": _measurement(10.0)}
        for seed in step33.SEEDS for spec in step33.FIT_SPEC
    ]
    cases = []
    for seed in step33.SEEDS:
        for spec in step33.POPULATION_SPEC:
            gain = 2.0 if spec in step33.HELD_SPEC else (0.0 if spec in step33.NULL_SPEC else 1.0)
            cases.append({
                "protocol": step33.PROTOCOL,
                "seed": seed,
                "label": spec.label,
                "product": _measurement(gain),
                "known_response": _measurement(3.0),
                "final_attacks": [
                    {"replicate": replicate, "measurement": _measurement(0.0)}
                    for replicate in range(step33.ATTACK_REPLICATES)
                ],
            })
    hashes = {str(path): digest for path, digest in step33.FROZEN_ARTIFACTS.items()}
    report = step33.finalize_report(
        artifact_hashes=hashes,
        small_game_control={"passed": True}, parity_control={"passed": True},
        observation=observation, calibration_rows=calibration, case_rows=cases,
    )
    assert report["all_pass"] is True
    assert all(report["gates"].values())

    incomplete = step33.finalize_report(
        artifact_hashes=hashes,
        small_game_control={"passed": True}, parity_control={"passed": True},
        observation=observation, calibration_rows=calibration, case_rows=cases[:-1],
    )
    assert incomplete["all_pass"] is False
    assert incomplete["gates"]["artifact/completeness"] is False
