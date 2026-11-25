from __future__ import annotations

import pytest

from parciver.rmapss import step32


def test_locked_constants_and_population_are_complete():
    assert step32.PROTOCOL == "s3.2-full-v1"
    assert step32.SEEDS == (103, 211, 307)
    assert step32.ALPHA_GRID == (1.0, 0.70, 0.40, 0.20, 0.10, 0.0)
    assert len(step32.FIT_SPEC) == 3
    assert len(step32.HELD_SPEC) == 9
    assert len(step32.NULL_SPEC) == 3
    assert len({spec.label for spec in step32.POPULATION_SPEC}) == 15
    assert step32.OBS_FIT_HANDS + step32.OBS_VALIDATION_HANDS == 250_000
    assert step32.EVAL_HANDS // step32.BLOCK_SIZE == step32.BLOCKS == 200
    assert "not safety" in step32.frozen_metadata()["safety_language"]


def test_holm_adjust_is_monotone_in_sorted_order_and_never_shrinks_p():
    p = [0.03, 0.001, 0.2, 0.01]
    adjusted = step32.holm_adjust(p)
    assert all(a >= b for a, b in zip(adjusted, p))
    ranked = sorted(range(len(p)), key=p.__getitem__)
    assert [adjusted[i] for i in ranked] == sorted(adjusted[i] for i in ranked)


def test_block_estimate_detects_positive_and_exact_zero():
    positive = step32.block_estimate([1.0 + (i % 3) * 0.01 for i in range(200)],
                                     family_size=27)
    assert positive.lower > 0.0
    assert positive.raw_p_greater < 0.01 / 27

    zero = step32.block_estimate([0.0] * 200, family_size=6)
    assert zero.mean == zero.lower == zero.upper == 0.0
    assert zero.raw_p_greater == 1.0
    assert zero.raw_p_less == 1.0


@pytest.mark.parametrize("blocks,family", [([], 1), ([1.0], 1), ([1.0, float("nan")], 1), ([1, 2], 0)])
def test_block_estimate_rejects_invalid_inputs(blocks, family):
    with pytest.raises(ValueError):
        step32.block_estimate(blocks, family_size=family)


def test_response_seed_streams_do_not_overlap():
    streams = {
        step32.response_seed(seed, responder, replicate)
        for seed in step32.SEEDS
        for responder in (0, 1)
        for replicate in range(step32.ATTACK_REPLICATES)
    }
    assert len(streams) == len(step32.SEEDS) * 2 * step32.ATTACK_REPLICATES


def test_all_sampling_stage_streams_are_disjoint():
    streams = {
        step32.stream_seed(seed, stage, case_index=case, alpha_index=alpha,
                           replicate=replicate, responder=responder)
        for seed in step32.SEEDS
        for stage in step32.frozen_metadata()["stream_stages"]
        for case in range(len(step32.POPULATION_SPEC))
        for alpha in range(len(step32.ALPHA_GRID))
        for replicate in range(step32.ATTACK_REPLICATES)
        for responder in (0, 1)
    }
    expected = (len(step32.SEEDS) * len(step32.frozen_metadata()["stream_stages"])
                * len(step32.POPULATION_SPEC) * len(step32.ALPHA_GRID)
                * step32.ATTACK_REPLICATES * 2)
    assert len(streams) == expected


def _measurement(gain: float) -> dict:
    return {
        "hands": step32.EVAL_HANDS,
        "block_size": step32.BLOCK_SIZE,
        "candidate_bb_per_100": gain,
        "anchor_bb_per_100": 0.0,
        "gain_bb_per_100": gain,
        "difference_stddev_bb_per_hand": 0.0,
        "block_gains_bb_per_100": [gain] * step32.BLOCKS,
    }


def _synthetic_report_rows():
    calibrations = [
        {"seed": seed, "label": spec.label, "measurement": _measurement(10.0)}
        for seed in step32.SEEDS for spec in step32.FIT_SPEC
    ]
    cases = []
    for seed in step32.SEEDS:
        for spec in step32.POPULATION_SPEC:
            gain = 2.0 if spec in step32.HELD_SPEC else 0.0
            if spec in step32.FIT_SPEC:
                gain = 1.0
            cases.append({
                "protocol": step32.PROTOCOL,
                "seed": seed,
                "label": spec.label,
                "model": {
                    "coverage": 0.99,
                    "model_log_loss": 0.1,
                    "reference_log_loss": 0.2,
                },
                "public_only_keys": True,
                "product": _measurement(gain),
                "known_response": _measurement(3.0),
                "final_attacks": [
                    {"replicate": replicate, "measurement": _measurement(0.0)}
                    for replicate in range(step32.ATTACK_REPLICATES)
                ],
            })
    return calibrations, cases


def test_finalize_report_requires_every_locked_case_and_rederives_all_gates():
    calibrations, cases = _synthetic_report_rows()
    hashes = {str(path): digest for path, digest in step32.FROZEN_ARTIFACTS.items()}
    report = step32.finalize_report(
        artifact_hashes=hashes,
        small_game_control={"passed": True},
        calibration_rows=calibrations,
        case_rows=cases,
    )
    assert report["all_pass"] is True
    assert all(report["gates"].values())
    assert all(row["scale_S"] == 10.0 for row in report["thresholds"].values())

    incomplete = step32.finalize_report(
        artifact_hashes=hashes,
        small_game_control={"passed": True},
        calibration_rows=calibrations,
        case_rows=cases[:-1],
    )
    assert incomplete["all_pass"] is False
    assert incomplete["gates"]["artifact/completeness"] is False


def test_known_response_sanity_rejects_significant_super_response():
    calibrations, cases = _synthetic_report_rows()
    held = next(row for row in cases if row["label"] == step32.HELD_SPEC[0].label)
    held["product"] = _measurement(4.0)
    hashes = {str(path): digest for path, digest in step32.FROZEN_ARTIFACTS.items()}
    report = step32.finalize_report(
        artifact_hashes=hashes,
        small_game_control={"passed": True},
        calibration_rows=calibrations,
        case_rows=cases,
    )
    assert report["gates"]["known-response sanity"] is False
    assert report["all_pass"] is False
