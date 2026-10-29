"""Synthetic uniqueness checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL
from p12_bank_liquidity.candidate_a.uniqueness import (
    EligibleTarget,
    decide_uniqueness,
    summarize_arm,
)


def _row(**overrides: object) -> EligibleTarget:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "target_timestamp": "1977-03-31",
        "origin_timestamp": "1976-12-31",
        "observed": True,
        "synthetic_grid_row": False,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return EligibleTarget(**values)  # type: ignore[arg-type]


def test_duplicate_entity_target_is_detected_on_each_arm() -> None:
    arm_a = [
        _row(origin_timestamp="1976-12-30"),
        _row(origin_timestamp="1976-12-31"),
    ]
    arm_b = [
        _row(entity_id="other", target_timestamp="1977-06-30", origin_timestamp="1977-03-31"),
        _row(entity_id="other", target_timestamp="1977-06-30", origin_timestamp="1977-03-30"),
    ]
    decision = decide_uniqueness(arm_a, arm_b, inherited_common_keys=0)
    assert decision.arm_a.duplicated_keys == 1
    assert decision.arm_a.max_multiplicity == 2
    assert decision.arm_b.duplicated_keys == 1
    assert decision.status == "BLOCKED_DUPLICATE_ENTITY_TARGET_KEYS"
    assert decision.arm_a.eligible_rows == 2
    assert decision.arm_a.distinct_keys == 1


def test_duplicate_detection_uses_the_target_not_origin_equality() -> None:
    same_origin = summarize_arm(
        [
            _row(origin_timestamp="1976-12-31"),
            _row(origin_timestamp="1976-12-31"),
        ]
    )
    different_origin = summarize_arm(
        [
            _row(origin_timestamp="1976-12-30"),
            _row(origin_timestamp="1976-12-31"),
        ]
    )
    assert same_origin.duplicated_keys == 1
    assert different_origin.duplicated_keys == 1
    assert different_origin.distinct_keys == 1


def test_different_origins_stay_in_the_primary_key() -> None:
    arm_a = [_row(origin_timestamp="1976-12-30")]
    arm_b = [_row(origin_timestamp="1976-12-31")]
    decision = decide_uniqueness(arm_a, arm_b, inherited_common_keys=1)
    assert decision.status == "VERIFIED_FOR_AUDITED_ELIGIBLE_KEYS"
    assert decision.common_unique_keys == 1
    assert decision.common_keys_with_different_origins == 1
    assert decision.arm_a.duplicated_keys == 0


def test_synthetic_grid_rows_are_not_eligible_targets() -> None:
    rows = [
        _row(observed=True, synthetic_grid_row=True, target_timestamp="1977-03-30"),
        _row(observed=False, synthetic_grid_row=False, target_timestamp="1977-06-30"),
    ]
    summary = summarize_arm(rows)
    assert summary.eligible_rows == 0
    assert summary.distinct_keys == 0
    assert summary.max_multiplicity == 0


def test_inherited_count_mismatch_does_not_replace_the_inherited_value() -> None:
    arm_a = [_row(entity_id="a", target_timestamp="1977-03-31")]
    arm_b = [_row(entity_id="a", target_timestamp="1977-03-31")]
    decision = decide_uniqueness(arm_a, arm_b, inherited_common_keys=3)
    assert decision.status == "BLOCKED_INHERITED_COUNT_MISMATCH"
    assert decision.common_unique_keys == 1
    assert decision.inherited_common_keys == 3
    assert decision.difference_from_inherited_common_keys == -2
    assert decision.evidence_label == EVIDENCE_LABEL
