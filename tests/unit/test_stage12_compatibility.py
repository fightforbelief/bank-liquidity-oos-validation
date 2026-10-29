"""Synthetic compatibility checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from datetime import date

import pytest

from p12_bank_liquidity.candidate_a.alignment import (
    DuplicateEntityTarget,
    TargetRecord,
    align_observed_targets,
)
from p12_bank_liquidity.candidate_a.bootstrap import (
    CONFIDENCE_LEVEL,
    INTERVAL,
    REPLICATES,
    SEED,
)
from p12_bank_liquidity.candidate_a.execution_policy import (
    ModelAttempt,
    classify_model_attempt,
)
from p12_bank_liquidity.candidate_a.grid_arms import (
    NonQuarterEndSourceDate,
    SourceRow,
    build_arm_b,
)
from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL, seasonal_mase


def _target(**overrides: object) -> TargetRecord:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "origin_timestamp": "1976-09-30",
        "target_timestamp": "1976-12-31",
        "observed": True,
        "synthetic_grid_date": False,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return TargetRecord(**values)  # type: ignore[arg-type]


def test_duplicate_entity_target_raises_a_clear_error() -> None:
    arm_a = [
        _target(entity_id="fixture-entity", target_timestamp="1977-03-31"),
        _target(
            entity_id="fixture-entity",
            origin_timestamp="1976-12-31",
            target_timestamp="1977-03-31",
        ),
    ]
    arm_b = [_target(entity_id="fixture-entity", target_timestamp="1977-03-31")]
    with pytest.raises(DuplicateEntityTarget, match="fixture-entity 1977-03-31"):
        align_observed_targets(arm_a, arm_b)


def test_primary_and_secondary_stay_distinct_and_keep_both_origins() -> None:
    arm_a = [
        _target(entity_id="shared", origin_timestamp="1976-12-30", target_timestamp="1977-03-31")
    ]
    arm_b = [
        _target(entity_id="shared", origin_timestamp="1976-12-31", target_timestamp="1977-03-31")
    ]
    result = align_observed_targets(arm_a, arm_b)
    assert len(result.primary_pairs) == 1
    assert result.secondary_pairs == ()
    pair = result.primary_pairs[0]
    assert pair.origin_timestamp_arm_a == "1976-12-30"
    assert pair.origin_timestamp_arm_b == "1976-12-31"
    assert pair.evidence_label == EVIDENCE_LABEL


def test_arm_b_keeps_quarter_ends_through_a_leap_year() -> None:
    rows = [
        SourceRow("fixture-entity", date(2019, 12, 31), 1.0, "q4"),
        SourceRow("fixture-entity", date(2020, 3, 31), 2.0, "q1"),
        SourceRow("fixture-entity", date(2020, 6, 30), 3.0, "q2"),
        SourceRow("fixture-entity", date(2020, 12, 31), 4.0, "q4b"),
    ]
    grid = build_arm_b(rows)
    by_day = {row.grid_timestamp: row for row in grid}
    assert set(by_day) == {
        date(2019, 12, 31),
        date(2020, 3, 31),
        date(2020, 6, 30),
        date(2020, 9, 30),
        date(2020, 12, 31),
    }
    assert date(2020, 2, 29) not in by_day
    september = by_day[date(2020, 9, 30)]
    assert september.is_synthetic_grid_row is True
    assert september.observed_target is False
    assert september.target_value is None
    assert september.source_reporting_date is None
    for day in (date(2020, 3, 31), date(2020, 6, 30), date(2020, 12, 31)):
        observed = by_day[day]
        assert observed.is_synthetic_grid_row is False
        assert observed.observed_target is True
        assert observed.source_reporting_date == observed.grid_timestamp
    assert all(row.evidence_label == EVIDENCE_LABEL for row in grid)


def test_arm_b_refuses_to_move_a_leap_day() -> None:
    with pytest.raises(NonQuarterEndSourceDate):
        build_arm_b([SourceRow("fixture-entity", date(2020, 2, 29), 1.0, "leap")])


def test_failed_model_does_not_keep_a_fallback_forecast() -> None:
    failed = classify_model_attempt(
        ModelAttempt(
            model_id="AutoARIMA",
            forecast_value=None,
            error="fit failed",
            fallback_forecast_value=99.0,
            evidence_label=EVIDENCE_LABEL,
        )
    )
    assert failed.status == "EXCLUDED"
    assert failed.reason == "model_failed"
    assert failed.forecast_value is None
    missing = classify_model_attempt(
        ModelAttempt(
            model_id="Naive",
            forecast_value=None,
            error=None,
            fallback_forecast_value=7.0,
            evidence_label=EVIDENCE_LABEL,
        )
    )
    assert missing.status == "EXCLUDED"
    assert missing.reason == "forecast_not_finite"
    assert missing.forecast_value is None
    kept = classify_model_attempt(
        ModelAttempt(
            model_id="HistoricAverage",
            forecast_value=1.25,
            error=None,
            fallback_forecast_value=99.0,
            evidence_label=EVIDENCE_LABEL,
        )
    )
    assert kept.status == "RECORDED_FORECAST"
    assert kept.forecast_value == 1.25


def test_mase_and_bootstrap_rules_remain_the_stage11_rules() -> None:
    undefined = seasonal_mase([3, 3, 3, 3, 3, 3, 3, 3], [3], [4])
    assert undefined.status == "UNDEFINED"
    assert undefined.value is None
    assert undefined.reason == "zero_denominator"
    assert REPLICATES == 2000
    assert SEED == 20261027
    assert CONFIDENCE_LEVEL == 0.95
    assert INTERVAL == "PERCENTILE"
