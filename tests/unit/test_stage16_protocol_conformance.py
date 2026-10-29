"""Stage 16 conformance fixtures. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from __future__ import annotations

import pytest

from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL, seasonal_mase
from p12_bank_liquidity.candidate_a.timestamp_adapter import (
    HistoryPoint,
    ScheduleRow,
    UnimputedTrainingGap,
    primary_pairs_keep_separate_origins,
    run_one_step,
    training_values,
)


def _row(**overrides: object) -> ScheduleRow:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "arm_id": "A",
        "origin_timestamp": "2020-09-30",
        "target_timestamp": "2020-12-31",
        "target_quarter": "2020Q4",
        "observed_finite_source_target": True,
        "synthetic_grid_row": False,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return ScheduleRow(**values)  # type: ignore[arg-type]


def _point(
    timestamp: str,
    value: float | None,
    y_imputed: float | None,
    *,
    arm_id: str = "A",
    synthetic: bool = False,
) -> HistoryPoint:
    return HistoryPoint(
        entity_id="fixture-entity",
        arm_id=arm_id,
        timestamp=timestamp,
        value=value,
        evidence_label=EVIDENCE_LABEL,
        y_imputed=y_imputed,
        imputation_applied=True,
        synthetic_grid_row=synthetic,
    )


class _Echo:
    model_id = "HistoricAverage"

    def __init__(self) -> None:
        self.seen: list[list[float]] = []

    def predict_one(self, y_train: list[float]) -> float:
        self.seen.append(list(y_train))
        return 1.0


def _grid() -> list[HistoryPoint]:
    # Frozen forward fill is applied by the fixture, not by the adapter.
    # Original interior gap at 2020-06-30 becomes the previous finite value.
    return [
        _point("2020-03-31", 1.0, 1.0),
        _point("2020-06-30", None, 1.0, synthetic=True),
        _point("2020-09-30", 3.0, 3.0),
        _point("2020-12-31", 4.0, 4.0),
    ]


def test_imputed_regular_grid_keeps_interior_position() -> None:
    row = _row()
    assert training_values(_grid(), row) == [1.0, 1.0, 3.0]
    echo = _Echo()
    result = run_one_step([row], _grid(), echo)[0]
    assert echo.seen == [[1.0, 1.0, 3.0]]
    assert result.target_timestamp == "2020-12-31"
    assert result.prediction == 1.0


def test_unimputed_interior_gap_is_not_compressed() -> None:
    history = [
        HistoryPoint("fixture-entity", "A", "2020-03-31", 1.0, EVIDENCE_LABEL),
        HistoryPoint("fixture-entity", "A", "2020-06-30", None, EVIDENCE_LABEL, synthetic_grid_row=True),
        HistoryPoint("fixture-entity", "A", "2020-09-30", 3.0, EVIDENCE_LABEL),
    ]
    with pytest.raises(UnimputedTrainingGap):
        training_values(history, _row())


def test_synthetic_filled_row_is_not_an_observed_target() -> None:
    synthetic_target = _row(
        origin_timestamp="2020-03-31",
        target_timestamp="2020-06-30",
        target_quarter="2020Q2",
        observed_finite_source_target=False,
        synthetic_grid_row=True,
    )
    result = run_one_step([synthetic_target], _grid(), _Echo())[0]
    assert result.reason == "synthetic_grid_row"
    assert result.prediction is None


def test_mase_denominator_uses_only_the_prepared_training_grid() -> None:
    prepared = training_values(
        [
            _point("2019-03-31", 1.0, 1.0),
            _point("2019-06-30", 2.0, 2.0),
            _point("2019-09-30", 3.0, 3.0),
            _point("2019-12-31", 4.0, 4.0),
            _point("2020-03-31", 2.0, 2.0),
            _point("2020-06-30", None, 2.0, synthetic=True),
            _point("2020-09-30", 5.0, 5.0),
            _point("2020-12-31", 9.0, 9.0),
        ],
        _row(origin_timestamp="2020-09-30", target_timestamp="2020-12-31"),
    )
    assert prepared == [1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 5.0]
    assert 9.0 not in prepared
    defined = seasonal_mase(prepared, [6.0], [5.0])
    assert defined.status == "DEFINED"
    assert defined.seasonality == 4
    undefined = seasonal_mase([1.0, 1.0, 1.0, 1.0, 1.0], [1.0], [1.0])
    assert undefined.status == "UNDEFINED"
    assert undefined.reason == "zero_denominator"


def test_same_origin_quarter_with_different_origin_dates_stays_primary() -> None:
    arm_a = _row(arm_id="A", origin_timestamp="2020-03-30", target_timestamp="2020-06-30", target_quarter="2020Q2")
    arm_b = _row(arm_id="B", origin_timestamp="2020-03-31", target_timestamp="2020-06-30", target_quarter="2020Q2")
    paired = primary_pairs_keep_separate_origins([arm_a], [arm_b])[0]
    assert paired["origin_timestamp_arm_a"] == "2020-03-30"
    assert paired["origin_timestamp_arm_b"] == "2020-03-31"
    assert paired["origin_timestamp_equal"] is False
    assert paired["origin_quarter_equal"] is True
    assert paired["in_secondary_exact_timestamp_set"] is False
    assert paired["target_timestamp"] == "2020-06-30"
    history = [
        _point("2019-12-31", 2.0, 2.0, arm_id="A"),
        _point("2020-03-30", None, 2.0, arm_id="A", synthetic=True),
        _point("2020-06-30", 4.0, 4.0, arm_id="A"),
        _point("2019-12-31", 8.0, 8.0, arm_id="B"),
        _point("2020-03-31", 9.0, 9.0, arm_id="B"),
        _point("2020-06-30", 7.0, 7.0, arm_id="B"),
    ]
    echo_a = _Echo()
    echo_b = _Echo()
    scored_a = run_one_step([arm_a], history, echo_a)[0]
    scored_b = run_one_step([arm_b], history, echo_b)[0]
    assert scored_a.target_timestamp == "2020-06-30"
    assert scored_b.target_timestamp == "2020-06-30"
    assert echo_a.seen == [[2.0, 2.0]]
    assert echo_b.seen == [[8.0, 9.0]]
    origin_as_target = run_one_step(
        [
            _row(
                origin_timestamp="2019-12-31",
                target_timestamp="2020-03-30",
                target_quarter="2020Q1",
                observed_finite_source_target=False,
                synthetic_grid_row=True,
            )
        ],
        history,
        _Echo(),
    )[0]
    assert origin_as_target.reason == "synthetic_grid_row"
    assert origin_as_target.prediction is None


def test_duplicate_and_failed_model_rules_remain() -> None:
    row = _row()
    with pytest.raises(Exception, match="fixture-entity A 2020-12-31"):
        run_one_step([row, row], _grid(), _Echo())

    class _Fail:
        model_id = "AutoARIMA"

        def predict_one(self, y_train: list[float]) -> float:
            raise RuntimeError("synthetic failure")

    class _Nan:
        model_id = "Naive"

        def predict_one(self, y_train: list[float]) -> float:
            return float("nan")

    failed = run_one_step([row], _grid(), _Fail())[0]
    invalid = run_one_step([row], _grid(), _Nan())[0]
    assert failed.model_id == "AutoARIMA"
    assert failed.reason == "model_failed"
    assert failed.prediction is None
    assert invalid.reason == "forecast_not_finite"
    assert invalid.prediction is None
