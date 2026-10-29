"""Synthetic adapter checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from datetime import date

import pytest

from p12_bank_liquidity.candidate_a.grid_arms import SourceRow, build_arm_b
from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL
from p12_bank_liquidity.candidate_a.timestamp_adapter import (
    PANEL_WINDOW_CAP,
    REAL_MODEL_OUTPUT_BINDING,
    DuplicateEvaluationKey,
    HistoryPoint,
    ScheduleRow,
    primary_pairs_keep_separate_origins,
    run_one_step,
    training_values,
)


class _Fake:
    model_id = "FakeForecaster"

    def __init__(self, value: float = 1.5, error: bool = False) -> None:
        self.value = value
        self.error = error
        self.seen: list[list[float]] = []

    def predict_one(self, y_train: list[float]) -> float:
        self.seen.append(list(y_train))
        if self.error:
            raise RuntimeError("fixture model failure")
        return self.value


def _schedule(**overrides: object) -> ScheduleRow:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "arm_id": "B",
        "origin_timestamp": "2020-06-30",
        "target_timestamp": "2020-09-30",
        "target_quarter": "2020Q3",
        "observed_finite_source_target": True,
        "synthetic_grid_row": False,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return ScheduleRow(**values)  # type: ignore[arg-type]


def _history(timestamp: str, value: float | None, **overrides: object) -> HistoryPoint:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "arm_id": "B",
        "timestamp": timestamp,
        "value": value,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return HistoryPoint(**values)  # type: ignore[arg-type]


def test_leap_year_quarter_ends_are_not_replaced() -> None:
    rows = [
        SourceRow("fixture-entity", date(2019, 12, 31), 1.0, "a"),
        SourceRow("fixture-entity", date(2020, 3, 31), 2.0, "b"),
        SourceRow("fixture-entity", date(2020, 6, 30), 3.0, "c"),
        SourceRow("fixture-entity", date(2020, 9, 30), 4.0, "d"),
        SourceRow("fixture-entity", date(2020, 12, 31), 5.0, "e"),
    ]
    grid = build_arm_b(rows)
    targets = [row.grid_timestamp.isoformat() for row in grid]
    assert "2020-02-29" not in targets
    assert targets == [
        "2019-12-31",
        "2020-03-31",
        "2020-06-30",
        "2020-09-30",
        "2020-12-31",
    ]
    schedule = [
        _schedule(origin_timestamp="2020-06-30", target_timestamp=target, target_quarter="kept")
        for target in ("2020-09-30", "2020-12-31")
    ]
    fake = _Fake(2.0)
    results = run_one_step(schedule, [], fake)
    assert [item.target_timestamp for item in results] == ["2020-09-30", "2020-12-31"]
    assert "2020-02-29" not in {item.target_timestamp for item in results}
    assert all(item.prediction == 2.0 for item in results)


def test_supplied_target_is_not_derived_from_a_frequency_alias() -> None:
    assert "freq" not in run_one_step.__code__.co_varnames
    result = run_one_step(
        [_schedule(target_timestamp="2020-09-30")],
        [_history("2020-06-30", 1.0)],
        _Fake(4.0),
    )[0]
    assert result.target_timestamp == "2020-09-30"
    assert result.prediction == 4.0


def test_different_origins_remain_outside_the_secondary_set() -> None:
    arm_a = [_schedule(arm_id="A", origin_timestamp="2020-06-30", target_timestamp="2020-09-30")]
    arm_b = [_schedule(arm_id="B", origin_timestamp="2020-06-29", target_timestamp="2020-09-30")]
    paired = primary_pairs_keep_separate_origins(arm_a, arm_b)
    assert paired[0]["origin_timestamp_arm_a"] == "2020-06-30"
    assert paired[0]["origin_timestamp_arm_b"] == "2020-06-29"
    assert paired[0]["origin_timestamp_equal"] is False
    assert paired[0]["in_secondary_exact_timestamp_set"] is False


def test_synthetic_and_missing_targets_are_excluded() -> None:
    history = [_history("2020-06-30", 1.0)]
    synthetic = run_one_step(
        [_schedule(synthetic_grid_row=True, observed_finite_source_target=False)],
        history,
        _Fake(),
    )[0]
    missing = run_one_step(
        [_schedule(target_timestamp="2020-12-31", observed_finite_source_target=False)],
        history,
        _Fake(),
    )[0]
    assert synthetic.reason == "synthetic_grid_row"
    assert synthetic.prediction is None
    assert missing.reason == "target_not_observed"
    assert missing.prediction is None


def test_training_stops_at_the_origin_and_excludes_the_target() -> None:
    row = _schedule(origin_timestamp="2020-06-30", target_timestamp="2020-09-30")
    history = [
        _history("2020-03-31", 1.0),
        _history("2020-06-30", 2.0),
        _history("2020-07-15", 9.0),
        _history("2020-09-30", 8.0),
    ]
    assert training_values(history, row) == [1.0, 2.0]
    fake = _Fake(3.0)
    result = run_one_step([row], history, fake)[0]
    assert fake.seen == [[1.0, 2.0]]
    assert result.target_timestamp == "2020-09-30"


def test_model_failure_and_nonfinite_forecast_are_exclusions() -> None:
    row = _schedule()
    failed = run_one_step([row], [], _Fake(error=True))[0]
    assert failed.reason == "model_failed"
    assert failed.prediction is None
    assert failed.model_id == "FakeForecaster"
    invalid = run_one_step([row], [], _Fake(value=float("nan")))[0]
    assert invalid.reason == "forecast_not_finite"
    assert invalid.prediction is None


def test_duplicate_schedule_keys_raise() -> None:
    row = _schedule()
    with pytest.raises(DuplicateEvaluationKey, match="fixture-entity B 2020-09-30"):
        run_one_step([row, row], [], _Fake())


def test_schedule_is_not_expanded_to_six_windows_per_entity() -> None:
    schedule = [
        _schedule(target_timestamp="2020-09-30"),
        _schedule(origin_timestamp="2020-09-30", target_timestamp="2020-12-31", target_quarter="2020Q4"),
    ]
    results = run_one_step(schedule, [], _Fake(1.0), panel_window_cap=PANEL_WINDOW_CAP)
    assert len(results) == 2
    assert [item.target_timestamp for item in results] == ["2020-09-30", "2020-12-31"]
    assert REAL_MODEL_OUTPUT_BINDING == "BLOCKED_PENDING_REAL_API_SMOKE_TEST"
