"""Stage 15 bridge checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE. No StatsForecast import."""

from __future__ import annotations

import numpy as np
import pytest

from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL
from p12_bank_liquidity.candidate_a.model_output import UnexpectedModelOutput, extract_one_step_scalar
from p12_bank_liquidity.candidate_a.statsforecast_bridge import StatsForecastOneStep
from p12_bank_liquidity.candidate_a.timestamp_adapter import (
    PANEL_WINDOW_CAP,
    HistoryPoint,
    ScheduleRow,
    primary_pairs_keep_separate_origins,
    run_one_step,
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


def _point(timestamp: str, value: float, arm_id: str = "A") -> HistoryPoint:
    return HistoryPoint(entity_id="fixture-entity", arm_id=arm_id, timestamp=timestamp, value=value, evidence_label=EVIDENCE_LABEL)


class _Mean:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.shape = (len(values),)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


class _Model:
    def __init__(self, output: object | None = None, error: bool = False) -> None:
        self.output = output if output is not None else {"mean": _Mean([4.0])}
        self.error = error
        self.seen: list[list[float]] = []

    def forecast(self, y: object, h: int) -> object:
        self.seen.append([float(item) for item in list(y)])  # type: ignore[arg-type]
        if h != 1:
            raise AssertionError(h)
        if self.error:
            raise RuntimeError("synthetic model failure")
        return self.output


def test_extraction_accepts_only_the_observed_shape() -> None:
    assert extract_one_step_scalar({"mean": _Mean([1.25])}) == 1.25
    with pytest.raises(UnexpectedModelOutput):
        extract_one_step_scalar({"other": _Mean([1.0])})
    with pytest.raises(UnexpectedModelOutput):
        extract_one_step_scalar({"mean": _Mean([1.0, 2.0])})
    with pytest.raises(UnexpectedModelOutput):
        extract_one_step_scalar(1.0)


def test_bridge_keeps_supplied_timestamp_and_training_cut() -> None:
    model = _Model()
    bridge = StatsForecastOneStep(model, "HistoricAverage")
    history = [_point("2020-06-30", 1.0), _point("2020-09-30", 2.0), _point("2020-12-31", 9.0)]
    result = run_one_step([_row()], history, bridge)[0]
    assert result.target_timestamp == "2020-12-31"
    assert result.origin_timestamp == "2020-09-30"
    assert result.status == "RECORDED_FORECAST"
    assert result.prediction == 4.0
    assert model.seen == [[1.0, 2.0]]


def test_failure_nonfinite_and_bad_shape_stay_distinct() -> None:
    history = [_point("2020-09-30", 1.0)]
    failed = run_one_step([_row()], history, StatsForecastOneStep(_Model(error=True), "AutoARIMA"))[0]
    assert failed.reason == "model_failed"
    assert failed.prediction is None
    assert failed.model_id == "AutoARIMA"
    nonfinite = run_one_step(
        [_row()],
        history,
        StatsForecastOneStep(_Model({"mean": np.array([np.nan])}), "Naive"),
    )[0]
    assert nonfinite.reason == "forecast_not_finite"
    assert nonfinite.prediction is None
    bad_shape = run_one_step(
        [_row()],
        history,
        StatsForecastOneStep(_Model({"mean": np.array([1.0, 2.0])}), "AutoTheta"),
    )[0]
    assert bad_shape.reason == "unexpected_output_shape"
    assert bad_shape.prediction is None


def test_schedule_rules_are_unchanged_by_the_bridge() -> None:
    synthetic = run_one_step(
        [_row(synthetic_grid_row=True, observed_finite_source_target=False)],
        [],
        StatsForecastOneStep(_Model(), "Naive"),
    )[0]
    missing = run_one_step(
        [_row(target_timestamp="2020-09-30", observed_finite_source_target=False)],
        [],
        StatsForecastOneStep(_Model(), "Naive"),
    )[0]
    assert synthetic.reason == "synthetic_grid_row"
    assert synthetic.prediction is None
    assert missing.reason == "target_not_observed"
    paired = primary_pairs_keep_separate_origins(
        [_row(arm_id="A", origin_timestamp="2020-09-30")],
        [_row(arm_id="B", origin_timestamp="2020-06-30")],
    )
    assert paired[0]["origin_timestamp_arm_a"] == "2020-09-30"
    assert paired[0]["origin_timestamp_arm_b"] == "2020-06-30"
    assert paired[0]["in_secondary_exact_timestamp_set"] is False
    with pytest.raises(Exception, match="fixture-entity A 2020-12-31"):
        run_one_step([_row(), _row()], [], StatsForecastOneStep(_Model(), "Naive"))
    results = run_one_step(
        [_row(), _row(origin_timestamp="2020-12-31", target_timestamp="2021-03-31", target_quarter="2021Q1")],
        [],
        StatsForecastOneStep(_Model(), "Naive"),
        panel_window_cap=PANEL_WINDOW_CAP,
    )
    assert [item.target_timestamp for item in results] == ["2020-12-31", "2021-03-31"]
