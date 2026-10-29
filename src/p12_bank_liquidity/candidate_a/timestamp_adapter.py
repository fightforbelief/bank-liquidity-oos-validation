"""One-step adapter that keeps the supplied target timestamp.

The adapter does not import StatsForecast and does not generate dates from a
frequency alias. A real library return value is not parsed here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol

from p12_bank_liquidity.candidate_a.alignment import calendar_quarter
from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL
from p12_bank_liquidity.candidate_a.model_output import UnexpectedModelOutput

PANEL_WINDOW_CAP = 6
REAL_MODEL_OUTPUT_BINDING = "BLOCKED_PENDING_REAL_API_SMOKE_TEST"
FROZEN_CONSTRUCTORS = (
    "HistoricAverage()",
    "Naive()",
    "SeasonalNaive(season_length=4)",
    "AutoARIMA(season_length=4)",
    "SimpleExponentialSmoothing(alpha=0.1)",
    "AutoTheta(season_length=4)",
)
FUTURE_WRAPPER_FALLBACK = "fallback_model=None"


class ResearchRowRejected(ValueError):
    """The adapter accepts labelled synthetic rows only."""


class UnimputedTrainingGap(ValueError):
    """An interior training-grid position has no prepared imputed value."""


class DuplicateEvaluationKey(ValueError):
    """The same entity, arm, and target timestamp was supplied twice."""


class OneStepForecaster(Protocol):
    """Fake or future one-step model. Stage 14 tests use fakes only."""

    model_id: str

    def predict_one(self, y_train: list[float]) -> float:
        """Return one finite forecast, or raise if the model fails."""


@dataclass(frozen=True)
class ScheduleRow:
    entity_id: str
    arm_id: str
    origin_timestamp: str
    target_timestamp: str
    target_quarter: str
    observed_finite_source_target: bool
    synthetic_grid_row: bool
    evidence_label: str


@dataclass(frozen=True)
class HistoryPoint:
    """One arm-specific grid position.

    ``value`` is the original source observation. ``y_imputed`` is the training
    value after the frozen training-only forward fill. The adapter does not
    compute that fill.
    """

    entity_id: str
    arm_id: str
    timestamp: str
    value: float | None
    evidence_label: str
    y_imputed: float | None = None
    imputation_applied: bool = False
    synthetic_grid_row: bool = False


@dataclass(frozen=True)
class AdapterResult:
    entity_id: str
    arm_id: str
    model_id: str
    origin_timestamp: str
    target_timestamp: str
    target_quarter: str
    status: str
    prediction: float | None
    reason: str | None
    evidence_label: str


def _finite(value: float | None) -> bool:
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


ALLOWED_ROW_LABELS = {
    EVIDENCE_LABEL,
    "PILOT_ONLY_NOT_PRIMARY_RESEARCH_RESULTS",
    "STAGE18_FULL_SAMPLE_PRIMARY_RESEARCH_RUN",
}


def _require_fixture(label: str) -> None:
    if label not in ALLOWED_ROW_LABELS:
        raise ResearchRowRejected(label)


def _prepared_training_value(point: HistoryPoint) -> float | None:
    if point.imputation_applied:
        return point.y_imputed
    return point.value


def training_values(
    history: list[HistoryPoint],
    row: ScheduleRow,
) -> list[float]:
    """Prepared training values on the arm grid, in timestamp order.

    Leading positions that are still missing after imputation are omitted.
    A later missing position is not dropped, because that would compress the
    quarterly spacing.
    """

    _require_fixture(row.evidence_label)
    selected: list[HistoryPoint] = []
    for point in history:
        _require_fixture(point.evidence_label)
        if point.entity_id != row.entity_id or point.arm_id != row.arm_id:
            continue
        if point.timestamp > row.origin_timestamp or point.timestamp >= row.target_timestamp:
            continue
        selected.append(point)
    selected.sort(key=lambda point: point.timestamp)
    numbers = [_prepared_training_value(point) for point in selected]
    if not any(_finite(value) for value in numbers):
        return []
    first_finite = next(index for index, value in enumerate(numbers) if _finite(value))
    tail = numbers[first_finite:]
    if any(not _finite(value) for value in tail):
        raise UnimputedTrainingGap(row.entity_id + " " + row.arm_id + " " + row.origin_timestamp)
    return [float(value) for value in tail if value is not None]


def run_one_step(
    schedule: list[ScheduleRow],
    history: list[HistoryPoint],
    forecaster: OneStepForecaster,
    *,
    panel_window_cap: int = PANEL_WINDOW_CAP,
) -> list[AdapterResult]:
    """Score the supplied schedule. Do not add windows or rewrite timestamps."""

    if panel_window_cap != PANEL_WINDOW_CAP:
        raise ValueError("panel window cap is frozen at 6 and is not a per-entity assignment")
    if "freq" in inspect.signature(run_one_step).parameters:
        raise RuntimeError("the adapter must not accept a frequency alias")
    seen: set[tuple[str, str, str]] = set()
    results: list[AdapterResult] = []
    for row in schedule:
        _require_fixture(row.evidence_label)
        identity = (row.entity_id, row.arm_id, row.target_timestamp)
        if identity in seen:
            raise DuplicateEvaluationKey(f"{row.entity_id} {row.arm_id} {row.target_timestamp}")
        seen.add(identity)
        base = dict(
            entity_id=row.entity_id,
            arm_id=row.arm_id,
            model_id=forecaster.model_id,
            origin_timestamp=row.origin_timestamp,
            target_timestamp=row.target_timestamp,
            target_quarter=row.target_quarter,
            evidence_label=row.evidence_label,
        )
        if row.synthetic_grid_row or not row.observed_finite_source_target:
            reason = "synthetic_grid_row" if row.synthetic_grid_row else "target_not_observed"
            results.append(AdapterResult(status="EXCLUDED", prediction=None, reason=reason, **base))
            continue
        try:
            raw = forecaster.predict_one(training_values(history, row))
        except UnimputedTrainingGap:
            results.append(
                AdapterResult(status="EXCLUDED", prediction=None, reason="unimputed_training_gap", **base)
            )
            continue
        except UnexpectedModelOutput:
            results.append(
                AdapterResult(status="EXCLUDED", prediction=None, reason="unexpected_output_shape", **base)
            )
            continue
        except Exception:
            results.append(AdapterResult(status="EXCLUDED", prediction=None, reason="model_failed", **base))
            continue
        if not _finite(raw):
            results.append(
                AdapterResult(status="EXCLUDED", prediction=None, reason="forecast_not_finite", **base)
            )
            continue
        results.append(AdapterResult(status="RECORDED_FORECAST", prediction=float(raw), reason=None, **base))
    return results


def primary_pairs_keep_separate_origins(
    arm_a: list[ScheduleRow],
    arm_b: list[ScheduleRow],
) -> list[dict[str, object]]:
    """Pair on entity and target. Different origins stay out of the secondary set."""

    index_b = {(row.entity_id, row.target_timestamp): row for row in arm_b if not row.synthetic_grid_row}
    paired: list[dict[str, object]] = []
    for row in arm_a:
        _require_fixture(row.evidence_label)
        if row.synthetic_grid_row:
            continue
        other = index_b.get((row.entity_id, row.target_timestamp))
        if other is None:
            continue
        _require_fixture(other.evidence_label)
        equal = row.origin_timestamp == other.origin_timestamp
        paired.append(
            {
                "entity_id": row.entity_id,
                "origin_timestamp_arm_a": row.origin_timestamp,
                "origin_timestamp_arm_b": other.origin_timestamp,
                "target_timestamp": row.target_timestamp,
                "origin_timestamp_equal": equal,
                "origin_quarter_equal": calendar_quarter(row.origin_timestamp)
                == calendar_quarter(other.origin_timestamp),
                "in_secondary_exact_timestamp_set": equal,
                "evidence_label": EVIDENCE_LABEL,
            }
        )
    return paired
