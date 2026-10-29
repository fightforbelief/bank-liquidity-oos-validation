"""Hand-checked MASE rules for Candidate A.

A zero seasonal denominator or a series without a seasonal pair is undefined.
No epsilon is substituted. This is not a check against ``utilsforecast.losses.mase``;
that package is not installed in the project environment.
"""

from __future__ import annotations

from dataclasses import dataclass

EVIDENCE_LABEL = "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"


@dataclass(frozen=True)
class MaseResult:
    status: str
    value: float | None
    reason: str | None
    seasonality: int
    evidence_label: str


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def seasonal_mase(
    y_train: list[float],
    y_observed: list[float],
    y_forecast: list[float],
    *,
    seasonality: int = 4,
    evidence_label: str = EVIDENCE_LABEL,
) -> MaseResult:
    if seasonality != 4:
        return MaseResult("UNDEFINED", None, "seasonality_not_frozen_at_4", seasonality, evidence_label)
    if len(y_observed) != len(y_forecast) or not y_observed:
        return MaseResult("UNDEFINED", None, "forecast_actual_misaligned", seasonality, evidence_label)
    if any(not _finite(value) for value in y_observed + y_forecast):
        return MaseResult("UNDEFINED", None, "non_finite_forecast_or_target", seasonality, evidence_label)
    pairs = []
    for index in range(seasonality, len(y_train)):
        current = y_train[index]
        lagged = y_train[index - seasonality]
        if _finite(current) and _finite(lagged):
            pairs.append(abs(current - lagged))
    if not pairs:
        return MaseResult("UNDEFINED", None, "short_history", seasonality, evidence_label)
    scale = sum(pairs) / len(pairs)
    if scale == 0:
        return MaseResult("UNDEFINED", None, "zero_denominator", seasonality, evidence_label)
    numerator = sum(abs(actual - forecast) for actual, forecast in zip(y_observed, y_forecast))
    numerator /= len(y_observed)
    return MaseResult("DEFINED", numerator / scale, None, seasonality, evidence_label)
