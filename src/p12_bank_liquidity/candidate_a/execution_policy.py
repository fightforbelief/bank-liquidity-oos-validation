"""Failure recording for the frozen six-model comparison.

This module does not import StatsForecast and does not fit a model.
A fallback value is never stored as the original model's forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL


class ResearchRowRejected(ValueError):
    """Stage 12 policy helpers do not accept unlabeled research rows."""


@dataclass(frozen=True)
class ModelAttempt:
    model_id: str
    forecast_value: float | None
    error: str | None
    fallback_forecast_value: float | None
    evidence_label: str


@dataclass(frozen=True)
class ModelOutcome:
    model_id: str
    status: str
    forecast_value: float | None
    reason: str | None
    evidence_label: str


def _finite(value: float | None) -> bool:
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def classify_model_attempt(attempt: ModelAttempt) -> ModelOutcome:
    """Record the original model's own forecast, or an explicit exclusion.

    ``fallback_forecast_value`` is ignored. It cannot fill a failure.
    """

    if attempt.evidence_label != EVIDENCE_LABEL:
        raise ResearchRowRejected(attempt.evidence_label)
    if attempt.error is not None:
        return ModelOutcome(
            model_id=attempt.model_id,
            status="EXCLUDED",
            forecast_value=None,
            reason="model_failed",
            evidence_label=EVIDENCE_LABEL,
        )
    if not _finite(attempt.forecast_value):
        return ModelOutcome(
            model_id=attempt.model_id,
            status="EXCLUDED",
            forecast_value=None,
            reason="forecast_not_finite",
            evidence_label=EVIDENCE_LABEL,
        )
    return ModelOutcome(
        model_id=attempt.model_id,
        status="RECORDED_FORECAST",
        forecast_value=attempt.forecast_value,
        reason=None,
        evidence_label=EVIDENCE_LABEL,
    )
