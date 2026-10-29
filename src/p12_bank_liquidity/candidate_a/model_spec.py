"""Confirmatory Candidate A model list. No forecasting library is imported."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    protocol_id: str
    saved_source_key: str | None
    constructor: str
    parameter_status: str
    source_status: str
    notes: str


CONFIRMATORY_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        protocol_id="HistoricAverage",
        saved_source_key="historic_average",
        constructor="HistoricAverage()",
        parameter_status="NO_EXTRA_ARGUMENTS_IN_SAVED_MAP",
        source_status="SOURCE_STATED",
        notes="Saved forecast_stats.py model map. No extra constructor arguments are written there.",
    ),
    ModelSpec(
        protocol_id="Naive",
        saved_source_key=None,
        constructor="Naive",
        parameter_status="API_NOT_VERIFIED",
        source_status="SOURCE_ABSENT",
        notes=(
            "Owner-specified member of the confirmatory set. The saved import list and "
            "model map do not contain Naive. No constructor arguments are invented here."
        ),
    ),
    ModelSpec(
        protocol_id="SeasonalNaive",
        saved_source_key="seasonal_naive",
        constructor="SeasonalNaive(season_length=4)",
        parameter_status="SEASON_LENGTH_FROM_DATASET_SEASONALITY",
        source_status="SOURCE_STATED",
        notes="Saved map passes season_length=seasonality. Bank Cash Liquidity seasonality in datasets.toml is 4.",
    ),
    ModelSpec(
        protocol_id="AutoARIMA",
        saved_source_key="auto_arima",
        constructor="AutoARIMA(season_length=4)",
        parameter_status="SEASON_LENGTH_FROM_DATASET_SEASONALITY",
        source_status="SOURCE_STATED",
        notes="Saved map passes only season_length. Other AutoARIMA arguments are not specified.",
    ),
    ModelSpec(
        protocol_id="SimpleExponentialSmoothing",
        saved_source_key="ses",
        constructor="SimpleExponentialSmoothing(alpha=0.1)",
        parameter_status="ALPHA_EXPLICIT_IN_SAVED_MAP",
        source_status="SOURCE_STATED",
        notes="Selected instead of AutoETS because the saved SES constructor and the printed SES label agree. This does not establish the paper constructor.",
    ),
    ModelSpec(
        protocol_id="AutoTheta",
        saved_source_key=None,
        constructor="AutoTheta(season_length=4)",
        parameter_status="API_NOT_VERIFIED",
        source_status="SOURCE_ABSENT",
        notes=(
            "Owner-specified confirmatory constructor. The saved map uses "
            "Theta(season_length=seasonality), not AutoTheta. The two names are not treated as the same model."
        ),
    ),
)

EXCLUDED_FROM_CONFIRMATORY: dict[str, str] = {
    "Ridge": "Absent from the saved statistical model map. Alpha and feature scaling are not established. Removed before any forecast, not because of a result.",
    "AutoETS": "Present in the saved map as AutoETS(season_length=seasonality). Not the confirmatory M5 choice.",
    "Theta": "Present in the saved map as Theta(season_length=seasonality). Not the confirmatory AutoTheta choice.",
}

SAVED_STATSFORECAST_WRAPPER: dict[str, str] = {
    "freq_for_dataset_code_QE": "3mo",
    "n_jobs": "-1",
    "fallback_model": "SeasonalNaive(season_length=seasonality)",
    "verbose": "True",
    "cross_validation_h": "test_size",
    "cross_validation_step_size": "test_size",
    "fallback_rows": "UNRESOLVED_MUST_BE_EXCLUSIONS_IF_SUBSTITUTED",
    "arm_b_freq_token": "UNRESOLVED_DO_NOT_REGRID_WITH_3MO",
}
