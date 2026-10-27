"""Candidate metric formulas for tests.

OFR reports MASE, pooled out-of-sample R², and RMSE. The forecasting code
imports ``mase``, ``mse``, and ``rmse`` from ``utilsforecast.losses``. The
functions here are the usual Hyndman / Campbell-Thompson definitions used to
lock software behaviour. They are not yet checked observation-by-observation
against that library, so they are not a reproduction of Table 6.
"""

from __future__ import annotations

import math


def mean_absolute_scaled_error(
    y_true: list[float],
    y_pred: list[float],
    y_insample: list[float],
    seasonality: int,
) -> float:
    if seasonality < 1:
        raise ValueError("seasonality must be positive")
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and aligned")
    if len(y_insample) <= seasonality:
        raise ValueError("in-sample series is shorter than the seasonal lag")
    scale_terms = [
        abs(y_insample[index] - y_insample[index - seasonality])
        for index in range(seasonality, len(y_insample))
    ]
    scale = sum(scale_terms) / len(scale_terms)
    if scale == 0:
        raise ValueError("MASE scale is zero")
    absolute = [abs(actual - forecast) for actual, forecast in zip(y_true, y_pred)]
    return (sum(absolute) / len(absolute)) / scale


def root_mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and aligned")
    squared = [(actual - forecast) ** 2 for actual, forecast in zip(y_true, y_pred)]
    return math.sqrt(sum(squared) / len(squared))


def pooled_oos_r2(
    y_true: list[float],
    y_pred: list[float],
    y_benchmark: list[float],
) -> float:
    """1 - SSE(model) / SSE(historical-average benchmark), pooled over supplied rows."""
    if not (len(y_true) == len(y_pred) == len(y_benchmark)) or not y_true:
        raise ValueError("metric inputs must be non-empty and aligned")
    sse_model = sum((actual - forecast) ** 2 for actual, forecast in zip(y_true, y_pred))
    sse_benchmark = sum(
        (actual - benchmark) ** 2 for actual, benchmark in zip(y_true, y_benchmark)
    )
    if sse_benchmark == 0:
        raise ValueError("benchmark SSE is zero")
    return 1.0 - sse_model / sse_benchmark
