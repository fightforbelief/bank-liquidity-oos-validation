"""Chronological splits. Training statistics are fit on the train slice only."""

from __future__ import annotations

import statistics

from p12_bank_liquidity.schemas import parse_quarter_end


def rolling_origin_windows(rows: list[dict], horizon: int, max_windows: int) -> list[dict]:
    """One-entity rolling origin. Train dates are strictly before the first test date."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    ordered = sorted(rows, key=lambda row: parse_quarter_end(row["date"]))
    windows = []
    last_train_end = len(ordered) - horizon
    start = max(1, last_train_end - max_windows + 1)
    for train_end in range(start, last_train_end + 1):
        train = ordered[:train_end]
        test = ordered[train_end : train_end + horizon]
        if len(test) < horizon:
            continue
        if parse_quarter_end(train[-1]["date"]) >= parse_quarter_end(test[0]["date"]):
            raise RuntimeError("future observation entered the training window")
        windows.append({"train": train, "test": test})
    return windows[-max_windows:]


def fit_train_standardizer(train_values: list[float]) -> tuple[float, float]:
    if len(train_values) < 2:
        raise ValueError("train-only standardizer needs at least two observations")
    center = statistics.fmean(train_values)
    scale = statistics.stdev(train_values)
    if scale == 0:
        raise ValueError("train slice has zero sample standard deviation")
    return center, scale


def apply_standardizer(values: list[float], center: float, scale: float) -> list[float]:
    return [(value - center) / scale for value in values]


def train_only_clip_bounds(train_values: list[float], iqr_multiplier: float = 1.0) -> tuple[float, float]:
    """Clip bounds from the training slice only.

    The source uses ``[train_min - k*IQR, train_max + k*IQR]`` with k = 1,
    computed from history available at the first forecast origin. This helper
    is the same arithmetic for a single training slice. It is not a claim that
    a confirmatory run has been executed.
    """
    ordered = sorted(train_values)
    n = len(ordered)
    if n < 4:
        raise ValueError("IQR clip needs at least four training observations")

    def quantile(p: float) -> float:
        position = (n - 1) * p
        low = int(position)
        high = min(low + 1, n - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    q1 = quantile(0.25)
    q3 = quantile(0.75)
    iqr = q3 - q1
    return ordered[0] - iqr_multiplier * iqr, ordered[-1] + iqr_multiplier * iqr
