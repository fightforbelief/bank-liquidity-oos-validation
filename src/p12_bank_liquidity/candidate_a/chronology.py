"""Chronological split checks. No model is fit."""

from __future__ import annotations

from datetime import date


class FutureLeakage(ValueError):
    """A training timestamp is not strictly before the target date."""


def training_dates_precede_target(train_dates: list[date], target_date: date) -> bool:
    return all(item < target_date for item in train_dates)


def require_training_before_target(train_dates: list[date], target_date: date) -> None:
    if not training_dates_precede_target(train_dates, target_date):
        raise FutureLeakage("training date is not strictly before the target date")


def train_only_location(train_values: list[float], train_dates: list[date], target_date: date) -> float:
    """Location fitted on training values only.

    The target date is used solely as a leakage boundary. It is not a feature.
    """

    require_training_before_target(train_dates, target_date)
    if not train_values:
        raise ValueError("training values are empty")
    return sum(train_values) / len(train_values)
