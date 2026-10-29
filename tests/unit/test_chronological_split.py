"""Chronology, leakage, and failure logging. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from datetime import date

import pytest

from p12_bank_liquidity.candidate_a.chronology import (
    FutureLeakage,
    train_only_location,
    training_dates_precede_target,
)
from p12_bank_liquidity.candidate_a.failures import record_failure


def test_training_dates_precede_target_quarter() -> None:
    assert training_dates_precede_target(
        [date(2019, 12, 31), date(2020, 3, 31)],
        date(2020, 6, 30),
    )


def test_future_training_date_is_rejected() -> None:
    with pytest.raises(FutureLeakage):
        train_only_location(
            [0.1, 0.2],
            [date(2020, 3, 31), date(2020, 6, 30)],
            date(2020, 6, 30),
        )


def test_location_uses_training_values_only() -> None:
    value = train_only_location(
        [1.0, 3.0],
        [date(2020, 3, 31), date(2020, 6, 30)],
        date(2020, 9, 30),
    )
    assert value == 2.0


def test_failure_is_retained() -> None:
    log = record_failure(
        [],
        entity_id="TEST_FIXTURE_NOT_RESEARCH_EVIDENCE",
        model_id="M2",
        reason="synthetic exception",
        evidence_label="TEST_FIXTURE_NOT_RESEARCH_EVIDENCE",
    )
    assert len(log) == 1
    assert log[0].reason == "synthetic exception"
