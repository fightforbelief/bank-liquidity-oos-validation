"""Synthetic cap checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

import pytest

from p12_bank_liquidity.candidate_a.pilot_guard import (
    PILOT_CANDIDATE_POOL_CAP,
    PILOT_ENTITY_CAP,
    candidate_pool,
    entity_rank,
    select_fitted_entities,
)


def test_rank_is_sha256_of_the_utf8_text() -> None:
    assert entity_rank("fixture-a") == __import__("hashlib").sha256(b"fixture-a").hexdigest()


def test_pool_and_fitted_caps_have_no_override() -> None:
    identifiers = [f"fixture-{index}" for index in range(150)]
    pool = candidate_pool(identifiers)
    assert len(pool) == PILOT_CANDIDATE_POOL_CAP
    assert pool == sorted(identifiers, key=entity_rank)[:PILOT_CANDIDATE_POOL_CAP]
    eligible = pool
    chosen = select_fitted_entities(eligible)
    assert len(chosen) == PILOT_ENTITY_CAP
    assert chosen == pool[:PILOT_ENTITY_CAP]
    with pytest.raises(ValueError, match="insufficient pilot coverage"):
        select_fitted_entities(pool[:9])


def test_pilot_label_is_accepted_and_unlabeled_rows_are_rejected() -> None:
    from p12_bank_liquidity.candidate_a.pilot_guard import PILOT_LABEL
    from p12_bank_liquidity.candidate_a.timestamp_adapter import (
        HistoryPoint,
        ResearchRowRejected,
        ScheduleRow,
        training_values,
    )

    row = ScheduleRow(
        "fixture-entity",
        "A",
        "2020-03-31",
        "2020-06-30",
        "2020Q2",
        True,
        False,
        PILOT_LABEL,
    )
    point = HistoryPoint(
        "fixture-entity",
        "A",
        "2020-03-31",
        1.0,
        PILOT_LABEL,
        y_imputed=1.0,
        imputation_applied=True,
    )
    assert training_values([point], row) == [1.0]
    rejected = HistoryPoint("fixture-entity", "A", "2020-03-31", 1.0, "research-row")
    with pytest.raises(ResearchRowRejected):
        training_values([rejected], row)
