"""Stage 18 guard tests. No research counts and no research rows."""

import numpy as np
import pytest

from p12_bank_liquidity.candidate_a.pilot_guard import PILOT_ENTITY_CAP
from p12_bank_liquidity.candidate_a.stage18_guard import (
    STAGE18_LABEL,
    assert_full_sample_label,
    assert_pilot_cap_unchanged,
)
from p12_bank_liquidity.candidate_a.stage18_inference import cluster_bootstrap
from p12_bank_liquidity.candidate_a.timestamp_adapter import (
    HistoryPoint,
    ResearchRowRejected,
    ScheduleRow,
    training_values,
)


def test_pilot_cap_stays_at_20_and_full_run_is_a_separate_label() -> None:
    assert PILOT_ENTITY_CAP == 20
    assert_pilot_cap_unchanged()
    assert_full_sample_label(STAGE18_LABEL)
    with pytest.raises(RuntimeError):
        assert_full_sample_label("PILOT_ONLY_NOT_PRIMARY_RESEARCH_RESULTS")


def test_stage18_label_is_accepted_by_the_adapter() -> None:
    row = ScheduleRow("fixture", "A", "2020-03-31", "2020-06-30", "2020Q2", True, False, STAGE18_LABEL)
    point = HistoryPoint("fixture", "A", "2020-03-31", 1.0, STAGE18_LABEL, y_imputed=1.0, imputation_applied=True)
    assert training_values([point], row) == [1.0]
    rejected = HistoryPoint("fixture", "A", "2020-03-31", 1.0, "not-a-permitted-label")
    with pytest.raises(ResearchRowRejected):
        training_values([rejected], row)


def test_constant_cluster_interval_stays_on_the_constant() -> None:
    result = cluster_bootstrap(np.array([5.0, 5.0, 5.0]), seed=20261027, replicates=30)
    assert result["valid"] == 30
    assert result["failed"] == 0
    assert result["estimate"] == 5.0
    assert result["low"] == 5.0
    assert result["high"] == 5.0


def test_empty_bootstrap_does_not_invent_a_zero_interval() -> None:
    result = cluster_bootstrap(np.array([], dtype=float), replicates=2000)
    assert result["estimate"] is None
    assert result["low"] is None
    assert result["failed"] == 2000
    assert result["interval"] == "unavailable"
