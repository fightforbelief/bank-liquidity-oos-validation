"""Synthetic checks for the constructed-target audit helpers.

TEST_FIXTURE_NOT_RESEARCH_EVIDENCE
These rows are invented arithmetic. They are not bank filings.
"""

import importlib.util
from pathlib import Path

import numpy as np

from p12_bank_liquidity.target.bank_cash_liquidity import cash_liquidity_ratio

FIXTURE_BANNER = "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"


def _audit():
    path = Path(__file__).resolve().parents[2] / "scripts" / "constructed_target_audit.py"
    spec = importlib.util.spec_from_file_location("constructed_target_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_banner_and_explicit_ratio_match_project_function():
    module = _audit()
    assert module.FIXTURE_BANNER == FIXTURE_BANNER
    assert module.independent_ratio(20, 100) == 0.2
    assert module.independent_ratio(20, 100) == cash_liquidity_ratio(20, 100)
    assert module.independent_ratio(-1, 100) is None
    assert cash_liquidity_ratio(-1, 100) is None
    assert module.independent_ratio(5, 1e-12) is None


def test_selection_order_is_fixed_before_ratio_values():
    module = _audit()
    dates = np.array([19760331, 19760331, 19760630, 20200331, 20200331])
    rssdid = np.array([2, 1, 1, 9, 1])
    rows = np.array([1, 2, 3, 4, 5])
    eligible = np.ones(5, dtype=bool)
    chosen = module.select_three(dates, rssdid, rows, eligible)
    assert rows[chosen["A"]] == 2
    assert rows[chosen["B"]] == 1
    assert rows[chosen["C"]] == 4
    assert chosen["B_distinct_bank"] is True
    assert chosen["C_distinct_bank"] is True


def test_qe_requirement_arithmetic():
    module = _audit()
    requirements = module.qe_quality_requirements(test_size=1, seasonality=4)
    assert requirements == {
        "min_total_obs": 27,
        "min_train_obs": 18,
        "min_test_obs": 1,
        "variance_threshold": 0.01,
        "max_gap_ratio": 0.6,
    }
    assert module.forecasting_length_threshold(1, 4) == 18
    cv_rules = module.cv_requirement_thresholds()
    assert cv_rules["min_total_obs"] == 27
    assert cv_rules["min_train_obs"] == 24
    assert cv_rules["min_test_obs"] == 3
