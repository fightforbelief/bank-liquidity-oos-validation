"""Hand-calculated MASE. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from p12_bank_liquidity.candidate_a.metrics import seasonal_mase


def test_hand_calculated_mase_is_one_half() -> None:
    # Seasonal differences are 4, 4, 4, and 4. Mean absolute error is 2.
    result = seasonal_mase(
        [10, 12, 11, 13, 14, 16, 15, 17],
        [16, 21],
        [18, 19],
    )
    assert result.status == "DEFINED"
    assert result.value == 0.5
    assert result.evidence_label == "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"


def test_zero_denominator_is_undefined_without_epsilon() -> None:
    result = seasonal_mase([3, 3, 3, 3, 3, 3, 3, 3], [3], [4])
    assert result.status == "UNDEFINED"
    assert result.value is None
    assert result.reason == "zero_denominator"


def test_short_series_is_undefined() -> None:
    result = seasonal_mase([1, 2, 3], [4], [5])
    assert result.status == "UNDEFINED"
    assert result.reason == "short_history"
