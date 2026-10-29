"""Common-sample tests. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from datetime import date

from p12_bank_liquidity.candidate_a.common_sample import common_evaluable_keys
from p12_bank_liquidity.candidate_a.grid_arms import (
    ARM_A,
    EVIDENCE_LABEL,
    GridRow,
    SourceRow,
    build_arm_b,
)


def _grid(
    arm: str,
    entity: str,
    day: date,
    *,
    observed: bool,
    synthetic: bool,
    value: float | None,
) -> GridRow:
    return GridRow(
        arm_id=arm,
        entity_id=entity,
        source_reporting_date=None if synthetic else day,
        grid_timestamp=day,
        calendar_quarter=f"{day.year}Q{(day.month - 1) // 3 + 1}",
        observed_target=observed,
        target_value=value,
        is_synthetic_grid_row=synthetic,
        source_row_id=None if synthetic else "src",
        evidence_label=EVIDENCE_LABEL,
    )


def test_common_sample_keeps_only_rows_observed_on_both_arms() -> None:
    shared = _grid(ARM_A, "E1", date(2020, 6, 30), observed=True, synthetic=False, value=0.2)
    synthetic = _grid(ARM_A, "E1", date(2020, 9, 30), observed=False, synthetic=True, value=None)
    arm_a_only = _grid(ARM_A, "E2", date(2020, 6, 30), observed=True, synthetic=False, value=0.4)
    arm_b_shared = _grid("B_NATURAL_QE", "E1", date(2020, 6, 30), observed=True, synthetic=False, value=0.2)
    arm_b_gap = _grid("B_NATURAL_QE", "E1", date(2020, 9, 30), observed=True, synthetic=False, value=0.3)
    keys = common_evaluable_keys([shared, synthetic, arm_a_only], [arm_b_shared, arm_b_gap])
    assert keys == [("E1", "2020-03-31", "2020Q2")]


def test_arm_b_builder_rows_trace_to_source_dates() -> None:
    built = build_arm_b(
        [SourceRow("E", date(2020, 3, 31), 0.1, "row-1"), SourceRow("E", date(2020, 9, 30), 0.2, "row-2")]
    )
    observed = [row for row in built if row.observed_target]
    assert [row.source_reporting_date for row in observed] == [date(2020, 3, 31), date(2020, 9, 30)]
    assert [row.source_row_id for row in observed] == ["row-1", "row-2"]
    assert all(row.evidence_label == "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE" for row in built)
