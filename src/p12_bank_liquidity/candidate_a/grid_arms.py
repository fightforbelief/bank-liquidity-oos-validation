"""Two quarterly grid arms for Candidate A.

Arm B is implemented here. Arm A is the inspected ``fill_gaps(freq="3mo")``
path and is checked by a subprocess test against the saved preprocessing
environment. This module does not modify upstream FTSFR code.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date

EVIDENCE_LABEL = "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"
ARM_A = "A_IMPL_3MO"
ARM_B = "B_NATURAL_QE"


class DuplicateEntityQuarter(ValueError):
    """One entity has two source rows in the same calendar quarter.

    No aggregation rule is frozen. Callers stop instead of choosing a row.
    """


class NonQuarterEndSourceDate(ValueError):
    """Arm B does not move a source date onto a different day."""


def calendar_quarter(day: date) -> str:
    quarter = (day.month - 1) // 3 + 1
    return f"{day.year}Q{quarter}"


def is_natural_quarter_end(day: date) -> bool:
    if day.month not in (3, 6, 9, 12):
        return False
    return day.day == monthrange(day.year, day.month)[1]


def quarter_index(day: date) -> int:
    return day.year * 4 + (day.month - 1) // 3


def date_from_quarter_index(index: int) -> date:
    year, zero_based = divmod(index, 4)
    month = (zero_based + 1) * 3
    return date(year, month, monthrange(year, month)[1])


def previous_quarter_end(day: date) -> date:
    return date_from_quarter_index(quarter_index(day) - 1)


@dataclass(frozen=True)
class SourceRow:
    entity_id: str
    source_reporting_date: date
    target_value: float | None
    source_row_id: str


@dataclass(frozen=True)
class GridRow:
    arm_id: str
    entity_id: str
    source_reporting_date: date | None
    grid_timestamp: date
    calendar_quarter: str
    observed_target: bool
    target_value: float | None
    is_synthetic_grid_row: bool
    source_row_id: str | None
    evidence_label: str


def _is_finite_number(value: float | None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if value != value:
        return False
    return value not in (float("inf"), float("-inf"))


def build_arm_b(
    rows: list[SourceRow],
    *,
    evidence_label: str = EVIDENCE_LABEL,
    grid_end: date | None = None,
) -> list[GridRow]:
    """Enumerate natural quarter-ends from the first to the last source quarter.

    A missing quarter becomes a synthetic row with a missing target. The
    original ``source_reporting_date`` is copied onto observed rows and is not
    replaced by the grid timestamp field.
    """

    by_entity: dict[str, list[SourceRow]] = {}
    for row in rows:
        if not is_natural_quarter_end(row.source_reporting_date):
            raise NonQuarterEndSourceDate(
                "Arm B refuses to relocate a source date that is not a natural quarter-end"
            )
        by_entity.setdefault(row.entity_id, []).append(row)

    if grid_end is not None and not is_natural_quarter_end(grid_end):
        raise NonQuarterEndSourceDate("grid_end must be a natural quarter-end")

    result: list[GridRow] = []
    for entity_id, entity_rows in by_entity.items():
        seen: dict[str, SourceRow] = {}
        for row in entity_rows:
            key = calendar_quarter(row.source_reporting_date)
            if key in seen:
                raise DuplicateEntityQuarter(entity_id)
            seen[key] = row
        indexes = [quarter_index(row.source_reporting_date) for row in entity_rows]
        end_index = max(indexes)
        if grid_end is not None:
            end_index = max(end_index, quarter_index(grid_end))
        for index in range(min(indexes), end_index + 1):
            grid_day = date_from_quarter_index(index)
            quarter = calendar_quarter(grid_day)
            source = seen.get(quarter)
            if source is None:
                result.append(
                    GridRow(
                        arm_id=ARM_B,
                        entity_id=entity_id,
                        source_reporting_date=None,
                        grid_timestamp=grid_day,
                        calendar_quarter=quarter,
                        observed_target=False,
                        target_value=None,
                        is_synthetic_grid_row=True,
                        source_row_id=None,
                        evidence_label=evidence_label,
                    )
                )
                continue
            observed = _is_finite_number(source.target_value)
            result.append(
                GridRow(
                    arm_id=ARM_B,
                    entity_id=entity_id,
                    source_reporting_date=source.source_reporting_date,
                    grid_timestamp=grid_day,
                    calendar_quarter=quarter,
                    observed_target=observed,
                    target_value=source.target_value if observed else None,
                    is_synthetic_grid_row=False,
                    source_row_id=source.source_row_id,
                    evidence_label=evidence_label,
                )
            )
    return result
