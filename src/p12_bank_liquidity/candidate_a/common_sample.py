"""Common evaluable sample keys. No forecast values are stored."""

from __future__ import annotations

from p12_bank_liquidity.candidate_a.grid_arms import GridRow, previous_quarter_end


def is_evaluable(row: GridRow) -> bool:
    return bool(
        row.observed_target
        and not row.is_synthetic_grid_row
        and row.target_value is not None
        and row.source_reporting_date is not None
    )


def evaluable_keys(rows: list[GridRow]) -> set[tuple[str, str, str]]:
    """Return entity, forecast-origin date, and target-quarter keys.

    The origin is the previous natural quarter-end. A synthetic row or a row
    without an observed target contributes no key.
    """

    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        if not is_evaluable(row):
            continue
        origin = previous_quarter_end(row.grid_timestamp)
        if origin >= row.grid_timestamp:
            continue
        keys.add((row.entity_id, origin.isoformat(), row.calendar_quarter))
    return keys


def common_evaluable_keys(
    arm_a: list[GridRow], arm_b: list[GridRow]
) -> list[tuple[str, str, str]]:
    return sorted(evaluable_keys(arm_a) & evaluable_keys(arm_b))
