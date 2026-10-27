"""Panel integrity checks shared by every future model."""

from __future__ import annotations

from p12_bank_liquidity.gates import GateBlocked
from p12_bank_liquidity.schemas import duplicate_entity_quarters, missing_quarter_ends, parse_quarter_end


def eligible_entity_ids(rows: list[dict]) -> frozenset[str]:
    return frozenset(str(row["entity_id"]) for row in rows)


def assert_same_eligible_panel(panels: dict[str, list[dict]]) -> frozenset[str]:
    if not panels:
        raise GateBlocked("no model panels were supplied")
    snapshots = {name: eligible_entity_ids(rows) for name, rows in panels.items()}
    unique = set(snapshots.values())
    if len(unique) != 1:
        raise GateBlocked(
            "models do not see the same eligible panel: "
            + ", ".join(f"{name}={len(ids)}" for name, ids in sorted(snapshots.items()))
        )
    return next(iter(unique))


def reject_survivor_filter(rows: list[dict], terminal_date: str) -> None:
    """Selecting entities by presence on a future terminal date is not allowed."""
    terminal = parse_quarter_end(terminal_date)
    survivors = {
        str(row["entity_id"])
        for row in rows
        if parse_quarter_end(row["date"]) == terminal
    }
    raise GateBlocked(
        "refusing a survivor filter on "
        f"{terminal.isoformat()} ({len(survivors)} entities would have been kept)"
    )


def quality_report(rows: list[dict]) -> dict:
    entities = sorted(eligible_entity_ids(rows))
    return {
        "n_rows": len(rows),
        "n_entities": len(entities),
        "duplicate_entity_quarters": duplicate_entity_quarters(rows),
        "missing_by_entity": {
            entity_id: missing_quarter_ends(rows, entity_id) for entity_id in entities
        },
        "interpolated_rows": 0,
    }
