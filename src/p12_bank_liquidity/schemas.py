"""Quarter, duplicate, and missingness checks. Missing quarters are reported, not filled."""

from __future__ import annotations

import calendar
import datetime as dt
from collections import Counter


QUARTER_END_MD = ((3, 31), (6, 30), (9, 30), (12, 31))


def parse_quarter_end(value: str | dt.date | dt.datetime) -> dt.date:
    """Parse a quarter-end date. Non-quarter-end dates are rejected."""
    if isinstance(value, dt.datetime):
        parsed = value.date()
    elif isinstance(value, dt.date):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if len(text) == 6 and text[4] == "Q" and text[:4].isdigit() and text[5] in "1234":
            year = int(text[:4])
            quarter = int(text[5])
            month, day = QUARTER_END_MD[quarter - 1]
            return dt.date(year, month, day)
        parsed = dt.date.fromisoformat(text[:10])
    else:
        raise TypeError(f"unsupported date value: {value!r}")
    if (parsed.month, parsed.day) not in QUARTER_END_MD:
        raise ValueError(f"not a calendar quarter-end: {parsed.isoformat()}")
    if parsed.day != calendar.monthrange(parsed.year, parsed.month)[1]:
        raise ValueError(f"not a month-end: {parsed.isoformat()}")
    return parsed


def next_quarter_end(value: dt.date) -> dt.date:
    parsed = parse_quarter_end(value)
    index = QUARTER_END_MD.index((parsed.month, parsed.day))
    if index < 3:
        month, day = QUARTER_END_MD[index + 1]
        return dt.date(parsed.year, month, day)
    return dt.date(parsed.year + 1, 3, 31)


def expected_quarter_ends(start: dt.date, end: dt.date) -> list[dt.date]:
    start = parse_quarter_end(start)
    end = parse_quarter_end(end)
    if end < start:
        raise ValueError("end is before start")
    cursor = start
    found = []
    while cursor <= end:
        found.append(cursor)
        cursor = next_quarter_end(cursor)
    return found


def duplicate_entity_quarters(rows: list[dict]) -> list[tuple[str, str]]:
    counts = Counter((str(row["entity_id"]), parse_quarter_end(row["date"]).isoformat()) for row in rows)
    return sorted(key for key, count in counts.items() if count > 1)


def missing_quarter_ends(rows: list[dict], entity_id: str) -> list[str]:
    """Return missing quarter-ends inside an entity's observed span. Do not fill them."""
    dates = sorted(
        parse_quarter_end(row["date"])
        for row in rows
        if str(row["entity_id"]) == str(entity_id)
    )
    if not dates:
        return []
    present = set(dates)
    return [
        item.isoformat()
        for item in expected_quarter_ends(dates[0], dates[-1])
        if item not in present
    ]


def panel_without_interpolation(rows: list[dict]) -> list[dict]:
    """Identity copy. This function must not invent missing quarters."""
    return list(rows)
