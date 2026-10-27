"""Bank Cash Liquidity as implemented by FTSFR, with the NYU column definition.

The bank-level target is cash / assets on the preconstructed NYU columns.
It is not BHC Cash Liquidity and it is not the 2025 reserves-to-deposits ratio.

FTSFR ``create_ftsfr_datasets.py`` (jmbejara/ftsfr @ e740156e) computes::

    cash_liquidity = cash / assets
    keep if assets > 1e-10 and cash >= 0
    drop non-finite values
    drop an rssdid whose series has sample standard deviation 0

NYU ``callreport_1976_2020_WRDS.sas`` defines the columns before that ratio::

    assets = rcfd2170
    cash = rcfd0010
    if cash is missing: cash = rcfd0071 + rcfd0081
    equity = rcfd3210

The ratio function below copies the FTSFR filters. The numerator helper copies
the SAS fill. They are separate steps because FTSFR does not recompute the
numerator from MDRM codes; it reads the columns NYU already built.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

ASSET_ITEM = "RCFD2170"
CASH_ITEM = "RCFD0010"
CASH_FILL_ITEMS = ("RCFD0071", "RCFD0081")
EQUITY_ITEM = "RCFD3210"
MIN_ASSETS = 1e-10
BHC_TARGET_NAME = "BHC_CASH_LIQUIDITY"
BANK_TARGET_NAME = "BANK_CASH_LIQUIDITY"


def nyu_cash_numerator(rcfd0010, rcfd0071, rcfd0081):
    """SAS rule: use RCFD0010 when present, otherwise RCFD0071 + RCFD0081."""
    if rcfd0010 is not None:
        return float(rcfd0010)
    if rcfd0071 is None or rcfd0081 is None:
        return None
    return float(rcfd0071) + float(rcfd0081)


def cash_liquidity_ratio(cash, assets):
    """Return cash/assets, or None when the FTSFR row filter would drop the row."""
    if cash is None or assets is None:
        return None
    cash_value = float(cash)
    asset_value = float(assets)
    if not (asset_value > MIN_ASSETS and cash_value >= 0):
        return None
    ratio = cash_value / asset_value
    if not math.isfinite(ratio):
        return None
    return ratio


def drop_constant_series(rows: list[dict]) -> list[dict]:
    """Drop an entity when its retained ratios have sample standard deviation 0.

    A one-row series has undefined sample standard deviation in the pandas
    default (ddof=1) and is kept, matching ``std() == 0`` being false for NaN.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["entity_id"])].append(float(row["y"]))
    drop = set()
    for entity_id, values in grouped.items():
        if len(values) < 2:
            continue
        if statistics.stdev(values) == 0:
            drop.add(entity_id)
    return [row for row in rows if str(row["entity_id"]) not in drop]


def build_bank_cash_liquidity(rows: list[dict]) -> list[dict]:
    """Apply the bank-level FTSFR construction. BHC rows are rejected."""
    kept = []
    for row in rows:
        if row.get("target_scope", BANK_TARGET_NAME) != BANK_TARGET_NAME:
            raise ValueError(
                "BHC Cash Liquidity is a separate target and cannot be mixed into "
                f"{BANK_TARGET_NAME}"
            )
        ratio = cash_liquidity_ratio(row.get("cash"), row.get("assets"))
        if ratio is None:
            continue
        kept.append(
            {
                "entity_id": str(row["entity_id"]),
                "date": row["date"],
                "y": ratio,
                "target": BANK_TARGET_NAME,
            }
        )
    return drop_constant_series(kept)
