"""Three historical bank-quarter checks for the NYU Call Report cash ratio.

This script does not import p12_bank_liquidity. The float calculation below
is a direct copy of the audited source rules:

- assets = rcfd2170
- cash = rcfd0010 when that item is observed
- otherwise cash = rcfd0071 + rcfd0081 when both items are observed
- y = cash / assets
- keep the row when assets > 1e-10, cash >= 0, and y is finite

The panel-level zero standard-deviation filter is intentionally not applied.
A three-row check cannot verify it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if any(name == "p12_bank_liquidity" or name.startswith("p12_bank_liquidity.") for name in sys.modules):
    raise SystemExit("Refusing to run: the project target module is already imported.")

SCRATCH = Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch")
PROJECT = Path(r"C:\Users\hanqi\Desktop\projects\p12-bank-liquidity-oos-validation")
ZIP_NAME = "callreports_1976_2020_WRDS.dta.zip"
EXPECTED_BYTES = 455791391
MIN_ASSETS = 1e-10
DATE_B_MIN = 19840331
DATE_C_MIN = 20150331
DATE_C_MAX = 20200331
NEEDED = ("rssdid", "date", "rcfd2170", "rcfd0010", "rcfd0071", "rcfd0081")
OPTIONAL = ("dateq", "cash", "assets", "year", "quarter", "chartertype", "bhcid")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_zip_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or normalized.startswith("/") or re_drive(normalized):
        return False
    return ".." not in parts


def re_drive(name: str) -> bool:
    head = name.replace("\\", "/").split("/")[0]
    return len(head) >= 2 and head[1] == ":"


def inspect_zip(zip_path: Path, extract_dir: Path) -> dict:
    info = {
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "expected_bytes": EXPECTED_BYTES,
        "size_matches_content_length": zip_path.stat().st_size == EXPECTED_BYTES,
        "sha256": sha256_file(zip_path),
        "inspected_utc": utc_now(),
    }
    if not info["size_matches_content_length"]:
        raise SystemExit(
            f"ZIP size {info['zip_bytes']} does not match Content-Length {EXPECTED_BYTES}"
        )
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        info["zip_test_bad_member"] = bad
        info["zip_test_ok"] = bad is None
        members = []
        for member in archive.infolist():
            members.append(
                {
                    "name": member.filename,
                    "file_size": member.file_size,
                    "compress_size": member.compress_size,
                    "safe_path": safe_zip_name(member.filename),
                }
            )
        info["members"] = members
        if bad is not None or any(not item["safe_path"] for item in members):
            raise SystemExit(f"ZIP integrity or path check failed: {info}")
        def data_member(name: str) -> bool:
            normalized = name.replace("\\", "/")
            base = normalized.rsplit("/", 1)[-1]
            return (
                normalized.lower().endswith(".dta")
                and not normalized.endswith("/")
                and "__MACOSX/" not in normalized
                and not base.startswith("._")
            )

        dta_members = [item["name"] for item in members if data_member(item["name"])]
        info["ignored_metadata_members"] = [
            item["name"]
            for item in members
            if item["name"].lower().endswith(".dta") and item["name"] not in dta_members
        ]
        if len(dta_members) != 1:
            raise SystemExit(f"Expected one Stata member, found {dta_members}")
        member_name = dta_members[0]
        extract_dir.mkdir(parents=True, exist_ok=True)
        destination = extract_dir / Path(member_name).name
        if not destination.exists() or destination.stat().st_size == 0:
            archive.extract(member_name, path=extract_dir)
            extracted = extract_dir / member_name
            if extracted != destination and extracted.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted.replace(destination)
        info["extracted_member"] = member_name
        info["extracted_path"] = str(destination)
        info["extracted_bytes"] = destination.stat().st_size
    info["extracted_sha256"] = sha256_file(destination)
    return info


def column_map(names: list[str]) -> dict[str, str]:
    folded = {name.casefold(): name for name in names}
    resolved = {}
    missing = []
    for wanted in NEEDED:
        actual = folded.get(wanted.casefold())
        if actual is None:
            missing.append(wanted)
        else:
            resolved[wanted] = actual
    if missing:
        raise SystemExit(f"Required columns missing: {missing}. Header: {names}")
    for wanted in OPTIONAL:
        actual = folded.get(wanted.casefold())
        if actual is not None:
            resolved[wanted] = actual
    return resolved


def peek_header(dta_path: Path) -> dict:
    iterator = pd.read_stata(
        dta_path,
        convert_categoricals=False,
        convert_dates=False,
        order_categoricals=False,
        chunksize=1,
    )
    first = next(iterator)
    closer = getattr(iterator, "close", None)
    if closer:
        closer()
    names = [str(name) for name in first.columns]
    try:
        resolved = column_map(names)
        column_error = None
    except SystemExit as exc:
        resolved = {}
        column_error = str(exc)
    interesting = {"cash", "assets", "date", "rssdid", "dateq", "year", "quarter", "bhcid"}
    dtypes = {
        str(col): str(dtype)
        for col, dtype in first.dtypes.items()
        if str(col).casefold() in interesting or str(col) in resolved.values()
    }
    return {
        "n_columns": len(names),
        "columns": names,
        "resolved": resolved,
        "required_columns_present": column_error is None,
        "required_column_error": column_error,
        "schema_dtypes": dtypes,
    }


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        return False
    return False


def date_key(value) -> int | None:
    if is_missing(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.year * 10000 + value.month * 100 + value.day
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        digits = text.replace("-", "")[:8]
        return int(digits)
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number)


def rssd_key(value) -> str | None:
    if is_missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return f"{int(number):020d}"
    return format(number, ".17g")


def display_number(value):
    if is_missing(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 2**53:
        return int(number)
    return float(number).hex()


def better_earliest(new: dict, old: dict | None) -> bool:
    if old is None:
        return True
    return (new["date_key"], new["rssd_key"], new["row_number"]) < (
        old["date_key"],
        old["rssd_key"],
        old["row_number"],
    )


def better_latest(new: dict, old: dict | None) -> bool:
    if old is None:
        return True
    if new["date_key"] != old["date_key"]:
        return new["date_key"] > old["date_key"]
    if new["rssd_key"] != old["rssd_key"]:
        return new["rssd_key"] < old["rssd_key"]
    return new["row_number"] < old["row_number"]


def float_cash(primary, fill_a, fill_b):
    """Explicit source numerator. Missing primary uses the sum only when both addends exist."""
    if not is_missing(primary):
        cash = float(primary)
        return cash, "primary_rcfd0010"
    if is_missing(fill_a) or is_missing(fill_b):
        return None, "missing_cash"
    return float(fill_a) + float(fill_b), "fallback_rcfd0071_plus_rcfd0081"


def float_ratio(cash, assets):
    if cash is None or is_missing(assets):
        return None, False
    assets_value = float(assets)
    if not (assets_value > MIN_ASSETS and cash >= 0):
        return None, False
    ratio = cash / assets_value
    if not math.isfinite(ratio):
        return None, False
    return ratio, True


def row_record(row_number: int, row: pd.Series, resolved: dict[str, str], category: str) -> dict | None:
    raw = {logical: row[actual] for logical, actual in resolved.items()}
    identifier = rssd_key(raw["rssdid"])
    reporting_date = date_key(raw["date"])
    if identifier is None or reporting_date is None:
        return None
    cash, path = float_cash(raw["rcfd0010"], raw["rcfd0071"], raw["rcfd0081"])
    ratio, eligible = float_ratio(cash, raw["rcfd2170"])
    record = {
        "category": category,
        "row_number_1based": row_number,
        "rssd_key": identifier,
        "rssdid_display": display_number(raw["rssdid"]),
        "date_key": reporting_date,
        "date_iso": f"{reporting_date // 10000:04d}-{(reporting_date // 100) % 100:02d}-{reporting_date % 100:02d}",
        "inputs": {name: display_number(raw[name]) for name in ("rcfd2170", "rcfd0010", "rcfd0071", "rcfd0081")},
        "inputs_hex": {
            name: None if is_missing(raw[name]) else float(raw[name]).hex()
            for name in ("rcfd2170", "rcfd0010", "rcfd0071", "rcfd0081")
        },
        "cash_path": path,
        "cash_display": display_number(cash) if cash is not None else None,
        "cash_hex": None if cash is None else float(cash).hex(),
        "assets_display": display_number(raw["rcfd2170"]),
        "assets_hex": None if is_missing(raw["rcfd2170"]) else float(raw["rcfd2170"]).hex(),
        "script_ratio_hex": None if ratio is None else float(ratio).hex(),
        "script_ratio_repr": None if ratio is None else repr(float(ratio)),
        "row_eligible": eligible,
        "dateq_display": display_number(raw["dateq"]) if "dateq" in raw else None,
        "stored_cash_display": display_number(raw["cash"]) if "cash" in raw else None,
        "stored_assets_display": display_number(raw["assets"]) if "assets" in raw else None,
    }
    return record


def numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def date_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        stamped = series.dt.year * 10000 + series.dt.month * 100 + series.dt.day
        values = stamped.to_numpy(dtype=np.float64)
        values[series.isna().to_numpy()] = np.nan
        return values
    if series.dtype == object:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().any():
            return date_array(parsed)
    return numeric_array(series)


def rssd_array(series: pd.Series) -> np.ndarray:
    keys = np.empty(len(series), dtype=object)
    for index, value in enumerate(series.to_numpy()):
        keys[index] = rssd_key(value)
    return keys


def pick_index(mask: np.ndarray, dates: np.ndarray, rssd_keys: np.ndarray, row_numbers: np.ndarray, latest: bool) -> int | None:
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return None
    date_order = -dates[candidates] if latest else dates[candidates]
    order = np.lexsort((row_numbers[candidates], rssd_keys[candidates], date_order))
    return int(candidates[order[0]])


def select_rows(dta_path: Path, resolved: dict[str, str]) -> dict:
    columns = list(dict.fromkeys(resolved.values()))
    reader = pd.read_stata(
        dta_path,
        columns=columns,
        convert_categoricals=False,
        convert_dates=False,
        order_categoricals=False,
        chunksize=100_000,
    )
    winners = {"A": None, "B": None, "C_primary": None, "C_fallback": None}
    rows_read = 0
    for chunk in reader:
        start = rows_read
        count = len(chunk)
        rows_read += count
        row_numbers = np.arange(start + 1, start + count + 1)
        assets = numeric_array(chunk[resolved["rcfd2170"]])
        primary_cash = numeric_array(chunk[resolved["rcfd0010"]])
        fill_a = numeric_array(chunk[resolved["rcfd0071"]])
        fill_b = numeric_array(chunk[resolved["rcfd0081"]])
        dates = date_array(chunk[resolved["date"]])
        identifiers = rssd_array(chunk[resolved["rssdid"]])
        primary = np.isfinite(primary_cash)
        fallback = ~primary & np.isfinite(fill_a) & np.isfinite(fill_b)
        cash = np.where(primary, primary_cash, np.where(fallback, fill_a + fill_b, np.nan))
        assets_ok = np.isfinite(assets) & (assets > MIN_ASSETS)
        cash_ok = np.isfinite(cash) & (cash >= 0)
        ratio = np.divide(cash, assets, out=np.full_like(cash, np.nan), where=assets_ok)
        identified = np.fromiter((value is not None for value in identifiers), dtype=bool, count=count)
        eligible = assets_ok & cash_ok & np.isfinite(ratio) & np.isfinite(dates) & identified
        dated = dates.astype(np.float64)
        choices = {
            "A": (eligible & primary, False, better_earliest),
            "B": (eligible & fallback & (dated >= DATE_B_MIN), False, better_earliest),
            "C_primary": (
                eligible & primary & (dated >= DATE_C_MIN) & (dated <= DATE_C_MAX),
                True,
                better_latest,
            ),
            "C_fallback": (
                eligible & fallback & (dated >= DATE_C_MIN) & (dated <= DATE_C_MAX),
                True,
                better_latest,
            ),
        }
        for label, (mask, latest, chooser) in choices.items():
            picked = pick_index(mask, dated, identifiers, row_numbers, latest)
            if picked is None:
                continue
            record = row_record(int(row_numbers[picked]), chunk.iloc[picked], resolved, "candidate")
            if record is not None and record["row_eligible"] and chooser(record, winners[label]):
                winners[label] = record
        print(
            f"SCANNED {rows_read} A={None if winners['A'] is None else winners['A']['date_iso']} "
            f"B={None if winners['B'] is None else winners['B']['date_iso']}",
            flush=True,
        )
    chosen_c = winners["C_primary"]
    substitution = None
    if chosen_c is None:
        chosen_c = winners["C_fallback"]
        substitution = "Record C uses the verified fallback path because no eligible primary-cash row was found from 2015-03-31 through 2020-03-31."
    selected = {"A": winners["A"], "B": winners["B"], "C": chosen_c}
    if any(value is None for value in selected.values()):
        raise SystemExit(f"Selection incomplete: { {key: value is not None for key, value in selected.items()} }")
    for category, record in selected.items():
        record["category"] = category
    selected["C"]["substitution"] = substitution
    keys = [(record["rssdid_display"], record["date_key"]) for record in selected.values()]
    banks = [record["rssdid_display"] for record in selected.values()]
    return {
        "rows_read": rows_read,
        "selected": selected,
        "distinct_bank_quarter_keys": len(set(keys)) == 3,
        "distinct_banks": len(set(banks)) == 3,
        "bank_quarter_keys": keys,
        "selection_rules": {
            "A": "earliest eligible row with observed rcfd0010; ties: lower rssdid, then lower 1-based row number",
            "B": "earliest eligible row with missing rcfd0010, both rcfd0071 and rcfd0081 observed, date >= 1984-03-31; same tie break",
            "C": "latest eligible row from 2015-03-31 through 2020-03-31 with observed rcfd0010; ties: lower rssdid, then lower row number",
            "date_field": "date interpreted as the SAS YYYYMMDD reporting date",
            "panel_std_filter": "not applied",
            "bhc_aggregation": "not applied",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=("inspect", "select"), required=True)
    args = parser.parse_args()
    zip_path = SCRATCH / ZIP_NAME
    extract_dir = SCRATCH / "extract"
    if args.command == "inspect":
        payload = inspect_zip(zip_path, extract_dir)
        payload["header"] = peek_header(Path(payload["extracted_path"]))
        destination = SCRATCH / "nyu_zip_inspect.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(destination)
        print("ZIP_BYTES", payload["zip_bytes"])
        print("ZIP_SHA256", payload["sha256"])
        print("DTA", payload["extracted_member"], payload["extracted_bytes"])
        print("DTA_SHA256", payload["extracted_sha256"])
        print("COLUMNS", payload["header"]["n_columns"])
        print("REQUIRED_PRESENT", payload["header"]["required_columns_present"])
        print("RESOLVED", json.dumps(payload["header"]["resolved"]))
        if not payload["header"]["required_columns_present"]:
            print("COLUMN_BLOCKER", payload["header"]["required_column_error"])
        return
    inspect_path = SCRATCH / "nyu_zip_inspect.json"
    inspected = json.loads(inspect_path.read_text(encoding="utf-8"))
    if not inspected["header"]["required_columns_present"]:
        raise SystemExit(
            "Selection not run. The Stata file does not contain the raw items "
            "required by the selection rules: rcfd2170, rcfd0010, rcfd0071, rcfd0081."
        )
    selected = select_rows(Path(inspected["extracted_path"]), inspected["header"]["resolved"])
    selected["versions"] = {
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    destination = SCRATCH / "three_row_selection.json"
    destination.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(destination)
    for category in ("A", "B", "C"):
        record = selected["selected"][category]
        print(
            category,
            record["date_iso"],
            record["rssdid_display"],
            record["cash_path"],
            record["script_ratio_repr"],
            record["row_number_1based"],
        )


if __name__ == "__main__":
    main()
