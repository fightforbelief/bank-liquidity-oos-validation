"""Build one FTSFR QE target panel (unique_id, ds, y) from the NYU Call Report file.

Follows jmbejara/ftsfr @ e740156, src/nyu_call_report/create_ftsfr_datasets.py and
pull_nyu_call_report.py, then the forecast loader's float32 cast with non-finite
values dropped (as Stage 10 did for Bank Cash Liquidity). Usage:
    python confirm/build_panel.py bank_leverage
Writes <run>/panel.parquet and <run>/panel_summary.json. No grid, screen, or model.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import common as C


def load_raw() -> pd.DataFrame:
    df = pd.read_stata(C.DTA, columns=["rssdid", "bhcid", "date", "assets", "equity", "cash"],
                       convert_categoricals=False, convert_dates=False)
    # pull_nyu_call_report: ids and date -> int -> str, 99999 placeholder -> NA
    for col in ("rssdid", "bhcid", "date"):
        df[col] = df[col].fillna(99999).astype("int64").astype(str).replace("99999", pd.NA)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def drop_constant(frame: pd.DataFrame, key: str) -> tuple[pd.DataFrame, int]:
    std = frame.groupby(key)["y"].std()
    constant = std.index[std == 0]
    return frame.loc[~frame[key].isin(constant)], int(len(constant))


def construct(target: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info: dict = {"raw_rows": int(len(df))}
    if target == "bank_leverage":
        d = df[(df["equity"] > 1e-10) & (df["assets"] > 0)].copy()
        d["y"] = d["assets"] / d["equity"]
        d = d[["rssdid", "date", "y"]].sort_values(["rssdid", "date"]).dropna()
        d = d[~d["y"].isin([np.inf, -np.inf])].rename(columns={"rssdid": "unique_id", "date": "ds"})
    elif target == "bhc_leverage":
        g = df[["bhcid", "date", "assets", "equity", "cash"]].groupby(["bhcid", "date"]).sum().reset_index()
        g = g[(g["equity"] > 1e-10) & (g["assets"] > 0)].copy()
        g["y"] = g["assets"] / g["equity"]
        g = g.rename(columns={"bhcid": "unique_id", "date": "ds"})
        g = g[g["unique_id"] != "0"].dropna()
        d = g[~g["y"].isin([np.inf, -np.inf])][["unique_id", "ds", "y"]]
    elif target == "bhc_cash_liquidity":
        g = (df[["bhcid", "date", "cash", "assets"]].groupby(["bhcid", "date"]).sum()
             .sort_values(by=["bhcid", "date"]).reset_index())
        g = g[(g["assets"] > 1e-10) & (g["cash"] >= 0)].copy()
        g["y"] = g["cash"] / g["assets"]
        g = g.rename(columns={"bhcid": "unique_id", "date": "ds"})
        g = g[g["unique_id"] != "0"].dropna()
        g = g[~g["y"].isin([np.inf, -np.inf])]
        g, info["constant_series_removed"] = drop_constant(g, "unique_id")
        d = g[["unique_id", "ds", "y"]]
    else:
        raise SystemExit(f"unknown target {target}")
    # forecast loader: y as float32, non-finite -> null (dropped here, as in Stage 10)
    y32 = d["y"].to_numpy(dtype=np.float32).astype(np.float64)
    d = d.assign(y=y32)
    info["nonfinite_after_float32"] = int((~np.isfinite(y32)).sum())
    d = d.loc[np.isfinite(d["y"].to_numpy())].reset_index(drop=True)
    d["unique_id"] = d["unique_id"].astype(str)
    md = d["ds"].dt.strftime("%m-%d")
    info.update({
        "rows": int(len(d)),
        "entities": int(d["unique_id"].nunique()),
        "non_quarter_end_dates": int((~md.isin(["03-31", "06-30", "09-30", "12-31"])).sum()),
        "duplicate_keys": int(d.duplicated(["unique_id", "ds"]).sum()),
        "date_min": str(d["ds"].min().date()),
        "date_max": str(d["ds"].max().date()),
    })
    if info["duplicate_keys"]:
        raise SystemExit("STOP duplicate entity-date keys")
    return d, info


def main() -> int:
    target = sys.argv[1]
    run = C.run_dir(target)
    df = load_raw()
    panel, info = construct(target, df)
    info["dta_sha256"] = C.sha256(C.DTA)
    panel.to_parquet(run / "panel.parquet", index=False)
    info["panel_sha256"] = C.sha256(run / "panel.parquet")
    (run / "panel_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(target, json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
