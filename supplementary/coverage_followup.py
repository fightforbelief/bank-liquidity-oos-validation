"""Descriptive follow-up to A6 (logged in DEVIATIONS.md, 2026-11-18).

For Arm B-only keys whose bank has no Arm A schedule rows, describe those banks
on the saved grids: grid anchor, whether the Arm A grid has an off-quarter-end
stamp, and the share of grid stamps carrying a finite stored value in each arm.
Model-independent. No bootstrap, no estimand, no refit. Reads the same frozen
inputs as run_posthoc.py and writes supplementary/out/coverage_followup.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import coverage
import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out" / "coverage_followup.json"
QUARTER_ENDS = {"03-31", "06-30", "09-30", "12-31"}


def main() -> int:
    inputs = load.read_inputs()
    schedule, prepared = inputs["schedule"], inputs["prepared"]
    grids = coverage._grids_by_arm(prepared)
    ent = {"A": set(), "B": set()}
    keys = {"A": set(), "B": set()}
    for row in schedule.itertuples(index=False):
        ent[row.arm_id].add(row.entity_hash)
        keys[row.arm_id].add((row.entity_hash, row.target_timestamp))
    b_only = keys["B"] - keys["A"]
    missing = sorted({e for e, _ in b_only if e not in ent["A"]})
    n_keys_missing = sum(1 for e, _ in b_only if e not in ent["A"])

    def describe(entities: list[str]) -> dict:
        anchor, drift, in_grid, anchor_b = Counter(), Counter(), Counter(), Counter()
        fin_a, fin_b = [], []
        for e in entities:
            ga, gb = grids["A"].get(e), grids["B"].get(e)
            in_grid["A" if ga is not None else "not_in_A_grid"] += 1
            if ga is not None:
                anchor[coverage._monthday(ga["ts"][0])] += 1
                off = any(coverage._monthday(t) not in QUARTER_ENDS for t in ga["ts"])
                drift["off_quarter_end_stamp" if off else "all_quarter_end"] += 1
                fin_a.append(float(np.mean(ga["finite"])))
            if gb is not None:
                fin_b.append(float(np.mean(gb["finite"])))
                # First Arm B stamp that carries a finite stored value: the quarter of the first filing.
                first = next((t for t, f in zip(gb["ts"], gb["finite"]) if f), None)
                anchor_b[coverage._monthday(first) if first else "none"] += 1
        return {
            "n_entities": len(entities),
            "in_A_grid": dict(in_grid),
            "A_anchor_monthday": dict(sorted(anchor.items())),
            "A_grid_drift": dict(drift),
            "B_first_filing_quarter_end": dict(sorted(anchor_b.items())),
            "finite_stamp_share_A_mean": float(np.mean(fin_a)) if fin_a else None,
            "finite_stamp_share_A_median": float(np.median(fin_a)) if fin_a else None,
            "finite_stamp_share_B_mean": float(np.mean(fin_b)) if fin_b else None,
            "finite_stamp_share_B_median": float(np.median(fin_b)) if fin_b else None,
        }

    common = sorted(ent["A"] & ent["B"])
    payload = {
        "b_only_keys_entity_not_in_A": n_keys_missing,
        "entities_not_in_A_schedule": describe(missing),
        "entities_in_both_schedules": describe(common),
        "n_entities_schedule": {"A": len(ent["A"]), "B": len(ent["B"])},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
