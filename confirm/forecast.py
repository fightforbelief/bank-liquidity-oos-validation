"""One-step forecasts for one target, both arms, six frozen StatsForecast baselines.

The per-batch logic is the Stage 18 run_batch (same adapter, same MASE), run in
parallel worker processes. Resumable: an existing batch file is skipped. Usage:
    python confirm/forecast.py bank_leverage [workers]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

import common as C

BATCH_SIZE = 200


def _factories():
    from statsforecast.models import (AutoARIMA, AutoTheta, HistoricAverage, Naive, SeasonalNaive,
                                      SimpleExponentialSmoothing)
    return (
        ("HistoricAverage", HistoricAverage),
        ("Naive", Naive),
        ("SeasonalNaive", lambda: SeasonalNaive(season_length=4)),
        ("AutoARIMA", lambda: AutoARIMA(season_length=4)),
        ("SimpleExponentialSmoothing", lambda: SimpleExponentialSmoothing(alpha=0.1)),
        ("AutoTheta", lambda: AutoTheta(season_length=4)),
    )


def history_for(frame: pd.DataFrame):
    from p12_bank_liquidity.candidate_a.timestamp_adapter import HistoryPoint
    grouped = {}
    for r in frame.itertuples(index=False):
        grouped.setdefault((r.entity_hash, r.arm_id), []).append(HistoryPoint(
            entity_id=r.entity_hash, arm_id=r.arm_id, timestamp=str(r.timestamp)[:10],
            value=None if pd.isna(r.value) else float(r.value), evidence_label=C.LABEL,
            y_imputed=None if pd.isna(r.y_imputed) else float(r.y_imputed),
            imputation_applied=True, synthetic_grid_row=bool(r.synthetic_grid_row)))
    return grouped


def run_batch(schedule_rows, history):
    """Stage 18 run_batch with the confirmatory label; per-model failures are recorded, not raised."""
    from p12_bank_liquidity.candidate_a.metrics import seasonal_mase
    from p12_bank_liquidity.candidate_a.statsforecast_bridge import StatsForecastOneStep
    from p12_bank_liquidity.candidate_a.timestamp_adapter import ScheduleRow, run_one_step, training_values
    results = []
    by_entity = {}
    for rec in schedule_rows:
        by_entity.setdefault(rec.entity_hash, {}).setdefault(rec.arm_id, []).append(rec)
    for entity, arms in by_entity.items():
        for arm, rows in arms.items():
            points = history.get((entity, arm), [])
            index = {p.timestamp: p for p in points}
            schedule, observed = [], {}
            for rec in rows:
                tp = index.get(rec.target_timestamp)
                if tp is None:
                    raise SystemExit("STOP target timestamp missing from prepared grid")
                observed[rec.target_timestamp] = tp.value
                schedule.append(ScheduleRow(
                    entity_id=entity, arm_id=arm, origin_timestamp=rec.origin_timestamp,
                    target_timestamp=rec.target_timestamp, target_quarter=rec.target_quarter,
                    observed_finite_source_target=tp.value is not None and not tp.synthetic_grid_row,
                    synthetic_grid_row=bool(tp.synthetic_grid_row), evidence_label=C.LABEL))
            for model_id, factory in _factories():
                outcomes = run_one_step(schedule, points, StatsForecastOneStep(factory(), model_id))
                for out in outcomes:
                    row = next(r for r in schedule if r.target_timestamp == out.target_timestamp)
                    if out.origin_timestamp != row.origin_timestamp:
                        raise SystemExit("STOP origin timestamp was rewritten")
                    status = reason = value = None
                    if out.status == "RECORDED_FORECAST":
                        scored = seasonal_mase(training_values(points, row), [float(observed[out.target_timestamp])],
                                               [float(out.prediction)], evidence_label=C.LABEL)
                        status, reason, value = scored.status, scored.reason, scored.value
                    results.append({
                        "entity_hash": entity, "arm_id": arm, "model_id": model_id,
                        "origin_timestamp": out.origin_timestamp, "target_timestamp": out.target_timestamp,
                        "status": out.status, "reason": out.reason, "prediction": out.prediction,
                        "y_true": observed[out.target_timestamp],
                        "mase_status": status, "mase_reason": reason, "mase_value": value})
    return results


def work(job):
    run, index, entities = job
    final = Path(run) / "batches" / f"batch_{index:05d}.parquet"
    if final.exists():
        return index, 0
    warnings.filterwarnings("ignore")
    sched = pd.read_parquet(Path(run) / "schedule.parquet", filters=[("entity_hash", "in", entities)])
    table = ds.dataset(Path(run) / "prepared_series.parquet").to_table(filter=ds.field("entity_hash").isin(entities))
    produced = run_batch(list(sched.itertuples(index=False)), history_for(table.to_pandas()))
    if len(produced) != len(sched) * 6:
        raise SystemExit(f"STOP batch {index} produced {len(produced)} for {len(sched)} rows")
    tmp = final.with_suffix(".parquet.tmp")
    pd.DataFrame(produced).to_parquet(tmp, index=False)
    os.replace(tmp, final)
    return index, len(produced)


def main() -> int:
    target = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run = C.run_dir(target)
    entities = sorted(pd.read_parquet(run / "schedule.parquet", columns=["entity_hash"])["entity_hash"].unique())
    jobs = [(str(run), i + 1, entities[o:o + BATCH_SIZE]) for i, o in enumerate(range(0, len(entities), BATCH_SIZE))]
    started = time.perf_counter()
    done = 0
    with Pool(workers) as pool:
        for index, rows in pool.imap_unordered(work, jobs):
            done += 1
            progress = {"target": target, "batches_done": done, "batches_total": len(jobs),
                        "elapsed_seconds": round(time.perf_counter() - started, 1)}
            (run / "progress.json").write_text(json.dumps(progress) + "\n", encoding="utf-8")
            print("BATCH", index, rows, f"{done}/{len(jobs)}", flush=True)
    print("FORECAST_DONE", target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
