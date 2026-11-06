"""Hash the read-only inputs and write POSTHOC_PROTOCOL.md. No estimator is run."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(r"C:\Users\hanqi\Desktop\projects\_p12_stage18_full_run\run_20261028")
PROTOCOL = ROOT / "supplementary" / "POSTHOC_PROTOCOL.md"
PLACEHOLDER = "<paths + SHA-256 filled in before the first run>"
TEXT = """P12 post-hoc protocol, written 2026-11-06, after the pre-specified results (Tables 2-5) were seen
and before any analysis below was run on real data.

Status: exploratory. None of these analyses alters the pre-specified estimand, its intervals, or
its interpretation. No model is refitted, no forecast regenerated, no grid rebuilt for scoring.

Inputs (read-only): <paths + SHA-256 filled in before the first run>

Gate: reproduce the 36 pre-specified values in Tables 4-5 exactly with seed 20261027; otherwise stop.

Analyses, on the 85,205 primary common keys and 17,377 entities, per model, in the frozen order:
  A1 MASE levels and relative change (estimate / mean entity MASE on Arm A).
  A2 Distribution of delta_i: quantiles 5/25/50/75/95, sign shares, median, 5% trimmed mean.
  A3 Exact Shapley split of delta into an error term and a scale term; unscaled MAE change;
     lag-4 zero-difference and imputation shares by arm.
  A4 Exact split of the primary estimate into different-origin and same-origin contributions;
     origin synthetic share and staleness by arm.
  A5 Ranking stability: per-entity Kendall tau-b, same-best share, identical-ranking share,
     mean ranks, aggregate order by mean entity MASE.
  A6 Coverage (model-independent): Arm A grid stamp month-days, anchor dependence, unjoined filings,
     rule-based classification of the 12,594 Arm B-only and 18 Arm A-only keys.

Inference: entity-cluster bootstrap, 2,000 replicates, the project's exact draw function, fresh
PCG64(20261106) per model per analysis, type-7 percentile 95% intervals, non-finite replicates
counted as failed and not replaced. Intervals are not multiplicity-adjusted. No equivalence margin.

No additional models, metrics, grids, subsets, or seeds will be added after results are seen;
any deviation is logged with a date in DEVIATIONS.md.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files() -> list[Path]:
    batches = sorted((RUN / "batches").glob("batch_*.parquet"))
    files = [
        RUN / "schedule.parquet",
        RUN / "primary_pairs.parquet",
        RUN / "prepared_series.parquet",
        RUN / "primary_results.json",
        RUN / "secondary_results.json",
        *batches,
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing or len(batches) != 98:
        raise SystemExit(f"STOP input list is incomplete batches={len(batches)} missing={len(missing)}")
    return files


def main() -> None:
    if (ROOT / "supplementary" / "out" / "gate.json").exists() or (ROOT / "supplementary" / "out" / "results.json").exists():
        raise SystemExit("STOP results already exist; the protocol is not rewritten")
    if PLACEHOLDER not in TEXT:
        raise SystemExit("STOP protocol template lost its placeholder")
    lines = []
    for path in input_files():
        digest = sha256(path)
        lines.append(f"- {path.as_posix()} SHA-256 {digest}")
        print(path.name, digest, flush=True)
    filled = TEXT.replace(PLACEHOLDER, "\n" + "\n".join(lines))
    if PLACEHOLDER in filled:
        raise SystemExit("STOP placeholder remained")
    PROTOCOL.write_text(filled, encoding="utf-8")
    print("WROTE", PROTOCOL, flush=True)


if __name__ == "__main__":
    main()
