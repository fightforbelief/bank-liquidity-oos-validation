# Deviations and pre-registered implementation notes

## 2026-11-06 — repository location

The task names `Desktop/projects/p12-bank-liquidity-oos-validation`. That directory is absent. The live git repository is `Desktop/projects/Research_Portfolio_2026/p12-bank-liquidity-oos-validation`. Frozen files were not copied or rewritten.

## 2026-11-06 — gitignore

`*.md` is ignored, which would drop this protocol from the required commit. Narrow exceptions were added for `supplementary/POSTHOC_PROTOCOL.md`, `supplementary/DEVIATIONS.md`, `supplementary/BLOCKERS.md`, and `supplementary/out/RESULTS_SUMMARY.md`. No other ignore rule was broadened.

## 2026-11-06 — draw function

The supplied `default_draw` uses `rng.integers`. The frozen Stage 18 bootstrap uses `rng.choice(n, size=n, replace=True)`. Every bootstrap in this run, including the reproduction gate, receives that draw. This is the substitution the task specifies.

## 2026-11-06 — MASE denominator

Forecast batches have no denominator column. `load.py` derives `mase_den` from `prepared_series.parquet` column `y_imputed` by the retained-position rule: drop leading non-finite imputed values, reject a later non-finite imputed value, and take the mean absolute lag-4 difference. The loader then checks the stored MASE identity. The call-report panel is not read.

## 2026-11-06 — unjoined filings

`prepared_series` stores grid rows only. Source dates that do not meet an Arm A stamp are not in that file. Task A6's unjoined-filing count is not recomputed from the call-report panel.

## 2026-11-06 — exclusive-key eligibility

The Stage 18 run directory has no per-entity quality-screen log, so `entity_not_in_A` and `entity_not_in_B` are not sub-split by a failed criterion. When a key reaches the eligibility rule, the frozen thresholds are applied in the frozen first-failure order on the saved grid: total length below 27, fewer than 18 finite source values on stamps before the target, training standard deviation below 0.01 with divisor n-1 when more than one finite value is present, then gap fraction above 0.6. A target that passes these checks and is still absent from the other arm's schedule is class `other`.

## 2026-11-06 — ranking seed

The supplied `ranking` function draws one joint bootstrap for all six models. It is not split into six generators.

## 2026-11-06 — test runner

The Stage 15 environment has no pytest package, and no package was installed. The synthetic tests run with the Stage 15 interpreter. Pytest is imported from the repository `.venv` after that directory is appended to `sys.path`, so NumPy, pandas, and SciPy remain the Stage 15 copies.

## 2026-11-06 — table formatting

Generated tables are only `tabular` environments. `tab_origin` and `tab_ranking` contain the two requested panels, and `tab_origin` adds the month-day tabulation required by the different-origin task.
