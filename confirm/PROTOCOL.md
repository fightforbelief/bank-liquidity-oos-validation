# P12 confirmatory replication protocol (QE datasets)

Written 2026-11-18, before any grid, quality screen, schedule, or forecast was produced for the three datasets below. The hypotheses come from exploratory analyses of Bank Cash Liquidity (the discovery dataset). This file is frozen by its SHA-256, which is recorded in `confirm/FREEZE.txt` before the first `prepare.py` run. Any later change is logged with a date in `confirm/DEVIATIONS.md`, and the frozen version stays authoritative.

## Datasets

The other three FTSFR quarterly (`QE`) datasets, built from the same NYU Call Report file (SHA-256 `40692c2e…33ad0`), following jmbejara/ftsfr @ `e740156` `create_ftsfr_datasets.py`:

1. `bank_leverage`: assets / equity by `rssdid`
2. `bhc_leverage`: summed assets / summed equity by `bhcid`
3. `bhc_cash_liquidity`: summed cash / summed assets by `bhcid`

Bank Cash Liquidity is reported alongside but is not confirmatory evidence.

## Fixed design (identical to the discovery run)

- **Arm A:** upstream `build_canonical_grid(frame, "QE")`, which calls `fill_gaps` with a `3mo` step from per-series starts.
- **Arm B:** strict calendar quarter-end grid.
- **Quality screen:** frozen thresholds (total 27, train 18, std 0.01, gap 0.6), the same on both arms.
- **Imputation:** forward-fill of training values only.
- **Targets:** the last six grid positions; one-step forecasts.
- **Models:** six StatsForecast baselines (HistoricAverage, Naive, SeasonalNaive(4), AutoARIMA(4), SES(0.1), AutoTheta(4)).
- **Metric:** seasonal MASE (period 4) on each forecast's own training history.
- **Failures:** a key enters the model comparisons only if all six models have a defined MASE on that arm. Excluded counts are reported by model.
- **Covariates:**
  - *first filing*: the earliest stored date of the entity in the constructed panel.
  - *entrant*: first filing after 1976-03-31 (the panel start).
  - *incumbent*: first filing on 1976-03-31.

## Hypotheses and decision rules (per dataset)

- **H1 (selection mechanism).** Every Arm B-only target belongs to an entity whose first filing is dated 06-30 or 09-30. Confirmed if there is at least one B-only target and the share is exactly 100%.
- **H2 (composition).** The entrant share among Arm A eligible entities is lower than among Arm B eligible entities. Confirmed if strictly lower.
- **H3 (primary; ranking heterogeneity).** On Arm B, with entity-level means over keys, let D = mean over entrants of (MASE_Naive − MASE_AutoTheta) minus the same mean over incumbents. Prediction: D < 0, meaning Naive does relatively better for entrants. Inference: independent entity resampling within each group, 2,000 replicates, `numpy.random.default_rng(20261119)`, type-7 percentile 95% interval. Confirmed if the upper limit is below 0.

**Overall claim.** H3 is "confirmed" across datasets if it is confirmed in at least 2 of the 3. All three results are reported whatever the outcome. Intervals are not multiplicity-adjusted.

## Secondary and descriptive (no decision rule)

- H3 with AutoARIMA in place of AutoTheta.
- H3 within training-length bins (18–30, 31–45, 46–60, 61–80, 81–200 finite training values).
- The pre-specified-style paired Arm B − Arm A estimate on common keys for each model, using the frozen Stage 18 bootstrap (seed 20261027), relative to the Arm A level.
- Full-population model order on each arm.
- Screen first-failure counts by arm.

## Code

`confirm/build_panel.py`, `prepare.py`, `forecast.py`, `analyze.py`, at the commit or hashes recorded in `FREEZE.txt`. No model, threshold, metric, covariate definition, subgroup, or seed will be added or changed after results are seen.
