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

## 2026-11-18 — gate uses the frozen Stage 18 code path

The 2026-11-06 gate failed only in the last unit of precision (see BLOCKERS.md). Cause: the reference `reproduce_primary` forms each entity delta as a pandas mean of `B_mase - A_mase` and bootstraps with pandas means, whereas the frozen pipeline forms `sum(B)/n - sum(A)/n` in Python floats (`candidate_a/aggregation.py`) and bootstraps a NumPy array with `candidate_a/stage18_inference.cluster_bootstrap`. The quantities are algebraically identical; only floating-point summation order differs.

The gate in `run_posthoc.py` now builds entity deltas with the frozen formula, in first-appearance order of the saved pair file, and calls `stage18_inference.cluster_bootstrap` itself. The pass criterion is unchanged: exact `repr` equality for all 36 values, 2,000 valid replicates, 0 failed. `reproduce_primary`, `cluster_bootstrap`, and every post-hoc estimator in `p12_supplementary.py` are unchanged. `supplementary/tests/test_frozen_gate.py` checks on synthetic data that the gate's per-entity deltas equal the frozen aggregator's bit for bit and that the gate returns the frozen bootstrap's output.

The post-hoc analyses use the reference estimators, so a post-hoc point estimate of the primary contrast may differ from the recorded value in the last unit of precision. Every post-hoc identity check uses a 1e-12 tolerance.

## 2026-11-06 — decomposition sentence
The supplied template says the scale term is the remainder of an error-term share between 0% and 100%. At least one share lies outside that interval, so abs-decomp-summary states the signed range instead.

## 2026-11-18 — coverage follow-up (added after results were seen)

The A6 classification put 10,937 of the 12,594 Arm B-only keys in `entity_not_in_A`, which does not by itself say why those banks are absent. `supplementary/coverage_followup.py` describes those banks on the same saved inputs: the month-day of the first Arm B stamp carrying a stored filing, presence in the saved Arm A grid, and the share of grid stamps with a finite stored value. It is model-independent and descriptive: no bootstrap, no estimand, no refit, no new seed. Output: `supplementary/out/coverage_followup.json`, and the generated key `coverage-followup-text`. Because it was specified after the A6 counts were seen, the manuscript labels it exploratory.

## 2026-11-18 — degenerate distribution summaries

Every quantile of delta_i, its median, and its 5% trimmed mean are exactly zero for every model, because delta_i is exactly zero for 95.2–95.3% of banks. The template sentence for `dist-text` then reads "ranges from 0.0000 to 0.0000" and lists every interval as containing zero. In that case `make_tex.py` now writes a sentence that states the zero share and the sign shares, and `tab_dist` reports the sign split (delta_i < 0, = 0, > 0) in place of the five all-zero quantile columns. The median and trimmed-mean columns are kept. No estimate changed. The same pass reports an error-term share that rounds to 0.0% as 0 rather than as a signed zero, so `abs-decomp-summary` uses the supplied template; this supersedes the 2026-11-06 "decomposition sentence" entry above, which was caused by a share of -0.0% for SeasonalNaive.
