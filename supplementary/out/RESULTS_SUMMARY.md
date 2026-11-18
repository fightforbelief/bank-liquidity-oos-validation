# Post-hoc results summary

Gate exact_match: True.

## Levels

| Model | Mean A | Mean B | Median entity A | Median entity B | Median key A | Median key B | Est./Mean A | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HistoricAverage | 1.8914 | 1.8871 | 1.4183 | 1.4191 | 1.4101 | 1.4096 | -0.2% | [-0.4%, -0.0%] |
| Naive | 0.8851 | 0.8742 | 0.6540 | 0.6496 | 0.4999 | 0.4974 | -1.2% | [-1.7%, -0.8%] |
| SeasonalNaive | 1.1744 | 1.1708 | 0.8173 | 0.8163 | 0.6492 | 0.6491 | -0.3% | [-0.6%, -0.0%] |
| AutoARIMA | 0.8828 | 0.8722 | 0.6470 | 0.6437 | 0.5119 | 0.5102 | -1.2% | [-1.6%, -0.8%] |
| SES | 1.1528 | 1.1461 | 0.8155 | 0.8136 | 0.7249 | 0.7238 | -0.6% | [-0.9%, -0.3%] |
| AutoTheta | 0.8648 | 0.8532 | 0.6268 | 0.6229 | 0.4877 | 0.4859 | -1.3% | [-1.8%, -0.9%] |

## Distribution

| Model | q05 | q25 | q50 | q75 | q95 | Share <0 | Share =0 | Share >0 | Median [CI] | Trim5 [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| HistoricAverage | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6% | 95.2% | 2.2% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| Naive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0% | 95.3% | 1.8% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| SeasonalNaive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.5% | 95.3% | 2.2% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| AutoARIMA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0% | 95.2% | 1.8% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| SES | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.8% | 95.2% | 1.9% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| AutoTheta | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.1% | 95.2% | 1.7% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

## Decomposition

| Model | Primary | Error [CI] | Scale [CI] | dMAE x 1e-3 [CI] | dMAE % | Share sB>sA | Share sB=sA | Median log(sB/sA) | Share errB<errA | Share err equal |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| HistoricAverage | -0.0043 | -0.0009 [-0.0014, -0.0004] | -0.0034 [-0.0076, 0.0003] | -0.0322 [-0.0520, -0.0135] | -0.1% | 1.5% | 97.1% | 0.0000 | 1.6% | 97.1% |
| Naive | -0.0108 | -0.0082 [-0.0102, -0.0062] | -0.0027 [-0.0061, 0.0007] | -0.3176 [-0.4136, -0.2322] | -1.6% | 1.5% | 97.1% | 0.0000 | 0.8% | 98.6% |
| SeasonalNaive | -0.0036 | 0.0000 [0.0000, 0.0000] | -0.0036 [-0.0075, -0.0000] | 0.0000 [0.0000, 0.0000] | 0.0% | 1.5% | 97.1% | 0.0000 | 0.0% | 100.0% |
| AutoARIMA | -0.0106 | -0.0080 [-0.0101, -0.0062] | -0.0026 [-0.0058, 0.0005] | -0.3113 [-0.4118, -0.2174] | -1.5% | 1.5% | 97.1% | 0.0000 | 1.5% | 97.2% |
| SES | -0.0067 | -0.0032 [-0.0041, -0.0024] | -0.0034 [-0.0073, 0.0000] | -0.1246 [-0.1617, -0.0856] | -0.4% | 1.5% | 97.1% | 0.0000 | 1.8% | 97.1% |
| AutoTheta | -0.0116 | -0.0086 [-0.0106, -0.0068] | -0.0030 [-0.0067, 0.0004] | -0.3269 [-0.4159, -0.2419] | -1.6% | 1.5% | 97.1% | 0.0000 | 1.6% | 97.1% |

Zero lag-4 share A 0.3%, B 0.3%.
Imputed training share A 1.8%, B 0.4%.

## Origin contribution

| Model | Diff term | Same term | Total | Share from diff |
|---|---:|---:|---:|---:|
| HistoricAverage | -0.0017 | -0.0026 | -0.0043 | 39.1% |
| Naive | -0.0092 | -0.0017 | -0.0108 | 84.7% |
| SeasonalNaive | -0.0014 | -0.0022 | -0.0036 | 39.3% |
| AutoARIMA | -0.0090 | -0.0016 | -0.0106 | 84.9% |
| SES | -0.0032 | -0.0035 | -0.0067 | 48.1% |
| AutoTheta | -0.0092 | -0.0023 | -0.0116 | 80.0% |

| Keys | Arm | Synthetic share | Median staleness | Mean staleness |
|---|---|---:|---:|---:|
| diff | A | 100.0% | 2.00 | 2.01 |
| diff | B | 0.2% | 0.00 | 0.01 |
| same | A | 0.0% | 0.00 | 0.00 |
| same | B | 0.0% | 0.00 | 0.00 |

| Arm A month-day | Arm B month-day | Keys |
|---|---|---:|
| 03-30 | 03-31 | 1190 |

## Grid anchor

Drift phrase: only from day-30 anchors.
Entities with an off-quarter-end stamp: 835 of 17383.

| Month-day | Stamps |
|---|---:|
| 03-30 | 13738 |
| 03-31 | 412053 |
| 06-30 | 418982 |
| 09-30 | 416106 |
| 12-30 | 13738 |
| 12-31 | 400052 |

| First stamp | Entities | Entities with an off-quarter-end stamp |
|---|---:|---:|
| 03-31 | 15058 | 0 |
| 06-30 | 456 | 456 |
| 09-30 | 379 | 379 |
| 12-31 | 1490 | 0 |

## Exclusive keys

### arm_b_only (12594)

| Rule | Count |
|---|---:|
| entity_not_in_A | 10937 |
| - no per-entity screen log in the saved run | 10937 |
| target_not_on_A_grid | 1657 |
| - 03-30 | 828 |
| - 12-30 | 829 |
| target_on_A_grid_missing | 0 |
| failed_A_eligibility | 0 |
| other | 0 |

### arm_a_only (18)

| Rule | Count |
|---|---:|
| entity_not_in_B | 18 |
| - no per-entity screen log in the saved run | 18 |
| target_not_on_B_grid | 0 |
| target_on_B_grid_missing | 0 |
| failed_B_eligibility | 0 |
| other | 0 |

## Notes against the manuscript text

Saved Arm A grids: drift occurs only from day-30 anchors. The manuscript illustrates a 30 June step of 30 September, 30 December, 30 March, 30 June. Error and scale terms are not both strictly negative or both strictly positive for: SeasonalNaive error 0.0000 scale -0.0036. The aggregate model order is the same on Arm A and Arm B. Unjoined source filings are not counted. The saved grid does not contain dates that failed to join it.

## BLOCKERS.md

# Blockers

## 2026-11-06 — unjoined source filings

The saved Arm A grid and the saved schedule do not contain source filing dates that failed to join a grid stamp. Counting those filings would require the call-report panel. That file was not opened. No unjoined-filing count is reported.

## 2026-11-06 — reproduction gate failed

T2--T10 were not run. `reproduce_primary` and `cluster_bootstrap` were not modified.
The draw passed to every gate bootstrap was `Generator.choice(n, size=n, replace=True)`.

- primary HistoricAverage: got -0.004283616827738098 [-0.008591823072073467, -0.0006109514586281041], recorded -0.0042836168277380965 [-0.008591823072073467, -0.000610951458628103], valid 2000, failed 0.
- primary Naive: got -0.010815696110028987 [-0.015259849607384432, -0.006857415193080016], recorded -0.010815696110028986 [-0.015259849607384432, -0.006857415193080016], valid 2000, failed 0.
- primary SeasonalNaive: got -0.003606282624498774 [-0.00774629104284172, -0.00026764583688035294], recorded -0.003606282624498774 [-0.00774629104284172, -0.0002676458368803533], valid 2000, failed 0.
- primary AutoARIMA: got -0.010605197991587254 [-0.014542165346988973, -0.006979203762667185], recorded -0.010605197991587254 [-0.014542165346988973, -0.006979203762667186], valid 2000, failed 0.
- primary SimpleExponentialSmoothing: got -0.006683875293450963 [-0.011230451149514733, -0.003131373227894662], recorded -0.006683875293450963 [-0.011230451149514731, -0.0031313732278946617], valid 2000, failed 0.
- primary AutoTheta: got -0.011553343325476726 [-0.015839678595276147, -0.007630739438669055], recorded -0.011553343325476726 [-0.015839678595276144, -0.0076307394386690534], valid 2000, failed 0.
- secondary HistoricAverage: got -0.005146063610992731 [-0.009309246208334955, -0.001854565732614158], recorded -0.005146063610992731 [-0.009309246208334955, -0.0018545657326141576], valid 2000, failed 0.
- secondary Naive: got -0.0030933316554803522 [-0.005993370429479464, -0.000970532083135954], recorded -0.0030933316554803522 [-0.005993370429479462, -0.0009705320831359538], valid 2000, failed 0.
- secondary SeasonalNaive: got -0.004292877329541735 [-0.008381815923470195, -0.0011985408033110146], recorded -0.004292877329541734 [-0.008381815923470195, -0.0011985408033110146], valid 2000, failed 0.
- secondary AutoARIMA: got -0.0027435544033859714 [-0.005794732292470179, -0.0002886055457325237], recorded -0.0027435544033859705 [-0.0057947322924701795, -0.0002886055457325237], valid 2000, failed 0.
- secondary SimpleExponentialSmoothing: got -0.0068507957197966665 [-0.010837846649293232, -0.0036323368098023084], recorded -0.006850795719796667 [-0.010837846649293232, -0.003632336809802309], valid 2000, failed 0.

Point-only diagnostic using mean(arm B) - mean(arm A) inside each entity, then a NumPy mean across entities:
- HistoricAverage: primary -0.0042836168277380965 recorded_match True; secondary -0.005146063610992731 recorded_match True.
- Naive: primary -0.010815696110028986 recorded_match True; secondary -0.0030933316554803522 recorded_match True.
- SeasonalNaive: primary -0.003606282624498774 recorded_match True; secondary -0.004292877329541734 recorded_match True.
- AutoARIMA: primary -0.010605197991587254 recorded_match True; secondary -0.0027435544033859705 recorded_match True.
- SimpleExponentialSmoothing: primary -0.006683875293450963 recorded_match True; secondary -0.006850795719796667 recorded_match True.
- AutoTheta: primary -0.011553343325476726 recorded_match True; secondary -0.004192768516977067 recorded_match True.
The recorded point estimates match a NumPy mean of the entity-level contrast mean(arm B) - mean(arm A). `reproduce_primary` instead averages the pandas mean of `B_mase - A_mase`. Where those two differ, the difference is one unit in the last place. Some interval endpoints also differ by one unit in the last place. All 2,000 replicates were finite. The supplied estimator was not edited to force the recorded reprs.

## 2026-11-18 — reproduction gate: resolution

The 2026-11-06 gate blocker above came from floating-point summation order in the reference re-implementation, not from the inputs. The gate now runs the frozen Stage 18 code path (see DEVIATIONS.md, 2026-11-18) with the unchanged exact-match criterion. The gate result for that run is in `supplementary/out/gate.json`.

## DEVIATIONS.md

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
