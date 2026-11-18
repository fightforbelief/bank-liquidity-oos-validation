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
