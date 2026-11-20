# Who Gets Benchmarked? A quarterly date grid that silently drops one in three entrants

Paper: [`paper.pdf`](paper.pdf). LaTeX source: [`paper_v3/`](paper_v3/).

The open financial forecasting benchmark of Bejarano et al. (OFR Working Paper 26-05, FTSFR) builds its quarterly grid by stepping three months from each series' first date. For a series that starts on 30 June or 30 September, that grid lands on 30 December and 30 March. Year-end and first-quarter Call Report filings, dated the 31st, then fail to join. We compare this grid (Arm A) with a calendar quarter-end grid (Arm B) on the benchmark's four quarterly Call Report datasets. Target, quality screen, six StatsForecast baselines, horizon, and seasonal MASE are held fixed.

## Findings

**For banks, the grid barely changes the forecasts it keeps.** On Bank Cash Liquidity, the pre-specified paired difference is negative for all six models, with all 95% intervals below zero. The differences are only 0.2–1.3% of the MASE level, and model rankings are unchanged.

**It changes who is evaluated.** The quality screen removes 2,190 banks under Arm A, all with a 30 June or 30 September first filing. Arm A matches the published count of 17,383 banks; the calendar grid retains 19,567. Arm A loses 35% of entrants eligible on either arm, where entry means first observation after 31 March 1976.

**The removed population ranks the models differently.** Among entrants, Naive beats AutoTheta and AutoARIMA. Among banks present since 1976, the order reverses. The difference persists in several exploratory strata and is largest after abrupt level shifts. These analyses are exploratory.

**Protocol-based replication across related panels.** The protocol was frozen by SHA-256 before the other three datasets were processed.

| Dataset | H1: all lost targets from day-30 starters | H2: entrant share lower on Arm A | H3: entrants relatively favour Naive (vs AutoTheta) |
|---|---|---|---|
| Bank Leverage | yes (100%) | yes | confirmed, −0.009 [−0.019, −0.001] |
| BHC Leverage | yes (100%) | yes | not confirmed, +0.011 [−0.006, +0.028] |
| BHC Cash Liquidity | yes (100%) | yes | confirmed, −0.024 [−0.042, −0.005] |

Arm A loses 34–36% of entering entities in every dataset.

## Repository map

| Path | Contents |
|---|---|
| `src/p12_bank_liquidity/` | Frozen project code: grid semantics, schedule, MASE, Stage 18 bootstrap |
| `tests/` | Unit tests for the frozen code |
| `supplementary/` | Post-hoc protocol (`POSTHOC_PROTOCOL.md`), reproduction gate, exploratory analyses, `DEVIATIONS.md`, `BLOCKERS.md` |
| `supplementary/out/` | Aggregate results only: gate, post-hoc results, coverage and entrant analyses |
| `confirm/` | Confirmatory protocol (`PROTOCOL.md`), freeze hashes (`FREEZE.txt`), pipeline, `DEVIATIONS.md` |
| `confirm/out/` | Aggregate confirmatory results for each dataset |
| `paper_v3/` | Manuscript source and generated tables |
| `manuscripts/`, `paper_v2/` | Earlier manuscript versions |

## Study timeline

1. **2026-10.** Pre-specified protocol and full-sample forecasts for Bank Cash Liquidity: the paired estimate.
2. **2026-11-06.** Post-hoc protocol committed before the exploratory analyses.
3. **2026-11-18.** Reproduction gate passes bit-for-bit, using the frozen Stage 18 code path. Exploratory entrant analyses follow.
4. **2026-11-18.** Confirmatory protocol frozen, then run on the three other datasets.

Every deviation is logged with a date in the `DEVIATIONS.md` files.

## Data

The Call Report file, constructed panels, saved forecasts, and grids are not in this repository. The NYU Call Report data page permits non-commercial use. The repository contains only code and aggregate statistics, with no bank identifiers.

## Run

With `PYTHONPATH=src`:

```text
python -m pytest -q tests supplementary/tests
```

The confirmatory pipeline uses `confirm/build_panel.py`, `prepare.py`, `forecast.py`, and `analyze.py`. Each takes a target name. Paths are set in `confirm/common.py`.

### Rebuild the reported tables

From a clean checkout run `python confirm/make_tex.py confirm/out`, then compile `paper_v3/main.tex` from its own directory with Tectonic or two passes of pdfLaTeX. The three confirmatory table/value files are generated from archived JSON, with no microdata or model fits required. On the Windows project host use the root `.venv\Scripts\python.exe`. The test dependencies are in `requirements-lock.txt`.

### Recompute the three additional panels

Obtain the basic 1976–2020 bank-level file from the [NYU Call Report data page](https://pages.stern.nyu.edu/~pschnabl/data/data_callreport.htm). The extracted Stata file has SHA-256 `40692c2ec9af404b3148ffbd7ad970a4ce191d9ac8f781b415e8a10909533ad0`. Set `P12_DTA` to that file, `P12_UPSTREAM` to the directory containing [robust_preprocessing.py at the inspected commit](https://github.com/jmbejara/ftsfr/blob/e740156e22b04498c74525f8db6b95f359ab596a/src/forecasting/robust_preprocessing.py), and `P12_RUNS` to an empty output directory. The forecast runner resumes any existing batch files in that directory.

For each of `bank_leverage`, `bhc_leverage`, and `bhc_cash_liquidity`, pass the target as the first argument to these scripts in order:

1. `confirm/build_panel.py` constructs the panel and records input hashes.
2. `confirm/prepare.py` creates the grids and evaluation schedule.
3. `confirm/forecast.py` fits the six models.
4. `confirm/analyze.py` computes the contrasts. Compare each target's analysis.json with its archived counterpart in `confirm/out/`.

For a raw-data rerun, run construction and preparation for all three targets using NumPy 2.0.2, pandas 2.2.3, PyArrow 17.0.0, Polars 1.31.0 and utilsforecast 0.2.16. For the forecast step install StatsForecast 2.1.1, utilsforecast 0.2.17 and PyArrow 26.0.0 with the same NumPy and pandas versions. Restore the preparation versions for the analysis step. On the project host all phases use the project-root interpreter. The paper records the core scientific package versions. The discovery analysis additionally uses the historical Stage 18 saved forecasts. The repository distributes aggregate discovery results.

In the discovery panel, changing only the entrant share from 31.64% to 23.05% increases the Naive minus AutoTheta gap by 0.0053 MASE, holding the calendar-grid subgroup means fixed. This post-hoc standardization uses the shares in H2 and the two group means in H3 of `confirm/out/discovery_analysis.json`. The overall winner stays AutoTheta.

### Verify saved forecasts

With the original forecast batches available, run `python confirm/verify_archive.py`. Set `P12_DISCOVERY_RUN` to the Stage 18 run and `P12_RUNS` to the parent of the three replication runs. The verifier checks model and target coverage, reconstructs date-related losses and entrant shares, and recomputes the primary entrant-minus-incumbent accuracy contrast from individual forecast errors. The test and archive-verification environment is pinned in `requirements-lock.txt`. For historical numerical reproduction the preprocessing versions included NumPy 2.0.2, pandas 2.2.3 and PyArrow 17.0.0; the historical forecast stage used NumPy 2.0.2, pandas 2.2.3 and PyArrow 26.0.0.

On 20 November 2026 the Windows project environment passed 102 tests. On the archived batches the verifier matched all four panels (1,098,132 discovery forecasts, 1,086,906 Bank Leverage, 475,938 BHC Leverage, and 453,924 BHC Cash Liquidity). The three replication panels' nine archived analysis, construction and preparation summaries matched their original run directories. Regenerating the confirmatory tables and TeX values reproduced the committed contents. This release audit used saved forecasts.
