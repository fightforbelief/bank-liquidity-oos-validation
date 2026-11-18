# Who Gets Benchmarked? A quarterly date grid that silently drops one in three entrants

Paper: [`paper.pdf`](paper.pdf). LaTeX source: [`paper_v3/`](paper_v3/).

The open financial forecasting benchmark of Bejarano et al. (OFR Working Paper 26-05, FTSFR) builds its quarterly grid by stepping three months from each series' first date. For a series that starts on 30 June or 30 September, that grid lands on 30 December and 30 March. Year-end and first-quarter Call Report filings, dated the 31st, then fail to join. We compare this grid (Arm A) with a calendar quarter-end grid (Arm B) on the benchmark's four quarterly Call Report datasets. Target, quality screen, six StatsForecast baselines, horizon, and seasonal MASE are held fixed.

## Findings

**For banks, the grid barely changes the forecasts it keeps.** On Bank Cash Liquidity, the pre-specified paired difference is negative for all six models, with all 95% intervals below zero. The differences are only 0.2–1.3% of the MASE level, and model rankings are unchanged.

**It changes who is evaluated.** The quality screen removes 2,190 banks under Arm A, all with a 30 June or 30 September first filing. The benchmark's published panel of 17,383 banks is exactly the Arm A panel; the calendar grid retains 19,567. Arm A loses 35% of all banks that entered after 1976.

**The removed population ranks the models differently.** Among entrants, Naive beats AutoTheta and AutoARIMA. Among banks present since 1976, the order reverses. The difference is not explained by history length, period, persistence, or bank age, and it is concentrated after abrupt level shifts. These analyses are exploratory.

**Pre-registered replication.** The protocol was frozen by SHA-256 before the other three datasets were processed.

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
