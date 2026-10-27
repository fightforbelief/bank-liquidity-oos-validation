"""Render the Gate 0 / Gate 1 audit reports from inspected sources."""

from __future__ import annotations

from pathlib import Path

from p12_bank_liquidity import SYNTHETIC_BANNER

INSPECTED_AT = "2026-10-26"
FTSFR_COMMIT = "e740156e22b04498c74525f8db6b95f359ab596a"
NYU_MODULE_COMMIT = "0fd99cd20fe219328769d5b1793d26290d81786e"
OFR_PDF_SHA256 = "c1396f58e2c4885925b1fe562767034cd95b41448ebd508a9c56ec2102f6cb90"
OFR_PDF_BYTES = 7682954


def _banner(evidence_class: str) -> str:
    if evidence_class == "TEST_FIXTURE":
        return f"\n\n{SYNTHETIC_BANNER}\n"
    return ""


def data_access_audit(evidence_class: str = "SOURCE_DOCUMENT") -> str:
    body = f"""# Data access audit

**Inspected:** {INSPECTED_AT}
**Evidence class:** {evidence_class}
**Bulk download this run:** no
**Model fit this run:** no

## NYU Call Report page

| Item | Value | Status |
| --- | --- | --- |
| Page | https://pages.stern.nyu.edu/~pschnabl/data/data_callreport.htm | `VERIFIED` |
| Page SHA-256 | cbab9ebd667d8bc7304e51efc0b69be978e94e9f431c8eeb17b37228bbae10b9 | `VERIFIED` |
| Stated terms | "Data may be used for non-commercial purposes free of charge." | `VERIFIED` |
| Stated last update | November 2020 | `VERIFIED` |
| Stated coverage | 1976–2020 bank-level file | `VERIFIED` |
| Zip URL | https://pages.stern.nyu.edu/~pschnabl/research/callreports_1976_2020_WRDS.dta.zip | `VERIFIED` |
| Zip HEAD Content-Length | 455,791,391 bytes | `VERIFIED` |
| Zip page label | 354MB | `VERIFIED` as page text; disagrees with Content-Length |
| Zip SHA-256 | not computed | `NOT_AVAILABLE` because the zip was not downloaded |
| Dictionary URL | https://pages.stern.nyu.edu/~pschnabl/research/CallReportVariableKey_WRDS_2021.xlsx | `VERIFIED` |
| Dictionary bytes / SHA-256 | 27,153 / 89b149b34eda0ea0acab4fe82e7bce1afb1538007f1a586ea0229b5970fcea69 | `VERIFIED` |
| Same dictionary in jmbejara/ftsfr | identical SHA-256 at commit {FTSFR_COMMIT} | `VERIFIED` |
| SAS construction code | https://pages.stern.nyu.edu/~pschnabl/research/callreport_1976_2020_WRDS.sas | `VERIFIED` |
| SAS bytes / SHA-256 | 31,931 / 66dd859c37804772255685c0dfbc36429d8ff17dc7ec073b3a44aa3809559d59 | `VERIFIED` |
| Readme PDF | https://pages.stern.nyu.edu/~pschnabl/research/callreport_readme.pdf | `VERIFIED` |
| Readme bytes / SHA-256 | 494,712 / a061906fc51b405541a2340704fb4c5dd71dea7e774950c1db015391311b7040 | `VERIFIED` |
| Human acceptance of non-commercial terms | not recorded | `BLOCKED` for download |
| Commercial use | not granted by the page sentence | `NOT_AVAILABLE` |
| WRDS reconstruction route | page points at WRDS SAS Studio | `BLOCKED` until authorised access is confirmed |

The readme says the code, data, and dictionary are provided free of charge and asks for a citation. It describes a WRDS SAS plus Stata cleaning route. It does not itself contain the sentence "non-commercial"; that sentence is on the HTML page.

## FFIEC CDR

| Item | Value | Status |
| --- | --- | --- |
| Bulk page | https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx | `VERIFIED` HEAD 200, 26,971 bytes |
| Help page | https://cdr.ffiec.gov/Public/HelpFiles/WelcomeAdditionalInfo.htm | `VERIFIED` |
| Amendment statement | bulk files are regenerated monthly to capture amendments | `VERIFIED` |
| Availability lag | bulk files begin 45 calendar days after the report date | `VERIFIED` as the help-page statement |
| Exact release timestamp of each bank report | not established by the 45-day schedule | `NOT_CHECKED` |
| Current FFIEC 031 forms page | https://www.ffiec.gov/resources/reporting-forms/ffiec031 | `BLOCKED` on this run (HTTP 403) |
| Bulk file downloaded | no | `VERIFIED` |

## FTSFR access boundary

`subscriptions.toml` at {FTSFR_COMMIT} sets `nyu_call_report = true` and also sets `bloomberg_terminal = true` and every listed `wrds_*` flag to true. Running `doit -f dodo_01_pull.py` on that file would be able to invoke subscription pulls. This project did not run that task. WRDS and Bloomberg stay disabled here regardless of the upstream defaults.

FTSFR repository licence file: GitHub licence API returned 404. SPDX licence of the code is `NOT_AVAILABLE` from that check.

## What this audit did not do

No Call Report zip, Stata extract, parquet panel, or FFIEC bulk quarter was saved into the project.
"""
    return body + _banner(evidence_class)


def target_definition_audit(evidence_class: str = "SOURCE_DOCUMENT") -> str:
    body = f"""# Target definition audit

**Inspected:** {INSPECTED_AT}
**Evidence class:** {evidence_class}
**Config `target_definition_status`:** `NOT_VERIFIED` (left unchanged; raw bank-quarter checks are still blocked)

## Decision on the formula text

The historical Bank Cash Liquidity target used by FTSFR is the ratio of two NYU columns, after row filters. That ratio is copied below. It is not inferred from the series name. Three calculations on real bank-quarters were not done, because the microdata zip was not downloaded. Those checks stay `BLOCKED`.

| Piece | Definition | Where read | Status |
| --- | --- | --- | --- |
| Target name | Bank Cash Liquidity, bank level, `rssdid` | `datasets.toml` key `ftsfr_nyu_call_report_cash_liquidity`; paper Table 2 | `VERIFIED` |
| Not this target | BHC Cash Liquidity, summed by `bhcid`, dropping id `"0"` | same module, separate block | `VERIFIED` |
| Assets column | `assets = rcfd2170` | SAS line `assets=rcfd2170`, dictionary row "Total assets / assets / RCFD2170 / 197603-" | `VERIFIED` |
| Cash column | `cash = rcfd0010`; if missing, `cash = rcfd0071 + rcfd0081` | SAS lines for `cash=rcfd0010` and the following `if cash=.` assignment; dictionary rows for RCFD0010 and `RCFD0071+RCFD0081` | `VERIFIED` |
| Ratio | `y = cash / assets` | `create_ftsfr_datasets.py` in jmbejara/ftsfr @ {FTSFR_COMMIT} and ftsfr/nyu_call_report @ {NYU_MODULE_COMMIT} | `VERIFIED` |
| Row filter | keep `assets > 1e-10` and `cash >= 0`; drop non-finite ratios | same Python file | `VERIFIED` |
| Constant series | drop an `rssdid` whose sample standard deviation is 0 | same Python file, to avoid a zero MASE scale | `VERIFIED` |
| Entity key | `rssdid` (dictionary string RSSD9001); BHC key `bhcid` (RSSD9348) | pull script column list and dictionary shared strings | `VERIFIED` |
| Units | ratio of dollar stocks, no extra transformation in the ratio step | code | `VERIFIED` |
| Frequency | quarterly, `QE`, seasonality 4 in `datasets.toml` | `datasets.toml` | `VERIFIED` |
| Sample end in the paper | 1976-03-31 to 2020-03-31 | OFR PDF Table 4 and Table 15 | `VERIFIED` |
| Winsorization `[5.0, 95.0]` in `datasets.toml` | declared on the dataset key | whether the published forecast run applied it | `NOT_CHECKED` |
| Real raw-to-target rows | three hand checks against the Stata file | not available in this run | `BLOCKED` |

The BHC docstring says "Cash flow / Total assets". The executable line is `cash / assets` after summing `cash` and `assets`. The code, not that docstring, is the BHC definition. BHC remains out of scope.

## Paper counts, read from the PDF

PDF: OFR WP 26-05, {OFR_PDF_BYTES} bytes, SHA-256 `{OFR_PDF_SHA256}`.

Table 4, "Dataset Statistics Summary", Bank Cash Liquidity row:

- frequency quarterly
- unique entities 23,862
- minimum length 1
- median length 66
- maximum length 177
- 1976-03-31 to 2020-03-31

Status: `VERIFIED` from the extracted table text. These are entity and length counts. 23,862 is not an observation count, and 66 is not an entity count.

Table 15, "Impact of Robust Forecasting Preprocessing", Bank Cash Liquidity row:

- entities before 23,862
- entities after 17,383
- retention 72.8%
- median length before 66
- median length after 82
- 1976-03-31 to 2020-03-31

Status: `VERIFIED` from the extracted table text. The earlier research-pack sentence that called 23,862 and 17,383 observations, and 66 an entity count, does not match this table.

## Published error tables

Table 6–8 cells for Bank Cash Liquidity were reconstructed from PDF text order under the header HistAvg, ARIMA, SES, Theta, DLinear, DeepAR, KAN, NBEATS, NHITS, NLinear, TiDE, Transformer. Status: `SOURCE CLAIMED` until someone checks the visual columns. They are replication targets, not results of this project.

- MASE: 1.83, 0.83, 1.09, 0.81, 0.79, 0.87, 0.79, 0.78, 0.77, 0.79, 0.80, 0.80
- RMSE: 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.03, 0.03
- out-of-sample R²: 0.00, 0.68, 0.49, 0.68, 0.70, 0.63, 0.70, 0.72, 0.72, 0.71, 0.71, 0.67

## Forecast protocol read from the paper and code

- Quarterly horizon in `get_test_size_from_frequency` is 1 quarter. `VERIFIED` in `forecast_utils.py`.
- Rolling-origin evaluation, step equal to the horizon, at most 6 windows. `VERIFIED` in the paper preprocessing section and `MAX_CV_WINDOWS = 6`.
- Neural forecasts average five seeds. `VERIFIED` on PDF page 30.
- Validation length for quarterly neural tuning is 8 quarters in `VAL_SIZE_BY_FREQUENCY`, separate from the 1-quarter horizon. `VERIFIED` in code.
- MASE in code is `utilsforecast.losses.mase`. The local metric helper is a Hyndman-style candidate for tests and is `NOT_CHECKED` against that function.
- Several filter functions coexist (`filter_series_for_forecasting`, `robust_preprocessing.get_data_requirements`, `get_minimum_requirements_by_frequency`). Which one produced the 17,383 entity count was not re-run. Status: `NOT_CHECKED`.
- Forecast clipping uses training history only, bounds `[train_min - k*IQR, train_max + k*IQR]` with `k = 1`. `VERIFIED` as code. Not applied to real data here.

## Synthetic worked examples

The next three rows are arithmetic checks of the copied formula. They are not bank filings.

| cash (RCFD0010) | fill inputs | assets (RCFD2170) | ratio | note |
| --- | --- | --- | --- | --- |
| 20 | unused | 100 | 0.20 | direct column |
| missing | 0071=5, 0081=7 | 40 | 0.30 | SAS fill, then ratio |
| -1 | unused | 100 | dropped | negative cash fails `cash >= 0` |

## Post-2020 field continuity

The SAS comment says RCFD0010 is sparsely reported starting 201503, and the fill uses RCFD0071 and RCFD0081 from 198403 onward. The current FFIEC form instructions were not retrieved (HTTP 403). Whether those three items still mean the same stocks after 2020 is `NOT_CHECKED`. L2 cannot treat field continuity as established.
"""
    return body + _banner(evidence_class)


def data_vintage_audit(evidence_class: str = "SOURCE_DOCUMENT") -> str:
    body = f"""# Data vintage audit

**Inspected:** {INSPECTED_AT}
**Evidence class:** {evidence_class}
**Config `vintage_mode`:** `UNSET` until a person accepts a label
**Required label if later work uses current amended files:** `FINAL_VINTAGE_RETROSPECTIVE_TEMPORAL_VALIDATION`

## What was checked

| Question | Finding | Status |
| --- | --- | --- |
| Does the NYU page expose a vintage history? | One preconstructed file, last update stated as November 2020. No as-of or first-release field is described. | `VERIFIED` that the page does not offer vintages |
| Does the NYU readme describe revision snapshots? | No. It documents a WRDS download plus a Stata clean. | `VERIFIED` |
| Does FFIEC say bulk files change after release? | Yes. After the initial file, data are regenerated monthly to capture amendments. | `VERIFIED` |
| Is the 45-day lag the publication time of every report? | The help page does not say that. Individual filings can appear earlier through institution download. | `VERIFIED` as a limitation |
| Public revision-history report | CDR revision history in the help material is an analyst login task, not a public bulk vintage archive. | `SOURCE CLAIMED` from the CDR help pages retrieved in search |
| Chicago Fed commercial-bank files | The Chicago Fed page says that effective with 2021Q2 no further as-of dates will be added. That is a stopped snapshot, not a continuing first-release archive. | `SOURCE CLAIMED` |
| First-release snapshot archive located | none in the sources opened for this run | `NOT_AVAILABLE` |

## Consequence

A chronological split of today's FFIEC file can still be computed later, but predictors may contain amendments filed after the original quarter. That design is a final-vintage retrospective temporal validation. It is not a real-time or vintage-valid forecast.

If the scientific claim requires the information set that existed at each forecast origin, missing first-release vintages are a stop for that claim. The config does not switch `vintage_mode` on by itself.
"""
    return body + _banner(evidence_class)


def entity_panel_audit(evidence_class: str = "SOURCE_DOCUMENT") -> str:
    body = f"""# Entity and panel audit

**Inspected:** {INSPECTED_AT}
**Evidence class:** {evidence_class}

## Identifiers in the source module

| Item | Source behaviour | Status |
| --- | --- | --- |
| Bank key | `rssdid`, renamed to `unique_id` | `VERIFIED` in code |
| Holding-company key | `bhcid`; id `"0"` dropped only on the BHC outputs | `VERIFIED` in code |
| Dictionary names | RSSD9001 for rssdid, RSSD9348 for bhcid | `VERIFIED` as dictionary strings |
| Name as key | not used | `VERIFIED` |
| Mergers, charter changes, failures, exits | no transition table in `create_ftsfr_datasets.py` or `pull_nyu_call_report.py` | `NOT_CHECKED` against an official structure file; the forecasting module does not build one |
| Survivor filter | not present in the cash-liquidity constructor | `VERIFIED` that this module does not keep only banks alive at the sample end |
| Duplicate `rssdid`-date rows | not collapsed in the bank-level constructor | `NOT_CHECKED` on real rows |
| Missing quarters | forecasting preprocessing can build a grid and forward-fill the training slice | `VERIFIED` as code behaviour; whether Bank Cash Liquidity gaps were filled in the published run was not re-executed |
| Same panel for every model | paper says the filtered panel is shared | `VERIFIED` as a stated rule; not re-run |

## Counts that are replication checks

Do not change filters to force these numbers.

- Table 4: 23,862 entities, median length 66, maximum length 177.
- Table 15: 17,383 entities after preprocessing, 72.8% retention, median length after 82.

Both are `VERIFIED` as published table text. Neither has been recomputed from microdata. Observation totals are `NOT_CHECKED`.

## Panel rules this repository enforces in tests

- quarter-ends are 31 March, 30 June, 30 September, and 31 December
- duplicate entity-quarter keys are reported
- missing quarters are listed and not interpolated by the audit helper
- a filter that keeps only entities present on a terminal date is refused
- two models must receive the same eligible entity set

These tests use synthetic rows and are not evidence about U.S. banks.
"""
    return body + _banner(evidence_class)


def gate0_gate1_decision(evidence_class: str = "SOURCE_DOCUMENT") -> str:
    body = f"""# Gate 0 and Gate 1 decision

**Date:** {INSPECTED_AT}
**Evidence class:** {evidence_class}
**Project status:** provisional audit. No confirmatory forecast was run.

## Gate 0 — paper and novelty

**Outcome:** `PIVOT`

The anchor is real, and a novelty claim is not supported.

| Check | Status |
| --- | --- |
| OFR WP 26-05 PDF retrieved and hashed | `VERIFIED` |
| Bank Cash Liquidity is a Call Report series following Drechsler, Savov, and Schnabl (2017), distinct from BHC Cash Liquidity | `VERIFIED` |
| Twelve-model roster and MASE / pooled OOS R² / RMSE | `VERIFIED` as the paper's reported design |
| Independent reproduction of any forecast number | `NOT_CHECKED` |
| Sonmez, Kuang, Nal, and Marzolf-Miller (2025), JRFM 18(10):576 | `VERIFIED` as adjacent, not the same study |

The 2025 paper estimates Gaussian hidden Markov regime shifts in a reserves-at-Fed to domestic-deposits ratio, RCFD0090 over RCON2200 or RCFD2200, mainly for 2002:Q4–2015:Q4, with a four-bank illustration for 2017–2023. That is a different target and a different task from reproducing the OFR forecasting table. It does occupy bank-level Call Report liquidity work through and after 2020, so this project cannot say that post-2020 bank liquidity dynamics are unstudied.

A search of every 2021–2026 forecasting paper on Call Reports was not completed. That literature pass remains `NOT_CHECKED`. Gate 0 therefore does not pass a contribution claim. The provisional question can continue only as reproduction and external validation.

## Gate 1 — data and target

**Outcome:** `BLOCKED`

| Required item | Status | Effect |
| --- | --- | --- |
| Historical ratio `cash / assets` with the FTSFR filters | `VERIFIED` from SAS, dictionary, and both FTSFR copies | formula text is known |
| NYU column codes RCFD2170, RCFD0010, and RCFD0071+RCFD0081 | `VERIFIED` | upstream definition is known |
| Three real raw-to-target calculations | `BLOCKED` | no microdata downloaded |
| Non-commercial terms on the NYU page | `VERIFIED` as text | human acceptance not recorded, so `licence_status` stays `NOT_VERIFIED` |
| FTSFR code licence | `NOT_AVAILABLE` from the GitHub licence endpoint | |
| Post-2020 MDRM continuity for the cash items | `NOT_CHECKED` | forms page returned HTTP 403 |
| Merger and charter identity | `NOT_CHECKED` | |
| First-release vintages | `NOT_AVAILABLE` from the sources opened | L2 label, if later approved, is `FINAL_VINTAGE_RETROSPECTIVE_TEMPORAL_VALIDATION` |
| Bank module isolated from the full pull | `VERIFIED` | files were read; `dodo_01_pull.py` was not run |
| WRDS / Bloomberg | left disabled | upstream `subscriptions.toml` would enable both |

`BLOCKED` means bulk acquisition and model fitting stay off. It is not a finding that the historical formula is unknown. A `STOP` would be appropriate for a real-time claim, because first-release vintages were not found. A retrospective claim is still a later, separate decision.

## Commands that must keep failing

- `p12 data-audit` until target, licence, and vintage fields leave `NOT_VERIFIED` / `UNSET`
- `p12 reproduce` until target status is `VERIFIED`, licence status is `ACCEPTED`, download is authorized, the protocol is `APPROVED`, and the compute flag is true
- `p12 validate-later-period` until those conditions hold and a person accepts the vintage label, the external period, the primary metric, and a frozen model list

## Owner decision requested

1. Accept the NYU non-commercial terms, or stop before any historical file is stored.
2. After that acceptance, authorize only the historical zip so three real raw-to-target rows can be checked. Do not authorize model fitting in the same step.
3. Accept that any later FFIEC evaluation using current bulk files must carry the final-vintage retrospective label.
4. Accept Gate 0 as a pivot away from novelty while the 2021–2026 literature pass is still open.
"""
    return body + _banner(evidence_class)


REPORTS = {
    "data_access_audit.md": data_access_audit,
    "target_definition_audit.md": target_definition_audit,
    "data_vintage_audit.md": data_vintage_audit,
    "entity_panel_audit.md": entity_panel_audit,
    "gate0_gate1_decision.md": gate0_gate1_decision,
}


def write_reports(directory: Path, evidence_class: str = "SOURCE_DOCUMENT") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in REPORTS.items():
        path = directory / name
        path.write_text(builder(evidence_class), encoding="utf-8")
        written.append(path)
    return written
