"""FFIEC CDR bulk source record. No file is downloaded."""

from __future__ import annotations

from p12_bank_liquidity.gates import GateBlocked

BULK_PAGE = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
HELP_PAGE = "https://cdr.ffiec.gov/Public/HelpFiles/WelcomeAdditionalInfo.htm"
FORMS_PAGE = "https://www.ffiec.gov/resources/reporting-forms/ffiec031"

VINTAGE_LABEL = "FINAL_VINTAGE_RETROSPECTIVE_TEMPORAL_VALIDATION"
HELP_AMENDMENT_STATEMENT = (
    "After the initial bulk data files are created, data will be regenerated "
    "monthly to capture amendments."
)


def refuse_bulk_download() -> None:
    raise GateBlocked(
        "FFIEC CDR bulk download is not authorized. Current bulk files are "
        "regenerated for amendments and are not a first-release vintage."
    )
