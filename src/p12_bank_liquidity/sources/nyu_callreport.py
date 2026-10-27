"""NYU Call Report source record. Metadata only; the zip is not downloaded."""

from __future__ import annotations

from p12_bank_liquidity.gates import GateBlocked
from p12_bank_liquidity.manifests import assert_bulk_download_refused, assert_not_subscription_url

PAGE_URL = "https://pages.stern.nyu.edu/~pschnabl/data/data_callreport.htm"
ZIP_URL = "https://pages.stern.nyu.edu/~pschnabl/research/callreports_1976_2020_WRDS.dta.zip"
DICTIONARY_URL = "https://pages.stern.nyu.edu/~pschnabl/research/CallReportVariableKey_WRDS_2021.xlsx"
SAS_URL = "https://pages.stern.nyu.edu/~pschnabl/research/callreport_1976_2020_WRDS.sas"
README_URL = "https://pages.stern.nyu.edu/~pschnabl/research/callreport_readme.pdf"

PAGE_SHA256 = "cbab9ebd667d8bc7304e51efc0b69be978e94e9f431c8eeb17b37228bbae10b9"
DICTIONARY_SHA256 = "89b149b34eda0ea0acab4fe82e7bce1afb1538007f1a586ea0229b5970fcea69"
SAS_SHA256 = "66dd859c37804772255685c0dfbc36429d8ff17dc7ec073b3a44aa3809559d59"
README_SHA256 = "a061906fc51b405541a2340704fb4c5dd71dea7e774950c1db015391311b7040"

# HEAD on 2026-10-26. The HTML page says "354MB"; the response length does not.
ZIP_CONTENT_LENGTH_BYTES = 455791391
ZIP_SHA256 = None

TERMS = "Data may be used for non-commercial purposes free of charge."
LAST_UPDATE_STATED = "November 2020"


def refuse_zip_download(authorized: bool) -> None:
    assert_not_subscription_url(ZIP_URL)
    assert_bulk_download_refused("callreports_1976_2020_WRDS.dta.zip", authorized)
    if not authorized:
        raise GateBlocked(
            "NYU Call Report zip download is not authorized "
            f"({ZIP_CONTENT_LENGTH_BYTES} bytes by HEAD; page label 354MB)"
        )
