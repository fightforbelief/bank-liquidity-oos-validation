"""Immutable-source manifests. Raw bytes stay outside version control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from p12_bank_liquidity import SYNTHETIC_BANNER
from p12_bank_liquidity.gates import GateBlocked

BULK_NAME_MARKERS = (
    "callreports_1976_2020",
    ".dta",
    "ffiec",
    "cdr_bulk",
)
SUBSCRIPTION_HOST_MARKERS = (
    "wrds",
    "wharton",
    "bloomberg",
)


@dataclass(frozen=True)
class SourceManifest:
    source_id: str
    url: str
    retrieved_at: str
    licence_summary: str
    file_name: str
    size_bytes: int | None
    sha256: str | None
    vintage: str
    evidence_class: str
    notes: str

    def to_json(self) -> str:
        payload = asdict(self)
        if self.evidence_class == "TEST_FIXTURE":
            payload["banner"] = SYNTHETIC_BANNER
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_subscription_url(url: str) -> None:
    lowered = url.lower()
    if any(marker in lowered for marker in SUBSCRIPTION_HOST_MARKERS):
        raise GateBlocked(
            f"refusing URL that points at WRDS or Bloomberg: {url}"
        )


def assert_bulk_download_refused(file_name: str, authorized: bool) -> None:
    lowered = file_name.lower()
    if any(marker in lowered for marker in BULK_NAME_MARKERS) and not authorized:
        raise GateBlocked(
            f"bulk Call Report acquisition is not authorized: {file_name}"
        )
    if not authorized and lowered.endswith((".zip", ".parquet")):
        raise GateBlocked(f"download of {file_name} is not authorized")
