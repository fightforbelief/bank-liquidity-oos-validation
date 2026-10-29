"""Byte-level SHA-256 comparison. This module does not parse data rows."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def compare_sha256(path: Path, expected: str) -> dict[str, object]:
    actual = sha256_file(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "match": actual == expected,
        "rows_parsed": 0,
    }
