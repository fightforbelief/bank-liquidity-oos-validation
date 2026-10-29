"""Compare cached file bytes with the Stage 4 manifest. Do not parse rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from p12_bank_liquidity.candidate_a.hashes import compare_sha256

EXPECTED = {
    Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\callreports_1976_2020_WRDS.dta.zip"):
        "83abc553c7c55e04d4b0ea1cc2a46566ba330b0eb992a25cdc451fb68e0bae25",
    Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\extract\callreports_1976_2020_WRDS.dta"):
        "40692c2ec9af404b3148ffbd7ad970a4ce191d9ac8f781b415e8a10909533ad0",
    Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\OFRwp26-05.pdf"):
        "c1396f58e2c4885925b1fe562767034cd95b41448ebd508a9c56ec2102f6cb90",
    Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\stage4_constructor_panel.csv.gz"):
        "01cc1a400039f7fb6af079ce7df16591d13769e5c4cf679e3e10ff6574ac1ee4",
}


def main() -> int:
    results = []
    for path, digest in EXPECTED.items():
        if not path.exists():
            results.append({"path": str(path), "match": False, "reason": "MISSING"})
            continue
        results.append(compare_sha256(path, digest))
    print(json.dumps(results, indent=2))
    return 0 if all(item.get("match") is True for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
