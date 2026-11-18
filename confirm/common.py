"""Shared paths and settings for the confirmatory replication on the other FTSFR QE datasets."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch")
DTA = SCRATCH / "extract" / "callreports_1976_2020_WRDS.dta"
RUNS = Path(r"C:\Users\hanqi\Desktop\projects\_p12_confirm_runs")
DISCOVERY_RUN = Path(r"C:\Users\hanqi\Desktop\projects\_p12_stage18_full_run\run_20261028")
PROTOCOL = REPO / "confirm" / "PROTOCOL.md"

LABEL = "CONFIRMATORY_QE_REPLICATION_RUN"
TARGETS = ("bank_leverage", "bhc_leverage", "bhc_cash_liquidity")
MODELS = ("HistoricAverage", "Naive", "SeasonalNaive", "AutoARIMA", "SimpleExponentialSmoothing", "AutoTheta")
PANEL_START = "1976-03-31"
SEED = 20261119
N_BOOT = 2000
LENGTH_BINS = ((18, 30), (31, 45), (46, 60), (61, 80), (81, 200))

sys.path.insert(0, str(REPO / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage10():
    """The frozen Stage 10 module: upstream build_canonical_grid, quarter_grid, screen, imputation."""
    spec = importlib.util.spec_from_file_location("stage10_dry_run", SCRATCH / "stage10_dry_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dir(target: str) -> Path:
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target}")
    path = RUNS / target
    (path / "batches").mkdir(parents=True, exist_ok=True)
    return path
