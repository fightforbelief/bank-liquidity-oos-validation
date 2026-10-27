"""Unit tests for audit behaviour. Fixtures are not research evidence."""

import datetime as dt
from pathlib import Path

import pytest

from p12_bank_liquidity import SYNTHETIC_BANNER
from p12_bank_liquidity.audit.report import write_reports
from p12_bank_liquidity.cli import main
from p12_bank_liquidity.config import load_config
from p12_bank_liquidity.evaluation.metrics import (
    mean_absolute_scaled_error,
    pooled_oos_r2,
    root_mean_squared_error,
)
from p12_bank_liquidity.evaluation.uncertainty import (
    assert_uncertainty_not_selected,
    refuse_iid_row_bootstrap,
)
from p12_bank_liquidity.gates import GateBlocked
from p12_bank_liquidity.manifests import assert_not_subscription_url
from p12_bank_liquidity.models.adapters import fit_model
from p12_bank_liquidity.preprocessing.panel import (
    assert_same_eligible_panel,
    quality_report,
    reject_survivor_filter,
)
from p12_bank_liquidity.preprocessing.splits import (
    fit_train_standardizer,
    rolling_origin_windows,
    train_only_clip_bounds,
)
from p12_bank_liquidity.schemas import (
    duplicate_entity_quarters,
    missing_quarter_ends,
    panel_without_interpolation,
    parse_quarter_end,
)
from p12_bank_liquidity.sources.ffiec_cdr import refuse_bulk_download
from p12_bank_liquidity.sources.nyu_callreport import refuse_zip_download
from p12_bank_liquidity.target.bank_cash_liquidity import (
    build_bank_cash_liquidity,
    cash_liquidity_ratio,
    nyu_cash_numerator,
)

ROOT = Path(__file__).resolve().parents[2]


def test_quarter_end_parsing():
    assert parse_quarter_end("1976-03-31") == dt.date(1976, 3, 31)
    assert parse_quarter_end("1976Q2") == dt.date(1976, 6, 30)
    assert parse_quarter_end("2020-12-31") == dt.date(2020, 12, 31)
    with pytest.raises(ValueError):
        parse_quarter_end("1976-03-30")
    with pytest.raises(ValueError):
        parse_quarter_end("1976-04-30")


def test_duplicate_entity_quarters():
    rows = [
        {"entity_id": "1", "date": "1976-03-31"},
        {"entity_id": "1", "date": "1976-03-31"},
        {"entity_id": "2", "date": "1976-03-31"},
    ]
    assert duplicate_entity_quarters(rows) == [("1", "1976-03-31")]


def test_missing_quarters_are_not_interpolated():
    rows = [
        {"entity_id": "1", "date": "1976-03-31", "y": 0.1},
        {"entity_id": "1", "date": "1976-09-30", "y": 0.2},
    ]
    assert missing_quarter_ends(rows, "1") == ["1976-06-30"]
    unchanged = panel_without_interpolation(rows)
    assert len(unchanged) == 2
    assert all("filled" not in row for row in unchanged)
    report = quality_report(rows)
    assert report["interpolated_rows"] == 0
    assert report["missing_by_entity"]["1"] == ["1976-06-30"]


def test_target_formula_on_synthetic_rows():
    assert cash_liquidity_ratio(20, 100) == pytest.approx(0.2)
    assert nyu_cash_numerator(None, 5, 7) == pytest.approx(12)
    assert cash_liquidity_ratio(nyu_cash_numerator(None, 5, 7), 40) == pytest.approx(0.3)
    assert cash_liquidity_ratio(-1, 100) is None
    assert cash_liquidity_ratio(1, 0) is None
    rows = [
        {"entity_id": "a", "date": "1976-03-31", "cash": 10, "assets": 100},
        {"entity_id": "a", "date": "1976-06-30", "cash": 10, "assets": 100},
        {"entity_id": "b", "date": "1976-03-31", "cash": 5, "assets": 50},
        {"entity_id": "c", "date": "1976-03-31", "cash": -3, "assets": 50},
    ]
    built = build_bank_cash_liquidity(rows)
    entities = {row["entity_id"] for row in built}
    assert "a" not in entities
    assert "b" in entities
    assert "c" not in entities
    assert built[0]["y"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        build_bank_cash_liquidity(
            [{"entity_id": "h", "date": "1976-03-31", "cash": 1, "assets": 2, "target_scope": "BHC_CASH_LIQUIDITY"}]
        )


def test_rolling_origin_has_no_future_observations():
    rows = [
        {"entity_id": "1", "date": f"1976-{month:02d}-{day:02d}", "y": index}
        for index, (month, day) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1)
    ]
    windows = rolling_origin_windows(rows, horizon=1, max_windows=6)
    assert windows
    for window in windows:
        train_end = parse_quarter_end(window["train"][-1]["date"])
        test_start = parse_quarter_end(window["test"][0]["date"])
        assert train_end < test_start
        assert len(window["test"]) == 1


def test_train_only_scaling_and_clipping_ignore_future_values():
    train = [1.0, 2.0, 3.0, 4.0]
    center, scale = fit_train_standardizer(train)
    future = [100.0, 200.0]
    assert fit_train_standardizer(train) == (center, scale)
    assert center == pytest.approx(2.5)
    low, high = train_only_clip_bounds(train, iqr_multiplier=1.0)
    low_again, high_again = train_only_clip_bounds(train + [], iqr_multiplier=1.0)
    assert (low, high) == (low_again, high_again)
    assert min(future) > high or max(future) > high


def test_models_share_one_eligible_panel():
    panel = [
        {"entity_id": "1", "date": "1976-03-31"},
        {"entity_id": "2", "date": "1976-03-31"},
    ]
    assert_same_eligible_panel({"ARIMA": panel, "Theta": list(panel)})
    with pytest.raises(GateBlocked):
        assert_same_eligible_panel({"ARIMA": panel, "Theta": panel[:1]})
    with pytest.raises(GateBlocked):
        reject_survivor_filter(panel, "1976-03-31")


def test_metric_definitions():
    insample = [1.0, 2.0, 3.0, 4.0]
    assert mean_absolute_scaled_error([4.0, 5.0], [3.0, 3.0], insample, 1) == pytest.approx(1.5)
    assert root_mean_squared_error([0.0, 0.0], [3.0, 4.0]) == pytest.approx((12.5) ** 0.5)
    assert pooled_oos_r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_fail_closed_gates(tmp_path: Path):
    config = load_config(ROOT / "configs" / "project.yaml")
    assert main(["--root", str(ROOT), "--config", str(ROOT / "configs" / "project.yaml"), "data-audit"]) == 2
    assert main(["--root", str(ROOT), "reproduce"]) == 2
    assert main(["--root", str(ROOT), "validate-later-period"]) == 2
    assert main(["--root", str(ROOT), "source-audit"]) == 0
    with pytest.raises(GateBlocked):
        refuse_zip_download(authorized=False)
    with pytest.raises(GateBlocked):
        refuse_bulk_download()
    with pytest.raises(GateBlocked):
        assert_not_subscription_url("https://wrds-web.wharton.upenn.edu/example")
    with pytest.raises(GateBlocked):
        assert_not_subscription_url("https://bloomberg.example/pull")
    with pytest.raises(GateBlocked):
        fit_model("ARIMA", authorized=False)
    with pytest.raises(GateBlocked):
        fit_model("NotASourceModel", authorized=True)
    with pytest.raises(GateBlocked):
        refuse_iid_row_bootstrap()
    with pytest.raises(GateBlocked):
        assert_uncertainty_not_selected("iid_row")
    assert config.data["raw_data_download_authorized"] is False
    assert config.access["wrds_authorized"] is False
    assert config.access["bloomberg_authorized"] is False


def test_synthetic_banner_propagates(tmp_path: Path):
    written = write_reports(tmp_path, evidence_class="TEST_FIXTURE")
    assert written
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert SYNTHETIC_BANNER in text
        assert "TEST_FIXTURE" in text


def test_source_reports_do_not_pretend_to_be_a_forecast(tmp_path: Path):
    written = write_reports(tmp_path, evidence_class="SOURCE_DOCUMENT")
    joined = "\n".join(path.read_text(encoding="utf-8") for path in written)
    assert SYNTHETIC_BANNER not in joined
    assert "BLOCKED" in joined
    assert "NOT_AVAILABLE" in joined
    assert "NOT_CHECKED" in joined
    assert "FINAL_VINTAGE_RETROSPECTIVE_TEMPORAL_VALIDATION" in joined
    assert "cash / assets" in joined
    assert "RCFD0010" in joined
    assert "RCFD2170" in joined
    assert "23,862" in joined
    assert "17,383" in joined
