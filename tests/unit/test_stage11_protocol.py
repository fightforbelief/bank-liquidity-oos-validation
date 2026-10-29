"""Synthetic protocol checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""

from pathlib import Path

import pytest
import yaml

from p12_bank_liquidity.candidate_a.aggregation import (
    PairedMaseRecord,
    aggregate_paired_mase,
)
from p12_bank_liquidity.candidate_a.alignment import (
    DuplicateEntityTarget,
    TargetRecord,
    align_observed_targets,
)
from p12_bank_liquidity.candidate_a.bootstrap import (
    CONFIDENCE_LEVEL,
    FORMAL_EQUIVALENCE_CLAIM,
    INTERVAL,
    PRACTICAL_EQUIVALENCE_MARGIN,
    REPLICATES,
    SEED,
    entity_cluster_bootstrap,
)
from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL, seasonal_mase
from p12_bank_liquidity.candidate_a.model_spec import (
    CONFIRMATORY_MODELS,
    EXCLUDED_FROM_CONFIRMATORY,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "research" / "P12_QUARTER_END_ROBUSTNESS" / "CANDIDATE_A_PROTOCOL_V1.yaml"


def _record(**overrides: object) -> TargetRecord:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "origin_timestamp": "1976-09-30",
        "target_timestamp": "1976-12-31",
        "observed": True,
        "synthetic_grid_date": False,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return TargetRecord(**values)  # type: ignore[arg-type]


def _mase(**overrides: object) -> PairedMaseRecord:
    values: dict[str, object] = {
        "entity_id": "fixture-entity",
        "target_timestamp": "1976-12-31",
        "mase_arm_a": 1.0,
        "mase_arm_b": 2.0,
        "target_observed_both": True,
        "forecast_finite_both": True,
        "denominator_defined_both": True,
        "evidence_label": EVIDENCE_LABEL,
    }
    values.update(overrides)
    return PairedMaseRecord(**values)  # type: ignore[arg-type]


def test_primary_pair_keeps_both_origins_when_only_the_target_matches() -> None:
    arm_a = [_record(origin_timestamp="1976-12-30", target_timestamp="1977-03-31")]
    arm_b = [_record(origin_timestamp="1976-12-31", target_timestamp="1977-03-31")]
    result = align_observed_targets(arm_a, arm_b)
    assert len(result.primary_pairs) == 1
    pair = result.primary_pairs[0]
    assert pair.origin_timestamp_arm_a == "1976-12-30"
    assert pair.origin_timestamp_arm_b == "1976-12-31"
    assert pair.target_timestamp == "1977-03-31"
    assert pair.origin_timestamp_equal is False
    assert pair.origin_quarter_equal is True
    assert pair.target_is_observed_arm_a is True
    assert pair.target_is_observed_arm_b is True
    assert pair.evidence_label == EVIDENCE_LABEL
    assert result.secondary_pairs == ()


def test_secondary_pair_requires_the_same_origin_and_target_timestamps() -> None:
    arm_a = [
        _record(entity_id="same", origin_timestamp="1976-12-31", target_timestamp="1977-03-31"),
        _record(entity_id="differs", origin_timestamp="1976-12-30", target_timestamp="1977-03-31"),
    ]
    arm_b = [
        _record(entity_id="same", origin_timestamp="1976-12-31", target_timestamp="1977-03-31"),
        _record(entity_id="differs", origin_timestamp="1976-12-31", target_timestamp="1977-03-31"),
    ]
    result = align_observed_targets(arm_a, arm_b)
    assert [pair.entity_id for pair in result.primary_pairs] == ["differs", "same"]
    assert [pair.entity_id for pair in result.secondary_pairs] == ["same"]


def test_arm_only_keys_stay_unmatched() -> None:
    arm_a = [
        _record(entity_id="shared", target_timestamp="1977-03-31"),
        _record(entity_id="only-a", target_timestamp="1977-06-30"),
    ]
    arm_b = [
        _record(entity_id="shared", target_timestamp="1977-03-31"),
        _record(entity_id="only-b", origin_timestamp="1977-06-30", target_timestamp="1977-09-30"),
    ]
    result = align_observed_targets(arm_a, arm_b)
    assert [pair.entity_id for pair in result.primary_pairs] == ["shared"]
    assert result.unmatched_arm_a == (("only-a", "1977-06-30"),)
    assert result.unmatched_arm_b == (("only-b", "1977-09-30"),)
    assert result.secondary_pairs[0].entity_id == "shared"


def test_pairing_rejects_duplicate_entity_target_keys() -> None:
    arm_a = [
        _record(target_timestamp="1977-03-31"),
        _record(origin_timestamp="1976-12-31", target_timestamp="1977-03-31"),
    ]
    arm_b = [_record(target_timestamp="1977-03-31")]
    with pytest.raises(DuplicateEntityTarget):
        align_observed_targets(arm_a, arm_b)


def test_primary_keys_are_unique_for_distinct_targets() -> None:
    arm_a = [
        _record(target_timestamp="1977-03-31"),
        _record(origin_timestamp="1977-03-31", target_timestamp="1977-06-30"),
    ]
    arm_b = [
        _record(target_timestamp="1977-03-31"),
        _record(origin_timestamp="1977-03-31", target_timestamp="1977-06-30"),
    ]
    result = align_observed_targets(arm_a, arm_b)
    keys = [(pair.entity_id, pair.target_timestamp) for pair in result.primary_pairs]
    assert len(keys) == len(set(keys)) == 2


def test_synthetic_grid_date_is_not_an_observed_target() -> None:
    arm_a = [_record(target_timestamp="1977-03-31")]
    arm_b = [
        _record(
            origin_timestamp="1976-12-30",
            target_timestamp="1977-03-30",
            observed=True,
            synthetic_grid_date=True,
        )
    ]
    result = align_observed_targets(arm_a, arm_b)
    assert result.primary_pairs == ()
    assert result.secondary_pairs == ()
    assert result.unmatched_arm_a == (("fixture-entity", "1977-03-31"),)
    assert result.unmatched_arm_b == ()


def test_entity_mean_is_unweighted_across_entities() -> None:
    records = [
        _mase(entity_id="e1", target_timestamp="1977-03-31", mase_arm_a=1.0, mase_arm_b=3.0),
        _mase(entity_id="e1", target_timestamp="1977-06-30", mase_arm_a=3.0, mase_arm_b=5.0),
        _mase(entity_id="e2", target_timestamp="1977-03-31", mase_arm_a=0.0, mase_arm_b=1.0),
    ]
    result = aggregate_paired_mase(records)
    assert result.delta_by_entity == {"e1": 2.0, "e2": 1.0}
    assert result.estimate == 1.5
    assert result.exclusions == ()
    assert result.evidence_label == EVIDENCE_LABEL


def test_undefined_mase_denominator_is_excluded_and_not_replaced() -> None:
    undefined = seasonal_mase([3, 3, 3, 3, 3, 3, 3, 3], [3], [4])
    assert undefined.status == "UNDEFINED"
    assert undefined.value is None
    assert undefined.reason == "zero_denominator"
    short = seasonal_mase([1, 2, 3], [4], [5])
    assert short.status == "UNDEFINED"
    assert short.reason == "short_history"
    result = aggregate_paired_mase(
        [
            _mase(entity_id="e1", target_timestamp="1977-03-31", mase_arm_a=1.0, mase_arm_b=2.0),
            _mase(
                entity_id="e1",
                target_timestamp="1977-06-30",
                mase_arm_a=None,
                mase_arm_b=None,
                denominator_defined_both=False,
            ),
        ]
    )
    assert result.delta_by_entity == {"e1": 1.0}
    assert result.estimate == 1.0
    assert [item.reason for item in result.exclusions] == ["mase_denominator_undefined"]


def test_bootstrap_configuration_is_deterministic_and_keeps_failed_replicates() -> None:
    assert REPLICATES == 2000
    assert SEED == 20261027
    assert CONFIDENCE_LEVEL == 0.95
    assert INTERVAL == "PERCENTILE"
    assert PRACTICAL_EQUIVALENCE_MARGIN == "NOT_SET"
    assert FORMAL_EQUIVALENCE_CLAIM == "NOT_AUTHORIZED"
    defined = entity_cluster_bootstrap({"a": 1.0, "b": 3.0})
    repeat = entity_cluster_bootstrap({"a": 1.0, "b": 3.0})
    assert defined == repeat
    assert defined.failed_replicate_indices == ()
    assert defined.lower is not None and defined.upper is not None
    assert defined.lower <= defined.upper
    partial = entity_cluster_bootstrap({"a": 1.0, "b": None}, replicates=40, seed=SEED)
    again = entity_cluster_bootstrap({"a": 1.0, "b": None}, replicates=40, seed=SEED)
    assert partial == again
    assert partial.failed_replicate_indices
    assert len(partial.estimates) + len(partial.failed_replicate_indices) == 40
    assert all(math_isfinite(value) for value in partial.estimates)
    assert all(value == 1.0 for value in partial.estimates)


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def test_confirmatory_model_list_matches_the_frozen_scope() -> None:
    assert [item.protocol_id for item in CONFIRMATORY_MODELS] == [
        "HistoricAverage",
        "Naive",
        "SeasonalNaive",
        "AutoARIMA",
        "SimpleExponentialSmoothing",
        "AutoTheta",
    ]
    ses = CONFIRMATORY_MODELS[4]
    assert ses.constructor == "SimpleExponentialSmoothing(alpha=0.1)"
    assert CONFIRMATORY_MODELS[1].parameter_status == "API_NOT_VERIFIED"
    assert CONFIRMATORY_MODELS[5].source_status == "SOURCE_ABSENT"
    assert "Ridge" in EXCLUDED_FROM_CONFIRMATORY
    assert "AutoETS" in EXCLUDED_FROM_CONFIRMATORY


def test_protocol_file_freezes_rules_without_freezing_the_environment() -> None:
    document = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert document["scientific_protocol_status"] == "FROZEN"
    assert document["execution_environment"]["status"] == "NOT_FROZEN"
    assert document["authorization"]["model_fitting_authorized"] is False
    assert document["authorization"]["forecast_generation_authorized"] is False
    assert document["authorization"]["dependency_installation_authorized"] is False
    assert document["authorization"]["external_data_download_authorized"] is False
    assert document["authorization"]["model_run_authorized"] is False
    assert document["matching"]["primary"]["rule"] == "ENTITY_AND_EXACT_TARGET_QUARTER"
    assert document["matching"]["secondary"]["rule"] == (
        "ENTITY_AND_EXACT_ORIGIN_TIMESTAMP_AND_TARGET_TIMESTAMP"
    )
    assert document["metric"]["primary"] == "MASE"
    assert document["inference"]["formal_equivalence_claim"] == "NOT_AUTHORIZED"
    assert [item["protocol_id"] for item in document["models"]["confirmatory"]] == [
        item.protocol_id for item in CONFIRMATORY_MODELS
    ]
