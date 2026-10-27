"""Project configuration. Unapproved fields block downloads and model runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from p12_bank_liquidity.gates import GateBlocked


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict

    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def protocol(self) -> dict:
        return self.raw["protocol"]

    @property
    def compute(self) -> dict:
        return self.raw["compute"]

    @property
    def access(self) -> dict:
        return self.raw["access"]


def load_config(path: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise GateBlocked(f"config at {path} is not a mapping")
    for section in ("project", "data", "protocol", "compute", "access"):
        if section not in payload or not isinstance(payload[section], dict):
            raise GateBlocked(f"config missing section: {section}")
    return ProjectConfig(raw=payload)


def assert_audit_only(config: ProjectConfig) -> None:
    """Refuse anything that would acquire bulk data or fit models."""
    reasons = []
    if config.data.get("raw_data_download_authorized") is True:
        reasons.append("raw_data_download_authorized must stay false during the audit run")
    if config.access.get("wrds_authorized") is True:
        reasons.append("wrds_authorized is true")
    if config.access.get("bloomberg_authorized") is True:
        reasons.append("bloomberg_authorized is true")
    if config.compute.get("full_model_run_authorized") is True:
        reasons.append("full_model_run_authorized is true")
    if config.protocol.get("final_test_authorized") is True:
        reasons.append("final_test_authorized is true")
    if reasons:
        raise GateBlocked("audit run refused: " + "; ".join(reasons))


def assert_reproduction_allowed(config: ProjectConfig) -> None:
    reasons = []
    if config.data.get("target_definition_status") != "VERIFIED":
        reasons.append(
            f"target_definition_status={config.data.get('target_definition_status')!r}, required VERIFIED"
        )
    if config.data.get("licence_status") != "ACCEPTED":
        reasons.append(
            f"licence_status={config.data.get('licence_status')!r}, required ACCEPTED"
        )
    if config.data.get("raw_data_download_authorized") is not True:
        reasons.append("raw_data_download_authorized is not true")
    if config.protocol.get("status") != "APPROVED":
        reasons.append(f"protocol.status={config.protocol.get('status')!r}, required APPROVED")
    if config.protocol.get("human_approval") is not True:
        reasons.append("protocol.human_approval is not true")
    if config.compute.get("full_model_run_authorized") is not True:
        reasons.append("compute.full_model_run_authorized is not true")
    if reasons:
        raise GateBlocked("reproduction blocked: " + "; ".join(reasons))


def assert_later_period_allowed(config: ProjectConfig) -> None:
    assert_reproduction_allowed(config)
    reasons = []
    vintage = config.data.get("vintage_mode")
    allowed = {
        "VINTAGE_VALID",
        "FINAL_VINTAGE_RETROSPECTIVE_TEMPORAL_VALIDATION",
    }
    if vintage not in allowed:
        reasons.append(f"vintage_mode={vintage!r} is not an approved label")
    if config.data.get("vintage_human_accepted") is not True:
        reasons.append("vintage_human_accepted is not true")
    if config.protocol.get("external_test_period") in (None, "UNSET"):
        reasons.append("external_test_period is unset")
    if config.protocol.get("primary_metric") in (None, "UNSET"):
        reasons.append("primary_metric is unset")
    if config.protocol.get("model_list_frozen") is not True:
        reasons.append("model_list_frozen is not true")
    if config.protocol.get("final_test_authorized") is not True:
        reasons.append("final_test_authorized is not true")
    if reasons:
        raise GateBlocked("later-period validation blocked: " + "; ".join(reasons))


def assert_data_audit_ready(config: ProjectConfig) -> None:
    """The confirmatory data audit refuses while licence, target, or vintage is unset."""
    reasons = []
    if config.data.get("target_definition_status") in (None, "UNSET", "NOT_VERIFIED"):
        reasons.append("target_definition_status is unset")
    if config.data.get("licence_status") in (None, "UNSET", "NOT_VERIFIED"):
        reasons.append("licence_status is unset")
    if config.data.get("vintage_mode") in (None, "UNSET"):
        reasons.append("vintage_mode is unset")
    if reasons:
        raise GateBlocked(
            "data-audit refused because required fields are unset: " + "; ".join(reasons)
        )
