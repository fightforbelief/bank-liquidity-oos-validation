"""Command line. Audit commands do not download panels or fit models."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from p12_bank_liquidity import __version__
from p12_bank_liquidity.audit.report import write_reports
from p12_bank_liquidity.config import (
    assert_audit_only,
    assert_data_audit_ready,
    assert_later_period_allowed,
    assert_reproduction_allowed,
    load_config,
)
from p12_bank_liquidity.gates import GateBlocked
from p12_bank_liquidity.sources.ffiec_cdr import refuse_bulk_download


def _project_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[2]


def _config_path(root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return root / "configs" / "project.yaml"


def cmd_env_audit() -> int:
    print(f"p12 {__version__}")
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print("env-audit is read-only and does not install packages")
    return 0


def cmd_source_audit(config) -> int:
    assert_audit_only(config)
    print("source audit is metadata-only")
    print("NYU zip download: refused")
    print("WRDS and Bloomberg pulls: refused")
    if config.data.get("raw_data_download_authorized") is True:
        raise GateBlocked("source-audit will not run while a bulk download is armed")
    return 0


def cmd_target_audit(config) -> int:
    assert_audit_only(config)
    print("target formula text is recorded in reports/target_definition_audit.md")
    print("real raw-to-target checks remain BLOCKED")
    return 0


def cmd_vintage_audit(config) -> int:
    assert_audit_only(config)
    print("first-release vintages: NOT_AVAILABLE from the sources opened")
    print("config vintage_mode remains UNSET pending human acceptance")
    return 0


def cmd_data_audit(config) -> int:
    assert_audit_only(config)
    assert_data_audit_ready(config)
    return 0


def cmd_reproduce(config) -> int:
    assert_reproduction_allowed(config)
    return 0


def cmd_validate(config) -> int:
    assert_later_period_allowed(config)
    refuse_bulk_download()
    return 0


def cmd_report(root: Path) -> int:
    paths = write_reports(root / "reports", evidence_class="SOURCE_DOCUMENT")
    for path in paths:
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p12")
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "env-audit",
        "source-audit",
        "target-audit",
        "vintage-audit",
        "data-audit",
        "reproduce",
        "validate-later-period",
        "report",
    ):
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = _project_root(args.root)
    try:
        if args.command == "env-audit":
            return cmd_env_audit()
        if args.command == "report":
            return cmd_report(root)
        config = load_config(_config_path(root, args.config))
        if args.command == "source-audit":
            return cmd_source_audit(config)
        if args.command == "target-audit":
            return cmd_target_audit(config)
        if args.command == "vintage-audit":
            return cmd_vintage_audit(config)
        if args.command == "data-audit":
            return cmd_data_audit(config)
        if args.command == "reproduce":
            return cmd_reproduce(config)
        if args.command == "validate-later-period":
            return cmd_validate(config)
    except GateBlocked as blocked:
        print(f"BLOCKED: {blocked.reason}", file=sys.stderr)
        return blocked.exit_code
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
