"""Audit entry points that do not acquire bulk data."""

from p12_bank_liquidity.audit.report import write_reports

__all__ = ["write_reports"]
