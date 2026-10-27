# P12 bank cash-liquidity out-of-time validation

Provisional reproduction and external-validation audit. This checkout does not contain Call Report microdata and does not fit forecasting models.

Gate 0 outcome recorded in `reports/gate0_gate1_decision.md`: `PIVOT` (no novelty claim).
Gate 1 outcome: `BLOCKED` until a person accepts the data terms, three real raw-to-target checks are done, and the vintage label is chosen.

## Audit commands

From the repository root, with `PYTHONPATH=src`:

```text
python -m p12_bank_liquidity.cli env-audit
python -m p12_bank_liquidity.cli source-audit
python -m p12_bank_liquidity.cli target-audit
python -m p12_bank_liquidity.cli vintage-audit
python -m p12_bank_liquidity.cli data-audit
python -m p12_bank_liquidity.cli reproduce
python -m p12_bank_liquidity.cli validate-later-period
python -m p12_bank_liquidity.cli report
```

`data-audit`, `reproduce`, and `validate-later-period` exit 2 while `configs/project.yaml` keeps target, licence, vintage, and compute flags unset. That is the intended fail-closed state.

Research notes copied into `docs/` are the handoff pack. Empirical claims belong only in `reports/` after an approved run.

## Dependencies for this stage

The audit environment installs `pytest` and `PyYAML` only. PyTorch, CUDA, pandas, and statsmodels are not required and are not installed for this gate.
