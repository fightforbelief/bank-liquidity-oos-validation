# P12 bank cash-liquidity out-of-time validation

Paired comparison of quarterly grids for U.S. bank cash-liquidity forecasts. The paper is [`paper.pdf`](paper.pdf). The LaTeX source is `manuscripts/p12_candidate_a_grid_sensitivity/`.

The pre-specified primary contrast is entity-level mean MASE on the natural quarter-end grid minus mean MASE on the FTSFR implementation-faithful grid. A negative estimate means lower average MASE on the quarter-end grid. All six baseline models have negative estimates, and all six 95% intervals lie entirely below zero. Each interval uses 2,000 successful paired bootstrap replicates and zero failures, on 17,377 entities and 85,205 paired keys.

| Model | Estimate | 95% interval |
|---|---:|---|
| HistoricAverage | -0.0042836168277380965 | [-0.008591823072073467, -0.000610951458628103] |
| Naive | -0.010815696110028986 | [-0.015259849607384432, -0.006857415193080016] |
| SeasonalNaive | -0.003606282624498774 | [-0.00774629104284172, -0.0002676458368803533] |
| AutoARIMA | -0.010605197991587254 | [-0.014542165346988973, -0.006979203762667186] |
| SimpleExponentialSmoothing | -0.006683875293450963 | [-0.011230451149514731, -0.0031313732278946617] |
| AutoTheta | -0.011553343325476726 | [-0.015839678595276144, -0.0076307394386690534] |

This is a model-specific grid contrast. It does not test whether the ordering of the six models is unchanged, and the six intervals are not a familywise-adjusted joint region. The practical equivalence margin remains unset, so formal equivalence was not tested. The result does not establish robustness, stability, or a causal effect of the grid, and it is not a reproduction of published Tables 6–8.

## Run

From the repository root, with `PYTHONPATH=src`:

```text
python -m p12_bank_liquidity.cli env-audit
python -m pytest -q
```

Call Report microdata and fitted forecast artifacts are not in this repository.
