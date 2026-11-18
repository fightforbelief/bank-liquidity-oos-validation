# Deviations from the frozen confirmatory protocol

## 2026-11-19 — analyze.py type error (code fix, no change to the analysis)

The first `analyze.py bhc_cash_liquidity` run stopped with `AttributeError: 'Index' object has no attribute 'sum'`, before any result was printed or written. `eb.index.map(...)` returns a pandas Index, which has no `.sum()`. The line now builds the same boolean entrant mask as a NumPy array. Hypotheses, covariates, estimators, seeds, and decision rules are unchanged. The frozen hash of `analyze.py` in `FREEZE.txt` refers to the version before this fix; the fixed file's hash is appended below.
4a5b6bc5f20eaa15aecadfa7fcec7c35adce7225df29348c6d76cefa267bc718  confirm/analyze.py
