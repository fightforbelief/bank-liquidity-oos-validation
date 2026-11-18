"""Run the pre-registered post-hoc analyses. The reproduction gate blocks every later task."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import coverage
import load
from p12_bank_liquidity.candidate_a import stage18_inference
import make_tex
import p12_supplementary as S

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out"
BLOCKERS = ROOT / "supplementary" / "BLOCKERS.md"
DEVIATIONS = ROOT / "supplementary" / "DEVIATIONS.md"


def project_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    """Frozen Stage 18 draw. The supplied default uses rng.integers."""
    return rng.choice(n, size=n, replace=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def append_block(path: Path, text: str) -> None:
    current = path.read_text(encoding="utf-8").rstrip()
    if text.strip() in current:
        return
    path.write_text(current + "\n\n" + text.strip() + "\n", encoding="utf-8")


def recorded(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["model_id"]: row for row in payload["models"]}


def pack(result: S.BootResult, expected: dict) -> dict:
    got_estimate = repr(float(result.estimate[0]))
    got_lower = repr(float(result.lower[0]))
    got_upper = repr(float(result.upper[0]))
    match = (
        got_estimate == expected["estimate_repr"]
        and got_lower == expected["low_repr"]
        and got_upper == expected["high_repr"]
        and result.n_valid == 2000
        and result.n_failed == 0
    )
    return {
        "estimate": float(result.estimate[0]),
        "lower": float(result.lower[0]),
        "upper": float(result.upper[0]),
        "estimate_repr": got_estimate,
        "lower_repr": got_lower,
        "upper_repr": got_upper,
        "expected_estimate_repr": expected["estimate_repr"],
        "expected_lower_repr": expected["low_repr"],
        "expected_upper_repr": expected["high_repr"],
        "n_valid": result.n_valid,
        "n_failed": result.n_failed,
        "match": bool(match),
    }


def stage19_point(frame: pd.DataFrame) -> float:
    """Diagnostic only: frozen entity contrast, mean(B) - mean(A), then a NumPy mean."""
    deltas = []
    for _, group in frame.groupby("entity", sort=False):
        arm_a = [float(value) for value in group["A_mase"]]
        arm_b = [float(value) for value in group["B_mase"]]
        deltas.append(sum(arm_b) / len(arm_b) - sum(arm_a) / len(arm_a))
    return float(np.asarray(deltas, dtype=float).mean())


def environment() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "git_commit": commit,
        "draw": "numpy.Generator.choice(n, size=n, replace=True)",
        "primary_seed": S.PRIMARY_SEED,
        "posthoc_seed": S.POSTHOC_SEED,
        "n_boot": S.N_BOOT,
        "quantile_method": "linear",
    }


def frozen_entity_deltas(frame: pd.DataFrame) -> np.ndarray:
    """Entity deltas exactly as frozen `aggregate_paired_mase` builds them.

    Python-float sum / len per arm inside each entity, entities in order of first
    appearance in the saved pair file (dict insertion order in the frozen code).
    """
    pairs: dict[object, list[tuple[float, float]]] = {}
    for entity, a, b in zip(frame["entity"], frame["A_mase"], frame["B_mase"]):
        pairs.setdefault(entity, []).append((float(a), float(b)))
    deltas = []
    for items in pairs.values():
        mean_a = sum(item[0] for item in items) / len(items)
        mean_b = sum(item[1] for item in items) / len(items)
        deltas.append(mean_b - mean_a)
    return np.asarray(deltas, dtype=float)


def frozen_gate_bootstrap(frame: pd.DataFrame) -> S.BootResult:
    """Run the frozen Stage 18 bootstrap code itself, not a re-implementation."""
    out = stage18_inference.cluster_bootstrap(frozen_entity_deltas(frame), seed=S.PRIMARY_SEED, replicates=S.N_BOOT)
    if out["interval"] != "available":
        raise RuntimeError(f"frozen bootstrap unavailable: {out['reason']}")
    return S.BootResult(
        np.array([out["estimate"]]),
        np.array([out["low"]]),
        np.array([out["high"]]),
        int(out["valid"]),
        int(out["failed"]),
    )


def run_gate(frame: pd.DataFrame) -> dict:
    primary = recorded(load.RUN / "primary_results.json")
    secondary = recorded(load.RUN / "secondary_results.json")
    report: dict = {
        "primary": {},
        "secondary": {},
        "draw": environment()["draw"],
        "gate_code_path": "p12_bank_liquidity.candidate_a.stage18_inference.cluster_bootstrap on frozen aggregate_paired_mase entity deltas",
    }
    exact = True
    for model in S.MODELS:
        print(f"GATE {model}", flush=True)
        wide = S.pair_wide(frame, model)
        got_primary = frozen_gate_bootstrap(wide)
        got_secondary = frozen_gate_bootstrap(wide.loc[wide["same_origin"]].copy())
        report["primary"][model] = pack(got_primary, primary[model])
        report["secondary"][model] = pack(got_secondary, secondary[model])
        exact = exact and report["primary"][model]["match"] and report["secondary"][model]["match"]
        print(f"GATE {model} {report['primary'][model]['match']} {report['secondary'][model]['match']}", flush=True)
    report["exact_match"] = bool(exact)
    if not exact:
        report["stage19_point_diagnostic"] = {}
        for model in S.MODELS:
            wide = S.pair_wide(frame, model)
            primary_point = stage19_point(wide)
            secondary_point = stage19_point(wide.loc[wide["same_origin"]].copy())
            report["stage19_point_diagnostic"][model] = {
                "primary_repr": repr(primary_point),
                "primary_matches_recorded": repr(primary_point) == report["primary"][model]["expected_estimate_repr"],
                "secondary_repr": repr(secondary_point),
                "secondary_matches_recorded": repr(secondary_point) == report["secondary"][model]["expected_estimate_repr"],
            }
    return report


def blocker_for_gate(report: dict) -> str:
    lines = [
        "## 2026-11-06 — reproduction gate failed",
        "",
        "T2--T10 were not run. `reproduce_primary` and `cluster_bootstrap` were not modified.",
        "The draw passed to every gate bootstrap was `Generator.choice(n, size=n, replace=True)`.",
        "",
    ]
    for role in ("primary", "secondary"):
        for model in S.MODELS:
            row = report[role][model]
            if row["match"]:
                continue
            lines.append(
                f"- {role} {model}: got {row['estimate_repr']} "
                f"[{row['lower_repr']}, {row['upper_repr']}], "
                f"recorded {row['expected_estimate_repr']} "
                f"[{row['expected_lower_repr']}, {row['expected_upper_repr']}], "
                f"valid {row['n_valid']}, failed {row['n_failed']}."
            )
    diagnostic = report.get("stage19_point_diagnostic", {})
    if diagnostic:
        lines.append("")
        lines.append("Point-only diagnostic using mean(arm B) - mean(arm A) inside each entity, then a NumPy mean across entities:")
        for model in S.MODELS:
            item = diagnostic[model]
            lines.append(
                f"- {model}: primary {item['primary_repr']} recorded_match {item['primary_matches_recorded']}; "
                f"secondary {item['secondary_repr']} recorded_match {item['secondary_matches_recorded']}."
            )
    return "\n".join(lines)


def mechanism(frame: pd.DataFrame) -> dict:
    reference = None
    for model in S.MODELS:
        part = frame.loc[frame["model"] == model, ["arm", "zero_seasonal_share", "n_imputed_train", "n_train_obs"]]
        part = part.reset_index(drop=True)
        if reference is None:
            reference = part
            continue
        if not np.array_equal(reference["zero_seasonal_share"].to_numpy(), part["zero_seasonal_share"].to_numpy()):
            raise RuntimeError("zero_seasonal_share differs across models")
        if not np.array_equal(reference["n_imputed_train"].to_numpy(), part["n_imputed_train"].to_numpy()):
            raise RuntimeError("imputed-training counts differ across models")
        if not np.array_equal(reference["n_train_obs"].to_numpy(), part["n_train_obs"].to_numpy()):
            raise RuntimeError("training lengths differ across models")
    assert reference is not None
    out = {}
    for arm, label in (("A", "A"), ("B", "B")):
        rows = reference.loc[reference["arm"] == arm]
        ratio = rows["n_imputed_train"].to_numpy(dtype=float) / rows["n_train_obs"].to_numpy(dtype=float)
        out[f"zero_seasonal_share_{label}"] = float(rows["zero_seasonal_share"].mean())
        out[f"imputed_share_{label}"] = float(ratio.mean())
    return out


def origin_descriptive(frame: pd.DataFrame) -> dict:
    signature = None
    for model in S.MODELS:
        wide = S.pair_wide(frame, model)
        current = wide[
            ["same_origin", "A_origin_is_synthetic", "B_origin_is_synthetic", "A_staleness_q", "B_staleness_q"]
        ].reset_index(drop=True)
        if signature is None:
            signature = current
            continue
        for column in current.columns:
            if not np.array_equal(signature[column].to_numpy(), current[column].to_numpy()):
                raise RuntimeError(f"origin metadata differs across models in {column}")
    assert signature is not None
    out = {}
    for key, mask in (("diff", ~signature["same_origin"]), ("same", signature["same_origin"])):
        out[key] = {}
        part = signature.loc[mask]
        for arm in ("A", "B"):
            stale = part[f"{arm}_staleness_q"].to_numpy(dtype=float)
            synthetic = part[f"{arm}_origin_is_synthetic"].to_numpy(dtype=bool)
            out[key][arm] = {
                "n_keys": int(len(part)),
                "synthetic_share": float(synthetic.mean()),
                "staleness_median": float(np.median(stale)),
                "staleness_mean": float(stale.mean()),
            }
    if out["diff"]["A"]["n_keys"] != 1190 or out["same"]["A"]["n_keys"] != 84015:
        raise RuntimeError("origin descriptive split is not 1190/84015")
    return out


def other_blocker(exclusive: dict) -> str:
    lines = []
    for title, key in (("Arm B-only", "arm_b_only"), ("Arm A-only", "arm_a_only")):
        for rule in exclusive[key]["rules"]:
            if rule["rule"] != "other" or int(rule["count"]) == 0:
                continue
            lines.append(f"## 2026-11-06 — {title} class other ({int(rule['count'])} keys)")
            lines.append("")
            lines.append("Entity hashes are not listed. Every key in a row has the stated description.")
            for sub in rule["subs"]:
                lines.append(
                    f"- {int(sub['count'])} keys: {sub['label']}. "
                    "The target is on the other arm's saved grid, the stored source value is finite, "
                    "and the frozen numeric checks pass for the grid stamps before that target. "
                    "The target is not a saved schedule row of the other arm."
                )
    return "\n".join(lines)


def notes(results: dict) -> str:
    anchor = results["coverage"]["anchor"]["drift"]
    pieces = [
        f"Saved Arm A grids: drift occurs {anchor}. "
        "The manuscript illustrates a 30 June step of 30 September, 30 December, 30 March, 30 June."
    ]
    opposite = []
    for model in S.MODELS:
        row = results["decomposition"][model]
        num, den = float(row["num_term"]), float(row["den_term"])
        if num == 0.0 or den == 0.0 or (num > 0.0) != (den > 0.0):
            opposite.append(f"{model} error {num:.4f} scale {den:.4f}")
    if opposite:
        pieces.append("Error and scale terms are not both strictly negative or both strictly positive for: " + "; ".join(opposite) + ".")
    else:
        pieces.append("For every model the error term and the scale term are both strictly negative or both strictly positive.")
    ranking = results["ranking"]
    if ranking["aggregate_order_identical"]:
        pieces.append("The aggregate model order is the same on Arm A and Arm B.")
    else:
        pieces.append(
            "Aggregate order A: " + ", ".join(ranking["aggregate_order_A"]) + ". "
            "Aggregate order B: " + ", ".join(ranking["aggregate_order_B"]) + "."
        )
    pieces.append("Unjoined source filings are not counted. The saved grid does not contain dates that failed to join it.")
    return " ".join(pieces)


def protocol_inputs() -> list[str]:
    text = (ROOT / "supplementary" / "POSTHOC_PROTOCOL.md").read_text(encoding="utf-8")
    return [line[2:] for line in text.splitlines() if line.startswith("- ") and "SHA-256" in line]


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    versions = environment()
    if not versions["python"].startswith("3.12") or versions["numpy"] != "2.0.2":
        append_block(
            BLOCKERS,
            "\n".join(
                [
                    "## 2026-11-06 — environment",
                    "",
                    f"The run requires Python 3.12 and NumPy 2.0.2. This interpreter is Python {versions['python']} and NumPy {versions['numpy']}.",
                    "No estimator was started.",
                ]
            ),
        )
        print("ENV_FAIL", versions["python"], versions["numpy"], flush=True)
        return 2
    print("LOAD_LONG", flush=True)
    frame = load.load_long()
    print("IDENTITY", load.LAST_IDENTITY, flush=True)
    report = run_gate(frame)
    report["environment"] = environment()
    report["identity"] = dict(load.LAST_IDENTITY)
    write_json(OUT / "gate.json", report)
    if not report["exact_match"]:
        append_block(BLOCKERS, blocker_for_gate(report))
        print("GATE_FAIL", flush=True)
        return 2
    print("POSTHOC", flush=True)
    levels, distribution, decomposition, origin = {}, {}, {}, {}
    estimates = {}
    for model in S.MODELS:
        print(f"MODEL {model}", flush=True)
        wide = S.pair_wide(frame, model)
        estimate = float(S.entity_table(wide)["delta"].mean())
        estimates[model] = estimate
        levels[model] = S.levels(wide, draw=project_draw)
        distribution[model] = S.distribution(wide, draw=project_draw)
        decomposition[model] = S.decomposition(wide, draw=project_draw)
        origin[model] = S.origin_contribution(wide)
        num = float(decomposition[model]["num_term"])
        den = float(decomposition[model]["den_term"])
        if abs(num + den - estimate) > 1e-12:
            raise RuntimeError(f"{model} Shapley split does not match the primary estimate")
        if abs(float(origin[model]["total"]) - estimate) > 1e-12:
            raise RuntimeError(f"{model} origin split does not match the primary estimate")
    print("RANKING", flush=True)
    ranking = S.ranking(frame, draw=project_draw)
    print("COVERAGE", flush=True)
    inputs = load.read_inputs()
    anchor = coverage.grid_anchor(inputs["prepared"])
    exclusive = coverage.classify_exclusive(inputs["schedule"], inputs["prepared"])
    monthdays = coverage.origin_monthdays(inputs["pairs"])
    extra = other_blocker(exclusive)
    if extra:
        append_block(BLOCKERS, extra)
    results = {
        "gate": {"exact_match": True},
        "primary_estimate": estimates,
        "levels": levels,
        "distribution": distribution,
        "decomposition": decomposition,
        "mechanism": mechanism(frame),
        "origin": origin,
        "origin_descriptive": origin_descriptive(frame),
        "origin_monthdays": monthdays,
        "coverage": {
            "anchor": anchor,
            "exclusive": exclusive,
            "unjoined_source_filings": None,
            "unjoined_source_filings_reason": "saved prepared_series contains only grid rows",
        },
        "ranking": ranking,
    }
    results["notes"] = notes(results)
    write_json(OUT / "results.json", results)
    manifest = environment()
    manifest.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "identity": dict(load.LAST_IDENTITY),
            "inputs_sha256": protocol_inputs(),
            "batch_names": _CACHE_batches(),
        }
    )
    write_json(OUT / "manifest.json", manifest)
    deviation = make_tex.emit(
        json.loads((OUT / "results.json").read_text(encoding="utf-8")),
        ROOT / "paper_v2" / "generated",
        OUT / "RESULTS_SUMMARY.md",
        BLOCKERS,
        DEVIATIONS,
    )
    if deviation:
        append_block(DEVIATIONS, deviation)
        make_tex.write_summary(
            json.loads((OUT / "results.json").read_text(encoding="utf-8")),
            BLOCKERS.read_text(encoding="utf-8"),
            DEVIATIONS.read_text(encoding="utf-8"),
            OUT / "RESULTS_SUMMARY.md",
        )
    print("DONE", flush=True)
    return 0


def _CACHE_batches() -> list[str]:
    names = load._CACHE.get("batch_names")
    return list(names) if isinstance(names, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
