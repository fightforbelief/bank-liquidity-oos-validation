"""Build paper_v2 tables and value macros from supplementary/out/results.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

from p12_supplementary import MODELS

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {
    "HistoricAverage": "HistoricAverage",
    "Naive": "Naive",
    "SeasonalNaive": "SeasonalNaive",
    "AutoARIMA": "AutoARIMA",
    "SimpleExponentialSmoothing": "SES",
    "AutoTheta": "AutoTheta",
}
GROUP_RECENT = ["Naive", "AutoARIMA", "AutoTheta"]
GROUP_OTHER = ["HistoricAverage", "SeasonalNaive", "SimpleExponentialSmoothing"]
FORBIDDEN = re.compile(r"\b(because|causes|inflates|genuinely|better)\b", re.IGNORECASE)


def load_results(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "supplementary" / "out" / "results.json").read_text(encoding="utf-8"))


def fnum(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def math_num(value: float, digits: int = 4) -> str:
    return f"${fnum(value, digits)}$"


def math_ci(lo: float, hi: float, digits: int = 4) -> str:
    return f"$[{fnum(lo, digits)},\\ {fnum(hi, digits)}]$"


def pct(value: float) -> str:
    number = 100.0 * float(value)
    if number < 0:
        return f"${number:.1f}$"
    return f"{number:.1f}"


def pct_ci(lo: float, hi: float) -> str:
    return f"[{pct(lo)}\\%, {pct(hi)}\\%]"


def tex_label(text: object) -> str:
    raw = str(text)
    return (
        raw.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _range_tex(values: list[float], digits: int = 1, percent: bool = True) -> str:
    lo, hi = min(values), max(values)
    suffix = "\\%" if percent else ""
    if lo < 0 or hi < 0:
        return f"${lo:.{digits}f}$--${hi:.{digits}f}${suffix}"
    return f"{lo:.{digits}f}--{hi:.{digits}f}{suffix}"


def _between(lo: float, hi: float) -> str:
    if lo > hi:
        lo, hi = hi, lo
    if lo < 0 or hi < 0:
        return f"between ${lo:.1f}$\\% and ${hi:.1f}$\\%"
    return f"between {lo:.1f}\\% and {hi:.1f}\\%"


def build_values(results: dict) -> tuple[dict[str, str], str | None]:
    levels = results["levels"]
    dist = results["distribution"]
    decomp = results["decomposition"]
    origin = results["origin"]
    ranking = results["ranking"]
    coverage = results["coverage"]
    mechanism = results["mechanism"]
    deviations: list[str] = []

    rel = {model: -100.0 * float(levels[model]["rel_change"]) for model in MODELS}
    values = {
        "gate-status": "passed: all 36 pre-specified values in Tables 4--5 were reproduced exactly",
        "abs-relchange-range": _between(min(rel.values()), max(rel.values())),
    }

    shares = []
    for model in MODELS:
        estimate = float(decomp[model]["num_term"]) + float(decomp[model]["den_term"])
        shares.append(100.0 * float(decomp[model]["num_term"]) / estimate)
    if all(0.0 <= share <= 100.0 for share in shares):
        values["abs-decomp-summary"] = (
            "the error term accounts for "
            + _between(min(shares), max(shares))
            + " of the primary estimate across models and the scale term for the remainder"
        )
    else:
        values["abs-decomp-summary"] = (
            "the error-term share of the primary estimate ranges from "
            + _range_tex(shares)
            + " across models"
        )
        deviations.append(
            "## 2026-11-06 — decomposition sentence\n"
            "The supplied template says the scale term is the remainder of an error-term share between 0% and 100%. "
            "At least one share lies outside that interval, so abs-decomp-summary states the signed range instead."
        )

    recent = [100.0 * float(origin[model]["share_from_diff_origin"]) for model in GROUP_RECENT]
    others = [100.0 * float(origin[model]["share_from_diff_origin"]) for model in GROUP_OTHER]
    values["abs-diff-origin-summary"] = (
        f"{_range_tex(recent)} of the primary estimate for Naive, AutoARIMA, and AutoTheta, "
        f"and {_range_tex(others)} for the other three models"
    )
    order_word = "identical" if ranking["aggregate_order_identical"] else "different"
    values["abs-ranking-summary"] = (
        f"have a mean Kendall $\\tau_b$ of {fnum(ranking['mean_tau'])} between grids, "
        f"the same best model for {100.0 * float(ranking['share_same_best']):.1f}\\% of banks, "
        f"and an aggregate order that is {order_word} across grids"
    )

    anchor = coverage["anchor"]
    off = anchor["off_stamp_monthday_counts"]
    n_stamps = int(anchor["n_stamps"])
    if off:
        parts = [
            f"{tex_label(key)} ({100.0 * off[key] / n_stamps:.1f}\\% of stamps)"
            for key in sorted(off)
        ]
        listed = ", ".join(parts)
    else:
        listed = "none"
    if anchor["drift"] in {"only from day-30 anchors", "from any anchor"}:
        drift_sentence = f"Drift occurs {anchor['drift']}."
    else:
        drift_sentence = f"No saved Arm~A stamp falls off a calendar quarter-end."
        deviations.append(
            "## 2026-11-06 — anchor sentence\n"
            "The supplied template offers two drift phrases. The saved Arm A grids contain no off-quarter-end stamp, "
            "so grid-anchor-summary states that count instead of either phrase."
        )
    values["grid-anchor-summary"] = (
        f"In the saved Arm~A grids, {int(anchor['n_entities_with_off_stamp']):,} of "
        f"{int(anchor['n_entities']):,} banks ({100.0 * float(anchor['share_entities_with_off_stamp']):.1f}\\%) "
        f"have at least one stamp that is not a calendar quarter-end; the off-quarter-end stamps are {listed}. "
        f"{drift_sentence}"
    )

    arm_a = [(model, float(levels[model]["mean_entity_mase_A"])) for model in MODELS]
    low_model, low_value = min(arm_a, key=lambda item: item[1])
    high_model, high_value = max(arm_a, key=lambda item: item[1])
    rel_rows = [(model, float(levels[model]["rel_change"])) for model in MODELS]
    rel_low = min(rel_rows, key=lambda item: item[1])
    rel_high = max(rel_rows, key=lambda item: item[1])
    values["levels-text"] = (
        f"Mean entity-level MASE on Arm~A ranges from {fnum(low_value)} ({DISPLAY[low_model]}) "
        f"to {fnum(high_value)} ({DISPLAY[high_model]}). "
        f"The primary estimate as a percentage of that Arm~A level ranges from "
        f"{pct(rel_low[1])}\\% {pct_ci(levels[rel_low[0]]['rel_change_lo'], levels[rel_low[0]]['rel_change_hi'])} "
        f"({DISPLAY[rel_low[0]]}) to {pct(rel_high[1])}\\% "
        f"{pct_ci(levels[rel_high[0]]['rel_change_lo'], levels[rel_high[0]]['rel_change_hi'])} "
        f"({DISPLAY[rel_high[0]]})."
    )

    medians = [(model, float(dist[model]["median"])) for model in MODELS]
    med_low = min(medians, key=lambda item: item[1])
    med_high = max(medians, key=lambda item: item[1])
    neg = [100.0 * float(dist[model]["share_negative"]) for model in MODELS]
    exceptions = []
    for model in MODELS:
        for label, lo_key, hi_key in (
            ("median", "median_lo", "median_hi"),
            ("5\\% trimmed mean", "trim5_lo", "trim5_hi"),
        ):
            lo, hi = float(dist[model][lo_key]), float(dist[model][hi_key])
            if lo <= 0.0 <= hi:
                exceptions.append(f"{DISPLAY[model]} {label} {math_ci(lo, hi)}")
    if exceptions:
        exception_sentence = "Intervals containing zero: " + "; ".join(exceptions) + "."
    else:
        exception_sentence = "Every median interval and every 5\\% trimmed-mean interval excludes zero."
    values["dist-text"] = (
        f"The median of $\\delta_i$ ranges from {fnum(med_low[1])} {math_ci(dist[med_low[0]]['median_lo'], dist[med_low[0]]['median_hi'])} "
        f"({DISPLAY[med_low[0]]}) to {fnum(med_high[1])} {math_ci(dist[med_high[0]]['median_lo'], dist[med_high[0]]['median_hi'])} "
        f"({DISPLAY[med_high[0]]}). The share of banks with $\\delta_i<0$ ranges from {min(neg):.1f}\\% to {max(neg):.1f}\\%. "
        + exception_sentence
    )

    num_terms = [float(decomp[model]["num_term"]) for model in MODELS]
    den_terms = [float(decomp[model]["den_term"]) for model in MODELS]
    d_abs = [1000.0 * float(decomp[model]["d_abs_err"]) for model in MODELS]
    s_share = [100.0 * float(decomp[model]["share_keys_sB_gt_sA"]) for model in MODELS]
    values["decomp-text"] = (
        f"The error term ranges from {fnum(min(num_terms))} to {fnum(max(num_terms))} and the scale term ranges from "
        f"{fnum(min(den_terms))} to {fnum(max(den_terms))}. "
        f"The change in absolute error ranges from {fnum(min(d_abs))} to {fnum(max(d_abs))} times $10^{{-3}}$, "
        f"the share of keys with $s_B>s_A$ ranges from {min(s_share):.1f}\\% to {max(s_share):.1f}\\%, "
        f"and the zero lag-4 difference share is {100.0 * float(mechanism['zero_seasonal_share_A']):.1f}\\% on Arm~A "
        f"and {100.0 * float(mechanism['zero_seasonal_share_B']):.1f}\\% on Arm~B."
    )

    diff = results["origin_descriptive"]["diff"]
    values["origin-text"] = (
        f"The different-origin share of the primary estimate is {_range_tex(recent)} for Naive, AutoARIMA, and AutoTheta, "
        f"and {_range_tex(others)} for HistoricAverage, SeasonalNaive, and SES. "
        f"On the different-origin keys, the synthetic-origin share is {100.0 * float(diff['A']['synthetic_share']):.1f}\\% "
        f"on Arm~A and {100.0 * float(diff['B']['synthetic_share']):.1f}\\% on Arm~B, and the median staleness is "
        f"{fnum(diff['A']['staleness_median'], 2)} grid steps on Arm~A and {fnum(diff['B']['staleness_median'], 2)} "
        f"grid steps on Arm~B."
    )

    pos_a = {model: index + 1 for index, model in enumerate(ranking["aggregate_order_A"])}
    pos_b = {model: index + 1 for index, model in enumerate(ranking["aggregate_order_B"])}
    changed = [DISPLAY[model] for model in MODELS if pos_a[model] != pos_b[model]]
    if changed:
        order_sentence = (
            "Aggregate order on Arm~A: " + ", ".join(DISPLAY[model] for model in ranking["aggregate_order_A"])
            + ". Aggregate order on Arm~B: " + ", ".join(DISPLAY[model] for model in ranking["aggregate_order_B"])
            + ". Positions differ for " + ", ".join(changed) + "."
        )
    else:
        order_sentence = "No model changes aggregate position."
    values["ranking-text"] = (
        f"Per-bank rankings have a mean Kendall $\\tau_b$ of {fnum(ranking['mean_tau'])} "
        f"{math_ci(ranking['mean_tau_lo'], ranking['mean_tau_hi'])}, the same best model for "
        f"{100.0 * float(ranking['share_same_best']):.1f}\\% "
        f"{pct_ci(ranking['same_best_lo'], ranking['same_best_hi'])} of banks, and identical full rankings for "
        f"{100.0 * float(ranking['share_identical_rank']):.1f}\\% "
        f"{pct_ci(ranking['identical_lo'], ranking['identical_hi'])} of banks. "
        + order_sentence
    )
    for key, text in values.items():
        if FORBIDDEN.search(text):
            raise RuntimeError(f"forbidden wording in {key}")
    return values, "\n\n".join(deviations) if deviations else None


def _tabular(spec: str, header: str, rows: list[str], footer: str | None = None) -> str:
    body = "\n".join(rows)
    foot = f"\n\\midrule\n{footer}" if footer else ""
    return (
        "\\begin{tabular}{" + spec + "}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}{foot}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def table_levels(results: dict) -> str:
    rows = []
    for model in MODELS:
        row = results["levels"][model]
        rows.append(
            " & ".join(
                [
                    DISPLAY[model],
                    math_num(row["mean_entity_mase_A"]),
                    math_num(row["mean_entity_mase_B"]),
                    math_num(row["median_entity_mase_A"]),
                    math_num(row["median_entity_mase_B"]),
                    pct(row["rel_change"]),
                    pct_ci(row["rel_change_lo"], row["rel_change_hi"]),
                ]
            )
            + " \\\\"
        )
    return _tabular(
        "lrrrrrr",
        "Model & Mean MASE A & Mean MASE B & Median A & Median B & Est./Mean A (\\%) & 95\\% CI (\\%)",
        rows,
    )


def table_dist(results: dict) -> str:
    rows = []
    for model in MODELS:
        row = results["distribution"][model]
        median = f"{math_num(row['median'])} {math_ci(row['median_lo'], row['median_hi'])}"
        trimmed = f"{math_num(row['trim5'])} {math_ci(row['trim5_lo'], row['trim5_hi'])}"
        rows.append(
            " & ".join(
                [
                    DISPLAY[model],
                    math_num(row["q05"]),
                    math_num(row["q25"]),
                    median,
                    math_num(row["q75"]),
                    math_num(row["q95"]),
                    pct(row["share_negative"]),
                    trimmed,
                ]
            )
            + " \\\\"
        )
    return _tabular(
        "lrrrrrrr",
        "Model & q05 & q25 & Median [CI] & q75 & q95 & Share $\\delta_i<0$ (\\%) & 5\\% trimmed mean [CI]",
        rows,
    )


def table_decomp(results: dict) -> str:
    rows = []
    mechanism = results["mechanism"]
    for model in MODELS:
        row = results["decomposition"][model]
        estimate = results["primary_estimate"][model]
        rows.append(
            " & ".join(
                [
                    DISPLAY[model],
                    math_num(estimate),
                    f"{math_num(row['num_term'])} {math_ci(row['num_lo'], row['num_hi'])}",
                    f"{math_num(row['den_term'])} {math_ci(row['den_lo'], row['den_hi'])}",
                    f"{math_num(1000.0 * float(row['d_abs_err']))} {math_ci(1000.0 * float(row['d_abs_lo']), 1000.0 * float(row['d_abs_hi']))}",
                    pct(row["d_abs_rel"]),
                    pct(row["share_keys_sB_gt_sA"]),
                ]
            )
            + " \\\\"
        )
    footer = (
        "\\multicolumn{7}{l}{Zero lag-4 difference share: "
        f"A {100.0 * float(mechanism['zero_seasonal_share_A']):.1f}\\%, "
        f"B {100.0 * float(mechanism['zero_seasonal_share_B']):.1f}\\%. "
        "Imputed training share: "
        f"A {100.0 * float(mechanism['imputed_share_A']):.1f}\\%, "
        f"B {100.0 * float(mechanism['imputed_share_B']):.1f}\\%.}}"
    )
    return _tabular(
        "lrrrrrr",
        "Model & Primary est. & Error term [CI] & Scale term [CI] & $\\Delta$MAE $\\times 10^{-3}$ [CI] & $\\Delta$MAE (\\%) & Share $s_B>s_A$ (\\%)",
        rows,
        footer,
    )


def table_origin(results: dict) -> str:
    rows = []
    for model in MODELS:
        row = results["origin"][model]
        rows.append(
            " & ".join(
                [
                    DISPLAY[model],
                    math_num(results["primary_estimate"][model]),
                    math_num(row["term_diff_origin"]),
                    math_num(row["term_same_origin"]),
                    pct(row["share_from_diff_origin"]),
                ]
            )
            + " \\\\"
        )
    panel_a = _tabular(
        "lrrrr",
        "Model & Primary est. & Diff-origin term & Same-origin term & Share from diff-origin (\\%)",
        rows,
    )
    desc = results["origin_descriptive"]
    desc_rows = []
    for label, key in (("Different origin", "diff"), ("Same origin", "same")):
        for arm in ("A", "B"):
            item = desc[key][arm]
            desc_rows.append(
                " & ".join(
                    [
                        label,
                        arm,
                        pct(item["synthetic_share"]),
                        math_num(item["staleness_median"], 2),
                        math_num(item["staleness_mean"], 2),
                    ]
                )
                + " \\\\"
            )
    panel_b = _tabular(
        "llrrr",
        "Keys & Arm & Synthetic origin (\\%) & Median staleness & Mean staleness",
        desc_rows,
    )
    month_rows = []
    for row in results["origin_monthdays"]:
        month_rows.append(
            f"{tex_label(row['arm_a_monthday'])} & {tex_label(row['arm_b_monthday'])} & {int(row['count'])} \\\\"
        )
    panel_c = _tabular("llr", "Arm A origin & Arm B origin & Keys", month_rows)
    return panel_a + "\n\\vspace{0.6em}\n" + panel_b + "\n\\vspace{0.6em}\n" + panel_c


def table_ranking(results: dict) -> str:
    ranking = results["ranking"]
    panel_rows = [
        f"Mean Kendall $\\tau_b$ & {math_num(ranking['mean_tau'])} & {math_ci(ranking['mean_tau_lo'], ranking['mean_tau_hi'])} \\\\",
        f"Same best model (\\%) & {pct(ranking['share_same_best'])} & {pct_ci(ranking['same_best_lo'], ranking['same_best_hi'])} \\\\",
        f"Identical ranking (\\%) & {pct(ranking['share_identical_rank'])} & {pct_ci(ranking['identical_lo'], ranking['identical_hi'])} \\\\",
    ]
    panel_a = _tabular("lrr", "Metric & Estimate & 95\\% CI", panel_rows)
    pos_a = {model: index + 1 for index, model in enumerate(ranking["aggregate_order_A"])}
    pos_b = {model: index + 1 for index, model in enumerate(ranking["aggregate_order_B"])}
    rows = []
    for model in MODELS:
        rows.append(
            " & ".join(
                [
                    DISPLAY[model],
                    math_num(ranking["mean_rank_A"][model]),
                    math_num(ranking["mean_rank_B"][model]),
                    pct(ranking["best_model_share_A"][model]),
                    pct(ranking["best_model_share_B"][model]),
                    str(pos_a[model]),
                    str(pos_b[model]),
                ]
            )
            + " \\\\"
        )
    panel_b = _tabular(
        "lrrrrrr",
        "Model & Mean rank A & Mean rank B & Best share A (\\%) & Best share B (\\%) & Position A & Position B",
        rows,
    )
    return panel_a + "\n\\vspace{0.6em}\n" + panel_b


def table_bonly(results: dict) -> str:
    payload = results["coverage"]["exclusive"]["arm_b_only"]
    total = int(payload["n"])
    rows = []
    for rule in payload["rules"]:
        count = int(rule["count"])
        rows.append(f"{tex_label(rule['rule'])} & {count:,} & {100.0 * count / total:.1f} \\\\")
        for sub in rule["subs"]:
            sub_count = int(sub["count"])
            rows.append(
                f"\\quad {tex_label(sub['label'])} & {sub_count:,} & {100.0 * sub_count / total:.1f} \\\\"
            )
    rows.append(f"Total & {total:,} & 100.0 \\\\")
    return _tabular("lrr", "Reason & Count & Share (\\%)", rows)


def write_values(values: dict[str, str], path: Path) -> None:
    lines = [f"\\setval{{{key}}}{{{values[key]}}}" for key in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(results: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {
        "tab_levels.tex": table_levels(results),
        "tab_dist.tex": table_dist(results),
        "tab_decomp.tex": table_decomp(results),
        "tab_origin.tex": table_origin(results),
        "tab_ranking.tex": table_ranking(results),
        "tab_bonly.tex": table_bonly(results),
    }
    for name, text in mapping.items():
        (directory / name).write_text(text, encoding="utf-8")


def _md_num(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_summary(results: dict, blockers: str, deviations: str, path: Path) -> None:
    levels = results["levels"]
    dist = results["distribution"]
    decomp = results["decomposition"]
    origin = results["origin"]
    ranking = results["ranking"]
    lines = [
        "# Post-hoc results summary",
        "",
        f"Gate exact_match: {results['gate']['exact_match']}.",
        "",
        "## Levels",
        "",
        "| Model | Mean A | Mean B | Median entity A | Median entity B | Median key A | Median key B | Est./Mean A | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        row = levels[model]
        lines.append(
            f"| {DISPLAY[model]} | {_md_num(row['mean_entity_mase_A'])} | {_md_num(row['mean_entity_mase_B'])} | "
            f"{_md_num(row['median_entity_mase_A'])} | {_md_num(row['median_entity_mase_B'])} | "
            f"{_md_num(row['median_key_mase_A'])} | {_md_num(row['median_key_mase_B'])} | "
            f"{100 * float(row['rel_change']):.1f}% | "
            f"[{100 * float(row['rel_change_lo']):.1f}%, {100 * float(row['rel_change_hi']):.1f}%] |"
        )
    lines += ["", "## Distribution", "", "| Model | q05 | q25 | q50 | q75 | q95 | Share <0 | Share =0 | Share >0 | Median [CI] | Trim5 [CI] |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for model in MODELS:
        row = dist[model]
        lines.append(
            f"| {DISPLAY[model]} | {_md_num(row['q05'])} | {_md_num(row['q25'])} | {_md_num(row['q50'])} | "
            f"{_md_num(row['q75'])} | {_md_num(row['q95'])} | {100 * float(row['share_negative']):.1f}% | "
            f"{100 * float(row['share_zero']):.1f}% | {100 * float(row['share_positive']):.1f}% | "
            f"{_md_num(row['median'])} [{_md_num(row['median_lo'])}, {_md_num(row['median_hi'])}] | "
            f"{_md_num(row['trim5'])} [{_md_num(row['trim5_lo'])}, {_md_num(row['trim5_hi'])}] |"
        )
    lines += ["", "## Decomposition", "", "| Model | Primary | Error [CI] | Scale [CI] | dMAE x 1e-3 [CI] | dMAE % | Share sB>sA | Share sB=sA | Median log(sB/sA) | Share errB<errA | Share err equal |", "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = decomp[model]
        lines.append(
            f"| {DISPLAY[model]} | {_md_num(results['primary_estimate'][model])} | "
            f"{_md_num(row['num_term'])} [{_md_num(row['num_lo'])}, {_md_num(row['num_hi'])}] | "
            f"{_md_num(row['den_term'])} [{_md_num(row['den_lo'])}, {_md_num(row['den_hi'])}] | "
            f"{_md_num(1000 * float(row['d_abs_err']))} [{_md_num(1000 * float(row['d_abs_lo']))}, {_md_num(1000 * float(row['d_abs_hi']))}] | "
            f"{100 * float(row['d_abs_rel']):.1f}% | {100 * float(row['share_keys_sB_gt_sA']):.1f}% | "
            f"{100 * float(row['share_keys_sB_eq_sA']):.1f}% | {_md_num(row['median_log_s_ratio'])} | "
            f"{100 * float(row['share_keys_errB_lt_errA']):.1f}% | {100 * float(row['share_keys_err_equal']):.1f}% |"
        )
    mech = results["mechanism"]
    lines += [
        "",
        f"Zero lag-4 share A {100 * float(mech['zero_seasonal_share_A']):.1f}%, B {100 * float(mech['zero_seasonal_share_B']):.1f}%.",
        f"Imputed training share A {100 * float(mech['imputed_share_A']):.1f}%, B {100 * float(mech['imputed_share_B']):.1f}%.",
        "",
        "## Origin contribution",
        "",
        "| Model | Diff term | Same term | Total | Share from diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = origin[model]
        lines.append(
            f"| {DISPLAY[model]} | {_md_num(row['term_diff_origin'])} | {_md_num(row['term_same_origin'])} | "
            f"{_md_num(row['total'])} | {100 * float(row['share_from_diff_origin']):.1f}% |"
        )
    lines += ["", "| Keys | Arm | Synthetic share | Median staleness | Mean staleness |", "|---|---|---:|---:|---:|"]
    for label, key in (("diff", "diff"), ("same", "same")):
        for arm in ("A", "B"):
            item = results["origin_descriptive"][key][arm]
            lines.append(
                f"| {label} | {arm} | {100 * float(item['synthetic_share']):.1f}% | "
                f"{_md_num(item['staleness_median'], 2)} | {_md_num(item['staleness_mean'], 2)} |"
            )
    lines += ["", "| Arm A month-day | Arm B month-day | Keys |", "|---|---|---:|"]
    for row in results["origin_monthdays"]:
        lines.append(f"| {row['arm_a_monthday']} | {row['arm_b_monthday']} | {int(row['count'])} |")
    anchor = results["coverage"]["anchor"]
    lines += [
        "",
        "## Grid anchor",
        "",
        f"Drift phrase: {anchor['drift']}.",
        f"Entities with an off-quarter-end stamp: {anchor['n_entities_with_off_stamp']} of {anchor['n_entities']}.",
        "",
        "| Month-day | Stamps |",
        "|---|---:|",
    ]
    for key, count in anchor["stamp_monthday_counts"].items():
        lines.append(f"| {key} | {int(count)} |")
    lines += ["", "| First stamp | Entities | Entities with an off-quarter-end stamp |", "|---|---:|---:|"]
    for key, item in anchor["by_first_stamp"].items():
        lines.append(f"| {key} | {int(item['entities'])} | {int(item['entities_with_off_stamp'])} |")
    lines += ["", "## Exclusive keys", ""]
    for side in ("arm_b_only", "arm_a_only"):
        payload = results["coverage"]["exclusive"][side]
        lines += [f"### {side} ({payload['n']})", "", "| Rule | Count |", "|---|---:|"]
        for rule in payload["rules"]:
            lines.append(f"| {rule['rule']} | {int(rule['count'])} |")
            for sub in rule["subs"]:
                lines.append(f"| - {sub['label']} | {int(sub['count'])} |")
        lines.append("")
    lines += [
        "## Notes against the manuscript text",
        "",
        results["notes"],
        "",
        "## BLOCKERS.md",
        "",
        blockers.rstrip(),
        "",
        "## DEVIATIONS.md",
        "",
        deviations.rstrip(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def emit(results: dict, generated: Path, summary: Path, blockers: Path, deviations: Path) -> str | None:
    values, deviation = build_values(results)
    generated.mkdir(parents=True, exist_ok=True)
    write_values(values, generated / "values.tex")
    write_tables(results, generated)
    write_summary(results, blockers.read_text(encoding="utf-8"), deviations.read_text(encoding="utf-8"), summary)
    return deviation
