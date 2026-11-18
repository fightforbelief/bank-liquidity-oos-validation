"""Generate paper_v3/generated/{confirm_values.tex, tab_confirm.tex, tab_confirm_coverage.tex}
from the analysis.json files (three confirmatory runs + the discovery run).

Usage: python confirm/make_tex.py <dir containing discovery_analysis.json and <target>_analysis.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "paper_v3" / "generated"
NAMES = {"discovery": "Bank Cash Liquidity", "bank_leverage": "Bank Leverage",
         "bhc_leverage": "BHC Leverage", "bhc_cash_liquidity": "BHC Cash Liquidity"}
ORDER = ("discovery", "bank_leverage", "bhc_leverage", "bhc_cash_liquidity")


def ci(r, d=3):
    return f"${r['est']:.{d}f}$ $[{r['lo']:.{d}f},\\ {r['hi']:.{d}f}]$"


def main() -> int:
    src = Path(sys.argv[1])
    res = {}
    for k in ORDER:
        p = src / ("discovery_analysis.json" if k == "discovery" else f"{k}_analysis.json")
        if p.exists():
            res[k] = json.loads(p.read_text(encoding="utf-8"))
    GEN.mkdir(parents=True, exist_ok=True)

    cov = []
    for k, r in res.items():
        h1, h2 = r["H1"], r["H2"]
        nb = r["descriptive"]["paired"]
        tot_b = h1["arm_b_only_keys"]
        cov.append(" & ".join([
            NAMES[k] + ("$^{\\dagger}$" if k == "discovery" else ""),
            f"{h1['entities']['A']:,}", f"{h1['entities']['B']:,}", f"{tot_b:,}",
            f"{100 * h1['share']:.1f}" if h1["share"] is not None else "--",
            f"{100 * h2['entrant_share_A']:.1f}", f"{100 * h2['entrant_share_B']:.1f}",
            f"{100 * h2['share_of_entrants_absent_from_A']:.1f}",
            f"{100 * min(-v['rel_change'] for v in nb.values()):.1f}--{100 * max(-v['rel_change'] for v in nb.values()):.1f}",
        ]) + " \\\\")
    (GEN / "tab_confirm_coverage.tex").write_text(
        "\\fittable{%\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Entities} & B-only & Day-30 & \\multicolumn{2}{c}{Entrant share (\\%)} & Entrants & $|$Est.$|$/level \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}\n"
        "Dataset & Arm A & Arm B & targets & first (\\%) & Arm A & Arm B & lost (\\%) & (\\%) \\\\\n\\midrule\n"
        + "\n".join(cov) + "\n\\bottomrule\n\\end{tabular}}\n", encoding="utf-8")

    rows = []
    for k, r in res.items():
        h3 = r["H3"]
        t, a = h3["naive_minus_theta"], h3["naive_minus_arima"]
        verdict = "--" if k == "discovery" else ("confirmed" if h3["confirmed"] else "not confirmed")
        rows.append(" & ".join([
            NAMES[k] + ("$^{\\dagger}$" if k == "discovery" else ""),
            f"{h3['n_entrants']:,} / {h3['n_incumbents']:,}",
            ci(t["entrants"]), ci(t["incumbents"]), ci(t["contrast"]), ci(a["contrast"]), verdict,
        ]) + " \\\\")
    (GEN / "tab_confirm.tex").write_text(
        "\\fittable{%\n\\begin{tabular}{lrrrrrl}\n\\toprule\n"
        " & Entrants / & \\multicolumn{2}{c}{Naive $-$ AutoTheta} & \\multicolumn{2}{c}{Contrast $D$ (entrants $-$ incumbents)} & H3 \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Dataset & incumbents & Entrants & Incumbents & vs AutoTheta & vs AutoARIMA & \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n", encoding="utf-8")

    conf = [k for k in ORDER[1:] if k in res]
    n_ok = sum(res[k]["H3"]["confirmed"] for k in conf)
    h1_ok = sum(res[k]["H1"]["confirmed"] for k in conf)
    h2_ok = sum(res[k]["H2"]["confirmed"] for k in conf)
    vals = {
        "confirm-n": str(len(conf)),
        "confirm-h1": f"{h1_ok} of {len(conf)}",
        "confirm-h2": f"{h2_ok} of {len(conf)}",
        "confirm-h3": f"{n_ok} of {len(conf)}",
        "confirm-overall": "confirmed" if n_ok >= 2 else "not confirmed",
    }
    fails = [k for k in conf if not res[k]["H3"]["confirmed"]]
    oks = [k for k in conf if res[k]["H3"]["confirmed"]]
    parts = []
    if oks:
        parts.append("The contrast was negative with an interval below zero for "
                     + " and ".join(f"{NAMES[k]} ({ci(res[k]['H3']['naive_minus_theta']['contrast'])})" for k in oks) + ".")
    for k in fails:
        c = res[k]["H3"]["naive_minus_theta"]["contrast"]
        side = "the opposite sign to the prediction" if c["est"] > 0 else "the predicted sign"
        parts.append(f"For {NAMES[k]} the contrast was {ci(c)}, with a point estimate of {side} and an interval "
                     "that includes zero, so H3 is not confirmed there.")
    h1_fail = [NAMES[k] for k in conf if not res[k]["H1"]["confirmed"]]
    if h1_fail:
        parts.append("H1 failed for " + ", ".join(h1_fail) + ".")
    vals["confirm-text"] = " ".join(parts)
    if "discovery" in res:
        c = res["discovery"]["H3"]["naive_minus_theta"]["contrast"]
        vals["disc-h3-contrast"] = ci(c)
    (GEN / "confirm_values.tex").write_text("".join(f"\\setval{{{k}}}{{{v}}}\n" for k, v in vals.items()),
                                            encoding="utf-8")
    print(json.dumps(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
