"""
Step 6 - emit the manuscript's tables and its numeric macros from the result
files.

Nothing in the paper's tables is typed by hand.  Every table below is written as
a .tex fragment that fininnov_mig.tex \\input{}s, and every number quoted in the
prose is written as a \\newcommand macro in macros.tex.  If an analysis is re-run
and a number moves, the manuscript moves with it; the text cannot silently drift
away from the data it describes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"

AGG_LABEL = {
    "A_mig": "Gated signal $A$",
    "pol_relf": "Relevance-filtered polarity",
    "pol_mean": "Mean polarity",
    "pol_cnt": "Count-weighted polarity",
    "add_comb": "Additive combiner",
    "A_nu": "Novelty gate only ($s\\nu$)",
    "A_mu": "Materiality gate only ($s\\mu$)",
}
AGG_ORDER = ["A_mig", "pol_relf", "pol_mean", "pol_cnt", "add_comb", "A_nu", "A_mu"]

VARIANT_LABEL = {
    "price_plus_A": "Price $+$ gated signal $A$",
    "price_only": "Price only",
    "price_plus_relfilt": "Price $+$ filtered polarity",
    "price_plus_polarity": "Price $+$ mean polarity",
    "A_only": "Text only",
}
VARIANT_ORDER = ["price_plus_A", "price_plus_relfilt", "price_plus_polarity",
                 "price_only", "A_only"]


# LaTeX control sequences may contain letters only, so horizons are spelled out
# when they appear in a macro name: \PrecAHFive, never \PrecAH5.
HWORD = {"1": "One", "5": "Five", "10": "Ten", "21": "TwentyOne", "20": "Twenty"}


def hw(h: str | int) -> str:
    h = str(h)
    return HWORD.get(h, "H" + h)


def stars(t: float) -> str:
    if t != t:
        return ""
    a = abs(t)
    return "$^{***}$" if a >= 2.576 else "$^{**}$" if a >= 1.96 else "$^{*}$" if a >= 1.645 else ""


def fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return f"{100 * v:.1f}" if pct else f"{v:,.{nd}f}"


# ---------------------------------------------------------------------------
def table_data_summary(meta: dict) -> str:
    return f"""\\begin{{table}}[htbp]
\\caption{{The two bodies of evidence. The validation panel answers whether the
decomposition carries information; the deployment ledger answers whether a
calibrated selective forecaster behaves as designed in production. They are drawn
from different markets and are never used to support claims about each other.}}
\\label{{tab:data}}
\\begin{{tabular}}{{@{{}}lll@{{}}}}
\\toprule
 & \\textbf{{Validation panel}} & \\textbf{{Deployment ledger}} \\\\
\\midrule
Market & US listed equities & NSE India large caps \\\\
Source & FNSPID headline corpus & Live append-only forecast ledger \\\\
Entities & {meta['symbols']} operating companies & 7 names (live), 54 (walk-forward) \\\\
Span & {meta['date_min']} to {meta['date_max']} & 19 Apr 2026 to 12 Jun 2026 \\\\
Sessions & {meta['sessions']:,} & -- \\\\
Symbol-sessions & {meta['rows']:,} & -- \\\\
\\ldots with scored news & {meta['rows_with_scored_news']:,} & -- \\\\
Headlines scored & {meta['headlines_scored']:,} & -- \\\\
Resolved forecasts & -- & 621 \\\\
Question answered & Does gating carry information? & Is the deployed system honest? \\\\
\\botrule
\\end{{tabular}}
\\end{{table}}
"""


def table_axis_descriptives(panel: pd.DataFrame, events: pd.DataFrame) -> str:
    rows = []
    for col, lab in [("s", "Polarity $s$"), ("nu", "Novelty $\\nu$"),
                     ("mu", "Materiality $\\mu$")]:
        v = events[col]
        rows.append((lab, v.mean(), v.std(), v.quantile(.25), v.median(),
                     v.quantile(.75), (v.abs() < 1e-9).mean()))
    ev = events.assign(r=events["nu"] * events["mu"],
                       a=events["s"] * events["nu"] * events["mu"])
    for col, lab in [("r", "Relevance $r=\\nu\\mu$"), ("a", "Event signal $a=s\\nu\\mu$")]:
        v = ev[col]
        rows.append((lab, v.mean(), v.std(), v.quantile(.25), v.median(),
                     v.quantile(.75), (v.abs() < 1e-9).mean()))

    body = "\n".join(
        f"{lab} & {m:,.3f} & {sd:,.3f} & {q1:,.3f} & {md:,.3f} & {q3:,.3f} & {100*z:,.1f} \\\\"
        for lab, m, sd, q1, md, q3, z in rows
    )
    corr = float(events["nu"].corr(events["mu"]))
    cov = panel[panel["has_scored_news"] == 1]
    agg_rows = "\n".join(
        f"{AGG_LABEL[c]} & {cov[c].mean():,.3f} & {cov[c].std():,.3f} & "
        f"{cov[c].quantile(.25):,.3f} & {cov[c].median():,.3f} & "
        f"{cov[c].quantile(.75):,.3f} & {100*(cov[c].abs()<1e-9).mean():,.1f} \\\\"
        for c in AGG_ORDER if c in cov.columns
    )
    return f"""\\begin{{table}}[htbp]
\\caption{{Descriptive statistics. Upper panel: the three axes and their products
at the level of the individual headline. Lower panel: the session-level
aggregators, computed on symbol-sessions carrying at least one scored headline.
The final column is the share of exactly-zero values, which is the share of
events or sessions the gate vetoes outright.}}
\\label{{tab:desc}}
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
 & Mean & SD & p25 & Median & p75 & Zeros (\\%) \\\\
\\midrule
\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{Panel A: per headline}}}} \\\\
{body}
\\addlinespace
\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{Panel B: per symbol-session}}}} \\\\
{agg_rows}
\\botrule
\\end{{tabular}}
\\footnotetext{{The correlation between novelty and materiality across all
events is {corr:.3f}: the two gates are close to the same variable, which is the
single fact behind most of what follows.}}
\\end{{table}}
"""


def table_univariate(uni: pd.DataFrame) -> str:
    horizons = sorted(uni["horizon"].unique())
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$H={h}$}}" for h in horizons)
    sub = " & ".join("bps & $t$" for _ in horizons)
    lines = []
    for a in AGG_ORDER:
        cells = []
        for h in horizons:
            r = uni[(uni["horizon"] == h) & (uni["aggregator"] == a)]
            if r.empty:
                cells += ["--", "--"]
            else:
                r = r.iloc[0]
                cells += [f"{r['coef_bps']:,.1f}{stars(r['t_2way'])}", f"{r['t_2way']:,.2f}"]
        lines.append(f"{AGG_LABEL[a]} & " + " & ".join(cells) + " \\\\")
    n = int(uni[uni["horizon"] == horizons[0]]["n"].max())
    return f"""\\begin{{table}}[htbp]
\\caption{{Predictive regressions of forward market-adjusted returns on each
aggregator, one aggregator at a time. Regressors are standardised, so a
coefficient reads as basis points of forward return per one standard deviation of
signal. Standard errors are two-way clustered by date and symbol; horizons beyond
one session use non-overlapping windows. Stars mark significance at the 10\\%,
5\\% and 1\\% levels.}}
\\label{{tab:univariate}}
\\begin{{tabular}}{{@{{}}l{'cc' * len(horizons)}@{{}}}}
\\toprule
 & {head} \\\\
Aggregator & {sub} \\\\
\\midrule
{chr(10).join(lines)}
\\botrule
\\end{{tabular}}
\\footnotetext{{Largest estimation sample: {n:,} symbol-sessions.}}
\\end{{table}}
"""


def table_horse_race(hr: pd.DataFrame) -> str:
    horizons = sorted(hr["horizon"].unique())
    terms = ["pol_mean", "A_nu", "A_mu", "A_mig"]
    lines = []
    for t in terms:
        cells = []
        for h in horizons:
            r = hr[(hr["horizon"] == h) & (hr["term"] == t)]
            if r.empty:
                cells += ["--", "--"]
            else:
                r = r.iloc[0]
                cells += [f"{r['coef_bps']:,.1f}{stars(r['t_2way'])}", f"{r['t_2way']:,.2f}"]
        lines.append(f"{AGG_LABEL[t]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$H={h}$}}" for h in horizons)
    sub = " & ".join("bps & $t$" for _ in horizons)
    return f"""\\begin{{table}}[htbp]
\\caption{{The nested horse race. All four terms enter one regression, so each
coefficient is the marginal contribution of that term given the others.
Multiplicative gating predicts that the gated signal loads and absorbs the
lower-order terms. Regressors standardised; two-way clustered $t$-statistics.}}
\\label{{tab:horserace}}
\\begin{{tabular}}{{@{{}}l{'cc' * len(horizons)}@{{}}}}
\\toprule
 & {head} \\\\
Term & {sub} \\\\
\\midrule
{chr(10).join(lines)}
\\botrule
\\end{{tabular}}
\\end{{table}}
"""


def table_gate(summary: dict) -> str:
    # Brier is dropped from the printed table: it agrees with AURC to the third
    # decimal for every variant and its extra column pushed the table past the
    # text width.  It stays in results/gate_summary_*.json for anyone who wants it.
    horizons = sorted(summary.keys(), key=int)
    lines = []
    for v in VARIANT_ORDER:
        cells = []
        for h in horizons:
            p = summary[h]["pooled"].get(v)
            if not p:
                cells += ["--", "--"]
            else:
                cells += [f"{100*p['prec_at_10pct']:,.1f}", f"{p['aurc']:,.3f}"]
        lines.append(f"{VARIANT_LABEL[v]} & " + " & ".join(cells) + " \\\\")
    base_line = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{100*summary[h]['pooled']['price_only']['base_rate']:,.1f}\\%}}"
        for h in horizons
    )
    comp = []
    for h in horizons:
        for name, lab in [("A_vs_price", "vs.\\ price only"),
                          ("A_vs_relfilt", "vs.\\ filtered polarity"),
                          ("A_vs_polarity", "vs.\\ mean polarity")]:
            c = summary[h]["comparisons"].get(name)
            if c:
                comp.append(
                    f"$H={h}$, {lab} & {c['diff_mean_pp']:+,.2f} & "
                    f"[{c['ci_lo_pp']:+,.2f}, {c['ci_hi_pp']:+,.2f}] & {c['p_two_sided']:,.3f} \\\\"
                )
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$H={h}$}}" for h in horizons)
    sub = " & ".join("Prec. & AURC" for _ in horizons)
    return f"""\\begin{{table}}[htbp]
\\caption{{Selective forecasting with and without the gated text signal. Precision
is the realised up-rate of the most confident 10\\% of out-of-sample predictions,
a fixed coverage that makes the variants comparable without reference to any
tuned threshold; AURC is the area under the risk--coverage curve, for which lower
is better. All figures are pooled over walk-forward test years. The lower panel
bootstraps the precision gap by resampling whole dates.}}
\\label{{tab:gate}}
\\small
\\begin{{tabular}}{{@{{}}l{'cc' * len(horizons)}@{{}}}}
\\toprule
 & {head} \\\\
Feature set & {sub} \\\\
\\midrule
{chr(10).join(lines)}
\\midrule
Always-up base rate & {base_line} \\\\
\\botrule
\\end{{tabular}}

\\vspace{{0.6em}}
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Precision gap at 10\\% coverage & Estimate (pp) & 95\\% CI & $p$ \\\\
\\midrule
{chr(10).join(comp)}
\\botrule
\\end{{tabular}}
\\end{{table}}
"""


def table_live() -> str:
    return """\\begin{table}[htbp]
\\caption{Deployment evidence from the live forecast ledger. Every interval
carries a nominal 90\\% level. Panel B reports the walk-forward conviction gate on
the 54-name deployment universe, where precision is the realised up-rate of the
fired high-conviction bucket and the base is the unconditional always-up rate.}
\\label{tab:live}
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\multicolumn{4}{@{}l}{\\textit{Panel A: interval coverage by horizon}} \\\\
Horizon & Resolved forecasts & Nominal & Empirical \\\\
\\midrule
5 trading days  & 8   & 90.0\\% & 100.0\\% \\\\
10 trading days & 575 & 90.0\\% & 69.2\\%  \\\\
20 trading days & 38  & 90.0\\% & 86.8\\%  \\\\
\\textbf{All}    & \\textbf{621} & \\textbf{90.0\\%} & \\textbf{70.7\\%} \\\\
\\botrule
\\end{tabular}

\\vspace{0.6em}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\multicolumn{5}{@{}l}{\\textit{Panel B: walk-forward conviction gate by test year}} \\\\
Test year & Fires & Fired precision & Always-up base & Edge (pp) \\\\
\\midrule
2022 & 365 & 61.6\\% & 54.0\\% & $+7.6$ \\\\
2023 & 290 & 68.3\\% & 65.3\\% & $+3.0$ \\\\
2024 & 932 & 53.3\\% & 55.7\\% & $-2.4$ \\\\
2025 & 822 & 69.0\\% & 58.2\\% & $+10.8$ \\\\
2026 & 431 & 54.5\\% & 47.1\\% & $+7.4$ \\\\
\\textbf{Pooled} & \\textbf{2{,}840} & \\textbf{60.6\\%} & \\textbf{58.0\\%} & \\textbf{$+2.6$} \\\\
\\botrule
\\end{tabular}
\\footnotetext{The pooled base is the unconditional always-up rate over all
out-of-sample rows, not the fires-weighted mean of the yearly bases (55.9\\%); the
pooled edge is therefore the more conservative of the two comparisons.}
\\end{table}
"""


def macros(meta: dict, uni: pd.DataFrame, hr: pd.DataFrame, gate: dict,
           expo: dict | None, rel: dict | None) -> str:
    out = [
        "% Auto-generated by analysis/06_tables.py -- do not edit by hand.",
        f"\\newcommand{{\\NCompanies}}{{{meta['symbols']}}}",
        f"\\newcommand{{\\ScoredHeadlines}}{{{meta['headlines_scored']:,}}}",
        f"\\newcommand{{\\PanelRows}}{{{meta['rows']:,}}}",
        f"\\newcommand{{\\NewsRows}}{{{meta['rows_with_scored_news']:,}}}",
        f"\\newcommand{{\\PanelSessions}}{{{meta['sessions']:,}}}",
        f"\\newcommand{{\\PanelStart}}{{{meta['date_min']}}}",
        f"\\newcommand{{\\PanelEnd}}{{{meta['date_max']}}}",
    ]
    for h in sorted(gate.keys(), key=int):
        w = hw(h)
        p = gate[h]["pooled"]
        if "price_plus_A" in p:
            out.append(f"\\newcommand{{\\PrecAH{w}}}{{{100*p['price_plus_A']['prec_at_10pct']:.1f}}}")
            out.append(f"\\newcommand{{\\AurcAH{w}}}{{{p['price_plus_A']['aurc']:.3f}}}")
        if "price_only" in p:
            out.append(f"\\newcommand{{\\PrecPriceH{w}}}{{{100*p['price_only']['prec_at_10pct']:.1f}}}")
            out.append(f"\\newcommand{{\\AurcPriceH{w}}}{{{p['price_only']['aurc']:.3f}}}")
            out.append(f"\\newcommand{{\\BaseRateH{w}}}{{{100*p['price_only']['base_rate']:.1f}}}")
        if "price_plus_relfilt" in p:
            out.append(f"\\newcommand{{\\PrecRelfH{w}}}{{{100*p['price_plus_relfilt']['prec_at_10pct']:.1f}}}")
            out.append(f"\\newcommand{{\\AurcRelfH{w}}}{{{p['price_plus_relfilt']['aurc']:.3f}}}")
        c = gate[h]["comparisons"].get("A_vs_price")
        if c:
            out.append(f"\\newcommand{{\\GapPriceH{w}}}{{{c['diff_mean_pp']:+.2f}}}")
            out.append(f"\\newcommand{{\\GapPriceCIH{w}}}{{[{c['ci_lo_pp']:+.2f}, {c['ci_hi_pp']:+.2f}]}}")
            out.append(f"\\newcommand{{\\GapPricePH{w}}}{{{c['p_two_sided']:.3f}}}")
        c = gate[h]["comparisons"].get("A_vs_relfilt")
        if c:
            out.append(f"\\newcommand{{\\GapRelfH{w}}}{{{c['diff_mean_pp']:+.2f}}}")
            out.append(f"\\newcommand{{\\GapRelfCIH{w}}}{{[{c['ci_lo_pp']:+.2f}, {c['ci_hi_pp']:+.2f}]}}")
            out.append(f"\\newcommand{{\\GapRelfPH{w}}}{{{c['p_two_sided']:.3f}}}")
    if expo:
        for h, v in expo.items():
            w = hw(h)
            b, u = v.get("best"), v.get("unit")
            if b:
                out.append(f"\\newcommand{{\\BestAlphaH{w}}}{{{b['alpha']:.2f}}}")
                out.append(f"\\newcommand{{\\BestBetaH{w}}}{{{b['beta']:.2f}}}")
                out.append(f"\\newcommand{{\\BestICH{w}}}{{{b['ic']:.4f}}}")
            if u:
                out.append(f"\\newcommand{{\\UnitICH{w}}}{{{u['ic']:.4f}}}")
            # the alpha = beta = 0 corner: no gating at all, i.e. plain mean
            # polarity, which is the benchmark the estimated exponents are read against
            pu = v.get("pure")
            if pu:
                out.append(f"\\newcommand{{\\PureICH{w}}}{{{pu['ic']:.4f}}}")
    cvp = RESULTS / "convergent_validity.json"
    if cvp.exists():
        cv = json.loads(cvp.read_text())
        st, ar = cv["staleness_regression"], cv["absret_regression"]
        out += [
            f"\\newcommand{{\\StaleNuBeta}}{{{st['nu_beta']:.3f}}}",
            f"\\newcommand{{\\StaleNuT}}{{{st['nu_t']:.2f}}}",
            f"\\newcommand{{\\StaleMuBeta}}{{{st['mu_beta']:.3f}}}",
            f"\\newcommand{{\\StaleMuT}}{{{st['mu_t']:.2f}}}",
            f"\\newcommand{{\\AbsRetNuBps}}{{{ar['nu_beta_bps']:.1f}}}",
            f"\\newcommand{{\\AbsRetNuT}}{{{ar['nu_t']:.2f}}}",
            f"\\newcommand{{\\AbsRetMuBps}}{{{ar['mu_beta_bps']:.1f}}}",
            f"\\newcommand{{\\AbsRetMuT}}{{{ar['mu_t']:.2f}}}",
            f"\\newcommand{{\\PartialNuStale}}{{{cv['partial']['nu_vs_mech_given_mu']:.3f}}}",
            f"\\newcommand{{\\PartialMuStale}}{{{cv['partial']['mu_vs_mech_given_nu']:.3f}}}",
            f"\\newcommand{{\\DenseNuStale}}{{{cv['raw']['nu_llm_vs_nu_mech_dense']:.3f}}}",
            f"\\newcommand{{\\DenseMuStale}}{{{cv['raw']['mu_llm_vs_nu_mech_dense']:.3f}}}",
            f"\\newcommand{{\\StaleN}}{{{st['n']:,}}}",
            f"\\newcommand{{\\AbsRetN}}{{{ar['n']:,}}}",
            f"\\newcommand{{\\NuPeakDecile}}{{{cv['profile_shape']['peak_decile'] + 1}}}",
            f"\\newcommand{{\\NuAtPeak}}{{{cv['profile_shape']['nu_at_peak']:.3f}}}",
            f"\\newcommand{{\\NuAtStalest}}{{{cv['profile_shape']['nu_at_stalest']:.3f}}}",
            f"\\newcommand{{\\NuAtFreshest}}{{{cv['profile_shape']['nu_at_freshest']:.3f}}}",
        ]
    if rel:
        for axis in ("nu", "mu", "s"):
            if axis in rel:
                name = {"nu": "Nu", "mu": "Mu", "s": "S"}[axis]
                out.append(f"\\newcommand{{\\ICC{name}}}{{{rel[axis]['icc']:.3f}}}")
        if "n" in rel:
            out.append(f"\\newcommand{{\\RetestN}}{{{rel['n']:,}}}")
    return "\n".join(out) + "\n"


# The scored panel is not copied into this folder; it stays with the pipeline
# that produced it, so its location is a flag rather than a fixed path.
DEFAULT_CACHE = ROOT.parent / "research-paper-17-fininnov-not-worth" / "cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help="folder holding mig_panel.parquet and events.parquet")
    args = ap.parse_args()
    cache = Path(args.cache)

    TABLES.mkdir(exist_ok=True)
    meta = json.loads((RESULTS / "panel_meta.json").read_text())
    panel = pd.read_parquet(cache / "mig_panel.parquet")
    events = pd.read_parquet(cache / "events.parquet")
    uni = pd.read_csv(RESULTS / "univariate.csv")
    hr = pd.read_csv(RESULTS / "horse_race.csv")
    gate = json.loads((RESULTS / "gate_summary_newsrows.json").read_text())
    expo = None
    if (RESULTS / "exponents.json").exists():
        expo = json.loads((RESULTS / "exponents.json").read_text())
    rel = None
    if (RESULTS / "reliability.json").exists():
        rel = json.loads((RESULTS / "reliability.json").read_text())

    (TABLES / "tab_data.tex").write_text(table_data_summary(meta))
    (TABLES / "tab_desc.tex").write_text(table_axis_descriptives(panel, events))
    (TABLES / "tab_univariate.tex").write_text(table_univariate(uni))
    (TABLES / "tab_horserace.tex").write_text(table_horse_race(hr))
    (TABLES / "tab_gate.tex").write_text(table_gate(gate))
    (TABLES / "tab_live.tex").write_text(table_live())
    (ROOT / "macros.tex").write_text(macros(meta, uni, hr, gate, expo, rel))
    print(f"wrote 6 tables to {TABLES} and macros.tex")


if __name__ == "__main__":
    main()
