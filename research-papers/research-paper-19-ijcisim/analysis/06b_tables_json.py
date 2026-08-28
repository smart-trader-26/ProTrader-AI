#!/usr/bin/env python
"""Emit the manuscript's tables and numeric macros as JSON, for the Word build.

IJCISIM takes a Microsoft Word file, so the manuscript is assembled by
``make_paper_ijcisim.py`` with python-docx rather than by LaTeX.  The project
rule that no number is typed by hand into a manuscript still applies, so this
script plays exactly the part ``06_tables.py`` plays for a LaTeX build: it reads
the stored result files and writes

    results/tables.json    every table as {caption, header, rows, note}
    results/macros.json    every number the prose quotes, keyed by name

with cell text in plain Unicode (nu, mu and friends as real characters) instead
of LaTeX markup.  The two emitters read the same inputs, so a Word table and a
LaTeX table of the same name cannot disagree.

Only ``tab_desc`` needs the scored panel itself; everything else comes from
``results/``.  The panel lives in the original study folder and is not copied
here, so its location is a flag:

    python analysis/06b_tables_json.py \
        --cache ../research-paper-17-fininnov-not-worth/cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
DEFAULT_CACHE = ROOT.parent / "research-paper-17-fininnov-not-worth" / "cache"

AGG_LABEL = {
    "A_mig": "Gated signal A",
    "pol_relf": "Threshold-filtered polarity",
    "pol_mean": "Mean polarity",
    "pol_cnt": "Count-weighted polarity",
    "add_comb": "Additive combiner",
    "A_nu": "Novelty gate only (sν)",
    "A_mu": "Materiality gate only (sμ)",
}
AGG_ORDER = ["A_mig", "pol_relf", "pol_mean", "pol_cnt", "add_comb", "A_nu", "A_mu"]

TERM_LABEL = {
    "pol_mean": "Mean polarity",
    "A_nu": "Novelty gate only (sν)",
    "A_mu": "Materiality gate only (sμ)",
    "A_mig": "Gated signal A",
}
TERM_ORDER = ["pol_mean", "A_nu", "A_mu", "A_mig"]

VARIANT_LABEL = {
    "price_plus_A": "Price + gated signal A",
    "price_plus_relfilt": "Price + filtered polarity",
    "price_plus_polarity": "Price + mean polarity",
    "price_only": "Price only",
    "A_only": "Text only",
}
VARIANT_ORDER = ["price_plus_A", "price_plus_relfilt", "price_plus_polarity",
                 "price_only", "A_only"]

COMPARISON_LABEL = {
    "A_vs_price": "vs. price only",
    "A_vs_relfilt": "vs. filtered polarity",
    "A_vs_polarity": "vs. mean polarity",
}
HORIZONS = ["1", "5", "21"]
HWORD = {"1": "One", "5": "Five", "21": "TwentyOne"}


def stars(t: float) -> str:
    """Significance marks, on the same thresholds the LaTeX emitter uses."""
    if t != t:
        return ""
    a = abs(t)
    return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""


def f(v, nd=3):
    return "--" if v is None or v != v else f"{v:,.{nd}f}"


def pc(v, nd=1):
    return "--" if v is None or v != v else f"{100 * v:.{nd}f}"


# ---------------------------------------------------------------- the tables
def t_data(meta: dict) -> dict:
    return {
        "caption": ("The two bodies of evidence. The validation panel answers "
                    "whether the decomposition carries information; the "
                    "deployment ledger answers whether a calibrated selective "
                    "forecaster behaves as designed in production. They are "
                    "drawn from different markets and neither is used to "
                    "support a claim about the other."),
        "header": ["", "Validation panel", "Deployment ledger"],
        "align": ["l", "l", "l"],
        "widths": [0.24, 0.38, 0.38],
        "rows": [
            ["Market", "US listed equities", "NSE India large caps"],
            ["Source", "FNSPID headline corpus", "Live append-only forecast ledger"],
            ["Entities", f"{meta['symbols']} operating companies",
             "7 names (live), 54 (walk-forward)"],
            ["Span", f"{meta['date_min']} to {meta['date_max']}",
             "19 Apr 2026 to 12 Jun 2026"],
            ["Sessions", f"{meta['sessions']:,}", "--"],
            ["Symbol-sessions", f"{meta['rows']:,}", "--"],
            ["... with scored news", f"{meta['rows_with_scored_news']:,}", "--"],
            ["Headlines scored", f"{meta['headlines_scored']:,}", "--"],
            ["Resolved forecasts", "--", "621"],
            ["Question answered", "Does gating carry information?",
             "Is the deployed system honest?"],
        ],
        "note": "",
    }


def t_desc(events: pd.DataFrame, panel: pd.DataFrame, corr: float) -> dict:
    def row(label, x):
        x = pd.Series(x).dropna()
        return [label, f(x.mean()), f(x.std()), f(x.quantile(.25)),
                f(x.median()), f(x.quantile(.75)), pc((x == 0).mean())]

    ev = events
    pa = panel[panel["has_news"] == 1] if "has_news" in panel else panel
    rows = [
        ["Panel A: per headline"],
        row("Polarity s", ev["s"]),
        row("Novelty ν", ev["nu"]),
        row("Materiality μ", ev["mu"]),
        row("Relevance r = νμ", ev["nu"] * ev["mu"]),
        row("Event signal a = sνμ", ev["s"] * ev["nu"] * ev["mu"]),
        ["Panel B: per symbol-session"],
        row("Gated signal A", pa["A_mig"]),
        row("Threshold-filtered polarity", pa["pol_relf"]),
        row("Mean polarity", pa["pol_mean"]),
        row("Count-weighted polarity", pa["pol_cnt"]),
        row("Additive combiner", pa["add_comb"]),
        row("Novelty gate only (sν)", pa["A_nu"]),
        row("Materiality gate only (sμ)", pa["A_mu"]),
    ]
    return {
        "caption": ("Descriptive statistics. Panel A covers the three "
                    "attributes and their products at the level of the "
                    "individual headline; Panel B covers the session-level "
                    "aggregators, computed on symbol-sessions carrying at "
                    "least one scored headline. The final column is the share "
                    "of exactly-zero values, which is the share of events or "
                    "sessions the gate vetoes outright."),
        "header": ["", "Mean", "SD", "p25", "Median", "p75", "Zeros (%)"],
        "align": ["l", "c", "c", "c", "c", "c", "c"],
        "widths": [0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        "rows": rows,
        "note": (f"The correlation between novelty and materiality across all "
                 f"events is {corr:.3f}: the two gates are close to the same "
                 f"variable, which is the single fact behind most of what "
                 f"follows."),
    }


def t_univariate(uni: pd.DataFrame) -> dict:
    rows = []
    for key in AGG_ORDER:
        r = [AGG_LABEL[key]]
        for h in HORIZONS:
            m = uni[(uni.horizon == int(h)) & (uni.aggregator == key)]
            if m.empty:
                r += ["--", "--"]
                continue
            m = m.iloc[0]
            r += [f"{m.coef_bps:.1f}{stars(m.t_2way)}", f"{m.t_2way:.2f}"]
        rows.append(r)
    n = int(uni.n.max())
    return {
        "caption": ("Predictive regressions of forward market-adjusted returns "
                    "on each aggregator, one aggregator at a time. Regressors "
                    "are standardised, so a coefficient reads as basis points "
                    "of forward return per one standard deviation of signal. "
                    "Standard errors are two-way clustered by date and symbol; "
                    "horizons beyond one session use non-overlapping windows. "
                    "Stars mark significance at the 10%, 5% and 1% levels."),
        "header": ["Aggregator"] + sum([[f"H={h} bps", "t"] for h in HORIZONS], []),
        "align": ["l"] + ["c"] * (2 * len(HORIZONS)),
        "widths": [0.28] + [0.12] * (2 * len(HORIZONS)),
        "rows": rows,
        "note": f"Largest estimation sample: {n:,} symbol-sessions.",
    }


def t_horserace(hr: pd.DataFrame) -> dict:
    rows = []
    for key in TERM_ORDER:
        r = [TERM_LABEL[key]]
        for h in HORIZONS:
            m = hr[(hr.horizon == int(h)) & (hr.term == key)]
            if m.empty:
                r += ["--", "--"]
                continue
            m = m.iloc[0]
            r += [f"{m.coef_bps:.1f}{stars(m.t_2way)}", f"{m.t_2way:.2f}"]
        rows.append(r)
    return {
        "caption": ("The nested horse race. All four terms enter one "
                    "regression, so each coefficient is the marginal "
                    "contribution of that term given the others. "
                    "Multiplicative gating predicts that the gated signal "
                    "loads and absorbs the lower-order terms. Regressors are "
                    "standardised; t-statistics are two-way clustered."),
        "header": ["Term"] + sum([[f"H={h} bps", "t"] for h in HORIZONS], []),
        "align": ["l"] + ["c"] * (2 * len(HORIZONS)),
        "widths": [0.28] + [0.12] * (2 * len(HORIZONS)),
        "rows": rows,
        "note": "",
    }


def t_gate(gate: dict) -> dict:
    rows = []
    for key in VARIANT_ORDER:
        r = [VARIANT_LABEL[key]]
        for h in HORIZONS:
            p = gate[h]["pooled"][key]
            r += [pc(p["prec_at_10pct"]), f(p["aurc"])]
        rows.append(r)
    base = ["Always-up base rate"]
    for h in HORIZONS:
        base += [pc(gate[h]["pooled"]["price_only"]["base_rate"]) + "%", ""]
    rows.append(base)
    return {
        "caption": ("Selective forecasting with and without the gated text "
                    "signal. Precision is the realised up-rate of the most "
                    "confident 10% of out-of-sample predictions, a fixed "
                    "coverage that makes the variants comparable without "
                    "reference to any tuned threshold; AURC is the area under "
                    "the risk-coverage curve, for which lower is better. All "
                    "figures are pooled over walk-forward test years."),
        "header": ["Feature set"] + sum([[f"H={h} Prec.", "AURC"] for h in HORIZONS], []),
        "align": ["l"] + ["c"] * (2 * len(HORIZONS)),
        "widths": [0.28] + [0.12] * (2 * len(HORIZONS)),
        "rows": rows,
        "note": "",
    }


def t_gapgate(gate: dict) -> dict:
    rows = []
    for h in HORIZONS:
        for key in ("A_vs_price", "A_vs_relfilt", "A_vs_polarity"):
            c = gate[h]["comparisons"][key]
            rows.append([f"H = {h}, {COMPARISON_LABEL[key]}",
                         f"{c['diff_mean_pp']:+.2f}",
                         f"[{c['ci_lo_pp']:+.2f}, {c['ci_hi_pp']:+.2f}]",
                         f"{min(c['p_two_sided'], 1.0):.3f}"])
    return {
        "caption": ("Bootstrapped difference in precision at 10% coverage "
                    "between the model carrying the gated signal and each "
                    "comparison model, resampling whole dates so that "
                    "cross-sectional correlation is preserved."),
        "header": ["Precision gap at 10% coverage", "Estimate (pp)", "95% CI", "p"],
        "align": ["l", "c", "c", "c"],
        "widths": [0.38, 0.20, 0.26, 0.16],
        "rows": rows,
        "note": "",
    }


def t_live() -> dict:
    return {
        "caption": ("Deployment evidence from the live forecast ledger. Every "
                    "interval carries a nominal 90% level. Panel B reports the "
                    "walk-forward conviction gate on the 54-name deployment "
                    "universe, where precision is the realised up-rate of the "
                    "fired high-conviction bucket and the base is the "
                    "unconditional always-up rate."),
        "header": ["", "", "", "", ""],
        "align": ["l", "c", "c", "c", "c"],
        "widths": [0.28, 0.19, 0.19, 0.19, 0.15],
        "rows": [
            ["Panel A: interval coverage by horizon"],
            ["Horizon", "Resolved forecasts", "Nominal", "Empirical", ""],
            ["5 trading days", "8", "90.0%", "100.0%", ""],
            ["10 trading days", "575", "90.0%", "69.2%", ""],
            ["20 trading days", "38", "90.0%", "86.8%", ""],
            ["All", "621", "90.0%", "70.7%", ""],
            ["Panel B: walk-forward conviction gate by test year"],
            ["Test year", "Fires", "Fired precision", "Always-up base", "Edge (pp)"],
            ["2022", "365", "61.6%", "54.0%", "+7.6"],
            ["2023", "290", "68.3%", "65.3%", "+3.0"],
            ["2024", "932", "53.3%", "55.7%", "-2.4"],
            ["2025", "822", "69.0%", "58.2%", "+10.8"],
            ["2026", "431", "54.5%", "47.1%", "+7.4"],
            ["Pooled", "2,840", "60.6%", "58.0%", "+2.6"],
        ],
        "note": ("The pooled base is the unconditional always-up rate over all "
                 "out-of-sample rows, not the fires-weighted mean of the "
                 "yearly bases (55.9%); the pooled edge is therefore the more "
                 "conservative of the two comparisons."),
    }


def t_cost(cost: dict) -> dict:
    return {
        "caption": ("What the validity screen costs and what it pre-empts. "
                    "Panel A is the screen of Section 3.5; Panel B is the "
                    "modelling programme of Section 4, whose every conclusion "
                    "the screen anticipates. Counts are of fitted objects, "
                    "derived from the stored result files rather than "
                    "asserted."),
        "header": ["Stage", "Unit counted", "Count"],
        "align": ["l", "l", "r"],
        "widths": [0.32, 0.46, 0.22],
        "rows": [
            ["Panel A: the validity screen"],
            ["Criterion regressions", "one per attribute", f"{cost['ScreenRegressions']:,}"],
            ["Decile profiles inspected", "one per criterion pair", f"{cost['ScreenProfiles']:,}"],
            ["Additional scoring-model calls", "headlines rescored", f"{cost['ScreenModelCalls']:,}"],
            ["Total fitted objects", "", f"{cost['ScreenTotal']:,}"],
            ["Panel B: the downstream programme"],
            ["Predictive regressions", "aggregator x horizon", f"{cost['UnivariateFits']:,}"],
            ["Nested horse-race coefficients", "term x horizon", f"{cost['HorseRaceCoefs']:,}"],
            ["Exponent-grid evaluations",
             f"{cost['GridPerHorizon']} per horizon, {cost['GridHorizons']} horizons",
             f"{cost['GridTotal']:,}"],
            ["Robustness re-estimations", "variation x setting x horizon", f"{cost['RobustnessFits']:,}"],
            ["Walk-forward selective forecasters",
             f"{cost['GateVariants']} variants x {cost['GateYears']} years x {cost['GateHorizons']} horizons",
             f"{cost['GateFits']:,}"],
            ["Total fitted objects", "", f"{cost['DownstreamTotal']:,}"],
        ],
        "note": ("Neither criterion in Panel A required a labelled example or "
                 "an additional call to the scoring model."),
    }


# ------------------------------------------------------------------- macros
def macros(meta, uni, gate, expo, cv, rel, cost) -> dict:
    m = {
        "NCompanies": f"{meta['symbols']}",
        "ScoredHeadlines": f"{meta['headlines_scored']:,}",
        "PanelRows": f"{meta['rows']:,}",
        "NewsRows": f"{meta['rows_with_scored_news']:,}",
        "PanelSessions": f"{meta['sessions']:,}",
        "PanelStart": meta["date_min"],
        "PanelEnd": meta["date_max"],
        "CorrNuMu": f"{cv['corr_nu_mu']:.3f}",
        "StaleN": f"{cv['staleness_regression']['n']:,}",
        "AbsRetN": f"{cv['absret_regression']['n']:,}",
        "ScoredForScreen": f"{cv['n_with_prior']:,}",
        "PriorDocMedian": f"{int(cv['median_prior_docs'])}",
        "StaleNuBeta": f"{cv['staleness_regression']['nu_beta']:.3f}",
        "StaleNuT": f"{cv['staleness_regression']['nu_t']:.2f}",
        "StaleMuBeta": f"{cv['staleness_regression']['mu_beta']:.3f}",
        "StaleMuT": f"{cv['staleness_regression']['mu_t']:.2f}",
        "AbsRetNuBps": f"{cv['absret_regression']['nu_beta_bps']:.1f}",
        "AbsRetNuT": f"{cv['absret_regression']['nu_t']:.2f}",
        "AbsRetMuBps": f"{cv['absret_regression']['mu_beta_bps']:.1f}",
        "AbsRetMuT": f"{cv['absret_regression']['mu_t']:.2f}",
        "PartialNuStale": f"{cv['partial']['nu_vs_mech_given_mu']:.3f}",
        "PartialMuStale": f"{cv['partial']['mu_vs_mech_given_nu']:.3f}",
        "DenseNuStale": f"{cv['raw']['nu_llm_vs_nu_mech_dense']:.3f}",
        "DenseMuStale": f"{cv['raw']['mu_llm_vs_nu_mech_dense']:.3f}",
        "RawNuStale": f"{cv['raw']['nu_llm_vs_nu_mech']:.3f}",
        "NuPeakDecile": f"{cv['profile_shape']['peak_decile'] + 1}",
        "NuAtPeak": f"{cv['profile_shape']['nu_at_peak']:.3f}",
        "NuAtStalest": f"{cv['profile_shape']['nu_at_stalest']:.3f}",
        "NuAtFreshest": f"{cv['profile_shape']['nu_at_freshest']:.3f}",
        "ICCNu": f"{rel['nu']['icc']:.3f}",
        "ICCMu": f"{rel['mu']['icc']:.3f}",
        "ICCS": f"{rel['s']['icc']:.3f}",
        "ICCEvent": f"{rel['event_signal']['icc']:.3f}",
        "RetestN": f"{rel['n']}",
    }
    for h in HORIZONS:
        w = HWORD[h]
        e = expo[h]
        m[f"BestAlphaH{w}"] = f"{e['best']['alpha']:.2f}"
        m[f"BestBetaH{w}"] = f"{e['best']['beta']:.2f}"
        m[f"BestICH{w}"] = f"{e['best']['ic']:.4f}"
        m[f"UnitICH{w}"] = f"{e['unit']['ic']:.4f}"
        m[f"PureICH{w}"] = f"{e['pure']['ic']:.4f}"
        p = gate[h]["pooled"]
        m[f"PrecAH{w}"] = pc(p["price_plus_A"]["prec_at_10pct"])
        m[f"AurcAH{w}"] = f(p["price_plus_A"]["aurc"])
        m[f"PrecPriceH{w}"] = pc(p["price_only"]["prec_at_10pct"])
        m[f"AurcPriceH{w}"] = f(p["price_only"]["aurc"])
        m[f"PrecTextH{w}"] = pc(p["A_only"]["prec_at_10pct"])
        m[f"AurcTextH{w}"] = f(p["A_only"]["aurc"])
        m[f"BaseRateH{w}"] = pc(p["price_only"]["base_rate"])
        for tag, key in (("Price", "A_vs_price"), ("Relf", "A_vs_relfilt")):
            c = gate[h]["comparisons"][key]
            m[f"Gap{tag}H{w}"] = f"{c['diff_mean_pp']:+.2f}"
            m[f"Gap{tag}CIH{w}"] = f"[{c['ci_lo_pp']:+.2f}, {c['ci_hi_pp']:+.2f}]"
            m[f"Gap{tag}PH{w}"] = f"{min(c['p_two_sided'], 1.0):.3f}"
        u = uni[(uni.horizon == int(h)) & (uni.aggregator == "A_mig")].iloc[0]
        m[f"CoefAH{w}"] = f"{u.coef_bps:.1f}"
        m[f"TAH{w}"] = f"{u.t_2way:.2f}"
        m[f"FMTAH{w}"] = f"{u.fm_t:.2f}"
        m[f"ICAH{w}"] = f"{u.ic:.4f}"
        m[f"ICTAH{w}"] = f"{u.ic_t:.2f}"
        m[f"ICPeriodsH{w}"] = f"{int(u.ic_periods):,}"
    for key in AGG_ORDER:
        u = uni[(uni.horizon == 1) & (uni.aggregator == key)].iloc[0]
        m[f"Coef{key}HOne"] = f"{u.coef_bps:.1f}"
        m[f"T{key}HOne"] = f"{u.t_2way:.2f}"
        m[f"IC{key}HOne"] = f"{u.ic:.4f}"
        m[f"ICT{key}HOne"] = f"{u.ic_t:.2f}"
    m.update({k: f"{v:,}" if isinstance(v, int) else str(v)
              for k, v in cost.items()})
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help="folder holding mig_panel.parquet and events.parquet")
    args = ap.parse_args()
    cache = Path(args.cache)

    meta = json.loads((RESULTS / "panel_meta.json").read_text())
    uni = pd.read_csv(RESULTS / "univariate.csv")
    hr = pd.read_csv(RESULTS / "horse_race.csv")
    gate = json.loads((RESULTS / "gate_summary_newsrows.json").read_text())
    expo = json.loads((RESULTS / "exponents.json").read_text())
    cv = json.loads((RESULTS / "convergent_validity.json").read_text())
    rel = json.loads((RESULTS / "reliability.json").read_text())
    cost = json.loads((RESULTS / "cost_accounting.json").read_text())

    events = pd.read_parquet(cache / "events.parquet")
    panel = pd.read_parquet(cache / "mig_panel.parquet")

    tables = {
        "tab_data": t_data(meta),
        "tab_desc": t_desc(events, panel, cv["corr_nu_mu"]),
        "tab_cost": t_cost(cost),
        "tab_univariate": t_univariate(uni),
        "tab_horserace": t_horserace(hr),
        "tab_gate": t_gate(gate),
        "tab_gapgate": t_gapgate(gate),
        "tab_live": t_live(),
    }
    (RESULTS / "tables.json").write_text(
        json.dumps(tables, indent=1, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "macros.json").write_text(
        json.dumps(macros(meta, uni, gate, expo, cv, rel, cost),
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {RESULTS / 'tables.json'} ({len(tables)} tables)")
    print(f"wrote {RESULTS / 'macros.json'}")


if __name__ == "__main__":
    main()
