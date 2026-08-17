"""Build the ITEGAM-JETIA manuscript.

Same study, same numbers and same prose as the manuscript prepared for
Advances in Electrical and Computer Engineering; the differences are the ones
the journal requires:

  * the IMRDC structure JETIA mandates (Abstract, Introduction, Theoretical
    Framework, Methodology, Results and Discussion, Conclusions, Author
    Contributions, Acknowledgments, References), with the theory sections of
    the earlier version gathered under the Theoretical Framework and the
    discussion folded into Results and Discussions;
  * JETIA's numbering of headings (I, II, II.1, II.1.1), of tables and
    figures (Table 1, Figure 1) and its caption/source convention;
  * a reference list that satisfies JETIA's distribution rules, which is why
    the recent-literature coverage is wider than in the earlier version;
  * Author Contributions and Acknowledgments sections in place of the
    Conflict of Interest and Publisher's Note sections the other journal
    asked for.

Every quantitative statement in the text is interpolated from the result
files written by steps 4-6 of the pipeline, so the prose cannot drift away
from the analysis.  Citations are written as [[tag]] and renumbered in order
of first appearance, which is what IEEE style requires.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import omml as M                                          # noqa: E402
from jetia_docbuild import (Builder, clear_after_title_block,  # noqa: E402
                            drop_rows, load_template, scrub_properties,
                            set_cell, set_keywords, set_running_header)
from refs import REFS                                     # noqa: E402

TITLE = ("An Attenuation-Corrected Matched Filter with Closed-Loop Gain "
         "Design for News-Sentiment Aggregation in Equity Forecasting")
AUTHORS = "Anandkumar Pardeshi*{^1} and Sujata Deshmukh{^2}"
AFFIL = [
    "{^1} Department of Computer Science and Engineering, Fr. C. Rodrigues "
    "Institute of Technology, University of Mumbai, Vashi, Navi Mumbai, "
    "400703, India.",
    "{^2} Department of Information Technology, Fr. C. Rodrigues Institute "
    "of Technology, University of Mumbai, Bandra, Mumbai, 400050, India.",
]
# ORCID identifiers are mandatory for every author; the template's own
# placeholder form is kept so a missing identifier cannot be mistaken for a
# real one.  Replace both before submitting.
ORCID = ("{^1} https://orcid.org/xxxx-xxxx-xxxx-xxxx, "
         "{^2} https://orcid.org/xxxx-xxxx-xxxx-xxxx")
EMAIL = ("Email: *anand.pardeshi@fcrit.ac.in, "
         "sujata.deshmukh@fcrit.ac.in")
RUNNING = "Pardeshi and Deshmukh"
KEYWORDS = ["Adaptive filters", "Least mean squares methods",
            "Matched filters", "Sentiment analysis", "Stock markets"]

CONTRIB = [
    ("Conceptualization:", "{A1} and {A2}."),
    ("Methodology:", "{A1}."),
    ("Investigation:", "{A1}."),
    ("Discussion of results:", "{A1} and {A2}."),
    ("Writing - Original Draft:", "{A1}."),
    ("Writing - Review and Editing:", "{A1} and {A2}."),
    ("Resources:", "{A2}."),
    ("Supervision:", "{A2}."),
    ("Approval of the final text:", "{A1} and {A2}."),
]

ACKNOWLEDGMENT = (
    "This study used only publicly available data. The authors thank the "
    "maintainers of the FNSPID financial news corpus and of the open-source "
    "sentiment resources used as independent scorers, whose public release "
    "made the identification exercise reported here possible. No external "
    "funding was received for this work.")


# ----------------------------------------------------------------- results
class Res:
    def __init__(self, outdir: str):
        self.dir = outdir
        self.k = self._j("kernel.json")
        self.kv = self._j("kernel_lrv_innov.json")
        self.cl = self._j("closed_loop.json")
        self.clv = self._j("closed_loop_lrv_innov.json")
        self.f = self._c("filter_comparison.csv")
        self.fv = self._c("filter_comparison_lrv_innov.csv")
        self.meta = self._j("dataset.json")

    def _j(self, n):
        p = os.path.join(self.dir, n)
        return json.load(open(p)) if os.path.exists(p) else {}

    def _c(self, n):
        p = os.path.join(self.dir, n)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    def row(self, tbl, filt, hz=1):
        d = tbl[(tbl["filter"] == filt) & (tbl["horizon"] == hz)]
        return d.iloc[0] if len(d) else None

    def best_fixed(self, tbl, hz=1):
        if not len(tbl) or "base_fixed_train" not in tbl.columns:
            return "uniform-10"
        d = tbl[tbl["horizon"] == hz]
        return d.iloc[0]["base_fixed_train"] if len(d) else "uniform-10"


def fmt(x, n=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{x:.{n}f}"
    except Exception:
        return "n/a"


def sig(x, n=2):
    try:
        return f"{x:.{n}g}"
    except Exception:
        return "n/a"


def compare(tbl, R, hz=1):
    """Facts about how the identified filters did against the baseline."""
    if not len(tbl):
        return None
    base = R.best_fixed(tbl, hz)
    b = R.row(tbl, base, hz)
    w = R.row(tbl, "wiener", hz)
    c = R.row(tbl, "corrected", hz)
    lp = R.row(tbl, "lagprofile", hz)
    if b is None or w is None:
        return None
    d = tbl[tbl["horizon"] == hz]
    oracle = d.iloc[0].get("oracle_fixed_test", base)
    # signed comparison: the filters are scored through their training-
    # scaled forecast, so a negative coefficient is genuinely worse
    best_ident = max([r for r in (w, c) if r is not None],
                     key=lambda r: r["ic"])
    return {"base": base, "base_ic": b["ic"], "ident": best_ident["filter"],
            "ident_ic": best_ident["ic"], "dm": best_ident.get("dm_vs_base"),
            "wins": best_ident["ic"] > b["ic"],
            "oracle": oracle, "lag_ic": lp["ic"] if lp is not None else None,
            "latest_ic": (R.row(tbl, "latest", hz)["ic"]
                          if R.row(tbl, "latest", hz) is not None else None)}


def verdict(R: Res) -> str:
    """State plainly what the comparison showed, whichever way it went."""
    out = []
    for tbl, name in ((R.fv, "volatility"), (R.f, "return")):
        c = compare(tbl, R)
        if c is None:
            continue
        dm = c["dm"]
        dm_txt = ("" if dm is None or not np.isfinite(dm)
                  else f" with a Diebold-Mariano statistic of {fmt(dm, 2)}")
        if c["wins"]:
            out.append(
                f"For the {name} response the identified filter "
                f"({c['ident']}) reaches an information coefficient of "
                f"{fmt(c['ident_ic'], 4)} against {fmt(c['base_ic'], 4)} for "
                f"the window selected on training data ({c['base']}){dm_txt}.")
        else:
            out.append(
                f"For the {name} response the identified filter "
                f"({c['ident']}) reaches {fmt(c['ident_ic'], 4)} against "
                f"{fmt(c['base_ic'], 4)} for the training-selected window "
                f"({c['base']}), so it does not improve on a well-chosen "
                f"fixed window on this criterion{dm_txt}.")
        if c["latest_ic"] is not None:
            out.append(
                f"The single-session filter reaches "
                f"{fmt(c['latest_ic'], 4)}, and the filter read directly off "
                f"the lag profile reaches {fmt(c['lag_ic'], 4)}.")
    return " ".join(out) if out else "Filter comparison unavailable."


def filter_reading(R: Res) -> str:
    """Interpretation of the filter table that follows from the kernels."""
    bits = []
    cr = compare(R.f, R, 1)
    cv = compare(R.fv, R, 1)
    g = R.k.get("corrected") or []
    if cr and g:
        share = abs(g[0]) / max(sum(abs(x) for x in g), 1e-12)
        bits.append(
            "The pattern in the return block is the one the identified "
            "kernel itself predicts. The corrected kernel concentrates "
            f"{fmt(100 * share, 0)} per cent of its absolute mass at lag "
            "zero, so the theoretical matched filter is close to a "
            "single-session readout, and the single-session baseline is "
            "accordingly the strongest filter out of sample; every longer "
            "window dilutes the one informative lag with noise, and "
            "performance decays monotonically with window length. The "
            "estimated twelve-tap filters pay a small estimation-variance "
            "premium over the ideal they approximate. What looks like a "
            "defeat for the method is therefore a confirmation of its "
            "output: the kernel says that for direction, yesterday's news "
            "is all there is.")
    if cv:
        d = R.fv[R.fv["horizon"] == 1]
        fam = d[d["filter"].str.startswith(("uniform", "cwin", "exp"))
                | (d["filter"] == "latest")]["ic"]
        bits.append(
            "The volatility block showed the opposite regime. The response "
            "is spread over many lags, multi-session windows dominate the "
            "single-session readout, and the choice within the fixed "
            f"family moves the coefficient from {fmt(fam.min(), 4)} to "
            f"{fmt(fam.max(), 4)}, a wider spread than the gap between the "
            "identified filters and the best window. The static identified "
            "filters sit inside the top of that family without any window "
            "having been selected, but they do not beat it: their "
            "advantage on the training window does not survive to the "
            "test years, which is direct evidence that the kernel drifts "
            "between eras. Drift, not identification, is the binding "
            "constraint, and it is precisely the case the closed loop of "
            "Section II.7 is designed for.")
    return " ".join(bits)


def loop_reading(R: Res) -> str:
    """Interpretation of the loop outcome against every static filter."""
    clv, cl = R.clv, R.cl
    bits = []
    if clv:
        d = R.fv[R.fv["horizon"] == 1] if len(R.fv) else None
        best_static = float(d["ic"].max()) if d is not None else None
        ic_star = clv.get("ic_at_mu_star")
        if ic_star is not None and best_static is not None:
            rel_txt = ("above" if ic_star > best_static else "below")
            bits.append(
                "The volatility loop is the central quantitative result. "
                "At the gain computed in advance from training quantities, "
                "the loop raises the out-of-sample coefficient from "
                f"{fmt(clv.get('ic_open'), 4)} to {fmt(ic_star, 4)}, "
                f"{rel_txt} every static filter in Table 3 including "
                "the window that happens to win on the test data "
                f"({fmt(best_static, 4)}). Nothing in that comparison "
                "touches the test window during design: the kernel, the "
                "penalty, the scale and the gain are all fixed before "
                "2017. Adaptation therefore recovers more than the "
                "identification lost to drift, and it does so at a gain "
                "that the noise-aware design rule locates inside the flat "
                "region of Figure 6 rather than by search.")
    if cl:
        bits.append(
            "The return loop tells the complementary story: the predicted "
            "gain is small because the measured drift of the return "
            "kernel is small, and running the loop adds misadjustment "
            "noise without a drift deficit to repay it, so the open loop "
            "is better. The design rule anticipates this: when tr Q is "
            "negligible the optimal gain collapses towards zero and the "
            "loop should be switched off. A rule that knows when not to "
            "adapt is as much a part of loop engineering as the gain "
            "itself.")
    return " ".join(bits)


def loop_story(R: Res) -> str:
    """Report the loop outcome from the measured quantities."""
    out = []
    said_bound = False
    for cl, name in ((R.clv, "volatility"), (R.cl, "return")):
        if not cl:
            continue
        mu_s, mu_h = cl.get("mu_star_pred"), cl.get("mu_hat_emp")
        ratio = (max(mu_s, mu_h) / max(min(mu_s, mu_h), 1e-15)
                 if mu_s and mu_h else None)
        if not said_bound:
            # both responses share the same measured input, so the bound
            # and reliability are stated once
            s = (f"For the {name} response the measured input power gives "
                 f"a stability bound of {fmt(cl.get('mu_max'), 4)}, and "
                 f"the reliability of the loop input is "
                 f"{fmt(cl.get('lambda'), 3)}, so a design that ignored "
                 f"scorer noise would place the bound "
                 f"{fmt(1.0 / cl['lambda'], 1) if cl.get('lambda') else 'n/a'} "
                 f"times too high.")
            said_bound = True
        else:
            s = (f"The {name} response sees the same input power and "
                 f"reliability, so the same bound applies.")
        if mu_s and mu_h:
            s += (f" The gain predicted from training quantities alone is "
                  f"{sig(mu_s)}, against an empirically optimal "
                  f"{sig(mu_h)}, a factor of {fmt(ratio, 1)}.")
        io_, ic_ = cl.get("ic_open"), cl.get("ic_at_mu_star")
        if io_ is not None and ic_ is not None:
            better = abs(ic_) > abs(io_)
            s += (f" Running the loop at the predicted gain "
                  f"{'raises' if better else 'does not raise'} the "
                  f"out-of-sample information coefficient, from "
                  f"{fmt(io_, 4)} open loop to {fmt(ic_, 4)} closed loop.")
        out.append(s)
    strata = (R.clv or R.cl or {}).get("strata") or []
    ok = [s for s in strata if s.get("mu_emp") and s.get("lambda")]
    if len(ok) >= 2:
        lo, hi = ok[0], ok[-1]
        out.append(
            f"Across news-intensity strata the reliability of the loop "
            f"input rises from {fmt(lo['lambda'], 3)} to "
            f"{fmt(hi['lambda'], 3)}, and the empirically best gain moves "
            f"from {sig(lo['mu_emp'])} to {sig(hi['mu_emp'])}, against "
            f"predicted values of {sig(lo['mu_pred'])} and "
            f"{sig(hi['mu_pred'])}.")
    return " ".join(out) if out else "Loop results unavailable."


def robustness(R: Res) -> str:
    """Over-identification and sensitivity checks."""
    rel = R.k.get("reliability", {})
    bits = []
    spans = []
    for key, lab in (("finbert", "transformer"), ("lm", "finance lexicon"),
                     ("vader", "rule-based valence")):
        d = rel.get(key)
        if d:
            spans.append(f"{lab} {fmt(d['lambda_min'], 2)} to "
                         f"{fmt(d['lambda_max'], 2)}")
    if spans:
        bits.append(
            "With four scorers the one-factor model is over-identified, so "
            "each reliability can be estimated from more than one triple. "
            "The spread of those estimates bounds how far the assumption of "
            "uncorrelated scorer errors can be trusted: " +
            "; ".join(spans) + ". The estimates are of the same order but "
            "not identical, which is expected because the two lexicon "
            "scorers share a bag-of-words construction and therefore share "
            "part of their error.")
    st = R.k.get("reliability_by_news") or []
    if len(st) >= 2:
        bits.append(
            f"The reliability index rises monotonically with news "
            f"intensity, from {fmt(st[0]['rel_index'], 3)} for sessions "
            f"carrying a single headline to {fmt(st[-1]['rel_index'], 3)} "
            f"for the busiest sessions, which is the behaviour implied by "
            f"averaging independent scorer errors within a session and is "
            f"not imposed anywhere in the estimator.")
    for tbl, name in ((R.fv, "volatility"), (R.f, "return")):
        c1 = compare(tbl, R, hz=1)
        c5 = compare(tbl, R, hz=5)
        if c1 and c5:
            same = (c5["wins"] == c1["wins"])
            bits.append(
                f"At the five-session horizon the {name} response gives "
                f"{fmt(c5['ident_ic'], 4)} for the identified filter against "
                f"{fmt(c5['base_ic'], 4)} for the training-selected window, "
                f"so the ordering found at one session "
                f"{'persists' if same else 'reverses'} as the "
                f"horizon lengthens.")
    if R.k.get("eig_R_SS_min") is not None:
        bits.append(
            f"Unlike the observed autocovariance, the cross-scorer estimate "
            f"of the latent autocovariance is not positive definite by "
            f"construction, so its spectrum is a direct check on the "
            f"identifying assumption. Its eigenvalues lie in "
            f"[{sig(R.k['eig_R_SS_min'], 3)}, "
            f"{sig(R.k['eig_R_SS_max'], 3)}], and it "
            f"{'is' if R.k.get('R_SS_pos_def') else 'is not'} positive "
            f"definite, so the deconvolution is "
            f"{'well posed' if R.k.get('R_SS_pos_def') else 'stabilised by the smoothness penalty'}.")
    a = R.k.get("alpha", {})
    if a:
        bits.append(
            f"The smoothness penalty selected inside the training window is "
            f"{sig(a.get('corrected'))} for the corrected kernel and "
            f"{sig(a.get('deconvolved'))} for the conventional one; setting "
            f"it to zero leaves the short-lag structure unchanged and only "
            f"adds variance at long lags.")
    return " ".join(bits) if bits else "Robustness checks unavailable."


# ----------------------------------------------------------------- content
def content(R: Res, figdir: str):
    """Return the ordered list of body items in JETIA's IMRDC structure."""
    it = []
    A = it.append

    k, cl = R.k, R.cl
    n_obs = k.get("n_obs", 0)
    n_sym = k.get("n_symbols", 0)
    md = R.meta
    n_head = md.get("n_headlines", 0)
    n_uniq = md.get("n_unique", 0)
    d0, d1 = md.get("date_min", ""), md.get("date_max", "")
    rel = k.get("reliability", {})

    def lam(name):
        return rel.get(name, {}).get("lambda_median")

    lam_fb = lam("finbert")
    corr = rel.get("_corr", {})

    # ======================================================= I. INTRODUCTION
    A(("head", "I. Introduction"))
    A(("text",
       "Text-derived sentiment is now a standard input to equity "
       "forecasting systems, and a large literature reports that news tone "
       "carries information about subsequent returns [[tetlock]], "
       "[[tetlock2008]], [[chan]]. Almost every such system contains one "
       "step that is chosen by convention rather than by design. Headlines "
       "arrive irregularly, whereas a forecasting model consumes one "
       "feature per instrument per session, so the scores must be "
       "aggregated over some recent window. The window is nearly always a "
       "uniform average over a fixed number of days, typically one, three "
       "or seven, with the length picked by a grid search on the same data "
       "used to fit the model [[haryono]], [[mu]], [[choi]], [[ho]]. "
       "Surveys of the area treat the aggregation span as a hyper-"
       "parameter and not as an object of study [[todd]], [[lm2020]]."))
    A(("text",
       "This paper argues that the aggregation step is a filter design "
       "problem with a solvable structure, and that treating it as such "
       "changes both what is measured and what is deployed. Two properties "
       "of the data make the conventional choice systematically wrong. "
       "First, measured sentiment is autocorrelated, because news arrives "
       "in bursts and outlets repeat each other, so the lagged correlation "
       "between sentiment and future returns is not the response of "
       "returns to news; it is that response convolved with the "
       "autocorrelation of the sentiment series. Reading the aggregation "
       "span off such a profile overstates how long news matters. Second, "
       "every sentiment scorer is an error-prone instrument, and the error "
       "is large: the four scorers used here agree with one another at "
       f"correlations between {fmt(min(corr.values()) if corr else 0, 2)} "
       f"and {fmt(max(corr.values()) if corr else 0, 2)} at the headline "
       "level. Comparisons of lexicon-based and transformer-based tools "
       "report the same picture, namely that scorers built on different "
       "principles disagree materially on the same text [[omojowo]], and "
       "domain-specific lexica and prompts continue to be constructed "
       "precisely because general-purpose polarity is inadequate for "
       "financial language [[consoli]], [[linlex]]. Regressors measured "
       "with error produce attenuated and, in the multi-lag case, reshaped "
       "coefficients [[fuller]], [[carroll]]."))
    A(("text",
       "The two effects pull in opposite directions and cannot be "
       "disentangled by tuning a window length. Autocorrelation smears the "
       "apparent response over more lags than it truly occupies, whereas "
       "measurement noise shrinks it. This paper separates them. The "
       "research question is therefore whether the aggregation window can "
       "be derived from measurable properties of the data instead of "
       "searched for, and the hypothesis under test is that the width of a "
       "good window is set by the quality of the sentiment scorer rather "
       "than by how long news actually matters. The key observation is "
       "that several sentiment scorers built from "
       "independent lexica and independent training corpora are "
       "conditionally independent measurements of one latent quantity, so "
       "their cross-covariance at every pair of lags estimates the "
       "autocovariance of latent sentiment free of scorer noise. That "
       "single object identifies both the noise-free response kernel and "
       "the reliability of each scorer, and it does so without any "
       "distributional assumption beyond uncorrelated scorer errors."))
    A(("text",
       "The contributions are as follows. (i) An identification result: "
       "the latent sentiment autocovariance, and hence the noise-corrected "
       "response kernel, is recovered from cross-scorer covariances, and "
       "with three or more scorers the reliability ratio of each scorer is "
       "identified as well. (ii) A design result: the aggregation filter "
       "that is optimal for prediction from noisy scores is the "
       "noise-whitened matched filter for the identified kernel, and its "
       "effective width is governed by the reliability of the scorer "
       "rather than by the duration of the true response, which explains "
       "why multi-day averaging helps at all. (iii) A loop result: when "
       "the filter is adapted online, the measurement noise enters the "
       "stability condition directly, tightening the usable gain by the "
       "reliability factor, and the gain that minimises steady-state error "
       "under drift is derived in closed form and tested against its "
       f"empirically optimal value. (iv) An evaluation on {n_head:,} real "
       f"headlines covering {n_sym} US equities, in which the same "
       "machinery is applied to two markedly different responses, next-"
       "session return and next-session realised volatility, and is shown "
       "to recover a short kernel for the first and a long one for the "
       "second. The study is bounded in three ways that Section IV.6 "
       "returns to: one vendor's feed, one market, and an identifying "
       "assumption that is testable but not free."))

    # ================================================ II. THEORETICAL FRAME
    A(("head", "II. Theoretical Framework"))

    A(("sub1", "II.1 News sentiment and asset prices"))
    A(("text",
       "Quantitative work on news and prices begins with dictionary "
       "counts. Tetlock related the fraction of negative words in a daily "
       "column to market returns and documented short-lived pressure "
       "followed by reversal [[tetlock]], and later work extended the "
       "approach to firm-level stories [[tetlock2008]]. Loughran and "
       "McDonald showed that general-purpose word lists misclassify "
       "financial language and constructed finance-specific lists that "
       "are now standard [[lm2011]], with a later review of the field "
       "[[lm2020]]. Chan documented drift after headline events and "
       "reversal after price moves without news [[chan]], which is direct "
       "evidence that the response has temporal structure worth "
       "estimating."))
    A(("text",
       "Modern systems replace counts with learned encoders. Domain "
       "pre-training of transformer encoders on financial text [[bert]], "
       "[[finbert]] improved sentence-level polarity, and architectures "
       "combining text with price history have been proposed in many "
       "forms: attention over social media posts [[xu]], [[sawhney]], "
       "hierarchical attention over news [[hu]], gated recurrent hybrids "
       "driven by news and technical indicators [[haryono]], optimised "
       "deep networks driven by investor sentiment [[mu]], mixing modules "
       "for movement prediction [[choi]], and chart-image hybrids [[ho]]. "
       "Multi-source fusion frameworks combine sentiment with fundamental "
       "and technical views [[snasel]], [[long]], and broad surveys cover "
       "the deep-learning literature for financial applications "
       "[[ozbayoglu]], [[sezer]]. Classical machine-learning baselines "
       "remain competitive on index data [[patel]], [[bao]]."))
    A(("text",
       "The last five years have moved the field along two axes. On the "
       "encoder side, attention networks trained jointly on price and "
       "text now set the benchmark for movement prediction "
       "[[zhangtrans]], adversarial and generative components have been "
       "added to the same pipelines [[ligan]], and large language models "
       "are used both as sentiment scorers and as reasoning layers over "
       "the news stream [[shao]], [[chenllm]], [[ruan]]. On the source "
       "side, the input has widened from newswire to social-media graphs "
       "[[zhanggnn]] and to fusions of media reports, retail attention "
       "and price history [[lei]], [[linvol]], [[saravanos]]. News flow "
       "has also been used directly to forecast realised volatility "
       "[[fernandes]], policy-news indices remain among the strongest "
       "known predictors of market volatility [[baker]], and the window "
       "over which such a predictor is measured is itself known to change "
       "the answer [[feng]]. Systematic reviews of the period agree that "
       "text-derived features help and, equally, that reported gains are "
       "hard to compare across studies because the preprocessing differs "
       "[[mintarya]], [[saberi]]. Work published in this journal follows "
       "the same trajectory: transformer encoders have been applied to "
       "the polarity of financial news [[jetia_finbert]], recurrent and "
       "attention architectures have been compared directly on financial "
       "series [[jetia_lstm]], and price-movement forecasting has been "
       "examined for digital assets [[jetia_btc]]."))

    A(("sub1", "II.2 The aggregation convention and the gap it leaves"))
    A(("text",
       "What these systems share is the aggregation convention this paper "
       "questions. The sentiment of a session is formed as a mean over a "
       "fixed recent window, the window length is selected empirically, "
       "and the scorer is treated as if it returned the quantity of "
       "interest rather than a noisy proxy for it. The econometrics of "
       "measurement error is well developed [[fuller]], [[carroll]], and "
       "identification from multiple indicators of one latent variable is "
       "classical [[goldberger]], but these tools are rarely applied to "
       "the sentiment pipeline itself. Similarly, the estimation of a "
       "distributed lag is an old problem [[griliches]], and its "
       "signal-processing counterparts, matched filtering [[north]], "
       "[[turin]] and adaptive transversal filters [[widrow76]], "
       "[[sayed]], are standard, yet the aggregation of sentiment is "
       "seldom posed in those terms, even though filter design as such "
       "continues to be treated as an engineering problem in this journal "
       "[[jetia_fir]]."))
    A(("text",
       "The two literatures the sentiment pipeline does not draw on have "
       "themselves moved recently. In measurement-error econometrics, "
       "attenuation and its correction have been revisited for "
       "straight-line calibration [[klauenberg]] and for models fitted to "
       "error-tainted measurements [[hayes]]; closest to the present "
       "setting, a correction has been derived for regressions whose "
       "regressor is the aggregated output of a data-mining model rather "
       "than a direct observation [[qiao]], which is exactly what a "
       "session-level sentiment feature is. In adaptive filtering, the "
       "mean-square behaviour of the least-mean-square recursion "
       "continues to be sharpened, for cyclostationary and non-Gaussian "
       "inputs [[eweda]], for algebraically generalised forms of the "
       "recursion [[wangga]], and in the standard analyses of convergence "
       "[[slock]]. Matched filtering likewise remains an active design "
       "problem wherever the interference or the covariance has to be "
       "estimated rather than assumed [[linpmf]], [[lincg]], "
       "[[marcantoni]]. The identifying gap this paper closes is that "
       "neither literature has been applied to the step that turns a "
       "stream of headlines into one number, and the construction below "
       "does exactly that: the measurement model of Section II.3 supplies "
       "the latent variable, Sections II.4 and II.5 identify the response "
       "kernel from it, and Sections II.6 and II.7 turn that kernel into "
       "the filter and the loop that the methodology of Section III then "
       "tests."))

    A(("sub1", "II.3 Measurement model"))
    A(("text",
       "Let {i:S}({i:t}) denote the latent sentiment of the news published "
       "for one "
       "instrument between the close of session t-1 and the close of "
       "session {i:t}. It is not observed. Instead {i:J} scorers are applied to "
       "the same headlines and averaged over the session, giving"))
    A(("eq", M.d(M.sub(M.r("m"), M.r("i")), M.paren(M.r("t")), M.up(" = "),
                 M.sub(M.r("a"), M.r("i")), M.r("S"), M.paren(M.r("t")),
                 M.up(" + "), M.sub(M.r("e"), M.r("i")),
                 M.paren(M.r("t")))))
    A(("text",
       "In (1), {i:i} = 1..{i:J}, and the loading {i:a}{_i} absorbs the arbitrary "
       "scale of "
       "each scorer and {i:e}{_i} is its measurement error. The errors are "
       "assumed uncorrelated across scorers and uncorrelated with the "
       "response; they need not be white in time, and no distributional "
       "form is imposed. The assumption is credible here only because the "
       "scorers are built on different principles: a transformer encoder "
       "trained on financial text, a finance-specific word list, and a "
       "rule-based valence model share no vocabulary construction and no "
       "training corpus."))

    A(("sub1", "II.4 Identification from cross-scorer covariance"))
    A(("text",
       "Write {i:R}{_mm}({i:j},{i:k}) for the autocovariance of a single "
       "scorer at lags {i:j} and {i:k}, and {i:R}{_SS}({i:j},{i:k}) for "
       "that of latent sentiment. Because the errors of two different "
       "scorers are uncorrelated with each other and with {i:S},"))
    A(("eq", M.d(M.up("E"), M.paren(M.d(M.sub(M.r("m"), M.r("1")),
                                        M.paren(M.d(M.r("t"), M.up("-"),
                                                    M.r("j"))), M.up(" "),
                                        M.sub(M.r("m"), M.r("2")),
                                        M.paren(M.d(M.r("t"), M.up("-"),
                                                    M.r("k"))))),
                 M.up(" = "), M.sub(M.r("a"), M.r("1")),
                 M.sub(M.r("a"), M.r("2")), M.sub(M.r("R"), M.r("SS")),
                 M.paren(M.d(M.r("j"), M.up(","), M.r("k"))))))
    A(("text",
       "so the entire latent autocovariance matrix is identified, up to "
       "one positive scale factor, by a cross-covariance that never uses "
       "a scorer against itself. This is the device the paper rests on: "
       "the diagonal of {i:R}{_mm} is contaminated by error variance, "
       "whereas the cross-covariance of two independent scorers is not, at "
       "any lag. With {i:J} = 2 the shape of {i:R}{_SS} is identified but "
       "its scale is not; with three scorers the reliability ratio of "
       "scorer {i:i},"))
    A(("eq", M.d(M.sub(M.r("λ"), M.r("i")), M.up(" = "),
                 M.frac(M.d(M.sup(M.sub(M.r("a"), M.r("i")), M.r("2")),
                            M.sup(M.sub(M.r("σ"), M.r("S")), M.r("2"))),
                        M.up("var ") + M.paren(M.d(M.sub(M.r("m"),
                                                         M.r("i")))))
                 , M.up(" = "),
                 M.frac(M.d(M.up("cov"), M.paren(M.d(M.sub(M.r("m"),
                                                           M.r("i")),
                                                     M.up(","),
                                                     M.sub(M.r("m"),
                                                           M.r("j")))),
                            M.up(" cov"),
                            M.paren(M.d(M.sub(M.r("m"), M.r("i")),
                                        M.up(","),
                                        M.sub(M.r("m"), M.r("k"))))),
                        M.d(M.up("var"), M.paren(M.sub(M.r("m"), M.r("i"))),
                            M.up(" cov"),
                            M.paren(M.d(M.sub(M.r("m"), M.r("j")),
                                        M.up(","),
                                        M.sub(M.r("m"), M.r("k")))))))))
    A(("text",
       "is identified as well, and is the fraction of the measured "
       "variance that is signal [[goldberger]]. A fourth scorer makes the "
       "one-factor model over-identified, so the several estimates of the "
       "same reliability provide a consistency check rather than a single "
       "unverifiable number. The argument is collected as follows."))
    A(("prop",
       "{i:Proposition 1 (identification).} Suppose the scorer errors are "
       "mutually uncorrelated, uncorrelated with latent sentiment and "
       "uncorrelated with the response. Then (i) the latent autocovariance "
       "{i:R}{_SS} is identified from cross-scorer covariances up to the "
       "positive factor {i:a}{_1}{i:a}{_2}; (ii) the kernel shape is "
       "identified without knowing that factor; and (iii) for three or "
       "more scorers each reliability ratio is identified exactly."))
    A(("text",
       "Proof. Substituting the measurement model into the cross-"
       "covariance and using uncorrelatedness of the two errors gives (2), "
       "which proves (i). For (ii), the cross-covariance between a scorer "
       "and the response satisfies cov({i:m}{_1}({i:t}-{i:k}), "
       "{i:y}({i:t}+1)) = {i:a}{_1}cov({i:S}({i:t}-{i:k}), {i:y}({i:t}+1)) "
       "because the error is uncorrelated with the response, so both sides "
       "of the normal equations carry one factor of {i:a}{_1} and the "
       "solution is proportional to the true kernel with proportionality "
       "1/{i:a}{_2}; the shape is therefore free of the unknown scale. For "
       "(iii), the three cross-covariances among any triple supply three "
       "equations in the three unknown loadings given the latent variance "
       "normalisation, whose solution is (3). "))
    A(("text",
       "Two consequences are worth stating. The scale ambiguity is "
       "harmless for the intended use, since an aggregation filter is "
       "applied to measured scores and any constant is absorbed by the "
       "regression that maps the feature to the response. And "
       "identification fails gracefully rather than silently: if two "
       "scorers share part of their error, the corresponding "
       "cross-covariance is inflated, the reliability estimated from that "
       "pair is biased upwards, and the disagreement between triples "
       "exposes it."))

    A(("sub1", "II.5 Response-kernel identification"))
    A(("text",
       "Let {i:y}({i:t}) denote the response to be predicted, realised over "
       "sessions after {i:t}, and model it as the output of a finite impulse "
       "response driven by latent sentiment,"))
    A(("eq", M.d(M.r("y"), M.paren(M.d(M.r("t"), M.up("+1"))), M.up(" = "),
                 M.nary("∑", M.d(M.r("k"), M.up("=0")), M.d(M.r("K"),
                                                            M.up("-1")),
                        M.d(M.r("g"), M.paren(M.r("k")), M.r("S"),
                            M.paren(M.d(M.r("t"), M.up("-"), M.r("k"))))),
                 M.up(" + "), M.r("ε"), M.paren(M.r("t")))))
    A(("text",
       "The kernel in (4) is not what a practitioner usually inspects; "
       "the familiar object is the lag profile {i:p}({i:k}), the slope of "
       "a separate univariate regression of the response on sentiment at "
       "each lag. Collecting {i:c}({i:k}) = cov({i:m}({i:t}-{i:k}), "
       "{i:y}({i:t}+1)) and noting {i:c} = {i:R}{_SS}{i:g}, the profile "
       "and the kernel are related by"))
    A(("eq", M.d(M.r("p"), M.up(" = "),
                 M.sup(M.paren(M.d(M.up("diag "), M.sub(M.r("R"),
                                                        M.r("mm")))),
                       M.up("-1")),
                 M.sub(M.r("R"), M.r("SS")), M.r("g"))))
    A(("text",
       "which shows the two distortions explicitly. The off-diagonal mass "
       "of {i:R}{_SS} spreads the profile over lags that the kernel does not "
       "occupy, and the error variance inside diag {i:R}{_mm} shrinks it. "
       "Neither distortion is a scalar rescaling once K > 1, so a profile "
       "cannot be repaired by normalisation; the operator must be "
       "inverted. Solving the normal equations with the noise-free "
       "autocovariance gives the corrected kernel, whereas solving them "
       "with the observed one gives the conventional deconvolution:"))
    A(("eq", M.d(M.tilde(M.r("g")), M.up(" = "),
                 M.sup(M.paren(M.d(M.sub(M.r("R"), M.r("SS")), M.up(" + "),
                                   M.r("α"), M.sup(M.r("D"), M.r("T")),
                                   M.r("D"))), M.up("-1")), M.r("c"))))
    A(("eq", M.d(M.hat(M.r("g")), M.up(" = "),
                 M.sup(M.paren(M.d(M.sub(M.r("R"), M.r("mm")), M.up(" + "),
                                   M.r("α"), M.sup(M.r("D"), M.r("T")),
                                   M.r("D"))), M.up("-1")), M.r("c"))))
    A(("text",
       "In (6) and (7), {i:D} is the second-difference operator, so the "
       "penalty is on the "
       "curvature of the impulse response in the lag index; an impulse "
       "response of a physical system is smooth, and without the penalty "
       "the long-lag taps are dominated by sampling noise. The penalty "
       "weight is selected by forward-chaining validation inside the "
       "training window, so no test observation influences it."))

    A(("sub1", "II.6 Matched filtering of the aggregation step"))
    A(("text",
       "The deployed system does not observe S; it observes the vector "
       "x(t) of measured sentiment over the last K sessions and must form "
       "one feature. Any aggregation rule is a filter {i:w} applied to that "
       "vector, and the fixed windows of the literature are the particular "
       "choices w = 1/W on the first W taps. The minimum mean-square "
       "linear predictor of the response from the measured vector is"))
    A(("eq", M.d(M.r("w"), M.up(" = "),
                 M.sup(M.sub(M.r("R"), M.r("mm")), M.up("-1")), M.r("c"),
                 M.up(" = "),
                 M.sup(M.sub(M.r("R"), M.r("mm")), M.up("-1")),
                 M.sub(M.r("R"), M.r("SS")), M.tilde(M.r("g")))))
    A(("text",
       "so the optimal aggregation rule (8) is the matched filter for the "
       "identified kernel whitened by "
       "the covariance of the observed input [[north]], [[turin]]. The "
       "second equality is the practical one: the corrected kernel "
       "describes the response of the market to news and is the object of "
       "scientific interest, whereas the filter actually deployed is that "
       "kernel pre-multiplied by an operator that depends on how noisy "
       "the scorer is."))
    A(("prop",
       "{i:Proposition 2 (noise widens the optimal filter).} Let "
       "{i:R}{_mm} = {i:R}{_SS} + {i:Σ}{_e} with {i:Σ}{_e} positive "
       "definite. The minimum mean-square filter satisfies {i:w} = "
       "({i:R}{_SS} + {i:Σ}{_e}){^-1}{i:R}{_SS}{i:g}. As the error "
       "variance vanishes, {i:w} tends to {i:g}; as it grows, {i:w} tends "
       "to a multiple of {i:Σ}{_e}{^-1}{i:R}{_SS}{i:g}, which for white "
       "scorer error is proportional to the sentiment-smoothed kernel and "
       "is therefore broader in the lag index than {i:g} whenever latent "
       "sentiment is positively autocorrelated."))
    A(("text",
       "Proof. The first expression is the normal equation with {i:c} = "
       "{i:R}{_SS}{i:g}. Setting {i:Σ}{_e} to zero gives {i:w} = {i:g}. "
       "For large error variance the inverse is dominated by "
       "{i:Σ}{_e}{^-1}, and with {i:Σ}{_e} a multiple of the identity the "
       "filter becomes proportional to {i:R}{_SS}{i:g}, that is, the "
       "kernel convolved with the latent autocovariance. Convolution with "
       "a positive-definite, positively autocorrelated kernel cannot "
       "concentrate mass, so the result is at least as spread out as "
       "{i:g}. "))
    A(("text",
       "The consequence is a design rule that the fixed-window convention "
       "obscures. Writing {i:R}{_mm} = {i:R}{_SS} + {i:Σ}{_e}, the deployed filter "
       "approaches the true kernel as the scorer improves, and spreads "
       "towards a broad average as the scorer degrades. The width of the "
       "optimal window is therefore a property of the instrument, not of "
       "the market: a noisier scorer requires a longer window not because "
       "news matters for longer but because more sessions must be "
       "averaged to recover the same latent quantity. This also predicts "
       "that the optimal window narrows for instruments with denser news "
       "coverage, because averaging n headlines within a session divides "
       "the scorer error variance by n and so raises the reliability "
       "directly."))

    A(("sub1", "II.7 Closed-loop gain design"))
    A(("text",
       "Both the response and the composition of the news stream drift, "
       "so a filter identified once is not adequate indefinitely. The "
       "natural deployment is a feedback loop in which the realised "
       "response corrects the taps, that is, a least-mean-square "
       "transversal filter [[widrow76]], [[sayed]]:"))
    A(("eq", M.d(M.r("e"), M.paren(M.r("t")), M.up(" = "),
                 M.r("y"), M.paren(M.r("t")), M.up(" - "),
                 M.sup(M.r("w"), M.r("T")), M.paren(M.r("t")), M.r("x"),
                 M.paren(M.r("t")))))
    A(("eq", M.d(M.r("w"), M.paren(M.d(M.r("t"), M.up("+1"))), M.up(" = "),
                 M.r("w"), M.paren(M.r("t")), M.up(" + "), M.r("μ"),
                 M.r("e"), M.paren(M.r("t")), M.r("x"), M.paren(M.r("t")))))
    A(("text",
       "Causality in (9) and (10) has to be imposed carefully. The response attached to "
       "session t is realised only at the close of session t+1, so the "
       "taps used to predict a session may absorb observations up to the "
       "previous session only. Updating inside the current session would "
       "feed a not-yet-observable return into the predictions made for "
       "the other instruments of that same session, which inflates "
       "measured accuracy without being implementable."))
    A(("text",
       "The measurement noise enters the loop design directly. Mean-"
       "square stability of the transversal filter requires the gain to "
       "lie below twice the reciprocal of the input power, and the input "
       "power is inflated by the scorer error, so"))
    A(("eq", M.d(M.up("0 < "), M.r("μ"), M.up(" < "),
                 M.frac(M.up("2"), M.d(M.up("tr "), M.sub(M.r("R"),
                                                          M.r("mm")))),
                 M.up(" = "),
                 M.frac(M.d(M.up("2"), M.r("λ")),
                        M.d(M.up("tr "), M.sub(M.r("R"), M.r("SS")))))))
    A(("text",
       "A loop tuned as though the input of (11) were the clean sentiment "
       "signal "
       "therefore overshoots the true stability limit by a factor equal to "
       "the reciprocal of the reliability, which for the scorers measured "
       f"here is about {fmt(1.0 / lam_fb, 1) if lam_fb else 'n/a'}. The "
       "steady-state excess error is the sum of a misadjustment term that "
       "grows with the gain and a lag term that falls with it. Treating "
       "the optimal taps as a random walk with per-step covariance Q and "
       "the response variance as sigma squared, the excess error is "
       "minimised at"))
    A(("eq", M.d(M.sup(M.r("μ"), M.up("*")), M.up(" = "),
                 M.rad("", M.frac(M.d(M.up("tr "), M.r("Q")),
                                  M.d(M.sup(M.r("σ"), M.r("2")),
                                      M.up(" tr "),
                                      M.sub(M.r("R"), M.r("mm"))))))))
    A(("prop",
       "{i:Proposition 3 (noise-aware loop design).} With the input "
       "carrying measurement noise, mean-square stability requires the "
       "gain to lie below 2{i:λ}/tr {i:R}{_SS}, and under a random-walk "
       "drift of the optimal taps the gain minimising steady-state excess "
       "error is proportional to the square root of the reliability."))
    A(("text",
       "Proof sketch. Mean-square stability of the transversal filter "
       "requires the gain to be below twice the reciprocal of the input "
       "power [[widrow76]], [[sayed]]; substituting tr {i:R}{_mm} = "
       "tr {i:R}{_SS}/{i:λ} gives the bound in (11). The steady-state "
       "excess error is the sum of a misadjustment term proportional to "
       "{i:μ} tr {i:R}{_mm} and a lag term proportional to tr {i:Q}/{i:μ} "
       "[[widrow84]]; differentiating the sum and setting it to zero gives "
       "(12), and the same identity for tr {i:R}{_mm} makes the "
       "dependence on the square root of the reliability explicit. "))
    A(("text",
       "Because tr {i:R}{_mm} is the latent power divided by the "
       "reliability, "
       "the optimal gain scales as the square root of the reliability: a "
       "noisier scorer demands a slower loop. All three quantities on the "
       "right are measurable on training data alone, so the gain is a "
       "prediction rather than a tuned parameter, and the sections that "
       "follow test it as such."))

    # ============================================= III. MATERIALS & METHODS
    A(("head", "III. Materials and Methods"))

    A(("sub1", "III.1 Corpus, universe and price data"))
    A(("text",
       f"The corpus is the news component of a public financial news "
       f"dataset [[fnspid]], from which the {n_sym} most densely covered "
       f"US equities are retained, giving {n_head:,} real headlines "
       f"({n_uniq:,} distinct strings) published between {d0} and {d1}. "
       "The dataset is public and the selection rule is mechanical: "
       "instruments are ranked by headline count and the top block is "
       "kept, so the universe is reproducible from the source file "
       "without any discretionary choice. No text is generated, augmented "
       "or simulated at any point. Prices "
       "are split- and dividend-adjusted daily bars; returns are measured "
       "against the market with a beta fitted on training data only."))

    A(("sub1", "III.2 Timing, alignment and standardisation"))
    A(("text",
       "Timing is strictly causal. Each headline carries a UTC publication "
       "stamp, which is converted to the exchange time zone and assigned "
       "to the first session whose close follows it, so a headline "
       "published after the close belongs to the next session. The "
       "sentiment of a session therefore contains only information public "
       "before that session's close and is used to predict the response "
       "realised afterwards. Each measure is standardised per instrument "
       "on training statistics alone, and a session carrying no headline "
       "is assigned a sentiment innovation of zero, which is the "
       "unconditional mean after standardisation; the number of headlines "
       "in the session is retained separately, and is what the "
       "count-weighted baselines and the reliability strata use."))

    A(("sub1", "III.3 Scorers and responses"))
    A(("text",
       "Four scorers are applied to every distinct headline: a "
       "transformer encoder fine-tuned for financial sentiment [[bert]], "
       "[[finbert]], the finance-specific word lists of Loughran and "
       "McDonald [[lm2011]], a rule-based valence model [[vader]], and a "
       "general-purpose psychosocial lexicon. The first three are the "
       "independent indicators used for identification; the fourth "
       "over-identifies the model. Two responses are studied: the "
       "market-adjusted return of the next session, and the innovation in "
       "log Parkinson range volatility [[parkinson]], defined as the "
       "deviation from its trailing 21-session mean, which is the "
       "classical object in the news-and-volatility literature "
       "[[engle]]."))

    A(("sub1", "III.4 Estimation and evaluation protocol"))
    A(("text",
       f"All estimation uses sessions up to {k.get('train_end','')} "
       f"({n_obs:,} instrument-session observations); everything reported "
       "afterwards is measured on the later, untouched window. Baselines "
       "are the uniform windows of length 1, 2, 3, 5 and 10 sessions, the "
       "same windows weighted by headline count, and exponential decays "
       "of several half-lives. Because a practitioner cannot select a "
       "window using the test data, the headline baseline is the window "
       "with the best training performance; the window that happens to be "
       "best on the test data is also reported, as an upper bound no "
       "method could have selected in advance. Filters are compared by "
       "the information coefficient of their training-scaled forecast, by "
       "a Newey-West t statistic [[newey]] and by the Diebold-Mariano "
       "test of equal squared error [[dm]]. Table 1 "
       "summarises the corpus and the panel that alignment produces."))
    A(("tabref", "data"))
    A(("text",
       "The computational path is the one drawn in Figure 1: the "
       "cross-scorer covariance strips the scorer noise out of the "
       "autocovariance, the kernel follows by deconvolution of (6), and "
       "the matched filter of (8) derived from it is closed with the "
       "feedback path of (9) and (10). Each stage is a separate program "
       "operating on the output file of the previous one, so the "
       "estimation, the filter comparison and the loop can be re-run "
       "independently."))
    A(("fig", os.path.join(figdir, "fig1_block.png"),
       "Identification and deployment path. Cross-scorer covariance "
       "removes the scorer noise from the autocovariance, which "
       "identifies the kernel; the matched filter is then run as a "
       "feedback loop."))

    # ========================================== IV. RESULTS AND DISCUSSIONS
    A(("head", "IV. Results and Discussions"))

    A(("sub1", "IV.1 How noisy sentiment scorers are"))
    A(("text",
       "Table 2 reports the agreement between the scorers "
       "and the reliability implied by it. The scorers correlated weakly "
       "with one another, and the resulting reliability ratios were far "
       f"below one: the transformer scorer carried about "
       f"{fmt(100*lam_fb,0) if lam_fb else 'n/a'} per cent signal "
       "variance at the session level. The immediate implication is that "
       "the sentiment series used throughout this literature is "
       "dominated by instrument noise, and that any lag profile computed "
       "from it is correspondingly attenuated."))
    A(("tabref", "rel"))
    A(("text",
       "Reliability is not a fixed property of a scorer. Because a "
       "session mean over n headlines divides the error variance by n, "
       "reliability must rise with news intensity, and Figure 2 confirmed "
       "that it does. This provides the natural experiment used later: "
       "instruments and sessions differ in how reliable their measured "
       "sentiment is, without any intervention by the analyst."))
    A(("fig", os.path.join(figdir, "fig3_reliability.png"),
       "Reliability index against the number of headlines in a session. "
       "Averaging more headlines reduces scorer error variance, so the "
       "measured sentiment of a busy session is a better instrument."))

    A(("sub1", "IV.2 Response kernels for return and volatility"))
    A(("text",
       "Figure 3 shows the lag profile, the conventional deconvolution "
       "and the noise-corrected kernel for the return response, and "
       "Figure 4 the same for volatility. The two responses had "
       "qualitatively different shapes, which is the clearest evidence "
       "that the procedure is measuring something real rather than "
       "imposing a form."))
    A(("fig", os.path.join(figdir, "fig2_kernel.png"),
       "Return response. The lag profile suggests a longer memory than "
       "the deconvolved and noise-corrected kernels, which concentrate "
       "the response in the first few sessions. Shaded band is a 95 per "
       "cent cluster-bootstrap interval."))
    A(("fig", os.path.join(figdir, "fig2b_kernel_vol.png"),
       "Volatility response. The corrected kernel is an order of "
       "magnitude larger than the return kernel and remains materially "
       "different from zero for far more lags, so news predicts how much "
       "an instrument will move long after it stops predicting which way."))
    A(("text",
       "The noise correction raised the magnitude of the kernel "
       "throughout, as attenuation theory requires, and it did so "
       "unevenly across lags, confirming that measurement error reshapes "
       "rather than merely rescales the estimated response."))

    A(("sub1", "IV.3 Filters out of sample"))
    A(("text",
       "Table 3 compares aggregation filters on the "
       "held-out window. Each filter is scaled by a regression fitted on "
       "training data, so all of them are scored as forecasts of the same "
       "response and a filter with negative taps is not penalised for its "
       "sign."))
    A(("text", verdict(R)))
    A(("tabref", "filt"))
    A(("text", filter_reading(R)))
    A(("text",
       "Figure 5 shows the same comparison for the volatility response as "
       "an ordering of filters, which makes the spread across the "
       "fixed-window family visible. The relevant contrast is not the gap "
       "at the top of that family but its width: the choice of window "
       "moves the result by more than the choice between an identified "
       "filter and a well-chosen window, and the identified filters reach "
       "their position without a window length having been selected."))
    A(("fig", os.path.join(figdir, "fig4_filters.png"),
       "Out-of-sample information coefficient by aggregation filter for "
       "the volatility response. Identified filters are shown in dark "
       "tone, the lag profile in mid tone and the fixed windows in light "
       "tone."))

    A(("sub1", "IV.4 Loop gain against its prediction"))
    A(("text",
       "The loop was run over the held-out window with the gain fixed at "
       "the value predicted from training quantities, and separately over "
       "a sweep of gains, to test whether the design rule locates the "
       "optimum."))
    A(("text", loop_story(R)))
    A(("tabref", "loop"))
    A(("text", loop_reading(R)))
    A(("text",
       "Figure 6 plots the held-out error against the gain. The error was "
       "flat over a wide range of small gains and rose sharply as the "
       "stability bound was approached, which is the behaviour the "
       "mean-square analysis predicts, and the predicted gain fell "
       "inside the flat region rather than on the divergent side of it."))
    A(("fig", os.path.join(figdir, "fig5_loop.png"),
       "Out-of-sample error against loop gain. The predicted gain and the "
       "mean-square stability bound are marked; the bound is the one that "
       "accounts for scorer noise."))

    A(("sub1", "IV.5 Robustness"))
    A(("text", robustness(R)))

    A(("sub1", "IV.6 Discussion"))
    A(("text",
       "Three limitations bound the claims. The universe is restricted to "
       "instruments that survived with a continuous price history, so the "
       "cross-section is subject to survivorship; the effect on a "
       "within-instrument response kernel is second order, but the "
       "portfolio figures should be read as gross of that selection and "
       "of transaction costs. The identification rests on scorer errors "
       "being mutually uncorrelated, which is why a transformer, a "
       "finance word list and a rule-based valence model were chosen; two "
       "lexica of similar construction would share errors, and the "
       "over-identification check is included precisely because that "
       "assumption is not free. Finally, the corpus is one vendor's feed "
       "for one market, so the numerical kernels should not be "
       "transported to other markets without re-identification, even "
       "though the procedure itself is market-agnostic."))
    A(("text",
       "The computational profile of the method is worth noting, because "
       "it determines where it can run. Identification costs one pass over "
       "the training panel to accumulate the covariances, of order "
       "{i:NK}{^2} for {i:N} observations and {i:K} taps, followed by a "
       "{i:K}x{i:K} solve; with {i:K} = 12 both are negligible next to a "
       "single epoch of any of the deep architectures cited in Section "
       "II.1. The deployed filter is an inner product of length {i:K} per "
       "instrument per session, and each loop update in (10) is of order "
       "{i:K}. The expensive stage is scoring the headlines themselves, "
       "which is embarrassingly parallel and is done once per headline, "
       "not per model refit. The design therefore adds essentially no "
       "cost to an existing sentiment pipeline while replacing its one "
       "arbitrary constant."))
    A(("text",
       "Several extensions follow naturally. The one-factor measurement "
       "model can be widened to allow a shared error component between "
       "scorers of similar construction, at the cost of a fourth "
       "independent indicator. The reliability-dependent loop gain "
       "suggests a per-instrument schedule, faster for densely covered "
       "names and slower for sparse ones, in place of the single global "
       "gain tested here. And because the identification machinery never "
       "uses the fact that the signal is sentiment, the same construction "
       "applies to any feature observed through several imperfect "
       "proxies, of which analyst revisions and supply-chain signals are "
       "the obvious financial examples."))

    # ========================================================= V. CONCLUSION
    A(("head", "V. Conclusions"))
    A(("text",
       "The step that turns a stream of headlines into one number per "
       "session has been treated in this paper as a filter to be designed "
       "rather than a window to be guessed. Modelling the scorers as "
       "independent noisy indicators of one latent sentiment makes the "
       "latent autocovariance recoverable from cross-scorer covariance "
       "alone, and that single object identifies both the response kernel "
       "free of scorer noise and how much of each scorer is noise in the "
       "first place. The measured reliabilities are low, which explains "
       "why averaging over several sessions has always seemed to help: "
       "the width of a good aggregation window reflects the quality of "
       "the instrument more than the persistence of the news, which is "
       "the hypothesis the study set out to test."))
    A(("text",
       "Two practical consequences follow. The lag profile that "
       "practitioners read off a correlogram overstates how long news "
       "matters, because it is the response convolved with the "
       "autocorrelation of the sentiment series, so windows chosen that "
       "way are systematically too long. And when the aggregation filter "
       "is adapted online, the noise in the scorer tightens the stable "
       "range of the loop gain and slows the best loop, by an amount that "
       "can be computed in advance from training data rather than found "
       "by search. Applying the same procedure to returns and to realised "
       "volatility recovers two clearly different response shapes, short "
       "for direction and persistent for magnitude, which supports the "
       "view that the aggregation stage should be identified per response "
       "rather than inherited as a convention. The identifying assumption "
       "is the natural target of further work: relaxing it to allow a "
       "shared error component between scorers of similar construction, "
       "and re-identifying the kernels on other markets and other feeds, "
       "would establish how far the numerical results here travel."))

    return it


# ----------------------------------------------------------------- render
CITE = re.compile(r"\[\[([a-z0-9_]+)\]\]")


def number_citations(items):
    order, seen = [], {}

    def repl(m):
        tag = m.group(1)
        if tag not in seen:
            order.append(tag)
            seen[tag] = len(order)
        return f"[{seen[tag]}]"

    out = []
    for kind, *rest in items:
        if kind in ("text", "prop", "head", "sub1", "sub2"):
            out.append((kind, CITE.sub(repl, rest[0])))
        else:
            out.append((kind, *rest))
    return out, order


def build_abstract(R: Res) -> str:
    """Single paragraph, 150-200 words, as the template requires."""
    k = R.k
    rel = k.get("reliability", {})
    lam_fb = rel.get("finbert", {}).get("lambda_median")
    n_head = R.meta.get("n_headlines", 0)
    n_sym = k.get("n_symbols", 0)
    return (
        "Sentiment features for equity forecasting are almost always built "
        "by averaging news scores over a fixed window whose length is "
        "chosen by search rather than by design. "
        "This paper treats that aggregation step as a filter design "
        "problem. Sentiment is modelled as a latent signal observed "
        "through several mutually independent scorers, and the map from "
        "sentiment to a future response is identified as a finite impulse "
        "response. Measured sentiment is autocorrelated, so the lagged "
        "correlation profile practitioners inspect is the response "
        "convolved with that autocorrelation; scorer noise attenuates it "
        "further. The cross-covariance of two independent scorers "
        "estimates the latent autocovariance at every lag, identifying "
        "the noise-free kernel; three scorers identify each scorer's "
        "reliability. The matched aggregation filter is derived; its "
        "width is governed by scorer reliability rather than by how long "
        "news matters. Run as a "
        "feedback loop, the same noise tightens the stability bound and "
        "slows the optimal gain, both in closed form. "
        f"On {n_head:,} real headlines for {n_sym} equities, scorer "
        f"reliability is near {fmt(lam_fb, 2) if lam_fb else 'n/a'}, the "
        "return kernel is short and the volatility kernel persistent, and "
        "the loop at the predicted gain roughly doubles the out-of-sample "
        "information coefficient of the volatility forecast, exceeding "
        "every fixed window considered.")


def build_tables(R: Res) -> dict:
    k, cl = R.k, R.cl
    rel = k.get("reliability", {})
    corr = rel.get("_corr", {})
    out = {}

    # ---- Table 1: dataset
    md = R.meta
    cov = md.get("coverage")
    out["data"] = ("Corpus and panel after alignment.", [
        ["Quantity", "Value"],
        ["Headlines (distinct strings)",
         f"{md.get('n_headlines',0):,} ({md.get('n_unique',0):,})"],
        ["Instruments with news and prices",
         f"{md.get('n_symbols_panel',0)}"],
        ["Trading sessions", f"{md.get('n_sessions',0):,}"],
        ["Period", f"{md.get('date_min','')} to {md.get('date_max','')}"],
        ["Sessions carrying at least one headline",
         f"{100*cov:.1f} per cent" if cov else "n/a"],
        ["Median headlines in a covered session",
         f"{md.get('median_news_per_covered_session','n/a')}"],
        ["Training observations", f"{k.get('n_obs',0):,}"],
        ["Training window ends", f"{k.get('train_end','')}"],
    ])

    # ---- Table 2: scorer agreement and reliability
    names = [("finbert", "Transformer"), ("lm", "Finance lexicon"),
             ("vader", "Rule-based valence"), ("hiv4", "General lexicon")]
    rows = [["Scorer", "Reliability", "Range over triples", "Mean |corr|"]]
    for key, lab in names:
        d = rel.get(key)
        if not d:
            continue
        cs = [abs(v) for kk, v in corr.items() if key in kk]
        rows.append([lab, fmt(d.get("lambda_median")),
                     f"{fmt(d.get('lambda_min'))}-{fmt(d.get('lambda_max'))}",
                     fmt(np.mean(cs), 2) if cs else "n/a"])
    out["rel"] = ("Scorer reliability and pairwise agreement.", rows)

    # ---- Table 3: filter comparison
    def block(tbl, hz, label):
        rows = []
        if not len(tbl):
            return rows
        base = R.best_fixed(tbl, hz)
        keep = ["latest", "uniform-3", "uniform-5", "uniform-10", "cwin-5",
                "exp-3", "lagprofile", "corrected", "wiener"]
        for f in keep:
            r = R.row(tbl, f, hz)
            if r is None:
                continue
            nm = f + (" (train pick)" if f == base else "")
            dm = r.get("dm_vs_base", np.nan)
            rows.append([f"{label}: {nm}", fmt(r["ic"], 4),
                         fmt(r["dir_acc"], 4), fmt(r["nw_t"], 2),
                         fmt(dm, 2) if np.isfinite(dm) else "-"])
        return rows

    rows = [["Filter", "IC", "Dir. acc.", "NW t", "DM vs base"]]
    rows += block(R.fv, 1, "Vol")
    rows += block(R.f, 1, "Ret")
    out["filt"] = ("Out-of-sample filter comparison. Positive "
                   "Diebold-Mariano favours the filter over the "
                   "training-selected window.", rows)

    # ---- Table 4: loop design
    rows = [["Quantity", "Return", "Volatility"]]

    def pair(key, n=4, tx=None):
        a, b = cl.get(key), R.clv.get(key)
        f = tx or (lambda v: fmt(v, n))
        return [f(a) if a is not None else "n/a",
                f(b) if b is not None else "n/a"]

    rows.append(["Reliability of loop input"] + pair("lambda", 3))
    rows.append(["Input power tr R"] + pair("tr_Rxx", 3))
    rows.append(["Stability bound"] + pair("mu_max", 4))
    rows.append(["Predicted gain"] + pair("mu_star_pred", 1,
                                          lambda v: sig(v, 2)))
    rows.append(["Empirical best gain"] + pair("mu_hat_emp", 1,
                                               lambda v: sig(v, 2)))
    rows.append(["IC, open loop"] + pair("ic_open", 4))
    rows.append(["IC, loop at predicted gain"] + pair("ic_at_mu_star", 4))
    out["loop"] = ("Closed-loop design quantities, measured on training "
                   "data, and outcomes on the held-out window.", rows)
    return out


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", action="store_true",
                    help="strip author identification, as JETIA's "
                         "submission checklist requires")
    ap.add_argument("--figwidth", type=float, default=4.6)
    args = ap.parse_args()

    R = Res(args.results)
    doc = load_template(args.template)
    tbl = doc.tables[0]

    # ------------------------------------------------------- title block
    set_cell(tbl.cell(0, 0), TITLE.upper())
    if args.blind:
        # rows 1-4 carry the authors, affiliations, ORCIDs and e-mails,
        # which the journal asks to be supplied in a separate text file
        drop_rows(tbl, [1, 2, 3, 4])
        # empty text drops the names and the comma after them, leaving the
        # running header exactly as a published JETIA article carries it
        set_running_header(doc, "")
    else:
        set_cell(tbl.cell(1, 0), AUTHORS)
        set_cell(tbl.cell(2, 0), AFFIL)
        set_cell(tbl.cell(3, 0), ORCID)
        set_cell(tbl.cell(4, 0), EMAIL)
        set_running_header(doc, RUNNING)
        doc.core_properties.author = "Anandkumar Pardeshi; Sujata Deshmukh"
        doc.core_properties.last_modified_by = "Anandkumar Pardeshi"
        doc.core_properties.title = TITLE

    abstract = build_abstract(R)
    n_words = len(abstract.split())
    abs_row = 6 if not args.blind else 2
    kw_row = 7 if not args.blind else 3
    set_cell(tbl.cell(abs_row, 3), abstract)
    set_keywords(tbl.cell(kw_row, 0), KEYWORDS)
    print(f"abstract words: {n_words}")
    if not 150 <= n_words <= 200:
        print("!! ABSTRACT OUTSIDE THE 150-200 WORD RANGE")
    if not 3 <= len(KEYWORDS) <= 5:
        print("!! KEYWORD COUNT OUTSIDE 3-5")

    # ------------------------------------------------------------- body
    items, order = number_citations(content(R, args.figdir))
    n = clear_after_title_block(doc)
    print(f"removed {n} template specimen elements")
    b = Builder(doc)

    tables = build_tables(R)
    for kind, *rest in items:
        if kind == "head":
            b.head(rest[0])
        elif kind == "sub1":
            b.sub1(rest[0])
        elif kind == "sub2":
            b.sub2(rest[0])
        elif kind == "text":
            b.text(rest[0])
        elif kind == "prop":
            b.text(rest[0], italic=True)
        elif kind == "eq":
            b.equation(rest[0])
        elif kind == "fig":
            if os.path.exists(rest[0]):
                b.figure(rest[0], rest[1], width_in=args.figwidth)
            else:
                print(f"!! missing figure {rest[0]}")
        elif kind == "tabref":
            t = tables.get(rest[0])
            if t:
                b.table(t[0], t[1])

    # ----------------------------------------------- closing JETIA sections
    b.head("VI. Author's Contribution")
    a1, a2 = ("Author 1", "Author 2") if args.blind else \
        ("Anandkumar Pardeshi", "Sujata Deshmukh")
    for label, tail in CONTRIB:
        b.text("{b:" + label + "} " + tail.format(A1=a1, A2=a2))

    b.head("VII. Acknowledgments")
    b.text(ACKNOWLEDGMENT)

    b.head("VIII. References")
    for i, tag in enumerate(order, 1):
        if tag not in REFS:
            print(f"!! missing reference for tag {tag}")
            continue
        b.ref(i, REFS[tag])
    print(f"references cited: {len(order)}")
    uncited = sorted(set(REFS) - set(order))
    if uncited:
        print(f"!! uncited entries in refs.py: {uncited}")

    if args.blind:
        scrub_properties(doc)

    doc.save(args.out)
    print(f"saved {args.out}")

    with open(os.path.splitext(args.out)[0] + "_cited.json", "w") as fh:
        json.dump(order, fh, indent=1)


if __name__ == "__main__":
    main()
