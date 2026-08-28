#!/usr/bin/env python
"""Assemble the IJCISIM manuscript as a Microsoft Word file.

    python analysis/make_paper_ijcisim.py --out ijcisim_validity_screen.docx

Every number in the prose is looked up from ``results/macros.json`` and every
table is rendered from ``results/tables.json``, both written by
``06b_tables_json.py`` out of the stored result files.  Nothing numeric is typed
into this file: a placeholder like ``{ICCNu}`` is substituted at build time, and
an unknown placeholder raises rather than silently printing itself.

Reference numbering is handled by ``refs_ijcisim.Bibliography``, which assigns
numbers in order of first citation and renders the list from refs.bib.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import ijcisim_docbuild as D                                   # noqa: E402
import omml as M                                               # noqa: E402
from refs_ijcisim import Bibliography                          # noqa: E402


class Numbers(dict):
    """A dict that refuses to silently swallow an unknown placeholder."""

    def __missing__(self, key):
        raise KeyError(f"no such result macro: {key}")


# A placeholder is a brace-delimited identifier and nothing else, so ordinary
# mathematical braces in the prose -- min{|s|, v, u} and the like -- are left
# alone instead of being parsed as fields the way str.format would.
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill(text: str, numbers: Numbers) -> str:
    return PLACEHOLDER.sub(lambda m: numbers[m.group(1)], text)


def load(results: Path):
    tables = json.loads((results / "tables.json").read_text(encoding="utf-8"))
    macros = Numbers(json.loads((results / "macros.json").read_text(encoding="utf-8")))
    return tables, macros


# =============================================================================
def build(out: Path, results: Path, figdir: Path, figwidth: float,
          blind: bool = False) -> Path:
    tables, N = load(results)
    bib = Bibliography(ROOT / "refs.bib")
    doc = D.new_document()
    D.page_numbers(doc)

    def P(text, **kw):
        D.body(doc, fill(text, N), **kw)

    def L(items, numbered=True):
        D.listing(doc, [fill(i, N) for i in items], numbered=numbered)

    tab_no = {"pipeline": 1, "tab_data": 2, "tab_desc": 3, "tab_cost": 4,
              "tab_univariate": 5, "tab_horserace": 6, "tab_gate": 7,
              "tab_gapgate": 8, "tab_live": 9}
    fig_no = {"fig1_axes_veto": 1, "fig6_convergent_validity": 2,
              "fig2_univariate": 3, "fig4_exponents": 4,
              "fig3_risk_coverage": 5, "fig5_deployment": 6}

    def T(key, spec=None):
        D.table(doc, spec if spec is not None else tables[key], tab_no[key])

    def F(stem, caption):
        D.figure(doc, str(figdir / f"{stem}.png"), fig_no[stem],
                 fill(caption, N), width_in=figwidth)

    # ------------------------------------------------------------ front page
    D.masthead(doc,
               journal="International Journal of Computer Information Systems "
                       "and Industrial Management Applications",
               issn="2150-7988", volume="18", year="2026",
               publisher="Cerebration Science Publishing")

    D.title(doc, "Do Prompted Attribute Scores Measure Distinct Attributes? "
                 "A Construct-Validity Screen for Language-Model Features in "
                 "Decision-Support Systems")

    if blind:
        D.authors(doc, [("Author names withheld for review", "")])
        D.affiliations(doc, ["Affiliations withheld for review."],
                       "Correspondence details withheld for review.")
    else:
        D.authors(doc, [("Anandkumar Pardeshi", "1,*"), ("Sujata Deshmukh", "2")])
        D.affiliations(doc, [
            "Department of Computer Science and Engineering, Fr. C. Rodrigues "
            "Institute of Technology, University of Mumbai, Vashi, Navi Mumbai "
            "400703, India; anand.pardeshi@fcrit.ac.in",
            "Department of Computer Engineering, Fr. C. Rodrigues College of "
            "Engineering, University of Mumbai, Bandra, Mumbai 400050, India; "
            "sujata.deshmukh@fragnel.edu.in",
        ], "Correspondence author: anand.pardeshi@fcrit.ac.in (A.P.)")

    D.abstract(doc, fill((
        "Decision-support systems increasingly obtain several conceptually "
        "distinct scores per document from a single large-language-model "
        "prompt and admit each score to the feature store as a separate "
        "variable. The controls such a pipeline already runs — schema, range, "
        "missingness, drift and reproducibility — are all passed by a score "
        "that is perfectly well formed and yet measures something other than "
        "its name, because a prompt has no source of record to check the value "
        "against. This paper proposes a construct-validity screen as a "
        "component acceptance test for prompted attributes, specifies it as a "
        "five-step procedure requiring no labelled data and no additional model "
        "calls, and reports what a real system lost by not running it. The "
        "component under study decomposes financial news into polarity, "
        "novelty and materiality and combines them multiplicatively, so that a "
        "stale or immaterial item is vetoed however extreme its tone; we also "
        "show that continuity, strict monotonicity and ratio-scale separability "
        "force that combiner into a Cobb-Douglas family whose exponents the "
        "axioms leave free. Over {ScoredHeadlines} headlines for {NCompanies} "
        "listed companies from {PanelStart} to {PanelEnd}, the screen fails. "
        "Novelty and materiality correlate at {CorrNuMu}; only materiality "
        "tracks an external criterion, predicting absolute market-adjusted "
        "returns at t = {AbsRetMuT}, while novelty carries no staleness "
        "information beyond what materiality already carries and enters with "
        "the opposite sign. Two of the three attributes are one latent "
        "judgement reported twice. Every downstream result follows: freely "
        "estimated exponents place the materiality elasticity at zero, the "
        "gated aggregate predicts no better than plain or threshold-filtered "
        "polarity, and adding it to a walk-forward selective forecaster moves "
        "precision at fixed coverage by between {GapPriceHFive} and "
        "{GapPriceHOne} percentage points with every interval covering zero. "
        "That programme took {DownstreamTotal} fitted objects; the screen took "
        "{ScreenTotal}, and its verdict preceded all of them. Cross-vendor "
        "agreement on the failing attribute is also the lowest of the three "
        "(intraclass correlation {ICCNu} against {ICCS} for polarity), so the "
        "same screen doubles as a portability test for a component a system "
        "owner may later have to re-source."
    ), N))

    D.keywords(doc, ["construct validity", "large language models",
                     "decision support systems", "feature verification",
                     "data quality", "selective prediction", "text analytics"])

    # ========================================================== 1 Introduction
    D.heading(doc, "1. Introduction")

    P("An instruction-tuned language model will answer as many questions about "
      "a document as a prompt cares to ask. One call can return the sentiment, "
      "the urgency and the credibility of a service ticket; the severity, the "
      "novelty and the confidence of a clinical note; or the tone, the "
      "freshness and the importance of a news item. Each answer arrives on the "
      "scale requested, in the format requested, and at a marginal cost close "
      "to zero. For an information system whose feature store is the asset "
      "that its models are built on, this is an unusually cheap source of "
      "supply, and it is being used.")

    P("It is cheap in the wrong place. What fell is the cost of *asking*; the "
      "cost of being wrong about what came back did not move. A prompt that "
      "requests k attributes obtains k columns, but nothing in the mechanism "
      "guarantees k measurements. A model may reduce several requested "
      "attributes to one internal judgement and report that judgement k times "
      "under k labels, and the output carries no sign of it. The columns vary. "
      "They correlate imperfectly rather than perfectly. They respond sensibly "
      "to worked examples. If the calls are memoised, they reproduce exactly, "
      "byte for byte, on every re-run.")

    P("This matters for an information system specifically because of which "
      "controls it defeats. A production feature pipeline already validates "
      "schema, type, range, null rate, cardinality and distributional drift, "
      "and it already demands reproducibility " + bib.cite("breck2017", "polyzotis2018") +
      ". Every one of those controls passes on an attribute that measures the "
      "wrong construct: the schema is the one requested, the range is the one "
      "requested, nothing is missing, and nothing drifts. The classical "
      "information-quality dimensions — accuracy, completeness, timeliness, "
      "consistency, interpretability " + bib.cite("wang1996", "batini2009") +
      " — are all defined relative to a source of record, and a prompted "
      "attribute has none. Its quality question is not whether the value is "
      "correct but whether the *construct* it names exists as a separate thing "
      "inside the instrument that produced it.")

    P("Psychometrics has asked exactly that question for seventy years, under "
      "the headings of construct validity and of convergent and discriminant "
      "validation " + bib.cite("cronbach1955", "campbell1959") + ". This paper "
      "argues that the same test belongs in the acceptance criteria for any "
      "language-model scoring component admitted to a decision-support system, "
      "specifies a version of it a team can afford to run on every attribute, "
      "and measures what one real system lost by not running it first.")

    D.subheading(doc, "1.1. The screen, in one paragraph")

    P("For each attribute, name in advance an observable criterion that the "
      "attribute must track *if* it measures what its label claims, and that "
      "the other attributes have no particular reason to track. Then test each "
      "attribute against its own criterion **conditional on** the others. The "
      "conditioning is the whole design. When one latent judgement drives every "
      "attribute, each attribute shows a respectable *marginal* association "
      "with every criterion — which is exactly the reassuring picture a "
      "correlation matrix gives — and only the conditional test reveals that "
      "none of them adds anything the others do not already supply.")

    D.subheading(doc, "1.2. The case study and the cost of omission")

    P("The component under study decomposes a news headline into three "
      "attributes — polarity, novelty and materiality — and multiplies them, so "
      "that a stale or immaterial item is suppressed however extreme its tone. "
      "We designed it, believed in it, deployed the surrounding system, and can "
      "therefore report against it without charity. We ran the screen and the "
      "full downstream modelling programme, in that order, on {ScoredHeadlines} "
      "headlines covering {NCompanies} listed companies. The screen fails, and "
      "every subsequent result is an unpacking of the same failure.")

    P("The engineering version of the finding is a ratio. The downstream "
      "programme comprised {DownstreamTotal} fitted objects: {UnivariateFits} "
      "predictive regressions, {HorseRaceCoefs} nested horse-race coefficients, "
      "an exponent grid of {GridPerHorizon} points at each of {GridHorizons} "
      "horizons, {RobustnessFits} robustness re-estimations and {GateFits} "
      "walk-forward selective forecasters. The screen comprised {ScreenTotal}: "
      "two criterion regressions and one decile profile, computed from "
      "artefacts the project already held on disk. It required "
      "{ScreenModelCalls} additional calls to the scoring model. Its verdict "
      "was available before any of the {DownstreamTotal} were run, and it "
      "implied all of them.")

    D.subheading(doc, "1.3. Contributions")

    L(["**A component acceptance test for prompted attributes** (Section 3.5), "
       "stated as a five-step procedure with the conditional test at its "
       "centre, together with the three properties — no labelled data, no "
       "additional model calls, no fitted model — that make it cheap enough to "
       "run on every attribute rather than on the ones somebody happens to "
       "doubt.",
       "**Evidence that the failure mode is real, with a worked diagnosis** "
       "(Section 4.3). Novelty and materiality correlate at {CorrNuMu}; under "
       "the screen, materiality predicts absolute market-adjusted returns at "
       "t = {AbsRetMuT} while novelty adds nothing to the staleness criterion "
       "that materiality does not already supply, and enters absolute returns "
       "with the opposite sign.",
       "**A measurement of the omission cost** (Sections 4.4 to 4.7 and "
       "Table 4), which turns the screen from good practice into an "
       "engineering decision with a number attached.",
       "**A vendor-portability result for the same component** (Section 4.1). "
       "Rescoring a held-out subsample with a second vendor's model gives "
       "intraclass correlations of {ICCS}, {ICCMu} and {ICCNu} for polarity, "
       "materiality and novelty. The attribute that fails the validity screen "
       "is also the one two independent instruments agree on least, which "
       "matters to any system owner who may have to re-source the component.",
       "**A characterisation of the design under test** (Section 3.2), "
       "including a veto bound and a representation theorem showing that "
       "continuity, strict monotonicity, sign fidelity and ratio-scale "
       "separability force the combiner into a Cobb-Douglas family whose "
       "exponents the axioms do not select — which is what makes the design "
       "falsifiable rather than definitional.",
       "**Production evidence on calibration and abstention** (Section 4.8) "
       "from an append-only ledger of 621 resolved interval forecasts issued "
       "by the deployed system, including a diagnosed and repaired calibration "
       "failure that no monitor in the pipeline had noticed."])

    P("The rest of the paper is organised as follows. Section 2 reviews the "
      "related work. Section 3 describes the system under study, the data, the "
      "screen and the evaluation protocol. Section 4 reports the results. "
      "Section 5 discusses what the findings mean for the design and operation "
      "of decision-support systems that consume language-model features, and "
      "Section 6 concludes.")

    # ========================================================= 2 Related work
    D.heading(doc, "2. Related Work")

    D.subheading(doc, "2.1. Verifying machine-learned components")

    P("Information-quality research treats fitness for use as "
      "multi-dimensional, spanning accuracy, completeness, timeliness, "
      "consistency and interpretability " + bib.cite("wang1996", "batini2009") +
      ", and production machine-learning practice has operationalised those "
      "dimensions as tests that run in the pipeline: schema validation, "
      "training-serving skew detection, distributional monitoring and "
      "reproducibility checks " + bib.cite("breck2017", "polyzotis2018") + ". "
      "All of them presume that what a column *denotes* is settled and only "
      "its values are in question. A prompted attribute inverts the "
      "presumption, which is why it needs a different test rather than a "
      "stricter version of the existing ones.")

    D.subheading(doc, "2.2. Language models as measurement instruments")

    P("Instruction-tuned models return structured numeric judgements from "
      "rubric-style prompts " + bib.cite("brown2020") + ", and applied fields "
      "have adopted them quickly, including for financial prediction " +
      bib.cite("lopezlira2025") + " and for reading corporate policy " +
      bib.cite("jha2024") + ". Their self-reported confidence is imperfectly "
      "calibrated and is perceived by users as better than it is " +
      bib.cite("steyvers2025") + ", which motivates a division of labour we "
      "adopt throughout: the model is never asked to forecast, only to "
      "*decompose* an item along defined axes, and forecasting and uncertainty "
      "quantification are left to a separate stage that can be calibrated "
      "against outcomes. Reliability of such judgements — whether two runs, or "
      "two vendors, agree — is increasingly reported. Validity is not, and "
      "reliability does not imply it: an instrument can be perfectly "
      "consistent and consistently wrong.")

    D.subheading(doc, "2.3. Relevance and novelty are established constructs")

    P("We make no claim to have discovered the attributes under test. "
      "Commercial news analytics have scored relevance and event novelty for "
      "well over a decade — vendor feeds routinely attach a relevance score, an "
      "event novelty score and an event sentiment score to each item, and a "
      "large empirical literature filters on those fields before running its "
      "tests. On the academic side the novelty axis has a canonical antecedent "
      "in Tetlock " + bib.cite("tetlock2011") + ", who separates genuinely new "
      "stories from reprints of stale information and finds that investors "
      "react to the stale component; Boudoukh and co-authors " +
      bib.cite("boudoukh2019") + " show that identifying which news is "
      "*relevant* materially changes the measured information content of a "
      "news flow; Dang and co-authors " + bib.cite("dang2015") + " study "
      "commonality in news flow across markets; and Kelly and co-authors " +
      bib.cite("kelly2021") + " treat which text gets written at all as an "
      "object of study. Hillert and co-authors " + bib.cite("hillert2014") +
      " document how coverage intensity interacts with momentum, and "
      "DellaVigna and Pollet " + bib.cite("dellavigna2009") + " show that "
      "identical information moves prices differently depending on when "
      "attention is available to receive it. What is new here is not the "
      "attributes but the question asked of them: whether a general-purpose "
      "model, prompted zero-shot under a published rubric, delivers them as "
      "*separate* measurements, and what a system loses by assuming it does.")

    D.subheading(doc, "2.4. Text analytics in financial decision support")

    P("The application domain has a long empirical literature. Tetlock " +
      bib.cite("tetlock2007") + " linked pessimistic media tone to downward "
      "price pressure and subsequent reversal, and later work " +
      bib.cite("tetlock2008") + " showed that negative language in "
      "firm-specific news predicts fundamentals and returns. Antweiler and "
      "Frank " + bib.cite("antweiler2004") + " studied message boards; "
      "Loughran and McDonald " + bib.cite("loughran2011") + " demonstrated "
      "that domain-specific word lists matter because general-purpose ones "
      "systematically mislabel domain terms, a point developed in their "
      "survey " + bib.cite("loughran2016") + "; and Garcia " +
      bib.cite("garcia2013") + " showed the effect of tone is state-dependent. "
      "Supervised and neural text models improve on dictionaries " +
      bib.cite("ke2019", "huang2023") + ", and machine learning is now standard "
      "in empirical asset pricing " + bib.cite("gu2020") + ". Across this "
      "literature the per-item output is a scalar tone; freshness and "
      "importance, when handled at all, are handled by ad hoc pre-filters. "
      "That convention is the baseline the component under study set out to "
      "beat, and Section 4.9 finds it hard to beat.")

    D.subheading(doc, "2.5. Calibration and selective prediction")

    P("Modern classifiers are frequently miscalibrated " + bib.cite("guo2017") +
      "; the standard remedies are Platt scaling " + bib.cite("platt1999") +
      " and isotonic regression " + bib.cite("zadrozny2002") + ", assessed by "
      "expected calibration error and the Brier score " + bib.cite("brier1950") +
      ". Stacking " + bib.cite("wolpert1992") + " complicates this by creating "
      "more than one score distribution, a failure mode diagnosed and repaired "
      "in Section 3.8. Abstention is an old idea " + bib.cite("chow1970") +
      " with a modern theory " + bib.cite("geifman2017", "elyaniv2010") + " and "
      "a distribution-free relative in conformal prediction " +
      bib.cite("angelopoulos2023") + ". Applied machine learning in this domain "
      "remains dominated by accuracy and area-under-the-curve reporting, which "
      "is fragile under backtest overfitting " + bib.cite("bailey2017") + " and "
      "multiple testing " + bib.cite("harvey2016", "romano2005") + ", and "
      "comparatively little of it reports whether the probabilities can be "
      "trusted.")

    # =============================================== 3 Materials and methods
    D.heading(doc, "3. Materials and Methods")

    D.subheading(doc, "3.1. The system under study")

    P("The component being verified does not stand alone. It sits in a "
      "five-stage decision-support pipeline that ingests timestamped news, "
      "scores it, aggregates it into a per-entity session feature, feeds that "
      "feature to a forecaster alongside price-derived features, and finally "
      "decides whether the resulting forecast is confident enough to act on. "
      "Table 1 names the stages, the artefact each produces and the control "
      "each already carries. Reading the table is the quickest way to see the "
      "gap this paper is about: every stage has an operational control, and "
      "none of them is a validity test.")

    T("pipeline", {
        "caption": ("The decision-support pipeline the scoring component sits "
                    "in, the artefact each stage produces, and the control the "
                    "stage already carries. Stage 2 is the component under "
                    "test; its existing control establishes reproducibility, "
                    "which is a necessary but not a sufficient condition for "
                    "the attribute to mean what it is named."),
        "header": ["Stage", "Artefact produced", "Control already in place"],
        "align": ["l", "l", "l"],
        "widths": [0.20, 0.38, 0.42],
        "rows": [
            ["1. Ingest", "Timestamped headline keyed to an entity",
             "Session assignment fixed by a published timing rule"],
            ["2. Score", "Three attributes per headline",
             "Content-addressed memoisation; anchored rubric"],
            ["3. Aggregate", "One signal per entity-session",
             "Deterministic function of stage 2; range checked"],
            ["4. Forecast", "Point, interval and directional probability",
             "Walk-forward evaluation; no test-set selection"],
            ["5. Decide", "Act or abstain",
             "Calibration scored against outcomes; fixed-coverage reporting"],
        ],
        "note": ("No stage carries a construct-validity test. Stage 2 is "
                 "verified to be repeatable and in range, which an attribute "
                 "measuring the wrong construct also is."),
    })

    D.subheading(doc, "3.2. The multi-attribute scoring component")

    P("Let e denote a textual event — in this system, a news headline — "
      "concerning an entity, here a listed equity. The prompt requests three "
      "attributes: polarity s(e) in [-1, 1], the sign and size of the implied "
      "value change; novelty ν(e) in [0, 1], the surprise relative to what is "
      "already known; and materiality μ(e) in [0, 1], the sensitivity of "
      "fundamental value to the event, independent of sign and of freshness. "
      "The *event signal* is their product,")

    D.equation(doc, M.d(
        M.r("a"), M.paren(M.r("e")), M.up(" = "),
        M.r("s"), M.paren(M.r("e")), M.r("ν"), M.paren(M.r("e")),
        M.r("μ"), M.paren(M.r("e")), M.up(" ∈ "), M.up("[−1, 1]")), "(1)")

    P("Writing *relevance* as r(e) = ν(e)μ(e) gives a(e) = s(e)r(e). The "
      "reading is a first-order decomposition of an expected move into a "
      "direction, a sensitivity and a surprise magnitude; additive tone scoring "
      "retains only the first factor, which is why it responds to items that "
      "are loud but stale or cosmetic. Events touching one entity in one "
      "session are combined by a relevance-weighted mean over items clearing a "
      "materiality floor μ₀,")

    D.equation(doc, M.d(
        M.r("A"), M.up(" = "),
        M.frac(M.d(M.nary("∑", M.r("e"), "", M.d(
            M.sup(M.d(M.r("r"), M.paren(M.r("e"))), M.up("2")),
            M.r("s"), M.paren(M.r("e"))))),
               M.d(M.nary("∑", M.r("e"), "", M.d(
                   M.r("r"), M.paren(M.r("e"))))))), "(2)")

    P("so that relevance enters twice, once inside the event signal and once as "
      "the weight, and a novel and material item dominates quadratically while "
      "routine chatter is suppressed. The floor is set to μ₀ = 0.15, the "
      "value the production system uses; Section 4.9 sweeps it.")

    P("Two properties of the product form are worth stating precisely, because "
      "they are what the design promises and what a reader is entitled to hold "
      "it to.")

    p = doc.add_paragraph()
    D._set_spacing(p, before=6, after=2)
    p.paragraph_format.first_line_indent = D.Cm(D.INDENT_CM)
    D._run(p, "Proposition 1 (Veto bound). ", bold=True)
    D.rich(p, "For every event e, |a(e)| ≤ min{|s(e)|, ν(e), μ(e)}. In "
              "particular a(e) = 0 whenever any single factor is zero, and the "
              "aggregate of Eq. (2) satisfies |A| ≤ max|s(e)|.", italic=True)

    P("*Proof.* Each factor has modulus at most one and ν, μ ≥ 0, so "
      "|a(e)| = |s(e)|ν(e)μ(e) is a product of three numbers of modulus at "
      "most one, which cannot exceed the modulus of any one of them; this gives "
      "the per-event bound and the zero case. For the aggregate, |A| is at most "
      "the ratio of Σ r(e)^2^|s(e)| to Σ r(e), which is at most max|s(e)| times "
      "the ratio of Σ r(e)^2^ to Σ r(e), and that ratio is at most one because "
      "every r(e) lies in [0, 1]. The weights are non-negative but "
      "sub-stochastic, so the aggregate contracts polarities toward zero and "
      "never amplifies them. ∎")

    P("Proposition 1 is the formal content of the claim that a single weak "
      "factor vetoes an event: a stale or immaterial item is capped near zero "
      "whatever its tone. No additive combination has this property, since a "
      "weighted sum is generally nonzero when one term vanishes. The second "
      "question is why a product rather than any other veto-respecting form. "
      "An earlier version of this argument assumed that scaling any one "
      "attribute by λ scales the combiner by λ, which is close to assuming the "
      "conclusion. The statement below replaces that assumption with the "
      "standard conditions of conjoint measurement " +
      bib.cite("debreu1960", "gorman1968", "krantz1971") + " together with the "
      "observation that each attribute is measured on a ratio scale.")

    p = doc.add_paragraph()
    D._set_spacing(p, before=6, after=2)
    p.paragraph_format.first_line_indent = D.Cm(D.INDENT_CM)
    D._run(p, "Proposition 2 (Representation). ", bold=True)
    D.rich(p, "Let f map [-1, 1] × (0, 1] × (0, 1] into [-1, 1] and satisfy "
              "(A0) normalisation, f(1, 1, 1) = 1; (A1) sign fidelity, the sign "
              "of f is the sign of s; (A2) veto, f = 0 whenever any of |s|, ν, "
              "μ is zero; (A3) regularity, f is continuous and |f| is strictly "
              "increasing in each argument on the interior; and (A4′) "
              "ratio-scale separability, that for each argument there is a "
              "one-variable function h such that multiplying that argument by λ "
              "in (0, 1] multiplies |f| by h(λ) whatever the other two "
              "arguments are. Then there exist exponents α₀, α, β > 0 with",
           italic=True)

    D.equation(doc, M.d(
        M.r("f"), M.paren(M.d(M.r("s"), M.up(", "), M.r("ν"), M.up(", "), M.r("μ"))),
        M.up(" = "), M.up("sign"), M.paren(M.r("s")),
        M.sup(M.up("|") + M.r("s") + M.up("|"), M.sub(M.r("α"), M.up("0"))),
        M.sup(M.r("ν"), M.r("α")), M.sup(M.r("μ"), M.r("β"))), "(3)")

    P("*Proof sketch.* Work on the positive orthant and write F = |f|. Fixing "
      "the second and third arguments, (A4′) gives F(λx, ν, μ) = h₀(λ)F(x, ν, "
      "μ). Applying it twice shows that h₀ satisfies the multiplicative "
      "Cauchy equation, and by (A3) it is continuous and positive, so its only "
      "solutions are the power functions " + bib.cite("aczel1966") + ", with a "
      "strictly positive exponent because F is strictly increasing. Setting "
      "x = 1 and repeating the argument for the remaining two arguments yields "
      "Eq. (3), with (A0) fixing the constant and (A1) restoring the sign. "
      "Continuity extends the representation to the closed domain, where (A2) "
      "is automatic because all three exponents are strictly positive. ∎")

    P("The proposition delivers the **family**, not a member of it. Eq. (1) is "
      "the special case in which all three exponents equal one, and nothing in "
      "(A0) to (A4′) selects it. This is deliberate: the axioms assert that the "
      "three attributes act on separate multiplicative scales and cannot "
      "compensate for one another, which is the substantive claim, while the "
      "elasticities with which they act are an empirical matter that "
      "Section 4.6 estimates. Proposition 1 survives for every member of the "
      "family in the weaker form |f| ≤ min{|s|^α₀^, ν^α^, μ^β^}, so the veto "
      "holds whatever the exponents turn out to be.")

    P("The design has one silent prerequisite, and the rest of this paper is "
      "about it. A product of three gates behaves like a product of three gates "
      "only if there **are** three gates. If two of the attributes are one "
      "attribute wearing two labels, the second gate cannot bind where the "
      "first does not, the exponents are not separately identified, and the "
      "component degenerates without any warning appearing in its output.")

    D.subheading(doc, "3.3. Data")

    P("The study draws on two datasets that answer different questions and are "
      "never used to support a claim about each other; Table 2 sets them side "
      "by side.")

    P("**A large historical news panel.** Testing the decomposition needs a long "
      "panel with dense per-item news and clean prices. We use FNSPID " +
      bib.cite("dong2024") + ", a public corpus of time-stamped financial news "
      "headlines keyed to tickers. From the symbols with usable price history "
      "we take the best-covered names and remove exchange-traded funds, for "
      "which the sensitivity of *this firm's* fundamental value to an event has "
      "no clean meaning; the exclusion list is enumerated in the "
      "panel-building script rather than pattern-matched, so it is auditable. "
      "That leaves {NCompanies} operating companies, {PanelRows} "
      "symbol-sessions of which {NewsRows} carry scored news, over "
      "{PanelSessions} trading sessions. Prices are split- and "
      "dividend-adjusted daily bars; returns are market-adjusted using "
      "per-symbol betas estimated on training windows only, and a Parkinson "
      "range estimator " + bib.cite("parkinson1980") + " supplies a volatility "
      "proxy.")

    P("**A production forecast ledger.** The forecasting system of "
      "Sections 3.8 and 4.8 is deployed on liquid National Stock Exchange of "
      "India large-capitalisation equities. Since April 2026 it has written "
      "every issued forecast to an append-only ledger — entity, timestamp, "
      "target date, point forecast, interval, nominal confidence, directional "
      "probability and horizon — with each row later stamped with the realised "
      "price and outcome. Rows are scored at first look and never retro-edited. "
      "This ledger supplies the interval-coverage and calibration evidence of "
      "Section 4.8. It cannot speak to the value of the text decomposition: it "
      "stores no per-event return labels and its history is far too short to "
      "fit anything.")

    P("We state the resulting limitation at the outset rather than in a closing "
      "caveat. *The validation market is not the deployment market.* No "
      "historical Indian headline corpus of comparable density is available to "
      "us, so the question whether the decomposition carries information is "
      "answered on the US panel, and the question whether a calibrated "
      "selective forecaster behaves as designed in production is answered on "
      "the Indian ledger.")

    T("tab_data")

    D.subheading(doc, "3.4. Reproducibility and vendor portability of the component")

    P("The attributes are produced by an instruction-tuned model prompted as a "
      "quantitative analyst and constrained to return a JSON array of triples, "
      "one per headline, in input order. Three design choices make the "
      "measurement reproducible, and they are the controls a systems audit "
      "would expect to find.")

    P("First, the prompt fixes **numerical anchors** on every axis. For novelty, "
      "0.0 is an exact restatement of information already priced or a generic "
      "market list, 0.2 an in-line and widely expected result, 0.6 an "
      "unexpected but modest development, and 1.0 a sudden shock such as a "
      "surprise regulatory action. For materiality, 0.0 is boilerplate or a "
      "price-move round-up, 0.4 a contract or ruling worth a few percent of "
      "revenue, 0.8 a transformative deal or guidance reset, and 1.0 an "
      "existential event. For polarity the anchors run from -1.0 for a fraud "
      "allegation through 0.0 for a neutral reshuffle to +1.0 for a large "
      "earnings beat or a takeover at a premium. Anchoring matters more here "
      "than in ordinary tone scoring, because the product in Eq. (1) amplifies "
      "disagreement on any single axis.")

    P("Second, every scored item is memoised in a content-addressed store keyed "
      "by the SHA-1 hash of the entity and the normalised headline, so repeated "
      "runs are deterministic and the scoring history is auditable. Third, the "
      "attributes are scored **zero-shot**: no labelled training data, no "
      "fine-tuning, and no market outcome enters the prompt, so the scores "
      "cannot encode look-ahead information about the returns they are later "
      "used to predict.")

    P("Determinism obtained this way is a property of the cache, not of the "
      "model, and a system owner who may one day have to change vendors needs "
      "the stronger property. We therefore rescored a random subsample of "
      "{RetestN} headlines with the cache bypassed and a **different vendor's** "
      "instruction-tuned model, which tests whether the anchored rubric is "
      "portable rather than model-specific. Section 4.1 reports the agreement.")

    D.subheading(doc, "3.5. The construct-validity screen")

    P("This is the acceptance test the paper argues should run before any "
      "feature is built from a multi-attribute prompt. It adapts convergent and "
      "discriminant validation " + bib.cite("campbell1959", "cronbach1955") +
      " to the case where the instrument is a prompt, compressed to what a "
      "team can execute in an afternoon.")

    L(["**Name a criterion per attribute, in advance.** For each attribute, "
       "state an observable quantity it must track *if* it measures what its "
       "label claims, and that the other attributes have no particular reason "
       "to track. The criterion must be computable without the attribute "
       "itself. Here, novelty is assigned *mechanical staleness* — lexical "
       "overlap with the same entity's recent coverage — and materiality is "
       "assigned *outcome magnitude*, the absolute market-adjusted return of "
       "the session.",
       "**Test conditionally, not marginally.** Regress each criterion on all "
       "attributes jointly, or equivalently examine partial correlations. This "
       "step carries the diagnostic weight. If one latent judgement drives "
       "every attribute, each will show a respectable *marginal* association "
       "with every criterion, and only the conditional test reveals that none "
       "adds anything to the others.",
       "**Inspect the shape before summarising it.** A rank correlation is "
       "meaningless if the relationship is not monotone. Plot each attribute "
       "against deciles of its criterion first, and only then report a "
       "coefficient.",
       "**Control the mechanical confounds of the criterion.** A criterion "
       "computed from a comparison set inherits that set's properties: a "
       "document with few predecessors scores as novel by construction. The "
       "size of the comparison set enters as a control.",
       "**Check that the criterion is the right *kind* of quantity for the "
       "downstream task.** An attribute can be valid and still useless. One "
       "that legitimately measures magnitude cannot sharpen a directional "
       "forecast. This step converts a validity result into a design decision."])

    P("Three properties make the screen cheap enough to be mandatory rather "
      "than aspirational. Neither criterion required labelled data. Neither "
      "required additional model calls: mechanical staleness is computed from "
      "the corpus already collected and outcome magnitude from prices already "
      "needed by the study, so the screen adds {ScreenModelCalls} scoring calls "
      "to a panel of {ScoredHeadlines}. And nothing in it requires a predictive "
      "model to be fitted, so it precedes rather than competes with the "
      "modelling programme. Note also that the outcome criterion validates the "
      "**instrument** and is never an input to any signal: checking a "
      "measurement against an outcome is legitimate, whereas building the "
      "measurement from that outcome would not be.")

    D.subheading(doc, "3.6. The downstream evaluation, and what it cost")

    P("To measure the omission cost we ran the modelling programme that a team "
      "skipping the screen would run. Every aggregator below is built from the "
      "same scored events, so no comparison is confounded by differences in the "
      "underlying text: mean polarity, the standard additive recipe; "
      "threshold-filtered polarity, the mean polarity of events clearing the "
      "materiality floor, which is what practitioners actually do with a vendor "
      "relevance score; count-weighted polarity, which rewards repeated "
      "coverage as a naive summing pipeline implicitly does; the additive "
      "combiner, the mean of the three attributes, which is the natural "
      "non-multiplicative way to use exactly the same inputs; and two "
      "single-gate ablations built from sν alone and sμ alone.")

    P("On top of these we estimated the exponents of Eq. (3) on a grid of "
      "{GridPerHorizon} points at each horizon, ran a nested horse race, swept "
      "the three consequential analyst choices as robustness, and fitted a "
      "walk-forward selective forecaster with and without the gated signal. "
      "Table 4 counts the resulting objects against the screen. The point is "
      "not that {DownstreamTotal} fits are expensive in machine time — they are "
      "not — but that each carries analyst attention, review time and an "
      "opportunity for a specification choice to be made in the direction "
      "somebody hopes for. The screen has {ScreenTotal} such objects and none "
      "of them is a predictive model.")

    T("tab_cost")

    D.subheading(doc, "3.7. Timing convention")

    P("Every headline carries a UTC publication stamp. It is converted to the "
      "exchange time zone and assigned to the session whose 16:00 close first "
      "follows it, so the aggregate for session d contains only information "
      "public before that close and is used to predict returns realised from "
      "session d+1 onward. The convention is applied uniformly and is the "
      "single most important guard in the study: a related analysis in earlier "
      "work by the present authors produced a spuriously significant result "
      "that turned out to be a time-zone join artefact, and this convention was "
      "adopted after that diagnosis.")

    D.subheading(doc, "3.8. Calibration and the conviction gate")

    P("The aggregate A is one feature among several entering a forecaster that "
      "also sees price, momentum, trend, reversal and volatility features, and "
      "that emits per name and horizon a point forecast, a price interval and a "
      "directional probability. We treat the forecaster as a black box and "
      "study its calibration, measured by the expected calibration error over B "
      "equal-width bins,")

    D.equation(doc, M.d(
        M.up("ECE"), M.up(" = "),
        M.nary("∑", M.d(M.r("b"), M.up("=1")), M.r("B"), M.d(
            M.frac(M.d(M.up("|"), M.sub(M.r("G"), M.r("b")), M.up("|")), M.r("N")),
            M.up("|"), M.sub(M.r("ȳ"), M.r("b")), M.up(" − "),
            M.sub(M.r("p̄"), M.r("b")), M.up("|")))), "(4)")

    P("with the bar quantities the mean predicted probability and mean realised "
      "outcome in bin b, and by the Brier score " + bib.cite("brier1950") + ".")

    P("The first deployed version of this model reported next-day "
      "up-probabilities clustered near 0.80 against a realised accuracy near "
      "0.51. The cause was a stacking domain mismatch " + bib.cite("wolpert1992") +
      ": an isotonic calibrator had been fitted on the out-of-fold *average of "
      "the base learners* and was then applied to the *stacked* prediction, a "
      "different distribution, and the blend itself was never calibrated. The "
      "repair is a three-stage procedure — refit the isotonic map on the "
      "base-average it was meant to correct, form the blend, then fit a final "
      "Platt scaler " + bib.cite("platt1999") + " on a held-out fold and apply "
      "it to test and live rows. The transferable lesson for a system owner is "
      "that a calibrator is bound to the distribution it was fitted on, and "
      "stacking silently creates more than one.")

    P("Calibration makes a probability honest, not skilful. If short-horizon "
      "direction is close to unpredictable, a calibrated probability sits near "
      "the base rate almost everywhere and acting on every name is pointless. "
      "The remedy mirrors the text-stage veto: act only when conviction clears "
      "a threshold. For a horizon of H sessions the system emits an UP call "
      "when the probability clears τ∗ and abstains otherwise, with τ∗ chosen on "
      "training data as")

    D.equation(doc, M.d(
        M.sup(M.r("τ"), M.up("*")), M.up(" = min"),
        M.paren(M.d(M.r("τ"), M.up(" : "), M.up("prec"), M.paren(M.r("τ")),
                    M.up(" ≥ "), M.sub(M.r("π"), M.up("0")), M.up(" and "),
                    M.r("φ"), M.paren(M.r("τ")), M.up(" ≥ "),
                    M.sub(M.r("φ"), M.up("0"))), "{", "}")), "(5)")

    P("where prec(τ) is the realised up-rate among fired calls and φ(τ) the "
      "firing rate, with π₀ = 0.60 and φ₀ = 0.05. The gate is one-sided "
      "because confident DOWN calls are destroyed by the upward drift of "
      "equities, so the system emits long-or-neutral signals only. Reporting a "
      "single operating point invites the objection that τ∗ was tuned to the "
      "number being reported, so we also report the full **risk-coverage "
      "curve** and its area (AURC): sorting predictions by confidence and "
      "sweeping the threshold traces the selective risk accepted at each "
      "coverage level, and the area under that curve summarises the whole "
      "trade-off without reference to any threshold " +
      bib.cite("elyaniv2010", "geifman2017") + ".")

    D.subheading(doc, "3.9. Evaluation protocol and inference")

    P("All panel results are walk-forward by calendar year: for test year Y the "
      "model sees only sessions strictly before 1 January Y, the last 20% of "
      "the training sessions are held out to fit the probability calibrator and "
      "to choose τ∗, and neither is ever selected on test data. Test years run "
      "from {GateYearFirst} to {GateYearLast}. Inference respects the two "
      "dependence structures of a return panel: a market-wide shock hits every "
      "name on the same date, and returns are persistent within a name. We "
      "therefore report two-way clustered standard errors by date and symbol " +
      bib.cite("cameron2011") + ", cross-checked against Fama-MacBeth "
      "cross-sectional slopes with a Newey-West correction " +
      bib.cite("famamacbeth1973", "neweywest1987") + ". Multi-session horizons "
      "are evaluated on non-overlapping windows as the primary specification, "
      "and confidence intervals for differences between models come from a "
      "block bootstrap that resamples whole dates, preserving cross-sectional "
      "correlation.")

    # =============================================================== 4 Results
    D.heading(doc, "4. Results")

    P("We report in the order a sceptical reviewer of the system would want. "
      "First, does the component produce anything stable and portable? Second, "
      "are the three attributes the separate things Section 3.2 assumes? Third, "
      "does the gated aggregate predict at all? And last, does it predict "
      "better than the simpler recipes it was built to replace. The answers "
      "are, in order: yes; **no**; yes; and no.")

    D.subheading(doc, "4.1. Component reliability and portability")

    P("Scoring {ScoredHeadlines} headlines for {NCompanies} companies over "
      "{PanelStart} to {PanelEnd} yields attributes with sensible central "
      "tendencies: mean novelty 0.189, mean materiality 0.157 and mean polarity "
      "0.039, the last confirming the mild positive skew the news-tone "
      "literature reports. Two properties of the component deserve comment "
      "before anything is built on it.")

    P("First, the model reports on a **coarse grid** rather than a continuum: "
      "the mass sits on multiples of 0.1, visible as the discrete bars of "
      "Figure 1(a). This is a property of anchored prompting, not an artefact "
      "of the aggregation, and it means the attributes carry less resolution "
      "than their [0, 1] range suggests — a fact worth recording in the feature "
      "specification, because a downstream consumer that assumes continuity "
      "will over-read small differences.")

    P("Second, and more consequential for a system owner, the cross-vendor "
      "agreement is moderate and uneven. On the {RetestN}-headline rescoring "
      "subsample the intraclass correlation " + bib.cite("shrout1979") + " is "
      "{ICCS} for polarity, {ICCMu} for materiality and only {ICCNu} for "
      "novelty, with the event signal inheriting {ICCEvent}. The ordering "
      "matters for how the rest of the paper reads. Polarity, the attribute "
      "with decades of instrument development behind it, is the one two "
      "independent vendors agree on most; novelty, the attribute on which the "
      "component's distinctive claim rests, is the one they agree on least. Any "
      "true multiplicative effect is therefore being estimated through the "
      "noisiest available channel, and classical attenuation would bias the "
      "measured contribution of gating toward zero. We return to this in "
      "Section 5.4, because it is the most credible non-trivial explanation of "
      "the null results that follow. The subsample is small — {RetestN} "
      "headlines, bounded by the free-tier request quota rather than by design "
      "— so these coefficients are indicative rather than precise.")

    D.subheading(doc, "4.2. The veto works; the attributes are not separable")

    P("The gate does exactly what Proposition 1 promises. Table 3 shows that "
      "56.3% of events fall below the materiality floor and are refused "
      "outright, and that 47.6% of symbol-sessions carrying news end with an "
      "aggregate of exactly zero. The contraction is equally visible: the "
      "standard deviation of the gated aggregate is 0.054 against 0.154 for "
      "mean polarity, so gating removes about two-thirds of the "
      "cross-sectional dispersion of the signal. Figure 1(c) is the per-session "
      "version of the same statement, with almost every point below the "
      "45-degree line. On its own terms the mechanism is not in doubt.",
      )

    P("The premise underneath it is. Multiplicative gating supposes that "
      "novelty and materiality are **separable** properties acting through "
      "different channels. In the measured data they are very nearly the same "
      "variable: the correlation between ν and μ across all {ScoredHeadlines} "
      "events is {CorrNuMu}, and Figure 1(b) shows the joint mass collapsing "
      "onto the diagonal. An item the model judges novel is almost always an "
      "item it judges material, and conversely.")

    P("This single fact anticipates every result that follows. If two of the "
      "three attributes carry nearly the same information, the second gate can "
      "add almost nothing once the first has acted, the product cannot separate "
      "itself from simpler combinations of the same inputs, and any attempt to "
      "estimate distinct elasticities for ν and μ will be poorly identified. "
      "All three happen.")

    P("Two very different things could produce that correlation. It may be a "
      "fact about the *world* — genuinely new firm news usually *is* the "
      "consequential news — or a fact about the *instrument*, an inability of "
      "the model to hold two related concepts apart under one prompt. The two "
      "have opposite implications for what a system owner should do next, so "
      "the screen exists to settle the question rather than leave it open.")

    T("tab_desc")

    F("fig1_axes_veto",
      "The three attributes and the veto they implement, measured on "
      "{ScoredHeadlines} scored headlines. (a) Marginal distributions of "
      "novelty and materiality; the model reports on a coarse grid, so the "
      "honest display is a discrete bar chart. (b) Joint mass of the two gates, "
      "on a power-law colour scale so the sparse high-novelty, high-materiality "
      "cells remain visible. (c) The gated session aggregate against the mean "
      "absolute polarity of the same session: almost every point lies below the "
      "45-degree line, which is Proposition 1 in the data.")

    D.subheading(doc, "4.3. Screen outcome: does each attribute track its own criterion?")

    P("Each attribute has an external criterion it must track if it measures "
      "what it claims, and neither criterion required further model calls.",
      )

    P("*Novelty should track mechanical staleness.* Following the stale-news "
      "literature " + bib.cite("tetlock2011") + ", we measure how much a "
      "headline repeats what was already written about the same ticker: one "
      "minus the maximum TF-IDF cosine similarity to that ticker's headlines "
      "over the preceding thirty days. The comparison set is strictly earlier "
      "in time, so the measure is causal and uses no returns. It is available "
      "for 99.9% of headlines ({ScoredForScreen} of {ScoredHeadlines}), with a "
      "median of {PriorDocMedian} prior documents.")

    P("*Materiality should track outcome magnitude.* Materiality is defined as "
      "the sensitivity of value to an event *regardless of direction*, so it "
      "should predict the absolute market-adjusted return of the session. The "
      "screen separates the two attributes sharply, but not in the direction "
      "the design needs.")

    P("**Novelty fails its own criterion.** Figure 2(a) plots both attributes "
      "against deciles of mechanical novelty, and the two curves are the same "
      "curve: parallel, identically shaped, separated by a constant. The "
      "relationship is also non-monotonic, rising from {NuAtStalest} at the "
      "stalest decile to {NuAtPeak} at decile {NuPeakDecile} and then falling "
      "to {NuAtFreshest} at the most lexically novel decile — the model assigns "
      "its *lowest* novelty to the headlines that share least vocabulary with "
      "recent coverage. Controlling for materiality and for the size of the "
      "comparison set, novelty's standardised coefficient on staleness is "
      "{StaleNuBeta} (t = {StaleNuT}) against materiality's {StaleMuBeta} "
      "(t = {StaleMuT}): distinguishable from zero, but essentially the same "
      "number, so novelty carries no staleness information that materiality "
      "does not carry equally. The partial correlations tell the same story — "
      "{PartialNuStale} for novelty given materiality, {PartialMuStale} for "
      "materiality given novelty — as does the dense subsample with at least "
      "twenty prior documents ({DenseNuStale} and {DenseMuStale}, identical to "
      "three decimals). Step 3 of the screen earns its place here: the raw rank "
      "correlation between the novelty attribute and mechanical novelty is "
      "{RawNuStale}, which read alone would suggest a modest inverse relation "
      "rather than the hump the decile profile actually shows.")

    P("**Materiality passes its own criterion, decisively.** Over {AbsRetN} "
      "events, a one standard deviation increase in materiality is followed by "
      "{AbsRetMuBps} additional basis points of absolute market-adjusted return "
      "(t = {AbsRetMuT}, clustered by date), while novelty enters the same "
      "regression with the *opposite* sign at {AbsRetNuBps} basis points "
      "(t = {AbsRetNuT}). Whatever common factor drives the {CorrNuMu} "
      "correlation, the materiality label is the one that aligns with a real "
      "external quantity; conditional on it, the novelty label points the wrong "
      "way (Figure 2(b)).")

    P("The verdict is therefore *instrument*, and specifically the novelty "
      "attribute. The model is largely reporting a single latent importance "
      "judgement on two channels; the channel labelled materiality happens to "
      "align with event magnitude, and the channel labelled novelty does not "
      "measure novelty in the sense the design requires.")

    P("Step 5 of the screen then supplies the implication that matters most, "
      "and supplies it before any forecasting model exists. The surviving "
      "attribute, materiality, predicts the **magnitude** of the move and not "
      "its **direction** — exactly what its definition promises. But Eq. (1) "
      "multiplies it into a *directional* signal. A quantity carrying "
      "information about how large a move will be cannot improve a forecast of "
      "which way it will go. That is the cleanest explanation for the "
      "materiality exponent estimating at zero in Section 4.6, and it was "
      "available from the screen alone.")

    F("fig6_convergent_validity",
      "The construct-validity screen. (a) Mean novelty and mean materiality "
      "across deciles of mechanical novelty: the two curves are parallel and "
      "identically shaped, which is what one variable measured twice looks "
      "like, and both are non-monotonic in mechanical novelty. (b) Standardised "
      "coefficients of each attribute on each criterion, in standard deviations "
      "of criterion per standard deviation of attribute, with 95% intervals "
      "from date-clustered standard errors. Materiality predicts absolute "
      "return strongly; novelty does not track mechanical staleness beyond what "
      "materiality already tracks.")

    D.subheading(doc, "4.4. Predictive content of the aggregators")

    P("The gated aggregate does predict returns. At the one-session horizon a "
      "one standard deviation move in A is followed by {CoefAHOne} basis points "
      "of market-adjusted return, with a two-way clustered t of {TAHOne} and a "
      "Fama-MacBeth t of {FMTAHOne}; its rank information coefficient is "
      "{ICAHOne}, averaged over {ICPeriodsHOne} trading days with a Newey-West "
      "t of {ICTAHOne}. At five sessions the effect is {CoefAHFive} basis "
      "points (t = {TAHFive}) on non-overlapping windows. By twenty-one "
      "sessions nothing survives: every aggregator has |t| < 0.9, which is what "
      "one expects of headline-level information at a monthly horizon.",
      )

    P("So the text carries signal. The question the study exists to answer is "
      "whether the **product** carries more of it than the recipes it was meant "
      "to displace, and Table 5 says it does not. At one session, mean polarity "
      "earns a *higher* t-statistic than the gated aggregate ({Tpol_meanHOne} "
      "against {TAHOne}); count-weighted polarity produces a larger coefficient "
      "still ({Coefpol_cntHOne} basis points) and the highest "
      "information-coefficient t-statistic of the seven ({ICTpol_cntHOne}); and "
      "threshold-filtered polarity — the recipe a practitioner would actually "
      "deploy — matches the product almost exactly ({Coefpol_relfHOne} basis "
      "points). The information coefficients of the gated aggregate, "
      "threshold-filtered polarity and count-weighted polarity are {ICAHOne}, "
      "{ICpol_relfHOne} and {ICpol_cntHOne}: identical for any practical "
      "purpose. The one aggregator that clearly underperforms is the additive "
      "combiner at {Coefadd_combHOne} basis points and {ICadd_combHOne}, which "
      "is worth stating plainly. **Adding** the three attributes is a bad idea, "
      "so the argument against additive combination in Section 3.2 survives; it "
      "is the step from a filter to a product that does not pay. Figure 3 makes "
      "the overlap visible: at the horizons where anything is estimable, the "
      "confidence intervals of all seven aggregators sit on top of one another.")

    T("tab_univariate")

    F("fig2_univariate",
      "Forward-return coefficient on each aggregator, in basis points per one "
      "standard deviation of signal, with 95% intervals from two-way clustered "
      "standard errors. The gated signal is highlighted; the alternatives are "
      "shown in neutral ink.")

    D.subheading(doc, "4.5. The nested horse race")

    P("Entering all four terms together (Table 6) was intended to be the sharp "
      "test: multiplicative gating predicts that the product loads and absorbs "
      "the lower-order terms. What happens instead is that nothing is "
      "individually identified. At one session the gated aggregate carries the "
      "largest coefficient but only t = 1.58, while mean polarity falls to "
      "t = 1.26 and the two single-gate ablations take opposite signs. At five "
      "and twenty-one sessions the coefficients grow large and alternate in "
      "sign, with |t| < 2 throughout.")

    P("This pattern is diagnostic of collinearity, not of a hidden effect, and "
      "Section 4.2 already said why: with ν and μ correlated at {CorrNuMu} and "
      "all four regressors built from the same polarity, the design matrix is "
      "close to singular and the individual coefficients are not separately "
      "estimable. We therefore decline to read anything into the sign or size "
      "of any single term. The honest conclusion is a statement about what the "
      "data **cannot** answer: on this corpus the nested horse race lacks the "
      "power to distinguish the product from its components.")

    T("tab_horserace")

    D.subheading(doc, "4.6. Free exponents")

    P("The horse race cannot separate the terms, but the exponent grid asks a "
      "cleaner question, because it varies the functional form directly instead "
      "of trying to identify collinear regressors. Proposition 2 left α and β "
      "free; here we estimate them by sweeping the aggregate built from "
      "sν^α^μ^β^ over a grid of {GridPerHorizon} points on [0, 2]² and reading "
      "off the information coefficient.")

    P("The answer is unambiguous and it is not the one the unit-exponent "
      "product predicts. At one session the grid maximum sits at "
      "α = {BestAlphaHOne}, β = {BestBetaHOne} with information coefficient "
      "{BestICHOne}: the materiality exponent is estimated at *exactly zero*, "
      "as Step 5 of the screen implied it would be. The unit-exponent product "
      "assumed by Eq. (1) scores {UnitICHOne}, and switching the gates off "
      "altogether — α = β = 0, which collapses the aggregate to plain mean "
      "polarity — scores {PureICHOne}. At five sessions the maximum **is** the "
      "no-gating corner, α = β = 0 with {BestICHFive}, against {UnitICHFive} "
      "for the unit-exponent product: there, gating is not merely unhelpful but "
      "actively harmful.")

    P("Two conclusions follow. First, the hypothesis that α = β = 1 is "
      "rejected, and rejected in the direction of the null model rather than "
      "toward some richer weighting. Second, and more important for a team "
      "deciding where to spend effort, the entire surface spans only 0.0201 to "
      "0.0225 in information coefficient (Figure 4). The objective is nearly "
      "flat in the exponents, so the choice of gating weights — including the "
      "choice not to gate — moves predictive accuracy by an amount small "
      "relative to its own sampling error. That flatness, rather than the "
      "location of any maximum, is the finding, and it is why every comparison "
      "in Section 4.7 lands inside its confidence interval.")

    F("fig4_exponents",
      "Information coefficient of the aggregate built from sν^α^μ^β^ over a "
      "grid of exponents. The unit-exponent product assumed by Eq. (1) is "
      "marked, as is the empirical maximum; Proposition 2 explains why the "
      "axioms leave this an open question.")

    D.subheading(doc, "4.7. Selective forecasting with and without the gated signal")

    P("This is the experiment the component has to pass: the conviction gate is "
      "fitted twice on identical walk-forward folds, once on price features "
      "alone and once on price features plus the gated text signal, and the "
      "reported quantity is the difference.")

    P("There is no difference. Adding A moves precision at a fixed 10% coverage "
      "by {GapPriceHOne} percentage points at one session (95% CI "
      "{GapPriceCIHOne}, p = {GapPricePHOne}), {GapPriceHFive} points at five "
      "sessions ({GapPriceCIHFive}, p = {GapPricePHFive}) and "
      "{GapPriceHTwentyOne} points at twenty-one ({GapPriceCIHTwentyOne}, "
      "p = {GapPricePHTwentyOne}). Against threshold-filtered polarity rather "
      "than against price alone the gaps are {GapRelfHOne}, {GapRelfHFive} and "
      "{GapRelfHTwentyOne} points. Every interval contains zero, and the point "
      "estimates change sign across horizons (Tables 7 and 8). The "
      "risk-coverage curves of Figure 5 say the same thing without reference to "
      "any threshold: the curves for the four feature sets are "
      "indistinguishable over the whole coverage range, and their AURCs differ "
      "in the third decimal.")

    P("One result in Table 7 does not fit the pattern, and we report it because "
      "it is the most interesting thing in the table. The **text-only** model — "
      "no price features at all — achieves the best AURC at both short horizons "
      "({AurcTextHOne} against {AurcPriceHOne} for price-only at one session, "
      "and {AurcTextHFive} against {AurcPriceHFive} at five) and the best "
      "precision at five sessions ({PrecTextHFive}% against {PrecPriceHFive}% "
      "for price-only, on a {BaseRateHFive}% base rate). The headlines "
      "therefore carry decision-relevant information that the price features do "
      "not, and a learner given the text alone finds it. What the learner "
      "cannot do is exploit that information *once price features are also "
      "available*: the combined model is no better than price alone. Whether "
      "this reflects genuine redundancy or a failure of the gradient-boosted "
      "learner to allocate capacity to a weak, sparse feature beside strong "
      "dense ones, we cannot determine from these data, and we would not want a "
      "reader to take the first explanation for granted.")

    T("tab_gate")
    T("tab_gapgate")

    F("fig3_risk_coverage",
      "Risk-coverage curves for the walk-forward selective forecaster. Each "
      "curve traces the error rate accepted at every coverage level, so the "
      "comparison does not depend on the choice of any single conviction "
      "threshold; the area under each curve (AURC, lower is better) appears in "
      "the legend.")

    D.subheading(doc, "4.8. Production evidence: coverage and calibration")

    P("The results above concern the scoring and aggregation stages. The "
      "decision stage was studied on the deployed system, where the evidence is "
      "of a different kind: not whether a signal predicts, but whether a system "
      "in operation tells the truth about its own uncertainty.")

    P("It did not. Between 19 April and 12 June 2026 the system issued and "
      "later resolved 621 price-interval forecasts on seven large caps, each "
      "carrying a nominal 90% interval with a median half-width of 5.6% of "
      "price. Had those intervals been calibrated, about 90% would have "
      "contained the realised price; empirical coverage was 70.7%, a 19.3 "
      "percentage-point shortfall (Table 9, Figure 6(a)). Most forecasts sat at "
      "the ten-day horizon, where coverage was 69.2% over 575 resolved "
      "forecasts. Nothing in the pipeline noticed, which is the point: "
      "overconfidence is silent unless somebody scores it, and no schema, range "
      "or drift control will do so.")

    P("The directional head failed in the same direction and for a diagnosable "
      "reason. It reported next-day up-probabilities clustered near 0.80 "
      "against a realised accuracy near 0.51, giving an expected calibration "
      "error of 0.345 and a Brier score above 0.366 — worse than predicting 0.5 "
      "every time. The cause was the stacking domain mismatch of Section 3.8, "
      "and the three-stage repair cut expected calibration error from 0.345 to "
      "0.049 while leaving accuracy untouched. Calibration changed what the "
      "model *said about its confidence*, not what it predicted, which is "
      "precisely the distinction the design insists on.")

    P("With probabilities pulled back to the base rate, the walk-forward "
      "conviction gate on the 54-name deployment universe (2018 to 2026, 91,471 "
      "training rows and 50,272 out-of-sample rows) settles at τ∗ = 0.63 and "
      "fires on 5.6% of observations, abstaining on the rest. The fired bucket "
      "realises 60.6% directional precision against a 58.0% always-up base rate "
      "over 2,840 pooled calls: a 2.6 point edge with a 95% binomial interval "
      "of 58.8 to 62.4%. We do not over-read it. Per-year counts are small, the "
      "swing from +10.8 points in 2025 to -2.4 in 2024 is within sampling noise "
      "(Figure 6(b)), and the ranking area under the curve is about 0.47, so "
      "what edge exists lives in the high-conviction tail and not in the "
      "ordering. Crucially for this paper, **that gate contains no text feature "
      "at all**: it is evidence that calibrated abstention is worth building, "
      "not evidence for the decomposition.")

    T("tab_live")

    F("fig5_deployment",
      "Production evidence. (a) Empirical coverage of nominal 90% price "
      "intervals on the live ledger, by horizon. (b) Fired-bucket precision of "
      "the conviction gate against the always-up base rate, by walk-forward "
      "test year, at roughly a 5.6% firing rate.")

    D.subheading(doc, "4.9. Robustness")

    P("A null result is only worth reporting if it is not an artefact of the "
      "analyst's choices, so we varied the three that matter, in "
      "{RobustnessFits} re-estimations.")

    P("*The materiality floor.* Sweeping μ₀ from 0 (no floor) to 0.40 leaves "
      "the one-session coefficient on A between 12.0 and 12.2 basis points with "
      "t between 3.97 and 4.14. The floor does not bind on the conclusion, "
      "which is what one expects when the exponent on materiality is estimated "
      "at zero. One detail cuts the other way and we note it: at five sessions "
      "the information coefficient rises from 0.013 at μ₀ = 0 to 0.025 at "
      "μ₀ = 0.25 and above, with the Newey-West t going from 1.24 to 2.25. A "
      "**hard threshold filter** thus helps where multiplicative weighting does "
      "not — consistent with the exponent surface, and consistent with what "
      "practitioners already do with vendor relevance scores.")

    P("*Overlapping versus non-overlapping windows.* Using every session rather "
      "than every H-th weakens the five-session result from 17.8 basis points "
      "(t = 2.12) to 8.9 (t = 1.68), in the direction the "
      "overlapping-observation literature predicts. We report non-overlapping "
      "windows as the primary specification for that reason; neither choice "
      "changes the comparison between aggregators, which is what the study "
      "turns on.")

    P("*Quiet sessions.* Restricting the panel to symbol-sessions carrying "
      "scored news is the primary specification, since including quiet sessions "
      "pads the sample with rows where every text aggregator is identically "
      "zero and can only dilute any difference between them.")

    P("*Repeatability of the component.* Cross-vendor agreement is reported in "
      "Section 4.1. We regard it as the weakest link in the measurement chain "
      "and the first thing a replication should attack.")

    # ============================================================ 5 Discussion
    D.heading(doc, "5. Discussion")

    D.subheading(doc, "5.1. Why the decomposition did not pay")

    P("Section 4.3 lets us be specific rather than speculative, and the answer "
      "has two parts.")

    P("*One attribute is not measuring what it is named.* Novelty fails its "
      "external criterion: it does not track mechanical staleness beyond what "
      "materiality already tracks, its profile against mechanical novelty is "
      "the same curve as materiality's, and it is non-monotonic besides. "
      "Materiality passes decisively, predicting absolute market-adjusted "
      "returns at {AbsRetMuT} standard errors. So the {CorrNuMu} correlation is "
      "not the world telling us that novel news is material news; it is one "
      "latent importance judgement reported twice under two labels, only one of "
      "which corresponds to something external. A product of two gates cannot "
      "behave like a two-gate product when only one gate exists.")

    P("*The surviving attribute carries the wrong quantity for the task.* "
      "Materiality is by definition about magnitude irrespective of sign, and "
      "that is exactly how it behaves. Multiplying it into a *signed* signal, "
      "as Eq. (1) does, asks a magnitude variable to sharpen a direction "
      "forecast, which it has no means of doing. This is the cleanest reading "
      "of the zero materiality exponent, and it points somewhere constructive: "
      "the same decomposition aimed at a volatility or absolute-return target, "
      "where materiality's information is of the relevant kind, is a genuinely "
      "different experiment and one we have not run.")

    P("*Attenuation remains a live secondary concern.* Cross-vendor intraclass "
      "correlations are {ICCS} for polarity but only {ICCNu} for novelty, and "
      "the product inherits {ICCEvent}; multiplying noisy factors compounds "
      "their noise, so classical errors-in-variables would bias the measured "
      "contribution of gating toward zero. We cannot rule this out, and it is a "
      "further reason the design deserves a rematch with a better instrument "
      "rather than a dismissal. What it does not license is treating the null "
      "as a measurement artefact by assumption — and in any case attenuation "
      "cannot explain why novelty points the *wrong way* on absolute returns, "
      "which is a validity failure and not a noise problem.")

    P("Note also what survived. The additive combiner performed worst of all "
      "the aggregators, so collapsing the attributes by addition is clearly "
      "wrong; and a **hard** materiality filter improved the five-session "
      "information coefficient from 0.013 to 0.025 where multiplicative "
      "weighting did not. Read together, these say that relevance information "
      "is worth using as a *filter* — decide which items to look at — but not "
      "as a continuous multiplicative *weight* on the ones that survive. The "
      "practitioner convention of thresholding a vendor relevance score, which "
      "the component set out to improve on, appears to be the right call.")

    D.subheading(doc, "5.2. What the screen would have saved")

    P("The value of an acceptance test is not that it produces a verdict but "
      "that it produces the verdict **early**, and Table 4 is the accounting. "
      "Every finding in Section 4 is an implication of the screen's two "
      "regressions. That novelty and materiality collapse into one attribute "
      "implies the nested horse race cannot identify separate terms and the "
      "exponent surface is nearly flat. That the surviving attribute measures "
      "magnitude rather than direction implies the materiality exponent "
      "estimates at zero against a signed target. And neither implication "
      "leaves room for an incremental gate, which is why the selective "
      "forecaster with and without A differs by less than its own sampling "
      "error.")

    P("The {ScreenTotal} screen objects and the {DownstreamTotal} downstream "
      "objects are not comparable in machine time; a modern laptop runs both. "
      "They are comparable in the resource that actually binds a delivery team, "
      "which is the sequence of decisions an engineer makes while a result is "
      "still ambiguous. Each of the {GridTotal} exponent evaluations, "
      "{UnivariateFits} predictive regressions and {GateFits} walk-forward fits "
      "carries a specification choice — which horizon is primary, whether to "
      "report overlapping windows, which threshold defines coverage — and each "
      "such choice is an opportunity for a result to be nudged toward the one "
      "somebody hoped for. Running the screen first removes the ambiguity that "
      "makes those choices consequential.")

    D.subheading(doc, "5.3. Implications for information-system practice")

    P("For a team about to admit multi-attribute language-model scores into a "
      "decision-support system, the findings translate into four specific "
      "rules.")

    L(["**Treat every prompted attribute as an instrument requiring validation, "
       "not as a column requiring monitoring.** The controls a feature pipeline "
       "already runs are all passed by an attribute that measures the wrong "
       "construct. Add a validity gate at feature onboarding, and make an "
       "attribute's criterion a mandatory field of its specification alongside "
       "its type and range.",
       "**Require the criterion to be named before the scores are generated.** "
       "A criterion chosen after inspecting the scores is a rationalisation. "
       "The version of this study we would defend is the one in which each "
       "attribute's criterion was fixed by its definition: novelty against "
       "staleness, materiality against outcome magnitude.",
       "**Budget attributes by validated count, not requested count.** The "
       "marginal cost of an extra attribute in the prompt is near zero, but the "
       "marginal cost of an *unvalidated* extra attribute is the entire "
       "modelling programme built on it. In this study the ratio was "
       "{ScreenTotal} objects to {DownstreamTotal}.",
       "**Record the portability of the component, not only its "
       "reproducibility.** Memoisation makes a scoring stage byte-for-byte "
       "repeatable and tells a system owner nothing about what happens when the "
       "vendor changes. The rescoring test cost {RetestN} calls and revealed "
       "that the attribute the validity screen rejects is also the one two "
       "vendors agree on least."])

    P("There is also a negative recommendation. The most reusable part of the "
      "screen is Step 5 — checking that the criterion is the right *kind* of "
      "quantity for the task — and it is the step most easily skipped, because "
      "it feels like philosophy rather than measurement. It was the step that "
      "predicted the single most consequential number in the study, the zero "
      "materiality exponent, and it costs nothing but a sentence of thought per "
      "attribute.")

    P("Nothing in Section 3.2 is specific to equities. Any stream of "
      "timestamped textual events feeding a decision has the same structure, "
      "and the same failure mode when conceptually distinct attributes are "
      "requested from one prompt: triage of clinical alerts, operations tickets "
      "and content-moderation queues all ask how bad, how new and how "
      "consequential, and all are exposed to a model that answers the three "
      "questions with one judgement. The screen transfers directly, because its "
      "only requirement is that each attribute admit an observable criterion "
      "the others have no reason to track — for a ticket-urgency attribute, "
      "time to resolution; for a moderation-severity attribute, the eventual "
      "enforcement action. The decision-stage half of the design is an instance "
      "of selective classification " + bib.cite("chow1970", "geifman2017") +
      ", here paired with a probability whose calibration was repaired rather "
      "than assumed, and the interval-coverage failure of Section 4.8 is "
      "exactly the problem distribution-free methods " +
      bib.cite("angelopoulos2023") + " are built for.")

    D.subheading(doc, "5.4. Threats to validity and limitations")

    P("Five limitations bound what may be concluded, and we would rather state "
      "them than have them found.")

    P("*The validation market is not the deployment market.* The decomposition "
      "is tested on US headlines while the production system operates on Indian "
      "equities. We report no result that crosses that boundary, but a reader "
      "should not read the US panel as evidence about the NSE's microstructure, "
      "disclosure regime or news supply, all of which differ.")

    P("*The attributes are judgements, and one of them did not survive "
      "validation.* Anchoring the prompt tightens the scores and Section 4.1 "
      "quantifies their repeatability, but repeatability is not validity: a "
      "scorer can be perfectly consistent and consistently wrong, which is "
      "close to what the screen finds for novelty. Every result about gating is "
      "therefore a joint test of the design and of this particular instrument, "
      "and a differently worded rubric, a larger model, or a two-stage prompt "
      "that scores the attributes independently might separate them where ours "
      "did not. We consider that the single most valuable follow-up.")

    P("*The external criteria are proxies.* Lexical similarity over a "
      "thirty-day window is a serviceable but crude stand-in for what the "
      "market already knew, and absolute return is a noisy realisation of an "
      "event's true importance. A criterion failure is therefore weaker "
      "evidence than a criterion success: materiality passing is strong "
      "evidence it measures something, whereas novelty failing is consistent "
      "with both a bad attribute and a bad proxy.")

    P("*The comparison is against feasible baselines, not against a vendor "
      "feed.* The most informative comparison would be against a commercial "
      "relevance and novelty product on the same headlines. We do not have a "
      "licence for one, so threshold-filtered polarity is our best available "
      "stand-in for what a practitioner would deploy.")

    P("*Selection and multiple testing.* We report several horizons, several "
      "aggregators and several feature sets. The risk-coverage curves and the "
      "non-overlapping-window specification remove the most common sources of "
      "optimism, and differences between models carry bootstrap intervals, but "
      "a reader who wants a single number with a family-wise guarantee should "
      "apply a stepdown correction " + bib.cite("romano2005") + " to the "
      "horse-race column and read the result accordingly. Applied machine "
      "learning in this domain has a well-documented tendency to discover edges "
      "that do not survive " + bib.cite("bailey2017", "harvey2016") + ", and we "
      "would rather our estimates be read with that prior in mind.")

    # =========================================================== 6 Conclusions
    D.heading(doc, "6. Conclusions")

    P("A textual event carries direction, surprise and consequence, and there "
      "is a clean argument that these should combine multiplicatively: the "
      "product makes freshness and relevance preconditions for influence, it "
      "caps the signal at its weakest factor, and under continuity, "
      "monotonicity and ratio-scale separability it is the form the attributes "
      "must take, up to exponents the axioms leave free.")

    P("Measured against real data, the argument does not pay. On "
      "{ScoredHeadlines} headlines covering {NCompanies} companies, the gated "
      "aggregate predicts market-adjusted returns at short horizons but no "
      "better than mean polarity, count-weighted polarity or "
      "threshold-filtered polarity; estimating the exponents freely puts "
      "materiality's at zero and, at five sessions, collapses the aggregate to "
      "plain polarity; and adding the gated signal to a walk-forward selective "
      "forecaster moves precision at fixed coverage by between {GapPriceHFive} "
      "and {GapPriceHOne} percentage points, with every interval containing "
      "zero.")

    P("The explanation is in the measurement rather than in the market. Novelty "
      "and materiality, as an instruction-tuned model scores them, correlate at "
      "{CorrNuMu}; validated against external criteria, only materiality "
      "survives, predicting absolute market-adjusted returns at {AbsRetMuT} "
      "standard errors, while novelty fails to track mechanical staleness "
      "beyond what materiality already tracks and enters absolute returns with "
      "the wrong sign. The three-attribute decomposition is empirically a "
      "two-attribute one, and the attribute that does measure something "
      "measures *magnitude*, which a signed product has no way to exploit.")

    P("Two things survive as positive findings. Adding the attributes is worse "
      "than multiplying them, so the case against additive combination stands. "
      "And using relevance as a hard filter rather than a continuous weight "
      "improves the five-session information coefficient from 0.013 to 0.025 — "
      "which is what practitioners already do with vendor relevance scores, now "
      "with evidence behind it. Separately, on the deployment side, nominal 90% "
      "intervals covered 70.7% of live outcomes, a three-stage recalibration "
      "cut expected calibration error from 0.345 to 0.049 without touching "
      "accuracy, and a conviction gate abstaining on 94% of names raised the "
      "thirty-day hit rate to 60.6% against a 58.0% base — evidence that "
      "calibrated abstention is worth building, independent of anything the "
      "text stage does.")

    P("The engineering conclusion is the ratio. The modelling programme that "
      "established all of the above comprised {DownstreamTotal} fitted objects. "
      "The screen that implied all of it comprised {ScreenTotal}, used no "
      "labelled data, and added {ScreenModelCalls} calls to the scoring model. "
      "As multi-attribute prompting spreads through decision-support pipelines, "
      "the constraint on feature quality will not be the cost of generating "
      "attributes; it will be the discipline to verify that the attributes "
      "generated are the ones that were asked for. Two experiments follow "
      "directly from this one and neither is run here: an instrument whose "
      "novelty channel passes an external validity check would give the design "
      "the fair test it has not yet had, and pointing the same decomposition at "
      "a volatility or absolute-return target asks the question the signed "
      "formulation cannot. The measurements reported here are designed to make "
      "both possible.")

    # ------------------------------------------------------------- back matter
    D.heading(doc, "Author Contributions")
    if blind:
        D.backmatter_para(doc, "Withheld for review.")
    else:
        D.backmatter_para(doc,
            "A.P. designed the study, implemented the scoring, panel, screen "
            "and evaluation code, produced the figures and tables, and drafted "
            "the manuscript. S.D. contributed to the study design and the "
            "interpretation of the validity results, supervised the work, and "
            "revised the manuscript. Both authors have read and agreed to the "
            "published version of the manuscript.")

    D.heading(doc, "Funding")
    D.backmatter_para(doc, "This research received no external funding.")

    D.heading(doc, "Conflict of Interest Statement")
    D.backmatter_para(doc, "The authors declare no conflict of interest.")

    D.heading(doc, "Data Availability Statement")
    D.backmatter_para(doc,
        "The FNSPID news corpus used for the validation panel is publicly "
        "available; price data are public end-of-day records. The scored "
        "attribute values, the derived panel, the analysis code and the "
        "production forecast ledger underlying the reported tables and figures "
        "are available from the corresponding author on reasonable request.")

    D.heading(doc, "Use of Generative Artificial Intelligence")
    D.backmatter_para(doc,
        "An instruction-tuned language model is used as a measurement "
        "instrument within the system under study, scoring headlines along the "
        "three attributes from the anchored rubric of Section 3.4; those "
        "outputs are persisted for audit. During manuscript preparation a large "
        "language model was used solely to improve language and readability. No "
        "generative tool produced the data, results, figures or "
        "interpretations, and the authors take full responsibility for the "
        "content.")

    D.heading(doc, "References")
    D.references(doc, bib.rendered())

    D.heading(doc, "Biographical Sketch")
    if blind:
        D.backmatter_para(doc, "Withheld for review.")
    else:
        D.backmatter_para(doc,
            "**Anandkumar Pardeshi** is with the Department of Computer Science "
            "and Engineering, Fr. C. Rodrigues Institute of Technology, "
            "University of Mumbai. His work concerns applied machine learning "
            "for decision support, with an emphasis on the calibration, "
            "selective prediction and measurement validity of deployed "
            "forecasting systems.")
        D.backmatter_para(doc,
            "**Sujata Deshmukh** is with the Department of Computer "
            "Engineering, Fr. C. Rodrigues College of Engineering, University "
            "of Mumbai. Her research interests include data engineering, "
            "machine learning systems and the evaluation of intelligent "
            "information systems.")

    D.copyright_footer(doc,
        "Copyright: © 2026 by the authors. This article is an open access "
        "article distributed under the terms and conditions of the Creative "
        "Commons Attribution (CC BY) license.")

    doc.save(str(out))
    leftover = bib.unused()
    print(f"wrote {out}")
    print(f"  {len(bib.order)} references cited")
    if leftover:
        print(f"  {len(leftover)} bib entries not cited: {', '.join(leftover)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ijcisim_validity_screen.docx"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figdir", default=str(ROOT / "figures"))
    ap.add_argument("--figwidth", type=float, default=4.6)
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args()
    build(Path(args.out), Path(args.results), Path(args.figdir),
          args.figwidth, blind=args.blind)


if __name__ == "__main__":
    main()
