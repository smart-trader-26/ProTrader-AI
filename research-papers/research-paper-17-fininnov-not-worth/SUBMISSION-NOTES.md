# Financial Innovation — NOT SENT (folder renamed `-not-worth`, 2026-08-20)

> **This manuscript was never submitted to Financial Innovation and is not going
> to be.** FI is a hard sell for a null result, and the study was re-homed
> instead to two venues that suit it better:
>
> | Folder | Venue | Frame |
> |---|---|---|
> | `../research-paper-19-ijcisim` | IJCISIM (Word, APC USD 1,500) | a component acceptance test inside a decision-support pipeline |
> | `../research-paper-20-dsm` | Data Science and Management (free, Q1) | a pre-deployment screen and the cost of omitting it |
>
> Both are genuine rewrites, not reformats: different framing, related work,
> structure and emphasis, on the same real results. This folder stays because it
> owns the analysis pipeline and the 130 MB scoring cache both of them read from
> — **do not delete it**, and do not delete `cache/`.

# Financial Innovation submission notes — Multiplicative Information Gating

**Journal:** Financial Innovation (Springer / SWUFE) — <https://jfin-swufe.springeropen.com>
Q1, IF 7.2, CiteScore 12.9. Scope explicitly names AI / GenAI / big data in finance.
APC waived for up to 110 articles/year from 2026 (choose the SWUFE sponsor option at
submission). First-decision median ~18 days; submission→acceptance median ~456 days.

**Article type:** Research article
**Predecessors:** `../rejected-research-paper-8-tubitak` (TJEECS, returned on a page-count
technicality) and `../research-paper-16-ijase-not-sent` (IJASE, judged Q4 — not sent).

## Why this is a new paper, not a reformat

The two earlier versions were framework papers. An honest adversarial review of the
IJASE draft scored it 5/10 and named three defects. This version fixes all three, and
the first fix required new experimental work rather than new prose.

| # | Defect in the earlier drafts | What was done |
|---|---|---|
| 1 | §4.4's headline 60.6% came from a conviction gate built on momentum/trend/reversal/volatility — **the text feature `A` was not in the model**. Any competent referee kills the paper on this. | The MIG claim is now carried by a new experiment on a large real news panel in which `A` **is** a feature of the model that produces the headline number, with an explicit ablation against the same model without it. |
| 2 | The "to our knowledge no prior work…" novelty sentence was exposed to RavenPack's Relevance / Event Novelty Score / Event Sentiment Score triple. | Sentence deleted. §2.2 concedes the prior art explicitly (RavenPack-style vendor feeds; Tetlock 2011 for the novelty axis; Boudoukh et al. 2019 for relevance) and states three narrower residual claims. |
| 3 | Proposition 2's axiom (A4), per-argument homogeneity, was close to assuming the conclusion. | Replaced by (A4′) ratio-scale separability plus the standard conjoint-measurement conditions. The theorem now delivers the **Cobb–Douglas family** `s·ν^α·μ^β`, and Remark 1 states plainly that the axioms do *not* pin the exponents. §5 estimates α and β and tests `H₀: α=β=1` instead of assuming it. |

## What the experiment actually found — the paper is a negative result

Fix 1 was run and **multiplicative gating lost**. The paper reports that as the
finding rather than hiding it, and the title is now a question.

| Test | Result |
|---|---|
| Univariate prediction, H=1 | `A` works: 12.2 bps per SD, two-way clustered *t* = 4.14, IC 0.023 over 2,548 days |
| …versus the baselines | Mean polarity *t* = 5.52, count-weighted 14.1 bps and best IC. `A` does not win |
| Nested horse race | Nothing separately identified — collinear by construction |
| Free exponents, H=1 | Best at α = 0.75, **β = 0.00**. Unit exponents IC 0.0210; no gating at all 0.0209 |
| Free exponents, H=5 | Best is **α = β = 0** (plain polarity), IC 0.0156, beating unit exponents' 0.0126 |
| Whole exponent surface | Spans only 0.0201–0.0225 — nearly flat in the gating weights |
| Selective gate with vs without `A` | −1.23 to +0.37 pp at fixed 10% coverage; every 95% CI contains zero |

**The diagnosis, and the most quotable line in the paper: novelty and materiality
correlate at 0.871.** Two of the three axes are empirically one variable, which is
why the second gate cannot add anything, why the horse race is unidentified, and
why the materiality exponent estimates at zero.

Three things survive and are reported as positives:

1. The **additive** combiner is the worst aggregator of all (3.3 bps, IC 0.010), so
   the argument against adding the axes stands even though multiplying them fails.
2. A **hard relevance filter** raises the H=5 IC from 0.013 (μ₀ = 0) to 0.025
   (μ₀ ≥ 0.25), where multiplicative weighting does not — evidence for exactly the
   thresholding convention the paper set out to improve on.
3. The **text-only** model beats the price-only model on AURC at both short
   horizons while the combined model beats neither, which is an open architecture
   question rather than evidence of redundancy.

### The convergent-validity result answers "world or instrument?"

Added after the first pass, at no API cost, from the corpus already on disk. Each
axis is given an external criterion: novelty against **mechanical staleness**
(one minus max TF-IDF cosine similarity to that ticker's prior 30 days, strictly
causal), materiality against **absolute market-adjusted return** (used to validate
the instrument only, never as a feature).

- **Novelty fails.** Its profile across deciles of mechanical novelty is the *same
  curve* as materiality's, offset by a constant, and non-monotonic besides (0.167
  at the stalest decile, 0.230 at the peak, 0.093 at the freshest). Partial
  correlations given the other axis are −0.014 for both; on the dense subsample the
  two are identical to three decimals (−0.155 vs −0.157).
- **Materiality passes decisively:** +54.3 bps of |return| per SD, *t* = 12.70,
  while novelty enters with the *opposite* sign (−26.3 bps, *t* = −8.69).
- **The insight that ties the paper together:** materiality validly measures
  *magnitude*, but Eq. (1) multiplies it into a *signed* signal. A magnitude
  variable cannot sharpen a direction forecast — which is the cleanest explanation
  for the zero materiality exponent, and it implies the decomposition deserves a
  rematch against a volatility or |return| target.

Analysis trap worth remembering: the raw Spearman (−0.16) is misleading because the
relationship is non-monotonic, and `n_prior_30d` is a mechanical confound (few prior
documents ⇒ high staleness by construction, corr −0.209). Plot the decile profile
before quoting any rank correlation here.

Attenuation survives as a secondary caveat — cross-model ICC is 0.779 for polarity
but only **0.565 for novelty**, product 0.601 — but it cannot explain novelty
pointing the *wrong way* on |return|, which is a validity failure, not noise.

## The framing decision a reader will challenge first

**The validation market is not the deployment market**, and the paper says so in
§4.1 rather than burying it in a limitations paragraph.

There is no historical Indian headline corpus of usable density available to us —
the deployment's own store holds 23 scored headlines — so `A` cannot be
reconstructed over the NSE walk-forward at all. The decomposition is therefore
tested on FNSPID (US, Benzinga-sourced), and the NSE live ledger is used only for
what it can support: interval coverage and calibration of a deployed system.
Neither dataset is used to make a claim about the other market.

## Study universe

- FNSPID symbols with usable price history: 237
- Target was the top 60 by headline coverage minus 8 exchange-traded funds
  (EWJ, SLV, GXC, PGJ, QQQ, FXP, EWI, YINN) = 52 operating companies. The scoring
  run stopped at the free-tier daily cap (500 requests/day on
  `gemini-3.1-flash-lite`, ~125k headlines), so the shipped panel is **35 companies
  and 134,037 headlines**. Because the scorer is symbol-complete ordered, every one
  of those 35 is *fully* scored — no session is aggregated from a partial set.
  Topping up to 52 is one command on a later day and changes nothing structural.
- ETFs are excluded because "the sensitivity of *this firm's* fundamental value to
  the event" is undefined for a country or commodity fund. The list is spelled out
  in `analysis/02_build_mig_panel.py` rather than pattern-matched, so it is auditable.
- 52 names is deliberately close to the 54-name NSE deployment universe.

## Reproduction

`bash run_all.sh` runs everything in order. Only step 1 touches the network.

| Step | Script | Output |
|---|---|---|
| 1 | `01_score_axes.py` | `cache/scores.db` — the three axes per (symbol, headline) |
| 1b | `01_score_axes.py --retest` | second pass on a subsample, for reliability |
| 2 | `02_build_mig_panel.py` | `cache/mig_panel.parquet`, `cache/events.parquet` |
| 3 | `03_horse_race.py --exponents` | `results/univariate.csv`, `horse_race.csv`, `exponents.json` |
| 4 | `04_selective_gate.py` | `results/gate_summary_newsrows.json`, `gate_by_year_*.csv` |
| 5 | `07_reliability.py` | `results/reliability.json` |
| 6 | `08_robustness.py` | `results/robustness.csv` |
| 6b | `09_convergent_validity.py` | `results/convergent_validity.json`, `cache/staleness.parquet` — **no network** |
| 7 | `05_figures.py`, `06_tables.py` | `figures/*.pdf` (6), `tables/*.tex` (6), `macros.tex` |
| 8 | `verify_manuscript.py` | placeholders, macro coverage, float presence, log health |

**Nothing in the manuscript's tables is typed by hand.** `06_tables.py` writes the
table fragments the `.tex` inputs, and every number quoted in the prose is a
`\newcommand` in `macros.tex`. If an analysis moves, the text moves with it.

## Method safeguards worth naming in a response to referees

- **Timing.** Headlines are assigned to the session whose 16:00 ET close first
  follows their UTC stamp, so session *d*'s aggregate is strictly pre-close and
  predicts returns from *d+1*. This convention is inherited from an audited
  pipeline; an earlier study of ours produced a spuriously significant result that
  turned out to be a time-zone join artefact, which is why the convention is fixed
  and applied uniformly.
- **Zero-shot scoring.** No labels, no fine-tuning, no market outcome enters the
  prompt, so the axes cannot encode look-ahead about the returns they predict.
- **Determinism.** Content-addressed cache keyed by SHA-1 of entity + normalised
  headline; re-runs are free and auditable.
- **Inference.** Two-way clustered standard errors by date and symbol
  (Cameron–Gelbach–Miller), cross-checked with Fama–MacBeth + Newey–West;
  non-overlapping windows as the primary specification for multi-session horizons;
  block bootstrap over whole dates for differences between models.
- **No single-operating-point claim.** The conviction gate's τ* is reported, but
  the headline comparison is precision at a *fixed* 10% coverage plus the full
  risk–coverage curve and AURC, which removes the "τ* was tuned to the number you
  report" objection.

## LaTeX build gotchas

- `sn-jnl.cls` is **not on CTAN** and Springer's own template zip URL 404s. The
  working copy here came from the `godkingjay/springer-nature-latex-template`
  GitHub mirror.
- That copy needs `manyfoot`, `xcolor` and `amsthm` loaded explicitly or it dies at
  `\begin{document}`, and it does **not** define `thmstyleone`/`thmstyletwo` — the
  manuscript declares them itself.
- Use `[pdflatex,sn-basic]`. Verified against a real Financial Innovation article,
  `sn-basic.bst` reproduces their house style exactly:
  `Adcock R, Gradojevic N (2019) Non-fundamental… Physica A 531:121727`.
- LaTeX control sequences may contain **letters only** — `\PrecAH5` silently parses
  as `\PrecAH` followed by `5`. `06_tables.py` spells horizons out (`\PrecAHFive`).
- **Never write `.tex` through a bash heredoc**: `\\` collapses to `\` and every
  table row breaks silently.

## Before submitting — author actions

1. Author details are **author-confirmed** (2026-08-17): affiliations, ORCIDs and
   both e-mail addresses. Note the two authors sit on *different* domains --
   `fcrit.ac.in` for author 1, `fragnel.edu.in` for author 2 -- because they are at
   different institutes. See `../AUTHORS.md`.
2. ORCIDs appear in the Declarations and are the author-confirmed ones; enter them
   in the submission system too.
3. Select the **SWUFE-sponsored APC waiver** at submission if eligible.
4. Run a similarity check before upload.
5. Upload the LaTeX **source** as well as the PDF — Springer Nature requires
   editable source files and delays production without them.
