# IJASE submission notes — Multiplicative Information Gating

**Journal:** International Journal of Advanced Science and Engineering (IJASE),
Mahendra Publications — <https://mahendrapublications.com>
**Article type:** Original research article
**Source manuscript:** `../rejected-research-paper-8-tubitak/tubitak_mig.tex`
(rejected by TJEECS; scientific content carried over unchanged)
**Format precedent:** `../research-paper-5-not-sent/ijase_protrader.tex` (same journal)

## What was kept from the TUBITAK version

Verified by script — all 46 measured quantities appear verbatim:

- Live ledger: 621 resolved forecasts, 70.7% coverage overall, 69.2%/575 at 10 days,
  86.8%/38 at 20 days, 100%/8 at 5 days, 5.6% median half-width, 19.3 pp shortfall
- Calibration: ECE 0.345 → 0.049, Brier > 0.366, up-prob 0.80 vs accuracy 0.51
- Accuracy ceiling: 50.09% out-of-fold, ranking AUC ≈ 0.49
- Conviction gate: τ\* = 0.63, 60.6% precision vs 58.0% base, +2.6 pp, 5.6% firing,
  94% abstention, 2,840 pooled calls, 95% CI 58.8–62.4, z ≈ 2.9 (p ≈ 0.002),
  AUC ≈ 0.47, DOWN calls ≈ 10% precision
- Universe/protocol: 54 names, 2018–2026, 91,471 train rows, 50,272 OOS rows, μ₀ = 0.15
- Per-year table: 2022 (365/61.6/54.0), 2023 (290/68.3/65.3), 2024 (932/53.3/55.7),
  2025 (822/69.0/58.2), 2026 (431/54.5/47.1), pooled (2,840/60.6/58.0)
- Veto example: +0.018, +0.022, +0.003, −0.330
- Both propositions with proofs; all four numbered equations

## What changed, and why

IJASE caps an original article at **15 pages including references, tables and figures**,
with each table and figure on its own page — a much tighter budget than TJEECS. That
forced three presentation changes. No result, claim or number was altered.

1. **Prose condensed** from ~4,800 to ~3,000 words. Introduction, Related Work and
   Discussion are considerably tighter than the TJEECS version.
2. **Conceptual pipeline schematic dropped** (old Figure 1). It carried no data and
   its own caption described it as conceptual. The two gates are described in the text.
3. **Illustrative headline table folded inline** (old Table 3). Its four
   polarity/novelty/materiality/signal triples are now stated in the prose of
   Section 4.5, so the numbers survive without consuming a whole float page.
4. **Two result tables merged into one two-panel Table 1** — panel (a) live-ledger
   coverage, panel (b) walk-forward selective signal. Same device used by IJASE
   paper 5. Every cell is preserved.
5. **References reduced 28 → 20** (IJASE allows up to 35). Dropped: El-Yaniv &
   Wiener 2010, Araci 2019, BloombergGPT 2023, Naeini 2015, Zaffran 2022, Gibbs 2021,
   Harvey 2016, Kelly & Xiu 2023 — each redundant with an adjacent retained citation.
6. **Highlights removed from the manuscript body**, since the guide requires them as
   a separate submission file. They live only in `Highlights.txt`.

## Not included, deliberately

A **graphical abstract** is optional under the IJASE guide, so none is supplied. If
one is later wanted it must be a separate file, minimum 531 × 1328 px (h × w).

## IJASE format compliance (all verified by script)

| Requirement | Status |
|---|---|
| Max 15 pages incl. refs/tables/figures | **15** |
| A4, 12 pt, double spacing, 3 cm margins | yes |
| No full justification (ragged right) | yes (`ragged2e`, keeps hyphenation) |
| Paragraphs indented | yes (1.27 cm) |
| Section order: Title, Authors, Affiliations, Abstract, Keywords, Introduction, Materials and Methods, Results and Discussion, Conclusions, Acknowledgements, References, Figure Captions, Tables and Figures | yes (Theory is §3, explicitly permitted by the guide) |
| Numbered sections 1, 1.1, 1.1.1 | yes |
| Results and Discussion as a single section | yes (§4) |
| Abstract 100–150 words | 150 |
| Keywords ≤ 5 | 5 |
| Highlights, 3–5 bullets, ≤ 85 chars each, **as a separate file** | `Highlights.txt` — 5 bullets, longest 76 chars; deliberately **not** in the manuscript body |
| Conclusions ≤ 100 words | 98 |
| References ≤ 35, numbered in order of first appearance | 20, order verified |
| Reference style (Author,A., Year. Title. Journal, vol, pages.) | yes |
| Corresponding author with asterisk, e-mail, phone, full postal address | yes |
| Tables/figures each on a separate page at the end | yes |
| Figure captions listed separately | yes |
| Each illustration as a separate file | `figures/figure1.pdf` |

Compile: `pdflatex ijase_mig.tex` ×2 — 0 errors, 0 overfull boxes, 0 undefined
references or citations.

## Files to upload

| File | Purpose |
|---|---|
| `ijase_mig.pdf` | manuscript (15 pages) |
| `figures/figure1.pdf` | Figure 1, separate illustration file |
| `Highlights.txt` | mandatory Highlights, separate file with "Highlights" in the name |
| `IJASE_copyright_form.pdf` | journal copyright form — **sign before uploading** |
| `Cover_Letter_IJASE_template.docx` | journal cover-letter template — fill from the draft below |

## Before submitting — author actions

1. **Confirm the corresponding author's phone number.** `+91-22-2768-0000` is carried
   over from paper 5 and looks like a placeholder switchboard number. IJASE requires a
   working number with country and area code.
2. **Sign the copyright form** and complete the cover letter template.
3. **Run a similarity / AI-detection check** on the new PDF — IJASE screens every
   submission with plagiarism software.
4. **APC is USD 150 (₹13,500)** as of 1 January 2025, payable on acceptance. Waivers
   are considered case by case if funding is unavailable.
5. Register and submit at <https://mahendrapublications.com/submit_article>.
6. ORCIDs are deliberately **not** in this version (the TJEECS draft used ORCID's
   documentation example ID). Add real ORCIDs if the submission system asks.

## Draft cover letter

> Dear Editor,
>
> We submit for your consideration an original research article, "Multiplicative
> Information Gating: Decomposing News into Novelty, Materiality and Polarity for
> Calibrated Selective Forecasting", for the International Journal of Advanced
> Science and Engineering.
>
> The paper addresses a problem common to every system that turns a stream of text
> into a decision. The standard approach reduces each item to a single sentiment
> polarity and averages, which conflates three properties that act through different
> channels: the direction of the news, how surprising it is, and how much it bears on
> value. We propose decomposing each event into polarity, novelty and materiality and
> combining them as a product, so that any near-zero factor vetoes the signal. We
> prove a veto bound and show that the product is the canonical combiner under
> per-axis multiplicative scaling.
>
> The framework is instantiated on National Stock Exchange of India large-cap
> equities, and every quantitative result is a measured statistic from a deployed
> system or its live append-only ledger. On 621 resolved forecasts, nominal 90%
> price intervals achieved 70.7% empirical coverage; an offline three-stage
> recalibration cut expected calibration error from 0.35 to 0.05 without changing
> accuracy; and a conviction gate that abstains on roughly 94% of cases raised the
> thirty-day hit rate to 60.6% against a 58.0% base rate, beating the base in four of
> five test years. We report no result we cannot measure, and Section 4.6 states
> plainly what the available data does not let us claim.
>
> The manuscript is original, has not been published elsewhere, and is not under
> consideration by another journal. All authors have approved the submission and
> declare no conflict of interest. We confirm the manuscript follows the IJASE Guide
> for Authors: 15 pages, double spaced, 12 pt, 3 cm margins, ragged right, numbered
> sections with Results and Discussion combined, a 150-word abstract, five keywords,
> five highlights supplied as a separate file, a 98-word Conclusions section, and 20
> references in order of first appearance.
>
> Thank you for your consideration.
>
> Yours sincerely,
> Anandkumar Pardeshi (corresponding author)
> Department of Computer Science and Engineering
> Fr. C. Rodrigues Institute of Technology, Vashi, Navi Mumbai 400703, India
> anand.pardeshi@fcrit.ac.in
