# TJEECS resubmission notes (new submission; previous Ms. No. TJEECS-2026-01032)

## Why the paper was returned
The return was **technical, not scientific**: "The number of pages should not be more than 15."
The manuscript itself was 14 pages, but the submission package also carried 4 separate
figure PDFs, and Editorial Manager merges everything into one reviewing PDF
(14 + 4 = 18 pages), which is what the technical staff counted. The journal explicitly
invites resubmission with a new number.

## What changed for the resubmission
- **Manuscript: 14 → 12 pages** (`tubitak_mig.pdf`). Prose tightened throughout; no
  proposition, proof, table, or measured number was changed.
- **Figures: 4 → 2 separate files** (`figures/figure1.pdf`, `figures/figure2.pdf`):
  - old Figure 3 (event-signal line plot) removed — Table 3 already demonstrates the veto;
  - old Figures 2 and 4 (coverage bars, yearly bars) merged into one two-panel Figure 2 (a)/(b);
  - retired files moved to `old/superseded-figures/` — **do not upload those**.
- **Worst-case merged reviewing PDF: 12 + 2 = 14 pages ≤ 15**, with one page of headroom
  even if the system prepends a metadata sheet.
- **References: 35 → 28**, still in strict citation order (TJEECS rule, verified by script).
  Five recent works added to sharpen the novelty positioning the desk letter asks for
  (Huang et al., Contemporary Accounting Research 2023; BloombergGPT 2023; Kelly & Xiu,
  Foundations and Trends in Finance 2023; Zaffran et al., ICML 2022; Steyvers et al.,
  Nature Machine Intelligence 2025). Seven older/redundant entries dropped
  (Bollen 2011, Tetlock 2008, Malkiel 2003, Niculescu-Mizil 2005, Vovk 2005, Krauss 2017,
  Devlin 2019, Fischer 2018, Lopez de Prado 2018 — each covered by an adjacent citation).
- **Explicit novelty claim added** to Related Work ("To our knowledge, no prior work scores
  novelty and materiality as separate, continuous, per-item quantities and makes them
  multiplicative preconditions for influence with a provable veto…"), addressing the
  desk letter's "novelty must be emphasized by referring to recent literature" criterion.
- Conflict-of-interest statement folded into "Acknowledgment and disclaimers"; AI
  declaration tightened (substance unchanged).
- Abstract now 273 words (≤300); 6 keywords; 0 LaTeX errors; 0 overfull boxes.

## Before submitting — still to do (author actions)
1. **ORCIDs are placeholders** in the .tex byline (the 0000-0002-1825-0097 ID is ORCID's
   documentation example). Replace with the authors' real ORCID iDs.
2. Re-run the **iThenticate/similarity** and AI-report checks on the new PDF (old reports
   in `old/final/` are for the previous version).
3. Upload **only**: `tubitak_mig.pdf` (manuscript), `figures/figure1.pdf`,
   `figures/figure2.pdf`, plus any forms the system requires.
4. After Editorial Manager builds the merged PDF, **check its page count is ≤ 15 before
   approving** the submission.
5. In the "comments to the editor" box, note that this replaces TJEECS-2026-01032 and
   that the page-limit issue has been resolved (manuscript 12 pages; two figure files).

## Suggested cover note to the editor
> Dear Editor,
> This manuscript is a revised resubmission of TJEECS-2026-01032, returned on technical
> grounds (page limit). The manuscript has been shortened to 12 pages and the figure
> files consolidated from four to two, so the complete submission is within the 15-page
> limit. We also strengthened the positioning of the contribution with respect to the
> recent literature. No results or claims were altered.
