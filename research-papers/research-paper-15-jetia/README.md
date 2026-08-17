# Paper 15 — ITEGAM-JETIA

**An Attenuation-Corrected Matched Filter with Closed-Loop Gain Design for
News-Sentiment Aggregation in Equity Forecasting**

Target: *ITEGAM-JETIA — Journal of Engineering and Technology for Industrial
Applications* (<https://itegam-jetia.org>). E-ISSN 2447-0228, DOI prefix
10.5935, Scopus / DOAJ / Latindex / Crossref indexed, Qualis CAPES, English
only, double-blind, paid APC.

This is the **same paper** as `../research-paper-14-aece-not-sent`, rebuilt in
JETIA's format. Same study, same data, same figures, same numbers — every
quantitative statement is still interpolated straight from `results/`, so the
prose cannot drift away from the analysis. What changed is only what the
journal requires; the list is in "What differs from the AECE version" below.

## Files

| file | what it is |
|---|---|
| `jetia_matched_filter_BLIND.docx` | **the file to submit.** Author identification removed from the text and from the Word document properties, per JETIA's submission checklist |
| `jetia_matched_filter_BLIND.pdf` | the same, rendered, for reading and checking |
| `jetia_matched_filter_authors.docx` / `.pdf` | identical manuscript with the author block filled in, for your own record and for the camera-ready stage |
| `authors.txt` | names, affiliations, ORCIDs and e-mails, which the template says must be sent as a separate text file |
| `*_cited.json` | the citation tags in order of first appearance, consumed by `ref_stats.py` |
| `templates/JETIA_Template.docx` | the journal's own template, downloaded from itegam-jetia.org |
| `templates/JETIA_author_instructions.txt` | the author-information page as text, captured on 2026-08-01 |
| `analysis/` | manuscript builder plus the unchanged pipeline scripts |
| `results/`, `figures/` | the artefacts the manuscript is built from |

The manuscript is delivered as `.docx`. JETIA asks for "Microsoft Word format"
and distributes a `.docx` template, and `.docx` keeps the twelve equations as
real, editable Word (OMML) objects; the legacy `.doc` round trip degrades
them. The previous journal specifically demanded `.doc`, which is why the
AECE version carries that extension.

## Build

```bash
./build_paper.sh
```

Writes both manuscripts, converts each to `.doc` and `.pdf` through Word, and
then runs the reference-distribution check. The upstream pipeline (steps 1–7:
corpus, scoring, panel, kernel, filters, loop, figures) is unchanged and lives
with its 96 MB data cache in `../research-paper-14-aece-not-sent`; only the
manuscript step is duplicated here.

## Structure (JETIA's mandated IMRDC layout)

| JETIA section | contains |
|---|---|
| I. Introduction | unchanged |
| II. Theoretical Framework | II.1 news sentiment and asset prices, II.2 the aggregation convention and the gap it leaves, II.3 measurement model, II.4 identification from cross-scorer covariance, II.5 response-kernel identification, II.6 matched filtering, II.7 closed-loop gain design |
| III. Materials and Methods | III.1 corpus, III.2 timing, III.3 scorers and responses, III.4 estimation and evaluation protocol |
| IV. Results and Discussions | IV.1–IV.5 results, IV.6 discussion |
| V. Conclusions | unchanged |
| VI. Author's Contribution | new (JETIA requires it) |
| VII. Acknowledgments | new (JETIA requires it) |
| VIII. References | IEEE style, numbered by first appearance |

All twelve equations, all six figures and all four tables carry over in the
same order, so equation numbers (1)–(12) are unchanged from the AECE version.

## What differs from the AECE version, and why

1. **Section layout.** JETIA mandates IMRDC. The AECE version's Sections
   II–VI (related work, signal model, kernel identification, matched
   filtering, closed loop) are gathered as II.1–II.7 under Theoretical
   Framework; Results and Discussion are merged into Section IV, as the
   journal asks. No sentence of the argument was dropped.
2. **Reference list, 40 → 66 entries.** JETIA requires that *at least half*
   the references be no older than five years. The AECE list was 11/40
   recent (27.5 %), which would have failed. Twenty-six real, CrossRef-verified
   references published 2021–2026 were added, and the recent-literature
   coverage in II.1 and II.2 was widened to cite them properly — which is
   also what JETIA asks a theoretical framework to do. The three AECE
   self-citations were replaced by four JETIA articles, which keeps the
   "recent work in this journal" sentence honest and stays under JETIA's
   10 % self-citation ceiling. `analysis/ref_stats.py` checks all four of
   the journal's distribution rules.
3. **Numbering and captions.** Tables and figures are numbered in Arabic
   (`Table 1`, `Figure 1`) instead of Roman, figure captions moved below the
   image, table captions stayed above, and each carries the
   `Source: Authors, (2026).` line the template uses.
4. **Closing sections.** Author's Contribution and Acknowledgments replace
   the Conflict of Interest and Publisher's Note sections AECE required.
   Conflicts are declared through JETIA's submission system instead, and the
   text for that is in `authors.txt`.
5. **Blinding.** JETIA's checklist requires author identification to be
   removed from the manuscript file *and* from Word's document properties.
   That is what the `_BLIND` file is; `authors.txt` carries the details.
6. **Small wording additions** required by JETIA's section briefs: the
   research question and hypothesis are stated explicitly in the
   Introduction (JETIA asks for both), the universe-selection rule is spelt
   out in III.1 ("clearly define the study population"), a sentence on the
   computational path was added to III.4, and the Conclusions name the
   direction of future work. Results sentences were put in the past tense,
   as the journal's brief specifies.

Nothing else in the argument, the data, the numbers or the figures changed.

## Journal compliance checklist

- [x] Microsoft Word format (`.docx`)
- [x] Journal's own template used unmodified for the masthead, the
      yellow-highlighted volume/DOI/date fields and the copyright block
- [x] 13 pages, inside the template's 8–14 page range
- [x] Times New Roman 10 pt, single spacing, single column, A4
- [x] Abstract 198 words (template: 150–200), single paragraph, and it
      states objective, framework, methodology and principal findings
- [x] 5 keywords (rule: 3–5)
- [x] IMRDC structure with Author's Contribution and Acknowledgments
- [x] IEEE citation style, numbered in ascending order of first appearance
- [x] 66 references, every one carrying a DOI
- [x] ≥ 50 % of references from the last five years — 37/66 = 56.1 %
- [x] ≥ 70 % international relative to the corresponding author's country
- [x] ≥ 40 % available online — 100 %
- [x] < 10 % citations to JETIA — 4/66 = 6.1 %
- [x] Figures and tables inserted in the text, not appended
- [x] Author identification removed from the file and from Word properties
      (`_BLIND` build)
- [x] Every reference re-checked field-by-field against CrossRef
      (`python analysis/verify_refs.py`)
- [ ] **ORCID identifiers** — mandatory for both authors and not yet
      available. Placeholders sit in `authors.txt` and in the author build;
      register at <https://orcid.org/signin> and fill them in before
      submitting.
- [ ] Publication fee — JETIA charges an APC; confirm the current amount at
      <https://itegam-jetia.org/journal/index.php/jetia/publication-fee>

## References

`analysis/refs.py` holds all 66 entries with their publication years.
`analysis/verify_refs.py` re-checks every one field-by-field (title, first
author, year, volume, issue, pages) against the CrossRef API, and
`analysis/ref_stats.py` checks JETIA's distribution rules against the tags
the manuscript actually cited. Re-run both after any edit:

```bash
python analysis/verify_refs.py
python analysis/ref_stats.py --cited jetia_matched_filter_BLIND_cited.json
```
