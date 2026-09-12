# PhD Thesis — build notes

**Integrating Multi-Source Data with Sentiment Analysis and Language Models to Enhance Stock
Market Decision Making**
Anandkumar Pardeshi · guide Dr. Sujata Deshmukh · University of Mumbai

**Output:** `PhD_Thesis_Pardeshi.pdf` (= `main.pdf`) — **158 physical pages**

| Block | Pages | Count |
|---|---|---|
| Preliminary pages (title, declaration, certificate, acknowledgements, abstract, TOC, LoF, LoT, abbreviations, symbols) | i–xix | 19 |
| **Main content — Chapters 1–9** | **1–129** | **129** |
| Publications on PhD Work | 130–132 | 3 |
| References (153 entries, two-column) | 133–139 | 7 |

22 figures · 23 tables · **153 references** · 6 publications labelled `[P1]`–`[P6]`.

## Chapter map

| Ch | Title | Pages | Objective | Source publication |
|---|---|---|---|---|
| 1 | Introduction | 9 | — | — |
| 2 | Background and Preliminaries | 18 | — | — |
| 3 | Literature Survey and Research Gaps | 15 | — | `[P6]` |
| 4 | Multi-Source Data Integration Framework | 17 | RO1 | `[P1]` |
| 5 | Dynamic Uncertainty-Weighted Fusion | 13 | RO1 | `[P4]` |
| 6 | Hybrid Language Model and Indicator Framework | 13 | RO2 | `[P3]`, `[P1]` |
| 7 | Explainable and Causal Framework | 15 | RO3 | `[P2]`, `[P3]`, `[P5]` |
| 8 | Decision-Theoretic Allocation and Cross-Market Validation | 18 | RO4 | `[P5]` |
| 9 | Conclusion and Future Scope | 11 | — | — |

The nine-chapter structure is exactly the one announced in Chapter 8 of the pre-synopsis report
(`../presynopsis-report/report/`), chapter titles included.

## ▶ What you must fill in

Everything needing your input is in **one file: `placeholders.tex`** — the same file the
pre-synopsis uses, copied here so the two documents can be filled in independently or kept in
sync by copying one over the other. There are 19 placeholders; they print in **red** in the PDF
(including a red note on the title page) so nothing unverified reaches an examiner by accident.

| Group | Fields |
|---|---|
| Title page | degree discipline, PhD registration number, submission month/year, guide's designation |
| `[P3]` Liberte | indexing claim, article URL/DOI |
| `[P4]` IJCISIM | submission month/year, acknowledgement URL or manuscript ID |
| `[P5]` IJADS | Inderscience submission URL (optional) |
| `[P1]` ICSIT 2026 | host institute/city/dates, pages, IEEE Xplore DOI |
| `[P2]` DCHF | full conference name/host/city/dates, acknowledgement URL |
| `[P6]` ICETI4T 2025 | pages, IEEE Xplore DOI |
| Copyrights/patents | set `\HasIPRtrue` and fill `\IPRentries`; otherwise the section is omitted |

**Already verified from the repo — no action needed:** author names, ORCIDs and affiliations
(from `research-papers/AUTHORS.md`), and `[P5]` = *Int. J. of Applied Decision Sciences*
(Inderscience), manuscript **IJADS-312370**, under review after a major-revision decision.

## Build

```
python make_figs.py          # regenerates the 13 data figures in figs/
pdflatex main.tex            # run 3x for TOC / LoF / LoT / cross-references
pdflatex main.tex
pdflatex main.tex
```

Requires MiKTeX and matplotlib. No bibtex — the bibliography is a `thebibliography` block at the
end of `backmatter.tex`.

## Files

| File | Contents |
|---|---|
| `placeholders.tex` | **the only file you need to edit** — all 19 user-supplied fields |
| `main.tex` | preamble, layout, TikZ styles, `\input` of front matter and chapters |
| `front.tex` | title page, declaration, certificate, acknowledgements, abstract, TOC, LoF, LoT, abbreviations, symbols |
| `ch1.tex` … `ch9.tex` | the nine chapters |
| `backmatter.tex` | Publications on PhD Work + the 153-entry reference list |
| `make_figs.py` | generates the 13 data figures; every plotted number is transcribed from the source papers or from `research-paper-ijads-312370/3-analysis/*.json` |
| `figs/*.pdf` | vector figures produced by `make_figs.py` |

## Provenance of every number

Every quantitative claim traces to one of three sources, and none is estimated or rounded into a
more favourable form:

- **Chapters 4 and 6** — Tables II–V of `[P1]` and Table 4 of `[P3]`
  (`../presynopsis-report/My Title and data/`).
- **Chapter 5** — `[P4]`, which reports improvements *relative to the strongest static baseline*
  rather than absolute levels. That reporting convention is preserved in Table 5.2 rather than
  filled in with invented absolute figures.
- **Chapters 7 and 8** — the JSON artefacts in
  `../research-paper-ijads-312370/3-analysis/` (`final_tables.json`, `extend_results.json`,
  `mega_results.json`, `ic_results.json`, `sentiment_results.json`, `defense_results.json`),
  which are themselves gated by that folder's `verify_numbers.py` (223 checks).

A cross-document check confirms that all 41 headline figures are identical in the thesis and in
the pre-synopsis report.

## Scope notes worth knowing before the viva

These are stated in the thesis itself (§1.7 and each chapter's limitations section) and are
collected here so nothing is a surprise.

- **RO4 and transfer learning.** No study performs transfer *learning*; `[P5]` re-applies the
  framework **unmodified** to a second universe. §8.9.1 states this explicitly, calls it direct
  transfer, and §9.7 lists adaptation-based transfer as open work.
- **RO3 and XAI.** No SHAP/LIME experiments exist in the papers. Transparency rests on directed
  causal structure; attribution methods are reviewed and cited (§3.3.1, §7.1) but not run, and
  §9.6 records that no comparative claim between the two is supported.
- **The allocation universe is not Indian.** §8.2.1 states why (eighteen years of clean history
  plus volatility dispersion), presents the study as a test of the *mechanism*, and §9.7 lists an
  Indian multi-asset study as future work.
- **The signal-layer null is load-bearing.** §8.6 reports that removing the signal fusion layer
  *improves* both measures insignificantly, and §8.7 explains why through the information
  coefficients. This is deliberate, it is the central finding of the thesis, and §9.3 explains why
  the 84.7 % and 47.72 % figures are not in conflict.
- **Two documented implementation defects.** §8.4 reports a unit-scale mismatch and an inert risk
  layer that had invalidated an earlier version of the study. Both are reported because both are
  silent — the software runs correctly and the results look plausible.

## Page-budget levers

The main content is 129 pages against a 125–140 target. If a template pushes it outside that:

- `\setstretch{1.40}` in `main.tex` — 1.5 adds ≈ 9 pages, 1.3 removes ≈ 8
- `\setlength{\parskip}{0.55em}` — 0.4em removes ≈ 4 pages
- margins are `left=1.4in` (binding edge) `right=1.0in`; widening the right margin costs pages
- Chapters 2 and 3 are the most compressible; Chapters 4 and 8 carry the most evidence and should
  be cut last
