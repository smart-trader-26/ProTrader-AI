# Pre-Synopsis Report — build notes

**Output:** `Pre-Synopsis_Report_Pardeshi.pdf` — 57 physical pages = 1 title page +
7 roman front-matter pages + **49 numbered body pages** (within the 40–50 limit).

18 figures · 9 tables · 80 references · 6 publications labelled `[P1]`–`[P6]`.

## ▶ What you must fill in

Everything that needs your input lives in **one file: `placeholders.tex`**.
There are **19 placeholders**. They print in **red** in the PDF (including a red note on
the title page) so nothing unverified can reach the committee by accident.

| Group | Fields |
|---|---|
| Title page | degree discipline, PhD registration number, submission month/year, guide's designation |
| `[P3]` Liberte | indexing claim, article URL/DOI |
| `[P4]` IJCISIM | submission month/year, acknowledgement URL or manuscript ID |
| `[P5]` IJADS | Inderscience submission URL (optional) |
| `[P1]` ICSIT 2026 | host institute/city/dates, pages, IEEE Xplore DOI |
| `[P2]` DCHF | full conference name/host/city/dates, acknowledgement URL |
| `[P6]` ICETI4T 2025 | pages, IEEE Xplore DOI |
| Copyrights/patents | set `\HasIPRtrue` and fill `\IPRentries` if you have any; otherwise the section is omitted |

To fill one in, replace the whole `\ph{...}` with your text. For links use
`\url{https://...}`. If a field does not apply, use `\notapplicable`.

**Already verified from your repo — no action needed:** author names, ORCIDs,
affiliations, University of Mumbai (from `research-papers/AUTHORS.md`); and `[P5]` =
*Int. J. of Applied Decision Sciences* (Inderscience), manuscript **IJADS-312370**,
under review after a major-revision decision.

## Build

```
python make_figs.py          # regenerates the five data figures in figs/
pdflatex main.tex            # run 3x for TOC / List of Figures / List of Tables
pdflatex main.tex
pdflatex main.tex
```

Requires MiKTeX (installed) and matplotlib. No bibtex — the bibliography is a
`thebibliography` block at the end of `ch6_end.tex`.

## Files

| File | Contents |
|---|---|
| `placeholders.tex` | **the only file you need to edit** — all 19 user-supplied fields |
| `main.tex` | preamble, layout, TikZ styles, title page, `\input` of the chapters |
| `front.tex` | Abstract, Table of Contents, List of Figures/Tables, Abbreviations |
| `ch1_3.tex` | Ch.1 Introduction · Ch.2 Literature Survey · Ch.3 Problem Statement and Objectives |
| `ch4.tex` | Ch.4 Research Contributions + the publication↔objective mapping (Table 4.1) |
| `ch5.tex` | Ch.5 Methodology and Implementation, one section per objective, then integration |
| `ch6_end.tex` | Ch.6 Discussion · Ch.7 Conclusion · Ch.8 Thesis organisation · Publications · References |
| `make_figs.py` | generates the five data figures; every plotted number is transcribed from the source papers |
| `figs/*.pdf` | vector figures produced by `make_figs.py` |

## Scope notes worth knowing before the viva

- **RO4 and transfer learning.** None of the five papers performs transfer learning;
  `[P5]` re-applies the framework unmodified to a second universe. Section 5.5.3 states
  this explicitly and Section 7.2 lists adaptation-based transfer as open work.
- **RO3 and XAI.** No SHAP/LIME experiments exist in the papers, so transparency rests on
  directed causal structure; attribution methods are cited, not run.
- **`[P6]`** is listed in Publications and cited once as survey groundwork; it has no
  contribution section of its own.
- The report deliberately reports the papers' **null results** (`[P3]`'s near-chance
  single-name accuracy, `[P5]`'s signal-layer null and insignificant Sharpe). These are
  load-bearing — Section 6.2 explains why the 84.7% and 47.72% figures are not in conflict.

## Page-budget levers (if your template pushes it over 50)

- `\setstretch{1.28}` in `main.tex` (→ 1.20 saves ≈3 pages; → 1.5 adds ≈4)
- `\scalebox{0.86}` on the TikZ diagrams in `ch1_3/ch4/ch5.tex`
- Table 2.1 and Table 5.1 are the two largest tables
- §6.3 (*Research Significance*) is the most compressible prose block
- The reference list is two-column at `scriptsize` and occupies 4 pages
