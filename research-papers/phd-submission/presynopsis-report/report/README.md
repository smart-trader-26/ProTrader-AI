# Pre-Synopsis Report — build notes

**Output:** `Pre-Synopsis_Report_Pardeshi.pdf` (= `main.pdf`) — **36 physical pages**

| Block | Pages | Count |
|---|---|---|
| Front matter (title, abstract, TOC, LoF + LoT, abbreviations) | i–v | 5 |
| **Main content — Chapters 1–8 + Publications** | 1–27 | **27** |
| References (105 entries, two-column) | 28–31 | 4 |

5 figures · 7 tables · **105 references** · 6 publications labelled `[P1]`–`[P6]`.

Chapter 8 announces a nine-chapter thesis whose structure and chapter titles match
`../../thesis-report/` exactly.

## ▶ What you must fill in

Everything needing your input lives in **one file: `placeholders.tex`** — 19 placeholders, which
print in **red** in the PDF (including a red note on the title page) so nothing unverified reaches
the committee by accident. The thesis in `../../thesis-report/` uses an identical copy of this
file, so filling one in and copying it over keeps both documents in sync.

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

**Already verified from the repo — no action needed:** author names, ORCIDs, affiliations,
University of Mumbai (from `research-papers/AUTHORS.md`); and `[P5]` = *Int. J. of Applied
Decision Sciences* (Inderscience), manuscript **IJADS-312370**, under review after a
major-revision decision.

## Build

```
python make_figs.py          # regenerates the data figures in figs/
pdflatex main.tex            # run 3x for TOC / List of Figures / List of Tables
pdflatex main.tex
pdflatex main.tex
```

Requires MiKTeX and matplotlib. No bibtex — the bibliography is a `thebibliography` block at the
end of `ch6_end.tex`.

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
| `make_figs.py` | generates the data figures; every plotted number is transcribed from the source papers |
| `figs/*.pdf` | vector figures produced by `make_figs.py` (three are unused in this compressed edition and are retained for the thesis build) |

## Relationship to the thesis

This report is the compressed statement of the same research programme set out at full length in
`../../thesis-report/`. Both documents are built from the same evidence, and a cross-document
check confirms that all 41 headline figures — MAE 0.018, $R^2$ 0.988, DA 84.7 %, the 10.9 % joint
textual ablation, 22.7 % / −18.3 % for dynamic fusion, 47.72 % against 46.43 %, causal graph
density 0.850, ANOVA $p = 1.8\times10^{-8}$, Sharpe 0.878 against 0.628, drawdown −18.98 % against
−36.08 %, holdout Sharpe 1.614, and the rest — are identical in the two.

## Scope notes worth knowing before the viva

- **RO4 and transfer learning.** None of the publications performs transfer *learning*; `[P5]`
  re-applies the framework **unmodified** to a second universe. §5.5.3 states this explicitly and
  §7.2 lists adaptation-based transfer as open work.
- **RO3 and XAI.** No SHAP/LIME experiments exist in the papers, so transparency rests on directed
  causal structure; attribution methods are cited, not run.
- **`[P6]`** is listed in Publications and cited once as survey groundwork; it has no contribution
  section of its own.
- The report deliberately reports the papers' **null results** (`[P3]`'s near-chance single-name
  accuracy, `[P5]`'s signal-layer null and insignificant Sharpe). These are load-bearing — §6.2
  explains why the 84.7 % and 47.72 % figures are not in conflict.

## Page-budget levers

The main content is 27 pages against an ~25-page target. To go lower or higher:

- `\documentclass[11pt,...]` in `main.tex` — 12pt adds ≈ 6 pages
- `\setstretch{1.0}` — 1.15 adds ≈ 3 pages, 1.5 adds ≈ 12
- `\setlength{\parskip}{0.36em}` — 0.25em removes ≈ 1 page
- each chapter begins on a new page, so eight chapters carry ≈ 3 pages of unavoidable tail
  whitespace; merging Ch.8 into Ch.7 saves one page
- Table 2.1 and Table 5.5 are the two largest tables; §6.3 is the most compressible prose block
- the reference list is two-column at `footnotesize` and occupies 4 pages
