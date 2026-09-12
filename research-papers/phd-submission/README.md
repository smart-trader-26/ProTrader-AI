# PhD Submission — Pre-Synopsis Report and Thesis

**Integrating Multi-Source Data with Sentiment Analysis and Language Models to Enhance Stock
Market Decision Making**
Anandkumar Pardeshi · guide Dr. Sujata Deshmukh · University of Mumbai

Two paired LaTeX documents built from the same evidence, plus the source material they draw on.

| Deliverable | Path | Pages |
|---|---|---|
| **Pre-synopsis report** | `presynopsis-report/report/Pre-Synopsis_Report_Pardeshi.pdf` | 36 total = 5 front-matter + **27 main content** + 4 reference pages |
| **Thesis** | `thesis-report/PhD_Thesis_Pardeshi.pdf` | 158 total = 19 preliminary + **129 main content** + 3 publications + 7 reference pages |

| | Pre-synopsis | Thesis |
|---|---|---|
| Chapters | 8 | 9 |
| Figures | 5 | 22 |
| Tables | 7 | 23 |
| References | 105 | 153 |
| Type size / spacing | 11pt, 1.0 | 12pt, 1.4 |

## Folder layout

```
phd-submission/
├── presynopsis-report/
│   ├── report/                    the pre-synopsis LaTeX source and PDF
│   │   ├── main.tex, front.tex, ch1_3.tex, ch4.tex, ch5.tex, ch6_end.tex
│   │   ├── placeholders.tex       <- the 19 fields you fill in
│   │   ├── make_figs.py, figs/
│   │   ├── Pre-Synopsis_Report_Pardeshi.pdf
│   │   └── README.md              build notes, scope notes, page-budget levers
│   ├── My Title and data/         the six source publications + "Title with ROs.txt"
│   ├── sample report/             the format exemplar supplied by the department
│   └── synopsis_sequence.docx
└── thesis-report/                 the thesis LaTeX source and PDF
    ├── main.tex, front.tex, ch1.tex … ch9.tex, backmatter.tex
    ├── placeholders.tex           <- identical copy of the file above
    ├── make_figs.py, figs/        13 data figures, all from real numbers
    ├── PhD_Thesis_Pardeshi.pdf
    └── README.md                  build notes, chapter map, provenance, scope notes
```

## The one thing to do before submitting

Both documents share **`placeholders.tex`** — 19 fields that print in **red** in the PDF so
nothing unverified reaches an examiner by accident: degree discipline, PhD registration number,
submission month/year, guide's designation, and the venue/DOI details for publications
`[P1]`–`[P6]`.

Fill it in **once**, then copy it over the other:

```
cp presynopsis-report/report/placeholders.tex thesis-report/placeholders.tex
```

Author names, ORCIDs and affiliations are already filled from `../AUTHORS.md` and need no action.

## Build

Each document builds independently, in its own directory, with no bibtex:

```
cd presynopsis-report/report   # or: cd thesis-report
python make_figs.py            # regenerates figs/ from real numbers
pdflatex main.tex              # 3x, for TOC / LoF / LoT / cross-references
pdflatex main.tex
pdflatex main.tex
```

Requires MiKTeX and matplotlib. LaTeX build artefacts (`*.aux`, `*.log`, `*.toc`, `*.lof`,
`*.lot`, `*.out`, …) are git-ignored via `../.gitignore` and can be deleted at any time — nothing
depends on them.

## How the two documents relate

The pre-synopsis is the compressed statement of the research programme the thesis sets out at
full length. They are kept synchronous in three specific ways:

1. **Structure.** Chapter 8 of the pre-synopsis announces a nine-chapter thesis; the thesis has
   exactly those nine chapters, chapter titles included.
2. **Numbers.** All **41 headline figures are identical** across the two PDFs — MAE 0.018,
   $R^2$ 0.988, DA 84.7 %, the 10.9 % joint textual ablation, 22.7 % / −18.3 % for dynamic
   fusion, 47.72 % against 46.43 %, causal graph density 0.850, ANOVA $p = 1.8\times10^{-8}$,
   Sharpe 0.878 against 0.628, drawdown −18.98 % against −36.08 %, holdout Sharpe 1.614, and the
   rest.
3. **Provenance.** Every quantitative claim traces to one of three sources and none is estimated:
   Tables II–V of `[P1]` and Table 4 of `[P3]` (in `presynopsis-report/My Title and data/`);
   `[P4]`, whose relative-to-baseline reporting convention is preserved rather than filled in with
   invented absolutes; and the JSON artefacts in `../research-paper-ijads-312370/3-analysis/`,
   themselves gated by that folder's `verify_numbers.py`.

If you change a number in one document, change it in the other and re-check.

## Scope notes to have ready for the viva

Stated in both documents, collected here so nothing is a surprise.

- **RO4 and transfer learning.** No study performs transfer *learning*; `[P5]` re-applies the
  framework **unmodified** to a second universe. Both documents call this direct transfer and list
  adaptation-based transfer as open work.
- **RO3 and XAI.** No SHAP/LIME experiments exist in the publications. Transparency rests on
  directed causal structure; attribution methods are reviewed and cited but not run.
- **The allocation universe is not Indian.** The eighteen-year study needs clean history plus
  volatility dispersion; it is presented as a test of the *mechanism*, with an Indian multi-asset
  study listed as future work.
- **The signal-layer null is load-bearing.** Removing the signal fusion layer *improves* both
  measures insignificantly. This is deliberate and is the central finding; both documents explain
  why the 84.7 % and 47.72 % figures are not in conflict.
- **Two documented implementation defects.** A unit-scale mismatch and an inert risk layer had
  silently invalidated an earlier version of the allocation study. Both are reported because both
  produce no error message.
