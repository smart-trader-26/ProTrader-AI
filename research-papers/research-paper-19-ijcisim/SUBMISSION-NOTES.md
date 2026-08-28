# IJCISIM submission notes — construct-validity screen for LLM features

**Journal:** International Journal of Computer Information Systems and Industrial
Management Applications (IJCISIM), ISSN 2150-7988.
Publisher since 2024: **Cerebration Science Publishing** (Hong Kong).
Submission portal: <https://cspub-ijcisim.org/index.php/ijcisim/about/submissions>
(the old MIR Labs site is retired; do not submit there).

**Article type:** Original article
**Deliverable:** `ijcisim_validity_screen.docx` — the file to upload.
`ijcisim_validity_screen.pdf` is only that file as Word renders it, for review.

## Read this first — the money question

**APC is USD 1,500** (<https://cspub-ijcisim.org/index.php/ijcisim/publishing_fees>),
with no waiver mentioned on the fees page. The journal is real and indexed but its
SJR is around 0.23, and a large share of its recent volume is special issues.
`research-paper-18-dsm/PLAN.md` records the earlier judgement in one line: *do not
pay $1,500 for IJCISIM.* This manuscript was prepared because it was asked for and
is ready to send; the venue decision is still the author's, and
`../research-paper-20-dsm` is the same study prepared for a **free** Q1 journal.

## What this paper is, and how it differs from its siblings

Three manuscripts now exist from one study. They share the results and share
nothing else; each is written for the audience of its venue and none is a reformat
of another.

| Folder | Venue | Object of the paper | Frame |
|---|---|---|---|
| `../research-paper-17-fininnov-not-worth` | Financial Innovation (not sent) | multi-axis LLM features in finance | finance / asset pricing |
| **this one** | IJCISIM | a **component acceptance test** in a decision-support pipeline | information systems, verification, governance |
| `../research-paper-20-dsm` | Data Science and Management | a **pre-deployment screen** and its omission cost | analytics management, OR |

The IJCISIM version is the systems-engineering telling. Its distinctive material,
not present in the other two:

- **Section 3.1 and Table 1** describe the five-stage pipeline the scoring
  component sits in, name the control each stage already carries, and show that
  none of them is a validity test. This table is written for this version only.
- **Vendor portability is promoted to a contribution.** The cross-model intraclass
  correlations (0.779 / 0.727 / 0.565) are framed as a re-sourcing risk for a
  system owner rather than as an attenuation caveat, and the argument that
  memoised determinism is a property of the cache and not of the model is made
  explicitly.
- **Section 5.3** gives four operational rules — validity gate at feature
  onboarding, criterion named before scores are generated, budget by validated
  attribute count, record portability as well as reproducibility.
- The production ledger (Section 4.8) is presented as evidence about an operating
  system, with the point that no schema, range or drift control would have caught
  the coverage shortfall.

## Format, and where the format came from

IJCISIM ships **no LaTeX class and no downloadable Word template**. The house style
below was read off a published article in the current Cerebration style
(Vol. 16, 2024) and off the two guidance pages that do exist:

- OJS submission checklist: Microsoft Word format, single-spaced, **10-point font**,
  italics rather than underlining, figures and tables placed within the text.
- Author page: **maximum 25 typeset printed pages**; abstract 200–300 words;
  references **numbered in order of appearance**; required back matter is Author
  Contributions, Funding, Conflict of Interest Statement, Data Availability
  Statement, References, Biographical Sketch.

What the builder reproduces: A4, single column, Times New Roman throughout,
first-page masthead with journal name / ISSN / volume / publisher and an "Article"
label, centred bold title, bold authors with superscript markers, numbered
affiliations with e-mails, run-in bold `Abstract:` and `Keywords:` blocks separated
by rules, numbered section headings, bold-italic `Figure N.` captions beneath
figures and `Table N.` captions above tables, booktabs-style horizontal rules only,
centred page numbers, and a CC-BY line at the end.

**Current build: 20 pages, 13,178 words, 6 figures, 9 tables, 47 references.**
Comfortably inside the 25-page cap.

## Reproduction

```bash
./build_paper.sh          # regenerates numbers, assembles .docx, renders .pdf, verifies
```

| Step | Script | What it does |
|---|---|---|
| 1 | `analysis/10_cost_accounting.py --no-latex` | counts fitted objects into `results/cost_accounting.json` |
| 2 | `analysis/06b_tables_json.py` | rebuilds `results/tables.json` and `results/macros.json` from the result files |
| 3 | `analysis/make_paper_ijcisim.py` | assembles the manuscript |
| 4 | `ijcisim_docbuild.to_pdf` | renders through Word itself |
| 5 | `analysis/verify_manuscript.py` | hard checks; the build gates on it |

The upstream analysis (steps 01–09) is **not** duplicated here. It lives in
`../research-paper-17-fininnov-not-worth`, and step 2 reads the scored panel from
that folder's `cache/` (override with `CACHE=... ./build_paper.sh`). Only
`results/` was copied, because that is all the manuscript needs.

### Discipline carried over from the LaTeX papers

Nothing numeric is typed into the manuscript source. A number in the prose is a
`{Placeholder}` resolved from `results/macros.json` at build time, and an unknown
placeholder **raises** rather than printing itself. Tables are rendered from
`results/tables.json`. References are numbered by `refs_ijcisim.Bibliography` in
order of first citation and rendered from the same `refs.bib` the DSM sibling uses,
so an entry cannot be real in one manuscript and invented in the other.

`analysis/verify_manuscript.py` checks the built `.docx` for: surviving
placeholders; figures embedded versus captioned versus referenced; contiguous table
numbering with every table pointed at from the prose; every bracketed citation
resolving to a listed reference and every reference being cited; and the page count
against the 25-page limit.

## Build gotchas worth remembering

- **`\@floatboxreset` resets the font size**, so wrapping `\input{table}` in
  `{\footnotesize ...}` does nothing. (That bit the LaTeX sibling; recorded here
  because the same instinct will recur.) In the Word build the equivalent trap is
  Word's table autofit, which narrows the label column until three-word labels wrap
  onto three lines — the builder declares fixed column widths instead.
- **The inline markup parser treats `*` as an italic delimiter**, so `τ*` silently
  italicised the rest of a paragraph. Tau-star is written with U+2217 (`τ∗`), and
  an unmatched marker is now treated as a literal character.
- **Subscripts.** `μ₀`, `π₀`, `φ₀` use real Unicode subscript characters; the `^…^`
  markup is superscript only.
- **`refs.bib` carries TeX escapes** (`\&`, `M{\"u}ller`, `Acz{\'e}l`) because it is
  shared with the LaTeX build. `refs_ijcisim.detex` converts them; if a new entry
  introduces an accent form that is not in the `ACCENTS` table it will pass through
  as the bare letter, so check the rendered list after adding references.
- **PDF conversion goes through Word** (`win32com`), so the PDF is what an editor
  opening the `.docx` would see. It is skipped automatically if Word is absent.

## Before submitting — author actions

1. Author details are the author-confirmed ones from `../AUTHORS.md`. The two
   authors sit on **different** e-mail domains (`fcrit.ac.in`, `fragnel.edu.in`)
   because they are at different institutes; never reconstruct one from the other.
2. Register on the OJS portal before submitting; submission is a five-step form.
3. The journal's peer-review policy does not require a blinded file, so the
   author-identified `.docx` is the one to upload. `make_paper_ijcisim.py --blind`
   produces an anonymous variant if the editor asks for one.
4. Decide the APC question **before** submitting, not after acceptance.
5. Run a similarity check before upload.
6. Volume and year in the masthead are set to 18 / 2026 in
   `make_paper_ijcisim.py`; correct them if the editor assigns different ones.
