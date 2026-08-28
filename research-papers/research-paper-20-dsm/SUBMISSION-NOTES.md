# Data Science and Management submission notes — pre-deployment validity screen

**Journal:** *Data Science and Management* (DSM), KeAi / Xi'an Jiaotong University.
Guide for authors:
<https://www.keaipublishing.com/en/journals/data-science-and-management/guide-for-authors/>

Q1, CiteScore 14.4 (7.5 in 2023), SJR ~1.13, ISSN 2666-7649, quarterly, launched
2021. **No APC** — open-access costs are covered by Xi'an Jiaotong University.
Articles publish under CC BY-NC-ND. Roughly 12 weeks to publication.

**Article type:** Original research paper (no length limit; brevity encouraged).
**Deliverables:**
`dsm_validity_screen_BLIND.pdf` — the file to upload (DSM runs **double-blind** review).
`dsm_validity_screen_authors.pdf` — same paper with names, for the record.

## Read this first — the venue is double-booked

`../research-paper-18-dsm/PLAN.md` earmarks DSM for a *different*, unstarted paper
(source-level information valuation with Shapley values). This manuscript now
claims that slot. Two papers may not go to DSM at once. Paper 18's own fallback
list names **The Journal of Finance and Data Science** (KeAi) next, which is the
obvious re-home; a note to that effect has been added to its PLAN.md.

## What this paper is, and how it differs from its siblings

Three manuscripts now exist from one study. They share the results and share
nothing else; each is written for the audience of its venue and none is a reformat
of another.

| Folder | Venue | Object of the paper | Frame |
|---|---|---|---|
| `../research-paper-17-fininnov-not-worth` | Financial Innovation (not sent) | multi-axis LLM features in finance | finance / asset pricing |
| `../research-paper-19-ijcisim` | IJCISIM (APC $1,500) | a component acceptance test in a pipeline | information systems, verification |
| **this one** | Data Science and Management | a **pre-deployment screen** and the **cost of omitting it** | analytics management, OR |

DSM is Q1 in operations research and its readers are analytics and operations
people, so this version is built around a decision a manager makes: whether to
build on a multi-attribute language-model feature, and what it costs to find out
too late. Its distinctive material, not present in the other two:

- **Section 2.1 frames the problem as a gap in data quality**, not in finance.
  Classical information-quality dimensions (Wang & Strong; Batini et al.) and
  production ML controls (Breck et al.; Polyzotis et al.) all presume a source of
  record, and a prompted attribute has none.
- **Table 2 is the omission cost, counted.** The screen is 3 fitted objects; the
  downstream programme it should have pre-empted is 408. Both counts come from
  `analysis/10_cost_accounting.py`, which derives them from the stored result
  files rather than asserting them.
- **Section 6.2** walks each downstream finding back to the screen that implied it,
  and argues the binding resource is not machine time but the sequence of
  specification choices an analyst makes while a result is still ambiguous.
- **Section 6.3, Managerial implications**, gives four rules: validity gate at
  feature onboarding; criterion named before scores are generated; budget
  attributes by validated count rather than requested count; use relevance as a
  filter, not as a weight.

## Format

`elsarticle`, `[preprint,12pt,authoryear]`, `elsarticle-harv` — DSM accepts
Elsevier's LaTeX and asks for **author-date** citations, references alphabetical
then chronological, an abstract that stands alone, and **at most six keywords**
(this paper uses six). Required declarations are all present: funding, competing
interests, ethics, CRediT authorship, generative-AI use, data availability.

Current build: **38 pages, 6 figures, 7 tables, 47 references, zero overfull boxes.**

### Blinding

DSM operates double-blind review, so the manuscript builds both ways from one
source. `build_paper.sh` writes `blindflag.tex` (`\blindtrue` / `\blindfalse`) and
compiles each. Under `\blindtrue` the author block, acknowledgements, ORCIDs and
the CRediT breakdown are suppressed; the self-reference in Section 4.5 is worded as
"earlier work by the present authors" with no citation, which is standard for a
blinded manuscript and does not identify the group.

## Reproduction

```bash
./build_paper.sh          # cost table, both PDFs, then the verifier
```

The upstream analysis (steps 01–05, 07–09) is **not** duplicated here; it lives in
`../research-paper-17-fininnov-not-worth` together with the 130 MB scoring cache.
`results/` and `figures/` were copied. Four scripts are live in this folder:

| Script | What it does |
|---|---|
| `analysis/06_tables.py` | regenerates `tables/*.tex` and `macros.tex` from `results/` and the scored panel. Takes `--cache`, defaulting to the sibling folder's `cache/`, because the panel is not copied here |
| `analysis/10_cost_accounting.py` | **new for this version** — counts fitted objects into `tables/tab_cost.tex`, `macros_cost.tex` and `results/cost_accounting.json` |
| `analysis/verify_manuscript.py` | hard pre-submission checks; the build gates on it |
| `analysis/verify_refs.py` | checks every `refs.bib` entry against CrossRef by DOI |

**Nothing in the manuscript's tables is typed by hand.** Every number quoted in the
prose is a `\newcommand` in `macros.tex` or `macros_cost.tex`, and
`analysis/verify_manuscript.py` fails the build if a used macro is undefined, if a
figure or input is missing, if the log carries an error, an undefined reference or
an overfull box, or if the abstract falls outside 120–300 words.

`analysis/verify_refs.py` checks `refs.bib` **by DOI**, not by title — title search
confidently returns the SSRN or NBER preprint instead of the journal version. The
six references added for this version (Campbell & Fiske 1959; Cronbach & Meehl
1955; Wang & Strong 1996; Batini et al. 2009; Polyzotis et al. 2018; Breck et al.
2017) all verify OK. Three long-standing flags are benign and unchanged from the
parent paper: `garcia2013` (accent encoding in the CrossRef record), `elyaniv2010`
(JMLR has no DOI), `bailey2017` (journal issue year ambiguity).

## LaTeX gotchas specific to this build

- **`\@floatboxreset` resets the font size at `\begin{table}`**, so wrapping
  `\input{tables/tab_data}` in `{\footnotesize …}` has no effect at all. The tables
  were generated for the wider Springer measure and overflow elsarticle's 12pt
  text block; the fix is `\AtBeginEnvironment{tabular}{\footnotesize}`, which hooks
  the tabular *inside* the float, after the reset. That took `tab_data` from 68.7pt
  overfull to zero.
- **`microtype` needs a scalable font.** With the default bitmap Computer Modern,
  pdfTeX aborts at the first shipout with "auto expansion is only possible with
  scalable fonts". Loading `lmodern` + `[T1]{fontenc}` fixes it, and microtype then
  clears nine of the eleven remaining overfull boxes on its own.
- **`\botrule` is a Springer macro**, not booktabs. The generated fragments use it,
  so the preamble maps it: `\providecommand{\botrule}{\bottomrule}`.
- **Table notes.** The fragments carry their notes as `\footnotetext`, which the
  Springer class prints under the table and elsarticle would push into the page
  footer. `\footnotetext` is redefined **after** `\end{frontmatter}` — the
  frontmatter itself uses the original for the corresponding-author mark.
- **`authoryear` is a class option, not a bst choice.** Without
  `\documentclass[…,authoryear]{elsarticle}` the citations render as `[45, 5]` even
  with `elsarticle-harv`.
- **Never write `.tex` through a bash heredoc**: `\\` collapses to `\` and every
  table row breaks silently. Use an editor tool or a Python file.

## Before submitting — author actions

1. Author details are the author-confirmed ones from `../AUTHORS.md`. The two
   authors sit on **different** e-mail domains (`fcrit.ac.in`, `fragnel.edu.in`)
   because they are at different institutes; never reconstruct one from the other.
2. Upload `dsm_validity_screen_BLIND.pdf`. Enter author names, affiliations, ORCIDs
   and the CRediT roles in the submission system, not in the manuscript.
3. Figures are also wanted as separate editable files at 300–1000 dpi; the PDFs in
   `figures/` are vector and satisfy this. Captions are in the manuscript.
4. DSM asks that its stylesheet be used while drafting — that applies to Word
   submissions; a LaTeX submission using `elsarticle` is explicitly accepted.
5. Run a similarity check before upload.
6. Resolve the paper-18 venue clash first (see the top of this file).
