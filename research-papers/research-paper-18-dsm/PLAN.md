# Paper 18 — PLAN (not started)

> **VENUE CLASH, 2026-08-20.** DSM has since been claimed by
> `../research-paper-20-dsm`, which is the multi-attribute validity-screen study
> rewritten for that journal. Two papers cannot sit at DSM at once, so this one
> needs the next name on its own fallback list in §12: **The Journal of Finance
> and Data Science** (KeAi, free) — or DSM later, once paper 20 has resolved.
> Nothing else in this plan changes; the topic, the data and the build order are
> unaffected by the venue move.

**Target:** *Data Science and Management* (DSM), KeAi / Xi'an Jiaotong University.
Q1, SJR 1.134, CiteScore 14.4, **no APC** (university-funded), ~12 weeks to
publication, `elsarticle` LaTeX accepted, **author–date** references, no length limit.
Guide for authors: <https://www.keaipublishing.com/en/journals/data-science-and-management/guide-for-authors/>

**Working title:** *Not all news is worth buying: source-level information valuation
and budget-constrained feed selection for text-driven forecasting*

**Status:** planned only. Nothing has been run. This file is the complete build
order — pick it up cold from here.

---

## 1. Why this topic and not the others

DSM is Q1 in **Operations Research**, not finance. Its readers are analytics and
operations people, so the paper must be a *method for a decision a manager makes*,
validated on large real data — not a return-prediction study. The decision here is
**data procurement**: which news feeds are worth paying to ingest.

The enabling fact, verified in the repo on 2026-08-20:

> `research-papers/research-paper-14-aece-not-sent/data-cache/corpus.parquet` has a
> `Publisher` column — **877 distinct sources, 807,784 headlines, 300 symbols,
> 2009-02-14 to 2020-06-11 (11.3 years)** — and it has **never been used as a
> variable** in any paper in this repo. It appears once, as a descriptive count, at
> `analysis/03_build_panel.py:158`.

Concentration is already visible: top 10 sources = **71.8%** of all headlines;
**60 sources have >=1,000 headlines**, 188 have >=100.

Candidates that were rejected, and why — do not revisit without a new reason:

| Rejected topic | Reason |
|---|---|
| Multi-instrument measurement validity (MTMM) | Overlaps paper 15's II.4 "identification from cross-scorer covariance" and paper 17's validity section |
| Retraining cadence law | It *is* paper 8, re-homed; and 673 ledger rows is thin for Q1 |
| Deployed interval reliability | Overlaps papers 11/12; 673 rows |
| Volatility rematch of the 3-axis decomposition | Reads as an incremental sequel to paper 17 |
| Arrival vs content decomposition | Partly inside paper 17 |

## 2. Overlap firewall

This paper must not re-tread papers 14/15 and 17, which use the same corpus.

| Paper | Its object | Firewall rule for paper 18 |
|---|---|---|
| 14/15 (matched filter) | temporal response kernel, attenuation correction, LMS loop gain | Paper 18 uses a **contemporaneous** aggregate only. Do **not** estimate a kernel or a loop gain. Cite 14/15 for the measurement-error background in one paragraph |
| 17 (MIG validity) | whether the LLM's three axes measure distinct constructs | Paper 18 treats the scorer as **fixed and given**; its object is the *source*, not the instrument. The four scorers appear only as a robustness dimension |

Code reuse is fine and encouraged (staleness/TF-IDF, verification scripts, palette,
`macros.tex` discipline). Prose and results reuse is not.

## 3. Research questions

1. **Q1 (valuation).** What is the marginal contribution of an individual news source
   to out-of-sample forecast skill, and how concentrated is that value across sources?
2. **Q2 (procurement).** Under a budget, which subset of sources should a team ingest,
   and how much skill is lost by the naive policy of ranking sources by volume?
3. **Q3 (non-stationarity).** Is source value stable over 11 years, and does adaptive
   re-selection beat a portfolio fixed ex ante?

## 4. Contributions to claim

1. A **source-level information-value framework** for text-derived features: a
   cooperative-game formulation in which sources are players and forecast skill is the
   characteristic function.
2. An **exact sufficient-statistic evaluation scheme** that makes Shapley estimation
   tractable at this scale (Section 6.2) — the methodological workhorse, and reusable
   for any additively-aggregated text feature.
3. **Empirical evidence on value concentration** and on the gap between volume and
   value, from 807k headlines and 877 sources over 11 years.
4. A **budgeted selection procedure** with an empirical submodularity diagnostic, and
   an **adaptive online variant**, benchmarked against static and volume-ranked policies.
5. **Instrument-robustness**: every ranking replicated across four independent scorers,
   so conclusions are about sources rather than about one sentiment model.

## 5. Data — all on disk, zero API cost

| Asset | Path | Shape |
|---|---|---|
| Corpus with `Publisher` | `../research-paper-14-aece-not-sent/data-cache/corpus.parquet` | 807,784 x 4 (Date, Article_title, Stock_symbol, Publisher) |
| Four scorers per title | `.../data-cache/headline_scores.parquet` | 526,668 x 5 (m1 FinBERT, m2 Loughran–McDonald, m3 VADER, m4 Harvard IV-4) |
| Symbol-session panel | `.../data-cache/panel.parquet` | 786,347 x 17 — `ret`, `ret_adj`, `beta`, `lrv`, `lrv_innov`, `m1..m4`, `n_news`, `has_news`, z-scores |
| Prices | `.../data-cache/prices.parquet` | — |
| LLM three-axis scores (robustness only) | `../research-paper-17-fininnov/cache/events.parquet` | 134,037 x 5 (symbol, session, nu, mu, s) |

Environment: `.venv/Scripts/python.exe` (Python 3.11 — 3.13 has no wheels for the
pinned numpy/TF stack). `pyarrow` is only in that venv; the system python cannot read
these parquets.

## 6. Method

### 6.1 Setup

Let `P` be the source universe. For a portfolio `S ⊆ P`, build the text feature for
symbol `i` and session `t` using only headlines whose source is in `S`:

    A_S(i,t) = ( sum_{p in S} sum_p(i,t) ) / ( sum_{p in S} cnt_p(i,t) )

where `sum_p(i,t)` is the total scorer output and `cnt_p(i,t)` the headline count
contributed by source `p`. Timing follows the audited convention: a headline belongs
to the session whose 16:00 ET close first follows its UTC stamp, so session `t`'s
aggregate is strictly pre-close and predicts from `t+1`. **Do not change this** — a
previous study in this project produced a spurious result from a timezone join.

**Characteristic function.** `v(S)` = out-of-sample forecast skill of `A_S`, on a
walk-forward, net of a price-only baseline. Skill is measured as information
coefficient (primary), with incremental R-squared and precision at fixed coverage as
secondary. `v(empty) = 0` by construction.

**Target choice — decided in advance.** Primary target is **realised-volatility
innovation `lrv_innov`**, secondary is **`ret_adj`**. Rationale is principled, not
opportunistic: paper 17 established that the material/magnitude content of financial
text is what validates cleanly, while direction does not, and paper 14 found the
volatility loop's IC roughly doubles where the return loop shuts itself off. Volatility
is also the more decision-relevant target for a risk desk. **Both are reported**; the
choice of primary is fixed here, before any result is seen, and the plan says so in
the manuscript.

### 6.2 The evaluation trick (this is the methodological contribution)

Because the aggregator is a ratio of sums, `v(S)` never requires re-reading the
corpus. Precompute once a sparse array over `(source, symbol, session)` holding
`(sum_p, cnt_p)`. Any portfolio's feature is then a sparse column-sum. This turns each
`v(S)` evaluation into a sum plus a regression, which is what makes tens of thousands
of Monte-Carlo permutations affordable. State this as a proposition with the exact cost
in both schemes.

### 6.3 Shapley estimation

    phi_p = sum over S subset of P\{p} of [ |S|!(|P|-|S|-1)! / |P|! ] * ( v(S ∪ {p}) - v(S) )

Exact enumeration is infeasible, so use **Monte-Carlo permutation sampling**
(Castro, Gómez & Tejada 2009) restricted to the **60 sources with >=1,000 headlines**,
with everything else pooled into a synthetic "tail" player. Report standard errors and
a convergence diagnostic; use stratified/antithetic permutations to cut variance. Fix
the seed and log it.

### 6.4 Budgeted selection

Two cost models, both reported, because feed prices are not in the data and inventing
one number would be worse than showing the answer under both:
- **C1 volume cost**: cost proportional to documents ingested (processing/storage).
- **C2 unit cost**: each source costs the same (subscription-like).

Selection is cost-benefit greedy (CELF / Khuller–Moss–Naor). Submodularity is not
assumed — **test it**: sample many `(S, p, q)` triples and report the empirical
violation rate for diminishing returns. If violations are rare, the `(1 - 1/e)`
guarantee is quoted with that caveat; if common, greedy is presented as a heuristic
validated out of sample. Either outcome is publishable; do not fudge this.

Baselines: volume-ranked, random, single-best-source, all-sources, price-only.

### 6.5 Stability and adaptation (Q3)

- Split 2009–2020 into rolling windows (annual, plus a 2-year robustness variant).
- Per-window optimal portfolios; report **Kendall tau** between window rankings and
  portfolio turnover.
- Online policy: rolling re-optimisation with a lookback, and a combinatorial bandit
  (successive elimination under a budget) over sources.
- Benchmark against the ex-ante static portfolio and against the hindsight static
  oracle. Report regret. **This is the OR contribution and the reason DSM is the
  right venue** — give it a full results subsection, not a paragraph.

### 6.6 Confounds that referees will raise — handle each explicitly

1. **`Publisher` mixes outlets with bylines.** "Seeking Alpha", "Zacks", "GuruFocus"
   are outlets; "Paul Quintaro", "Lisa Levin", "Charles Gross", "Monica Gerson" are
   Benzinga reporters. Hand-classify the top 60 into outlet / byline / aggregator,
   store the mapping as a committed CSV so it is auditable, and report the valuation
   **at both levels**. Do not pattern-match this.
2. **Coverage confound.** Large sources cover more symbols and more sessions. Report
   value *per document* and value *per covered symbol-session* alongside total value,
   and control coverage in the cross-source regressions.
3. **Originality / reprints.** A source that reprints another's story late adds
   nothing. Measure source-level originality by reusing paper 17's staleness code
   (`research-paper-17-fininnov/analysis/09_convergent_validity.py` — TF-IDF cosine to
   the prior 30 days, strictly causal) aggregated per source, and use it as an
   explanatory variable for why some sources are valuable. Beware the confound found
   there: `n_prior_30d` correlates with staleness by construction (corr -0.209), and
   the relationship is **non-monotonic** — plot the decile profile before quoting any
   rank correlation.
4. **Symbol-mix specialisation.** "ETF Professor" covers ETFs. Apply the paper-17 ETF
   exclusion (EWJ, SLV, GXC, PGJ, QQQ, FXP, EWI, YINN) and add symbol fixed effects.
5. **Multiple comparisons.** 60 sources ranked on noisy skill invites false discovery.
   Apply Benjamini–Hochberg to the per-source significance claims and say so.

### 6.7 Inference

Two-way clustered standard errors by date and symbol (Cameron–Gelbach–Miller),
cross-checked with Fama–MacBeth plus Newey–West. Block bootstrap over whole dates for
any difference between portfolios. Keep Shapley Monte-Carlo error separate from
sampling error and report both. Bootstrap p-values must be capped at 1.0 and reported
alongside the share of exact ties (paper 17 hit a p-value of 1.013 without this).

## 7. Pipeline — mirrors the paper-17 layout, which worked

| Step | Script | Output |
|---|---|---|
| 1 | `01_build_source_panel.py` | `cache/source_stats.parquet` — (source, symbol, session) -> (sum, cnt) per scorer |
| 2 | `02_source_descriptives.py` | coverage, volume, originality, outlet/byline class; `cache/source_meta.csv` |
| 3 | `03_value_functional.py` | `v(S)` evaluator + walk-forward harness; unit tests against a brute-force recompute |
| 4 | `04_shapley.py` | `results/shapley.csv` with SEs + convergence trace |
| 5 | `05_budget_selection.py` | `results/frontier.csv`, `results/submodularity.json` |
| 6 | `06_stability.py` | `results/window_rankings.csv`, Kendall tau matrix |
| 7 | `07_adaptive.py` | `results/adaptive.json`, regret curves |
| 8 | `08_robustness.py` | 4 scorers x 2 aggregators x 2 cost models x 2 targets |
| 9 | `09_figures.py`, `10_tables.py` | `figures/*.pdf`, `tables/*.tex`, `macros.tex` |
| 10 | `verify_manuscript.py`, `verify_refs.py` | copied from paper 17 |

**Non-negotiable discipline, carried from paper 17:** no number is typed by hand into
the manuscript. Tables are generated fragments; every in-prose number is a
`\newcommand` in `macros.tex`. If an analysis moves, the text moves with it.

**Step 3 needs a correctness test.** The sufficient-statistic shortcut is the whole
paper; write a test that rebuilds `A_S` from raw headlines for a handful of random
portfolios and asserts equality with the sparse-sum path.

## 8. Figures (target 7) and tables (target 6)

Figures:
1. **Lorenz curves — volume vs information value.** The money shot. If the two curves
   separate, the paper's headline is made.
2. Shapley value by source, top 25, with Monte-Carlo error bars.
3. Value vs volume, log-log scatter, with value-per-document as a second panel;
   label over- and under-performers.
4. Budget–performance frontier: greedy vs volume-ranked vs random vs all-sources.
5. Source rank stability: heatmap by year plus the Kendall tau matrix.
6. Adaptive vs static: cumulative skill and regret.
7. Instrument robustness: rank correlation of source valuations across the four scorers.

Tables:
1. Corpus and source descriptives.
2. Top-20 sources: volume, Shapley, value/doc, originality, coverage, outlet/byline.
3. Out-of-sample portfolio performance at each budget level.
4. Submodularity diagnostic and greedy guarantee.
5. Adaptive vs static, with block-bootstrap CIs.
6. Robustness grid across scorers, aggregators, cost models, targets.

Use the validated palette from paper 17's `05_figures.py`
(`#0072B2 #D55E00 #009E73 #7B3294`, `#666666` for reference lines only) — it already
passes the dataviz validator. Load the `dataviz` skill before writing chart code.

## 9. Manuscript

`elsarticle`, author–date (`elsarticle-harv`). DSM imposes no length limit but asks
for concision; aim 30–35 pages including floats.

1. Introduction — the procurement decision, why volume is the default proxy, what we find
2. Related work — data valuation and Shapley-for-data; news and asset prices; source
   credibility and media economics; feature selection under budget
3. Problem formulation — the cooperative game, the characteristic function, the two cost models
4. Data — corpus, sources, the outlet/byline distinction, timing convention
5. Method — evaluation scheme (with the cost proposition), Shapley estimation, budgeted
   selection, adaptive policy
6. Results — Q1 concentration, Q2 frontier, Q3 stability and adaptation
7. Robustness — four scorers, aggregators, cost models, targets, subperiods
8. Managerial implications — an explicit procurement rule of thumb
9. Limitations — cost is modelled not observed; US/Benzinga-sourced corpus ends 2020-06;
   valuation is conditional on the chosen scorer and aggregator
10. Conclusion

**Author block:** use `../AUTHORS.md` verbatim. The two authors are at different
institutes on different e-mail domains (`fcrit.ac.in` vs `fragnel.edu.in`) — never
reconstruct one address from the other's pattern.

**References:** real, published, Google-findable works only. Never cite this project's
own unpublished or under-review papers. Verify by **DOI**, not title — title search
returns the SSRN/NBER preprint. Books and proceedings get flagged MANUAL.

## 10. Timeline — about 3 weeks

| Days | Work | Gate |
|---|---|---|
| 1–3 | Steps 1–2: source panel, descriptives, **Lorenz curves** | **GO/NO-GO.** If value tracks volume almost exactly, switch to the fallback framing in §11 R1 before investing further |
| 4–7 | Step 3–4: value functional (with its correctness test) and Shapley | Shapley SEs small enough to rank the top 20 |
| 8–11 | Step 5: budgeted selection, submodularity diagnostic, frontier | — |
| 12–15 | Steps 6–7: stability and adaptive policy | — |
| 16–18 | Step 8: robustness grid | — |
| 19–21 | Steps 9–10: figures, tables, manuscript, verification | all hard checks pass |

## 11. Risks, and what to do about each

- **R1 — value turns out proportional to volume.** The headline weakens but the paper
  survives: value-per-document and originality still separate sources, and "volume is a
  sufficient statistic for value" is itself a clean, useful procurement finding. Reframe
  around the frontier and the stability result rather than the concentration gap.
- **R2 — absolute signal is too weak to distinguish portfolios.** Real risk: paper 17's
  ICs were around 0.02 with a nearly flat surface. Mitigations, in order: volatility is
  already the primary target for exactly this reason; report *relative* rankings, which
  can be sharp even when absolute skill is small; use the full 300-symbol universe
  rather than paper 17's 35; and lean on the block bootstrap rather than eyeballing gaps.
- **R3 — Shapley Monte-Carlo too noisy.** Mitigated by the sufficient-statistic scheme,
  stratified sampling, and pooling sub-1,000-headline sources into one tail player.
- **R4 — submodularity fails.** Report the violation rate; greedy becomes an empirically
  validated heuristic. Do not quietly keep quoting the `(1-1/e)` bound.
- **R5 — outlet/byline ambiguity attacked by a referee.** Pre-empt with the committed
  classification CSV and results at both levels.
- **R6 — corpus ends 2020-06 and is US/Benzinga-sourced.** Own it in §9 as paper 17 owns
  "validation market is not deployment market". Do not claim generality to Indian markets.

## 12. Fallback venues if DSM declines

In order: **Journal of Finance and Data Science** (KeAi, free) -> **Digital Finance**
(Springer) -> **Financial Innovation** (Springer/SWUFE, free, though it is a harder sell
for anything that reads as negative). Do **not** pay $1,500 for IJCISIM — SJR 0.23 and
16 of its 18 issues in 2026 are special issues.
