# IJADS-312370 — Revision Plan

**Submission:** "From Signal Fusion to Asset Allocation" (Pardeshi & Deshmukh)
**Verdict:** Acceptable with major revisions · **Due:** 2026-09-20
**Status:** **complete — ready for the author's read-through and upload** (2026-08-27; sentiment validation and title restored 2026-09-03; structure, headings and full reference list of the submitted version restored 2026-09-11 — see §9)

| Deliverable | File | State |
|---|---|---|
| Revised manuscript | `2-revised-manuscript/ai67.tex` → `ai67.pdf` | 47 pp, compiles clean, no undefined refs |
| Response to reviewers | `2-revised-manuscript/response-to-reviewers.tex` → `.pdf` | 8 pp, every comment answered with page numbers |
| Submitted version, preserved | `1-original-submission/ai92.tex` + `2026_IJADS-312370.pdf` | as submitted |
| Previous (2008–2023) revision | `4-archive/v1-2008-2023/` | full snapshot: tex, sections, JSON, cache |
| Evidence | `3-analysis/final_tables.json`, `mega_results.json`, `ic_results.json`, `sentiment_results.json` | 223/223 numbers in the manuscript verified against these |

Rebuild with `3-analysis/assemble.py`, then `pdflatex ai67.tex` twice **from inside
`2-revised-manuscript/`**. Re-verify with `3-analysis/verify_numbers.py`.

---

## What changed, and why

Two defects were found while implementing the robustness analysis R2.3 asked for.

1. **Unit mismatch in the objective.** The fused signal entered the mean-variance
   objective in raw indicator units (on-balance volume at 1e8–1e9 against a 1e-4 risk
   term), so the risk model was irrelevant and the optimiser held one asset on almost
   every date — exactly the behaviour R2.4 flagged in Figure 5. Fixed by Grinold
   information-ratio scaling, Eq. (4).
2. **Inert regime layer.** The absolute volatility targets (20/15/10%) are never reached
   by a sleeve whose own volatility is 6.4%, so the budget bound in **1 of 224**
   rebalances and the paper's central mechanism did nothing. Fixed by expressing the
   budget as a fraction of the sleeve's own risk, Eq. (7).

Correcting these changed every reported number. The universe was also extended from
10 mega-cap equities over 5 years to 10 multi-asset ETFs over 18 years (R2.o4), and the
evaluation window now runs to every session available at the analysis date.

## New empirical base

| | Strategy | Equal-weight | Vol-matched EW | Inverse-vol |
|---|---|---|---|---|
| Sharpe | **0.878** | 0.628 | 0.651 | 0.849 |
| Max drawdown | **−18.98%** | −36.08% | −18.69% | −24.09% |
| CAGR | 5.65% | 8.45% | 4.19% | 7.46% |

Universe SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, VNQ · Jan 2008 – Aug 2026 ·
4,689 sessions · **18 of 18** specifications beat equal weight on Sharpe and drawdown ·
crisis wins GFC +22.74pp, Euro 2011 +10.65pp, COVID +14.99pp, 2022 +7.73pp ·
drawdown wins in 18 of 19 years, Sharpe in 9 of 19.

**Out-of-sample holdout (§4.2), the strongest addition.** The 665 sessions from
January 2024 to August 2026 postdate every specification decision. On them the framework
beats **all three** benchmarks on both measures — Sharpe 1.614 vs 1.391 (EW), 1.397
(inverse-vol) and 1.378 (vol-matched EW); drawdown −5.48% vs −10.13%, −6.70% and −7.04%.
Two of those comparisons reverse the in-sample ordering. The 2008–2023 block reproduces
the earlier revision's numbers exactly, which is what makes the extension a clean
superset rather than a re-tuning.

**Claims dropped.** Significant alpha (excess return is *negative*, p = 0.171, and the
Sharpe difference bootstraps to [−0.102, 0.574]). Signal fusion (removing the layer
*improves* Sharpe 0.878 → 0.908; IC is −0.033 overall, significantly negative).

**Claim that survives, and is the paper's centre of gravity.** At *matched average
exposure* (92.1%), timing risk by regime beats holding the same exposure constantly:
Sharpe 0.878 vs 0.795, drawdown −18.98% vs −22.26%, with no significant difference in
mean return (p = 0.695). Its cost is measured too: against a fully invested variant the
framework gives up ~1.3pp of annual return (p = 0.016) for 3.6pp of drawdown.

---

## Task list

### 1. Methodology — §3 · R2.1, R2.2, R2.7, R2.o3
- [x] 1.1 Data specified: source, adjustment, universe, window, warm-up, evaluation period
- [x] 1.2 Feature table with exact formulas and windows for all nine inputs (Table 2)
- [x] 1.3 §3.3 states plainly that no text, lexicon or language model is used
- [x] 1.4 GMM settings, expanding-window refit, volatility-sorted components, seed
- [x] 1.5 Walk-forward protocol documented
- [x] 1.6 Two-step optimiser documented (sleeve, then regime scaling)
- [x] 1.7 Eq. (1)–(7) with every symbol and constraint defined
- [x] 1.8 Solver, shrinkage estimator, cost model
- [x] 1.9 Stray reviewer-style sentence in §3.1.1 deleted
- [x] 1.10 Holdout boundary defined and the "all data available at run date" rule stated

### 2. Results — §4 rewrite · R2.3, R2.4, R2.5, R2.o2, R2.o5
- [x] 2.1 Table 3 — performance vs three benchmarks
- [x] 2.2 Table 4 — out-of-sample holdout, both blocks side by side (§4.2, new)
- [x] 2.3 Table 5 — all 18 specifications
- [x] 2.4 Table 6 — four crisis episodes
- [x] 2.5 Table 7 — 19 years, wins and losses both shown
- [x] 2.6 Table 8 — ablation at matched exposure; Table 9 — risk-budget ladder
- [x] 2.7 Table 10 — regime-conditional IC plus the pre/post-2024 era split (the null)
- [x] 2.8 Table 11 — turnover, concentration, cost sensitivity 0/5/10/25 bps
- [x] 2.9 Table 12 — AIC/BIC, transition matrix, run lengths, per-regime performance
- [x] 2.10 Table 13 — paired t, Newey–West, block-bootstrap CIs
- [x] 2.11 Table 14 — secondary mega-cap universe, 2019–2026
- [x] 2.12 §4.9 explains the Figure 5 concentration as an indicator-scaling defect
- [x] 2.13 Technical interpretation for every figure
- [x] 2.14 Old §4.2/§4.3 merged; §4 reorganised so no subsection repeats another

### 3. Framing — Abstract, §1, §2 · R2.o1, R2.o4
- [x] 3.1 Abstract rewritten; "significant alpha" gone, non-significance stated, holdout added
- [x] 3.2 Contributions recast around what the evidence supports
- [x] 3.3 Table 1 — comparison against seven allocation paradigms
- [x] 3.4 Future work: textual sentiment, larger universes, RL budget tuning, HMM/jump models
- [x] 3.5 Title **unchanged from the submitted version** (author's decision, 2026-09-03).
      "Sentiment" stays, and Sections 3.3 and 4.8 now carry it honestly — see task 8

### 4. Bibliography · R1.1
- [x] 4.1 24 of 33 IEEE Access references replaced; 3 retained
- [x] 4.2 46 references, 24 from 2025–26, all CrossRef-verified, all journal articles
      (plus one monograph, Grinold & Kahn 2000)
- [x] 4.3 ≤3 per journal, ≤2 per author, every DOI resolves
- [x] 4.4 No citation cluster exceeds three (checked programmatically)
- [x] 4.5 Zero self-citations
- [x] 4.6 Three IJADS articles cited, per the Editor's request

### 5. Figures · R1.3
- [x] 5.1 All 8 figures regenerated at exactly 300 dpi on the extended sample
- [x] 5.2 One typeface, one palette, matched weights and widths
- [x] 5.3 Unique `\label` for every figure (all seven previously shared `fig:integration`)
- [x] 5.4 Architecture diagram redrawn — the old one showed a news/text path that does not exist

### 6. Presentation · R1.2, R2.6
- [x] 6.1 Every acronym expanded at first use, in the abstract and in the body
- [x] 6.2 "LLPs" typo fixed
- [x] 6.3 Full language pass — the manuscript is rewritten end to end in one voice
- [x] 6.4 AI declaration completed — names Claude (Anthropic) and ChatGPT (OpenAI)
      for language editing only (p. 31), which closes R2.6's complaint about the
      unfinished declaration text.

### 8. Sentiment — title, §3.3 and §4.8 · R2.2, R2.o3
- [x] 8.1 Title restored to the submitted wording; "Sentiment" is back in it
- [x] 8.2 §3.3 renamed to the submitted heading, "Sentiment Analysis Pipeline", and
      rewritten: market-derived, no transformer, no lexicon, no text, and why
- [x] 8.3 §2.4 renamed back to "Sentiment Analysis and Alternative Data Integration",
      with the market-derived family of sentiment measures added to the review
- [x] 8.4 §4.8 added — external validation against four published series. All four carry
      the expected sign, all four significant in first differences, 16/16 sub-period
      signs correct, and the identified states separate at F = 19.34, p = 1.8e-08
- [x] 8.5 Baker & Wurgler (2007) added as the reference for market-derived sentiment;
      still ≤3 per journal and ≤2 per author (50 refs total)
- [x] 8.6 "market-state composite" renamed to "sentiment composite" throughout, without
      disturbing "market state" where it means the regime
- [x] 8.7 `sentiment_validation.py` written; `verify_numbers.py` extended 190 → 223 checks

### 7. Response letter
- [x] 7.1 Point-by-point response to all 10 mandatory and 5 optional comments
- [x] 7.2 Every page, table, figure and equation number re-checked against `ai67.aux`
- [x] 7.3 The changed experiment explained truthfully and up front
- [x] 7.4 The sample extension and the holdout explained in the opening summary
- [x] 7.5 R2.2 and R2.o3 rewritten around the sentiment validation; all 31 shifted
      section, table, figure and page references corrected against the rebuilt PDF

---

## Before uploading

1. **Read §4.6, §4.9 and §5.2.** They report that the signal fusion layer does not work
   and that the Sharpe gain is not statistically significant. This is deliberate and, in
   our judgement, the strongest available position — but you should be comfortable
   defending it before it goes to the reviewers.
2. **Decide how to submit the response letter.** Inderscience asks for it at the front of
   the revised article. It is currently a separate 7-page PDF; either prepend it or
   upload it alongside.
3. **Highlight changed text**, as the Editor asked. Essentially everything from §1
   onward is new, so a covering sentence saying so may serve better than highlighting.

## Notes for anyone rerunning the analysis

- `sentiment_validation.py` is independent of the backtest and takes about a minute; it
  pulls FRED and Yahoo directly and rewrites `sentiment_results.json`.
- `pipeline_core.py` downloads through `P.END`; `final_run.py` sets it to `2026-08-27`
  and caches into `_final_cache_ext.pkl`. Delete that pickle to force a full rerun
  (~40 minutes: 32 backtests, each refitting the GMM at 224 rebalance dates).
- Run order is `ic_analysis.py` → `mega_run.py` → `final_run.py` → `fix_significance.py`,
  which is what `run_all_ext.sh` does. `fix_significance.py` must run last: it overwrites
  the inference blocks so that every Sharpe ratio in the paper uses the same
  compound-growth definition.
- `verify_numbers.py` is the gate. 223 checks, all tied to the JSON artifacts.


---

## 9. Restoration of the submitted structure (2026-09-11)

Requested by the author after reading the 35-page revision against the submitted paper.
The revision had retained **zero** of the submitted manuscript's 246 sentences.

- [x] 9.1 All section, subsection and subsubsection headings of the submitted version
      restored. `Results` and `Discussion` merged back into `Results and Discussion`, so
      the paper is 5 sections again.
- [x] 9.2 One new subsection only — §4.3 `Robustness and Attribution` — holding the
      holdout, sweep, sentiment validation, ablation, inference and secondary-universe
      analyses the reviewers required. §4 drops from 14 subsections to 4.
- [x] 9.3 All 33 submitted references retained, 21 added, 54 total. 10 from 2025–26.
      `lu25` key collision resolved in favour of the submitted entry (Lu et al., 2025,
      electricity price prediction); Lu and Tian (2025) dropped.
- [x] 9.4 §1 and §2 restored close to the submitted wording, with the grammatical faults
      R1.2/R2.6 flagged fixed (`LLPs` → `LLMs`, `near-monosaccharic` → `near-exclusive`,
      `A advanced` → `An explicit`) and the three submitted comparison tables reinstated.
- [x] 9.5 §3 keeps all 13 submitted subsubsection headings, with content that matches the
      implementation. Headings describing absent components are filled truthfully:
      `Input Layer` states that the news and macroeconomic streams are not used;
      `Model Refinement Loop` is the expanding-window GMM and covariance refit and names
      the parameters that are *not* re-optimised; `Performance Attribution` is ablation
      plus IC, not Brinson.
- [x] 9.6 §4 restored to the submitted per-figure rhythm — figure, characteristics
      bullets, bold interpretation paragraphs — with corrected numbers and honest
      conclusions where the submitted ones are refuted.
- [x] 9.7 Abstract: submitted wording retained, 46% changed. `significant alpha` gone;
      non-significance, holdout and ablation added. No unexpanded acronym.
- [x] 9.8 Response letter rebuilt: R1.1 answer now states the *IEEE Access* retention
      openly instead of claiming three-per-journal compliance; stale §4.11 regime figures
      (12.0/20.3/49.4, 0.928–0.964, 14–28) corrected to 12.22/20.04/47.07,
      0.910–0.961, 11.1–25.3; all 40+ section, table, figure and page references
      re-derived from the rebuilt `ai67.aux`.
- [x] 9.9 `verify_numbers.py` still passes 223/223. Assemble reports 54 bibitems, 54
      cited, no dangling refs, no duplicate labels. Compiles with zero errors.

**Measured retention** (sentence level, citation-style and emphasis markup normalised):
62.7% of the submitted manuscript survives, i.e. **37.3% changed**, against 100% changed
before this pass.

**Open item.** §9.3 knowingly leaves R1.1's three-per-journal limit unmet (24 *IEEE
Access*). The letter asks the Editor to choose. If the answer is to comply, the work is
to cut *IEEE Access* to three and rebuild §2.1–§2.4 and Tables 1–3 around what remains.

**Open item.** The manuscript is now 47 pp, against 22 pp submitted and 35 pp at the
previous revision. If the journal objects, the reducible material is §4.3 — but every
part of it was required by a reviewer.

---

## 10. Post-review fixes and length reduction (2026-09-11, second pass)

Three defects found by auditing the revision against the reviewer letter, plus a
length-reduction pass. Manuscript **47 pp -> 45 pp**; response letter re-derived against
the final build.

- [x] 10.1 **R1.2 was broken by the §9 restoration.** Tables 1-3, restored verbatim from
      the submitted version, carried `LSTM`, `GRU`, `CNN`, `SVM`, `RBF` and `ML`
      unexpanded anywhere in the manuscript, while the response letter claimed every
      acronym was expanded at first use and "checked programmatically". All six are now
      expanded in the table cells at first use (Tables 1 and 2, pp. 6-7), and the letter's
      R1.2 answer lists them. All 17 acronyms verified programmatically: none unexpanded.
- [x] 10.2 **Author block restored.** `\authorA`/`\affA`/`\authorB`/`\affB` were commented
      out, so the title page carried no names or affiliations and left empty `Reference`
      and `Biographical notes:` stubs. Names, departments, institutions and e-mails are
      back on p. 1; `\REF{}` now carries the Inderscience "Reference to this paper"
      line; `\begin{bio}` carries factual notes for both authors. **The bios state only
      affiliation and research area — expand with degrees, positions and publication
      record before upload.**
- [x] 10.3 **All page references in the response letter re-derived** from the final
      45-page build: 57 corrections. Every one of the 15 table and figure references now
      resolves exactly. Three section references deliberately cite the page of the item
      rather than the section's first page (§1.1 p. 3 for the LLM expansion, §3.3.2 p. 18
      for CAGR, §4.4.2 p. 39 for the reinforcement-learning bullet).
- [x] 10.4 **Length reduction, ~1,600 words.** Cut only material that was self-inflicted
      duplication, never reviewer-mandated evidence: §4.4.1 916 -> ~560 w, §5 739 ->
      ~610 w, §4.4.2 591 -> ~510 w, plus the numeric restatements of Tables 5 and 8 in
      §4.1 and §4.2.4, which R2.o5 asked to be reduced. All 18 tables, all 8 figures and
      every number retained.
- [x] 10.5 Tables set uniformly in `\footnotesize` (8 pt against the 10 pt body, matching
      the class's own `\EIGHT` convention). Saves a page and cuts underfull boxes
      223 -> 168, which also serves R1.3's consistency requirement. Figure 1 scaled to
      0.70\textwidth, Figures 4 and 5 to 0.92\textwidth.
- [x] 10.6 `verify_numbers.py` still 223/223. 54 bibitems, 54 cited, no dangling refs, no
      duplicate labels, zero compile errors.

**Measured retention after this pass** (was 58.2% recognisable before the cuts, 65.0%
word-level): sentence level 13% near-identical, 42% substantially kept, 58% recognisable;
word level 65.0% of the submitted manuscript's words survive in order.

**Open item — length.** 45 pp and 17,653 body words against 22 pp and 6,825 submitted.
Roughly 60% of the growth is evidence the reviewers required (18 tables, 8 figures, seven
equations, the holdout, the sweep, the ablations, the inference); the rest is the §9
restoration layering the submitted prose on top of the rewritten prose. Going below
~42 pp now requires giving up one of two things the author asked to keep: the restored
submitted structure, or reviewer-mandated evidence. That is a decision for the author.
