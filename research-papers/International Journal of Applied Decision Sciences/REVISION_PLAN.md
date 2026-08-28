# IJADS-312370 — Revision Plan

**Submission:** "From Signal Fusion to Asset Allocation" (Pardeshi & Deshmukh)
**Verdict:** Acceptable with major revisions · **Due:** 2026-09-20
**Status:** **complete — ready for the author's read-through and upload** (2026-08-27)

| Deliverable | File | State |
|---|---|---|
| Revised manuscript | `single column/ai67.tex` → `ai67.pdf` | 33 pp, compiles clean, no undefined refs |
| Response to reviewers | `response-to-reviewers.tex` → `.pdf` | 7 pp, every comment answered with page numbers |
| Submitted version, preserved | `revision-analysis/ai67_submitted.tex.bak` | untouched backup |
| Previous (2008–2023) revision | `revision-analysis/_v1_2008_2023/` | full snapshot: tex, sections, JSON, cache |
| Evidence | `revision-analysis/final_tables.json`, `mega_results.json`, `ic_results.json` | 190/190 numbers in the manuscript verified against these |

Rebuild with `revision-analysis/assemble.py`, then `pdflatex ai67.tex` twice **from inside
`single column/`**. Re-verify with `revision-analysis/verify_numbers.py`.

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
- [x] 3.5 Title decided — "Sentiment" dropped (author's decision, 2026-08-26)

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
- [ ] 6.4 **AI declaration — needs the author.** The manuscript carries
      `[AUTHORS: NAME THE TOOL(S) ACTUALLY USED]` on p. 29. Only you know which tools
      were used; naming one would be a guess, so it is left for you to fill in.

### 7. Response letter
- [x] 7.1 Point-by-point response to all 10 mandatory and 5 optional comments
- [x] 7.2 Every page, table, figure and equation number re-checked against `ai67.aux`
- [x] 7.3 The changed experiment explained truthfully and up front
- [x] 7.4 The sample extension and the holdout explained in the opening summary

---

## Before uploading

1. **Fill the AI declaration** on p. 29 of the manuscript.
2. **Read §4.6, §4.8 and §5.2.** They report that the signal fusion layer does not work
   and that the Sharpe gain is not statistically significant. This is deliberate and, in
   our judgement, the strongest available position — but you should be comfortable
   defending it before it goes to the reviewers.
3. **Decide how to submit the response letter.** Inderscience asks for it at the front of
   the revised article. It is currently a separate 7-page PDF; either prepend it or
   upload it alongside.
4. **Highlight changed text**, as the Editor asked. Essentially everything from §1
   onward is new, so a covering sentence saying so may serve better than highlighting.

## Notes for anyone rerunning the analysis

- `pipeline_core.py` downloads through `P.END`; `final_run.py` sets it to `2026-08-27`
  and caches into `_final_cache_ext.pkl`. Delete that pickle to force a full rerun
  (~40 minutes: 32 backtests, each refitting the GMM at 224 rebalance dates).
- Run order is `ic_analysis.py` → `mega_run.py` → `final_run.py` → `fix_significance.py`,
  which is what `run_all_ext.sh` does. `fix_significance.py` must run last: it overwrites
  the inference blocks so that every Sharpe ratio in the paper uses the same
  compound-growth definition.
- `verify_numbers.py` is the gate. 190 checks, all tied to the JSON artifacts.
