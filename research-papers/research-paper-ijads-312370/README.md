# IJADS-312370 — revision

**"From Signal Fusion to Asset Allocation: A Decision-Theoretic Model for Portfolio
Construction Under Regime-Based Sentiment and Volatility"**
Pardeshi & Deshmukh · Int. J. of Applied Decision Sciences (Inderscience)
Editor: Prof. Madjid Tavana · Verdict: *acceptable with major revisions* · **Due 2026-09-20**

Status: **complete — ready for the author's read-through and upload.**

---

## What to upload

Everything the journal needs is in `2-revised-manuscript/`:

| File | What it is |
|---|---|
| `IJADS-312370-revised-with-response.pdf` | **the file to upload** — response letter (R1-R13) bound at the front of the 45 pp article, 58 pp |
| `ai67.pdf` | the revised manuscript alone, 45 pp |
| `response-to-reviewers.pdf` | point-by-point reply, 13 pp, all 10 mandatory and 5 optional comments |
| `ai67.tex`, `singlecol-new.cls`, `images/` | the source, if the journal asks for it |

Inderscience asks for the response letter at the *front of the revised article*. That is
already done: `make_submission_pdf.py` binds the two into
`IJADS-312370-revised-with-response.pdf`, with the letter numbered R1-R13 so every
"p. n" in it points at a manuscript page.

## Folder layout

```
1-original-submission/   what was submitted in August 2026, untouched
   2026_IJADS-312370.pdf     the submitted PDF
   ai92.tex                  the submitted LaTeX source
   code.ipynb                the submitted code — contains BOTH defects below
   reviewer-comments.pdf     the editor's letter and both reviews
   images/                   figures as submitted

2-revised-manuscript/    the deliverables (see above)

3-analysis/              the code and artifacts behind the revision
   pipeline_core.py          the corrected strategy
   final_run.py, ic_analysis.py, mega_run.py, fix_significance.py
   sentiment_validation.py   the external validation added for §4.3.3
   assemble.py               rebuilds ai67.tex from new_sec1..5.tex
   verify_numbers.py         the gate: 223 checks against the JSON artifacts
   run_all_ext.sh            runs the four analysis scripts in the required order

4-archive/
   v1-2008-2023/             the earlier revision, before the sample extension
   superseded/               exploratory scripts, stale caches, dead one-offs
   inderscience-template/    the journal's blank template
```

## Rebuilding

```bash
python 3-analysis/assemble.py          # rebuilds 2-revised-manuscript/ai67.tex
cd 2-revised-manuscript && pdflatex ai67.tex && pdflatex ai67.tex
python ../3-analysis/verify_numbers.py # must report 223 passed, 0 failed
```

A full re-run of the analysis is about 40 minutes (32 backtests, each refitting the
mixture model at 224 rebalance dates). `run_all_ext.sh` does it in the required order —
`fix_significance.py` **must** run last, because it rewrites the inference blocks so that
every Sharpe ratio in the paper uses the same compound-growth definition. Delete
`_final_cache_ext.pkl` to force the re-run.

## The two defects that drove the revision

Both were found while building the robustness analysis R2.3 asked for, and both are in
`1-original-submission/code.ipynb`.

1. **Unit mismatch.** The fused signal entered the mean–variance objective in raw
   indicator units — on-balance volume at 1e8–1e9 against a 1e-4 risk term — so the risk
   model was irrelevant and the optimiser held a single asset on almost every date. That
   is exactly the behaviour Reviewer 2 flagged in Figure 5. Fixed by Grinold
   information-ratio scaling, Eq. (4).
2. **Inert regime layer.** Absolute volatility targets of 20/15/10 per cent are never
   reached by a sleeve whose own volatility is 6.4 per cent, so the budget bound in
   **1 of 224** rebalances and the paper's central mechanism did nothing. Fixed by making
   the budget *relative* — a fraction of the sleeve's own risk, Eq. (7).

The lesson worth keeping: an absolute volatility target is not scale-free, so moving a
strategy between universes silently disables it.

## Where the revision stands empirically

Ten liquid multi-asset ETFs, 4,689 sessions, January 2008 – August 2026.

| | Strategy | Equal-weight | Inverse-vol | Vol-matched EW |
|---|---|---|---|---|
| Sharpe | **0.878** | 0.628 | 0.849 | 0.651 |
| Max drawdown | **−18.98%** | −36.08% | −24.09% | −18.69% |
| CAGR | 5.65% | 8.45% | 7.46% | 4.19% |

18 of 18 specifications beat equal weight on both measures; drawdown wins in 18 of 19
years, Sharpe in 9 of 19.

**The holdout (§4.3.1) is the strongest section.** The 665 sessions from January 2024
postdate every specification decision, and there the framework beats all three benchmarks
on both measures. Two of those comparisons reverse the in-sample ordering.

**Claims that survive.** At matched average exposure (92.1%), timing risk by market state
beats holding exposure constant — 0.878 vs 0.795 Sharpe, −18.98% vs −22.26% drawdown,
with no significant difference in mean return.

**Claims dropped, deliberately.** Significant alpha (excess return is *negative*,
p = 0.171). Signal fusion (removing it *improves* Sharpe to 0.908).

**Sentiment (§3.1.3 and §4.3.3).** The second composite is market-derived, not textual — no
transformer, no lexicon, no text of any kind. §4.3.3 validates it against four independently
published series: all four carry the expected sign, all four are significant in first
differences, and the sign holds in all sixteen series-by-block combinations. The
identified market states separate on it monotonically (F = 19.34, p = 1.8e-08) although
it takes no part in fitting them. §4.2.5 then reports that it does not predict returns. A
measure can be valid and still be unprofitable; the paper keeps the two findings apart.

## Structure and reference retention (author's decision, revision 2)

The submitted version's section headings are restored in full, so the revision reads
against the original section by section:

| | Submitted | Revised |
|---|---|---|
| Sections | 5 | 5 (same titles; `Results and Discussion` stays merged) |
| §3 | Phase 1 / Phase 2 / Phase 3 / Cross-Cutting, 13 subsubsections | identical headings, truthful content |
| §4 | Overview / Detailed Analysis (6 figures) / Implications | same, plus one new §4.3 `Robustness and Attribution` |
| References | 33 | 54 = all 33 retained + 21 added |
| Abstract | 245 words | 337 words |
| Pages | 22 | 45 |
| Body words | 6,825 | 17,653 |

Measured retention against `1-original-submission/ai92.tex`. The honest figure depends on
the threshold, so all three are given: **13% of sentences near-identical (>=95% similar),
42% substantially kept (>=80%), 58% recognisable (>=60%)**; at the word level
**65.0% of the submitted manuscript's words survive in order**. Per section, word level:
§1 75%, §2 75%, §3 55%, §4 53%, §5 65%. The §3 and §4 shortfall is not recoverable: the
submitted §3 describes components the implementation does not contain (LLM sentiment
scoring, named entity recognition, macroeconomic ingestion, Bayesian weight updating,
Brinson attribution, stop-loss, liquidity-adjusted sizing, an automated refinement loop),
and every number in the submitted §4 came from the two defective code paths below.

**Open risk — R1.1.** Reviewer 1 required no more than three references per journal.
24 of the 33 retained references are from *IEEE Access*. Retaining them is a deliberate
author decision; the response letter states this plainly rather than claiming compliance,
and offers to cut the list to three if the Editor prefers. Every other journal is within
the limit (max 3: IJADS, JPM, The Journal of Finance), no author appears more than twice,
no citation cluster exceeds two, there are no self-citations, and 10 of the 54 are from
2025–26.

## Before uploading

1. **Read §4.2.5, §4.3.3, §4.3.4 and §4.3.5.** They report that the signal fusion layer does not
   work and that the Sharpe gain is not statistically significant. This is deliberate and
   is, in our judgement, the strongest defensible position — but be comfortable with it
   before the reviewers see it.
2. Upload `IJADS-312370-revised-with-response.pdf`. The letter is already bound at the front.
3. **Highlighting changed text**, as the editor asked: §1 and §2 are close to the
   submitted text, so highlighting is practical there; §3 onward is substantially new and
   a covering sentence will serve better.

The AI declaration on p. 40 now names Claude (Anthropic) and ChatGPT (OpenAI) for
language editing only, which is what Reviewer 2 comment 6 asked to be completed.
