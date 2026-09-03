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
| `ai67.pdf` | the revised manuscript, 35 pp |
| `response-to-reviewers.pdf` | point-by-point reply, 7 pp, all 10 mandatory and 5 optional comments |
| `ai67.tex`, `singlecol-new.cls`, `images/` | the source, if the journal asks for it |

Inderscience asks for the response letter at the *front of the revised article*. It is
currently a separate PDF — either prepend it or upload it alongside.

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
   sentiment_validation.py   the external validation added for §4.8
   assemble.py               rebuilds ai67.tex from new_sec1..6.tex
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

**The holdout (§4.2) is the strongest section.** The 665 sessions from January 2024
postdate every specification decision, and there the framework beats all three benchmarks
on both measures. Two of those comparisons reverse the in-sample ordering.

**Claims that survive.** At matched average exposure (92.1%), timing risk by market state
beats holding exposure constant — 0.878 vs 0.795 Sharpe, −18.98% vs −22.26% drawdown,
with no significant difference in mean return.

**Claims dropped, deliberately.** Significant alpha (excess return is *negative*,
p = 0.171). Signal fusion (removing it *improves* Sharpe to 0.908).

**Sentiment (§3.3 and §4.8).** The second composite is market-derived, not textual — no
transformer, no lexicon, no text of any kind. §4.8 validates it against four independently
published series: all four carry the expected sign, all four are significant in first
differences, and the sign holds in all sixteen series-by-block combinations. The
identified market states separate on it monotonically (F = 19.34, p = 1.8e-08) although
it takes no part in fitting them. §4.9 then reports that it does not predict returns. A
measure can be valid and still be unprofitable; the paper keeps the two findings apart.

## Before uploading

1. **Read §4.6, §4.8, §5.2 and §6.** They report that the signal fusion layer does not
   work and that the Sharpe gain is not statistically significant. This is deliberate and
   is, in our judgement, the strongest defensible position — but be comfortable with it
   before the reviewers see it.
2. **Decide how to submit the response letter** (prepend, or upload alongside).
3. **Highlighting changed text**, as the editor asked: essentially everything from §1
   onward is new, so a covering sentence saying so will serve better than highlighting.

The AI declaration on p. 31 now names Claude (Anthropic) and ChatGPT (OpenAI) for
language editing only, which is what Reviewer 2 comment 6 asked to be completed.
