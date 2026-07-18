"""
LLM Sentiment Alpha — novelty × materiality scoring (Phase 3 — P3-4).

Two operating modes
-------------------
Mode A — Direct API (requires ANTHROPIC_API_KEY in .env or Streamlit secrets):
    Uses `claude-haiku-4-5-20251001` to score headlines automatically.
    Each call is cached in SQLite so the same headline is never billed twice.

Mode B — Clipboard / Manual (default when no API key is configured):
    1. `generate_manual_prompt(ticker, headlines)` returns a formatted prompt string
       the user can paste into claude.ai (free, no key needed).
    2. The user runs the prompt, copies Claude's JSON response.
    3. `parse_manual_response(response_text)` parses that JSON into scores.
    4. `score_ticker_from_manual_response(ticker, headlines, response_text)` wraps
       both steps and returns a `TickerSentimentResult`.

When the API is unavailable AND no manual response has been provided,
`score_ticker()` returns a `TickerSentimentResult` with:
    needs_manual_input = True
    manual_prompt      = <copy this into claude.ai>

The Streamlit UI should check `result.needs_manual_input`, show the prompt in a
copyable text area, accept the pasted response, then call
`score_ticker_from_manual_response()`.

Score dimensions
----------------
  novelty     [0.0–1.0]  : How new / surprising is this information?
  materiality [0.0–1.0]  : How much does this affect fundamental value?
  sentiment  [-1.0–+1.0] : Net directional impact for shareholders.

Combined Alpha Signal = sentiment × novelty × materiality

Usage — Mode A (API):
---------------------
    from services.llm_sentiment_alpha import LLMSentimentAlpha
    scorer = LLMSentimentAlpha()  # reads ANTHROPIC_API_KEY from env
    result = scorer.score_ticker("RELIANCE.NS", headlines)

Usage — Mode B (Clipboard):
----------------------------
    from services.llm_sentiment_alpha import LLMSentimentAlpha
    scorer = LLMSentimentAlpha()

    result = scorer.score_ticker("RELIANCE.NS", headlines)
    if result.needs_manual_input:
        # Show result.manual_prompt to user → they paste into claude.ai
        # Then accept their pasted response:
        result = scorer.score_ticker_from_manual_response(
            "RELIANCE.NS", headlines, pasted_claude_response
        )
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import textwrap
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_MODEL            = "claude-haiku-4-5-20251001"
_MAX_HEADLINES    = 10
_MAX_TOKENS_OUT   = 512
_CACHE_DB_PATH    = Path("data/ledger/llm_sentiment_cache.db")
_CACHE_LOCK       = threading.Lock()
_MIN_MATERIALITY  = 0.15


# ── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass
class HeadlineScore:
    """Scored dimensions for one news headline."""
    headline:     str   = ""
    ticker:       str   = ""
    novelty:      float = 0.5
    materiality:  float = 0.5
    sentiment:    float = 0.0
    alpha_signal: float = 0.0
    scored_at:    str   = ""
    from_cache:   bool  = False
    from_manual:  bool  = False
    used_fallback: bool = False
    error:        str   = ""


@dataclass
class TickerSentimentResult:
    """Aggregated LLM sentiment alpha for one ticker."""
    ticker:             str   = ""
    headlines_scored:   int   = 0
    headlines_used:     int   = 0

    combined_signal:    float = 0.0
    mean_novelty:       float = 0.0
    mean_materiality:   float = 0.0
    mean_sentiment:     float = 0.0
    top_headline:       str   = ""
    top_alpha:          float = 0.0

    available:          bool  = False
    used_fallback:      bool  = False

    # Clipboard-mode fields
    needs_manual_input: bool  = False
    manual_prompt:      str   = ""

    individual_scores: list[HeadlineScore] = field(default_factory=list)


# ── SQLite cache ──────────────────────────────────────────────────────────────

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS headline_scores (
    headline_hash TEXT PRIMARY KEY,
    ticker        TEXT NOT NULL,
    headline      TEXT NOT NULL,
    novelty       REAL NOT NULL,
    materiality   REAL NOT NULL,
    sentiment     REAL NOT NULL,
    alpha_signal  REAL NOT NULL,
    scored_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hs_ticker ON headline_scores(ticker);
"""


def _connect_cache() -> sqlite3.Connection:
    _CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_CACHE_SCHEMA)
    conn.commit()
    return conn


def _headline_hash(ticker: str, headline: str) -> str:
    raw = f"{ticker.upper()}||{headline.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _lookup_cache(ticker: str, headline: str) -> Optional[HeadlineScore]:
    key = _headline_hash(ticker, headline)
    with _CACHE_LOCK:
        try:
            with _connect_cache() as conn:
                row = conn.execute(
                    "SELECT * FROM headline_scores WHERE headline_hash = ?", (key,)
                ).fetchone()
            if row:
                return HeadlineScore(
                    headline    = str(row["headline"]),
                    ticker      = str(row["ticker"]),
                    novelty     = float(row["novelty"]),
                    materiality = float(row["materiality"]),
                    sentiment   = float(row["sentiment"]),
                    alpha_signal= float(row["alpha_signal"]),
                    scored_at   = str(row["scored_at"]),
                    from_cache  = True,
                )
        except Exception as exc:
            logger.debug("llm_sentiment_alpha: cache lookup failed: %s", exc)
    return None


def _write_cache(score: HeadlineScore) -> None:
    key = _headline_hash(score.ticker, score.headline)
    with _CACHE_LOCK:
        try:
            with _connect_cache() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO headline_scores
                       (headline_hash, ticker, headline, novelty, materiality, sentiment,
                        alpha_signal, scored_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (key, score.ticker, score.headline[:500],
                     score.novelty, score.materiality, score.sentiment,
                     score.alpha_signal, score.scored_at),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("llm_sentiment_alpha: cache write failed: %s", exc)


# ── Keyword fallback ──────────────────────────────────────────────────────────

_BULLISH_KW = frozenset([
    "profit", "beat", "upgrade", "acquisition", "buyback", "dividend",
    "record revenue", "strong", "growth", "expansion", "order win",
    "ipo", "qip", "capex", "partnership", "deal", "approval", "launch",
])
_BEARISH_KW = frozenset([
    "loss", "miss", "downgrade", "fraud", "raid", "sebi", "penalty",
    "npa", "default", "recall", "lawsuit", "ban", "halt", "plant shutdown",
    "pledge", "resign", "debt", "write-off",
])


def _fallback_score(ticker: str, headline: str) -> HeadlineScore:
    hl = headline.lower()
    bull = sum(1 for kw in _BULLISH_KW if kw in hl)
    bear = sum(1 for kw in _BEARISH_KW if kw in hl)
    if bull > bear:
        sentiment = min(0.4 * bull, 0.8)
    elif bear > bull:
        sentiment = max(-0.4 * bear, -0.8)
    else:
        sentiment = 0.0
    novelty     = 0.50
    materiality = 0.30 if (bull + bear) > 0 else 0.10
    alpha       = float(sentiment * novelty * materiality)
    return HeadlineScore(
        headline    = headline,
        ticker      = ticker,
        novelty     = novelty,
        materiality = materiality,
        sentiment   = sentiment,
        alpha_signal= alpha,
        scored_at   = datetime.utcnow().isoformat(),
        used_fallback = True,
    )


# ── Clipboard prompt generation ───────────────────────────────────────────────

def generate_manual_prompt(ticker: str, headlines: list[str]) -> str:
    """
    Generate a top-class quantitative analyst prompt for pasting into claude.ai.

    The prompt gives Claude full scoring philosophy, calibration anchors with
    concrete examples, NSE sector context, and explicit JSON output instructions.
    `parse_manual_response` handles minor formatting deviations gracefully.
    """
    company = ticker.replace(".NS", "").replace(".BO", "").upper()
    to_score = [h.strip() for h in headlines if h.strip()][:_MAX_HEADLINES]

    if not to_score:
        return ""

    numbered = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(to_score))
    n = len(to_score)

    prompt = textwrap.dedent(f"""
        ROLE
        ────
        You are a quantitative equity analyst at a top-tier Indian asset management firm
        (think DSP, Motilal, Mirae). Your job is to score raw news headlines for their
        information value in a systematic trading signal. The scores feed directly into a
        quant alpha model — they are NOT used for investment advice.

        COMPANY IN FOCUS
        ────────────────
        {company} (NSE-listed Indian equity)

        SCORING FRAMEWORK
        ─────────────────
        Score every headline on exactly three continuous dimensions:

        1. NOVELTY  [0.00 – 1.00]
           Measures how informationally new this headline is relative to the market's
           existing knowledge. High novelty = genuine price-relevant surprise.
           Anchors:
             0.00 — Exact repetition of news already priced in 3+ weeks ago.
             0.20 — Quarterly result that met consensus; nothing new.
             0.40 — Guidance update that was anticipated but now confirmed.
             0.60 — An unexpected but modest management change or order win.
             0.80 — A large acquisition announcement not previously rumoured.
             1.00 — A sudden regulatory ban, fraud disclosure, or CEO arrest.

        2. MATERIALITY  [0.00 – 1.00]
           Measures how much this event affects the company's intrinsic value or near-
           term earnings. Purely technical/noise events score near zero.
           Anchors:
             0.00 — Boilerplate PR or unrelated industry chatter.
             0.20 — Minor contract; <1% of annual revenue.
             0.40 — Meaningful contract or partnership; 2-5% revenue impact.
             0.60 — Structural change: new product line, capacity expansion.
             0.80 — Transformative deal (large M&A, major regulatory approval).
             1.00 — Existential event (insolvency risk, regulatory shutdown, fraud).

        3. SENTIMENT  [-1.00 – +1.00]
           Net directional impact on equity holders over the next 5–10 trading days.
           Positive = bullish for the stock. Negative = bearish.
           Anchors:
             +1.00 — Massive earnings beat + raised guidance; buyback at premium.
             +0.60 — Order win, margin expansion, credit rating upgrade.
             +0.30 — Mild positive surprise; in-line result with positive tone.
              0.00 — Neutral event (management reshuffle with no strategic change).
             -0.30 — Mild miss; cost pressure flagged without guidance cut.
             -0.60 — Profit warning, NPA recognition, SEBI notice.
             -1.00 — Fraud allegation, promoter pledge invocation, ED raid.

        COMBINED ALPHA SIGNAL
        ─────────────────────
        The model computes:  alpha = sentiment × novelty × materiality
        This means:
          • High materiality + high novelty + strong sentiment → large alpha signal
          • Even very bullish sentiment scores near-zero if the news is stale (novelty≈0)
            or irrelevant (materiality≈0). Be precise — calibration matters.

        HEADLINES TO SCORE  ({n} total)
        ────────────────────────────────
{numbered}

        OUTPUT RULES
        ────────────
        • Return ONLY a valid JSON array — no explanation, no markdown, no code fences.
        • One object per headline in the SAME ORDER as listed above.
        • Use the key "i" (1-based) to identify each headline.
        • All values must be numbers — no strings, no nulls.
        • If a headline is ambiguous or you are uncertain, use the midpoint of the range
          (e.g., novelty=0.5) rather than skewing arbitrarily.

        REQUIRED FORMAT (copy exactly, no deviations):
        [{{"i":1,"novelty":0.75,"materiality":0.60,"sentiment":0.55}},{{"i":2,"novelty":0.20,"materiality":0.15,"sentiment":-0.30}}]

        Return {n} objects total. Nothing else.
    """).strip()

    return prompt


def generate_signal_synthesis_prompt(
    ticker: str,
    current_price: float,
    forecast_return_pct: float,
    directional_prob: float,
    model_accuracy: float,
    hurst: Optional[float],
    rsi: float,
    macd_hist: float,
    volatility_20d: float,
    fii_net_5d_cr: float,
    dii_net_5d_cr: float,
    vix: float,
    llm_sentiment_signal: float,
    llm_mean_sentiment: float,
    llm_mean_materiality: float,
    llm_top_headline: str,
    multi_sentiment_score: float,
    options_pcr_signal: float,
    options_iv_skew_signal: float,
    options_combined: float,
    shap_top_features: list[str],
    forecast_days: int,
    regime: str,
) -> str:
    """
    Generate a comprehensive Claude signal synthesis prompt.

    This prompt synthesises EVERY signal layer of the Phase 3 pipeline into a
    single unified trade brief. Paste this into claude.ai after all model outputs
    are available.

    Returns
    -------
    str — the full prompt to paste into claude.ai
    """
    hurst_str = f"{hurst:.3f}" if hurst is not None else "N/A"
    market_char = (
        "Trending (momentum)" if (hurst or 0) > 0.55
        else "Mean-Reverting" if (hurst or 0) < 0.45
        else "Random Walk"
    ) if hurst is not None else "Unknown"

    shap_str = ", ".join(shap_top_features[:5]) if shap_top_features else "N/A"
    regime_clean = regime.replace("_", " ").title()

    prompt = textwrap.dedent(f"""
        ROLE
        ────
        You are the head of systematic equity research at a Nifty 500 quant fund.
        A multi-layer alpha model just completed its full pipeline on {ticker.replace('.NS','').replace('.BO','').upper()}.
        Your job: synthesise ALL signal layers into ONE actionable trade brief.
        Be quantitatively rigorous. No disclaimers. No hedging. Be decisive.

        ════════════════════════════════════════════════════════════════
        SIGNAL STACK — {ticker.replace('.NS','').replace('.BO','').upper()}  |  ₹{current_price:,.2f}
        ════════════════════════════════════════════════════════════════

        ── LAYER 1: ML MODEL OUTPUT ──────────────────────────────────
        Forecast horizon        : {forecast_days} trading days
        Median forecast return  : {forecast_return_pct:+.2f}%
        Directional probability : {directional_prob:.1f}% (bullish)
        Walk-forward accuracy   : {model_accuracy:.1f}%
        Market regime           : {regime_clean}
        Hurst exponent          : {hurst_str} → {market_char}
        Top SHAP features       : {shap_str}

        ── LAYER 2: TECHNICAL INDICATORS ─────────────────────────────
        RSI (14)                : {rsi:.1f}  {"[Overbought]" if rsi > 70 else "[Oversold]" if rsi < 30 else "[Neutral]"}
        MACD Histogram          : {macd_hist:+.4f}  {"[Bullish divergence]" if macd_hist > 0 else "[Bearish divergence]"}
        20-day realised vol     : {volatility_20d * 100:.2f}% annualised

        ── LAYER 3: INSTITUTIONAL FLOWS (FII / DII) ──────────────────
        FII net (5-day)         : ₹{fii_net_5d_cr:+.1f} Cr  {"[Buying]" if fii_net_5d_cr > 0 else "[Selling]"}
        DII net (5-day)         : ₹{dii_net_5d_cr:+.1f} Cr  {"[Buying]" if dii_net_5d_cr > 0 else "[Selling]"}
        India VIX               : {vix:.2f}  {"[Elevated fear]" if vix > 18 else "[Complacent]" if vix < 12 else "[Normal]"}

        ── LAYER 4: OPTIONS MARKET ALPHA ─────────────────────────────
        PCR Δ signal            : {options_pcr_signal:+.3f}  {"[Bearish unwinding]" if options_pcr_signal > 0.3 else "[Bullish build]" if options_pcr_signal < -0.3 else "[Neutral]"}
        IV skew Δ signal        : {options_iv_skew_signal:+.3f}  {"[Tail-risk demand rising]" if options_iv_skew_signal < -0.3 else "[Skew normalising]" if options_iv_skew_signal > 0.3 else "[Stable]"}
        Options combined alpha  : {options_combined:+.3f}

        ── LAYER 5: LLM SENTIMENT ALPHA ──────────────────────────────
        Combined alpha signal   : {llm_sentiment_signal:+.3f}
        Mean sentiment score    : {llm_mean_sentiment:+.3f}
        Mean materiality        : {llm_mean_materiality:.2f}
        Top headline            : "{llm_top_headline[:120]}"
        Multi-source sentiment  : {multi_sentiment_score:+.3f}

        ════════════════════════════════════════════════════════════════
        SYNTHESIS TASK — rules
        ════════════════════════════════════════════════════════════════
        • Every claim must cite a specific layer name and numeric value from above.
        • If layers disagree, state WHICH layer you trust more and WHY (give one sentence).
        • Rank the 5 layers by reliability for THIS specific situation (e.g., Layer 1 > Layer 3
          because walk-forward accuracy is 68% vs options with zero open interest context).
        • Be decisive. No hedging. I am a registered professional trader.
        • Max 400 words total.

        Produce your response in EXACTLY this markdown format:

        ### Signal Alignment Score: [X/5 signals aligned — bullish / bearish / mixed]
        Layer ranking for this trade (most reliable → least reliable): [Layer N > Layer N > ...]

        ### Composite View: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
        Conviction: [High / Medium / Low] — [one sentence explaining why]

        ### Bull Case (cite layer + number)
        - [Layer N: specific signal and value]
        - [Layer N: specific signal and value]

        ### Bear Case (cite layer + number)
        - [Layer N: specific signal and value — most important risk]
        - [Layer N: any conflicting or weakening signal]

        ### Key Conflict
        One sentence on the most critical signal divergence and which layer wins
        (e.g., "Layer 1 ML says +4.2% but Layer 3 FII is selling ₹1,200 Cr/day —
        trust ML because accuracy is 67% and FII sell-off may be rotation not exit").

        ### Execution Parameters
        - Entry zone    : ₹[specific range based on current price ± 20D vol band]
        - Stop loss     : ₹[price]  (−[%] from entry — sized to 20D vol)
        - Target 1 (1.5R): ₹[price]  (+[%])
        - Target 2 (2.5R): ₹[price]  (+[%])
        - Suggested size: [Aggressive / Normal / Half / Skip] — [one phrase: cite alignment score and VIX]

        ### Risk Flags
        - [Flag 1: extreme reading in any layer — RSI >75, VIX >20, LLM materiality <0.2, etc.]
        - [Flag 2: regime or Hurst-based warning — e.g., mean-reverting regime contradicts momentum trade]
    """).strip()

    return prompt


def parse_manual_response(
    response_text: str,
    ticker: str,
    headlines: list[str],
) -> list[HeadlineScore]:
    """
    Parse a Claude response (pasted by the user) into a list of HeadlineScore objects.

    Handles minor formatting issues:
      • JSON wrapped in markdown code blocks (```json ... ```)
      • Leading/trailing whitespace
      • Extra text before/after the JSON array

    Parameters
    ----------
    response_text : str
        The raw text the user copied from Claude's response.
    ticker : str
        NSE ticker (used to set HeadlineScore.ticker).
    headlines : list[str]
        The original headlines list (used for label matching by index).

    Returns
    -------
    list[HeadlineScore] — one entry per headline; fallback score on parse error.
    """
    headlines_clean = [h.strip() for h in headlines if h.strip()][:_MAX_HEADLINES]
    scored: list[HeadlineScore] = []

    # ── Extract JSON array from the pasted text ───────────────────────────────
    text = response_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Normalize typographic quotes that clipboards/editors substitute for ASCII ones
    text = (text.replace("“", '"').replace("”", '"').replace("„", '"')
                .replace("‘", "'").replace("’", "'"))
    text = text.strip()

    # Find the first JSON array (may have surrounding text)
    json_match = re.search(r"\[[\s\S]*?\]", text)
    if not json_match:
        logger.warning("parse_manual_response: no JSON array found in response")
        return [_fallback_score(ticker, h) for h in headlines_clean]

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("parse_manual_response: JSON decode error: %s", exc)
        return [_fallback_score(ticker, h) for h in headlines_clean]

    if not isinstance(parsed, list):
        logger.warning("parse_manual_response: expected a JSON array, got %s", type(parsed))
        return [_fallback_score(ticker, h) for h in headlines_clean]

    # Build index → score dict (Claude returns 1-based "i" keys)
    index_map: dict[int, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        # Support both "i" key (preferred) and positional fallback
        idx = item.get("i") or item.get("index") or item.get("n")
        if idx is not None:
            try:
                index_map[int(idx) - 1] = item
            except (TypeError, ValueError):
                continue

    # If no "i" keys, use positional order
    if not index_map:
        for pos, item in enumerate(parsed):
            if isinstance(item, dict):
                index_map[pos] = item

    now_str = datetime.utcnow().isoformat()
    for i, headline in enumerate(headlines_clean):
        item = index_map.get(i)
        if item is None:
            scored.append(_fallback_score(ticker, headline))
            continue

        try:
            novelty     = float(np.clip(item.get("novelty",     0.5), 0.0, 1.0))
            materiality = float(np.clip(item.get("materiality", 0.5), 0.0, 1.0))
            sentiment   = float(np.clip(item.get("sentiment",   0.0), -1.0, 1.0))
            alpha       = float(sentiment * novelty * materiality)

            sc = HeadlineScore(
                headline    = headline,
                ticker      = ticker,
                novelty     = novelty,
                materiality = materiality,
                sentiment   = sentiment,
                alpha_signal= alpha,
                scored_at   = now_str,
                from_manual = True,
            )
            scored.append(sc)
            _write_cache(sc)

        except Exception as exc:
            logger.warning("parse_manual_response: item %d parse error: %s", i, exc)
            scored.append(_fallback_score(ticker, headline))

    logger.info(
        "parse_manual_response: %d/%d headlines parsed from manual response for %s",
        sum(1 for s in scored if s.from_manual), len(scored), ticker,
    )
    return scored


# ── API scoring ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a quantitative financial analyst specialising in Indian equities. "
    "Score each headline on novelty, materiality, and sentiment. "
    "Respond ONLY with a JSON array: "
    '[{"i":1,"novelty":<f>,"materiality":<f>,"sentiment":<f>}, ...]. '
    "No explanation. No markdown."
)


def _call_claude_api(ticker: str, headlines: list[str], api_key: str) -> list[HeadlineScore]:
    """Call Claude API to score a list of headlines. Returns fallbacks on error."""
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; using fallback")
        return [_fallback_score(ticker, h) for h in headlines]

    company = ticker.replace(".NS", "").replace(".BO", "")
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    user_msg  = f"Company: {company}\nHeadlines:\n{numbered}"

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model      = _MODEL,
            max_tokens = _MAX_TOKENS_OUT,
            system     = [
                {"type": "text", "text": _SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        return parse_manual_response(raw, ticker, headlines)

    except Exception as exc:
        logger.warning("_call_claude_api: %s — using fallback", exc)
        return [_fallback_score(ticker, h) for h in headlines]


# ── Aggregation helper ────────────────────────────────────────────────────────

def _aggregate_scores(
    ticker: str,
    headlines: list[str],
    scores: list[HeadlineScore],
) -> TickerSentimentResult:
    """Aggregate a list of HeadlineScore objects into a TickerSentimentResult."""
    if not scores:
        return TickerSentimentResult(ticker=ticker, available=True)

    material = [s for s in scores if s.materiality >= _MIN_MATERIALITY]

    if not material:
        return TickerSentimentResult(
            ticker           = ticker,
            headlines_scored = len(scores),
            headlines_used   = 0,
            available        = True,
            individual_scores= scores,
        )

    weights      = np.array([s.novelty * s.materiality for s in material])
    alpha_arr    = np.array([s.alpha_signal   for s in material])
    novelty_arr  = np.array([s.novelty        for s in material])
    mat_arr      = np.array([s.materiality    for s in material])
    sent_arr     = np.array([s.sentiment      for s in material])

    w_sum    = float(weights.sum())
    combined = float(np.dot(weights, alpha_arr) / w_sum) if w_sum > 1e-10 else 0.0
    combined = float(np.clip(combined, -1.0, 1.0))

    top_idx  = int(np.argmax(np.abs(alpha_arr)))

    return TickerSentimentResult(
        ticker            = ticker,
        headlines_scored  = len(scores),
        headlines_used    = len(material),
        combined_signal   = combined,
        mean_novelty      = float(np.mean(novelty_arr)),
        mean_materiality  = float(np.mean(mat_arr)),
        mean_sentiment    = float(np.mean(sent_arr)),
        top_headline      = material[top_idx].headline,
        top_alpha         = float(alpha_arr[top_idx]),
        available         = True,
        used_fallback     = any(s.error or getattr(s, "used_fallback", False) for s in material),
        individual_scores = scores,
    )


# ── Main class ────────────────────────────────────────────────────────────────

class LLMSentimentAlpha:
    """
    Scores news headlines and returns a single ticker-level alpha signal.

    Operating modes (automatically selected):
      • API mode:      ANTHROPIC_API_KEY configured → calls Claude directly.
      • Clipboard mode: no key → returns needs_manual_input=True with
                        manual_prompt that the user pastes into claude.ai.

    The class is stateless between calls. Caching is at the DB layer.

    Parameters
    ----------
    api_key : str | None
        Explicit Anthropic API key. If None, resolved from
        config.settings.ANTHROPIC_API_KEY or ANTHROPIC_API_KEY env var.
    max_headlines : int
        Max headlines to score per ticker (keeps API cost bounded).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_headlines: int = _MAX_HEADLINES,
    ) -> None:
        self._max    = max_headlines
        self._key    = self._resolve_key(api_key)
        self.has_api = bool(self._key)

        if not self.has_api:
            logger.info(
                "LLMSentimentAlpha: no ANTHROPIC_API_KEY — clipboard mode active. "
                "Call score_ticker(); if result.needs_manual_input is True, show "
                "result.manual_prompt to the user and call "
                "score_ticker_from_manual_response() with their pasted response."
            )

    @staticmethod
    def _resolve_key(explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        try:
            from config.settings import ANTHROPIC_API_KEY
            if ANTHROPIC_API_KEY:
                return ANTHROPIC_API_KEY
        except ImportError:
            pass
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Core: score a list of headlines ──────────────────────────────────────

    def score_ticker(
        self,
        ticker: str,
        headlines: list[str],
    ) -> TickerSentimentResult:
        """
        Score headlines and return a TickerSentimentResult.

        • If API key is configured: calls Claude and returns scores immediately.
        • If no API key: returns result with needs_manual_input=True and
          manual_prompt set to the text the user should paste into claude.ai.
          Call score_ticker_from_manual_response() with their pasted response.

        Cache is checked first in both modes, so headlines already scored
        (from a prior session) are returned instantly without re-scoring.
        """
        if not headlines:
            return TickerSentimentResult(ticker=ticker, available=True)

        to_score = [h.strip() for h in headlines if h.strip()][: self._max]

        # ── Check cache first ─────────────────────────────────────────────────
        scores: list[HeadlineScore] = []
        uncached: list[str] = []
        uncached_idx: list[int] = []

        for i, h in enumerate(to_score):
            cached = _lookup_cache(ticker, h)
            if cached:
                scores.append(cached)
            else:
                scores.append(None)   # placeholder
                uncached.append(h)
                uncached_idx.append(i)

        # ── All headlines cached → aggregate and return ───────────────────────
        if not uncached:
            return _aggregate_scores(ticker, to_score,
                                     [s for s in scores if s is not None])

        # ── API mode: call Claude for uncached headlines ──────────────────────
        if self.has_api:
            new_scores = _call_claude_api(ticker, uncached, self._key)
            for idx, sc in zip(uncached_idx, new_scores):
                scores[idx] = sc
            final = [s for s in scores if s is not None]
            return _aggregate_scores(ticker, to_score, final)

        # ── Clipboard mode: return prompt for the user ────────────────────────
        prompt = generate_manual_prompt(ticker, uncached)
        return TickerSentimentResult(
            ticker              = ticker,
            headlines_scored    = len(to_score),
            headlines_used      = 0,
            needs_manual_input  = True,
            manual_prompt       = prompt,
            available           = False,
        )

    def score_ticker_from_manual_response(
        self,
        ticker: str,
        headlines: list[str],
        manual_response_text: str,
    ) -> TickerSentimentResult:
        """
        Parse the response the user pasted from claude.ai and aggregate scores.

        Call this after the user pastes Claude's response into the UI.

        Parameters
        ----------
        ticker : str
            NSE ticker.
        headlines : list[str]
            The same headlines that were shown in `score_ticker()`.
        manual_response_text : str
            The text the user copied from Claude's response.

        Returns
        -------
        TickerSentimentResult with available=True (or fallback scores if parse fails).
        """
        to_score = [h.strip() for h in headlines if h.strip()][: self._max]
        scores   = parse_manual_response(manual_response_text, ticker, to_score)
        return _aggregate_scores(ticker, to_score, scores)

    def score_batch(
        self,
        ticker_headlines: dict[str, list[str]],
    ) -> dict[str, TickerSentimentResult]:
        """
        Score multiple tickers. Returns dict[ticker → result].
        In clipboard mode, each result will have needs_manual_input=True.
        """
        results: dict[str, TickerSentimentResult] = {}
        for ticker, headlines in ticker_headlines.items():
            try:
                results[ticker] = self.score_ticker(ticker, headlines)
            except Exception as exc:
                logger.warning("score_batch: %s failed: %s", ticker, exc)
                results[ticker] = TickerSentimentResult(ticker=ticker, available=False)
        return results


# ── Feature dict helper ───────────────────────────────────────────────────────

def llm_sentiment_to_feature_dict(result: TickerSentimentResult) -> dict[str, float]:
    """Flatten a TickerSentimentResult into a numeric feature dict."""
    if not result.available or result.needs_manual_input:
        return {
            "LLM_Combined":    0.0,
            "LLM_Novelty":     0.5,
            "LLM_Materiality": 0.5,
            "LLM_Sentiment":   0.0,
            "LLM_Top_Alpha":   0.0,
            "LLM_N_Headlines": 0.0,
            "LLM_Available":   0.0,
        }
    return {
        "LLM_Combined":    float(result.combined_signal),
        "LLM_Novelty":     float(result.mean_novelty),
        "LLM_Materiality": float(result.mean_materiality),
        "LLM_Sentiment":   float(result.mean_sentiment),
        "LLM_Top_Alpha":   float(result.top_alpha),
        "LLM_N_Headlines": float(result.headlines_used),
        "LLM_Available":   1.0 if result.headlines_used > 0 else 0.0,
    }
