"""
Analysis router — AI Expert Analysis + Signal Synthesis (parity with Streamlit).

  POST /api/v1/stocks/{ticker}/ai-analysis     → DeepSeek/Gemini prose analysis
  POST /api/v1/stocks/{ticker}/signal-synthesis → 5-layer Claude synthesis prompt
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["analysis"])


# ── Shared request/response types ─────────────────────────────────────────────

class TechnicalSnapshot(BaseModel):
    """Technical indicators sent by AiAnalysisPanel — rsi maps from rsi_14 on the frontend."""
    rsi: float = 50.0            # frontend sends technicals.rsi_14 mapped to this field
    macd_histogram: float = 0.0
    volatility_5d: float = 0.01  # frontend sends volatility_20d as proxy
    volatility_20d: float = 0.01
    price_vs_ma20: float = 0.0


class AiAnalysisRequest(BaseModel):
    current_price: float
    forecast_target: float
    forecast_days: int = 10
    model_accuracy: float = 50.0
    directional_prob: float = 50.0
    technicals: TechnicalSnapshot = Field(default_factory=TechnicalSnapshot)
    fii_net_cr: float = 0.0
    dii_net_cr: float = 0.0
    vix: float | None = None
    forward_pe: float | None = None
    price_book: float | None = None
    peg: float | None = None
    patterns: list[str] = Field(default_factory=list)
    sentiment_label: str = "neutral"
    sentiment_pos: int = 0
    sentiment_neg: int = 0


class AiAnalysisResponse(BaseModel):
    analysis: str
    source: str          # "deepseek" | "gemini" | "claude_prompt" | "template"
    claude_prompt: str   # always populated so frontend can offer copy-paste


class SignalSynthesisRequest(BaseModel):
    current_price: float
    forecast_return_pct: float
    forecast_days: int = 10
    directional_prob: float = 50.0
    model_accuracy: float = 50.0
    hurst: float | None = None
    regime: str = "normal"
    rsi: float = 50.0
    macd_hist: float = 0.0
    volatility_20d: float = 0.01
    fii_net_5d_cr: float = 0.0
    dii_net_5d_cr: float = 0.0
    vix: float = 15.0
    llm_sentiment_signal: float = 0.0
    llm_mean_sentiment: float = 0.0
    llm_mean_materiality: float = 0.0
    llm_top_headline: str = "N/A"
    multi_sentiment_score: float = 0.0
    options_pcr_signal: float = 0.0
    options_iv_skew_signal: float = 0.0
    options_combined: float = 0.0
    shap_top_features: list[str] = Field(default_factory=list)


class SignalSynthesisResponse(BaseModel):
    prompt: str
    pasted_analysis: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_analysis_context(ticker: str, req: AiAnalysisRequest) -> str:
    forecast_return = ((req.forecast_target - req.current_price) / req.current_price) * 100
    pattern_str = ", ".join(req.patterns[:3]) if req.patterns else "None detected"
    fii_dii = (
        f"FII net: ₹{req.fii_net_cr:+,.2f} Cr | DII net: ₹{req.dii_net_cr:+,.2f} Cr"
        if (req.fii_net_cr or req.dii_net_cr)
        else "Unavailable — model used zeros"
    )
    vix_str = f"{req.vix:.2f} ({'Elevated fear' if req.vix and req.vix > 18 else 'Complacent' if req.vix and req.vix < 12 else 'Normal'})" if req.vix is not None else "N/A"
    rsi_signal = "Overbought" if req.technicals.rsi > 70 else "Oversold" if req.technicals.rsi < 30 else "Neutral"
    macd_signal = "Bullish divergence" if req.technicals.macd_histogram > 0 else "Bearish divergence"
    company = ticker.replace(".NS", "").replace(".BO", "").upper()

    return f"""STOCK: {company} (NSE/BSE Indian equity)
Current Price: ₹{req.current_price:,.2f}

━━ ML MODEL OUTPUT ({req.forecast_days}-day horizon) ━━
  Target price      : ₹{req.forecast_target:,.2f}  ({forecast_return:+.2f}%)
  Direction         : {'UP ↑' if forecast_return > 0 else 'DOWN ↓'}
  Walk-forward acc  : {req.model_accuracy:.1f}%  {'(research-grade ≥65%)' if req.model_accuracy >= 65 else '(production-grade)' if req.model_accuracy >= 55 else '(uncertain <55%)'}
  Bullish prob P(up): {req.directional_prob:.1f}%

━━ TECHNICAL INDICATORS ━━
  RSI (14)          : {req.technicals.rsi:.1f}  [{rsi_signal}]
  MACD Histogram    : {req.technicals.macd_histogram:+.4f}  [{macd_signal}]
  20D Volatility    : {req.technicals.volatility_20d * 100:.2f}%  annualised
  Price vs MA20     : {req.technicals.price_vs_ma20 * 100:+.2f}%
  Patterns          : {pattern_str}

━━ FUNDAMENTALS ━━
  Forward P/E       : {req.forward_pe if req.forward_pe is not None else 'N/A'}
  Price/Book        : {req.price_book if req.price_book is not None else 'N/A'}
  PEG Ratio         : {req.peg if req.peg is not None else 'N/A'}

━━ MARKET CONTEXT ━━
  Institutional flows : {fii_dii}
  News sentiment      : {req.sentiment_label.capitalize()}  ({req.sentiment_pos} positive, {req.sentiment_neg} negative headlines)
  India VIX           : {vix_str}""".strip()


def _build_system_prompt() -> str:
    return """You are the head of equity research at a top-tier Indian quant hedge fund.
You receive structured ML model output + technicals + fundamentals for an NSE/BSE stock.
Your job: produce a precise, institutionally rigorous trading verdict.

RULES:
1. No disclaimers. No hedging. No "consider consulting a financial advisor." Be decisive.
2. Every claim must cite a specific number from the data provided.
3. If ML signal contradicts technicals/fundamentals, resolve the conflict explicitly — state which you trust more and why.
4. Use clean markdown. Bullets only — no prose paragraphs.
5. Entry/stop/target levels must be specific rupee prices, not vague ranges.
6. Acknowledge ECE calibration warnings if mentioned — unreliable probabilities change the trade sizing.

You are writing for a registered professional trader who understands all standard financial concepts."""


def _build_user_prompt(context: str) -> str:
    return f"""{context}

Provide your analysis in EXACTLY this format. No extra sections. Max 350 words total.

### 🎯 Verdict: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
**Conviction:** [High / Medium / Low]

### 🧠 Expert Rationale (3 bullets, each cites a specific number)
- **ML Signal:** [Accuracy %, P(up) %, direction — is the model reliable here?]
- **Technical bias:** [RSI, MACD, patterns — confirming or contradicting the ML call?]
- **Institutional / macro:** [FII/DII flow direction, VIX level — tailwind or headwind?]

### ⚠️ Critical Risks
- [Risk 1 with specific data point — model accuracy limit, RSI extreme, VIX spike, etc.]
- [Risk 2 — fundamental concern, sector headwind, or conflicting signal]

### 💰 Execution Plan
- **Entry zone:** ₹[price range — near current if bullish, wait-for-bounce level if HOLD/SELL]
- **Stop loss:** ₹[specific level — use 20D volatility or nearest support as basis]
- **Target 1 (1.5R):** ₹[entry + 1.5 × risk]
- **Target 2 (2.5R):** ₹[entry + 2.5 × risk]
- **Position size:** [Full / Half / Quarter / Skip] — [one phrase justifying sizing based on confidence and model accuracy]"""


def _template_analysis(ticker: str, req: AiAnalysisRequest) -> str:
    forecast_return = ((req.forecast_target - req.current_price) / req.current_price) * 100
    company = ticker.replace(".NS", "").replace(".BO", "").upper()
    if req.model_accuracy < 50:
        conf, verdict = "Low Confidence", "HOLD"
    elif forecast_return > 5 and req.model_accuracy > 60:
        conf, verdict = "Good Confidence", "BUY"
    elif forecast_return < -5 and req.model_accuracy > 60:
        conf, verdict = "Good Confidence", "SELL"
    else:
        conf, verdict = "Moderate Confidence", "HOLD"
    rsi_signal = "Overbought" if req.technicals.rsi > 70 else "Oversold" if req.technicals.rsi < 30 else "Neutral"
    return f"""### 🎯 Verdict: {verdict}
**{conf}** | Model Accuracy: {req.model_accuracy:.1f}%

### 📊 Outlook
- **Short-term:** Predicted {forecast_return:+.1f}% move
- **RSI Signal:** {rsi_signal} ({req.technicals.rsi:.1f})
- **FII:** ₹{req.fii_net_cr:+.1f}Cr | **DII:** ₹{req.dii_net_cr:+.1f}Cr

### 💡 Key Insight
Hybrid ML model ({req.forecast_days}d horizon) predicts a {'positive' if forecast_return > 0 else 'negative'} move on {company}.
{'Model accuracy below 55% — weight technicals/fundamentals more.' if req.model_accuracy < 55 else 'Model shows reasonable directional accuracy on test data.'}

### ⚠️ Risk Factors
- Model predictions are probabilistic, not guarantees
- External market factors may override technical signals

*Analysis generated using template mode — provide DeepSeek/Gemini API keys for AI-powered analysis.*"""


def _build_claude_expert_prompt(ticker: str, req: AiAnalysisRequest) -> str:
    from ui.ai_analysis import generate_claude_expert_analysis_prompt
    import pandas as pd

    company = ticker.replace(".NS", "").replace(".BO", "").upper()
    forecast_return = ((req.forecast_target - req.current_price) / req.current_price) * 100
    pred_df = pd.DataFrame({
        "Predicted Price": [req.forecast_target],
    })
    metrics = {
        "accuracy": req.model_accuracy,
        "rmse": 0.0,
        "last_directional_prob": req.directional_prob,
    }
    fundamentals = {
        "Forward P/E": req.forward_pe,
        "Price/Book": req.price_book,
        "PEG Ratio": req.peg,
    }
    technical_indicators = {
        "RSI": req.technicals.rsi,
        "MACD_Histogram": req.technicals.macd_histogram,
        "Price_vs_MA20": req.technicals.price_vs_ma20,
        "Volatility_5D": req.technicals.volatility_5d,
    }
    try:
        return generate_claude_expert_analysis_prompt(
            stock_symbol=f"{company}.NS",
            current_price=req.current_price,
            predicted_prices=pred_df,
            metrics=metrics,
            fundamentals=fundamentals,
            sentiment_summary={},
            technical_indicators=technical_indicators,
        )
    except Exception:
        return ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/stocks/{ticker}/ai-analysis",
    response_model=AiAnalysisResponse,
    summary="AI Expert Analysis via DeepSeek / Gemini / template fallback",
)
def ai_analysis(ticker: str, req: AiAnalysisRequest) -> AiAnalysisResponse:
    """
    Replicates the Streamlit `generate_ai_analysis()` flow:
    1. Try DeepSeek V3 (primary)
    2. Fallback to Gemini 2.5 Flash
    3. Fallback to template
    Always returns `claude_prompt` so the UI can offer a copy-paste alternative.
    """
    from config.settings import DEEPSEEK_API_KEY, GEMINI_API_KEY

    context = _build_analysis_context(ticker, req)
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(context)

    # Build Claude expert prompt (always)
    claude_prompt = _build_claude_expert_prompt(ticker, req)

    # 1 — DeepSeek
    if DEEPSEEK_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            return AiAnalysisResponse(
                analysis=resp.choices[0].message.content,
                source="deepseek",
                claude_prompt=claude_prompt,
            )
        except Exception:
            pass

    # 2 — Gemini
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(user_prompt)
            return AiAnalysisResponse(
                analysis=resp.text,
                source="gemini",
                claude_prompt=claude_prompt,
            )
        except Exception:
            pass

    # 3 — Template fallback
    return AiAnalysisResponse(
        analysis=_template_analysis(ticker, req),
        source="template",
        claude_prompt=claude_prompt,
    )


@router.post(
    "/stocks/{ticker}/signal-synthesis",
    response_model=SignalSynthesisResponse,
    summary="Generate 5-layer signal synthesis prompt for claude.ai",
)
def signal_synthesis(ticker: str, req: SignalSynthesisRequest) -> SignalSynthesisResponse:
    """
    Builds the full-stack signal synthesis prompt (ML · Technicals · FII/DII ·
    Options · LLM sentiment) for pasting into claude.ai. Mirrors the Streamlit
    P3 Signal Synthesis expander exactly.
    """
    try:
        from services.llm_sentiment_alpha import generate_signal_synthesis_prompt
        prompt = generate_signal_synthesis_prompt(
            ticker=ticker,
            current_price=req.current_price,
            forecast_return_pct=req.forecast_return_pct,
            directional_prob=req.directional_prob,
            model_accuracy=req.model_accuracy,
            hurst=req.hurst,
            rsi=req.rsi,
            macd_hist=req.macd_hist,
            volatility_20d=req.volatility_20d,
            fii_net_5d_cr=req.fii_net_5d_cr,
            dii_net_5d_cr=req.dii_net_5d_cr,
            vix=req.vix,
            llm_sentiment_signal=req.llm_sentiment_signal,
            llm_mean_sentiment=req.llm_mean_sentiment,
            llm_mean_materiality=req.llm_mean_materiality,
            llm_top_headline=req.llm_top_headline,
            multi_sentiment_score=req.multi_sentiment_score,
            options_pcr_signal=req.options_pcr_signal,
            options_iv_skew_signal=req.options_iv_skew_signal,
            options_combined=req.options_combined,
            shap_top_features=req.shap_top_features,
            forecast_days=req.forecast_days,
            regime=req.regime,
        )
        return SignalSynthesisResponse(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Signal synthesis failed: {e}") from e
