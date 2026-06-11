"use client";

import { useState, useRef } from "react";
import type { PredictionBundle, PredictionJobAccepted, StockHistory, FiiDiiRow, TechnicalSnapshot } from "@/lib/types";
import Stat from "@/components/Stat";
import ProgressLoader from "@/components/ProgressLoader";
import HorizontalBarChart from "../charts/HorizontalBarChart";
import PredictedVsActualChart from "../charts/PredictedVsActualChart";
import PriceForecastBandsChart from "../charts/PriceForecastBandsChart";
import ModelAccuracyChart from "../charts/ModelAccuracyChart";
import { apiPost, apiGet, waitForJob } from "@/lib/api-client";
import usePersistedState from "@/lib/usePersistedState";
import { formatINR, formatPercent, formatRatio, toneFor } from "@/lib/format";
import { FII_DII_STORAGE_KEY } from "./FiiDiiTab";
import AiAnalysisPanel from "./AiAnalysisPanel";
import SignalSynthesisPanel from "./SignalSynthesisPanel";
import SwingSignalPanel from "./SwingSignalPanel";
const FII_DII_SAVED_AT_KEY = "fii_dii:saved_at:v1";

interface LlmSentimentScores {
  combined_signal: number;
  mean_sentiment: number;
  mean_materiality: number;
  mean_novelty: number;
  top_headline: string;
}

function isFiiDiiStale(): boolean {
  try {
    const savedAt = localStorage.getItem(FII_DII_SAVED_AT_KEY);
    if (!savedAt) return true;
    // IST "YYYY-MM-DD"
    const now = new Date(Date.now() + 5.5 * 60 * 60 * 1000);
    const today = now.toISOString().slice(0, 10);
    return savedAt < today;
  } catch {
    return false;
  }
}

interface Props {
  ticker: string;
  ohlcv: StockHistory | null;
}

interface RunState {
  startedAt: number | null;
  finishedAt: number | null;
  phase: string | null;
  error: string | null;
}

const INITIAL_STATE: RunState = {
  startedAt: null,
  finishedAt: null,
  phase: null,
  error: null,
};

export default function OverviewTab({ ticker, ohlcv }: Props) {
  // Persist horizon + bundle so flipping to Accuracy/Patterns and back doesn't
  // wipe a long-running prediction result.
  const [horizon, setHorizon] = usePersistedState<number>(
    `pred:${ticker}:horizon`,
    10,
  );
  const [bundle, setBundle] = usePersistedState<PredictionBundle | null>(
    `pred:${ticker}:bundle`,
    null,
  );
  const [run, setRun] = useState<RunState>(INITIAL_STATE);
  const [busy, setBusy] = useState(false);
  const [technicals, setTechnicals] = useState<TechnicalSnapshot | null>(null);

  // FII/DII inline-prompt state
  const [fiiDiiModalOpen, setFiiDiiModalOpen] = useState(false);
  const [fiiDiiPasteText, setFiiDiiPasteText] = useState("");
  const [fiiDiiPasteError, setFiiDiiPasteError] = useState<string | null>(null);
  // Resolved FII/DII rows to attach to the next prediction run
  const pendingFiiDiiRowsRef = useRef<FiiDiiRow[] | null>(null);
  // Pending prediction horizon when waiting for FII/DII modal
  const pendingHorizonRef = useRef<number>(horizon);

  // LLM pre-prediction sentiment scoring modal
  const [llmModalOpen, setLlmModalOpen] = useState(false);
  const [llmPrompt, setLlmPrompt] = useState("");
  const [llmPasteText, setLlmPasteText] = useState("");
  const [llmPasteError, setLlmPasteError] = useState<string | null>(null);
  const [llmCopied, setLlmCopied] = useState(false);
  // FII/DII rows held while waiting for user to complete LLM modal
  const pendingLlmFiiDiiRef = useRef<FiiDiiRow[] | null>(null);

  /** Read FII/DII rows from localStorage (written by FiiDiiTab). Returns null if stale or missing. */
  function loadSavedFiiDiiRows(): FiiDiiRow[] | null {
    try {
      if (isFiiDiiStale()) return null;
      const raw = localStorage.getItem(FII_DII_STORAGE_KEY);
      if (!raw) return null;
      const rows = JSON.parse(raw) as FiiDiiRow[];
      return rows?.length ? rows : null;
    } catch {
      return null;
    }
  }

  /** Parse JSON (NSE format or legacy) or tab/comma-separated paste into FiiDiiRow[]  */
  function parseFiiDiiPaste(raw: string): FiiDiiRow[] | null {
    try {
      const trimmed = raw.trim();
      // Try JSON first (NSE API format)
      if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
        const parsed = JSON.parse(trimmed);
        const arr: Record<string, unknown>[] = Array.isArray(parsed) ? parsed : parsed.data ?? [];
        // NSE category format: [{category, date, buyValue, sellValue, netValue},...]
        const isNseFormat = arr.length > 0 && "category" in arr[0] && "buyValue" in arr[0];
        if (isNseFormat) {
          const byDate: Record<string, FiiDiiRow> = {};
          for (const r of arr) {
            const date = String(r.date ?? "");
            const cat = String(r.category ?? "").toUpperCase();
            if (!byDate[date]) byDate[date] = { date, fii_buy: null, fii_sell: null, fii_net: null, dii_buy: null, dii_sell: null, dii_net: null };
            const pf = (v: unknown) => { const n = parseFloat(String(v ?? "")); return isNaN(n) ? null : n; };
            if (cat.includes("FII") || cat.includes("FPI")) {
              byDate[date].fii_buy = pf(r.buyValue);
              byDate[date].fii_sell = pf(r.sellValue);
              byDate[date].fii_net = pf(r.netValue);
            } else if (cat.includes("DII")) {
              byDate[date].dii_buy = pf(r.buyValue);
              byDate[date].dii_sell = pf(r.sellValue);
              byDate[date].dii_net = pf(r.netValue);
            }
          }
          const rows = Object.values(byDate).filter((r) => r.date && /\d/.test(r.date));
          return rows.length ? rows : null;
        }
        // Generic JSON row array
        const rows = arr.map((r) => ({
          date: String(r.date ?? r.tradeDate ?? ""),
          fii_buy: parseFloat(String(r.fii_buy ?? r.fiiBuy ?? "")) || null,
          fii_sell: parseFloat(String(r.fii_sell ?? r.fiiSell ?? "")) || null,
          fii_net: parseFloat(String(r.fii_net ?? r.fiiNet ?? "")) || null,
          dii_net: parseFloat(String(r.dii_net ?? r.diiNet ?? "")) || null,
          dii_buy: null,
          dii_sell: null,
        } as FiiDiiRow)).filter((r) => r.date && /\d/.test(r.date));
        return rows.length ? rows : null;
      }
      // Fall back to tab/comma-separated
      const lines = raw.trim().split("\n").filter(Boolean);
      const rows = lines
        .map((line) => {
          const cols = line.split(/[\t,|]/).map((c) => c.trim());
          return {
            date: cols[0],
            fii_buy: parseFloat(cols[1]) || null,
            fii_sell: parseFloat(cols[2]) || null,
            fii_net: parseFloat(cols[3]) || null,
            dii_net: parseFloat(cols[4]) || null,
            dii_buy: null,
            dii_sell: null,
          } as FiiDiiRow;
        })
        .filter((r) => r.date && /\d/.test(r.date));
      return rows.length ? rows : null;
    } catch {
      return null;
    }
  }

  async function _doRunPredict(fiiDiiRows: FiiDiiRow[] | null, llmScores: LlmSentimentScores | null = null) {
    setBusy(true);
    setRun({ startedAt: Date.now(), finishedAt: null, phase: "queued", error: null });
    setBundle(null);
    try {
      const body: Record<string, unknown> = {
        horizon_days: pendingHorizonRef.current,
      };
      if (fiiDiiRows && fiiDiiRows.length > 0) {
        body.fii_dii_rows = fiiDiiRows;
      }
      if (llmScores) {
        body.llm_sentiment_signal  = llmScores.combined_signal;
        body.llm_mean_sentiment    = llmScores.mean_sentiment;
        body.llm_mean_materiality  = llmScores.mean_materiality;
        body.llm_top_headline      = llmScores.top_headline;
      }
      const accepted = await apiPost<PredictionJobAccepted>(
        `/api/v1/stocks/${ticker}/predict`,
        body,
      );
      setRun((s) => ({ ...s, phase: "started" }));
      const result = await waitForJob<PredictionBundle>(accepted.job_id, {
        onPhase: (p) => setRun((s) => ({ ...s, phase: p })),
      });
      setBundle(result);
      setRun((s) => ({ ...s, phase: "succeeded", finishedAt: Date.now() }));
      // Fetch technicals for AiAnalysisPanel + SignalSynthesisPanel
      try {
        const tech = await apiGet<TechnicalSnapshot>(`/api/v1/stocks/${ticker}/technicals`);
        setTechnicals(tech);
      } catch { /* non-fatal */ }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setRun((s) => ({ ...s, error: message, phase: "failed", finishedAt: Date.now() }));
    } finally {
      setBusy(false);
    }
  }

  /**
   * Step 2 of the pre-prediction pipeline: after FII/DII data is resolved,
   * try to score LLM sentiment BEFORE starting the model job.
   * - ANTHROPIC_API_KEY set → auto-scores in ~3s, runs immediately.
   * - Not set → shows a modal so the user can paste into claude.ai first.
   * - Timeout / error → skips LLM and runs without sentiment scores.
   */
  async function _afterFiiDii(fiiDiiRows: FiiDiiRow[] | null) {
    // Show loading state while fetching/scoring headlines
    setBusy(true);
    setRun({ startedAt: Date.now(), finishedAt: null, phase: "scoring sentiment…", error: null });
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 8_000);
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const resp = await fetch(
        `${API_BASE}/api/v1/stocks/${ticker}/sentiment/llm-alpha?max_headlines=10`,
        { signal: ctrl.signal },
      );
      clearTimeout(timer);
      if (resp.ok) {
        const d = await resp.json() as {
          available: boolean; needs_manual_input: boolean; manual_prompt: string;
          combined_signal: number; mean_sentiment: number; mean_materiality: number;
          mean_novelty: number; top_headline: string; headlines_scored: number;
        };
        if (!d.needs_manual_input && d.available && d.headlines_scored > 0) {
          // Auto-scored via ANTHROPIC_API_KEY — feed directly into prediction
          await _doRunPredict(fiiDiiRows, {
            combined_signal: d.combined_signal,
            mean_sentiment:  d.mean_sentiment,
            mean_materiality: d.mean_materiality,
            mean_novelty:    d.mean_novelty,
            top_headline:    d.top_headline,
          });
          return;
        }
        if (d.needs_manual_input && d.manual_prompt) {
          // No API key — release the busy lock and show the LLM modal
          setBusy(false);
          setRun(INITIAL_STATE);
          pendingLlmFiiDiiRef.current = fiiDiiRows;
          setLlmPrompt(d.manual_prompt);
          setLlmPasteText("");
          setLlmPasteError(null);
          setLlmModalOpen(true);
          return;
        }
      }
    } catch { /* timeout / network — skip LLM step and run without scores */ }
    // Fall-through: no LLM scores available — release busy and run directly
    setBusy(false);
    setRun(INITIAL_STATE);
    await _doRunPredict(fiiDiiRows, null);
  }

  async function runPredict(e: React.FormEvent) {
    e.preventDefault();
    pendingHorizonRef.current = horizon;

    // Step 1: Check if FII/DII data is already saved from the FII/DII tab
    const savedRows = loadSavedFiiDiiRows();
    if (savedRows) {
      await _afterFiiDii(savedRows);
      return;
    }

    // No FII/DII data — try to auto-fetch quickly (timeout 4s)
    let autoFetchedRows: FiiDiiRow[] | null = null;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 4000);
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const resp = await fetch(`${API_BASE}/api/v1/stocks/fii-dii?lookback_days=30`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (resp.ok) {
        const data = await resp.json();
        if (data?.rows?.length) {
          autoFetchedRows = data.rows as FiiDiiRow[];
          try {
            localStorage.setItem(FII_DII_STORAGE_KEY, JSON.stringify(autoFetchedRows));
            window.dispatchEvent(new CustomEvent("fii-dii-updated"));
          } catch {}
        }
      }
    } catch {
      // NSE throttled or timeout — show the FII/DII modal
    }

    if (autoFetchedRows) {
      await _afterFiiDii(autoFetchedRows);
      return;
    }

    // NSE throttled and no saved data — prompt user for FII/DII first
    pendingFiiDiiRowsRef.current = null;
    setFiiDiiPasteText("");
    setFiiDiiPasteError(null);
    setFiiDiiModalOpen(true);
  }

  function handleFiiDiiModalSubmit() {
    setFiiDiiPasteError(null);
    const rows = parseFiiDiiPaste(fiiDiiPasteText);
    if (!rows) {
      setFiiDiiPasteError("Could not parse. Use tab-separated: Date | FII Buy | FII Sell | FII Net | DII Net");
      return;
    }
    try {
      localStorage.setItem(FII_DII_STORAGE_KEY, JSON.stringify(rows));
      window.dispatchEvent(new CustomEvent("fii-dii-updated"));
    } catch {}
    setFiiDiiModalOpen(false);
    _afterFiiDii(rows);
  }

  function handleFiiDiiModalSkip() {
    setFiiDiiModalOpen(false);
    _afterFiiDii(null);
  }

  async function handleLlmModalSubmit() {
    setLlmPasteError(null);
    if (!llmPasteText.trim()) {
      setLlmPasteError("Please paste Claude's JSON response.");
      return;
    }
    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const resp = await fetch(
        `${API_BASE}/api/v1/stocks/${ticker}/sentiment/llm-alpha/manual`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ manual_response: llmPasteText }),
        },
      );
      if (!resp.ok) throw new Error(`Status ${resp.status}`);
      const d = await resp.json() as {
        combined_signal: number; mean_sentiment: number;
        mean_materiality: number; mean_novelty: number; top_headline: string;
      };
      setLlmModalOpen(false);
      await _doRunPredict(pendingLlmFiiDiiRef.current, {
        combined_signal:  d.combined_signal,
        mean_sentiment:   d.mean_sentiment,
        mean_materiality: d.mean_materiality,
        mean_novelty:     d.mean_novelty,
        top_headline:     d.top_headline,
      });
    } catch (e) {
      setLlmPasteError(e instanceof Error ? e.message : "Failed to parse Claude response. Check the JSON format.");
    }
  }

  function handleLlmModalSkip() {
    setLlmModalOpen(false);
    _doRunPredict(pendingLlmFiiDiiRef.current, null);
  }



  function clearPrediction() {
    setBundle(null);
    setRun(INITIAL_STATE);
  }

  const lastClose = ohlcv?.bars.length ? ohlcv.bars[ohlcv.bars.length - 1].close : null;
  const firstClose = ohlcv?.bars.length ? ohlcv.bars[0].close : null;
  const windowChange =
    lastClose != null && firstClose != null && firstClose !== 0
      ? ((lastClose - firstClose) / firstClose) * 100
      : null;

  const lastPoint = bundle?.points?.[bundle.points.length - 1] ?? null;
  const predReturn =
    bundle && bundle.anchor_price && lastPoint
      ? ((lastPoint.pred_price - bundle.anchor_price) / bundle.anchor_price) * 100
      : null;

  return (
    <div className="space-y-6">
      {/* FII/DII data prompt modal — shown when NSE is throttled and no cached data exists */}
      {fiiDiiModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-surface border border-border rounded-xl shadow-2xl max-w-lg w-full p-6 space-y-4">
            <div>
              <h3 className="text-lg font-semibold">FII / DII data required</h3>
              <p className="text-sm text-muted mt-1">
                NSE is throttled and no saved FII/DII data was found. The model uses FII/DII institutional
                flows as a key feature. You can paste recent data from{" "}
                <a
                  href="https://www.nseindia.com/api/fiidiiTradeReact"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent underline"
                >
                  nseindia.com/api/fiidiiTradeReact
                </a>{" "}
                (visit{" "}
                <a href="https://www.nseindia.com" target="_blank" rel="noopener noreferrer" className="text-accent underline">
                  nseindia.com
                </a>
                {" "}first to get the session cookie, then copy-paste the raw JSON), or skip to run
                without this feature.
              </p>
            </div>
            <textarea
              className="input w-full h-40 font-mono text-xs resize-y"
              placeholder={`[{"category":"DII","date":"23-Apr-2026","buyValue":"18498.19","sellValue":"17556.84","netValue":"941.35"},{"category":"FII/FPI","date":"23-Apr-2026","buyValue":"12829.12","sellValue":"16083.83","netValue":"-3254.71"}]`}
              value={fiiDiiPasteText}
              onChange={(e) => setFiiDiiPasteText(e.target.value)}
            />
            {fiiDiiPasteError && (
              <p className="text-xs text-bear">{fiiDiiPasteError}</p>
            )}
            <div className="flex gap-3 justify-end">
              <button
                type="button"
                onClick={handleFiiDiiModalSkip}
                className="btn text-sm"
              >
                Skip (run without FII/DII)
              </button>
              <button
                type="button"
                onClick={handleFiiDiiModalSubmit}
                className="btn btn-primary text-sm"
                disabled={!fiiDiiPasteText.trim()}
              >
                Use this data &amp; run prediction
              </button>
            </div>
          </div>
        </div>
      )}

      {/* LLM sentiment scoring modal — scores headlines BEFORE the model runs */}
      {llmModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-surface border border-border rounded-xl shadow-2xl max-w-xl w-full p-6 space-y-4">
            <div>
              <h3 className="text-lg font-semibold">🧠 LLM Sentiment Scoring</h3>
              <p className="text-sm text-muted mt-1">
                Score these headlines <strong>before the prediction runs</strong> — the scores are fed into
                the model as sentiment features (better than keyword-only FinBERT). Copy the prompt,
                paste into{" "}
                <a href="https://claude.ai" target="_blank" rel="noopener noreferrer" className="text-accent underline">
                  claude.ai
                </a>
                {" "}(free), then paste Claude&apos;s JSON array back here.
              </p>
            </div>
            <div className="space-y-1">
              <textarea readOnly className="input w-full h-52 font-mono text-xs resize-y" value={llmPrompt} />
              <button
                type="button"
                className="btn text-xs"
                onClick={() => {
                  navigator.clipboard.writeText(llmPrompt);
                  setLlmCopied(true);
                  setTimeout(() => setLlmCopied(false), 2000);
                }}
              >
                {llmCopied ? "Copied!" : "Copy prompt"}
              </button>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted">Paste Claude&apos;s JSON response:</p>
              <textarea
                className="input w-full h-28 font-mono text-xs resize-y"
                placeholder={'[{"i":1,"novelty":0.8,"materiality":0.7,"sentiment":0.6},{"i":2,...}]'}
                value={llmPasteText}
                onChange={(e) => setLlmPasteText(e.target.value)}
              />
              {llmPasteError && <p className="text-xs text-bear">{llmPasteError}</p>}
            </div>
            <div className="flex gap-3 justify-end">
              <button type="button" onClick={handleLlmModalSkip} className="btn text-sm">
                Skip — run without LLM
              </button>
              <button
                type="button"
                onClick={handleLlmModalSubmit}
                className="btn btn-primary text-sm"
                disabled={!llmPasteText.trim()}
              >
                Use scores &amp; run prediction
              </button>
            </div>
          </div>
        </div>
      )}

      <section className="panel">
        {ohlcv && ohlcv.bars.length > 0 ? (
          <div className="flex items-baseline justify-between mb-2">
            <div>
              <p className="text-xs uppercase text-muted">Last close</p>
              <p className="text-2xl font-semibold">{formatINR(lastClose)}</p>
            </div>
            {windowChange != null && (
              <p
                className={`text-sm ${
                  toneFor(windowChange) === "bull"
                    ? "text-bull"
                    : toneFor(windowChange) === "bear"
                    ? "text-bear"
                    : "text-muted"
                }`}
              >
                {formatPercent(windowChange)} · {ohlcv.bars.length} bars · {ohlcv.start} →{" "}
                {ohlcv.end}
              </p>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted">
            OHLCV unavailable — upstream may be rate-limited.
          </p>
        )}
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Forecast</h2>
          <span className="text-xs text-muted">
            {bundle?.model_version && `Model: ${bundle.model_version}`}
          </span>
        </div>
        <form onSubmit={runPredict} className="panel flex items-end gap-3 flex-wrap">
          <label className="flex-1 min-w-[160px]">
            <span className="block text-xs uppercase text-muted mb-1">
              Horizon (days)
            </span>
            <input
              type="number"
              min={1}
              max={60}
              className="input"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              suppressHydrationWarning
            />
          </label>
          <button type="submit" disabled={busy} className="btn btn-primary" suppressHydrationWarning>
            {busy ? "Analyzing…" : bundle ? "Re-run prediction" : "Run prediction"}
          </button>
          {bundle && !busy && (
            <button type="button" onClick={clearPrediction} className="btn">
              Clear
            </button>
          )}
        </form>

        <ProgressLoader
          active={busy}
          phase={run.phase}
          startedAt={run.startedAt}
          estimatedSeconds={150}
          steps={[
            { key: "scoring sentiment…", label: "LLM sentiment" },
            { key: "queued", label: "Queued" },
            { key: "started", label: "Loading data" },
            { key: "running", label: "Fitting ensemble" },
            { key: "succeeded", label: "Done" },
          ]}
          error={run.error}
        />

        {bundle?.trading_suspended && (
          <div className="panel border border-bear bg-bear/10 text-sm text-bear flex gap-2 items-start">
            <span className="text-lg leading-none mt-0.5">⛔</span>
            <div>
              <p className="font-semibold">Trading Suspended</p>
              <p className="text-xs text-muted mt-0.5">
                Rolling 60-day Sharpe fell below 0 for 3 consecutive days. The model is in diagnostic-only
                mode. Do not trade on these signals until live Sharpe recovers.
              </p>
            </div>
          </div>
        )}

        {bundle?.calibration_gated && (
          <div className="panel border border-yellow-500 bg-yellow-500/10 text-sm text-yellow-400 flex gap-2 items-start">
            <span className="text-lg leading-none mt-0.5">⚠️</span>
            <div>
              <p className="font-semibold">Calibration Gate Active</p>
              <p className="text-xs text-muted mt-0.5">
                Rolling ECE exceeds 15% on ≥10 resolved predictions. Stated probabilities are
                unreliable — the model is systematically over- or under-confident. Check the
                Accuracy tab for details.
              </p>
            </div>
          </div>
        )}

        {bundle && (
          <>
            <VerdictBanner bundle={bundle} lastClose={lastClose} />

            <div className="panel grid gap-4 sm:grid-cols-4">
              <Stat
                label={`Target (${bundle.horizon_days}d)`}
                value={formatINR(lastPoint?.pred_price)}
              />
              <Stat
                label="Expected return"
                value={predReturn == null ? "—" : formatPercent(predReturn)}
                tone={toneFor(predReturn)}
              />
              <Stat
                label="P(up)"
                value={
                  bundle.last_directional_prob != null
                    ? formatPercent(bundle.last_directional_prob)
                    : "—"
                }
                tone={
                  bundle.last_directional_prob != null
                    ? bundle.last_directional_prob >= 50
                      ? "bull"
                      : "bear"
                    : "muted"
                }
                hint={
                  bundle.threshold_tuning
                    ? `τ* = ${(bundle.threshold_tuning.tau_star * 100).toFixed(1)}%`
                    : undefined
                }
              />
              <Stat
                label={`${Math.round(bundle.confidence_level * 100)}% band (day 1)`}
                value={
                  bundle.points[0]?.ci_low != null && bundle.points[0]?.ci_high != null
                    ? `${formatINR(bundle.points[0].ci_low)} – ${formatINR(
                        bundle.points[0].ci_high,
                      )}`
                    : "—"
                }
                hint={
                  bundle.conformal_halfwidth != null
                    ? `conformal ±\u20b9${bundle.conformal_halfwidth.toFixed(2)}`
                    : undefined
                }
              />
            </div>

            {/* Streamlit-parity charts */}
            <PredictedVsActualChart series={bundle.test_predictions ?? []} />

            {ohlcv && ohlcv.bars.length > 0 && (
              <PriceForecastBandsChart
                bars={ohlcv.bars}
                predictions={bundle.points}
                anchorPrice={bundle.anchor_price ?? lastClose}
                ticker={ticker}
                bullishProb={bundle.last_directional_prob}
                regime={bundle.regime}
              />
            )}

            {ohlcv && ohlcv.bars.length > 0 && (
              <ModelAccuracyChart
                bars={ohlcv.bars}
                testSeries={bundle.test_predictions ?? []}
                forecastPoints={bundle.points}
                anchorPrice={bundle.anchor_price ?? lastClose}
              />
            )}

            {bundle.regime && (
              <div className="panel grid gap-4 sm:grid-cols-3 text-sm">
                <div>
                  <p className="text-xs uppercase text-muted">Regime</p>
                  <p className="font-medium">{bundle.regime}</p>
                  {bundle.regime_detail && (
                    <p className="text-xs text-muted mt-0.5">{bundle.regime_detail}</p>
                  )}
                </div>
                {bundle.hurst_exponent != null && (
                  <div>
                    <p className="text-xs uppercase text-muted">Hurst exponent</p>
                    <p className="font-medium">{formatRatio(bundle.hurst_exponent, 3)}</p>
                    <p className="text-xs text-muted">
                      {bundle.hurst_exponent > 0.55
                        ? "Trending"
                        : bundle.hurst_exponent < 0.45
                        ? "Mean-reverting"
                        : "Random walk"}
                    </p>
                  </div>
                )}
                {bundle.walkforward && bundle.walkforward.accuracy != null && (
                  <div>
                    <p className="text-xs uppercase text-muted">Walk-forward accuracy</p>
                    <p className="font-medium">
                      {formatPercent((bundle.walkforward.accuracy ?? 0) * 100, 1)}
                      {bundle.naive_accuracy != null && (
                        <span className="ml-1 text-xs text-muted font-normal">
                          {" "}
                          vs naive {formatPercent(bundle.naive_accuracy, 1)}
                          {" "}
                          <span className={(bundle.walkforward.accuracy ?? 0) * 100 > bundle.naive_accuracy ? "text-green-500" : "text-red-400"}>
                            ({((bundle.walkforward.accuracy ?? 0) * 100 - bundle.naive_accuracy) >= 0 ? "+" : ""}
                            {((bundle.walkforward.accuracy ?? 0) * 100 - bundle.naive_accuracy).toFixed(1)}pp)
                          </span>
                        </span>
                      )}
                    </p>
                    <p className="text-xs text-muted">
                      {bundle.walkforward.n_windows} windows
                      {bundle.walkforward.std != null
                        ? ` · σ ${(bundle.walkforward.std * 100).toFixed(1)}%`
                        : ""}
                    </p>
                  </div>
                )}
                {bundle.conviction_precision != null && (
                  <div>
                    <p className="text-xs uppercase text-muted">Conviction precision</p>
                    <p className="font-medium">
                      {bundle.conviction_precision.bull_precision != null
                        ? formatPercent(bundle.conviction_precision.bull_precision * 100, 1)
                        : "—"}
                      <span className={
                        bundle.conviction_precision.bull_precision != null &&
                        bundle.conviction_precision.bull_precision >= 0.55
                          ? "ml-1 text-xs text-green-500"
                          : "ml-1 text-xs text-red-400"
                      }>
                        {bundle.conviction_precision.bull_precision != null
                          ? bundle.conviction_precision.bull_precision >= 0.55 ? "✓ Go" : "✗ No-go"
                          : ""}
                      </span>
                    </p>
                    <p className="text-xs text-muted">
                      Bull calls ≥{(bundle.conviction_precision.threshold * 100).toFixed(0)}%
                      · n={bundle.conviction_precision.n_high_bull}
                    </p>
                  </div>
                )}
              </div>
            )}

            {bundle.v2_blend?.used && (
              <div className="panel">
                <p className="text-sm font-semibold mb-2">🤖 Sentiment v2 blend</p>
                <div className="grid gap-4 sm:grid-cols-4 text-sm">
                  <Stat
                    label="Stacker P(up)"
                    value={formatPercent(bundle.v2_blend.stacker_prob * 100)}
                  />
                  <Stat
                    label="v2 P(up)"
                    value={
                      bundle.v2_blend.v2_prob != null
                        ? formatPercent(bundle.v2_blend.v2_prob * 100)
                        : "—"
                    }
                  />
                  <Stat
                    label="Blended"
                    value={formatPercent(bundle.v2_blend.blended_prob * 100)}
                    hint={`weight v2 = ${(bundle.v2_blend.weight_v2 * 100).toFixed(0)}%`}
                  />
                  <Stat
                    label="Headlines"
                    value={String(bundle.v2_blend.n_headlines)}
                    hint={
                      bundle.v2_blend.stacker_available
                        ? "stacker OK"
                        : "fallback weighted-avg"
                    }
                  />
                </div>
              </div>
            )}

            <SwingSignalPanel signal={bundle.swing_signal} />

            <AiAnalysisPanel ticker={ticker} bundle={bundle} technicals={technicals} />

            <SignalSynthesisPanel ticker={ticker} bundle={bundle} technicals={technicals} />

            <ClaudePromptPanel bundle={bundle} ticker={ticker} lastClose={lastClose} />

            {bundle.shap_top_features && bundle.shap_top_features.length > 0 && (
              <ShapPanel features={bundle.shap_top_features} method={bundle.shap_method} />
            )}

            {bundle.rmse_breakdown && (
              <div className="panel">
                <p className="text-sm font-semibold mb-2">Base-learner RMSE (test fold)</p>
                <div className="grid gap-4 sm:grid-cols-5 text-sm">
                  {(
                    [
                      ["XGB", bundle.rmse_breakdown.xgb],
                      ["LightGBM", bundle.rmse_breakdown.lgbm],
                      ["CatBoost", bundle.rmse_breakdown.catboost],
                      ["GRU", bundle.rmse_breakdown.rnn],
                      ["Stacked", bundle.rmse_breakdown.stacked],
                    ] as const
                  ).map(([k, v]) => (
                    <Stat key={k} label={k} value={formatRatio(v, 4)} />
                  ))}
                </div>
              </div>
            )}

            {bundle.calibration && (
              <div className="panel">
                <p className="text-sm font-semibold mb-2">Calibration (holdout)</p>
                <div className="grid gap-4 sm:grid-cols-3 text-sm">
                  <Stat
                    label="Expected Calibration Error"
                    value={formatPercent(bundle.calibration.ece * 100, 2)}
                    tone={bundle.calibration.ece <= 0.05 ? "bull" : "bear"}
                    hint="Trustworthy ≤ 5%"
                  />
                  <Stat
                    label="Brier score"
                    value={formatRatio(bundle.calibration.brier_score, 4)}
                  />
                  <Stat
                    label="n_samples"
                    value={String(bundle.calibration.n_samples)}
                  />
                </div>
                <ReliabilityChart
                  bins={bundle.calibration.bin_predicted.map((p, i) => ({
                    predicted: p,
                    actual: bundle.calibration!.bin_actual[i] ?? 0,
                    count: bundle.calibration!.bin_counts[i] ?? 0,
                  }))}
                />
              </div>
            )}

            {bundle.points.length > 0 && (
              <div className="panel overflow-x-auto">
                <p className="text-sm font-semibold mb-2">Forecast trajectory</p>
                <table className="w-full text-sm">
                  <thead className="text-xs uppercase text-muted">
                    <tr>
                      <th className="text-left pb-1">Date</th>
                      <th className="text-right pb-1">Price</th>
                      <th className="text-right pb-1">Dir</th>
                      <th className="text-right pb-1">P(up)</th>
                      <th className="text-right pb-1">CI low</th>
                      <th className="text-right pb-1">CI high</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bundle.points.map((p, i) => (
                      <tr key={i} className="border-t border-border">
                        <td className="py-1">{p.target_date}</td>
                        <td className="py-1 text-right">{formatINR(p.pred_price)}</td>
                        <td
                          className={`py-1 text-right ${
                            p.direction === "up"
                              ? "text-bull"
                              : p.direction === "down"
                              ? "text-bear"
                              : "text-muted"
                          }`}
                        >
                          {p.direction}
                        </td>
                        <td className="py-1 text-right">
                          {p.prob_up != null ? `${(p.prob_up * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="py-1 text-right">
                          {p.ci_low != null ? formatINR(p.ci_low) : "—"}
                        </td>
                        <td className="py-1 text-right">
                          {p.ci_high != null ? formatINR(p.ci_high) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}

// ─── Claude expert prompt panel ─────────────────────────────────────────────

function buildClaudePrompt(bundle: PredictionBundle, ticker: string, lastClose: number | null): string {
  const company = ticker.replace(/\.(NS|BO)$/, "").toUpperCase();
  const lastPoint = bundle.points[bundle.points.length - 1];
  const target = lastPoint?.pred_price ?? lastClose ?? 0;
  const anchor = bundle.anchor_price ?? lastClose ?? 0;
  const forecastReturn = anchor ? ((target - anchor) / anchor) * 100 : 0;
  const acc = (bundle.walkforward?.accuracy ?? 0) * 100;
  const prob = bundle.last_directional_prob ?? 50;
  const tau = bundle.threshold_tuning?.tau_star ?? 0.55;

  const hurst = bundle.hurst_exponent;
  const hurstStr = hurst != null
    ? `${hurst.toFixed(3)} — ${hurst > 0.55 ? "Trending (momentum biased)" : hurst < 0.45 ? "Mean-Reverting (fade extremes)" : "Random Walk (no regime edge)"}`
    : "N/A";
  const modelTier = acc >= 65 ? "Research-grade (≥65%)"
    : acc >= 55 ? "Production-grade (55–65%)"
    : "Uncertain (<55% — down-weight model signal)";

  // SHAP breakdown
  const shapFeatures = bundle.shap_top_features ?? [];
  const topShap = shapFeatures.slice(0, 6)
    .map((f) => `  ${f.feature.padEnd(22)} ${f.importance >= 0 ? "+" : ""}${f.importance.toFixed(4)}`)
    .join("\n");
  const bullDrivers = shapFeatures.filter((f) => f.importance > 0).map((f) => f.feature).join(", ") || "none";
  const bearDrivers = shapFeatures.filter((f) => f.importance < 0).map((f) => f.feature).join(", ") || "none";

  // Confidence interval (day 1)
  const ci1 = bundle.points[0];
  const ciStr = ci1?.ci_low != null && ci1?.ci_high != null
    ? `₹${ci1.ci_low.toFixed(2)} – ₹${ci1.ci_high.toFixed(2)}`
    : "N/A";
  const hwStr = bundle.conformal_halfwidth != null
    ? `±₹${bundle.conformal_halfwidth.toFixed(2)} (use for stop sizing)`
    : "N/A";

  // Conviction precision
  const cp = bundle.conviction_precision;
  const convStr = cp?.n_high_bull && cp.n_high_bull > 0
    ? `${cp.bull_precision != null ? (cp.bull_precision * 100).toFixed(1) + "%" : "?"} on ${cp.n_high_bull} high-conviction calls ≥${(cp.threshold * 100).toFixed(0)}% — ${cp.bull_precision != null && cp.bull_precision >= 0.55 ? "✓ Go" : "✗ No-go"}`
    : "Insufficient history";

  // v2 blend
  const v2 = bundle.v2_blend?.used
    ? `P(up) ${bundle.v2_blend.v2_prob != null ? (bundle.v2_blend.v2_prob * 100).toFixed(1) : "?"}% · weight ${(bundle.v2_blend.weight_v2 * 100).toFixed(0)}% · ${bundle.v2_blend.n_headlines} headlines`
    : "not used";

  // Forecast trajectory (first 5 days)
  const trajectory = bundle.points.slice(0, 5)
    .map((p, i) => `  Day ${i + 1} (${p.target_date}): ₹${p.pred_price.toFixed(2)}  P(up)=${p.prob_up != null ? (p.prob_up * 100).toFixed(1) + "%" : "?"}`)
    .join("\n");

  // Gate warnings
  const gates = [
    bundle.trading_suspended ? "⛔ TRADING SUSPENDED — 60-day Sharpe < 0. DIAGNOSTIC ONLY." : "",
    bundle.calibration_gated ? "⚠️ CALIBRATION GATE — rolling ECE > 15%. Stated probabilities unreliable." : "",
  ].filter(Boolean).join("\n");

  const eceStr = bundle.calibration
    ? `${(bundle.calibration.ece * 100).toFixed(1)}% (${bundle.calibration.ece <= 0.05 ? "Trustworthy ✓" : bundle.calibration.ece <= 0.10 ? "Fair" : "Poorly calibrated ✗"})`
    : "N/A";

  return `ROLE
────
You are a senior portfolio manager at a tier-1 Indian equity fund (AUM > ₹50,000 Cr).
A hybrid ML model (XGBoost + LightGBM + CatBoost + GRU-128 + Ridge meta-stacker,
${bundle.walkforward?.n_windows ?? "52+"}× walk-forward windows) just finished on ${company}.
Deliver a precise, decisive trading brief. No disclaimers. I am a registered professional trader.
${gates ? `\n${gates}\n` : ""}
════════════════════════════════════════════════════════
MODEL OUTPUT  —  ${company}  |  ₹${anchor.toFixed(2)}
════════════════════════════════════════════════════════
${bundle.horizon_days}-day target (median)   : ₹${target.toFixed(2)}  (${forecastReturn >= 0 ? "+" : ""}${forecastReturn.toFixed(2)}%)
90% confidence interval  : ${ciStr}
Conformal halfwidth      : ${hwStr}
Bullish probability P(up): ${prob.toFixed(1)}%  (threshold τ* = ${(tau * 100).toFixed(1)}%)
Verdict                  : ${prob / 100 >= tau ? "BUY" : prob / 100 <= 1 - tau ? "SELL" : "HOLD"} (${prob / 100 >= tau || prob / 100 <= 1 - tau ? "outside indifference band" : "inside indifference band"})
Walk-forward accuracy    : ${acc.toFixed(1)}%  —  ${modelTier}
Hurst exponent           : ${hurstStr}
Market regime            : ${bundle.regime ?? "N/A"}${bundle.regime_detail ? ` — ${bundle.regime_detail}` : ""}
ECE (calibration)        : ${eceStr}
Brier score              : ${bundle.calibration ? bundle.calibration.brier_score.toFixed(4) + " (random baseline = 0.25)" : "N/A"}
RMSE (stacked)           : ${bundle.rmse_breakdown?.stacked?.toFixed(4) ?? "N/A"}
v2 sentiment blend       : ${v2}
Conviction precision     : ${convStr}

════════════════════════════════════════════════════════
SHAP FEATURE IMPORTANCE (top 6 drivers)
════════════════════════════════════════════════════════
${topShap || "  N/A"}

  Bullish contributors : ${bullDrivers}
  Bearish headwinds    : ${bearDrivers}

════════════════════════════════════════════════════════
FORECAST TRAJECTORY
════════════════════════════════════════════════════════
${trajectory || "  N/A"}

════════════════════════════════════════════════════════
YOUR TASK — output in STRICT markdown below, max 400 words
════════════════════════════════════════════════════════

### 🎯 Verdict: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
**Conviction:** [High / Medium / Low]
**Model Override:** [Yes — cite the reason / No — model is reliable]

### 🧠 Rationale (3 bullets max — each must cite a specific number from above)
- **Primary driver:** [top SHAP feature name + its signal direction + numeric value]
- **Confirming signal:** [second layer pointing same way]
- **Key concern:** [main bear case with exact data point]

### ⚠️ Critical Risks
- [Risk + data point — ECE reliability, regime shift, or gate warning if active]
- [External / macro risk — VIX level, FII trend, option signal if available]

### 💰 Trade Plan (calculate levels precisely)
- **Entry zone:** ₹${(anchor * 0.997).toFixed(2)} – ₹${(anchor * 1.003).toFixed(2)}
- **Stop loss:** ₹[entry minus conformal halfwidth ${hwStr}; must be below nearest support]
- **Target 1 (1.5R):** ₹[entry + 1.5 × risk]
- **Target 2 (2.5R):** ₹[entry + 2.5 × risk]
- **Position size:** [Full / Half / Quarter] — [one phrase tied to conviction + ECE]
- **Hold period:** ${bundle.horizon_days} trading days`.trim();
}

function ClaudePromptPanel({
  bundle,
  ticker,
  lastClose,
}: {
  bundle: PredictionBundle;
  ticker: string;
  lastClose: number | null;
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const prompt = open ? buildClaudePrompt(bundle, ticker, lastClose) : "";

  function handleCopy() {
    navigator.clipboard.writeText(buildClaudePrompt(bundle, ticker, lastClose)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="panel">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">🤖 Claude Expert Prompt</p>
          <p className="text-xs text-muted mt-0.5">
            Paste into claude.ai for a professional trading brief based on this model output.
          </p>
        </div>
        <button
          className="btn text-sm"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Hide prompt" : "Show prompt"}
        </button>
      </div>
      {open && (
        <div className="mt-3 space-y-2">
          <textarea
            readOnly
            className="input w-full h-64 font-mono text-xs resize-y"
            value={prompt}
          />
          <div className="flex gap-2 justify-end">
            <button className="btn btn-primary text-sm" onClick={handleCopy}>
              {copied ? "Copied!" : "Copy to clipboard"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Feature catalogue (Feature 1 … N) ─────────────────────────────────────
const FEATURE_CATALOGUE: Record<string, { n: number; group: string; desc: string }> = {
  Log_Ret:             { n: 1,  group: "Core Technical",     desc: "Today's log-return — the raw daily price move, normalized by log scale to handle large price differences." },
  Volatility_5D:       { n: 2,  group: "Core Technical",     desc: "5-day rolling standard deviation of daily returns — measures short-term price turbulence." },
  RSI_Norm:            { n: 3,  group: "Core Technical",     desc: "14-day RSI, rescaled to [-1, +1]. Values > 0 = overbought pressure; < 0 = oversold bounce potential." },
  Vol_Ratio:           { n: 4,  group: "Core Technical",     desc: "Today's volume divided by 20-day average. High ratio = unusual participation (breakout / reversal signal)." },
  MA_Div:              { n: 5,  group: "Core Technical",     desc: "Distance of price from the 5-day MA minus 20-day MA — a fast-vs-slow momentum proxy." },
  MACD_Norm:           { n: 6,  group: "Enhanced Technical", desc: "MACD line (EMA12 − EMA26), normalized. Positive = bullish momentum; negative = bearish." },
  MACD_Hist_Norm:      { n: 7,  group: "Enhanced Technical", desc: "MACD histogram (MACD − Signal line). Diverging from zero signals strengthening momentum." },
  BB_PctB:             { n: 8,  group: "Enhanced Technical", desc: "Bollinger Band %B: 0 = lower band (oversold), 1 = upper band (overbought). Measures band position." },
  ATR_Norm:            { n: 9,  group: "Enhanced Technical", desc: "14-day Average True Range, normalized. High ATR = wide daily swings; low = tight consolidation." },
  OBV_Slope:           { n: 10, group: "Enhanced Technical", desc: "20-day slope of On-Balance Volume. Rising = buyers accumulating; falling = distribution." },
  Ret_2D:              { n: 11, group: "Enhanced Technical", desc: "2-day cumulative return — captures very short-term momentum and gap behavior." },
  Ret_5D:              { n: 12, group: "Enhanced Technical", desc: "5-day (1-week) return — the primary weekly trend signal in the model." },
  Ret_10D:             { n: 13, group: "Enhanced Technical", desc: "10-day (2-week) return — medium-term momentum." },
  Ret_20D:             { n: 14, group: "Enhanced Technical", desc: "20-day (1-month) return — captures monthly trend direction." },
  CMF_20:              { n: 15, group: "New Technical",      desc: "Chaikin Money Flow over 20 days: > 0 = money flowing in (bullish); < 0 = outflow (bearish)." },
  Williams_R_Norm:     { n: 16, group: "New Technical",      desc: "Williams %R, normalized. Near -1 = oversold; near +1 = overbought." },
  RSI_Bear_Div:        { n: 17, group: "New Technical",      desc: "RSI bearish divergence flag: 1 = price made new high but RSI didn't (warning of reversal)." },
  RSI_Bull_Div:        { n: 18, group: "New Technical",      desc: "RSI bullish divergence flag: 1 = price made new low but RSI didn't (potential reversal up)." },
  Sentiment:           { n: 19, group: "Sentiment",          desc: "Primary FinBERT news sentiment score for this ticker's headlines. Positive = bullish news." },
  Multi_Sentiment:     { n: 20, group: "Sentiment",          desc: "Multi-source sentiment (RSS + Reddit + NewsAPI aggregated). More robust than single-source." },
  Sentiment_Confidence:{ n: 21, group: "Sentiment",          desc: "Model's confidence in the sentiment reading. Low confidence → the score is less reliable." },
  FII_Net_Norm:        { n: 22, group: "Institutional",      desc: "FII (foreign institutional) net buying/selling, normalized. Strong FII buying = bullish signal." },
  DII_Net_Norm:        { n: 23, group: "Institutional",      desc: "DII (domestic institutional) net flow, normalized. Often counter-cyclical to FII flows." },
  FII_5D_Avg:          { n: 24, group: "Institutional",      desc: "5-day average of FII net flow — smoothed institutional sentiment trend." },
  DII_5D_Avg:          { n: 25, group: "Institutional",      desc: "5-day average of DII net flow — smoothed domestic institutional trend." },
  VIX_Norm:            { n: 26, group: "Market Fear",        desc: "India VIX normalized. High VIX = market fear / wide uncertainty bands for the model." },
  VIX_Change:          { n: 27, group: "Market Fear",        desc: "Day-over-day VIX change. Rising VIX = increasing fear; falling = growing complacency." },
};

const GROUP_COLOR: Record<string, string> = {
  "Core Technical":     "#2dd4bf",
  "Enhanced Technical": "#a78bfa",
  "New Technical":      "#f59e0b",
  "Sentiment":          "#22c55e",
  "Institutional":      "#60a5fa",
  "Market Fear":        "#ef4444",
  "Macro":              "#fb923c",
  "Option Chain":       "#e879f9",
};

function featureMeta(name: string): { n: number; group: string; desc: string } | null {
  if (name in FEATURE_CATALOGUE) return FEATURE_CATALOGUE[name];
  // Macro columns: usd_inr_ret_1d, crude_ret_1d, …
  if (name.endsWith("_ret_1d") || name.endsWith("_ret_5d")) {
    const base = name.replace(/_ret_[15]d$/, "").replace(/_/g, "/").toUpperCase();
    return { n: 28, group: "Macro", desc: `${base} 1/5-day log-return — cross-asset macro signal feeding into the ensemble.` };
  }
  // Option chain columns: opt_pcr, opt_max_pain_dist, …
  if (name.startsWith("opt_")) {
    const key = name.slice(4).replace(/_/g, " ");
    return { n: 29, group: "Option Chain", desc: `Option chain snapshot — ${key}. Captures real-money positioning in derivatives.` };
  }
  return null;
}

// ─── Verdict banner with SHAP reasoning ─────────────────────────────────────

function VerdictBanner({
  bundle,
  lastClose,
}: {
  bundle: PredictionBundle;
  lastClose: number | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const probPct = bundle.last_directional_prob ?? 50;
  const tau = bundle.threshold_tuning?.tau_star ?? 0.55;
  const prob = probPct / 100;
  const lastPoint = bundle.points[bundle.points.length - 1];
  const target = lastPoint?.pred_price ?? null;
  const anchor = bundle.anchor_price ?? lastClose;
  const predReturn =
    target != null && anchor != null && anchor !== 0
      ? ((target - anchor) / anchor) * 100
      : null;

  let verdict: "BUY" | "SELL" | "HOLD";
  let tone: "bull" | "bear" | "muted";
  if (prob >= tau) { verdict = "BUY"; tone = "bull"; }
  else if (prob <= 1 - tau) { verdict = "SELL"; tone = "bear"; }
  else { verdict = "HOLD"; tone = "muted"; }

  const ece = bundle.calibration?.ece;
  const caliTag =
    ece == null ? null
    : ece <= 0.05 ? "Trustworthy"
    : ece <= 0.1  ? "Fair"
    : "Poorly calibrated";

  // Build SHAP-driven narrative
  const topShap = (bundle.shap_top_features ?? []).slice(0, 5);
  const bullishDrivers = topShap.filter((f) => f.importance > 0);
  const bearishDrivers = topShap.filter((f) => f.importance < 0);

  // Regime explanation
  const hurstExpl =
    bundle.hurst_exponent == null ? null
    : bundle.hurst_exponent > 0.55 ? `Hurst exponent ${bundle.hurst_exponent.toFixed(3)} signals a persistent trend — the model trusts momentum signals more.`
    : bundle.hurst_exponent < 0.45 ? `Hurst exponent ${bundle.hurst_exponent.toFixed(3)} signals mean-reversion — the model treats extremes as fade opportunities.`
    : `Hurst exponent ${bundle.hurst_exponent.toFixed(3)} is near 0.5 (random walk) — no strong directional bias from regime.`;

  const tauExpl = bundle.threshold_tuning
    ? `τ* = ${(tau * 100).toFixed(1)}% was learned from the holdout set (Youden's J, AUC = ${bundle.threshold_tuning.auc != null ? bundle.threshold_tuning.auc.toFixed(3) : "?"}).`
    : null;

  const v2Expl = bundle.v2_blend?.used
    ? `A v2 sentiment ensemble (FinBERT fine-tuned) contributed ${(bundle.v2_blend.weight_v2 * 100).toFixed(0)}% of the final probability — its read: P(up) ${bundle.v2_blend.v2_prob != null ? (bundle.v2_blend.v2_prob * 100).toFixed(1) : "?"}%.`
    : null;

  return (
    <div
      className={`panel border-l-4 ${
        tone === "bull" ? "border-l-bull" : tone === "bear" ? "border-l-bear" : "border-l-muted"
      }`}
    >
      {/* Top row */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <span
            className={`inline-flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-full text-xs font-bold tracking-wider ${
              tone === "bull" ? "bg-bull/20 text-bull" : tone === "bear" ? "bg-bear/20 text-bear" : "bg-border text-muted"
            }`}
          >
            {verdict}
          </span>
          <div>
            <p className="text-xs uppercase text-muted tracking-wide">AI verdict · {bundle.horizon_days}d horizon</p>
            <p className="text-sm mt-0.5 font-medium">
              {prob >= tau
                ? `P(up) ${probPct.toFixed(1)}% clears decision threshold τ* = ${(tau * 100).toFixed(1)}% — model says go long.`
                : prob <= 1 - tau
                ? `P(up) ${probPct.toFixed(1)}% is below bearish threshold ${((1 - tau) * 100).toFixed(1)}% — model says go short.`
                : `P(up) ${probPct.toFixed(1)}% is inside the indifference band [${((1 - tau) * 100).toFixed(1)}%, ${(tau * 100).toFixed(1)}%] — not enough edge to trade.`}
            </p>
            {caliTag && (
              <p className="text-xs text-muted mt-0.5">
                Calibration (in-sample): <span className={ece! <= 0.05 ? "text-bull" : ece! <= 0.1 ? "text-fg" : "text-bear"}>{caliTag}</span>{" "}
                (ECE {(ece! * 100).toFixed(1)}%){ece! <= 0.05 ? " — check Accuracy tab for live resolved ECE." : ece! <= 0.1 ? " — probabilities approximately right on training data." : " — treat these probabilities with caution."}
              </p>
            )}
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-xs uppercase text-muted tracking-wide">Target · {bundle.horizon_days}d</p>
          <p className="text-2xl font-semibold tabular-nums">{target != null ? `₹${target.toFixed(2)}` : "—"}</p>
          {predReturn != null && (
            <p className={`text-sm tabular-nums ${predReturn >= 0 ? "text-bull" : "text-bear"}`}>
              {predReturn >= 0 ? "+" : ""}{predReturn.toFixed(2)}% expected
            </p>
          )}
        </div>
      </div>

      {/* Expand button */}
      <button
        className="mt-3 text-xs text-accent underline"
        onClick={() => setExpanded((e) => !e)}
      >
        {expanded ? "Hide reasoning" : "Show full reasoning"}
      </button>

      {expanded && (
        <div className="mt-4 space-y-4 text-sm">
          {/* SHAP drivers */}
          {topShap.length > 0 && (
            <div>
              <p className="text-xs uppercase text-muted mb-2 font-semibold">What's driving this prediction (SHAP)</p>
              {bullishDrivers.length > 0 && (
                <div className="mb-2">
                  <p className="text-xs text-bull mb-1">Bullish contributors</p>
                  <ul className="space-y-1">
                    {bullishDrivers.map((f) => {
                      const meta = featureMeta(f.feature);
                      return (
                        <li key={f.feature} className="flex gap-2">
                          <span className="text-bull font-mono min-w-[54px]">+{f.importance.toFixed(3)}</span>
                          <span>
                            <strong>{meta ? `Feature ${meta.n} (${f.feature})` : f.feature}</strong>
                            {meta ? ` — ${meta.desc}` : ""}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              {bearishDrivers.length > 0 && (
                <div>
                  <p className="text-xs text-bear mb-1">Bearish headwinds</p>
                  <ul className="space-y-1">
                    {bearishDrivers.map((f) => {
                      const meta = featureMeta(f.feature);
                      return (
                        <li key={f.feature} className="flex gap-2">
                          <span className="text-bear font-mono min-w-[54px]">{f.importance.toFixed(3)}</span>
                          <span>
                            <strong>{meta ? `Feature ${meta.n} (${f.feature})` : f.feature}</strong>
                            {meta ? ` — ${meta.desc}` : ""}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Regime */}
          {(hurstExpl || bundle.regime) && (
            <div className="border-t border-border pt-3">
              <p className="text-xs uppercase text-muted mb-1 font-semibold">Market regime</p>
              {bundle.regime && <p><strong>{bundle.regime}</strong>{bundle.regime_detail ? ` — ${bundle.regime_detail}` : ""}</p>}
              {hurstExpl && <p className="text-muted mt-0.5">{hurstExpl}</p>}
            </div>
          )}

          {/* Threshold */}
          {tauExpl && (
            <div className="border-t border-border pt-3">
              <p className="text-xs uppercase text-muted mb-1 font-semibold">Decision threshold τ*</p>
              <p className="text-muted">{tauExpl}</p>
              <p className="text-muted mt-0.5">
                The indifference band [{((1 - tau) * 100).toFixed(1)}%, {(tau * 100).toFixed(1)}%] means the model abstains when it doesn't have enough edge — a HOLD signal reduces false trades.
              </p>
            </div>
          )}

          {/* v2 blend */}
          {v2Expl && (
            <div className="border-t border-border pt-3">
              <p className="text-xs uppercase text-muted mb-1 font-semibold">Sentiment v2 ensemble</p>
              <p className="text-muted">{v2Expl}</p>
            </div>
          )}

          {/* Calibration detail */}
          {bundle.calibration && (
            <div className="border-t border-border pt-3">
              <p className="text-xs uppercase text-muted mb-1 font-semibold">Why trust these probabilities?</p>
              <p className="text-muted">
                ECE (Expected Calibration Error) measures how far predicted probabilities are from actual outcomes.
                ECE = {(bundle.calibration.ece * 100).toFixed(1)}% means if the model says 60%, the stock actually went up ~{(60 - bundle.calibration.ece * 100).toFixed(0)}%–{(60 + bundle.calibration.ece * 100).toFixed(0)}% of the time.
                Brier score {bundle.calibration.brier_score.toFixed(4)} (lower = better; random = 0.25).
                Tested on {bundle.calibration.n_samples} holdout samples.
              </p>
            </div>
          )}

          {/* Walk-forward */}
          {bundle.walkforward && bundle.walkforward.accuracy != null && (
            <div className="border-t border-border pt-3">
              <p className="text-xs uppercase text-muted mb-1 font-semibold">Walk-forward validation</p>
              <p className="text-muted">
                The model was tested across {bundle.walkforward.n_windows} expanding windows of history.
                Average directional accuracy: {((bundle.walkforward.accuracy ?? 0) * 100).toFixed(1)}%
                {bundle.walkforward.std != null ? ` ± ${(bundle.walkforward.std * 100).toFixed(1)}%` : ""}.
                This tests whether the model would have worked if trained on past data and applied to the next period.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── SHAP panel with feature numbers + glossary ─────────────────────────────

function ShapPanel({ features, method }: { features: { feature: string; importance: number }[]; method?: string | null }) {
  const [showGlossary, setShowGlossary] = useState(false);
  const top = features.slice(0, 12);

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold">
          Feature importance ({method ?? "SHAP"})
        </p>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted">top {top.length} of ~27–35 features</span>
          <button
            className="text-xs text-accent underline"
            onClick={() => setShowGlossary((g) => !g)}
          >
            {showGlossary ? "Hide glossary" : "Feature glossary"}
          </button>
        </div>
      </div>

      {/* Numbered SHAP bars */}
      <HorizontalBarChart
        bars={top.map((f) => {
          const meta = featureMeta(f.feature);
          return {
            label: meta ? `F${meta.n} · ${f.feature}` : f.feature,
            value: f.importance,
          };
        })}
      />

      {/* Inline legend: which group each feature belongs to */}
      <div className="flex flex-wrap gap-2 text-xs">
        {Array.from(new Set(top.map((f) => featureMeta(f.feature)?.group ?? "Other"))).map((g) => (
          <span key={g} className="flex items-center gap-1">
            <span
              className="inline-block w-2 h-2 rounded-sm"
              style={{ background: GROUP_COLOR[g] ?? "#8b98a5" }}
            />
            {g}
          </span>
        ))}
      </div>

      {/* Full glossary */}
      {showGlossary && (
        <div className="border-t border-border pt-3 space-y-1">
          <p className="text-xs uppercase text-muted mb-2">All model features (F1 – F27+)</p>
          <div className="grid gap-2 sm:grid-cols-2">
            {Object.entries(FEATURE_CATALOGUE).map(([key, meta]) => (
              <div key={key} className="text-xs flex gap-2">
                <span
                  className="font-mono font-bold min-w-[28px]"
                  style={{ color: GROUP_COLOR[meta.group] ?? "#8b98a5" }}
                >
                  F{meta.n}
                </span>
                <span>
                  <strong className="text-fg">{key}</strong>
                  <span className="text-muted"> — {meta.desc}</span>
                </span>
              </div>
            ))}
            <div className="text-xs flex gap-2 sm:col-span-2">
              <span className="font-mono font-bold min-w-[28px]" style={{ color: GROUP_COLOR["Macro"] }}>F28+</span>
              <span>
                <strong className="text-fg">Macro factors</strong>
                <span className="text-muted"> — USD/INR, Crude Oil, US 10Y yield, Gold, S&P 500, US VIX (1d + 5d log-returns). Added only when live data is available.</span>
              </span>
            </div>
            <div className="text-xs flex gap-2 sm:col-span-2">
              <span className="font-mono font-bold min-w-[28px]" style={{ color: GROUP_COLOR["Option Chain"] }}>F30+</span>
              <span>
                <strong className="text-fg">Option chain</strong>
                <span className="text-muted"> — PCR, max-pain distance, ATM IV, IV skew, OI concentration, call/put walls. Added from NSE live snapshot when available.</span>
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ReliabilityChart({
  bins,
}: {
  bins: { predicted: number; actual: number; count: number }[];
}) {
  if (bins.length === 0) return null;
  const w = 400;
  const h = 200;
  const pad = 24;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-sm mt-3" preserveAspectRatio="none">
      <rect x={pad} y={pad} width={plotW} height={plotH} fill="none" stroke="#1f2a36" />
      <line
        x1={pad}
        y1={pad + plotH}
        x2={pad + plotW}
        y2={pad}
        stroke="#8b98a5"
        strokeDasharray="3 3"
      />
      {bins.map((b, i) => {
        const x = pad + b.predicted * plotW;
        const y = pad + plotH - b.actual * plotH;
        const r = Math.max(2, Math.min(8, Math.sqrt(b.count)));
        return <circle key={i} cx={x} cy={y} r={r} fill="#2dd4bf" opacity={0.85} />;
      })}
      <text x={pad} y={h - 6} fill="#8b98a5" fontSize={10}>
        Predicted
      </text>
      <text x={2} y={pad + 4} fill="#8b98a5" fontSize={10}>
        Actual
      </text>
    </svg>
  );
}
