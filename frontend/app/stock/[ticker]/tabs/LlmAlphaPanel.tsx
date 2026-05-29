"use client";

import { useState } from "react";
import Stat from "@/components/Stat";
import { apiGet, apiPost } from "@/lib/api-client";
import { toneFor, formatRatio } from "@/lib/format";

interface HeadlineScore {
  headline: string;
  novelty: number;
  materiality: number;
  sentiment: number;
  alpha_signal: number;
}

interface LlmAlphaResult {
  ticker: string;
  available: boolean;
  needs_manual_input: boolean;
  manual_prompt: string;
  combined_signal: number;
  mean_novelty: number;
  mean_materiality: number;
  mean_sentiment: number;
  top_headline: string;
  top_alpha: number;
  headlines_scored: number;
  used_fallback: boolean;
  scores: HeadlineScore[];
}

/**
 * Lazy-fetch LLM novelty×materiality alpha panel (P3-4).
 * When ANTHROPIC_API_KEY is set on the backend, it auto-scores headlines.
 * When not set, it returns a manual_prompt for pasting into claude.ai.
 */
export default function LlmAlphaPanel({ ticker }: { ticker: string }) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<LlmAlphaResult | null>(null);
  const [manualPaste, setManualPaste] = useState("");
  const [pasteError, setPasteError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function run() {
    setLoading(true);
    setErr(null);
    setData(null);
    try {
      const r = await apiGet<LlmAlphaResult>(
        `/api/v1/stocks/${ticker}/sentiment/llm-alpha?max_headlines=10`,
      );
      setData(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function submitManual() {
    if (!manualPaste.trim() || !data) return;
    setSubmitting(true);
    setPasteError(null);
    try {
      const r = await apiPost<LlmAlphaResult>(
        `/api/v1/stocks/${ticker}/sentiment/llm-alpha/manual`,
        { manual_response: manualPaste },
      );
      setData(r);
    } catch (e) {
      setPasteError(e instanceof Error ? e.message : "Failed to parse response");
    } finally {
      setSubmitting(false);
    }
  }

  const signalTone = data ? toneFor(data.combined_signal * 100) : "muted";

  return (
    <section className="panel">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">🧠 LLM Alpha (Novelty × Materiality)</p>
          <p className="text-xs text-muted">
            Claude Haiku scores each headline on novelty, materiality, and sentiment direction.
            Combined alpha = sentiment × novelty × materiality.
          </p>
        </div>
        {!data && !loading && (
          <button className="btn text-sm" onClick={run}>
            Run LLM Alpha
          </button>
        )}
        {data && (
          <button className="btn text-sm" onClick={run}>
            Refresh
          </button>
        )}
      </div>

      {loading && (
        <p className="text-xs text-muted mt-3 animate-pulse">Scoring headlines…</p>
      )}
      {err && <p className="text-xs text-bear mt-3">{err}</p>}

      {data && !data.needs_manual_input && (
        <div className="mt-4 space-y-4">
          <div className="grid gap-4 sm:grid-cols-4 text-sm">
            <Stat
              label="Alpha signal"
              value={`${data.combined_signal >= 0 ? "+" : ""}${data.combined_signal.toFixed(3)}`}
              tone={signalTone}
              hint="sentiment × novelty × materiality"
            />
            <Stat
              label="Mean novelty"
              value={formatRatio(data.mean_novelty, 2)}
              hint="0=stale, 1=breaking"
            />
            <Stat
              label="Mean materiality"
              value={formatRatio(data.mean_materiality, 2)}
              hint="0=noise, 1=fundamental impact"
            />
            <Stat
              label="Headlines scored"
              value={String(data.headlines_scored)}
              hint={data.used_fallback ? "keyword fallback" : "LLM scored"}
            />
          </div>

          {data.top_headline && (
            <div className="text-xs text-muted border-t border-border pt-3">
              <p className="uppercase text-[10px] tracking-wide mb-1">Top alpha headline</p>
              <p className="text-fg">{data.top_headline}</p>
              <p className="mt-0.5">
                Alpha:{" "}
                <span className={data.top_alpha >= 0 ? "text-bull" : "text-bear"}>
                  {data.top_alpha >= 0 ? "+" : ""}
                  {data.top_alpha.toFixed(3)}
                </span>
              </p>
            </div>
          )}

          {data.scores.length > 0 && (
            <details className="border-t border-border pt-3">
              <summary className="cursor-pointer text-xs text-accent">
                Per-headline breakdown ({data.scores.length} headlines)
              </summary>
              <div className="mt-2 space-y-2">
                {data.scores.map((s, i) => (
                  <div key={i} className="text-xs flex gap-3 items-start">
                    <span
                      className={`font-mono font-bold min-w-[52px] ${
                        s.alpha_signal >= 0 ? "text-bull" : "text-bear"
                      }`}
                    >
                      {s.alpha_signal >= 0 ? "+" : ""}
                      {s.alpha_signal.toFixed(3)}
                    </span>
                    <span className="text-muted flex-1">
                      <span className="text-fg">{s.headline}</span>
                      <span className="ml-2 opacity-60">
                        N:{s.novelty.toFixed(2)} M:{s.materiality.toFixed(2)} S:{s.sentiment >= 0 ? "+" : ""}
                        {s.sentiment.toFixed(2)}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      {data?.needs_manual_input && (
        <div className="mt-4 space-y-3">
          <div className="panel border border-border bg-surface/50">
            <p className="text-xs font-semibold mb-1">
              ANTHROPIC_API_KEY not set — manual mode
            </p>
            <p className="text-xs text-muted mb-2">
              Copy the prompt below, paste it into{" "}
              <a
                href="https://claude.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent underline"
              >
                claude.ai
              </a>
              , then paste Claude&apos;s JSON response back here.
            </p>
            <textarea
              readOnly
              className="input w-full h-40 font-mono text-xs resize-y"
              value={data.manual_prompt}
            />
            <button
              className="btn text-xs mt-2"
              onClick={() => navigator.clipboard.writeText(data.manual_prompt)}
            >
              Copy prompt
            </button>
          </div>
          <div>
            <p className="text-xs text-muted mb-1">Paste Claude&apos;s JSON response:</p>
            <textarea
              className="input w-full h-28 font-mono text-xs resize-y"
              placeholder='[{"headline": "...", "novelty": 0.8, "materiality": 0.7, "sentiment": 0.6}]'
              value={manualPaste}
              onChange={(e) => setManualPaste(e.target.value)}
            />
            {pasteError && <p className="text-xs text-bear mt-1">{pasteError}</p>}
            <button
              className="btn btn-primary text-sm mt-2"
              disabled={!manualPaste.trim() || submitting}
              onClick={submitManual}
            >
              {submitting ? "Parsing…" : "Submit response"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
