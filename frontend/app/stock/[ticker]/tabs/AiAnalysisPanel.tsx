"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api-client";
import type { PredictionBundle, TechnicalSnapshot } from "@/lib/types";

interface AiAnalysisRequest {
  current_price: number;
  forecast_target: number;
  forecast_days: number;
  model_accuracy: number;
  directional_prob: number;
  technicals: { rsi: number; macd_histogram: number; volatility_5d: number; volatility_20d: number; price_vs_ma20: number };
  fii_net_cr: number;
  dii_net_cr: number;
  vix: number | null;
  forward_pe: number | null;
  price_book: number | null;
  peg: number | null;
  patterns: string[];
  sentiment_label: string;
  sentiment_pos: number;
  sentiment_neg: number;
}

interface AiAnalysisResponse {
  analysis: string;
  source: string;
  claude_prompt: string;
}

interface Props {
  ticker: string;
  bundle: PredictionBundle;
  technicals: TechnicalSnapshot | null;
}

const SOURCE_LABEL: Record<string, string> = {
  deepseek: "✨ DeepSeek V3",
  gemini: "✨ Gemini 2.5 Flash",
  template: "📝 Template mode",
  claude_prompt: "🤖 Claude (manual)",
};

export default function AiAnalysisPanel({ ticker, bundle, technicals }: Props) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<AiAnalysisResponse | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);
  const [pasted, setPasted] = useState("");
  const [applied, setApplied] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const lastPoint = bundle.points[bundle.points.length - 1];
  const anchor = bundle.anchor_price ?? 0;
  const target = lastPoint?.pred_price ?? anchor;

  async function run() {
    setLoading(true);
    setErr(null);
    setResult(null);
    setApplied(null);
    try {
      const body: AiAnalysisRequest = {
        current_price: anchor,
        forecast_target: target,
        forecast_days: bundle.horizon_days,
        model_accuracy: bundle.walkforward?.accuracy != null ? bundle.walkforward.accuracy * 100 : 50,
        directional_prob: bundle.last_directional_prob ?? 50,
        technicals: {
          rsi: technicals?.rsi_14 ?? 50,
          macd_histogram: technicals?.macd_histogram ?? 0,
          volatility_5d: technicals?.volatility_20d ?? 0.01,
          volatility_20d: technicals?.volatility_20d ?? 0.01,
          price_vs_ma20: technicals?.price_vs_ma20 ?? 0,
        },
        fii_net_cr: 0,
        dii_net_cr: 0,
        vix: null,
        forward_pe: null,
        price_book: null,
        peg: null,
        patterns: [],
        sentiment_label: "neutral",
        sentiment_pos: 0,
        sentiment_neg: 0,
      };
      const r = await apiPost<AiAnalysisResponse>(`/api/v1/stocks/${ticker}/ai-analysis`, body);
      setResult(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function applyPaste() {
    if (pasted.trim()) {
      setApplied(pasted.trim());
      setShowPrompt(false);
    }
  }

  function copyPrompt() {
    if (result?.claude_prompt) {
      navigator.clipboard.writeText(result.claude_prompt).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  }

  const displayAnalysis = applied ?? result?.analysis;
  const isTemplate = result?.source === "template" && !applied;

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">🤖 AI Expert Analysis</p>
          <p className="text-xs text-muted">
            DeepSeek V3 → Gemini 2.5 Flash → template fallback
          </p>
        </div>
        <div className="flex gap-2">
          {result && (
            <span className="text-xs text-muted self-center">
              {SOURCE_LABEL[result.source] ?? result.source}
            </span>
          )}
          <button className="btn text-sm" onClick={run} disabled={loading}>
            {loading ? "Analyzing…" : result ? "Re-run" : "Run AI Analysis"}
          </button>
        </div>
      </div>

      {err && <p className="text-xs text-bear">{err}</p>}

      {displayAnalysis && (
        <div className="prose prose-invert prose-sm max-w-none border-t border-border pt-3">
          {displayAnalysis.split("\n").map((line, i) => {
            if (line.startsWith("### "))
              return <p key={i} className="font-semibold text-fg mt-3 mb-1">{line.replace("### ", "")}</p>;
            if (line.startsWith("**") && line.endsWith("**"))
              return <p key={i} className="font-medium">{line.replace(/\*\*/g, "")}</p>;
            if (line.startsWith("- "))
              return <p key={i} className="text-muted pl-3">• {line.slice(2)}</p>;
            if (line.trim() === "") return <div key={i} className="h-1" />;
            return <p key={i} className="text-sm">{line}</p>;
          })}
        </div>
      )}

      {/* When template mode: offer Claude manual alternative */}
      {result && (isTemplate || result.source === "template") && (
        <div className="border-t border-border pt-3 space-y-2">
          <p className="text-xs text-muted">
            DeepSeek and Gemini are not configured. Get a richer analysis by pasting the prompt below into{" "}
            <a href="https://claude.ai" target="_blank" rel="noopener noreferrer" className="text-accent underline">
              claude.ai
            </a>{" "}
            (free), then paste Claude&apos;s response here.
          </p>
          <button className="btn text-xs" onClick={() => setShowPrompt((p) => !p)}>
            {showPrompt ? "Hide prompt" : "Show Claude prompt"}
          </button>
          {showPrompt && result.claude_prompt && (
            <div className="space-y-2">
              <textarea readOnly className="input w-full h-48 font-mono text-xs resize-y" value={result.claude_prompt} />
              <button className="btn text-xs" onClick={copyPrompt}>
                {copied ? "Copied!" : "Copy prompt"}
              </button>
              <textarea
                className="input w-full h-32 text-xs resize-y"
                placeholder="Paste Claude's response here…"
                value={pasted}
                onChange={(e) => setPasted(e.target.value)}
              />
              <button className="btn btn-primary text-xs" onClick={applyPaste} disabled={!pasted.trim()}>
                Apply Claude analysis
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
