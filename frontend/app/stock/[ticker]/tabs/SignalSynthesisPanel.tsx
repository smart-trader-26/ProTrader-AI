"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api-client";
import type { PredictionBundle, TechnicalSnapshot } from "@/lib/types";
import { FII_DII_STORAGE_KEY } from "./FiiDiiTab";

/** Sum of the last 5 days' FII/DII net (₹ Cr) from the collected FII/DII data. */
function fiiDii5dSum(): { fii: number; dii: number } {
  try {
    const raw = localStorage.getItem(FII_DII_STORAGE_KEY);
    if (!raw) return { fii: 0, dii: 0 };
    const rows = JSON.parse(raw) as { fii_net: number | null; dii_net: number | null }[];
    const last5 = rows.slice(-5);
    return {
      fii: last5.reduce((s, r) => s + (r.fii_net ?? 0), 0),
      dii: last5.reduce((s, r) => s + (r.dii_net ?? 0), 0),
    };
  } catch {
    return { fii: 0, dii: 0 };
  }
}

interface SignalSynthesisRequest {
  current_price: number;
  forecast_return_pct: number;
  forecast_days: number;
  directional_prob: number;
  model_accuracy: number;
  hurst: number | null;
  regime: string;
  rsi: number;
  macd_hist: number;
  volatility_20d: number;
  fii_net_5d_cr: number;
  dii_net_5d_cr: number;
  vix: number;
  llm_sentiment_signal: number;
  llm_mean_sentiment: number;
  llm_mean_materiality: number;
  llm_top_headline: string;
  multi_sentiment_score: number;
  options_pcr_signal: number;
  options_iv_skew_signal: number;
  options_combined: number;
  shap_top_features: string[];
}

interface SignalSynthesisResponse {
  prompt: string;
}

interface Props {
  ticker: string;
  bundle: PredictionBundle;
  technicals: TechnicalSnapshot | null;
}

export default function SignalSynthesisPanel({ ticker, bundle, technicals }: Props) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [prompt, setPrompt] = useState<string | null>(null);
  const [pasted, setPasted] = useState("");
  const [applied, setApplied] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const anchor = bundle.anchor_price ?? 0;
  const lastPoint = bundle.points[bundle.points.length - 1];
  const target = lastPoint?.pred_price ?? anchor;
  const forecastReturn = anchor ? ((target - anchor) / anchor) * 100 : 0;
  const accuracy = bundle.walkforward?.accuracy != null ? bundle.walkforward.accuracy * 100 : 50;
  const shapFeatures = (bundle.shap_top_features ?? []).slice(0, 5).map((f) => f.feature);

  async function run() {
    setLoading(true);
    setErr(null);
    setPrompt(null);
    setApplied(null);
    try {
      const fiiDii = fiiDii5dSum();
      const body: SignalSynthesisRequest = {
        current_price: anchor,
        forecast_return_pct: forecastReturn,
        forecast_days: bundle.horizon_days,
        directional_prob: bundle.last_directional_prob ?? 50,
        model_accuracy: accuracy,
        hurst: bundle.hurst_exponent ?? null,
        regime: bundle.regime ?? "normal",
        rsi: technicals?.rsi_14 ?? 50,
        macd_hist: technicals?.macd_histogram ?? 0,
        volatility_20d: technicals?.volatility_20d ?? 0.01,
        fii_net_5d_cr: fiiDii.fii,
        dii_net_5d_cr: fiiDii.dii,
        vix: 15,
        llm_sentiment_signal: 0,
        llm_mean_sentiment: 0,
        llm_mean_materiality: 0,
        llm_top_headline: "N/A",
        multi_sentiment_score: 0,
        options_pcr_signal: 0,
        options_iv_skew_signal: 0,
        options_combined: 0,
        shap_top_features: shapFeatures,
      };
      const r = await apiPost<SignalSynthesisResponse>(`/api/v1/stocks/${ticker}/signal-synthesis`, body);
      setPrompt(r.prompt);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function copyPrompt() {
    if (prompt) {
      navigator.clipboard.writeText(prompt).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  }

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">🧬 Full-Stack Signal Synthesis</p>
          <p className="text-xs text-muted">
            All 5 signal layers (ML · Technicals · FII/DII · Options · LLM sentiment) synthesised
            into one claude.ai prompt for a unified trade brief.
          </p>
        </div>
        <button className="btn text-sm" onClick={run} disabled={loading}>
          {loading ? "Building…" : prompt ? "Rebuild" : "Build prompt"}
        </button>
      </div>

      {err && <p className="text-xs text-bear">{err}</p>}

      {prompt && !applied && (
        <div className="space-y-2 border-t border-border pt-3">
          <p className="text-xs text-muted">
            Copy the prompt below → paste into{" "}
            <a href="https://claude.ai" target="_blank" rel="noopener noreferrer" className="text-accent underline">
              claude.ai
            </a>{" "}
            → paste Claude&apos;s response back here.
          </p>
          <textarea readOnly className="input w-full h-56 font-mono text-xs resize-y" value={prompt} />
          <button className="btn text-xs" onClick={copyPrompt}>
            {copied ? "Copied!" : "Copy prompt"}
          </button>
          <textarea
            className="input w-full h-40 text-xs resize-y"
            placeholder={"### Signal Alignment Score: 4/5 signals aligned\n..."}
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
          />
          <button
            className="btn btn-primary text-sm"
            disabled={!pasted.trim()}
            onClick={() => { setApplied(pasted.trim()); }}
          >
            Apply synthesis
          </button>
        </div>
      )}

      {applied && (
        <div className="border-t border-border pt-3 space-y-2">
          <div className="prose prose-invert prose-sm max-w-none">
            {applied.split("\n").map((line, i) => {
              if (line.startsWith("### "))
                return <p key={i} className="font-semibold text-fg mt-3 mb-1">{line.replace("### ", "")}</p>;
              if (line.startsWith("- "))
                return <p key={i} className="text-muted pl-3">• {line.slice(2)}</p>;
              if (line.trim() === "") return <div key={i} className="h-1" />;
              return <p key={i} className="text-sm">{line}</p>;
            })}
          </div>
          <button className="btn text-xs" onClick={() => { setApplied(null); setPasted(""); }}>
            Clear &amp; re-enter
          </button>
        </div>
      )}
    </div>
  );
}
