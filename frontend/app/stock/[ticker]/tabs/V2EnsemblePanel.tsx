"use client";

import { useState } from "react";
import Stat from "@/components/Stat";
import { apiGet } from "@/lib/api-client";
import HorizontalBarChart from "../charts/HorizontalBarChart";

interface V2Breakdown {
  logreg: number;
  random_forest: number;
  xgboost: number;
  lightgbm: number;
}

interface V2Prediction {
  ticker: string;
  made_at: string;
  n_headlines: number;
  top_category: string;
  weighted_sentiment: number;
  category_counts: Record<string, number>;
  prob_up: number;
  model_breakdown: V2Breakdown;
  stacker_available: boolean;
  model_version: string;
}

/**
 * Client-side panel that lazy-fetches the 4-model HuggingFace sentiment
 * ensemble. The v2 endpoint 503s cleanly when HF_TOKEN is unset — we show
 * that instead of a stack trace.
 */
export default function V2EnsemblePanel({ ticker }: { ticker: string }) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<V2Prediction | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const r = await apiGet<V2Prediction>(
        `/api/v1/stocks/${ticker}/sentiment/v2?max_headlines=30`,
      );
      setData(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">🤖 Sentiment Ensemble (v2)</p>
          <p className="text-xs text-muted">
            4-learner stack on HuggingFace — LogReg + RF + XGB + LGBM
          </p>
        </div>
        <button disabled={loading} onClick={run} className="btn btn-primary">
          {loading ? "Running…" : data ? "Re-run" : "Run v2 ensemble"}
        </button>
      </div>

      {err && <p className="text-sm text-bear mt-3">{err}</p>}

      {data && (
        <div className="mt-4 space-y-4">
          <div className="grid gap-4 sm:grid-cols-4">
            <Stat
              label="Stacked P(up)"
              value={`${(data.prob_up * 100).toFixed(1)}%`}
              tone={data.prob_up >= 0.5 ? "bull" : "bear"}
              hint={data.stacker_available ? "stacker OK" : "weighted-avg fallback"}
            />
            <Stat
              label="Weighted sentiment"
              value={data.weighted_sentiment.toFixed(3)}
              tone={
                data.weighted_sentiment > 0.1
                  ? "bull"
                  : data.weighted_sentiment < -0.1
                  ? "bear"
                  : "muted"
              }
            />
            <Stat label="Headlines" value={String(data.n_headlines)} />
            <Stat label="Top category" value={data.top_category} />
          </div>

          <div>
            <p className="text-xs uppercase text-muted mb-2">Base learner P(up)</p>
            <HorizontalBarChart
              signedColors={false}
              bars={[
                { label: "LogReg", value: data.model_breakdown.logreg },
                { label: "Random Forest", value: data.model_breakdown.random_forest },
                { label: "XGBoost", value: data.model_breakdown.xgboost },
                { label: "LightGBM", value: data.model_breakdown.lightgbm },
              ]}
              height={140}
            />
          </div>
        </div>
      )}

      {!data && !err && (
        <p className="text-xs text-muted mt-3">
          Gated behind <code>HF_TOKEN</code>. Click run to download the ensemble on
          first use (~340 MB, cached locally).
        </p>
      )}
    </section>
  );
}
