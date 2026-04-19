"use client";

import { useState } from "react";
import type { PredictionBundle, PredictionJobAccepted, StockHistory } from "@/lib/types";
import Stat from "@/components/Stat";
import PriceChart from "../charts/PriceChart";
import HorizontalBarChart from "../charts/HorizontalBarChart";
import { apiPost, waitForJob } from "@/lib/api-client";
import { formatINR, formatPercent, formatRatio, toneFor } from "@/lib/format";

interface Props {
  ticker: string;
  ohlcv: StockHistory | null;
}

/**
 * Tab 1 — chart, run-prediction button, forecast overlay, calibration +
 * threshold + SHAP + v2-blend diagnostics. Mirrors the first few Streamlit
 * tabs condensed into a single scrollable panel.
 */
export default function OverviewTab({ ticker, ohlcv }: Props) {
  const [horizon, setHorizon] = useState(10);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [bundle, setBundle] = useState<PredictionBundle | null>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setBundle(null);
    setPhase("queued");
    try {
      const accepted = await apiPost<PredictionJobAccepted>(
        `/api/v1/stocks/${ticker}/predict`,
        { horizon_days: horizon, use_v2_blend: null },
      );
      setPhase("running");
      const result = await waitForJob<PredictionBundle>(accepted.job_id, {
        onPhase: (p) => setPhase(p),
      });
      setBundle(result);
      setPhase("done");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setPhase(null);
    } finally {
      setBusy(false);
    }
  }

  const lastClose = ohlcv?.bars.length ? ohlcv.bars[ohlcv.bars.length - 1].close : null;
  const firstClose = ohlcv?.bars.length ? ohlcv.bars[0].close : null;
  const windowChange = lastClose != null && firstClose != null && firstClose !== 0
    ? ((lastClose - firstClose) / firstClose) * 100
    : null;

  const lastPoint = bundle?.points?.[bundle.points.length - 1] ?? null;
  const predReturn = bundle && bundle.anchor_price && lastPoint
    ? ((lastPoint.pred_price - bundle.anchor_price) / bundle.anchor_price) * 100
    : null;

  return (
    <div className="space-y-6">
      <section className="panel">
        {ohlcv && ohlcv.bars.length > 0 ? (
          <>
            <div className="flex items-baseline justify-between mb-2">
              <div>
                <p className="text-xs uppercase text-muted">Last close</p>
                <p className="text-2xl font-semibold">{formatINR(lastClose)}</p>
              </div>
              {windowChange != null && (
                <p
                  className={`text-sm ${toneFor(windowChange) === "bull" ? "text-bull" : toneFor(windowChange) === "bear" ? "text-bear" : "text-muted"}`}
                >
                  {formatPercent(windowChange)} · {ohlcv.bars.length} bars · {ohlcv.start} → {ohlcv.end}
                </p>
              )}
            </div>
            <PriceChart
              bars={ohlcv.bars}
              predictions={bundle?.points}
              anchorPrice={bundle?.anchor_price}
              height={420}
            />
          </>
        ) : (
          <p className="text-sm text-muted">OHLCV unavailable — upstream may be rate-limited.</p>
        )}
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Forecast</h2>
          <span className="text-xs text-muted">
            {bundle?.model_version && `Model: ${bundle.model_version}`}
          </span>
        </div>
        <form onSubmit={run} className="panel flex items-end gap-3">
          <label className="flex-1">
            <span className="block text-xs uppercase text-muted mb-1">Horizon (days)</span>
            <input
              type="number"
              min={1}
              max={60}
              className="input"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
            />
          </label>
          <button type="submit" disabled={busy} className="btn btn-primary">
            {busy ? (phase ?? "…") : "Run prediction"}
          </button>
        </form>
        {err && <p className="text-sm text-bear">{err}</p>}

        {bundle && (
          <>
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
                    ? bundle.last_directional_prob >= 50 ? "bull" : "bear"
                    : "muted"
                }
                hint={bundle.threshold_tuning
                  ? `τ* = ${(bundle.threshold_tuning.tau_star * 100).toFixed(1)}%`
                  : undefined}
              />
              <Stat
                label={`${Math.round(bundle.confidence_level * 100)}% band (day 1)`}
                value={
                  bundle.points[0]?.ci_low != null && bundle.points[0]?.ci_high != null
                    ? `${formatINR(bundle.points[0].ci_low)} – ${formatINR(bundle.points[0].ci_high)}`
                    : "—"
                }
                hint={bundle.conformal_halfwidth != null
                  ? `conformal ± ${(bundle.conformal_halfwidth * 100).toFixed(2)}%`
                  : undefined}
              />
            </div>

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
                    <p className="font-medium">{formatPercent((bundle.walkforward.accuracy ?? 0) * 100, 1)}</p>
                    <p className="text-xs text-muted">
                      {bundle.walkforward.n_windows} windows
                      {bundle.walkforward.std != null
                        ? ` · σ ${(bundle.walkforward.std * 100).toFixed(1)}%`
                        : ""}
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
                    hint={bundle.v2_blend.stacker_available ? "stacker OK" : "fallback weighted-avg"}
                  />
                </div>
              </div>
            )}

            {bundle.shap_top_features && bundle.shap_top_features.length > 0 && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-semibold">Feature importance ({bundle.shap_method ?? "SHAP"})</p>
                  <span className="text-xs text-muted">top {bundle.shap_top_features.length}</span>
                </div>
                <HorizontalBarChart
                  bars={bundle.shap_top_features.slice(0, 12).map((f) => ({
                    label: f.feature,
                    value: f.importance,
                  }))}
                />
              </div>
            )}

            {bundle.rmse_breakdown && (
              <div className="panel">
                <p className="text-sm font-semibold mb-2">Base-learner RMSE (test fold)</p>
                <div className="grid gap-4 sm:grid-cols-5 text-sm">
                  {([
                    ["XGB", bundle.rmse_breakdown.xgb],
                    ["LightGBM", bundle.rmse_breakdown.lgbm],
                    ["CatBoost", bundle.rmse_breakdown.catboost],
                    ["GRU", bundle.rmse_breakdown.rnn],
                    ["Stacked", bundle.rmse_breakdown.stacked],
                  ] as const).map(([k, v]) => (
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
                  <Stat label="n_samples" value={String(bundle.calibration.n_samples)} />
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
                        <td className="py-1 text-right">{(p.prob_up * 100).toFixed(1)}%</td>
                        <td className="py-1 text-right">{p.ci_low != null ? formatINR(p.ci_low) : "—"}</td>
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

function ReliabilityChart({ bins }: { bins: { predicted: number; actual: number; count: number }[] }) {
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
      <text x={pad} y={h - 6} fill="#8b98a5" fontSize={10}>Predicted</text>
      <text x={2} y={pad + 4} fill="#8b98a5" fontSize={10}>Actual</text>
    </svg>
  );
}
