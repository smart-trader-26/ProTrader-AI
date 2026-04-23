"use client";

import type { PredictionPoint, StockBar, TestSeriesPoint } from "@/lib/types";
import ZoomableChart from "@/components/ZoomableChart";

interface Props {
  bars: StockBar[];
  testSeries: TestSeriesPoint[];
  forecastPoints: PredictionPoint[];
  anchorPrice: number | null;
  height?: number;
}

/**
 * Streamlit-parity "Model Accuracy: Actual vs Predicted Prices" chart.
 * Shows full-history actuals, dotted model-predicted overlay across the test
 * fold, and a green dot-line for the forward forecast.
 */
export default function ModelAccuracyChart({
  bars,
  testSeries,
  forecastPoints,
  anchorPrice,
  height = 360,
}: Props) {
  if (!bars.length) {
    return <p className="text-sm text-muted">No price history.</p>;
  }

  const tsAt = (s: string) => new Date(s).getTime();
  const startTs = tsAt(bars[0].ts);
  const lastBarTs = tsAt(bars[bars.length - 1].ts);
  const lastForecast = forecastPoints.length
    ? tsAt(forecastPoints[forecastPoints.length - 1].target_date)
    : lastBarTs;
  const endTs = Math.max(lastBarTs, lastForecast);
  const tsRange = endTs - startTs || 1;

  const allPrices: number[] = [
    ...bars.map((b) => b.close),
    ...testSeries.filter((p) => p.predicted_price != null).map((p) => p.predicted_price as number),
    ...forecastPoints.map((p) => p.pred_price),
  ];
  if (anchorPrice != null) allPrices.push(anchorPrice);
  const yMin = Math.min(...allPrices);
  const yMax = Math.max(...allPrices);
  const yPad = (yMax - yMin) * 0.05 || yMax * 0.02;
  const lo = yMin - yPad;
  const hi = yMax + yPad;
  const yRange = hi - lo || 1;

  const pad = { l: 60, r: 16, t: 16, b: 36 };
  const w = 900;
  const h = height;
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;

  const xFor = (ts: number) =>
    pad.l + ((ts - startTs) / tsRange) * innerW;
  const yFor = (v: number) =>
    pad.t + innerH - ((v - lo) / yRange) * innerH;

  const histPath = bars
    .map((b, i) => `${i === 0 ? "M" : "L"}${xFor(tsAt(b.ts)).toFixed(1)},${yFor(b.close).toFixed(1)}`)
    .join(" ");

  const testWithPx = testSeries.filter((p) => p.predicted_price != null);
  const testPath = testWithPx
    .map(
      (p, i) =>
        `${i === 0 ? "M" : "L"}${xFor(tsAt(p.date)).toFixed(1)},${yFor(p.predicted_price as number).toFixed(1)}`,
    )
    .join(" ");

  const forecastNodes: { ts: number; v: number }[] = [];
  if (anchorPrice != null) forecastNodes.push({ ts: lastBarTs, v: anchorPrice });
  for (const p of forecastPoints) {
    forecastNodes.push({ ts: tsAt(p.target_date), v: p.pred_price });
  }
  const forecastPath = forecastNodes
    .map((n, i) => `${i === 0 ? "M" : "L"}${xFor(n.ts).toFixed(1)},${yFor(n.v).toFixed(1)}`)
    .join(" ");

  const ticksY = [hi, lo + (hi - lo) * 0.66, lo + (hi - lo) * 0.33, lo];
  const todayX = xFor(lastBarTs);

  return (
    <div className="panel">
      <p className="text-sm font-semibold mb-2">🎯 Model Accuracy: Actual vs Predicted Prices</p>
      <ZoomableChart>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none">
        {ticksY.map((t, i) => (
          <g key={i}>
            <line
              x1={pad.l}
              y1={yFor(t)}
              x2={w - pad.r}
              y2={yFor(t)}
              stroke="#1f2a36"
              strokeDasharray="2 4"
            />
            <text x={pad.l - 6} y={yFor(t) + 3} fontSize={10} fill="#8b98a5" textAnchor="end">
              ₹{t.toFixed(0)}
            </text>
          </g>
        ))}

        <path d={histPath} fill="none" stroke="#22d3ee" strokeWidth={1.6} />
        {testPath && (
          <path
            d={testPath}
            fill="none"
            stroke="#ef4444"
            strokeWidth={1.4}
            strokeDasharray="2 4"
            opacity={0.95}
          />
        )}
        {forecastPath && (
          <path
            d={forecastPath}
            fill="none"
            stroke="#22c55e"
            strokeWidth={2}
            strokeLinejoin="round"
          />
        )}
        {forecastNodes.slice(1).map((n, i) => (
          <circle key={i} cx={xFor(n.ts)} cy={yFor(n.v)} r={3} fill="#22c55e" />
        ))}

        <line
          x1={todayX}
          y1={pad.t}
          x2={todayX}
          y2={pad.t + innerH}
          stroke="#8b98a5"
          strokeDasharray="3 4"
        />
        <text x={todayX + 4} y={pad.t + 12} fontSize={10} fill="#8b98a5">
          Forecast Start
        </text>

        <text x={pad.l} y={h - 8} fontSize={10} fill="#8b98a5">
          {bars[0].ts.slice(0, 10)}
        </text>
        <text x={w - pad.r} y={h - 8} fontSize={10} fill="#8b98a5" textAnchor="end">
          {forecastPoints.length
            ? forecastPoints[forecastPoints.length - 1].target_date
            : bars[bars.length - 1].ts.slice(0, 10)}
        </text>
      </svg>
      <div className="flex flex-wrap items-center gap-4 text-xs text-muted mt-2">
        <Legend color="#22d3ee" label="Actual Price (Historical)" />
        <Legend color="#ef4444" label="Model Prediction (Test Period)" dashed />
        <Legend color="#22c55e" label="Future Forecast" />
      </div>
      </ZoomableChart>
    </div>
  );
}

function Legend({ color, label, dashed = false }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1">
      <span
        className="inline-block w-3"
        style={{ borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}` }}
      />
      {label}
    </span>
  );
}
