"use client";

import type { PredictionPoint, StockBar, TestSeriesPoint } from "@/lib/types";
import ZoomableChart, { type Domain } from "@/components/ZoomableChart";

interface Props {
  bars: StockBar[];
  testSeries: TestSeriesPoint[];
  forecastPoints: PredictionPoint[];
  anchorPrice: number | null;
  height?: number;
}

const W = 900;
const PAD = { l: 64, r: 16, t: 16, b: 36 };

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

  const H = height;
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  const tsAt = (s: string) => new Date(s).getTime();
  const startTs = tsAt(bars[0].ts);
  const lastBarTs = tsAt(bars[bars.length - 1].ts);
  const lastForecast = forecastPoints.length
    ? tsAt(forecastPoints[forecastPoints.length - 1].target_date)
    : lastBarTs;
  const endTs = Math.max(lastBarTs, lastForecast);

  // Build forecast nodes
  const forecastNodes: { ts: number; v: number }[] = [];
  if (anchorPrice != null) forecastNodes.push({ ts: lastBarTs, v: anchorPrice });
  for (const p of forecastPoints) {
    forecastNodes.push({ ts: tsAt(p.target_date), v: p.pred_price });
  }

  // full price range
  const allPrices: number[] = [
    ...bars.map((b) => b.close),
    ...testSeries.filter((p) => p.predicted_price != null).map((p) => p.predicted_price as number),
    ...forecastNodes.map((n) => n.v),
  ];
  if (anchorPrice != null) allPrices.push(anchorPrice);
  const yPad = (Math.max(...allPrices) - Math.min(...allPrices)) * 0.05 || Math.max(...allPrices) * 0.02;

  const fullDomain: Domain = {
    x0: startTs,
    x1: endTs,
    y0: Math.min(...allPrices) - yPad,
    y1: Math.max(...allPrices) + yPad,
  };

  return (
    <div className="panel">
      <p className="text-sm font-semibold mb-2">🎯 Model Accuracy: Actual vs Predicted Prices</p>

      <ZoomableChart fullDomain={fullDomain} pad={PAD} svgW={W} svgH={H}>
        {({ domain }) => {
          const dom = domain ?? fullDomain;
          const xRange = dom.x1 - dom.x0 || 1;
          const yRange = dom.y1 - dom.y0 || 1;

          const xFor = (ts: number) => PAD.l + ((ts - dom.x0) / xRange) * innerW;
          const yFor = (v: number) => PAD.t + innerH - ((v - dom.y0) / yRange) * innerH;

          // ── visible slices ─────────────────────────────────────────────
          const visBars = bars.filter(
            (b) => tsAt(b.ts) >= dom.x0 - 1 && tsAt(b.ts) <= dom.x1 + 1,
          );
          const histPath = visBars
            .map((b, i) => `${i === 0 ? "M" : "L"}${xFor(tsAt(b.ts)).toFixed(1)},${yFor(b.close).toFixed(1)}`)
            .join(" ");

          const visTest = testSeries.filter(
            (p) => p.predicted_price != null && tsAt(p.date) >= dom.x0 - 1 && tsAt(p.date) <= dom.x1 + 1,
          );
          const testPath = visTest
            .map((p, i) => `${i === 0 ? "M" : "L"}${xFor(tsAt(p.date)).toFixed(1)},${yFor(p.predicted_price as number).toFixed(1)}`)
            .join(" ");

          const visForecast = forecastNodes.filter(
            (n) => n.ts >= dom.x0 - 1 && n.ts <= dom.x1 + 1,
          );
          const forecastPath = visForecast
            .map((n, i) => `${i === 0 ? "M" : "L"}${xFor(n.ts).toFixed(1)},${yFor(n.v).toFixed(1)}`)
            .join(" ");

          // ── ticks ──────────────────────────────────────────────────────
          const nYTicks = 5;
          const yTicks = Array.from({ length: nYTicks }, (_, i) => dom.y0 + (i / (nYTicks - 1)) * yRange);
          const nXTicks = 5;
          const xTicks = Array.from({ length: nXTicks }, (_, i) => dom.x0 + (i / (nXTicks - 1)) * xRange);
          const fmtDate = (ts: number) => new Date(ts).toISOString().slice(0, 10);

          const todayInView = lastBarTs >= dom.x0 && lastBarTs <= dom.x1;
          const todayX = xFor(lastBarTs);

          return (
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: "block" }} preserveAspectRatio="none">
              <defs>
                <clipPath id="mac-clip">
                  <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH} />
                </clipPath>
              </defs>

              {/* y-axis */}
              {yTicks.map((t, i) => (
                <g key={i}>
                  <line x1={PAD.l} y1={yFor(t)} x2={W - PAD.r} y2={yFor(t)} stroke="#1f2a36" strokeDasharray="2 4" />
                  <text x={PAD.l - 6} y={yFor(t) + 3} fontSize={10} fill="#8b98a5" textAnchor="end">
                    ₹{t.toFixed(0)}
                  </text>
                </g>
              ))}

              <g clipPath="url(#mac-clip)">
                <path d={histPath} fill="none" stroke="#22d3ee" strokeWidth={1.6} />
                {testPath && (
                  <path d={testPath} fill="none" stroke="#ef4444" strokeWidth={1.4} strokeDasharray="2 4" opacity={0.95} />
                )}
                {forecastPath && (
                  <path d={forecastPath} fill="none" stroke="#22c55e" strokeWidth={2} strokeLinejoin="round" />
                )}
                {visForecast.slice(1).map((n, i) => (
                  <circle key={i} cx={xFor(n.ts)} cy={yFor(n.v)} r={3} fill="#22c55e" />
                ))}

                {todayInView && (
                  <>
                    <line x1={todayX} y1={PAD.t} x2={todayX} y2={PAD.t + innerH} stroke="#8b98a5" strokeDasharray="3 4" />
                    <text x={todayX + 4} y={PAD.t + 12} fontSize={10} fill="#8b98a5">Forecast Start</text>
                  </>
                )}
              </g>

              {/* x-axis labels */}
              {xTicks.map((ts, i) => (
                <text
                  key={i}
                  x={xFor(ts)}
                  y={H - 6}
                  fontSize={10}
                  fill="#8b98a5"
                  textAnchor={i === 0 ? "start" : i === nXTicks - 1 ? "end" : "middle"}
                >
                  {fmtDate(ts)}
                </text>
              ))}

              <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH} fill="none" stroke="#1f2a36" strokeWidth={0.5} />
            </svg>
          );
        }}
      </ZoomableChart>

      <div className="flex flex-wrap items-center gap-4 text-xs text-muted mt-2">
        <Legend color="#22d3ee" label="Actual Price (Historical)" />
        <Legend color="#ef4444" label="Model Prediction (Test Period)" dashed />
        <Legend color="#22c55e" label="Future Forecast" />
      </div>
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
