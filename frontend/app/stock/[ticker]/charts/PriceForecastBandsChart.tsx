"use client";

import type { PredictionPoint, StockBar } from "@/lib/types";
import ZoomableChart from "@/components/ZoomableChart";

interface Props {
  bars: StockBar[];
  predictions: PredictionPoint[];
  anchorPrice: number | null;
  ticker: string;
  bullishProb?: number | null;
  regime?: string | null;
  height?: number;
  historyDays?: number;
}

/**
 * Streamlit-parity "Price Forecast with Uncertainty Bands" chart.
 * Shows trailing history + median forecast + P5–P95 / P25–P75 fan bands +
 * pessimistic/optimistic dotted P5 / P95 lines. Pure SVG.
 */
export default function PriceForecastBandsChart({
  bars,
  predictions,
  anchorPrice,
  ticker,
  bullishProb,
  regime,
  height = 380,
  historyDays = 90,
}: Props) {
  if (!bars.length || !predictions.length) {
    return (
      <p className="text-sm text-muted">Forecast unavailable.</p>
    );
  }

  const trailing = bars.slice(-historyDays);
  const startTs = new Date(trailing[0].ts).getTime();
  const lastBarTs = new Date(trailing[trailing.length - 1].ts).getTime();
  const lastForecastTs = new Date(predictions[predictions.length - 1].target_date).getTime();
  const tsRange = lastForecastTs - startTs || 1;

  const allPrices: number[] = [
    ...trailing.map((b) => b.close),
    ...predictions.flatMap((p) => [
      p.pred_price,
      p.ci_low ?? p.pred_price,
      p.ci_high ?? p.pred_price,
      p.p25_price ?? p.pred_price,
      p.p75_price ?? p.pred_price,
    ]),
  ];
  if (anchorPrice != null) allPrices.push(anchorPrice);

  const yMin = Math.min(...allPrices);
  const yMax = Math.max(...allPrices);
  const yPad = (yMax - yMin) * 0.08 || yMax * 0.02;
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

  const histPath = trailing
    .map((b, i) => {
      const x = xFor(new Date(b.ts).getTime());
      const y = yFor(b.close);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const forecastNodes: { ts: number; pred: number; p5?: number; p95?: number; p25?: number; p75?: number }[] = [];
  if (anchorPrice != null) {
    forecastNodes.push({
      ts: lastBarTs,
      pred: anchorPrice,
      p5: anchorPrice,
      p95: anchorPrice,
      p25: anchorPrice,
      p75: anchorPrice,
    });
  }
  for (const p of predictions) {
    const ts = new Date(p.target_date).getTime();
    forecastNodes.push({
      ts,
      pred: p.pred_price,
      p5: p.ci_low ?? undefined,
      p95: p.ci_high ?? undefined,
      p25: p.p25_price ?? undefined,
      p75: p.p75_price ?? undefined,
    });
  }

  const haveOuter = forecastNodes.every((n) => n.p5 != null && n.p95 != null);
  const haveInner = forecastNodes.every((n) => n.p25 != null && n.p75 != null);

  const fanPath = (loKey: "p5" | "p25", hiKey: "p95" | "p75") => {
    if (forecastNodes.length === 0) return "";
    const top = forecastNodes
      .map((n, i) => `${i === 0 ? "M" : "L"}${xFor(n.ts).toFixed(1)},${yFor(n[hiKey] as number).toFixed(1)}`)
      .join(" ");
    const bot = forecastNodes
      .slice()
      .reverse()
      .map((n) => `L${xFor(n.ts).toFixed(1)},${yFor(n[loKey] as number).toFixed(1)}`)
      .join(" ");
    return `${top} ${bot} Z`;
  };

  const linePath = (key: "pred" | "p5" | "p95") =>
    forecastNodes
      .filter((n) => n[key] != null)
      .map((n, i) => `${i === 0 ? "M" : "L"}${xFor(n.ts).toFixed(1)},${yFor(n[key] as number).toFixed(1)}`)
      .join(" ");

  const ticksY = [hi, lo + (hi - lo) * 0.66, lo + (hi - lo) * 0.33, lo];
  const todayX = xFor(lastBarTs);

  const startLabel = trailing[0].ts.slice(0, 10);
  const lastLabel = predictions[predictions.length - 1].target_date;

  return (
    <div className="panel">
      <div className="flex flex-wrap items-baseline justify-between mb-2 gap-x-3">
        <p className="text-sm font-semibold">📈 Price Forecast with Uncertainty Bands</p>
        <p className="text-xs text-muted">
          {ticker} — {predictions.length}-Day Probabilistic Forecast
          {bullishProb != null && (
            <>
              {" "}| <span>Bullish Prob: <span className="text-fg">{bullishProb.toFixed(1)}%</span></span>
            </>
          )}
          {regime && (
            <>
              {" "}| <span>Regime: <span className="text-fg">{regime}</span></span>
            </>
          )}
        </p>
      </div>
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

        {haveOuter && (
          <path d={fanPath("p5", "p95")} fill="rgba(45, 212, 191, 0.18)" stroke="none" />
        )}
        {haveInner && (
          <path d={fanPath("p25", "p75")} fill="rgba(45, 212, 191, 0.32)" stroke="none" />
        )}

        <path d={histPath} fill="none" stroke="#22d3ee" strokeWidth={1.6} />

        {haveOuter && (
          <>
            <path
              d={linePath("p5")}
              fill="none"
              stroke="#ef4444"
              strokeWidth={1.2}
              strokeDasharray="4 3"
              opacity={0.85}
            />
            <path
              d={linePath("p95")}
              fill="none"
              stroke="#22c55e"
              strokeWidth={1.2}
              strokeDasharray="4 3"
              opacity={0.85}
            />
          </>
        )}
        <path d={linePath("pred")} fill="none" stroke="#ef4444" strokeWidth={2} />

        {/* Today divider */}
        <line
          x1={todayX}
          y1={pad.t}
          x2={todayX}
          y2={pad.t + innerH}
          stroke="#8b98a5"
          strokeDasharray="3 4"
        />
        <text x={todayX + 4} y={pad.t + 12} fontSize={10} fill="#8b98a5">
          Today
        </text>

        <text x={pad.l} y={h - 8} fontSize={10} fill="#8b98a5">
          {startLabel}
        </text>
        <text x={w - pad.r} y={h - 8} fontSize={10} fill="#8b98a5" textAnchor="end">
          {lastLabel}
        </text>
      </svg>
      </ZoomableChart>
      <div className="flex flex-wrap items-center gap-4 text-xs text-muted mt-2">
        <Legend color="#22d3ee" label="Historical" />
        <Legend color="rgba(45, 212, 191, 0.18)" label="P5–P95 Range" swatch />
        <Legend color="rgba(45, 212, 191, 0.32)" label="P25–P75 Range" swatch />
        <Legend color="#ef4444" label="Pessimistic (P5)" dashed />
        <Legend color="#22c55e" label="Optimistic (P95)" dashed />
        <Legend color="#ef4444" label="Median Forecast" />
      </div>
    </div>
  );
}

function Legend({
  color,
  label,
  swatch = false,
  dashed = false,
}: {
  color: string;
  label: string;
  swatch?: boolean;
  dashed?: boolean;
}) {
  return (
    <span className="flex items-center gap-1">
      <span
        className="inline-block w-3"
        style={
          swatch
            ? { background: color, height: "10px" }
            : { borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}` }
        }
      />
      {label}
    </span>
  );
}
