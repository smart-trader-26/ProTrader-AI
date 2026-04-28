"use client";

import type { PredictionPoint, StockBar } from "@/lib/types";
import ZoomableChart, { type Domain } from "@/components/ZoomableChart";

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

const W = 900;
const PAD = { l: 64, r: 16, t: 16, b: 36 };

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
    return <p className="text-sm text-muted">Forecast unavailable.</p>;
  }

  const H = height;
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  const trailing = bars.slice(-historyDays);
  const startTs = new Date(trailing[0].ts).getTime();
  const lastBarTs = new Date(trailing[trailing.length - 1].ts).getTime();
  const lastForecastTs = new Date(predictions[predictions.length - 1].target_date).getTime();

  // Build forecast nodes (anchor + predictions)
  const forecastNodes: {
    ts: number; pred: number;
    p5?: number; p95?: number; p25?: number; p75?: number;
  }[] = [];
  if (anchorPrice != null) {
    // No CI bands on the anchor — they'd be zero-width (p5=p95=anchor).
    // The fan area naturally converges toward the anchor via the first prediction node.
    forecastNodes.push({ ts: lastBarTs, pred: anchorPrice });
  }
  for (const p of predictions) {
    forecastNodes.push({
      ts: new Date(p.target_date).getTime(),
      pred: p.pred_price,
      p5: p.ci_low ?? undefined,
      p95: p.ci_high ?? undefined,
      p25: p.p25_price ?? undefined,
      p75: p.p75_price ?? undefined,
    });
  }

  // Only check prediction nodes (skip the anchor which has no CI bands).
  const predNodes = forecastNodes.filter((n) => n.p5 != null || n.p95 != null || n.p25 != null);
  const haveOuter = predNodes.length > 0 && predNodes.every((n) => n.p5 != null && n.p95 != null);
  const haveInner = predNodes.length > 0 && predNodes.every((n) => n.p25 != null && n.p75 != null);

  // full price range
  const allPrices: number[] = [
    ...trailing.map((b) => b.close),
    ...forecastNodes.flatMap((n) => [n.pred, n.p5 ?? n.pred, n.p95 ?? n.pred, n.p25 ?? n.pred, n.p75 ?? n.pred]),
  ];
  const yPad = (Math.max(...allPrices) - Math.min(...allPrices)) * 0.08 || Math.max(...allPrices) * 0.02;

  const fullDomain: Domain = {
    x0: startTs,
    x1: lastForecastTs,
    y0: Math.min(...allPrices) - yPad,
    y1: Math.max(...allPrices) + yPad,
  };

  const startLabel = trailing[0].ts.slice(0, 10);
  const lastLabel = predictions[predictions.length - 1].target_date;

  return (
    <div className="panel">
      <div className="flex flex-wrap items-baseline justify-between mb-2 gap-x-3">
        <p className="text-sm font-semibold">📈 Price Forecast with Uncertainty Bands</p>
        <p className="text-xs text-muted">
          {ticker} — {predictions.length}-Day Probabilistic Forecast
          {bullishProb != null && (
            <> | <span>Bullish Prob: <span className="text-fg">{bullishProb.toFixed(1)}%</span></span></>
          )}
          {regime && (
            <> | <span>Regime: <span className="text-fg">{regime}</span></span></>
          )}
        </p>
      </div>

      <ZoomableChart fullDomain={fullDomain} pad={PAD} svgW={W} svgH={H}>
        {({ domain }) => {
          const dom = domain ?? fullDomain;
          const xRange = dom.x1 - dom.x0 || 1;
          const yRange = dom.y1 - dom.y0 || 1;

          const xFor = (ts: number) => PAD.l + ((ts - dom.x0) / xRange) * innerW;
          const yFor = (v: number) => PAD.t + innerH - ((v - dom.y0) / yRange) * innerH;

          // ── paths ──────────────────────────────────────────────────────
          const visTrailing = trailing.filter(
            (b) => new Date(b.ts).getTime() >= dom.x0 - 1 && new Date(b.ts).getTime() <= dom.x1 + 1,
          );
          const histPath = visTrailing
            .map((b, i) => `${i === 0 ? "M" : "L"}${xFor(new Date(b.ts).getTime()).toFixed(1)},${yFor(b.close).toFixed(1)}`)
            .join(" ");

          const visForecast = forecastNodes.filter(
            (n) => n.ts >= dom.x0 - 1 && n.ts <= dom.x1 + 1,
          );

          const fanPath = (loKey: "p5" | "p25", hiKey: "p95" | "p75") => {
            if (!visForecast.length) return "";
            const top = visForecast
              .map((n, i) => `${i === 0 ? "M" : "L"}${xFor(n.ts).toFixed(1)},${yFor(n[hiKey] as number).toFixed(1)}`)
              .join(" ");
            const bot = [...visForecast]
              .reverse()
              .map((n) => `L${xFor(n.ts).toFixed(1)},${yFor(n[loKey] as number).toFixed(1)}`)
              .join(" ");
            return `${top} ${bot} Z`;
          };

          const linePath = (key: "pred" | "p5" | "p95") =>
            visForecast
              .filter((n) => n[key] != null)
              .map((n, i) => `${i === 0 ? "M" : "L"}${xFor(n.ts).toFixed(1)},${yFor(n[key] as number).toFixed(1)}`)
              .join(" ");

          // ── ticks ──────────────────────────────────────────────────────
          const nYTicks = 5;
          const yTicks = Array.from({ length: nYTicks }, (_, i) => dom.y0 + (i / (nYTicks - 1)) * yRange);
          const nXTicks = 5;
          const xTicks = Array.from({ length: nXTicks }, (_, i) => dom.x0 + (i / (nXTicks - 1)) * xRange);
          const fmtDate = (ts: number) => new Date(ts).toISOString().slice(0, 10);

          const todayX = xFor(lastBarTs);
          const todayInView = lastBarTs >= dom.x0 && lastBarTs <= dom.x1;

          return (
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: "block" }} preserveAspectRatio="none">
              <defs>
                <clipPath id="pfb-clip">
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

              <g clipPath="url(#pfb-clip)">
                {haveOuter && <path d={fanPath("p5", "p95")} fill="rgba(45,212,191,0.18)" stroke="none" />}
                {haveInner && <path d={fanPath("p25", "p75")} fill="rgba(45,212,191,0.32)" stroke="none" />}

                <path d={histPath} fill="none" stroke="#22d3ee" strokeWidth={1.6} />

                {haveOuter && (
                  <>
                    <path d={linePath("p5")} fill="none" stroke="#ef4444" strokeWidth={1.2} strokeDasharray="4 3" opacity={0.85} />
                    <path d={linePath("p95")} fill="none" stroke="#22c55e" strokeWidth={1.2} strokeDasharray="4 3" opacity={0.85} />
                  </>
                )}
                <path d={linePath("pred")} fill="none" stroke="#ef4444" strokeWidth={2} />

                {todayInView && (
                  <>
                    <line x1={todayX} y1={PAD.t} x2={todayX} y2={PAD.t + innerH} stroke="#8b98a5" strokeDasharray="3 4" />
                    <text x={todayX + 4} y={PAD.t + 12} fontSize={10} fill="#8b98a5">Today</text>
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
  color, label, swatch = false, dashed = false,
}: { color: string; label: string; swatch?: boolean; dashed?: boolean }) {
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
