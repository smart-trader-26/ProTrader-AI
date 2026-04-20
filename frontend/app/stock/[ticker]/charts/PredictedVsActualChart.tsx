"use client";

import type { TestSeriesPoint } from "@/lib/types";
import ZoomableChart from "@/components/ZoomableChart";

interface Props {
  series: TestSeriesPoint[];
  height?: number;
  title?: string;
}

/**
 * Streamlit-parity "Predicted vs Actual Returns" chart — two overlaid line
 * series across the held-out test fold. Pure SVG so it renders without the
 * lightweight-charts dep.
 */
export default function PredictedVsActualChart({
  series,
  height = 320,
  title = "Predicted vs Actual Returns",
}: Props) {
  if (!series || series.length < 2) {
    return (
      <div className="panel">
        <p className="text-sm font-semibold mb-2">{title}</p>
        <p className="text-sm text-muted">No test-fold series in this prediction.</p>
      </div>
    );
  }

  const pad = { l: 50, r: 16, t: 16, b: 32 };
  const w = 900;
  const h = height;
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;

  const allY = series.flatMap((p) => [p.actual_return, p.predicted_return]);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yPad = (yMax - yMin) * 0.05 || 0.01;
  const lo = yMin - yPad;
  const hi = yMax + yPad;
  const yRange = hi - lo || 1;

  const x = (i: number) => pad.l + (i / (series.length - 1)) * innerW;
  const y = (v: number) => pad.t + innerH - ((v - lo) / yRange) * innerH;

  const path = (key: "actual_return" | "predicted_return") =>
    series
      .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`)
      .join(" ");

  const ticks = [hi, (hi + lo) / 2, lo];
  const zeroY = lo <= 0 && hi >= 0 ? y(0) : null;

  const firstDate = series[0].date;
  const lastDate = series[series.length - 1].date;
  const midDate = series[Math.floor(series.length / 2)].date;

  return (
    <div className="panel">
      <p className="text-sm font-semibold mb-2">{title}</p>
      <ZoomableChart>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none">
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={pad.l}
              y1={y(t)}
              x2={w - pad.r}
              y2={y(t)}
              stroke="#1f2a36"
              strokeDasharray="2 4"
            />
            <text x={pad.l - 8} y={y(t) + 3} fontSize={10} fill="#8b98a5" textAnchor="end">
              {(t * 100).toFixed(2)}%
            </text>
          </g>
        ))}
        {zeroY != null && (
          <line
            x1={pad.l}
            y1={zeroY}
            x2={w - pad.r}
            y2={zeroY}
            stroke="#3a4a5a"
            strokeWidth={1}
          />
        )}
        <path d={path("actual_return")} fill="none" stroke="#3b82f6" strokeWidth={1.4} />
        <path
          d={path("predicted_return")}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={1.4}
          opacity={0.95}
        />
        <text x={pad.l} y={h - 8} fontSize={10} fill="#8b98a5">
          {firstDate}
        </text>
        <text
          x={pad.l + innerW / 2}
          y={h - 8}
          fontSize={10}
          fill="#8b98a5"
          textAnchor="middle"
        >
          {midDate}
        </text>
        <text
          x={w - pad.r}
          y={h - 8}
          fontSize={10}
          fill="#8b98a5"
          textAnchor="end"
        >
          {lastDate}
        </text>
      </svg>
      </ZoomableChart>
      <div className="flex flex-wrap items-center gap-4 text-xs text-muted mt-2">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5" style={{ background: "#3b82f6" }} />{" "}
          Actual Returns
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5" style={{ background: "#f59e0b" }} />{" "}
          Predicted Returns
        </span>
        <span className="ml-auto">n = {series.length} test bars</span>
      </div>
    </div>
  );
}
