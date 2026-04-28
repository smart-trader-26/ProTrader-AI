"use client";

import type { TestSeriesPoint } from "@/lib/types";
import ZoomableChart, { type Domain } from "@/components/ZoomableChart";

interface Props {
  series: TestSeriesPoint[];
  height?: number;
  title?: string;
}

const W = 900;
const PAD = { l: 54, r: 16, t: 16, b: 32 };

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

  const H = height;
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  // ── full data extents ──────────────────────────────────────────────────
  const allY = series.flatMap((p) => [p.actual_return, p.predicted_return]);
  const yPad = (Math.max(...allY) - Math.min(...allY)) * 0.05 || 0.01;
  const fullDomain: Domain = {
    x0: 0,
    x1: series.length - 1,
    y0: Math.min(...allY) - yPad,
    y1: Math.max(...allY) + yPad,
  };

  return (
    <div className="panel">
      <p className="text-sm font-semibold mb-2">{title}</p>
      <ZoomableChart
        fullDomain={fullDomain}
        pad={PAD}
        svgW={W}
        svgH={H}
      >
        {({ domain }) => {
          const dom = domain ?? fullDomain;

          // coordinate helpers bound to current domain
          const xRange = dom.x1 - dom.x0 || 1;
          const yRange = dom.y1 - dom.y0 || 1;
          const x = (i: number) =>
            PAD.l + ((i - dom.x0) / xRange) * innerW;
          const y = (v: number) =>
            PAD.t + innerH - ((v - dom.y0) / yRange) * innerH;

          // only render points inside the domain window
          const visible = series
            .map((p, i) => ({ p, i }))
            .filter(({ i }) => i >= Math.floor(dom.x0) - 1 && i <= Math.ceil(dom.x1) + 1);

          const path = (key: "actual_return" | "predicted_return") =>
            visible
              .map(({ p, i }, vi) =>
                `${vi === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`,
              )
              .join(" ");

          // y-axis ticks
          const nTicks = 5;
          const tickStep = yRange / (nTicks - 1);
          const ticks = Array.from({ length: nTicks }, (_, i) => dom.y0 + i * tickStep);

          // x-axis labels
          const nXLabels = 5;
          const xLabels = Array.from({ length: nXLabels }, (_, i) => {
            const idx = Math.round(dom.x0 + (i / (nXLabels - 1)) * xRange);
            const clamped = Math.max(0, Math.min(series.length - 1, idx));
            return { idx: clamped, label: series[clamped]?.date ?? "" };
          });

          const zeroY = dom.y0 <= 0 && dom.y1 >= 0 ? y(0) : null;

          return (
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: "block" }} preserveAspectRatio="none">
              {/* grid + y-axis */}
              {ticks.map((t, i) => (
                <g key={i}>
                  <line
                    x1={PAD.l} y1={y(t)}
                    x2={W - PAD.r} y2={y(t)}
                    stroke="#1f2a36" strokeDasharray="2 4"
                  />
                  <text x={PAD.l - 8} y={y(t) + 3} fontSize={10} fill="#8b98a5" textAnchor="end">
                    {(t * 100).toFixed(2)}%
                  </text>
                </g>
              ))}

              {/* zero line */}
              {zeroY != null && (
                <line x1={PAD.l} y1={zeroY} x2={W - PAD.r} y2={zeroY}
                  stroke="#3a4a5a" strokeWidth={1} />
              )}

              {/* clip to plot area */}
              <defs>
                <clipPath id="pvac-clip">
                  <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH} />
                </clipPath>
              </defs>

              <g clipPath="url(#pvac-clip)">
                <path d={path("actual_return")} fill="none" stroke="#3b82f6" strokeWidth={1.5} />
                <path d={path("predicted_return")} fill="none" stroke="#f59e0b" strokeWidth={1.5} opacity={0.95} />
              </g>

              {/* x-axis labels */}
              {xLabels.map(({ idx, label }, i) => (
                <text
                  key={i}
                  x={x(idx)}
                  y={H - 6}
                  fontSize={10}
                  fill="#8b98a5"
                  textAnchor={i === 0 ? "start" : i === nXLabels - 1 ? "end" : "middle"}
                >
                  {label}
                </text>
              ))}

              {/* border */}
              <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH}
                fill="none" stroke="#1f2a36" strokeWidth={0.5} />
            </svg>
          );
        }}
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
