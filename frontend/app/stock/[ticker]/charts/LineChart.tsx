"use client";

interface Series {
  label: string;
  color: string;
  data: { x: string | number; y: number }[];
}

interface Props {
  series: Series[];
  height?: number;
  yLabel?: string;
}

/**
 * Lightweight SVG line chart. Used for equity curves (backtest tab) and
 * any auxiliary overlay where pulling in lightweight-charts would be overkill.
 */
export default function LineChart({ series, height = 240, yLabel }: Props) {
  if (series.length === 0 || series.every((s) => s.data.length === 0)) {
    return <p className="text-sm text-muted">No data.</p>;
  }

  const allY = series.flatMap((s) => s.data.map((d) => d.y));
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yRange = yMax - yMin || 1;

  const maxLen = Math.max(...series.map((s) => s.data.length));
  const w = 800;
  const h = height;
  const padX = 40;
  const padY = 20;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const pathFor = (s: Series) => {
    if (s.data.length < 2) return "";
    const step = innerW / (s.data.length - 1);
    return s.data
      .map((d, i) => {
        const x = padX + i * step;
        const y = padY + innerH - ((d.y - yMin) / yRange) * innerH;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  };

  const ticksY = [yMin, yMin + yRange / 2, yMax];

  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none">
        {ticksY.map((t, i) => {
          const y = padY + innerH - ((t - yMin) / yRange) * innerH;
          return (
            <g key={i}>
              <line x1={padX} y1={y} x2={w - padX} y2={y} stroke="#1f2a36" strokeDasharray="2 3" />
              <text x={6} y={y + 4} fill="#8b98a5" fontSize={10}>
                {t.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </text>
            </g>
          );
        })}
        {series.map((s) => (
          <path key={s.label} d={pathFor(s)} fill="none" stroke={s.color} strokeWidth={2} />
        ))}
        {/* X labels: first + last */}
        {(() => {
          const first = series.find((s) => s.data.length > 0);
          if (!first) return null;
          return (
            <>
              <text x={padX} y={h - 4} fill="#8b98a5" fontSize={10}>
                {String(first.data[0].x)}
              </text>
              <text x={w - padX} y={h - 4} fill="#8b98a5" fontSize={10} textAnchor="end">
                {String(first.data[first.data.length - 1].x)}
              </text>
            </>
          );
        })()}
      </svg>
      <div className="flex flex-wrap gap-4 text-xs text-muted">
        {series.map((s) => (
          <span key={s.label} className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5" style={{ background: s.color }} /> {s.label}
          </span>
        ))}
        {yLabel && <span className="ml-auto">{yLabel}</span>}
        <span>n={maxLen}</span>
      </div>
    </div>
  );
}
