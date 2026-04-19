"use client";

interface Bar {
  label: string;
  value: number;
}

interface Props {
  bars: Bar[];
  height?: number;
  signedColors?: boolean;
}

/**
 * Sign-aware horizontal bar chart. Used for SHAP importances and for
 * category distributions where magnitude carries meaning.
 */
export default function HorizontalBarChart({ bars, height = 300, signedColors = true }: Props) {
  if (bars.length === 0) return <p className="text-sm text-muted">No data.</p>;

  const maxAbs = Math.max(...bars.map((b) => Math.abs(b.value)), 1e-9);
  const w = 640;
  const rowH = Math.max(16, Math.floor(height / bars.length));
  const chartH = rowH * bars.length;
  const labelW = 170;
  const innerW = w - labelW - 20;
  const midX = labelW + innerW / 2;

  return (
    <svg viewBox={`0 0 ${w} ${chartH}`} className="w-full" preserveAspectRatio="none">
      {bars.map((b, i) => {
        const yTop = i * rowH;
        const yMid = yTop + rowH / 2;
        const frac = b.value / maxAbs;
        const barW = (Math.abs(frac) * innerW) / 2;
        const x = frac >= 0 ? midX : midX - barW;
        const color = signedColors ? (b.value >= 0 ? "#22c55e" : "#ef4444") : "#2dd4bf";
        return (
          <g key={i}>
            <text x={labelW - 6} y={yMid + 3} fill="#e6edf3" fontSize={11} textAnchor="end">
              {b.label}
            </text>
            <rect x={x} y={yTop + 3} width={barW} height={rowH - 6} fill={color} opacity={0.85} rx={2} />
            <text
              x={frac >= 0 ? x + barW + 4 : x - 4}
              y={yMid + 3}
              fill="#8b98a5"
              fontSize={10}
              textAnchor={frac >= 0 ? "start" : "end"}
            >
              {b.value.toFixed(3)}
            </text>
          </g>
        );
      })}
      <line x1={midX} y1={0} x2={midX} y2={chartH} stroke="#1f2a36" />
    </svg>
  );
}
