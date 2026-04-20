"use client";

interface Slice {
  label: string;
  value: number;
  color?: string;
}

interface Props {
  slices: Slice[];
  size?: number;
  innerRadius?: number;
}

const PALETTE = [
  "#22d3ee",
  "#f97316",
  "#22c55e",
  "#a78bfa",
  "#f59e0b",
  "#60a5fa",
  "#ef4444",
  "#2dd4bf",
];

/**
 * Lightweight donut/pie chart with label + percentage callouts. Used for
 * the News Event Breakdown panel where category share matters more than
 * absolute count.
 */
export default function PieChart({ slices, size = 220, innerRadius = 0 }: Props) {
  const total = slices.reduce((a, s) => a + Math.max(0, s.value), 0);
  if (total <= 0 || slices.length === 0) {
    return <p className="text-sm text-muted">No data.</p>;
  }
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 6;

  let acc = 0;
  const arcs = slices.map((s, i) => {
    const value = Math.max(0, s.value);
    const startAng = (acc / total) * Math.PI * 2 - Math.PI / 2;
    const endAng = ((acc + value) / total) * Math.PI * 2 - Math.PI / 2;
    acc += value;
    const x1 = cx + r * Math.cos(startAng);
    const y1 = cy + r * Math.sin(startAng);
    const x2 = cx + r * Math.cos(endAng);
    const y2 = cy + r * Math.sin(endAng);
    const large = endAng - startAng > Math.PI ? 1 : 0;
    const midAng = (startAng + endAng) / 2;
    const lx = cx + (r * 0.6) * Math.cos(midAng);
    const ly = cy + (r * 0.6) * Math.sin(midAng);
    const color = s.color ?? PALETTE[i % PALETTE.length];
    const pct = (value / total) * 100;
    const path =
      innerRadius > 0
        ? donutPath(cx, cy, r, innerRadius, startAng, endAng, large)
        : `M${cx},${cy} L${x1.toFixed(2)},${y1.toFixed(2)} A${r},${r} 0 ${large} 1 ${x2.toFixed(2)},${y2.toFixed(2)} Z`;
    return { path, color, label: s.label, value, pct, lx, ly };
  });

  return (
    <div className="flex flex-wrap items-center gap-6">
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
        {arcs.map((a, i) => (
          <path key={i} d={a.path} fill={a.color} stroke="#0b1118" strokeWidth={1} />
        ))}
        {arcs.map((a, i) =>
          a.pct >= 6 ? (
            <text
              key={`l-${i}`}
              x={a.lx}
              y={a.ly}
              fontSize={11}
              fill="#0b1118"
              textAnchor="middle"
              dominantBaseline="middle"
              fontWeight={600}
            >
              {a.pct.toFixed(0)}%
            </text>
          ) : null,
        )}
      </svg>
      <ul className="space-y-1 text-sm">
        {arcs.map((a, i) => (
          <li key={i} className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: a.color }} />
            <span className="capitalize">{a.label}</span>
            <span className="text-muted tabular-nums ml-auto">
              {a.value} · {a.pct.toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function donutPath(
  cx: number,
  cy: number,
  rOuter: number,
  rInner: number,
  startAng: number,
  endAng: number,
  large: number,
): string {
  const x1 = cx + rOuter * Math.cos(startAng);
  const y1 = cy + rOuter * Math.sin(startAng);
  const x2 = cx + rOuter * Math.cos(endAng);
  const y2 = cy + rOuter * Math.sin(endAng);
  const x3 = cx + rInner * Math.cos(endAng);
  const y3 = cy + rInner * Math.sin(endAng);
  const x4 = cx + rInner * Math.cos(startAng);
  const y4 = cy + rInner * Math.sin(startAng);
  return `M${x1},${y1} A${rOuter},${rOuter} 0 ${large} 1 ${x2},${y2} L${x3},${y3} A${rInner},${rInner} 0 ${large} 0 ${x4},${y4} Z`;
}
