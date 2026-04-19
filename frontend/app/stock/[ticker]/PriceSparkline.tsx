import type { StockBar } from "@/lib/types";

/** SVG sparkline — no chart lib needed for a simple close-price ribbon. */
export default function PriceSparkline({ bars }: { bars: StockBar[] }) {
  if (bars.length < 2) return null;
  const closes = bars.map((b) => b.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;
  const w = 800;
  const h = 180;
  const stepX = w / (closes.length - 1);
  const path = closes
    .map((c, i) => {
      const x = i * stepX;
      const y = h - ((c - min) / range) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  const last = closes[closes.length - 1];
  const first = closes[0];
  const upColor = last >= first ? "#22c55e" : "#ef4444";

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <div>
          <p className="text-xs uppercase text-muted">Last close</p>
          <p className="text-2xl font-semibold">₹{last.toFixed(2)}</p>
        </div>
        <p className="text-sm" style={{ color: upColor }}>
          {(((last - first) / first) * 100).toFixed(2)}% · {bars.length} bars
        </p>
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        preserveAspectRatio="none"
      >
        <path d={path} fill="none" stroke={upColor} strokeWidth={2} />
      </svg>
    </div>
  );
}
