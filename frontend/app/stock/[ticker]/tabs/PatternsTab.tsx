import { apiFetch, ApiError } from "@/lib/api";
import Stat from "@/components/Stat";
import type { PatternBundle, StockHistory } from "@/lib/types";
import { formatINR, formatRatio } from "@/lib/format";
import PatternOverlayChart from "../charts/PatternOverlayChart";

export default async function PatternsTab({ ticker }: { ticker: string }) {
  const [data, ohlcv] = await Promise.all([
    safe<PatternBundle>(`/api/v1/stocks/${ticker}/patterns?lookback_days=365`),
    safe<StockHistory>(`/api/v1/stocks/${ticker}/ohlcv`),
  ]);

  if (!data) {
    return (
      <p className="panel text-sm text-muted">
        Pattern analysis unavailable — upstream may be rate-limited.
      </p>
    );
  }

  const marketTone =
    data.market_character === "Trending"
      ? "bull"
      : data.market_character === "Mean-Reverting"
      ? "bear"
      : "muted";

  const trendTone = (data.trend ?? "").toLowerCase().includes("up")
    ? "bull"
    : (data.trend ?? "").toLowerCase().includes("down")
    ? "bear"
    : "muted";

  return (
    <div className="space-y-5">
      <section className="panel grid gap-4 sm:grid-cols-4">
        <Stat label="Trend" value={data.trend ?? "—"} tone={trendTone} />
        <Stat
          label="Market character"
          value={data.market_character ?? "—"}
          tone={marketTone}
        />
        <Stat
          label="Hurst"
          value={formatRatio(data.hurst_exponent, 3)}
          hint={
            data.hurst_exponent == null
              ? undefined
              : data.hurst_exponent > 0.55
              ? "Persistent trend"
              : data.hurst_exponent < 0.45
              ? "Mean-reverting"
              : "Random walk"
          }
        />
        <Stat label="Bias" value={data.bias ?? "—"} />
      </section>

      <PatternOverlayChart
        bars={ohlcv?.bars ?? []}
        patterns={data.patterns}
        supportResistance={data.support_resistance}
      />

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Detected patterns</h2>
        {data.patterns.length === 0 ? (
          <p className="panel text-sm text-muted">
            No high-confidence patterns right now.
          </p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {data.patterns.map((p, i) => {
              const isBullish = (p.type ?? "").toLowerCase().includes("bull");
              const isBearish = (p.type ?? "").toLowerCase().includes("bear");
              const tone = isBullish ? "bull" : isBearish ? "bear" : "muted";
              return (
                <div key={i} className="panel">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className={`font-semibold text-${tone}`}>{p.name}</p>
                      {p.type && <p className="text-xs text-muted">{p.type}</p>}
                    </div>
                    <span className="chip tabular-nums">
                      {p.confidence.toFixed(0)}%
                    </span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    {p.neckline != null && (
                      <div>
                        <p className="text-xs uppercase text-muted">Neckline</p>
                        <p className="font-medium">{formatINR(p.neckline)}</p>
                      </div>
                    )}
                    {p.target != null && (
                      <div>
                        <p className="text-xs uppercase text-muted">Target</p>
                        <p className="font-medium">{formatINR(p.target)}</p>
                      </div>
                    )}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2 text-xs">
                    {p.volume_confirmed && (
                      <span className="chip text-bull">volume confirmed</span>
                    )}
                    {p.timeframe_confluence && (
                      <span className="chip text-accent">multi-timeframe</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {data.support_resistance && (
        <section className="panel">
          <p className="text-sm font-semibold mb-3">Support & Resistance</p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <p className="text-xs uppercase text-muted mb-1">Nearest support</p>
              <p className="text-lg font-semibold text-bull">
                {formatINR(data.support_resistance.nearest_support)}
              </p>
              <ul className="mt-2 space-y-1 text-sm">
                {data.support_resistance.strong_supports.slice(0, 5).map((s, i) => (
                  <li
                    key={i}
                    className="flex justify-between border-b border-border/40 pb-1"
                  >
                    <span className="text-muted">Support {i + 1}</span>
                    <span className="text-bull">{formatINR(s)}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-xs uppercase text-muted mb-1">
                Nearest resistance
              </p>
              <p className="text-lg font-semibold text-bear">
                {formatINR(data.support_resistance.nearest_resistance)}
              </p>
              <ul className="mt-2 space-y-1 text-sm">
                {data.support_resistance.strong_resistances
                  .slice(0, 5)
                  .map((s, i) => (
                    <li
                      key={i}
                      className="flex justify-between border-b border-border/40 pb-1"
                    >
                      <span className="text-muted">Resistance {i + 1}</span>
                      <span className="text-bear">{formatINR(s)}</span>
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

async function safe<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiError) return null;
    return null;
  }
}
