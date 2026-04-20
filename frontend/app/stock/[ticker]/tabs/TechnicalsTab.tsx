import { apiFetch, ApiError } from "@/lib/api";
import Stat from "@/components/Stat";
import type { TechnicalSnapshot } from "@/lib/types";
import { formatINR, formatPercent, formatRatio, toneFor } from "@/lib/format";

export default async function TechnicalsTab({ ticker }: { ticker: string }) {
  const snap = await safe<TechnicalSnapshot>(
    `/api/v1/stocks/${ticker}/technicals?lookback_days=365`,
  );

  if (!snap || snap.last_close == null) {
    return (
      <p className="panel text-sm text-muted">
        Technical indicators unavailable — upstream quote feed may be rate-limited.
      </p>
    );
  }

  const rsiTone =
    snap.rsi_14 == null ? "muted" : snap.rsi_14 > 70 ? "bear" : snap.rsi_14 < 30 ? "bull" : "muted";
  const macdTone = toneFor(snap.macd_histogram);
  const above = (price: number | null | undefined, ma: number | null | undefined) =>
    price != null && ma != null ? (price > ma ? "bull" : "bear") : "muted";

  const fib = fibLevels(snap);

  return (
    <div className="space-y-5">
      <section className="panel">
        <p className="text-sm font-semibold mb-3">Momentum & trend</p>
        <div className="grid gap-4 sm:grid-cols-4">
          <Stat
            label="RSI (14)"
            value={formatRatio(snap.rsi_14, 1)}
            tone={rsiTone}
            hint={
              snap.rsi_14 == null
                ? undefined
                : snap.rsi_14 > 70
                ? "Overbought"
                : snap.rsi_14 < 30
                ? "Oversold"
                : "Neutral"
            }
          />
          <Stat
            label="MACD"
            value={formatRatio(snap.macd, 3)}
            hint={snap.macd_signal != null ? `signal ${formatRatio(snap.macd_signal, 3)}` : undefined}
          />
          <Stat
            label="MACD histogram"
            value={formatRatio(snap.macd_histogram, 3)}
            tone={macdTone}
            hint={
              snap.macd_histogram != null && snap.macd_histogram > 0 ? "Bullish cross bias" : "Bearish cross bias"
            }
          />
          <Stat
            label="ATR (14)"
            value={formatRatio(snap.atr_14, 2)}
            hint={snap.atr_14 != null && snap.last_close ? `${formatPercent((snap.atr_14 / snap.last_close) * 100, 2)} of price` : undefined}
          />
        </div>
      </section>

      <section className="panel">
        <p className="text-sm font-semibold mb-3">Moving averages</p>
        <div className="grid gap-4 sm:grid-cols-4">
          <Stat label="MA 5"   value={formatINR(snap.ma_5)}   tone={above(snap.last_close, snap.ma_5)} />
          <Stat label="MA 20"  value={formatINR(snap.ma_20)}  tone={above(snap.last_close, snap.ma_20)}  hint={snap.price_vs_ma20 != null ? `${formatPercent(snap.price_vs_ma20 * 100, 2)} gap` : undefined} />
          <Stat label="MA 50"  value={formatINR(snap.ma_50)}  tone={above(snap.last_close, snap.ma_50)}  hint={snap.price_vs_ma50 != null ? `${formatPercent(snap.price_vs_ma50 * 100, 2)} gap` : undefined} />
          <Stat label="MA 200" value={formatINR(snap.ma_200)} tone={above(snap.last_close, snap.ma_200)} />
        </div>
      </section>

      <section className="panel">
        <p className="text-sm font-semibold mb-3">Volatility & volume</p>
        <div className="grid gap-4 sm:grid-cols-4">
          <Stat
            label="20D volatility"
            value={snap.volatility_20d != null ? formatPercent(snap.volatility_20d * 100, 2) : "—"}
          />
          <Stat
            label="Volume ratio"
            value={formatRatio(snap.volume_ratio, 2)}
            tone={
              snap.volume_ratio == null
                ? "muted"
                : snap.volume_ratio > 1.5
                ? "bull"
                : snap.volume_ratio < 0.7
                ? "bear"
                : "muted"
            }
            hint="vs 20-day mean"
          />
          <Stat label="OBV" value={formatRatio(snap.obv, 0)} />
          <Stat label="As of" value={snap.as_of ?? "—"} />
        </div>
      </section>

      <section className="panel">
        <p className="text-sm font-semibold mb-3">Key levels (Fibonacci + pivots)</p>
        <div className="grid gap-4 sm:grid-cols-4">
          <Stat label="Pivot" value={formatINR(snap.pivot_point)} />
          <Stat label="Support 1" value={formatINR(snap.support_1)} tone="bull" />
          <Stat label="Resistance 1" value={formatINR(snap.resistance_1)} tone="bear" />
          <Stat label="Last close" value={formatINR(snap.last_close)} />
        </div>
        {fib && (
          <div className="mt-4 grid gap-4 sm:grid-cols-5 text-sm">
            <Stat label="Fib 0%"     value={formatINR(fib.p0)} />
            <Stat label="Fib 38.2%"  value={formatINR(fib.p382)} />
            <Stat label="Fib 50%"    value={formatINR(fib.p50)} />
            <Stat label="Fib 61.8%"  value={formatINR(fib.p618)} />
            <Stat label="Fib 100%"   value={formatINR(fib.p100)} />
          </div>
        )}
      </section>
    </div>
  );
}

/** Derived Fibonacci retracements from R1/S1 when available. */
function fibLevels(snap: TechnicalSnapshot) {
  if (snap.support_1 == null || snap.resistance_1 == null) return null;
  const hi = Math.max(snap.support_1, snap.resistance_1);
  const lo = Math.min(snap.support_1, snap.resistance_1);
  const span = hi - lo;
  return {
    p0: lo,
    p382: lo + span * 0.382,
    p50: lo + span * 0.5,
    p618: lo + span * 0.618,
    p100: hi,
  };
}

async function safe<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiError) return null;
    return null;
  }
}
