import { createClient } from "@/utils/supabase/server";
import { apiFetch, ApiError } from "@/lib/api";
import type { StockHistory } from "@/lib/types";
import { redirect } from "next/navigation";
import AccuracyBadge from "./AccuracyBadge";
import StockTabs from "./StockTabs";
import { TAB_KEYS, DEFAULT_TAB } from "./tabs";
import OverviewTab from "./tabs/OverviewTab";
import TechnicalsTab from "./tabs/TechnicalsTab";
import SentimentTab from "./tabs/SentimentTab";
import FundamentalsTab from "./tabs/FundamentalsTab";
import FiiDiiTab from "./tabs/FiiDiiTab";
import PatternsTab from "./tabs/PatternsTab";
import BacktestTab from "./tabs/BacktestTab";
import AccuracyTab from "./tabs/AccuracyTab";
import { formatINR, formatPercent, toneFor } from "@/lib/format";

export default async function StockPage({
  params,
  searchParams,
}: {
  params: Promise<{ ticker: string }>;
  searchParams: Promise<{ tab?: string }>;
}) {
  const { ticker: raw } = await params;
  const ticker = raw.toUpperCase();
  const sp = await searchParams;
  const tab = TAB_KEYS.has(sp.tab ?? "") ? (sp.tab as string) : DEFAULT_TAB;

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect(`/login?next=/stock/${ticker}`);

  const ohlcv = await safe<StockHistory>(`/api/v1/stocks/${ticker}/ohlcv`);

  const lastClose = ohlcv?.bars.length ? ohlcv.bars[ohlcv.bars.length - 1].close : null;
  const firstClose = ohlcv?.bars.length ? ohlcv.bars[0].close : null;
  const windowChange =
    lastClose != null && firstClose != null && firstClose !== 0
      ? ((lastClose - firstClose) / firstClose) * 100
      : null;

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{ticker}</h1>
          <p className="text-sm text-muted">
            {ohlcv
              ? `${ohlcv.bars.length} bars · ${ohlcv.start} → ${ohlcv.end}`
              : "OHLCV unavailable — upstream may be rate-limited."}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastClose != null && (
            <div className="text-right">
              <p className="text-xs uppercase text-muted tracking-wide">Last close</p>
              <p className="text-xl font-semibold tabular-nums">
                {formatINR(lastClose)}
              </p>
              {windowChange != null && (
                <p
                  className={`text-xs ${
                    toneFor(windowChange) === "bull"
                      ? "text-bull"
                      : toneFor(windowChange) === "bear"
                      ? "text-bear"
                      : "text-muted"
                  }`}
                >
                  {formatPercent(windowChange)} over window
                </p>
              )}
            </div>
          )}
          <AccuracyBadge ticker={ticker} />
        </div>
      </header>

      <StockTabs ticker={ticker} active={tab} />

      <div className="pt-2">
        {tab === "overview" && <OverviewTab ticker={ticker} ohlcv={ohlcv} />}
        {tab === "technicals" && <TechnicalsTab ticker={ticker} />}
        {tab === "sentiment" && <SentimentTab ticker={ticker} />}
        {tab === "fundamentals" && <FundamentalsTab ticker={ticker} />}
        {tab === "fii-dii" && <FiiDiiTab />}
        {tab === "patterns" && <PatternsTab ticker={ticker} />}
        {tab === "backtest" && <BacktestTab ticker={ticker} />}
        {tab === "accuracy" && <AccuracyTab ticker={ticker} />}
      </div>
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
