import { createClient } from "@/utils/supabase/server";
import { apiFetch, ApiError } from "@/lib/api";
import type { StockHistory, SentimentAggregate } from "@/lib/types";
import { redirect } from "next/navigation";
import PriceSparkline from "./PriceSparkline";
import PredictPanel from "./PredictPanel";
import AccuracyBadge from "./AccuracyBadge";

export default async function StockPage({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const { ticker: raw } = await params;
  const ticker = raw.toUpperCase();

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect(`/login?next=/stock/${ticker}`);

  const ohlcv = await safe<StockHistory>(`/api/v1/stocks/${ticker}/ohlcv`);
  const sentiment = await safe<SentimentAggregate>(
    `/api/v1/stocks/${ticker}/sentiment`,
  );

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{ticker}</h1>
          <p className="text-sm text-muted">
            {ohlcv
              ? `${ohlcv.bars.length} bars · ${ohlcv.start} → ${ohlcv.end}`
              : "OHLCV unavailable"}
          </p>
        </div>
        <AccuracyBadge ticker={ticker} />
      </header>

      {ohlcv && ohlcv.bars.length > 0 ? (
        <section className="panel">
          <PriceSparkline bars={ohlcv.bars} />
        </section>
      ) : (
        <section className="panel text-sm text-muted">
          No OHLCV returned — the upstream may be rate-limited.
        </section>
      )}

      <PredictPanel ticker={ticker} />

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Sentiment</h2>
        {sentiment ? <SentimentBlock s={sentiment} /> : (
          <p className="panel text-sm text-muted">No headline sentiment available.</p>
        )}
      </section>
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

function SentimentBlock({ s }: { s: SentimentAggregate }) {
  const tone = s.mean_score > 0.1 ? "bull" : s.mean_score < -0.1 ? "bear" : "muted";
  return (
    <div className="space-y-3">
      <div className="panel flex items-center gap-6">
        <div>
          <p className="text-xs uppercase text-muted">Mean score</p>
          <p className={`text-2xl font-semibold text-${tone}`}>
            {s.mean_score.toFixed(3)}
          </p>
        </div>
        <div className="text-sm text-muted">
          {s.n_headlines} headlines · +{s.pos_count} / -{s.neg_count} / ={s.neu_count}
        </div>
      </div>
      <ul className="space-y-2">
        {s.headlines.slice(0, 8).map((h, i) => (
          <li key={i} className="panel text-sm">
            <div className="flex items-center justify-between gap-3">
              <a
                href={h.url ?? "#"}
                target="_blank"
                rel="noreferrer"
                className="hover:text-accent line-clamp-2"
              >
                {h.title}
              </a>
              <span
                className={`chip ${
                  h.score.label === "positive"
                    ? "text-bull"
                    : h.score.label === "negative"
                    ? "text-bear"
                    : ""
                }`}
              >
                {h.score.label} · {(h.score.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-xs text-muted mt-1">
              {h.source}
              {h.published_at && ` · ${new Date(h.published_at).toLocaleString()}`}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
