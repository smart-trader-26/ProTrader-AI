import { createClient } from "@/utils/supabase/server";
import { redirect } from "next/navigation";
import Link from "next/link";
import { apiFetch, ApiError } from "@/lib/api";
import Stat from "@/components/Stat";
import { formatINR, formatPercent, formatRatio } from "@/lib/format";
import type { AccuracyWindow, LedgerRow } from "@/lib/types";

const WINDOWS = [7, 30, 90] as const;

export default async function AccuracyPage({
  searchParams,
}: {
  searchParams: Promise<{ ticker?: string; days?: string }>;
}) {
  const sp = await searchParams;
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login?next=/accuracy");

  const ticker = sp.ticker?.toUpperCase();

  const qs = (days: number) => {
    const p = new URLSearchParams({ days: String(days) });
    if (ticker) p.set("ticker", ticker);
    return p.toString();
  };

  const [w7, w30, w90, rows] = await Promise.all([
    safe<AccuracyWindow>(`/api/v1/accuracy?${qs(7)}`),
    safe<AccuracyWindow>(`/api/v1/accuracy?${qs(30)}`),
    safe<AccuracyWindow>(`/api/v1/accuracy?${qs(90)}`),
    safe<LedgerRow[]>(
      `/api/v1/accuracy/recent?limit=50${ticker ? `&ticker=${ticker}` : ""}`,
    ),
  ]);

  const wins = [w7, w30, w90];
  const hasAny = wins.some((w) => w && w.n_resolved > 0);

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Global accuracy</h1>
        <p className="text-sm text-muted">
          Rolling windows from the prediction ledger. Every forecast lands here,
          directional correctness is scored once the target date closes.
        </p>
      </header>

      <form className="panel flex flex-wrap items-end gap-3" method="get">
        <label className="flex flex-col">
          <span className="text-xs uppercase text-muted tracking-wide mb-1">Ticker</span>
          <input
            name="ticker"
            defaultValue={ticker ?? ""}
            className="input w-44"
            placeholder="All tickers"
            suppressHydrationWarning
          />
        </label>
        <button className="btn btn-primary" suppressHydrationWarning>Refresh</button>
        {ticker && (
          <Link href="/accuracy" className="btn">
            Clear filter
          </Link>
        )}
        <p className="ml-auto text-xs text-muted self-center">
          {ticker ? `Filtered to ${ticker}` : "Aggregated across all tickers"}
        </p>
      </form>

      {!hasAny ? (
        <div className="panel text-sm text-muted space-y-2">
          <p>
            No resolved predictions yet
            {ticker ? ` for ${ticker}` : ""}.
          </p>
          <p className="text-xs">
            Predictions resolve once their target date passes. Run one from any
            stock page to seed the ledger.
          </p>
        </div>
      ) : (
        <section className="grid gap-4 md:grid-cols-3">
          {WINDOWS.map((days, i) => (
            <WindowCard key={days} days={days} win={wins[i]} />
          ))}
        </section>
      )}

      {rows && rows.length > 0 && (
        <section className="panel overflow-x-auto">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold">Recent predictions</p>
            <p className="text-xs text-muted">
              {rows.filter((r) => r.hit != null).length} resolved ·{" "}
              {rows.filter((r) => r.hit == null).length} pending
            </p>
          </div>
          <table className="w-full text-sm">
            <thead className="text-left text-muted text-xs uppercase">
              <tr>
                <th className="pb-2">Made</th>
                <th className="pb-2">Target</th>
                <th className="pb-2">Ticker</th>
                <th className="pb-2">Dir</th>
                <th className="pb-2 text-right">P(up)</th>
                <th className="pb-2 text-right">Pred ₹</th>
                <th className="pb-2 text-right">Actual ₹</th>
                <th className="pb-2 text-right">Hit</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={`${r.ticker}-${r.made_at}-${r.target_date}`}
                  className="border-t border-border"
                >
                  <td className="py-1">{new Date(r.made_at).toLocaleDateString()}</td>
                  <td className="py-1">{r.target_date}</td>
                  <td className="py-1">
                    <Link
                      href={`/stock/${r.ticker}?tab=accuracy`}
                      className="font-medium hover:text-accent"
                    >
                      {r.ticker}
                    </Link>
                  </td>
                  <td
                    className={`py-1 ${
                      r.pred_dir === "up"
                        ? "text-bull"
                        : r.pred_dir === "down"
                        ? "text-bear"
                        : "text-muted"
                    }`}
                  >
                    {r.pred_dir}
                  </td>
                  <td className="py-1 text-right tabular-nums">
                    {r.prob_up != null ? formatPercent(r.prob_up * 100, 1) : "—"}
                  </td>
                  <td className="py-1 text-right tabular-nums">{formatINR(r.pred_price)}</td>
                  <td className="py-1 text-right tabular-nums">
                    {r.actual_price != null ? formatINR(r.actual_price) : "—"}
                  </td>
                  <td className="py-1 text-right">
                    {r.hit == null ? (
                      <span className="text-muted">…</span>
                    ) : r.hit ? (
                      <span className="text-bull">✓</span>
                    ) : (
                      <span className="text-bear">✗</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

function WindowCard({
  days,
  win,
}: {
  days: number;
  win: AccuracyWindow | null;
}) {
  if (!win || win.n_resolved === 0) {
    return (
      <div className="panel">
        <p className="text-xs uppercase text-muted tracking-wide">{days}-day window</p>
        <p className="text-sm text-muted mt-2">No resolved predictions yet.</p>
      </div>
    );
  }
  const acc = win.directional_accuracy != null ? win.directional_accuracy * 100 : null;
  const tone = acc == null ? "muted" : acc >= 58 ? "bull" : acc >= 50 ? "muted" : "bear";
  return (
    <div className="panel space-y-3">
      <p className="text-xs uppercase text-muted tracking-wide">{days}-day window</p>
      <Stat
        label="Directional accuracy"
        value={acc == null ? "—" : `${acc.toFixed(1)}%`}
        tone={tone}
        hint={`${win.n_resolved} resolved · ${win.n_predictions} total`}
      />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-xs uppercase text-muted">Brier</p>
          <p className="font-medium">{formatRatio(win.brier_score, 3)}</p>
        </div>
        <div>
          <p className="text-xs uppercase text-muted">ECE</p>
          <p className="font-medium">
            {win.ece != null ? `${(win.ece * 100).toFixed(2)}%` : "—"}
          </p>
        </div>
        {win.mae_price != null && (
          <div className="col-span-2">
            <p className="text-xs uppercase text-muted">MAE (₹)</p>
            <p className="font-medium">{formatINR(win.mae_price)}</p>
          </div>
        )}
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
