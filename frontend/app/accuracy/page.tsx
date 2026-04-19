import { createClient } from "@/utils/supabase/server";
import { redirect } from "next/navigation";
import { apiFetch, ApiError } from "@/lib/api";
import type { AccuracyWindow, LedgerRow } from "@/lib/types";

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
  const days = Math.max(1, Math.min(365, Number(sp.days ?? 30) || 30));

  const qs = new URLSearchParams({ days: String(days) });
  if (ticker) qs.set("ticker", ticker);

  const [win, rows] = await Promise.all([
    safe<AccuracyWindow>(`/api/v1/accuracy?${qs.toString()}`),
    safe<LedgerRow[]>(
      `/api/v1/accuracy/recent?limit=50${ticker ? `&ticker=${ticker}` : ""}`,
    ),
  ]);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Accuracy</h1>
        <p className="text-sm text-muted">
          Rolling window read from the prediction ledger (A7).
        </p>
      </header>

      <form className="panel flex items-end gap-3" method="get">
        <label>
          <span className="block text-xs uppercase text-muted mb-1">Ticker</span>
          <input
            name="ticker"
            defaultValue={ticker ?? ""}
            className="input"
            placeholder="All tickers"
          />
        </label>
        <label>
          <span className="block text-xs uppercase text-muted mb-1">Days</span>
          <input name="days" type="number" defaultValue={days} className="input w-28" />
        </label>
        <button className="btn btn-primary">Refresh</button>
      </form>

      {win && (
        <section className="panel grid gap-4 sm:grid-cols-4">
          <Stat label="Window" value={`${win.window_days}d`} />
          <Stat
            label="Directional acc."
            value={win.directional_accuracy != null ? `${(win.directional_accuracy * 100).toFixed(1)}%` : "—"}
          />
          <Stat
            label="Brier"
            value={win.brier_score != null ? win.brier_score.toFixed(3) : "—"}
          />
          <Stat
            label="ECE"
            value={win.ece != null ? `${(win.ece * 100).toFixed(2)}%` : "—"}
          />
          <Stat label="Predictions" value={String(win.n_predictions)} />
          <Stat label="Resolved" value={String(win.n_resolved)} />
          {win.mae_price != null && (
            <Stat label="MAE (₹)" value={win.mae_price.toFixed(2)} />
          )}
        </section>
      )}

      {rows && rows.length > 0 && (
        <section className="panel overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-muted">
              <tr>
                <th className="py-2 pr-4">Made</th>
                <th className="py-2 pr-4">Target</th>
                <th className="py-2 pr-4">Ticker</th>
                <th className="py-2 pr-4">Dir</th>
                <th className="py-2 pr-4">Pred ₹</th>
                <th className="py-2 pr-4">Actual ₹</th>
                <th className="py-2 pr-4">Hit</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.ticker}-${r.made_at}-${r.target_date}`} className="border-t border-border">
                  <td className="py-1 pr-4">{new Date(r.made_at).toLocaleDateString()}</td>
                  <td className="py-1 pr-4">{r.target_date}</td>
                  <td className="py-1 pr-4 font-medium">{r.ticker}</td>
                  <td className={`py-1 pr-4 text-${r.pred_dir === "up" ? "bull" : r.pred_dir === "down" ? "bear" : "muted"}`}>
                    {r.pred_dir}
                  </td>
                  <td className="py-1 pr-4">₹{r.pred_price.toFixed(2)}</td>
                  <td className="py-1 pr-4">{r.actual_price != null ? `₹${r.actual_price.toFixed(2)}` : "—"}</td>
                  <td className="py-1 pr-4">
                    {r.hit == null ? "…" : r.hit ? <span className="text-bull">✓</span> : <span className="text-bear">✗</span>}
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase text-muted">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
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
