import { apiFetch, ApiError } from "@/lib/api";
import type { AccuracyWindow } from "@/lib/types";

export default async function AccuracyBadge({ ticker }: { ticker: string }) {
  let win: AccuracyWindow | null = null;
  try {
    win = await apiFetch<AccuracyWindow>(
      `/api/v1/accuracy?ticker=${encodeURIComponent(ticker)}&days=30`,
    );
  } catch (e) {
    if (!(e instanceof ApiError)) throw e;
  }
  if (!win || win.n_resolved === 0 || win.directional_accuracy == null) {
    return <span className="chip">30d accuracy: n/a</span>;
  }
  const pct = Math.round(win.directional_accuracy * 100);
  const tone = pct >= 58 ? "bull" : pct >= 50 ? "muted" : "bear";
  return (
    <span className={`chip text-${tone}`}>
      30d accuracy: {pct}% · {win.n_resolved} resolved
    </span>
  );
}
