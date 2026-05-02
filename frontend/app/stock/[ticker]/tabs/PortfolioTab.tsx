"use client";

import { useState, useEffect } from "react";

interface Position {
  type: "long" | "short";
  entry_price: number;
  shares: number | null;
  entry_date: string;
}

const STORAGE_PREFIX = "portfolio:position:v1:";

function loadPosition(ticker: string): Position | null {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + ticker);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function savePosition(ticker: string, pos: Position | null) {
  try {
    if (pos) localStorage.setItem(STORAGE_PREFIX + ticker, JSON.stringify(pos));
    else localStorage.removeItem(STORAGE_PREFIX + ticker);
  } catch {}
}

function badge(cls: string, children: React.ReactNode) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${cls}`}>
      {children}
    </span>
  );
}

export default function PortfolioTab({
  ticker,
  lastClose,
}: {
  ticker: string;
  lastClose: number | null;
}) {
  const [position, setPosition] = useState<Position | null>(null);
  const [hasPosition, setHasPosition] = useState(false);
  const [form, setForm] = useState<Partial<Position>>({ type: "long" });
  const [editing, setEditing] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const saved = loadPosition(ticker);
    if (saved) {
      setPosition(saved);
      setHasPosition(true);
      setForm(saved);
    }
    setMounted(true);
  }, [ticker]);

  if (!mounted) return null;

  function handleSave() {
    if (!form.entry_price || form.entry_price <= 0) return;
    const pos: Position = {
      type: form.type ?? "long",
      entry_price: Number(form.entry_price),
      shares: form.shares ? Number(form.shares) : null,
      entry_date: form.entry_date ?? "",
    };
    setPosition(pos);
    setHasPosition(true);
    setEditing(false);
    savePosition(ticker, pos);
  }

  function handleClear() {
    setPosition(null);
    setHasPosition(false);
    setForm({ type: "long" });
    setEditing(false);
    savePosition(ticker, null);
  }

  const price = lastClose;

  // P&L computation
  let pnlPct: number | null = null;
  let pnlRs: number | null = null;
  let pnlTone = "muted";

  if (position && price != null) {
    if (position.type === "long") {
      pnlPct = ((price - position.entry_price) / position.entry_price) * 100;
    } else {
      pnlPct = ((position.entry_price - price) / position.entry_price) * 100;
    }
    if (position.shares != null) {
      pnlRs =
        position.type === "long"
          ? (price - position.entry_price) * position.shares
          : (position.entry_price - price) * position.shares;
    }
    pnlTone = pnlPct >= 5 ? "bull" : pnlPct <= -5 ? "bear" : "muted";
  }

  function advice(pct: number, type: "long" | "short"): string {
    if (type === "long") {
      if (pct >= 20)
        return "Strong profit. Consider booking partial gains or tightening a trailing stop to protect your upside.";
      if (pct >= 10)
        return "Healthy gain. Trail a stop below recent support. Check the Overview tab for the model's current verdict before adding more.";
      if (pct >= 3)
        return "Small profit. Hold and monitor. Use the Overview tab to see if the model's directional signal still supports the trade.";
      if (pct >= -3)
        return "Near breakeven. No urgent action needed. Review the Overview tab verdict — a HOLD signal supports patience.";
      if (pct >= -10)
        return "Moderate loss. Review your original thesis. If the Overview tab shows a SELL signal, this may be a good exit point.";
      return "Significant loss. Evaluate whether the thesis is still intact. A strict stop-loss policy helps limit further downside.";
    } else {
      // short
      if (pct >= 20)
        return "Strong short profit. Consider covering partially or tightening your stop above resistance.";
      if (pct >= 5)
        return "Profitable short. Monitor for reversal signals on the Overview tab before adding to position.";
      if (pct >= -3)
        return "Near breakeven on a short. Hold if the Overview tab still shows a SELL signal.";
      return "Short position under pressure. Check the Overview tab — a BUY signal is a cue to cover.";
    }
  }

  const fmtINR = (v: number) =>
    new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 2,
    }).format(v);

  return (
    <div className="space-y-6">
      {/* Toggle */}
      <section className="rounded-xl border border-border bg-surface p-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-sm tracking-wide text-muted uppercase">
              My Position
            </h2>
            <p className="text-xs text-muted mt-0.5">
              Track your holding and get personalised guidance.
            </p>
          </div>
          {position && !editing && (
            <div className="flex gap-2">
              <button
                onClick={() => { setEditing(true); setForm(position); }}
                className="text-xs px-3 py-1.5 rounded border border-border hover:bg-surface-alt transition"
              >
                Edit
              </button>
              <button
                onClick={handleClear}
                className="text-xs px-3 py-1.5 rounded border border-border text-bear hover:bg-surface-alt transition"
              >
                Clear
              </button>
            </div>
          )}
        </div>

        {/* Not invested toggle */}
        {!hasPosition && !editing && (
          <div className="mt-4 flex gap-3">
            <button
              onClick={() => { setHasPosition(true); setEditing(true); }}
              className="px-4 py-2 rounded-lg bg-teal text-black text-sm font-semibold hover:opacity-90 transition"
            >
              I have a position
            </button>
            <button
              onClick={() => { setHasPosition(false); setEditing(false); }}
              className="px-4 py-2 rounded-lg border border-border text-sm hover:bg-surface-alt transition"
            >
              Not invested yet
            </button>
          </div>
        )}

        {/* Entry form */}
        {(editing || (hasPosition && !position)) && (
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1">Position type</label>
              <select
                className="w-full bg-surface border border-border rounded px-3 py-2 text-sm"
                value={form.type ?? "long"}
                onChange={(e) => setForm((f) => ({ ...f, type: e.target.value as "long" | "short" }))}
              >
                <option value="long">Long (bought)</option>
                <option value="short">Short (sold / shorted)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-muted mb-1">Entry price (₹)</label>
              <input
                type="number"
                min={0}
                step="0.01"
                className="w-full bg-surface border border-border rounded px-3 py-2 text-sm"
                placeholder="e.g. 1380"
                value={form.entry_price ?? ""}
                onChange={(e) => setForm((f) => ({ ...f, entry_price: parseFloat(e.target.value) }))}
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1">Shares / quantity (optional)</label>
              <input
                type="number"
                min={0}
                step="1"
                className="w-full bg-surface border border-border rounded px-3 py-2 text-sm"
                placeholder="e.g. 50"
                value={form.shares ?? ""}
                onChange={(e) => setForm((f) => ({ ...f, shares: parseFloat(e.target.value) || null }))}
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1">Entry date (optional)</label>
              <input
                type="date"
                className="w-full bg-surface border border-border rounded px-3 py-2 text-sm"
                value={form.entry_date ?? ""}
                onChange={(e) => setForm((f) => ({ ...f, entry_date: e.target.value }))}
              />
            </div>
            <div className="sm:col-span-2 flex gap-2">
              <button
                onClick={handleSave}
                disabled={!form.entry_price || form.entry_price <= 0}
                className="px-4 py-2 rounded-lg bg-teal text-black text-sm font-semibold disabled:opacity-50 hover:opacity-90 transition"
              >
                Save position
              </button>
              {position && (
                <button
                  onClick={() => setEditing(false)}
                  className="px-4 py-2 rounded-lg border border-border text-sm hover:bg-surface-alt transition"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        )}
      </section>

      {/* P&L dashboard */}
      {position && !editing && (
        <>
          <section className="rounded-xl border border-border bg-surface p-5 space-y-4">
            <div className="flex items-center gap-3">
              <h2 className="font-semibold text-sm tracking-wide text-muted uppercase">
                Position Summary
              </h2>
              {badge(
                position.type === "long"
                  ? "bg-bull/20 text-bull"
                  : "bg-bear/20 text-bear",
                position.type === "long" ? "LONG" : "SHORT"
              )}
              {position.entry_date &&
                badge("bg-surface-alt text-muted", `Entry ${position.entry_date}`)}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-muted uppercase tracking-wide">Entry price</p>
                <p className="text-lg font-semibold tabular-nums">
                  {fmtINR(position.entry_price)}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted uppercase tracking-wide">Last close</p>
                <p className="text-lg font-semibold tabular-nums">
                  {price != null ? fmtINR(price) : "—"}
                </p>
              </div>
              {position.shares != null && (
                <div>
                  <p className="text-xs text-muted uppercase tracking-wide">Quantity</p>
                  <p className="text-lg font-semibold tabular-nums">
                    {position.shares.toLocaleString("en-IN")} shares
                  </p>
                </div>
              )}
              {position.shares != null && price != null && (
                <div>
                  <p className="text-xs text-muted uppercase tracking-wide">Current value</p>
                  <p className="text-lg font-semibold tabular-nums">
                    {fmtINR(price * position.shares)}
                  </p>
                </div>
              )}
            </div>

            {/* P&L row */}
            {pnlPct != null && (
              <div className="rounded-lg border border-border bg-surface-alt p-4 flex flex-wrap gap-6">
                <div>
                  <p className="text-xs text-muted uppercase tracking-wide">Unrealised P&amp;L</p>
                  <p
                    className={`text-2xl font-bold tabular-nums ${
                      pnlTone === "bull"
                        ? "text-bull"
                        : pnlTone === "bear"
                        ? "text-bear"
                        : "text-foreground"
                    }`}
                  >
                    {pnlPct >= 0 ? "+" : ""}
                    {pnlPct.toFixed(2)}%
                  </p>
                  {pnlRs != null && (
                    <p
                      className={`text-sm tabular-nums ${
                        pnlRs >= 0 ? "text-bull" : "text-bear"
                      }`}
                    >
                      {pnlRs >= 0 ? "+" : ""}
                      {fmtINR(pnlRs)}
                    </p>
                  )}
                </div>
                <div>
                  <p className="text-xs text-muted uppercase tracking-wide">Status</p>
                  <p className="text-sm font-semibold mt-1">
                    {pnlPct >= 10
                      ? badge("bg-bull/20 text-bull", "Strong profit")
                      : pnlPct >= 3
                      ? badge("bg-bull/10 text-bull", "In profit")
                      : pnlPct >= -3
                      ? badge("bg-surface text-muted border border-border", "Near breakeven")
                      : pnlPct >= -10
                      ? badge("bg-bear/10 text-bear", "Moderate loss")
                      : badge("bg-bear/20 text-bear", "Significant loss")}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted uppercase tracking-wide">Breakeven at</p>
                  <p className="text-sm font-semibold tabular-nums mt-1">
                    {fmtINR(position.entry_price)}
                  </p>
                  <p className="text-xs text-muted">
                    {price != null
                      ? price >= position.entry_price
                        ? `₹${(price - position.entry_price).toFixed(2)} above`
                        : `₹${(position.entry_price - price).toFixed(2)} below`
                      : ""}
                  </p>
                </div>
              </div>
            )}
          </section>

          {/* Personalised advice */}
          {pnlPct != null && (
            <section className="rounded-xl border border-border bg-surface p-5">
              <h2 className="font-semibold text-sm tracking-wide text-muted uppercase mb-3">
                Personalised Guidance
              </h2>
              <p className="text-sm leading-relaxed">{advice(pnlPct, position.type)}</p>
              <p className="text-xs text-muted mt-3">
                For the model's directional forecast (BUY / SELL / HOLD),
                switch to the <strong>Overview</strong> tab and run a prediction.
                The verdict factors in calibrated probability, sentiment, and market regime.
              </p>
            </section>
          )}

          {/* Not invested nudge */}
          {price != null && (
            <section className="rounded-xl border border-border bg-surface p-5">
              <h2 className="font-semibold text-sm tracking-wide text-muted uppercase mb-2">
                Considering an entry?
              </h2>
              <p className="text-sm text-muted">
                Current last close is <strong>{fmtINR(price)}</strong>. Run a prediction on the{" "}
                <strong>Overview</strong> tab to get the AI verdict — if it says{" "}
                <span className="text-bull font-semibold">BUY</span>, P(up) has cleared the
                confidence threshold (τ*). If it says{" "}
                <span className="text-muted font-semibold">HOLD</span>, wait for a stronger signal.
              </p>
            </section>
          )}
        </>
      )}

      {/* No position — show guidance */}
      {!hasPosition && !editing && (
        <section className="rounded-xl border border-border bg-surface p-5">
          <h2 className="font-semibold text-sm tracking-wide text-muted uppercase mb-2">
            Not in a position?
          </h2>
          <p className="text-sm text-muted">
            Click <strong>I have a position</strong> above to track your entry. Or head to the{" "}
            <strong>Overview</strong> tab and run a prediction to see the AI verdict before deciding.
          </p>
          {price != null && (
            <p className="text-sm text-muted mt-2">
              Current last close: <span className="text-foreground font-semibold">{fmtINR(price)}</span>
            </p>
          )}
        </section>
      )}
    </div>
  );
}
