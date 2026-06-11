"use client";

import { apiGet } from "@/lib/api-client";
import type { SwingScanResult } from "@/lib/types";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

/**
 * P4-2 — Swing Screener: "which names is the 30-day signal firing on TODAY?"
 *
 * The signal abstains ~94% of the time by design, so the per-ticker view is
 * almost always NEUTRAL. This panel runs the persisted model across the whole
 * universe (one cached backend scan) and surfaces the fired names — the only
 * genuinely actionable output the platform produces.
 */
export default function SwingScreenerPanel() {
  const [scan, setScan] = useState<SwingScanResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async (refresh = false) => {
    setLoading(true);
    setErr(null);
    try {
      const res = await apiGet<SwingScanResult>(
        `/api/v1/swing/scan${refresh ? "?refresh=true" : ""}`,
      );
      setScan(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const pct = (x: number | null | undefined, d = 0) =>
    x == null ? "—" : `${(x * 100).toFixed(d)}%`;

  const fired = scan?.rows.filter((r) => r.signal === "UP") ?? [];
  const watchlist = scan?.rows.filter((r) => r.signal !== "UP").slice(0, 5) ?? [];

  return (
    <section className="panel space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">
            🎯 Swing Screener · {scan?.horizon_days ?? 30}-day signal across the universe
          </h2>
          <p className="text-xs text-muted">
            Names clearing the conviction threshold today. Long / neutral only —
            validated out-of-sample.
          </p>
        </div>
        <button
          onClick={() => void load(true)}
          disabled={loading}
          className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-fg disabled:opacity-50"
        >
          {loading ? "Scanning…" : "Re-scan"}
        </button>
      </div>

      {err && <p className="text-xs text-bear">Scan failed: {err}</p>}

      {loading && !scan && (
        <p className="text-xs text-muted animate-pulse">
          Scanning the universe (one batched download — a few seconds)…
        </p>
      )}

      {scan && !scan.trained && (
        <p className="text-xs text-muted">
          Swing model not trained. Run{" "}
          <code className="font-mono">DirectionalSignal(horizon=30).train()</code> to enable
          the screener.
        </p>
      )}

      {scan && scan.trained && (
        <>
          {/* Honesty header — the stats backing every call below */}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted border-b border-border pb-2">
            <span>
              Hit-rate when fired:{" "}
              <span className="text-accent font-semibold">{pct(scan.expected_hit_rate)}</span>
            </span>
            <span>
              vs. always-up base: <span className="text-fg">{pct(scan.base_rate)}</span>
            </span>
            <span>
              Fires {pct(scan.coverage)} of the time · τ* = {pct(scan.tau_star)}
            </span>
            <span>
              Scanned {scan.n_scanned} names
              {scan.cached ? " (cached)" : ""}
            </span>
          </div>

          {fired.length === 0 ? (
            <p className="text-xs text-muted">
              ● No high-conviction setups today — the signal is quiet by design (
              {pct(scan.coverage)} historical fire rate). Showing the closest names below.
            </p>
          ) : (
            <ul className="space-y-2">
              {fired.map((r) => (
                <li key={r.ticker} className="flex items-center gap-3">
                  <span className="rounded-full bg-bull/15 px-2 py-0.5 text-xs font-bold text-bull">
                    ▲ UP
                  </span>
                  <Link
                    href={`/stock/${r.ticker}`}
                    className="text-sm font-semibold hover:text-accent"
                  >
                    {r.ticker.replace(".NS", "")}
                  </Link>
                  <div className="flex-1 h-1.5 rounded-full bg-border/40 overflow-hidden">
                    <div
                      className="h-full bg-bull"
                      style={{ width: `${Math.min(100, r.prob_up * 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-bull font-semibold w-12 text-right">
                    {pct(r.prob_up, 1)}
                  </span>
                  {r.last_close != null && (
                    <span className="text-xs text-muted w-20 text-right">
                      ₹{r.last_close.toFixed(2)}
                    </span>
                  )}
                  <button
                    onClick={() =>
                      window.dispatchEvent(
                        new CustomEvent("protrader:propose", {
                          detail: { ticker: r.ticker },
                        }),
                      )
                    }
                    className="rounded border border-border px-2 py-0.5 text-[11px] text-muted hover:text-accent"
                    title="Send to Trade Desk — sizes an order and waits for your approval"
                  >
                    → Trade Desk
                  </button>
                </li>
              ))}
            </ul>
          )}

          {watchlist.length > 0 && (
            <div className="space-y-1">
              <p className="text-[11px] text-muted uppercase tracking-wide">
                Closest to firing
              </p>
              <ul className="flex flex-wrap gap-x-4 gap-y-1">
                {watchlist.map((r) => (
                  <li key={r.ticker} className="text-xs text-muted">
                    <Link href={`/stock/${r.ticker}`} className="hover:text-accent">
                      {r.ticker.replace(".NS", "")}
                    </Link>{" "}
                    {pct(r.prob_up, 1)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="text-[11px] text-muted leading-snug">
            Honest note: the hit-rate shown is the measured walk-forward precision of fired
            calls; most of it is equity drift — only the edge over the base rate is skill.
            As of {scan.rows[0]?.asof ?? "latest close"}.
          </p>
        </>
      )}
    </section>
  );
}
