"use client";

import { apiGet, apiPost } from "@/lib/api-client";
import type {
  BrokerStatus,
  ProposalCreateResponse,
  SkippedTicker,
  TradeProposal,
} from "@/lib/types";
import TickerPicker from "@/components/TickerPicker";
import { useCallback, useEffect, useState } from "react";

/**
 * P5 — Trade Desk: analyze → propose → YOU approve → execute.
 *
 * Safety model (mirrors the backend):
 *   • Orders are proposals until you explicitly approve each one.
 *   • Approvals are simulated (DRY-RUN) unless LIVE_TRADING=1 + Kite keys are
 *     configured on the backend — the banner always shows which mode you're in.
 *   • Every placed order is also logged to the accuracy ledger, so live
 *     results stay comparable with the paper book and the backtest.
 */
export default function TradeDeskPanel() {
  const [broker, setBroker] = useState<BrokerStatus | null>(null);
  const [proposals, setProposals] = useState<TradeProposal[]>([]);
  const [skipped, setSkipped] = useState<SkippedTicker[]>([]);
  const [tickers, setTickers] = useState<string[]>([]);
  const [pickerValue, setPickerValue] = useState("");
  const [capital, setCapital] = useState("25000");
  const [analyzing, setAnalyzing] = useState(false);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    try {
      const [st, rows] = await Promise.all([
        apiGet<BrokerStatus>("/api/v1/trade/broker/status"),
        apiGet<TradeProposal[]>("/api/v1/trade/proposals?limit=50"),
      ]);
      setBroker(st);
      setProposals(rows);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  const analyze = useCallback(
    async (list: string[]) => {
      if (list.length === 0) return;
      setAnalyzing(true);
      setErr(null);
      setSkipped([]);
      try {
        const res = await apiPost<ProposalCreateResponse>("/api/v1/trade/proposals", {
          tickers: list,
          capital_per_slot: Number(capital) > 0 ? Number(capital) : null,
        });
        setSkipped(res.skipped);
        setTickers([]);
        await loadAll();
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setAnalyzing(false);
      }
    },
    [capital, loadAll],
  );

  // The Swing Screener's "→ Trade Desk" buttons dispatch this event.
  useEffect(() => {
    const onPropose = (ev: Event) => {
      const ticker = (ev as CustomEvent<{ ticker: string }>).detail?.ticker;
      if (ticker) void analyze([ticker]);
    };
    window.addEventListener("protrader:propose", onPropose);
    return () => window.removeEventListener("protrader:propose", onPropose);
  }, [analyze]);

  function addTicker(sym: string) {
    const s = sym.trim().toUpperCase();
    if (!s) return;
    const full = s.endsWith(".NS") || s.endsWith(".BO") ? s : `${s}.NS`;
    setTickers((xs) => (xs.includes(full) ? xs : [...xs, full]));
    setPickerValue("");
  }

  async function decide(p: TradeProposal, action: "approve" | "reject") {
    if (action === "approve") {
      const mode = broker?.live_trading ? "LIVE — real money at Zerodha" : "DRY-RUN (simulated)";
      const ok = window.confirm(
        `${mode}\n\nBUY ${p.qty} × ${p.ticker.replace(".NS", "")} ≈ ₹${inr(p.capital_required)}\n` +
          `Stop-loss: ₹${inr(p.stop_price)} (−10%) · Exit by: ${p.exit_by}\n\nPlace this order?`,
      );
      if (!ok) return;
    }
    setBusyId(p.id);
    setErr(null);
    try {
      const updated = await apiPost<TradeProposal>(
        `/api/v1/trade/proposals/${p.id}/${action}`,
        {},
      );
      setProposals((xs) => xs.map((x) => (x.id === updated.id ? updated : x)));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      await loadAll(); // FAILED status is persisted server-side — resync
    } finally {
      setBusyId(null);
    }
  }

  const pending = proposals.filter((p) => p.status === "PROPOSED");
  const history = proposals.filter((p) => p.status !== "PROPOSED").slice(0, 8);

  return (
    <section className="panel space-y-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">💼 Trade Desk · approve-before-execute</h2>
          <p className="text-xs text-muted">
            Runs the validated 30-day swing signal on your picks, sizes the order, and
            waits for your approval. Nothing is ever placed automatically.
          </p>
        </div>
        {broker && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-bold ${
              broker.live_trading
                ? "bg-bear/15 text-bear"
                : "bg-accent/15 text-accent"
            }`}
            title={broker.note}
          >
            {broker.live_trading ? "🔴 LIVE · ZERODHA KITE" : "🛡 DRY-RUN · SIMULATED"}
          </span>
        )}
      </div>
      {broker && <p className="text-[11px] text-muted -mt-2">{broker.note}</p>}

      {/* ── Analyze form ── */}
      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-[220px] flex-1">
          <label className="mb-1 block text-[11px] text-muted">Add stocks to analyze</label>
          <TickerPicker
            value={pickerValue}
            onChange={setPickerValue}
            onCommit={addTicker}
            placeholder="Search ticker (e.g. RELIANCE)"
          />
        </div>
        <div>
          <label className="mb-1 block text-[11px] text-muted">₹ per position</label>
          <input
            type="number"
            min={1000}
            step={1000}
            value={capital}
            onChange={(e) => setCapital(e.target.value)}
            className="w-28 rounded border border-border bg-transparent px-2 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={() => void analyze(tickers)}
          disabled={analyzing || tickers.length === 0}
          className="rounded bg-accent px-4 py-1.5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40"
        >
          {analyzing ? "Analyzing…" : `Analyze ${tickers.length || ""}`.trim()}
        </button>
      </div>

      {tickers.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {tickers.map((t) => (
            <li
              key={t}
              className="flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs"
            >
              {t.replace(".NS", "")}
              <button
                onClick={() => setTickers((xs) => xs.filter((x) => x !== t))}
                className="text-muted hover:text-bear"
                aria-label={`remove ${t}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {err && <p className="text-xs text-bear">⚠ {err}</p>}

      {/* ── Honest skip reasons ── */}
      {skipped.length > 0 && (
        <div className="space-y-1 rounded border border-border/60 p-2">
          <p className="text-[11px] font-semibold text-muted uppercase tracking-wide">
            Analyzed but not proposed
          </p>
          {skipped.map((s) => (
            <p key={s.ticker} className="text-xs text-muted">
              <span className="font-semibold text-fg">{s.ticker.replace(".NS", "")}</span>{" "}
              — {s.reason}
            </p>
          ))}
        </div>
      )}

      {/* ── Pending approvals ── */}
      {pending.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] font-semibold text-muted uppercase tracking-wide">
            Awaiting your approval
          </p>
          {pending.map((p) => (
            <div key={p.id} className="rounded-lg border border-accent/40 p-3 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-bull/15 px-2 py-0.5 text-xs font-bold text-bull">
                    ▲ BUY
                  </span>
                  <span className="text-sm font-semibold">{p.ticker.replace(".NS", "")}</span>
                  <span className="text-xs text-muted">
                    P(up {30}d) = {(p.prob_up * 100).toFixed(1)}%
                    {p.expected_hit_rate != null &&
                      ` · historically right ${(p.expected_hit_rate * 100).toFixed(0)}% when fired`}
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => void decide(p, "approve")}
                    disabled={busyId === p.id}
                    className="rounded bg-bull px-3 py-1 text-xs font-bold text-white hover:opacity-90 disabled:opacity-40"
                  >
                    {busyId === p.id ? "Placing…" : broker?.live_trading ? "Approve · LIVE" : "Approve · dry-run"}
                  </button>
                  <button
                    onClick={() => void decide(p, "reject")}
                    disabled={busyId === p.id}
                    className="rounded border border-border px-3 py-1 text-xs text-muted hover:text-bear disabled:opacity-40"
                  >
                    Reject
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-5">
                <Field label="Entry (ref.)" value={`₹${inr(p.entry_price)}`} />
                <Field label="Quantity" value={String(p.qty)} />
                <Field label="Capital" value={`₹${inr(p.capital_required)}`} />
                <Field label="Stop-loss" value={p.stop_price ? `₹${inr(p.stop_price)} (−10%)` : "—"} />
                <Field label="Exit by" value={p.exit_by} />
              </div>
              <p className="text-[11px] text-muted">{p.note}</p>
            </div>
          ))}
        </div>
      )}

      {pending.length === 0 && !analyzing && (
        <p className="text-xs text-muted">
          No pending proposals. Add stocks above (or send a fired name from the Swing
          Screener) to analyze.
        </p>
      )}

      {/* ── History ── */}
      {history.length > 0 && (
        <div className="space-y-1 border-t border-border pt-2">
          <p className="text-[11px] font-semibold text-muted uppercase tracking-wide">Recent</p>
          {history.map((p) => (
            <div key={p.id} className="flex flex-wrap items-center gap-2 text-xs">
              <StatusBadge status={p.status} dryRun={p.dry_run} />
              <span className="font-semibold">{p.ticker.replace(".NS", "")}</span>
              <span className="text-muted">
                {p.qty} × ₹{inr(p.entry_price)}
              </span>
              {p.broker_order_id && (
                <span className="font-mono text-[10px] text-muted">{p.broker_order_id}</span>
              )}
              {p.status === "FAILED" && <span className="text-bear">{p.note}</span>}
            </div>
          ))}
        </div>
      )}

      <p className="text-[11px] text-muted leading-snug border-t border-border pt-2">
        Honesty contract: every approved order is also logged to the accuracy ledger and
        resolved against the real price at its exit date — so live, paper and backtest
        results stay comparable on the Accuracy page. Exits: a 10% disaster stop is
        placed server-side (GTT when live); otherwise the position is held to the signal
        horizon and should be exited by the date shown.
      </p>
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] text-muted">{label}</p>
      <p className="font-semibold">{value}</p>
    </div>
  );
}

function StatusBadge({
  status,
  dryRun,
}: {
  status: TradeProposal["status"];
  dryRun: boolean | null;
}) {
  const styles: Record<string, string> = {
    PLACED: "bg-bull/15 text-bull",
    REJECTED: "bg-muted/15 text-muted",
    FAILED: "bg-bear/15 text-bear",
    EXPIRED: "bg-muted/15 text-muted",
    PROPOSED: "bg-accent/15 text-accent",
  };
  const label =
    status === "PLACED" ? (dryRun === false ? "PLACED · LIVE" : "PLACED · DRY") : status;
  return (
    <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${styles[status]}`}>
      {label}
    </span>
  );
}

function inr(x: number | null | undefined): string {
  if (x == null) return "—";
  return x.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}
