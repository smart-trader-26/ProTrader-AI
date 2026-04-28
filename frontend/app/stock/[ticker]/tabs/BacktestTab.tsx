"use client";

import { useState } from "react";
import Stat from "@/components/Stat";
import LineChart from "../charts/LineChart";
import type { BacktestResult, PredictionJobAccepted } from "@/lib/types";
import { apiPost, waitForJob } from "@/lib/api-client";
import { formatINR, formatPercent, formatRatio, toneFor } from "@/lib/format";

type Strategy = "ma_crossover" | "momentum";

function defaultStart() {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 3);
  return d.toISOString().slice(0, 10);
}

export default function BacktestTab({ ticker }: { ticker: string }) {
  const [strategy, setStrategy] = useState<Strategy>("ma_crossover");
  const [initialCapital, setInitialCapital] = useState<number>(100_000);
  const [includeCosts, setIncludeCosts] = useState(true);
  const [startDate, setStartDate] = useState<string>(defaultStart());
  const [endDate, setEndDate] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setPhase("queued");
    try {
      const accepted = await apiPost<PredictionJobAccepted>(
        `/api/v1/stocks/${ticker}/backtest`,
        {
          strategy,
          initial_capital: initialCapital,
          include_costs: includeCosts,
          start: startDate || undefined,
          end: endDate || undefined,
        },
      );
      setPhase("running");
      const r = await waitForJob<BacktestResult>(accepted.job_id, {
        timeoutMs: 240_000,
        onPhase: (p) => setPhase(p),
      });
      setResult(r);
      setPhase("done");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setPhase(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <section className="panel space-y-3">
        <div>
          <h2 className="text-lg font-semibold">Strategy backtest</h2>
          <p className="text-xs text-muted">
            NSE line-item costs (brokerage + STT + exchange + SEBI + stamp + GST) applied when enabled.
          </p>
        </div>
        <form onSubmit={run} className="grid gap-3 sm:grid-cols-4 items-end">
          <label>
            <span className="block text-xs uppercase text-muted mb-1">Strategy</span>
            <select
              className="input"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as Strategy)}
            >
              <option value="ma_crossover">MA crossover</option>
              <option value="momentum">Momentum</option>
            </select>
          </label>
          <label>
            <span className="block text-xs uppercase text-muted mb-1">Start date</span>
            <input
              type="date"
              className="input"
              value={startDate}
              max={endDate || undefined}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </label>
          <label>
            <span className="block text-xs uppercase text-muted mb-1">End date (blank = today)</span>
            <input
              type="date"
              className="input"
              value={endDate}
              min={startDate || undefined}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </label>
          <label>
            <span className="block text-xs uppercase text-muted mb-1">Initial capital (₹)</span>
            <input
              type="number"
              className="input"
              min={10_000}
              step={10_000}
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
            />
          </label>
          <label className="flex items-center gap-2 text-sm pt-5 sm:pt-0">
            <input
              type="checkbox"
              checked={includeCosts}
              onChange={(e) => setIncludeCosts(e.target.checked)}
            />
            Include NSE costs
          </label>
          <button disabled={busy} className="btn btn-primary sm:col-start-4">
            {busy ? (phase ?? "…") : "Run backtest"}
          </button>
        </form>
        {err && <p className="text-sm text-bear">{err}</p>}
      </section>

      {result && (
        <>
          {/* Dim all metrics when there aren't enough trades for reliable statistics */}
          <section className={`panel grid gap-4 sm:grid-cols-4 ${result.metrics.n_trades < 50 ? "opacity-40 pointer-events-none" : ""}`}>
            <Stat
              label="Total return"
              value={formatPercent(result.metrics.total_return_pct, 2)}
              tone={result.metrics.n_trades < 50 ? "muted" : toneFor(result.metrics.total_return_pct)}
            />
            <Stat
              label="CAGR"
              value={formatPercent(result.metrics.cagr_pct, 2)}
              tone={result.metrics.n_trades < 50 ? "muted" : toneFor(result.metrics.cagr_pct)}
            />
            <Stat
              label="Sharpe"
              value={formatRatio(result.metrics.sharpe)}
              tone={
                result.metrics.n_trades < 50 ? "muted"
                : result.metrics.sharpe >= 1.2 ? "bull"
                : result.metrics.sharpe < 0 ? "bear"
                : "muted"
              }
              hint={result.metrics.n_trades < 50 ? `Only ${result.metrics.n_trades} trades — need ≥50` : "Target ≥ 1.2"}
            />
            <Stat
              label="Max drawdown"
              value={formatPercent(result.metrics.max_drawdown_pct, 2)}
              tone={result.metrics.n_trades < 50 ? "muted" : "bear"}
            />
          </section>

          <section className={`panel grid gap-4 sm:grid-cols-4 ${result.metrics.n_trades < 50 ? "opacity-40 pointer-events-none" : ""}`}>
            <Stat label="Sortino" value={formatRatio(result.metrics.sortino)} />
            <Stat label="Calmar" value={formatRatio(result.metrics.calmar)} />
            <Stat
              label="Profit factor"
              value={formatRatio(result.metrics.profit_factor)}
              tone={
                result.metrics.n_trades < 50 || result.metrics.profit_factor == null
                  ? "muted"
                  : result.metrics.profit_factor > 1.5
                  ? "bull"
                  : result.metrics.profit_factor < 1
                  ? "bear"
                  : "muted"
              }
            />
            <Stat
              label="Win rate"
              value={formatPercent(result.metrics.win_rate_pct, 1)}
            />
            <Stat
              label="Avg win"
              value={formatPercent(result.metrics.avg_win_pct, 2)}
              tone="bull"
            />
            <Stat
              label="Avg loss"
              value={formatPercent(result.metrics.avg_loss_pct, 2)}
              tone="bear"
            />
            <Stat
              label="Expectancy"
              value={formatPercent(result.metrics.expectancy_pct, 3)}
              tone={toneFor(result.metrics.expectancy_pct)}
            />
            <Stat label="Trades" value={String(result.metrics.n_trades)} />
          </section>

          {result.metrics.validity_warning && (
            <div className="rounded border border-red-400 bg-red-400/10 px-4 py-3 text-sm text-red-400">
              ⚠ {result.metrics.validity_warning}
            </div>
          )}

          <section className="panel">
            <div className="flex items-baseline justify-between mb-2">
              <p className="text-sm font-semibold">Equity curve vs Buy &amp; Hold</p>
              <p className="text-xs text-muted">
                Final: {formatINR(result.final_equity)} · initial {formatINR(result.initial_capital)}
              </p>
            </div>
            <LineChart
              height={260}
              series={[
                {
                  label: `Strategy (${result.strategy})`,
                  color: "#2dd4bf",
                  data: result.equity_curve.map((p) => ({ x: p.date, y: p.equity })),
                },
                {
                  label: "Buy & Hold",
                  color: "#8b98a5",
                  data: result.benchmark_equity_curve.map((p) => ({ x: p.date, y: p.equity })),
                },
              ]}
            />
          </section>

          {result.metrics.dm_pvalue != null && (
            <section className="panel text-sm">
              <p className="font-semibold mb-1">Diebold-Mariano (vs Buy &amp; Hold)</p>
              <p className="text-muted">
                p-value ={" "}
                <span
                  className={
                    result.metrics.dm_pvalue < 0.05
                      ? "text-bull font-medium"
                      : "text-muted"
                  }
                >
                  {result.metrics.dm_pvalue.toFixed(4)}
                </span>{" "}
                — {result.metrics.dm_pvalue < 0.05
                  ? "strategy forecasts are statistically better."
                  : "no significant edge over buy-and-hold at the 5% level."}
              </p>
            </section>
          )}
        </>
      )}
    </div>
  );
}
