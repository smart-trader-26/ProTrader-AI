"use client";

import type { SwingSignal } from "@/lib/types";

/**
 * P4-1 — Honest 1-2 month "Swing Direction" signal.
 *
 * This is the model's most accurate, *validated* directional call. Unlike the
 * next-day P(up) (which is ~50/50 noise on liquid names and is now correctly
 * reported near 50%), this targets a ~30-trading-day horizon where a real,
 * out-of-sample-measured edge exists. Every number shown here is the persisted
 * walk-forward statistic — not an in-sample fit — so it never overpromises.
 */
export default function SwingSignalPanel({ signal }: { signal: SwingSignal | null }) {
  if (!signal || !signal.trained) {
    return (
      <div className="panel space-y-1">
        <p className="text-sm font-semibold">📅 Swing Direction (1–2 month)</p>
        <p className="text-xs text-muted">
          Model not trained yet. Run{" "}
          <code className="font-mono">DirectionalSignal(horizon=30).train()</code> to enable.
        </p>
      </div>
    );
  }

  const isUp = signal.signal === "UP";
  const pct = (x: number | null | undefined, d = 0) =>
    x == null ? "—" : `${(x * 100).toFixed(d)}%`;
  const probPct = signal.prob_up * 100;

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">📅 Swing Direction · {signal.horizon_days}-day (~1.5 mo)</p>
          <p className="text-xs text-muted">
            The model&apos;s most reliable directional call — validated out-of-sample.
            Long / neutral only.
          </p>
        </div>
        <span
          className={`rounded-full px-3 py-1 text-sm font-bold ${
            isUp ? "bg-bull/15 text-bull" : "bg-muted/15 text-muted"
          }`}
        >
          {isUp ? "▲ LEAN UP" : "● NEUTRAL"}
        </span>
      </div>

      {/* Calibrated probability bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-muted">Calibrated P(up over {signal.horizon_days}d)</span>
          <span className={isUp ? "text-bull font-semibold" : "text-fg"}>
            {probPct.toFixed(1)}%
          </span>
        </div>
        <div className="h-2 w-full rounded-full bg-border/40 overflow-hidden">
          <div
            className={`h-full ${isUp ? "bg-bull" : "bg-muted"}`}
            style={{ width: `${Math.min(100, Math.max(0, probPct))}%` }}
          />
        </div>
        {signal.tau_star != null && (
          <p className="text-[11px] text-muted">
            Fires UP only above {pct(signal.tau_star, 0)} conviction — otherwise neutral.
          </p>
        )}
      </div>

      {/* Honesty stats — measured out-of-sample */}
      <div className="grid grid-cols-3 gap-2 border-t border-border pt-3 text-center">
        <Stat
          label="Hit-rate when it fires"
          value={pct(signal.expected_hit_rate, 0)}
          tone="accent"
        />
        <Stat label="vs. always-up base" value={pct(signal.base_rate, 0)} />
        <Stat label="Fires this often" value={pct(signal.coverage, 0)} />
      </div>

      <p className="text-[11px] text-muted leading-snug">
        {signal.note}.{" "}
        {signal.expected_hit_rate != null && signal.base_rate != null && (
          <>
            Edge over drift:{" "}
            <span className={signal.expected_hit_rate > signal.base_rate ? "text-bull" : "text-bear"}>
              {((signal.expected_hit_rate - signal.base_rate) * 100).toFixed(1)} pp
            </span>
            .{" "}
          </>
        )}
        Honest note: most of the hit-rate is equity drift; only the edge over base is genuine skill.
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "accent" | "bull";
}) {
  const color = tone === "accent" ? "text-accent" : tone === "bull" ? "text-bull" : "text-fg";
  return (
    <div>
      <p className={`text-base font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-muted leading-tight">{label}</p>
    </div>
  );
}
