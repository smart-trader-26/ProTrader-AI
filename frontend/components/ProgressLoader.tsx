"use client";

import { useEffect, useState } from "react";

interface Props {
  active: boolean;
  phase: string | null;
  startedAt: number | null;
  estimatedSeconds?: number;
  steps?: { key: string; label: string }[];
  error?: string | null;
}

const DEFAULT_STEPS = [
  { key: "queued", label: "Queued" },
  { key: "started", label: "Loading data" },
  { key: "running", label: "Fitting ensemble" },
  { key: "calibrating", label: "Calibration & SHAP" },
  { key: "succeeded", label: "Done" },
];

/**
 * Wide progress bar with phase, elapsed timer and step pips. Used during
 * predict / backtest jobs so the user has a visible signal that the long
 * job is alive and roughly where it is.
 */
export default function ProgressLoader({
  active,
  phase,
  startedAt,
  estimatedSeconds = 45,
  steps = DEFAULT_STEPS,
  error,
}: Props) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [active]);

  if (!active && !error) return null;

  const elapsed = startedAt ? Math.max(0, (now - startedAt) / 1000) : 0;
  const baseFrac = startedAt ? Math.min(0.98, elapsed / estimatedSeconds) : 0;

  const phaseIdx = phaseToIndex(phase, steps);
  const stepFrac = phaseIdx >= 0 ? phaseIdx / Math.max(1, steps.length - 1) : 0;
  const fraction = error ? 1 : Math.max(baseFrac, stepFrac);

  const minutes = Math.floor(elapsed / 60);
  const seconds = Math.floor(elapsed % 60).toString().padStart(2, "0");

  return (
    <div className="panel">
      <div className="flex items-center justify-between text-xs uppercase tracking-wide text-muted mb-2">
        <span>{error ? "Failed" : labelFor(phase, steps) ?? "Working…"}</span>
        <span className="tabular-nums">
          {minutes}:{seconds} elapsed
          {!error && startedAt && elapsed < estimatedSeconds && (
            <> · ~{Math.max(0, Math.round(estimatedSeconds - elapsed))}s left</>
          )}
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-border overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-200 ${
            error ? "bg-bear" : "bg-accent"
          }`}
          style={{ width: `${(fraction * 100).toFixed(1)}%` }}
        />
      </div>
      <ol className="mt-3 grid grid-cols-5 gap-2 text-[11px]">
        {steps.map((s, i) => {
          const reached = phaseIdx >= i;
          return (
            <li key={s.key} className="flex flex-col items-center gap-1">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  error && i === phaseIdx
                    ? "bg-bear"
                    : reached
                    ? "bg-accent"
                    : "bg-border"
                }`}
              />
              <span
                className={
                  reached ? "text-fg text-center" : "text-muted text-center"
                }
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>
      {error && <p className="text-sm text-bear mt-3">{error}</p>}
    </div>
  );
}

function phaseToIndex(phase: string | null, steps: { key: string; label: string }[]): number {
  if (!phase) return 0;
  const p = phase.toLowerCase();
  // Map common Celery / in-process states onto the 5 visible steps.
  if (p === "queued" || p === "pending") return 0;
  if (p === "started" || p === "running" || p === "fetching") return 2;
  if (p === "calibrating" || p === "scoring") return 3;
  if (p === "succeeded" || p === "done" || p === "finished") return steps.length - 1;
  if (p === "failed") return Math.max(0, steps.length - 1);
  // Unknown phase — keep some forward motion.
  return 1;
}

function labelFor(phase: string | null, steps: { key: string; label: string }[]) {
  const idx = phaseToIndex(phase, steps);
  return steps[idx]?.label;
}
