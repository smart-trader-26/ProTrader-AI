"use client";

import { createClient } from "@/utils/supabase/client";
import type { PredictionBundle, PredictionJobAccepted } from "@/lib/types";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function authFetch(path: string, init: RequestInit = {}) {
  const supabase = createClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (session?.access_token) {
    headers.set("Authorization", `Bearer ${session.access_token}`);
  }
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}

export default function PredictPanel({ ticker }: { ticker: string }) {
  const [horizon, setHorizon] = useState(10);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [bundle, setBundle] = useState<PredictionBundle | null>(null);
  const [phase, setPhase] = useState<string | null>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setBundle(null);
    setPhase("queued");

    try {
      const res = await authFetch(`/api/v1/stocks/${ticker}/predict`, {
        method: "POST",
        body: JSON.stringify({ horizon_days: horizon }),
      });
      if (res.status === 429) throw new Error("Rate limited — try again in a minute.");
      if (!res.ok) throw new Error(`Enqueue failed (${res.status})`);
      const accepted: PredictionJobAccepted = await res.json();
      setPhase("running");

      const deadline = Date.now() + 180_000;
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2000));
        const pollRes = await authFetch(`/api/v1/jobs/${accepted.job_id}`);
        if (!pollRes.ok) throw new Error(`Poll failed (${pollRes.status})`);
        const job = await pollRes.json();
        if (job.status === "succeeded") {
          setBundle(job.result as PredictionBundle);
          setPhase("done");
          return;
        }
        if (job.status === "failed") {
          throw new Error(job.error ?? "prediction failed");
        }
      }
      throw new Error("Timed out waiting for prediction (>3 min).");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setPhase(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold">Predict</h2>
      <form onSubmit={run} className="panel flex items-end gap-3">
        <label className="flex-1">
          <span className="block text-xs uppercase text-muted mb-1">Horizon (days)</span>
          <input
            type="number"
            min={1}
            max={60}
            className="input"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </label>
        <button disabled={busy} className="btn btn-primary">
          {busy ? (phase ?? "…") : "Run prediction"}
        </button>
      </form>

      {err && <p className="text-sm text-bear">{err}</p>}

      {bundle && (
        <div className="panel grid gap-4 sm:grid-cols-4">
          <Stat label="Predicted price" value={`₹${bundle.pred_price.toFixed(2)}`} />
          <Stat
            label="Expected return"
            value={`${(bundle.pred_return * 100).toFixed(2)}%`}
            tone={bundle.pred_return >= 0 ? "bull" : "bear"}
          />
          <Stat
            label="P(up)"
            value={`${(bundle.prob_up * 100).toFixed(1)}%`}
            tone={bundle.prob_up >= 0.5 ? "bull" : "bear"}
          />
          <Stat
            label={`${Math.round(bundle.confidence_level * 100)}% interval`}
            value={
              bundle.ci_low != null && bundle.ci_high != null
                ? `₹${bundle.ci_low.toFixed(0)} – ₹${bundle.ci_high.toFixed(0)}`
                : "—"
            }
          />
        </div>
      )}
    </section>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "bull" | "bear";
}) {
  return (
    <div>
      <p className="text-xs uppercase text-muted">{label}</p>
      <p className={`text-lg font-semibold ${tone ? `text-${tone}` : ""}`}>{value}</p>
    </div>
  );
}
