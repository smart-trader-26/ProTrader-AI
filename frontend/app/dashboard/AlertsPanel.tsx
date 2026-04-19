"use client";

import { createClient } from "@/utils/supabase/client";
import type { Alert, AlertKind } from "@/lib/types";
import { useState } from "react";

const KINDS: { value: AlertKind; label: string }[] = [
  { value: "price_above", label: "Price above" },
  { value: "price_below", label: "Price below" },
  { value: "pct_move", label: "Percent move" },
];

export default function AlertsPanel({ initial }: { initial: Alert[] }) {
  const [alerts, setAlerts] = useState<Alert[]>(initial);
  const [ticker, setTicker] = useState("");
  const [kind, setKind] = useState<AlertKind>("price_above");
  const [threshold, setThreshold] = useState("");
  const [err, setErr] = useState<string | null>(null);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    try {
      const supabase = createClient();
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) throw new Error("Not signed in");
      const num = Number(threshold);
      if (!Number.isFinite(num)) throw new Error("Threshold must be a number");
      const { data, error } = await supabase
        .from("alerts")
        .insert({
          user_id: user.id,
          ticker: ticker.trim().toUpperCase(),
          kind,
          threshold: num,
          active: true,
        })
        .select()
        .single();
      if (error) throw error;
      setAlerts((xs) => [data as Alert, ...xs]);
      setTicker("");
      setThreshold("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function toggle(a: Alert) {
    const supabase = createClient();
    // Re-arming (active: true) also clears triggered_at — matches backend.
    const patch = a.active
      ? { active: false }
      : { active: true, triggered_at: null };
    const { data, error } = await supabase
      .from("alerts")
      .update(patch)
      .eq("id", a.id)
      .select()
      .single();
    if (error) {
      setErr(error.message);
      return;
    }
    setAlerts((xs) => xs.map((x) => (x.id === a.id ? (data as Alert) : x)));
  }

  async function remove(id: number) {
    const supabase = createClient();
    const { error } = await supabase.from("alerts").delete().eq("id", id);
    if (error) {
      setErr(error.message);
      return;
    }
    setAlerts((xs) => xs.filter((x) => x.id !== id));
  }

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold">Alerts</h2>
      <form onSubmit={create} className="panel grid gap-2 sm:grid-cols-[1fr_1fr_1fr_auto]">
        <input
          className="input"
          placeholder="Ticker"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          required
        />
        <select
          className="input"
          value={kind}
          onChange={(e) => setKind(e.target.value as AlertKind)}
        >
          {KINDS.map((k) => (
            <option key={k.value} value={k.value}>
              {k.label}
            </option>
          ))}
        </select>
        <input
          className="input"
          placeholder="Threshold"
          value={threshold}
          onChange={(e) => setThreshold(e.target.value)}
          required
        />
        <button className="btn btn-primary">Add</button>
      </form>
      {err && <p className="text-sm text-bear">{err}</p>}
      {alerts.length === 0 ? (
        <p className="text-muted text-sm">No alerts yet.</p>
      ) : (
        <ul className="space-y-2">
          {alerts.map((a) => (
            <li key={a.id} className="panel flex items-center justify-between">
              <div className="flex items-center gap-3 text-sm">
                <span className="font-medium">{a.ticker}</span>
                <span className="text-muted">
                  {KINDS.find((k) => k.value === a.kind)?.label} {a.threshold}
                </span>
                {a.triggered_at && (
                  <span className="chip text-bull">
                    triggered {new Date(a.triggered_at).toLocaleString()}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => toggle(a)} className="btn text-xs">
                  {a.active ? "Disable" : "Re-arm"}
                </button>
                <button onClick={() => remove(a.id)} className="text-xs text-muted hover:text-bear">
                  delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
