"use client";

import { createClient } from "@/utils/supabase/client";
import type { Watchlist } from "@/lib/types";
import Link from "next/link";
import { useState } from "react";
import TickerPicker from "@/components/TickerPicker";

export default function WatchlistsPanel({ initial }: { initial: Watchlist[] }) {
  const [watchlists, setWatchlists] = useState<Watchlist[]>(initial);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const supabase = createClient();
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) throw new Error("Not signed in");
      const { data, error } = await supabase
        .from("watchlists")
        .insert({ name: name.trim(), user_id: user.id })
        .select()
        .single();
      if (error) throw error;
      setWatchlists((xs) => [...xs, { ...data, tickers: [] }]);
      setName("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function addTicker(wlId: number, ticker: string) {
    const sym = ticker.trim().toUpperCase();
    if (!sym) return;
    const supabase = createClient();
    const { error } = await supabase
      .from("watchlist_tickers")
      .upsert({ watchlist_id: wlId, ticker: sym }, { onConflict: "watchlist_id,ticker" });
    if (error) {
      setErr(error.message);
      return;
    }
    setWatchlists((xs) =>
      xs.map((w) =>
        w.id === wlId
          ? {
              ...w,
              tickers: [
                ...w.tickers.filter((t) => t.ticker !== sym),
                { watchlist_id: wlId, ticker: sym, added_at: new Date().toISOString() },
              ],
            }
          : w,
      ),
    );
  }

  async function removeTicker(wlId: number, ticker: string) {
    const supabase = createClient();
    const { error } = await supabase
      .from("watchlist_tickers")
      .delete()
      .eq("watchlist_id", wlId)
      .eq("ticker", ticker);
    if (error) {
      setErr(error.message);
      return;
    }
    setWatchlists((xs) =>
      xs.map((w) =>
        w.id === wlId ? { ...w, tickers: w.tickers.filter((t) => t.ticker !== ticker) } : w,
      ),
    );
  }

  async function deleteList(wlId: number) {
    const supabase = createClient();
    const { error } = await supabase.from("watchlists").delete().eq("id", wlId);
    if (error) {
      setErr(error.message);
      return;
    }
    setWatchlists((xs) => xs.filter((w) => w.id !== wlId));
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Watchlists</h2>
      </div>
      <form onSubmit={create} className="flex gap-2">
        <input
          className="input"
          placeholder="New watchlist name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button disabled={busy} className="btn btn-primary whitespace-nowrap">
          Add list
        </button>
      </form>
      {err && <p className="text-sm text-bear">{err}</p>}
      {watchlists.length === 0 ? (
        <p className="text-muted text-sm">No watchlists yet.</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {watchlists.map((w) => (
            <WatchlistCard
              key={w.id}
              wl={w}
              onAdd={(t) => addTicker(w.id, t)}
              onRemove={(t) => removeTicker(w.id, t)}
              onDelete={() => deleteList(w.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function WatchlistCard({
  wl,
  onAdd,
  onRemove,
  onDelete,
}: {
  wl: Watchlist;
  onAdd: (t: string) => void;
  onRemove: (t: string) => void;
  onDelete: () => void;
}) {
  const [t, setT] = useState("");
  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">{wl.name}</h3>
        <button onClick={onDelete} className="text-xs text-muted hover:text-bear">
          delete
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {wl.tickers.length === 0 && <span className="text-xs text-muted">No tickers.</span>}
        {wl.tickers.map((tk) => (
          <div key={tk.ticker} className="chip flex items-center gap-2">
            <Link href={`/stock/${tk.ticker}`} className="hover:text-accent">
              {tk.ticker}
            </Link>
            <button
              onClick={() => onRemove(tk.ticker)}
              className="text-muted hover:text-bear"
              aria-label={`remove ${tk.ticker}`}
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          // No-op: `onCommit` from TickerPicker handles validated picks.
        }}
        className="flex gap-2"
      >
        <TickerPicker
          value={t}
          onChange={setT}
          onCommit={(sym) => {
            onAdd(sym);
            setT("");
          }}
          placeholder="Add ticker (e.g. RELIANCE)"
        />
      </form>
    </div>
  );
}
