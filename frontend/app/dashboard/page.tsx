import { createClient } from "@/utils/supabase/server";
import { redirect } from "next/navigation";
import WatchlistsPanel from "./WatchlistsPanel";
import AlertsPanel from "./AlertsPanel";
import type { Alert, Watchlist } from "@/lib/types";

export default async function DashboardPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const [wlRes, tkRes, alRes] = await Promise.all([
    supabase.from("watchlists").select("*").order("created_at", { ascending: true }),
    supabase.from("watchlist_tickers").select("*").order("added_at", { ascending: true }),
    supabase.from("alerts").select("*").order("created_at", { ascending: false }),
  ]);

  const watchlists: Watchlist[] = (wlRes.data ?? []).map((w) => ({
    ...w,
    tickers: (tkRes.data ?? []).filter((t) => t.watchlist_id === w.id),
  }));
  const alerts: Alert[] = (alRes.data ?? []) as Alert[];

  return (
    <div className="space-y-8">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-muted text-sm">
          Signed in as <span className="text-fg">{user.email}</span>. Watchlists
          and alerts are scoped to your account via Supabase row-level security.
        </p>
      </header>

      <WatchlistsPanel initial={watchlists} />
      <AlertsPanel initial={alerts} />
    </div>
  );
}
