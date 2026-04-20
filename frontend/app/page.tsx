import Link from "next/link";
import { createClient } from "@/utils/supabase/server";
import HomeSearch from "@/components/HomeSearch";

const QUICK_PICKS = [
  "RELIANCE.NS",
  "TCS.NS",
  "INFY.NS",
  "HDFCBANK.NS",
  "ICICIBANK.NS",
  "SBIN.NS",
  "LT.NS",
  "ITC.NS",
];

export default async function Home() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <div className="mx-auto max-w-3xl space-y-10 pt-8">
      <section className="text-center space-y-5">
        <h1 className="text-4xl font-semibold tracking-tight">
          Predict Indian equities with a <span className="text-accent">hybrid AI ensemble</span>.
        </h1>
        <p className="text-muted">
          XGBoost + LightGBM + CatBoost + GRU stacked with a Ridge meta-learner,
          blended with a 4-model FinBERT sentiment ensemble. Conformal intervals,
          rolling calibration, and a ledger that scores every prediction.
        </p>

        <div className="panel text-left">
          <p className="text-xs uppercase text-muted mb-2 tracking-wide">Search a ticker</p>
          <HomeSearch requireAuth={!user} />
          <div className="mt-3 flex flex-wrap gap-1.5 text-xs">
            <span className="text-muted">Quick picks:</span>
            {QUICK_PICKS.map((s) => (
              <Link
                key={s}
                href={user ? `/stock/${s}` : `/login?next=/stock/${s}`}
                className="chip hover:text-accent hover:border-accent"
              >
                {s}
              </Link>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-center gap-3 pt-2">
          {user ? (
            <>
              <Link href="/dashboard" className="btn btn-primary">
                Open dashboard
              </Link>
              <Link href="/accuracy" className="btn">
                Global accuracy
              </Link>
            </>
          ) : (
            <>
              <Link href="/login" className="btn btn-primary">
                Sign in
              </Link>
              <Link href="/login?mode=signup" className="btn">
                Create account
              </Link>
            </>
          )}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3 text-sm">
        <Feature
          title="Hybrid ensemble"
          body="Direction-calibrated via isotonic regression on walk-forward OOF folds. Per-ticker Youden-J thresholds."
        />
        <Feature
          title="Live market feeds"
          body="yfinance OHLCV + macros, NSE option-chain cookie-dance, RSS / NewsAPI / Reddit headlines, FinBERT categorisation."
        />
        <Feature
          title="Accountable forecasts"
          body="Every prediction lands in a SQLite ledger. Rolling hit-rate, Brier, and ECE on every stock page."
        />
      </section>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="panel">
      <p className="font-semibold mb-1">{title}</p>
      <p className="text-muted">{body}</p>
    </div>
  );
}
