import { apiFetch, ApiError } from "@/lib/api";
import Stat from "@/components/Stat";
import type { Fundamentals } from "@/lib/types";
import { formatCompactINR, formatPercent, formatRatio, toneFor } from "@/lib/format";

type InfoDict = Record<string, unknown>;

/** Key/value pairs we surface from the raw info dict, in display order. */
const INFO_KEYS: Array<[string, string]> = [
  ["longName", "Company"],
  ["sector", "Sector"],
  ["industry", "Industry"],
  ["website", "Website"],
  ["country", "Country"],
  ["fullTimeEmployees", "Employees"],
  ["beta", "Beta"],
  ["fiftyTwoWeekHigh", "52W High"],
  ["fiftyTwoWeekLow", "52W Low"],
  ["averageVolume", "Avg volume"],
  ["sharesOutstanding", "Shares outstanding"],
  ["floatShares", "Float"],
  ["bookValue", "Book value"],
  ["earningsGrowth", "Earnings growth"],
  ["returnOnAssets", "Return on assets"],
];

export default async function FundamentalsTab({ ticker }: { ticker: string }) {
  const [f, info] = await Promise.all([
    safe<Fundamentals>(`/api/v1/stocks/${ticker}/fundamentals`),
    safe<InfoDict>(`/api/v1/stocks/${ticker}/info`),
  ]);

  if (!f) {
    return (
      <p className="panel text-sm text-muted">
        Fundamentals unavailable — upstream may be rate-limited.
      </p>
    );
  }

  const hasAny =
    f.market_cap != null ||
    f.pe_ratio != null ||
    f.price_to_book != null ||
    f.roe != null ||
    f.debt_to_equity != null;

  return (
    <div className="space-y-5">
      <section className="panel">
        <p className="text-sm font-semibold mb-3">Valuation & profitability</p>
        {!hasAny ? (
          <p className="text-sm text-muted">No ratios returned.</p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-4">
            <Stat label="Market cap" value={formatCompactINR(f.market_cap)} />
            <Stat label="P/E" value={formatRatio(f.pe_ratio)} />
            <Stat label="Forward P/E" value={formatRatio(f.forward_pe)} />
            <Stat label="PEG" value={formatRatio(f.peg_ratio)} />
            <Stat label="Price / Book" value={formatRatio(f.price_to_book)} />
            <Stat
              label="Debt / Equity"
              value={formatRatio(f.debt_to_equity)}
              tone={
                f.debt_to_equity == null
                  ? "muted"
                  : f.debt_to_equity < 1
                  ? "bull"
                  : f.debt_to_equity > 2
                  ? "bear"
                  : "muted"
              }
            />
            <Stat
              label="ROE"
              value={f.roe != null ? formatPercent(f.roe * 100, 1) : "—"}
              tone={f.roe != null ? (f.roe > 0.15 ? "bull" : f.roe < 0 ? "bear" : "muted") : "muted"}
            />
            <Stat
              label="Profit margin"
              value={f.profit_margin != null ? formatPercent(f.profit_margin * 100, 1) : "—"}
              tone={toneFor(f.profit_margin)}
            />
          </div>
        )}
      </section>

      <section className="panel">
        <p className="text-sm font-semibold mb-3">Growth & cash</p>
        <div className="grid gap-4 sm:grid-cols-4">
          <Stat
            label="Revenue growth"
            value={f.revenue_growth != null ? formatPercent(f.revenue_growth * 100, 1) : "—"}
            tone={toneFor(f.revenue_growth)}
          />
          <Stat label="Free cashflow" value={formatCompactINR(f.free_cashflow)} />
          <Stat label="Target price" value={formatCompactINR(f.target_price)} />
          <Stat
            label="Dividend yield"
            value={f.dividend_yield != null ? formatPercent(f.dividend_yield * 100, 2) : "—"}
          />
        </div>
      </section>

      {info && (
        <section className="panel">
          <p className="text-sm font-semibold mb-3">Company info</p>
          <div className="grid gap-x-6 gap-y-2 sm:grid-cols-2 text-sm">
            {INFO_KEYS.map(([k, label]) => {
              const v = info[k];
              if (v == null || v === "") return null;
              return (
                <div key={k} className="flex justify-between gap-3 border-b border-border/40 pb-1">
                  <span className="text-muted">{label}</span>
                  <span className="font-medium text-right truncate max-w-[60%]">
                    {formatInfoValue(k, v)}
                  </span>
                </div>
              );
            })}
          </div>
          {typeof info.longBusinessSummary === "string" && (
            <details className="mt-4 text-sm text-muted">
              <summary className="cursor-pointer text-fg">Business summary</summary>
              <p className="mt-2 leading-relaxed">{info.longBusinessSummary as string}</p>
            </details>
          )}
        </section>
      )}
    </div>
  );
}

function formatInfoValue(key: string, v: unknown): string {
  if (typeof v === "number") {
    if (key === "fullTimeEmployees" || key === "averageVolume" || key === "sharesOutstanding" || key === "floatShares") {
      return v.toLocaleString("en-IN");
    }
    if (key === "fiftyTwoWeekHigh" || key === "fiftyTwoWeekLow" || key === "bookValue") {
      return `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
    }
    if (key === "earningsGrowth" || key === "returnOnAssets") {
      return `${(v * 100).toFixed(2)}%`;
    }
    return v.toFixed(3);
  }
  if (typeof v === "string" && v.startsWith("http")) return v.replace(/^https?:\/\//, "");
  return String(v);
}

async function safe<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiError) return null;
    return null;
  }
}
