import { apiFetch, ApiError } from "@/lib/api";
import Stat from "@/components/Stat";
import PieChart from "@/components/PieChart";
import type {
  EventBreakdownItem,
  HeadlineSentiment,
  SentimentAggregate,
  SourceBreakdown,
} from "@/lib/types";
import V2EnsemblePanel from "./V2EnsemblePanel";

const CAT_COLOR: Record<string, string> = {
  earnings: "#22c55e",
  guidance: "#a78bfa",
  "m&a": "#60a5fa",
  regulatory: "#f97316",
  management: "#f59e0b",
  product: "#22d3ee",
  general: "#22d3ee",
  other: "#8b98a5",
};

export default async function SentimentTab({ ticker }: { ticker: string }) {
  const agg = await safe<SentimentAggregate>(
    `/api/v1/stocks/${ticker}/sentiment?max_headlines=40`,
  );

  if (!agg) {
    return (
      <p className="panel text-sm text-muted">
        Sentiment unavailable — headline sources may be offline.
      </p>
    );
  }

  const tone =
    agg.mean_score > 0.1 ? "bull" : agg.mean_score < -0.1 ? "bear" : "muted";
  const overallLabel = (agg.overall_label ?? "neutral").toUpperCase();
  const bySource = groupBy(agg.headlines, (h) => h.source || "Unknown");

  return (
    <div className="space-y-5">
      <SentimentHeroBanner ticker={ticker} agg={agg} />

      <SourceAgreementPanel agreement={agg.source_agreement} label={agg.source_agreement_label} breakdown={agg.source_breakdown} />

      <section className="panel grid gap-4 sm:grid-cols-4">
        <Stat
          label="Mean score"
          value={agg.mean_score.toFixed(3)}
          tone={tone}
          hint={`overall ${overallLabel}`}
        />
        <Stat
          label="Confidence"
          value={`${(agg.confidence * 100).toFixed(0)}%`}
          hint="mean per-headline"
        />
        <Stat label="Headlines" value={String(agg.n_headlines)} />
        <Stat
          label="Pos / Neg"
          value={`${agg.pos_count} / ${agg.neg_count}`}
          hint={`${agg.neu_count} neutral`}
        />
      </section>

      <NewsEventBreakdown items={agg.event_breakdown} />

      <V2EnsemblePanel ticker={ticker} />

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Headlines by source</h2>
        {Object.entries(bySource)
          .sort(([, a], [, b]) => b.length - a.length)
          .map(([source, items], idx) => (
            <details key={source} open={idx === 0}>
              <summary className="cursor-pointer py-2 text-sm flex items-center justify-between">
                <span className="font-medium">{source}</span>
                <span className="chip">{items.length}</span>
              </summary>
              <ul className="space-y-2 mt-2">
                {items.slice(0, 25).map((h, i) => (
                  <HeadlineRow h={h} key={`${source}-${i}`} />
                ))}
              </ul>
            </details>
          ))}
      </section>
    </div>
  );
}

function SentimentHeroBanner({
  ticker,
  agg,
}: {
  ticker: string;
  agg: SentimentAggregate;
}) {
  const overall = agg.overall_label ?? "neutral";
  const isBull = overall === "positive";
  const isBear = overall === "negative";
  const accent = isBull ? "#22c55e" : isBear ? "#ef4444" : "#8b98a5";
  const bg = isBull
    ? "linear-gradient(120deg, rgba(34,197,94,0.16), rgba(15,23,30,0.6))"
    : isBear
    ? "linear-gradient(120deg, rgba(239,68,68,0.16), rgba(15,23,30,0.6))"
    : "linear-gradient(120deg, rgba(139,152,165,0.12), rgba(15,23,30,0.6))";
  return (
    <section
      className="panel flex flex-wrap items-center justify-between gap-4"
      style={{ borderColor: accent, background: bg }}
    >
      <div>
        <p className="text-sm font-semibold">{ticker} Sentiment</p>
        <p className="text-xs text-muted mt-0.5">
          {agg.n_headlines} headlines · window {new Date(agg.window_start).toLocaleDateString()}
        </p>
      </div>
      <div className="text-right">
        <p
          className="text-2xl font-bold tracking-wide"
          style={{ color: accent }}
        >
          {overall.toUpperCase()}
        </p>
        <p className="text-xs text-muted">
          Score: {agg.mean_score >= 0 ? "+" : ""}
          {agg.mean_score.toFixed(3)} · Confidence: {(agg.confidence * 100).toFixed(1)}%
        </p>
      </div>
    </section>
  );
}

function SourceAgreementPanel({
  agreement,
  label,
  breakdown,
}: {
  agreement: number | null;
  label: string | null;
  breakdown: SourceBreakdown[];
}) {
  if (!breakdown || breakdown.length === 0) {
    return null;
  }
  const tone =
    agreement == null
      ? "muted"
      : agreement >= 0.8
      ? "bull"
      : agreement >= 0.5
      ? "muted"
      : "bear";
  const toneColor = tone === "bull" ? "#22c55e" : tone === "bear" ? "#ef4444" : "#8b98a5";
  const heading =
    label ??
    (breakdown.length === 1
      ? "Single source — agreement undefined"
      : "Source breakdown");
  const checkmark = tone === "bull" ? "✓" : tone === "bear" ? "✗" : "·";

  return (
    <section
      className="panel"
      style={{ borderLeft: `4px solid ${toneColor}` }}
    >
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <p className="text-sm font-semibold" style={{ color: toneColor }}>
            {heading} {checkmark}
          </p>
          <p className="text-xs text-muted mt-0.5">
            {agreement == null
              ? "Need ≥ 2 sources for an agreement score."
              : `Agreement score: ${(agreement * 100).toFixed(0)}% — ${
                  agreement >= 0.8
                    ? "Low inter-source disagreement."
                    : agreement >= 0.5
                    ? "Some divergence between sources."
                    : "High inter-source disagreement — read with caution."
                }`}
          </p>
        </div>
        {agreement != null && (
          <div className="min-w-[160px]">
            <div className="h-2 rounded-full bg-border overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${(agreement * 100).toFixed(0)}%`, background: toneColor }}
              />
            </div>
          </div>
        )}
      </div>
      <ul className="mt-3 grid gap-2 sm:grid-cols-2 text-sm">
        {breakdown.map((s) => {
          const sourceTone =
            s.label === "positive"
              ? "#22c55e"
              : s.label === "negative"
              ? "#ef4444"
              : "#8b98a5";
          return (
            <li
              key={s.source}
              className="flex items-center gap-2 border-b border-border/40 pb-1"
            >
              <span className="font-medium flex-1 truncate">{s.source}</span>
              <span className="text-xs text-muted">{s.n_headlines}</span>
              <span
                className="chip whitespace-nowrap"
                style={{ color: sourceTone, borderColor: sourceTone }}
              >
                {s.mean_score >= 0 ? "+" : ""}
                {s.mean_score.toFixed(2)} · {s.label}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function NewsEventBreakdown({ items }: { items: EventBreakdownItem[] }) {
  if (!items || items.length === 0) {
    return null;
  }
  const slices = items.map((i) => ({
    label: i.category,
    value: i.count,
    color: CAT_COLOR[i.category.toLowerCase()] ?? undefined,
  }));
  return (
    <section className="panel">
      <p className="text-sm font-semibold mb-3">News Event Breakdown (All Sources)</p>
      <PieChart slices={slices} size={220} innerRadius={50} />
    </section>
  );
}

function HeadlineRow({ h }: { h: HeadlineSentiment }) {
  const labelClass =
    h.score.label === "positive"
      ? "text-bull"
      : h.score.label === "negative"
      ? "text-bear"
      : "text-muted";
  return (
    <li className="panel text-sm">
      <div className="flex items-start justify-between gap-3">
        {h.url ? (
          <a
            href={h.url}
            target="_blank"
            rel="noreferrer"
            className="hover:text-accent line-clamp-2"
          >
            {h.title}
          </a>
        ) : (
          <span className="line-clamp-2">{h.title}</span>
        )}
        <span className={`chip whitespace-nowrap ${labelClass}`}>
          {h.score.label} · {(h.score.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <div className="text-xs text-muted mt-1 flex items-center gap-3">
        <span>{h.category || "Other"}</span>
        {h.published_at && (
          <span>· {new Date(h.published_at).toLocaleString()}</span>
        )}
      </div>
    </li>
  );
}

function groupBy<T>(items: T[], key: (x: T) => string): Record<string, T[]> {
  const out: Record<string, T[]> = {};
  for (const x of items) {
    const k = key(x);
    (out[k] ??= []).push(x);
  }
  return out;
}

async function safe<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch (e) {
    if (e instanceof ApiError) return null;
    return null;
  }
}
