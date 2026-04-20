"use client";

import { useState, useEffect } from "react";
import Stat from "@/components/Stat";
import type { FiiDiiBundle } from "@/lib/types";
import { formatCompactINR, toneFor } from "@/lib/format";

const NSE_FIIDII_URL = "https://www.nseindia.com/api/fiidiiTradeReact";

export default function FiiDiiTab() {
  const [data, setData] = useState<FiiDiiBundle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [pasteMode, setPasteMode] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [pasteError, setPasteError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/v1/stocks/fii-dii?lookback_days=30")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d: FiiDiiBundle) => {
        if (!d || !d.rows || d.rows.length === 0) throw new Error("empty");
        setData(d);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  function parsePaste(raw: string): FiiDiiBundle | null {
    try {
      // Try JSON first
      if (raw.trim().startsWith("[") || raw.trim().startsWith("{")) {
        const parsed = JSON.parse(raw.trim());
        const arr = Array.isArray(parsed) ? parsed : parsed.data ?? [];
        return buildBundle(arr);
      }
      // Try tab/comma separated: Date | FII Buy | FII Sell | FII Net | DII Net
      const lines = raw.trim().split("\n").filter(Boolean);
      const rows = lines.map((line) => {
        const cols = line.split(/[\t,|]/).map((c) => c.trim());
        return {
          date: cols[0],
          fii_buy: parseFloat(cols[1]) || null,
          fii_sell: parseFloat(cols[2]) || null,
          fii_net: parseFloat(cols[3]) || null,
          dii_net: parseFloat(cols[4]) || null,
        };
      }).filter((r) => r.date && r.date.match(/\d/));
      if (rows.length === 0) return null;
      return buildBundle(rows);
    } catch {
      return null;
    }
  }

  function buildBundle(arr: Record<string, unknown>[]): FiiDiiBundle {
    const rows = arr.map((r) => ({
      date: String(r.date ?? r.tradeDate ?? ""),
      fii_buy: parseNum(r.fii_buy ?? r.fiiBuy ?? r["FII Buy"]),
      fii_sell: parseNum(r.fii_sell ?? r.fiiSell ?? r["FII Sell"]),
      fii_net: parseNum(r.fii_net ?? r.fiiNet ?? r["FII Net"]),
      dii_buy: parseNum(r.dii_buy ?? r.diiBuy ?? r["DII Buy"]),
      dii_sell: parseNum(r.dii_sell ?? r.diiSell ?? r["DII Sell"]),
      dii_net: parseNum(r.dii_net ?? r.diiNet ?? r["DII Net"]),
    }));
    const last5 = rows.slice(-5);
    const fii5 = last5.reduce((s, r) => s + (r.fii_net ?? 0), 0);
    const dii5 = last5.reduce((s, r) => s + (r.dii_net ?? 0), 0);
    return { rows, fii_net_5d: fii5, dii_net_5d: dii5, fii_net_streak: null, dii_net_streak: null };
  }

  function handlePasteSubmit() {
    setPasteError(null);
    const parsed = parsePaste(pasteText);
    if (!parsed || parsed.rows.length === 0) {
      setPasteError("Could not parse data. Try tab-separated or JSON format.");
      return;
    }
    setData(parsed);
    setError(false);
    setPasteMode(false);
  }

  if (loading) {
    return <p className="panel text-sm text-muted animate-pulse">Loading FII / DII data…</p>;
  }

  if ((error || !data) && !pasteMode) {
    return (
      <div className="panel space-y-4">
        <p className="text-sm text-bear font-semibold">NSE endpoint throttled — data unavailable.</p>
        <p className="text-sm text-muted">
          FII/DII flows are a model feature. You can paste the data manually from the NSE website:
        </p>
        <ol className="list-decimal list-inside text-sm space-y-1">
          <li>
            Open{" "}
            <a
              href="https://www.nseindia.com/market-data/fii-dii-activity"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent underline"
            >
              nseindia.com/market-data/fii-dii-activity
            </a>
          </li>
          <li>Copy the data table (Ctrl+A in the table, then Ctrl+C)</li>
          <li>Paste it below and press <strong>Submit</strong></li>
        </ol>
        <button
          className="btn btn-primary"
          onClick={() => setPasteMode(true)}
        >
          Paste FII / DII data manually
        </button>
      </div>
    );
  }

  if (pasteMode) {
    return (
      <div className="panel space-y-4">
        <p className="text-sm font-semibold">Paste FII / DII table data</p>
        <p className="text-xs text-muted">
          Accepted: tab-separated, comma-separated, or JSON array. Expected columns:{" "}
          <code className="bg-border/40 px-1 rounded">Date | FII Buy | FII Sell | FII Net | DII Net</code>
        </p>
        <textarea
          className="w-full h-48 bg-surface border border-border rounded p-2 text-sm font-mono resize-y"
          placeholder={"2025-04-01\t12345\t9876\t2469\t-1200\n..."}
          value={pasteText}
          onChange={(e) => setPasteText(e.target.value)}
        />
        {pasteError && <p className="text-sm text-bear">{pasteError}</p>}
        <div className="flex gap-2">
          <button className="btn btn-primary" onClick={handlePasteSubmit}>
            Submit
          </button>
          <button className="btn" onClick={() => setPasteMode(false)}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const rows = data.rows.slice().sort((a, b) => a.date.localeCompare(b.date));
  const maxAbs = Math.max(
    1,
    ...rows.flatMap((r) => [Math.abs(r.fii_net ?? 0), Math.abs(r.dii_net ?? 0)]),
  );

  let fiiCum = 0;
  let diiCum = 0;
  const cumulative = rows.map((r) => {
    fiiCum += r.fii_net ?? 0;
    diiCum += r.dii_net ?? 0;
    return { date: r.date, fii: fiiCum, dii: diiCum };
  });

  return (
    <div className="space-y-5">
      <section className="panel grid gap-4 sm:grid-cols-4">
        <Stat label="FII net (5d)" value={formatCompactINR(data.fii_net_5d)} tone={toneFor(data.fii_net_5d)} />
        <Stat label="DII net (5d)" value={formatCompactINR(data.dii_net_5d)} tone={toneFor(data.dii_net_5d)} />
        <Stat
          label="FII streak"
          value={data.fii_net_streak == null ? "—" : `${data.fii_net_streak} day${Math.abs(data.fii_net_streak) === 1 ? "" : "s"}`}
          tone={toneFor(data.fii_net_streak)}
        />
        <Stat
          label="DII streak"
          value={data.dii_net_streak == null ? "—" : `${data.dii_net_streak} day${Math.abs(data.dii_net_streak) === 1 ? "" : "s"}`}
          tone={toneFor(data.dii_net_streak)}
        />
      </section>

      <section className="panel">
        <p className="text-sm font-semibold mb-3">Daily net flows (₹ cr)</p>
        <FlowBars rows={rows} maxAbs={maxAbs} />
      </section>

      <section className="panel">
        <p className="text-sm font-semibold mb-3">Cumulative net over window</p>
        <CumulativeLines points={cumulative} />
      </section>

      <section className="panel overflow-x-auto">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-semibold">Raw data</p>
          <button className="text-xs text-accent underline" onClick={() => { setData(null); setError(true); }}>
            Re-paste data
          </button>
        </div>
        <table className="w-full text-sm">
          <thead className="text-left text-muted text-xs uppercase">
            <tr>
              <th className="pb-2">Date</th>
              <th className="pb-2 text-right">FII buy</th>
              <th className="pb-2 text-right">FII sell</th>
              <th className="pb-2 text-right">FII net</th>
              <th className="pb-2 text-right">DII net</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice().reverse().slice(0, 15).map((r) => (
              <tr key={r.date} className="border-t border-border">
                <td className="py-1">{r.date}</td>
                <td className="py-1 text-right">{formatCompactINR(r.fii_buy)}</td>
                <td className="py-1 text-right">{formatCompactINR(r.fii_sell)}</td>
                <td className={`py-1 text-right ${toneFor(r.fii_net) === "bull" ? "text-bull" : toneFor(r.fii_net) === "bear" ? "text-bear" : ""}`}>
                  {formatCompactINR(r.fii_net)}
                </td>
                <td className={`py-1 text-right ${toneFor(r.dii_net) === "bull" ? "text-bull" : toneFor(r.dii_net) === "bear" ? "text-bear" : ""}`}>
                  {formatCompactINR(r.dii_net)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function parseNum(v: unknown): number | null {
  const n = parseFloat(String(v ?? ""));
  return isNaN(n) ? null : n;
}

function FlowBars({
  rows,
  maxAbs,
}: {
  rows: { date: string; fii_net: number | null; dii_net: number | null }[];
  maxAbs: number;
}) {
  const w = 800;
  const h = 240;
  const padY = 20;
  const innerH = h - padY * 2;
  const midY = padY + innerH / 2;
  const barW = Math.max(2, Math.floor((w / (rows.length || 1)) * 0.35));
  const step = w / Math.max(1, rows.length);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none">
      <line x1={0} y1={midY} x2={w} y2={midY} stroke="#1f2a36" />
      {rows.map((r, i) => {
        const x = i * step;
        const fh = Math.abs(((r.fii_net ?? 0) / maxAbs) * (innerH / 2));
        const dh = Math.abs(((r.dii_net ?? 0) / maxAbs) * (innerH / 2));
        const fColor = (r.fii_net ?? 0) >= 0 ? "#22c55e" : "#ef4444";
        const dColor = (r.dii_net ?? 0) >= 0 ? "#2dd4bf" : "#fb923c";
        return (
          <g key={r.date}>
            <rect x={x} y={(r.fii_net ?? 0) >= 0 ? midY - fh : midY} width={barW} height={fh} fill={fColor} opacity={0.9} />
            <rect x={x + barW + 1} y={(r.dii_net ?? 0) >= 0 ? midY - dh : midY} width={barW} height={dh} fill={dColor} opacity={0.85} />
          </g>
        );
      })}
      <text x={4} y={padY - 4} fill="#8b98a5" fontSize={10}>Net ₹cr (+)</text>
      <text x={4} y={h - 4} fill="#8b98a5" fontSize={10}>Net ₹cr (−)</text>
      <g transform={`translate(${w - 170}, ${padY - 6})`} fontSize={10}>
        <rect x={0} y={0} width={10} height={10} fill="#22c55e" />
        <text x={14} y={9} fill="#e6edf3">FII</text>
        <rect x={60} y={0} width={10} height={10} fill="#2dd4bf" />
        <text x={74} y={9} fill="#e6edf3">DII</text>
      </g>
    </svg>
  );
}

function CumulativeLines({ points }: { points: { date: string; fii: number; dii: number }[] }) {
  if (points.length < 2) return <p className="text-sm text-muted">Not enough points.</p>;
  const allY = points.flatMap((p) => [p.fii, p.dii]);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const range = yMax - yMin || 1;
  const w = 800;
  const h = 220;
  const padX = 40;
  const padY = 20;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;
  const step = innerW / (points.length - 1);

  const path = (ys: number[]) =>
    ys.map((y, i) => {
      const px = padX + i * step;
      const py = padY + innerH - ((y - yMin) / range) * innerH;
      return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none">
      <line x1={padX} y1={padY + innerH - ((0 - yMin) / range) * innerH} x2={w - padX}
        y2={padY + innerH - ((0 - yMin) / range) * innerH} stroke="#1f2a36" strokeDasharray="3 3" />
      <path d={path(points.map((p) => p.fii))} fill="none" stroke="#22c55e" strokeWidth={2} />
      <path d={path(points.map((p) => p.dii))} fill="none" stroke="#2dd4bf" strokeWidth={2} />
      <text x={padX} y={h - 4} fill="#8b98a5" fontSize={10}>{points[0].date}</text>
      <text x={w - padX} y={h - 4} fill="#8b98a5" fontSize={10} textAnchor="end">{points[points.length - 1].date}</text>
      <g transform={`translate(${w - 170}, 4)`} fontSize={10}>
        <rect x={0} y={0} width={10} height={2} fill="#22c55e" />
        <text x={14} y={5} fill="#e6edf3">Cum FII</text>
        <rect x={60} y={0} width={10} height={2} fill="#2dd4bf" />
        <text x={74} y={5} fill="#e6edf3">Cum DII</text>
      </g>
    </svg>
  );
}
