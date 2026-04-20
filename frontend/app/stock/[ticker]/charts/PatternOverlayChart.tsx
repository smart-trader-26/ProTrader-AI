"use client";

import { useEffect, useRef, useState } from "react";
import {
  ColorType,
  createChart,
  IChartApi,
  LineStyle,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { DetectedPattern, StockBar, SupportResistance } from "@/lib/types";

interface Props {
  bars: StockBar[];
  patterns: DetectedPattern[];
  supportResistance?: SupportResistance | null;
  height?: number;
  initialWindowDays?: number;
}

const PATTERN_COLORS = [
  "#ef4444", "#22c55e", "#a78bfa", "#f59e0b", "#22d3ee", "#f97316",
];

export default function PatternOverlayChart({
  bars,
  patterns,
  supportResistance,
  height = 520,
  initialWindowDays = 90,
}: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [windowDays, setWindowDays] = useState(initialWindowDays);

  useEffect(() => {
    if (!wrapRef.current || bars.length === 0) return;

    const chart = createChart(wrapRef.current, {
      width: wrapRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#111820" },
        textColor: "#8b98a5",
      },
      grid: {
        vertLines: { color: "#1f2a36" },
        horzLines: { color: "#1f2a36" },
      },
      rightPriceScale: { borderColor: "#1f2a36" },
      timeScale: { borderColor: "#1f2a36", timeVisible: true },
      crosshair: { mode: 1 },
      handleScroll: true,
      handleScale: true,
    });
    chartRef.current = chart;

    const candle = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      borderVisible: false,
    });

    const trimmed = bars.slice(-windowDays);
    const barMap = new Map<string, StockBar>();
    trimmed.forEach((b) => { barMap.set(b.ts.slice(0, 10), b); });

    candle.setData(
      trimmed.map((b) => ({
        time: Math.floor(new Date(b.ts).getTime() / 1000) as UTCTimestamp,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    );

    // ── Pattern overlays ──────────────────────────────────────────────────────
    patterns.forEach((p, i) => {
      const color = PATTERN_COLORS[i % PATTERN_COLORS.length];
      const isBull = (p.type ?? "").toLowerCase().includes("bull");
      const pColor = isBull ? "#22c55e" : color;

      // 1. Neckline horizontal line
      if (p.neckline != null && Number.isFinite(p.neckline)) {
        candle.createPriceLine({
          price: p.neckline,
          color: pColor,
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `${p.name} neckline (${p.confidence.toFixed(0)}%)`,
        });
      }

      // 2. Target horizontal line
      if (p.target != null && Number.isFinite(p.target)) {
        candle.createPriceLine({
          price: p.target,
          color: pColor,
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: `${p.name} target → ₹${p.target.toFixed(0)}`,
        });
      }

      // 3. Keypoint shape overlay (actual pattern lines)
      if (p.keypoints && p.keypoints.length >= 2) {
        // Build a line series through the keypoints
        const kpData: { time: UTCTimestamp; value: number }[] = [];
        for (const kp of p.keypoints) {
          const ts = Math.floor(new Date(kp.date).getTime() / 1000) as UTCTimestamp;
          kpData.push({ time: ts, value: kp.price });
        }
        kpData.sort((a, b) => (a.time as number) - (b.time as number));

        const shapeSeries = chart.addLineSeries({
          color: pColor,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 5,
        });
        shapeSeries.setData(kpData);

        // Add markers at each keypoint
        const markers = p.keypoints
          .map((kp) => ({
            time: Math.floor(new Date(kp.date).getTime() / 1000) as UTCTimestamp,
            position: (kp.label.toLowerCase().includes("bottom") || kp.label.toLowerCase().includes("head") && isBull || kp.label === "Head" && !isBull ? "belowBar" : "aboveBar") as "aboveBar" | "belowBar",
            color: pColor,
            shape: "circle" as const,
            text: kp.label,
            size: 1,
          }))
          .sort((a, b) => (a.time as number) - (b.time as number));
        shapeSeries.setMarkers(markers);
      }
    });

    // ── Support / resistance ──────────────────────────────────────────────────
    if (supportResistance) {
      (supportResistance.strong_supports ?? []).slice(0, 4).forEach((s) => {
        if (Number.isFinite(s)) {
          candle.createPriceLine({
            price: s,
            color: "#22c55e",
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: `S ₹${Math.round(s)}`,
          });
        }
      });
      (supportResistance.strong_resistances ?? []).slice(0, 4).forEach((r) => {
        if (Number.isFinite(r)) {
          candle.createPriceLine({
            price: r,
            color: "#ef4444",
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: `R ₹${Math.round(r)}`,
          });
        }
      });
    }

    chart.timeScale().fitContent();

    const onResize = () => {
      if (wrapRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: wrapRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, patterns, supportResistance, height, windowDays]);

  if (bars.length === 0) {
    return (
      <div className="panel text-sm text-muted">
        OHLCV unavailable — chart cannot render pattern overlays.
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
        <div>
          <p className="text-sm font-semibold">Pattern Visualization</p>
          <p className="text-xs text-muted mt-0.5">
            Scroll to zoom · drag to pan · colored lines = pattern keypoints overlaid on candlesticks
          </p>
        </div>
        <label className="text-xs text-muted flex items-center gap-2">
          Window
          <input
            type="range"
            min={20}
            max={Math.min(365, bars.length)}
            value={windowDays}
            onChange={(e) => setWindowDays(Number(e.target.value))}
            className="w-32 accent-accent"
          />
          <span className="tabular-nums w-10 text-right">{windowDays}d</span>
        </label>
      </div>
      <div ref={wrapRef} style={{ height }} />
      <div className="mt-3 flex flex-wrap gap-3 text-xs">
        {patterns.slice(0, 8).map((p, i) => {
          const pColor = PATTERN_COLORS[i % PATTERN_COLORS.length];
          const isBull = (p.type ?? "").toLowerCase().includes("bull");
          const c = isBull ? "#22c55e" : pColor;
          return (
            <span key={`${p.name}-${i}`} className="chip flex items-center gap-1" style={{ borderColor: c }}>
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: c }} />
              <span>{p.name}</span>
              <span className="text-muted">· {p.confidence.toFixed(0)}%</span>
              {p.keypoints.length > 0 && (
                <span className="text-muted">· {p.keypoints.length} points</span>
              )}
            </span>
          );
        })}
      </div>

      {/* Keypoint table for any pattern with keypoints */}
      {patterns.some((p) => p.keypoints.length > 0) && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-muted">Pattern keypoints</summary>
          <div className="mt-2 space-y-3">
            {patterns.filter((p) => p.keypoints.length > 0).map((p, i) => (
              <div key={i}>
                <p className="font-semibold mb-1">{p.name}</p>
                <table className="w-full">
                  <thead>
                    <tr className="text-muted uppercase">
                      <th className="text-left pb-1">Label</th>
                      <th className="text-right pb-1">Date</th>
                      <th className="text-right pb-1">Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.keypoints.map((kp, j) => (
                      <tr key={j} className="border-t border-border">
                        <td className="py-0.5 text-fg">{kp.label}</td>
                        <td className="py-0.5 text-right text-muted">{kp.date}</td>
                        <td className="py-0.5 text-right">₹{kp.price.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
