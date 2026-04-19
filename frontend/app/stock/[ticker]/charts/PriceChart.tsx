"use client";

import { useEffect, useRef } from "react";
import {
  ColorType,
  createChart,
  IChartApi,
  LineStyle,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { PredictionPoint, StockBar } from "@/lib/types";

interface Props {
  bars: StockBar[];
  predictions?: PredictionPoint[];
  anchorPrice?: number | null;
  height?: number;
}

/**
 * OHLCV candlestick + forecast overlay. Shaded band between CI low/high when
 * the model returned them. Designed for the Overview + Predict tab so the
 * same chart renders raw history and the predicted trajectory.
 */
export default function PriceChart({ bars, predictions, anchorPrice, height = 420 }: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!wrapRef.current) return;
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
    });
    chartRef.current = chart;

    const candle = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      borderVisible: false,
    });

    candle.setData(
      bars.map((b) => ({
        time: Math.floor(new Date(b.ts).getTime() / 1000) as UTCTimestamp,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    );

    let predSeries: ISeriesApi<"Line"> | null = null;
    let bandHi: ISeriesApi<"Line"> | null = null;
    let bandLo: ISeriesApi<"Line"> | null = null;
    if (predictions && predictions.length > 0) {
      predSeries = chart.addLineSeries({
        color: "#2dd4bf",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
      });
      const predLine: { time: UTCTimestamp; value: number }[] = [];
      if (anchorPrice && bars.length > 0) {
        predLine.push({
          time: Math.floor(new Date(bars[bars.length - 1].ts).getTime() / 1000) as UTCTimestamp,
          value: anchorPrice,
        });
      }
      for (const p of predictions) {
        predLine.push({
          time: Math.floor(new Date(p.target_date).getTime() / 1000) as UTCTimestamp,
          value: p.pred_price,
        });
      }
      predSeries.setData(predLine);

      const ciValid = predictions.every((p) => p.ci_low != null && p.ci_high != null);
      if (ciValid) {
        bandHi = chart.addLineSeries({
          color: "rgba(45, 212, 191, 0.35)",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
        });
        bandLo = chart.addLineSeries({
          color: "rgba(45, 212, 191, 0.35)",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
        });
        bandHi.setData(
          predictions.map((p) => ({
            time: Math.floor(new Date(p.target_date).getTime() / 1000) as UTCTimestamp,
            value: p.ci_high as number,
          })),
        );
        bandLo.setData(
          predictions.map((p) => ({
            time: Math.floor(new Date(p.target_date).getTime() / 1000) as UTCTimestamp,
            value: p.ci_low as number,
          })),
        );
      }
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
  }, [bars, predictions, anchorPrice, height]);

  return <div ref={wrapRef} className="w-full" style={{ height }} />;
}
