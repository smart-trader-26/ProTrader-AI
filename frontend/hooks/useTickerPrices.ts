/**
 * useTickerPrices — live price stream via WebSocket (B4.7 / B5.4).
 *
 * Connects to the backend's `/api/v1/ws/prices?tickers=X,Y,Z` and
 * returns a reactive map of `{ ticker → price }` that updates in
 * real-time as the tick publisher emits frames.
 *
 * Falls back gracefully — if the WS can't connect (backend down, no
 * Redis), the map stays empty and `status` is "error" or "closed".
 *
 * Usage:
 *   const { prices, status } = useTickerPrices(["RELIANCE.NS", "TCS.NS"]);
 *   // prices.get("RELIANCE.NS") → 2841.5 (or undefined if not yet received)
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl } from "@/lib/api-client";

export type WsStatus = "connecting" | "connected" | "closed" | "error";

export interface TickerPricesResult {
  /** Live prices — updated on each WebSocket frame. */
  prices: Map<string, number>;
  /** Connection status. */
  status: WsStatus;
  /** Last tick timestamp (ISO string). */
  lastTick: string | null;
}

const RECONNECT_DELAY_MS = 5000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useTickerPrices(tickers: string[]): TickerPricesResult {
  const [prices, setPrices] = useState<Map<string, number>>(new Map());
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [lastTick, setLastTick] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const tickersKey = tickers.sort().join(",");

  const connect = useCallback(() => {
    if (!tickersKey) return;

    const url = wsUrl(`/api/v1/ws/prices?tickers=${tickersKey}`);
    setStatus("connecting");

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      attemptsRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.ticker && data.price != null) {
          setPrices((prev) => {
            const next = new Map(prev);
            next.set(data.ticker, data.price);
            return next;
          });
          setLastTick(data.ts ?? new Date().toISOString());
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus("closed");
      wsRef.current = null;
      // Auto-reconnect with backoff
      if (attemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        attemptsRef.current += 1;
        setTimeout(connect, RECONNECT_DELAY_MS * attemptsRef.current);
      }
    };
  }, [tickersKey]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return { prices, status, lastTick };
}
