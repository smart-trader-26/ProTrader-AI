/**
 * LivePriceBadge — shows a real-time price from the WebSocket stream (B5.4).
 *
 * Displays a pulsing dot + live price for a single ticker.
 * Uses the useTickerPrices hook to subscribe to the backend WS.
 *
 * Usage:
 *   <LivePriceBadge ticker="RELIANCE.NS" />
 */

"use client";

import { useTickerPrices, WsStatus } from "@/hooks/useTickerPrices";

interface Props {
  ticker: string;
  className?: string;
}

const STATUS_DOT: Record<WsStatus, string> = {
  connecting: "bg-yellow-400 animate-pulse",
  connected: "bg-green-400 animate-pulse",
  closed: "bg-gray-400",
  error: "bg-red-400",
};

export default function LivePriceBadge({ ticker, className = "" }: Props) {
  const { prices, status, lastTick } = useTickerPrices([ticker]);
  const price = prices.get(ticker);

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs tabular-nums ${className}`}
      title={lastTick ? `Last tick: ${lastTick}` : `Status: ${status}`}
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${STATUS_DOT[status]}`} />
      {price != null ? (
        <span className="font-medium">₹{price.toFixed(2)}</span>
      ) : (
        <span className="text-muted">—</span>
      )}
    </span>
  );
}
