"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TickerSearchResponse {
  count: number;
  results: string[];
}

/**
 * Typeahead bound to `GET /api/v1/stocks?q=`. Forces users to pick from the
 * known Indian universe instead of free-typing, which was producing invalid
 * tickers (typos, missing suffixes) that failed downstream at yfinance.
 *
 * Public endpoint — no auth header needed. Debounced at 200 ms.
 */
export default function TickerPicker({
  value,
  onChange,
  onCommit,
  placeholder = "Search ticker (e.g. RELIANCE)",
  autoFocus = false,
}: {
  value: string;
  onChange: (v: string) => void;
  onCommit?: (v: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState<string[]>([]);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!value.trim()) {
      setResults([]);
      return;
    }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/stocks?q=${encodeURIComponent(value)}&limit=10`,
          { cache: "no-store" },
        );
        if (!res.ok) throw new Error(`search failed (${res.status})`);
        const body: TickerSearchResponse = await res.json();
        setResults(body.results ?? []);
        setActive(0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [value]);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function pick(sym: string) {
    onChange(sym);
    setOpen(false);
    onCommit?.(sym);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || results.length === 0) {
      if (e.key === "Enter" && onCommit && value.trim()) {
        // No dropdown: refuse free-text submissions.
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(results.length - 1, a + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      pick(results[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div ref={boxRef} className="relative flex-1">
      <input
        className="input"
        placeholder={placeholder}
        autoFocus={autoFocus}
        value={value}
        onChange={(e) => {
          onChange(e.target.value.toUpperCase());
          setOpen(true);
        }}
        onFocus={() => value && setOpen(true)}
        onKeyDown={onKeyDown}
        autoComplete="off"
      />
      {open && (results.length > 0 || loading) && (
        <ul className="absolute z-10 mt-1 max-h-64 w-full overflow-auto rounded-md border border-border bg-panel shadow-lg">
          {loading && results.length === 0 && (
            <li className="px-3 py-2 text-xs text-muted">Searching…</li>
          )}
          {results.map((sym, i) => (
            <li key={sym}>
              <button
                type="button"
                onClick={() => pick(sym)}
                onMouseEnter={() => setActive(i)}
                className={`flex w-full items-center px-3 py-2 text-left text-sm ${
                  i === active ? "bg-[#17202c] text-accent" : "hover:bg-[#17202c]"
                }`}
              >
                {sym}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
