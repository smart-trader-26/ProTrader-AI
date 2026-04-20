"use client";

import Link from "next/link";
import { TABS } from "./tabs";

export default function StockTabs({
  ticker,
  active,
}: {
  ticker: string;
  active: string;
}) {
  return (
    <nav className="border-b border-border -mx-4 px-4 overflow-x-auto">
      <ul className="flex items-end gap-1 min-w-max text-sm">
        {TABS.map((t) => {
          const isActive = t.key === active;
          return (
            <li key={t.key}>
              <Link
                scroll={false}
                href={`/stock/${ticker}?tab=${t.key}`}
                className={[
                  "inline-flex items-center gap-1.5 px-3 py-2 border-b-2 transition",
                  isActive
                    ? "border-accent text-fg"
                    : "border-transparent text-muted hover:text-fg hover:border-border",
                ].join(" ")}
              >
                <span aria-hidden>{t.icon}</span>
                <span>{t.label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
