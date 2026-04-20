"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import TickerPicker from "./TickerPicker";

/** Wrapper that routes TickerPicker selections into the stock detail page. */
export default function HomeSearch({ requireAuth }: { requireAuth: boolean }) {
  const [value, setValue] = useState("");
  const router = useRouter();
  return (
    <TickerPicker
      value={value}
      onChange={setValue}
      onCommit={(sym) => {
        const href = `/stock/${sym}`;
        router.push(requireAuth ? `/login?next=${encodeURIComponent(href)}` : href);
      }}
      autoFocus
    />
  );
}
