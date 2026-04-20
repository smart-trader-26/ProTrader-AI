"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import TickerPicker from "./TickerPicker";

/** Condensed stock picker for the top nav. */
export default function HeaderSearch() {
  const [value, setValue] = useState("");
  const router = useRouter();
  return (
    <div className="w-56 max-w-xs">
      <TickerPicker
        value={value}
        onChange={setValue}
        onCommit={(sym) => {
          setValue("");
          router.push(`/stock/${sym}`);
        }}
        placeholder="Search ticker…"
      />
    </div>
  );
}
