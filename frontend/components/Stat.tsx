/** Small KPI tile. `tone` colours the value (bull/bear/muted). */

export default function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "bull" | "bear" | "muted";
}) {
  return (
    <div>
      <p className="text-xs uppercase text-muted tracking-wide">{label}</p>
      <p
        className={`text-lg font-semibold ${
          tone === "bull" ? "text-bull" : tone === "bear" ? "text-bear" : ""
        }`}
      >
        {value}
      </p>
      {hint && <p className="text-xs text-muted mt-0.5">{hint}</p>}
    </div>
  );
}
