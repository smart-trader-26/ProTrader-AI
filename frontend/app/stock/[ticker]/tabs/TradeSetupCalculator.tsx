"use client";

import { useState } from "react";
import type { TechnicalSnapshot } from "@/lib/types";
import { formatINR } from "@/lib/format";

interface Props {
  snap: TechnicalSnapshot;
}

/**
 * Risk Management / Trade Setup Calculator — mirrors Streamlit Tab 3.
 * Computes entry zone, stop loss (ATR-based), and targets from support/resistance.
 * All values are editable so traders can override with their own levels.
 */
export default function TradeSetupCalculator({ snap }: Props) {
  const [open, setOpen] = useState(false);
  const atr = snap.atr_14 ?? 0;
  const price = snap.last_close ?? 0;
  const support = snap.support_1 ?? (price * 0.97);
  const resistance = snap.resistance_1 ?? (price * 1.05);

  const defaultEntry = price;
  const defaultSl = Math.max(support, price - atr * 1.5);
  const risk = defaultEntry - defaultSl;
  const defaultT1 = defaultEntry + risk * 1.5;
  const defaultT2 = defaultEntry + risk * 2.5;

  const [entry, setEntry] = useState(defaultEntry.toFixed(2));
  const [sl, setSl] = useState(defaultSl.toFixed(2));
  const [t1, setT1] = useState(defaultT1.toFixed(2));
  const [t2, setT2] = useState(defaultT2.toFixed(2));
  const [capital, setCapital] = useState("100000");
  const [riskPct, setRiskPct] = useState("1");

  const entryN = parseFloat(entry) || 0;
  const slN = parseFloat(sl) || 0;
  const t1N = parseFloat(t1) || 0;
  const t2N = parseFloat(t2) || 0;
  const capitalN = parseFloat(capital) || 100000;
  const riskPctN = parseFloat(riskPct) || 1;

  const riskPerShare = Math.abs(entryN - slN);
  const riskRupees = (capitalN * riskPctN) / 100;
  const positionSize = riskPerShare > 0 ? Math.floor(riskRupees / riskPerShare) : 0;
  const positionValue = positionSize * entryN;
  const rrT1 = riskPerShare > 0 ? (t1N - entryN) / riskPerShare : 0;
  const rrT2 = riskPerShare > 0 ? (t2N - entryN) / riskPerShare : 0;

  const slPct = entryN > 0 ? ((slN - entryN) / entryN) * 100 : 0;
  const t1Pct = entryN > 0 ? ((t1N - entryN) / entryN) * 100 : 0;
  const t2Pct = entryN > 0 ? ((t2N - entryN) / entryN) * 100 : 0;

  return (
    <section className="panel">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">💰 Risk Management / Trade Setup</p>
          <p className="text-xs text-muted">
            ATR-based stop loss · position sizing · R:R calculator
          </p>
        </div>
        <button className="btn text-sm" onClick={() => setOpen((o) => !o)}>
          {open ? "Hide" : "Show calculator"}
        </button>
      </div>

      {open && (
        <div className="mt-4 space-y-4">
          {/* ATR context */}
          <div className="grid gap-3 sm:grid-cols-4 text-xs text-muted border border-border rounded-lg p-3">
            <div><p className="uppercase tracking-wide">ATR (14)</p><p className="text-fg font-semibold text-sm">{formatINR(atr)}</p></div>
            <div><p className="uppercase tracking-wide">Support 1</p><p className="text-bull font-semibold text-sm">{formatINR(support)}</p></div>
            <div><p className="uppercase tracking-wide">Resistance 1</p><p className="text-bear font-semibold text-sm">{formatINR(resistance)}</p></div>
            <div><p className="uppercase tracking-wide">Last close</p><p className="text-fg font-semibold text-sm">{formatINR(price)}</p></div>
          </div>

          {/* Position sizing inputs */}
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="space-y-1">
              <span className="text-xs uppercase text-muted">Capital (₹)</span>
              <input type="number" className="input w-full" value={capital} onChange={(e) => setCapital(e.target.value)} />
            </label>
            <label className="space-y-1">
              <span className="text-xs uppercase text-muted">Risk per trade (%)</span>
              <input type="number" step="0.5" className="input w-full" value={riskPct} onChange={(e) => setRiskPct(e.target.value)} />
            </label>
          </div>

          {/* Trade levels */}
          <div className="grid gap-3 sm:grid-cols-4">
            <label className="space-y-1">
              <span className="text-xs uppercase text-muted">Entry (₹)</span>
              <input type="number" step="0.05" className="input w-full" value={entry} onChange={(e) => setEntry(e.target.value)} />
            </label>
            <label className="space-y-1">
              <span className="text-xs uppercase text-muted">Stop Loss (₹)</span>
              <input type="number" step="0.05" className="input w-full" value={sl} onChange={(e) => setSl(e.target.value)} />
              <span className="text-xs text-bear">{slPct.toFixed(2)}%</span>
            </label>
            <label className="space-y-1">
              <span className="text-xs uppercase text-muted">Target 1 (₹)</span>
              <input type="number" step="0.05" className="input w-full" value={t1} onChange={(e) => setT1(e.target.value)} />
              <span className="text-xs text-bull">{t1Pct.toFixed(2)}%</span>
            </label>
            <label className="space-y-1">
              <span className="text-xs uppercase text-muted">Target 2 (₹)</span>
              <input type="number" step="0.05" className="input w-full" value={t2} onChange={(e) => setT2(e.target.value)} />
              <span className="text-xs text-bull">{t2Pct.toFixed(2)}%</span>
            </label>
          </div>

          {/* Results */}
          <div className="grid gap-4 sm:grid-cols-4 border-t border-border pt-3">
            <div className="text-sm">
              <p className="text-xs uppercase text-muted mb-1">Position size</p>
              <p className="font-semibold text-lg">{positionSize} shares</p>
              <p className="text-xs text-muted">≈ {formatINR(positionValue)} deployed</p>
            </div>
            <div className="text-sm">
              <p className="text-xs uppercase text-muted mb-1">Risk exposure</p>
              <p className="font-semibold text-lg text-bear">₹{(riskPerShare * positionSize).toFixed(0)}</p>
              <p className="text-xs text-muted">₹{riskPerShare.toFixed(2)} / share</p>
            </div>
            <div className="text-sm">
              <p className="text-xs uppercase text-muted mb-1">R:R at Target 1</p>
              <p className={`font-semibold text-lg ${rrT1 >= 1.5 ? "text-bull" : "text-bear"}`}>
                1 : {rrT1.toFixed(2)}
              </p>
              <p className="text-xs text-muted">{rrT1 >= 1.5 ? "✓ Acceptable" : "✗ Below 1.5R"}</p>
            </div>
            <div className="text-sm">
              <p className="text-xs uppercase text-muted mb-1">R:R at Target 2</p>
              <p className={`font-semibold text-lg ${rrT2 >= 2.5 ? "text-bull" : "text-muted"}`}>
                1 : {rrT2.toFixed(2)}
              </p>
              <p className="text-xs text-muted">{rrT2 >= 2.5 ? "✓ Strong" : "Moderate"}</p>
            </div>
          </div>

          <p className="text-xs text-muted border-t border-border pt-2">
            Stop loss placed {Math.abs(slPct).toFixed(2)}% below entry using 1.5× ATR ({formatINR(atr)}).
            Min risk/reward of 1.5R required. Position sized to risk {riskPct}% of ₹{parseInt(capital).toLocaleString("en-IN")}.
          </p>
        </div>
      )}
    </section>
  );
}
