"use client";

import {
  useRef,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";

export interface Domain {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export interface ZoomState {
  /** Current visible data domain (null = full extent) */
  domain: Domain | null;
  /** Pixel coord of mouse inside plot area, or null */
  crosshair: { px: number; py: number } | null;
  /** Convert a data-x value to a fraction [0,1] across the SVG inner width */
  toFracX: (v: number) => number;
  /** Convert a data-y value to a fraction [0,1] across the SVG inner height */
  toFracY: (v: number) => number;
}

interface Props {
  /** Full data extents — required so reset knows the full range */
  fullDomain: Domain;
  /** SVG padding object so we can map pixel ↔ data correctly */
  pad: { l: number; r: number; t: number; b: number };
  /** SVG logical width / height */
  svgW: number;
  svgH: number;
  children: (state: ZoomState) => ReactNode;
  className?: string;
}

const ZOOM_FACTOR = 0.15; // fraction of domain to shrink per wheel tick

/**
 * Data-domain zoom controller — like Plotly/Matplotlib.
 *
 * Interactions:
 *   Drag (no modifier) → rubber-band box-select to zoom
 *   Scroll wheel       → zoom X-axis under cursor (Y stays fixed)
 *   Shift + drag       → pan
 *   Double-click       → reset to full extent
 *   Reset button       → same as double-click
 */
export default function ZoomableChart({
  fullDomain,
  pad,
  svgW,
  svgH,
  children,
  className = "",
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [domain, setDomain] = useState<Domain | null>(null);
  const [box, setBox] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);
  const [crosshair, setCrosshair] = useState<{
    px: number;
    py: number;
  } | null>(null);
  const dragRef = useRef<{
    mode: "box" | "pan";
    startPx: number;
    startPy: number;
    domainAtStart: Domain;
  } | null>(null);

  const current = domain ?? fullDomain;

  // ── helpers ──────────────────────────────────────────────────────────────
  const innerW = svgW - pad.l - pad.r;
  const innerH = svgH - pad.t - pad.b;

  /** Convert pixel offset inside the div → data value */
  const pxToDataX = useCallback(
    (px: number, dom: Domain) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return dom.x0;
      const svgPx = (px / rect.width) * svgW;
      const frac = Math.max(0, Math.min(1, (svgPx - pad.l) / innerW));
      return dom.x0 + frac * (dom.x1 - dom.x0);
    },
    [pad.l, innerW, svgW],
  );

  const pxToDataY = useCallback(
    (py: number, dom: Domain) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return dom.y0;
      const svgPy = (py / rect.height) * svgH;
      const frac = Math.max(0, Math.min(1, (svgPy - pad.t) / innerH));
      // SVG Y is flipped: top = y1, bottom = y0
      return dom.y0 + (1 - frac) * (dom.y1 - dom.y0);
    },
    [pad.t, innerH, svgH],
  );

  // ── zoom state → fraction mappers (passed to children) ──────────────────
  const toFracX = useCallback(
    (v: number) => (v - current.x0) / (current.x1 - current.x0 || 1),
    [current.x0, current.x1],
  );
  const toFracY = useCallback(
    (v: number) => (v - current.y0) / (current.y1 - current.y0 || 1),
    [current.y0, current.y1],
  );

  // ── wheel → zoom X under cursor ──────────────────────────────────────────
  const onWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const relX = e.clientX - rect.left;

      setDomain((prev) => {
        const dom = prev ?? fullDomain;
        const pivotX = pxToDataX(relX, dom);
        const span = dom.x1 - dom.x0;
        const dir = e.deltaY > 0 ? 1 : -1; // >0 = zoom out, <0 = zoom in
        const delta = dir * ZOOM_FACTOR * span;
        const lo = pivotX - (span + delta) * ((pivotX - dom.x0) / span);
        const hi = lo + span + delta;
        // Clamp to fullDomain
        const nx0 = Math.max(fullDomain.x0, lo);
        const nx1 = Math.min(fullDomain.x1, hi);
        if (nx1 - nx0 < (fullDomain.x1 - fullDomain.x0) * 0.01) return prev; // too small
        return { ...dom, x0: nx0, x1: nx1 };
      });
    },
    [fullDomain, pxToDataX],
  );

  // Attach wheel non-passively so preventDefault works
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  // ── mouse down → start box or pan ────────────────────────────────────────
  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const dom = domain ?? fullDomain;

      if (e.shiftKey) {
        dragRef.current = {
          mode: "pan",
          startPx: px,
          startPy: py,
          domainAtStart: dom,
        };
      } else {
        dragRef.current = {
          mode: "box",
          startPx: px,
          startPy: py,
          domainAtStart: dom,
        };
        setBox({ x0: px, y0: py, x1: px, y1: py });
      }
    },
    [domain, fullDomain],
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      setCrosshair({ px, py });

      const drag = dragRef.current;
      if (!drag) return;

      if (drag.mode === "box") {
        setBox({ x0: drag.startPx, y0: drag.startPy, x1: px, y1: py });
      } else {
        // pan: shift domain by delta
        const dom = drag.domainAtStart;
        const rect2 = containerRef.current?.getBoundingClientRect();
        if (!rect2) return;
        const dxData =
          ((drag.startPx - px) / rect2.width) *
          svgW /
          innerW *
          (dom.x1 - dom.x0);
        const dyData =
          ((py - drag.startPy) / rect2.height) *
          svgH /
          innerH *
          (dom.y1 - dom.y0);
        const spanX = dom.x1 - dom.x0;
        const spanY = dom.y1 - dom.y0;
        let nx0 = dom.x0 + dxData;
        let nx1 = dom.x1 + dxData;
        if (nx0 < fullDomain.x0) { nx0 = fullDomain.x0; nx1 = nx0 + spanX; }
        if (nx1 > fullDomain.x1) { nx1 = fullDomain.x1; nx0 = nx1 - spanX; }
        let ny0 = dom.y0 + dyData;
        let ny1 = dom.y1 + dyData;
        if (ny0 < fullDomain.y0) { ny0 = fullDomain.y0; ny1 = ny0 + spanY; }
        if (ny1 > fullDomain.y1) { ny1 = fullDomain.y1; ny0 = ny1 - spanY; }
        setDomain({ x0: nx0, x1: nx1, y0: ny0, y1: ny1 });
      }
    },
    [fullDomain, innerH, innerW, svgH, svgW],
  );

  const onMouseUp = useCallback(
    (e: React.MouseEvent) => {
      const drag = dragRef.current;
      dragRef.current = null;

      if (drag?.mode === "box" && box) {
        const minDrag = 8; // px threshold
        const bw = Math.abs(box.x1 - box.x0);
        const bh = Math.abs(box.y1 - box.y0);
        if (bw > minDrag && bh > minDrag) {
          const dom = drag.domainAtStart;
          const xa = pxToDataX(Math.min(box.x0, box.x1), dom);
          const xb = pxToDataX(Math.max(box.x0, box.x1), dom);
          const ya = pxToDataY(Math.max(box.y0, box.y1), dom); // note: higher px = lower data
          const yb = pxToDataY(Math.min(box.y0, box.y1), dom);
          setDomain({
            x0: Math.max(fullDomain.x0, xa),
            x1: Math.min(fullDomain.x1, xb),
            y0: Math.max(fullDomain.y0, ya),
            y1: Math.min(fullDomain.y1, yb),
          });
        }
      }
      setBox(null);
    },
    [box, fullDomain, pxToDataX, pxToDataY],
  );

  const onMouseLeave = useCallback(() => {
    dragRef.current = null;
    setBox(null);
    setCrosshair(null);
  }, []);

  const onDblClick = useCallback(() => {
    setDomain(null);
  }, []);

  const reset = useCallback(() => setDomain(null), []);

  const isZoomed = domain !== null;

  return (
    <div className={`relative select-none ${className}`}>
      {/* Toolbar */}
      <div className="absolute top-1.5 right-2 z-20 flex items-center gap-2">
        {isZoomed && (
          <button
            onClick={reset}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium border border-border bg-surface text-muted hover:text-fg hover:border-accent transition-colors"
          >
            <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor">
              <path d="M13.5 3A1.5 1.5 0 0 0 12 1.5H4A1.5 1.5 0 0 0 2.5 3v10A1.5 1.5 0 0 0 4 14.5h8A1.5 1.5 0 0 0 13.5 13V3ZM8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6Z"/>
            </svg>
            Reset
          </button>
        )}
        <span className="text-[10px] text-muted pointer-events-none">
          {isZoomed ? "Drag·Scroll·Shift+Drag" : "Drag to zoom · Scroll · Shift+drag to pan"}
        </span>
      </div>

      {/* Chart area */}
      <div
        ref={containerRef}
        className="w-full"
        style={{ cursor: box ? "crosshair" : "default" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onDoubleClick={onDblClick}
      >
        {children({ domain, crosshair, toFracX, toFracY })}

        {/* Rubber-band selection box */}
        {box && (
          <div
            className="pointer-events-none absolute border border-accent/70 bg-accent/10"
            style={{
              left: Math.min(box.x0, box.x1),
              top: Math.min(box.y0, box.y1),
              width: Math.abs(box.x1 - box.x0),
              height: Math.abs(box.y1 - box.y0),
            }}
          />
        )}
      </div>
    </div>
  );
}
