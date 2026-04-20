"use client";

import { useRef, useState, useCallback, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
}

/**
 * Wrapper that adds scroll-to-zoom + drag-to-pan to any SVG chart child.
 * The SVG uses preserveAspectRatio="none" and viewBox so this works by
 * applying a CSS transform to the inner container.
 *
 * Controls:
 *   Wheel → zoom in/out (centred on cursor)
 *   Drag  → pan
 *   Double-click → reset
 */
export default function ZoomableChart({ children, className = "" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef<{ startX: number; startY: number; tx: number; ty: number } | null>(null);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const cursorX = e.clientX - rect.left;
    const cursorY = e.clientY - rect.top;

    setTransform((prev) => {
      const delta = e.deltaY > 0 ? 0.85 : 1.18;
      const newScale = Math.min(10, Math.max(0.5, prev.scale * delta));

      // Keep the point under the cursor fixed
      const newX = cursorX - (cursorX - prev.x) * (newScale / prev.scale);
      const newY = cursorY - (cursorY - prev.y) * (newScale / prev.scale);

      return { x: newX, y: newY, scale: newScale };
    });
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startY: e.clientY, tx: transform.x, ty: transform.y };
  }, [transform.x, transform.y]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setTransform((prev) => ({ ...prev, x: dragRef.current!.tx + dx, y: dragRef.current!.ty + dy }));
  }, []);

  const onMouseUp = useCallback(() => { dragRef.current = null; }, []);

  const onDblClick = useCallback(() => {
    setTransform({ x: 0, y: 0, scale: 1 });
  }, []);

  return (
    <div className={`relative overflow-hidden ${className}`}>
      {/* Controls hint */}
      <div className="absolute top-1 right-2 z-10 text-xs text-muted pointer-events-none select-none">
        Scroll to zoom · drag to pan · dbl-click to reset
      </div>
      <div
        ref={containerRef}
        className="w-full cursor-grab active:cursor-grabbing"
        style={{ userSelect: "none" }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onDoubleClick={onDblClick}
      >
        <div
          style={{
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
            transformOrigin: "0 0",
            willChange: "transform",
          }}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
