import { useEffect, useRef, useState } from "react";

export interface Size {
  width: number;
  height: number;
}

/** Tracks the content-box size of a referenced element via ResizeObserver. */
export function useResize<T extends HTMLElement>(): [
  React.RefObject<T>,
  Size,
] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const next = {
        width: Math.round(width),
        height: Math.round(height),
      };
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        raf = 0;
        setSize((prev) =>
          prev.width === next.width && prev.height === next.height ? prev : next
        );
      });
    });
    ro.observe(el);
    return () => {
      if (raf) cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return [ref, size];
}
