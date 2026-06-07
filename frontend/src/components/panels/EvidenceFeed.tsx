import { useEffect, useRef, useState } from "react";
import { useSessionStore, type EvidenceEntry } from "@/store/sessionStore";

const ROW_H = 58; // px, fixed-height rows enable cheap virtualization
const OVERSCAN = 4;

const SOURCE_TONE: Record<string, string> = {
  yFinance: "text-positive",
  DuckDuckGo: "text-teal",
  "Google News": "text-orange",
  Wikipedia: "text-data",
  Pytrends: "text-teal",
  "IPIP-300": "text-orange",
  "TimesFM-ICF": "text-positive",
};

/**
 * Streaming evidence feed. Fixed-height rows + a windowed render keep it smooth
 * even as agent_result events accumulate (store caps at 200 entries).
 */
export function EvidenceFeed() {
  const evidence = useSessionStore((s) => s.evidence);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewport(el.clientHeight));
    ro.observe(el);
    setViewport(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  if (!evidence.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        no evidence ingested yet
      </div>
    );
  }

  const total = evidence.length * ROW_H;
  const start = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const end = Math.min(
    evidence.length,
    Math.ceil((scrollTop + viewport) / ROW_H) + OVERSCAN
  );
  const slice = evidence.slice(start, end);

  return (
    <div
      ref={scrollRef}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      className="h-full w-full overflow-auto"
    >
      <div style={{ height: total, position: "relative" }}>
        {slice.map((item, i) => (
          <Row
            key={`${item.agentId}-${start + i}-${item.title}`}
            item={item}
            top={(start + i) * ROW_H}
          />
        ))}
      </div>
    </div>
  );
}

function Row({ item, top }: { item: EvidenceEntry; top: number }) {
  const tone = SOURCE_TONE[item.source] ?? "text-data";
  return (
    <div
      className="absolute left-0 right-0 border-b border-[color:var(--hairline)] px-1 py-1.5"
      style={{ top, height: ROW_H }}
    >
      <div className="flex items-center justify-between">
        <span className={`font-mono text-2xs font-semibold uppercase tracking-wider ${tone}`}>
          {item.source}
        </span>
        <span className="font-mono text-2xs tabular-nums text-muted">
          conf {(item.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <div className="truncate font-mono text-xs text-data">
        {item.title}
        {item.value != null && (
          <span className="ml-1 text-orange">
            {item.value}
            {item.unit ? ` ${item.unit}` : ""}
          </span>
        )}
      </div>
      <div className="truncate font-mono text-2xs text-muted">{item.detail}</div>
    </div>
  );
}
