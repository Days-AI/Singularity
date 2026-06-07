import { useEffect, useRef, useState } from "react";
import { useSessionStore, type EvidenceEntry } from "@/store/sessionStore";
import { COLORS } from "@/lib/theme";

const ROW_H = 58;
const OVERSCAN = 4;

const SOURCE_TONE: Record<string, string> = {
  yFinance: "text-positive",
  DuckDuckGo: "text-teal",
  "Google News": "text-orange",
  Wikipedia: "text-data",
  "IPIP-300": "text-orange",
  "TimesFM-ICF": "text-positive",
  "TimesFM+Prophet-ICF": "text-positive",
  Serper: "text-teal",
  GDELT: "text-orange",
  arXiv: "text-data",
  Parallel: "text-teal",
  Simulation: "text-positive",
  PsychometricEngine: "text-teal",
};

const SENTIMENT_POSITIVE = 0.15;
const SENTIMENT_NEGATIVE = -0.15;

type SentimentTone = { label: string; color: string };

function sentimentTone(sentiment: number | undefined): SentimentTone {
  if (sentiment == null) return { label: "n/a", color: COLORS.muted };
  if (sentiment >= SENTIMENT_POSITIVE)
    return { label: "positive", color: COLORS.positive };
  if (sentiment <= SENTIMENT_NEGATIVE)
    return { label: "negative", color: COLORS.alert };
  return { label: "neutral", color: COLORS.orange };
}

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
  const end = Math.min(evidence.length, Math.ceil((scrollTop + viewport) / ROW_H) + OVERSCAN);
  const slice = evidence.slice(start, end);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-1">
      <div
        ref={scrollRef}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        className="min-h-0 flex-1 overflow-auto"
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
    </div>
  );
}

function Row({ item, top }: { item: EvidenceEntry; top: number }) {
  const tone = SOURCE_TONE[item.source] ?? "text-data";
  const sentiment = sentimentTone(item.sentiment);
  return (
    <div
      className="absolute left-0 right-0 border-b border-[color:var(--hairline)] py-1.5 pl-2.5 pr-1"
      style={{
        top,
        height: ROW_H,
        borderLeft: `3px solid ${sentiment.color}`,
        background: `linear-gradient(90deg, ${sentiment.color}1f 0%, transparent 60%)`,
      }}
    >
      <div className="flex items-center justify-between gap-1">
        <span className={`shrink-0 font-mono text-2xs font-semibold uppercase tracking-wider ${tone}`}>
          {item.source}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {item.sentiment != null && (
            <span
              className="font-mono text-2xs tabular-nums"
              style={{ color: sentiment.color }}
              title={`sentiment ${sentiment.label}`}
            >
              {item.sentiment >= 0 ? "+" : ""}
              {item.sentiment.toFixed(2)}
            </span>
          )}
          <span className="font-mono text-2xs tabular-nums text-muted">
            conf {(item.confidence * 100).toFixed(0)}%
          </span>
        </div>
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
