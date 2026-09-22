import { useEffect, useState } from "react";
import { fetchNodeDetail, fetchPaths } from "@/api/causalIntel";
import { COLORS } from "@/lib/theme";
import type { NodeDetail } from "@/types/causalIntel";

interface NodeInspectorProps {
  nodeId: string | null;
  sessionId: string | null;
  onClose: () => void;
}

export function NodeInspector({ nodeId, sessionId, onClose }: NodeInspectorProps) {
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [paths, setPaths] = useState<string[][]>([]);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<"overview" | "paths" | "explain">("overview");

  useEffect(() => {
    if (!nodeId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    fetchNodeDetail(nodeId, sessionId, tab === "explain")
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [nodeId, sessionId, tab]);

  useEffect(() => {
    if (!nodeId || !sessionId) return;
    fetchPaths(sessionId, nodeId)
      .then((p) => setPaths(p.labels))
      .catch(() => setPaths([]));
  }, [nodeId, sessionId]);

  if (!nodeId) return null;

  return (
    <aside className="flex h-full w-52 shrink-0 flex-col border-l border-[color:var(--hairline)] bg-panel/95">
      <header className="flex items-center justify-between border-b border-[color:var(--hairline)] px-2 py-1">
        <span className="font-mono text-[10px] uppercase tracking-wider text-teal">Inspector</span>
        <button
          type="button"
          onClick={onClose}
          className="font-mono text-xs text-muted hover:text-data"
          aria-label="Close inspector"
        >
          ×
        </button>
      </header>

      <div className="flex shrink-0 gap-0.5 border-b border-[color:var(--hairline)] p-0.5">
        {(["overview", "paths", "explain"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`flex-1 rounded-sm px-1 py-0.5 font-mono text-[9px] uppercase ${
              tab === t ? "bg-teal/15 text-teal" : "text-muted hover:text-data"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2 font-mono text-[10px]">
        {loading && <p className="text-muted">loading…</p>}
        {!loading && detail && tab === "overview" && (
          <div className="space-y-2">
            <div>
              <p className="font-semibold text-data">{detail.node.label}</p>
              <p className="mt-0.5 text-muted">{detail.node.question}</p>
            </div>
            <div className="grid grid-cols-2 gap-1">
              <Stat label="Probability" value={`${detail.node.probability.toFixed(1)}%`} color={COLORS.orange} />
              <Stat label="Confidence" value={`${(detail.node.confidence * 100).toFixed(0)}%`} />
            </div>
            {detail.incoming.length > 0 && (
              <Section title="Incoming">
                {detail.incoming.map((e) => (
                  <Factor key={e.id} label={e.source_label} badge={e.label} positive={e.polarity === "positive"} />
                ))}
              </Section>
            )}
            {detail.outgoing.length > 0 && (
              <Section title="Outgoing">
                {detail.outgoing.map((e) => (
                  <Factor key={e.id} label={e.target_label} badge={e.label} positive={e.polarity === "positive"} />
                ))}
              </Section>
            )}
            {detail.evidence.length > 0 && (
              <Section title="Evidence">
                {detail.evidence.map((e) => (
                  <p key={e.index} className="truncate text-data/90">
                    [{e.source}] {e.title}
                  </p>
                ))}
              </Section>
            )}
          </div>
        )}
        {!loading && tab === "paths" && (
          <div className="space-y-1.5">
            {paths.length === 0 && <p className="text-muted">No paths to goal</p>}
            {paths.map((p, i) => (
              <p key={i} className="rounded-sm border border-[color:var(--hairline)] bg-bg/30 px-1 py-0.5 text-data/90">
                {p.join(" → ")}
              </p>
            ))}
          </div>
        )}
        {!loading && tab === "explain" && (
          <p className="leading-relaxed text-data/90">
            {detail?.explanation ?? "Explanation unavailable."}
          </p>
        )}
      </div>
    </aside>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/30 px-1 py-0.5">
      <span className="text-muted">{label}</span>
      <p className="font-semibold" style={{ color: color ?? "var(--color-data)" }}>
        {value}
      </p>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="mb-0.5 text-[9px] uppercase tracking-wider text-muted">{title}</p>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

function Factor({
  label,
  badge,
  positive,
}: {
  label: string;
  badge: string;
  positive: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-1 truncate">
      <span className="truncate text-data">{label}</span>
      <span style={{ color: positive ? COLORS.teal : COLORS.alert }}>{badge}</span>
    </div>
  );
}
