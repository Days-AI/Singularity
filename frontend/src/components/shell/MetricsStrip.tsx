import { useSessionStore } from "@/store/sessionStore";
import { fmtNum } from "@/lib/format";

/** Bottom ticker-style strip summarizing the live simulation metrics. */
export function MetricsStrip() {
  const dagNodes = useSessionStore((s) => s.dagNodes);
  const nodeStatus = useSessionStore((s) => s.nodeStatus);
  const personas = useSessionStore((s) => s.personasSimulated);
  const personaTarget = useSessionStore((s) => s.personaTarget);
  const forecast = useSessionStore((s) => s.forecast);
  const causal = useSessionStore((s) => s.causal);
  const evidence = useSessionStore((s) => s.evidence);
  const oceanMean = useSessionStore((s) => s.oceanMean);

  const resolved = Object.values(nodeStatus).filter((v) => v === "done").length;
  const sigEdges = causal
    ? causal.edges.filter((e) => e.p_value < 0.05).length
    : 0;
  const netSent = oceanMean ? (oceanMean.E - oceanMean.N) / 100 : null;

  const items: { label: string; value: string; tone?: string }[] = [
    {
      label: "NODES RESOLVED",
      value: `${resolved}/${dagNodes.length || "-"}`,
    },
    {
      label: "PERSONAS",
      value: `${fmtNum(personas)}/${fmtNum(personaTarget)}`,
    },
    { label: "EVIDENCE ITEMS", value: fmtNum(evidence.length) },
    {
      label: "MASE",
      value: forecast ? forecast.mase_score.toFixed(2) : "--",
      tone: "text-positive",
    },
    {
      label: "SIG. CAUSAL EDGES",
      value: causal ? String(sigEdges) : "--",
    },
    {
      label: "NET AFFECT",
      value: netSent == null ? "--" : netSent.toFixed(2),
      tone:
        netSent == null
          ? undefined
          : netSent >= 0
            ? "text-positive"
            : "text-alert",
    },
  ];

  return (
    <footer className="flex h-9 shrink-0 items-center gap-6 overflow-hidden border-t border-[color:var(--hairline)] bg-panel px-4">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-2 whitespace-nowrap">
          <span className="text-2xs uppercase tracking-widest text-muted">
            {it.label}
          </span>
          <span className={`stat-value text-xs ${it.tone ?? "text-data"}`}>
            {it.value}
          </span>
        </div>
      ))}
      <div className="ml-auto flex items-center gap-2 whitespace-nowrap">
        <span className="h-1.5 w-1.5 rounded-full bg-teal animate-pulse-stream" />
        <span className="text-2xs uppercase tracking-widest text-muted">
          Kerala Startup Mission // LeapX
        </span>
      </div>
    </footer>
  );
}
