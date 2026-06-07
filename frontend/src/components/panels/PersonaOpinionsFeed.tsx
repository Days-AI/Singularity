import { useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useSessionStore } from "@/store/sessionStore";
import { COLORS, OCEAN_COLORS } from "@/lib/theme";
import type { FacetScore, PersonaOpinion } from "@/types/events";

const CLUSTER_COLORS: Record<string, string> = {
  Skeptics: COLORS.alert,
  Pragmatists: COLORS.orange,
  Enthusiasts: COLORS.positive,
};

const BAND_COLORS: Record<FacetScore["band"], string> = {
  high: COLORS.teal,
  low: COLORS.orange,
  moderate: COLORS.muted,
};

function oceanBandColor(score: number): string {
  if (score >= 66) return COLORS.teal;
  if (score <= 33) return COLORS.orange;
  return COLORS.muted;
}

function FacetChip({ facet }: { facet: FacetScore }) {
  const color = BAND_COLORS[facet.band] ?? COLORS.muted;
  return (
    <span
      className="rounded-sm px-1 py-0.5 font-mono text-2xs uppercase tracking-wide"
      style={{ color, backgroundColor: `${color}1f` }}
      title={`${facet.band} ${facet.name}`}
    >
      {facet.name} {facet.score.toFixed(0)}
    </span>
  );
}

function OceanScoreStrip({ ocean }: { ocean: PersonaOpinion["ocean"] }) {
  const dims: Array<[string, number]> = [
    ["O", ocean.O],
    ["C", ocean.C],
    ["E", ocean.E],
    ["A", ocean.A],
    ["N", ocean.N],
  ];
  return (
    <div className="flex flex-wrap items-center gap-1.5 font-mono text-2xs">
      {dims.map(([dim, score]) => (
        <span key={dim} style={{ color: oceanBandColor(score) }}>
          <span style={{ color: OCEAN_COLORS[dim] }}>{dim}</span>
          {score.toFixed(0)}
        </span>
      ))}
    </div>
  );
}

function PopulationHeader({
  population,
  target,
}: {
  population: number;
  target: number;
}) {
  const pct = target > 0 ? Math.min(100, Math.round((population / target) * 100)) : 0;
  return (
    <div className="shrink-0 space-y-2 border-b border-[color:var(--hairline)] pb-2">
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="font-mono text-2xs uppercase tracking-widest text-teal">
            Synthetic World Environment
          </p>
          <p className="font-mono text-2xs text-muted">
            Cognitive agents · {population.toLocaleString()}/{target.toLocaleString()} active
          </p>
        </div>
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-bg">
        <div
          className="h-full bg-teal/70 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function PopulationFeed({
  opinions,
  filter,
  onFilter,
}: {
  opinions: PersonaOpinion[];
  filter: string;
  onFilter: (f: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const parentRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    if (filter === "all") return opinions;
    return opinions.filter((o) => o.cluster_label === filter);
  }, [opinions, filter]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 110,
    overscan: 8,
  });

  if (!opinions.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        agents entering the world…
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="flex shrink-0 flex-wrap items-center gap-1.5">
        {(["all", "Skeptics", "Pragmatists", "Enthusiasts"] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => onFilter(f)}
            className={`rounded-sm border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider transition-colors ${
              filter === f
                ? "border-teal/60 bg-teal/15 text-teal"
                : "border-[color:var(--hairline)] text-muted hover:text-data"
            }`}
          >
            {f === "all" ? `All (${opinions.length})` : f}
          </button>
        ))}
      </div>
      <div ref={parentRef} className="min-h-0 flex-1 overflow-y-auto pr-1">
        <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
          {virtualizer.getVirtualItems().map((row) => {
            const op = filtered[row.index];
            const accent = CLUSTER_COLORS[op.cluster_label] ?? COLORS.teal;
            const facets = op.top_facets ?? [];
            return (
              <div
                key={op.id}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  transform: `translateY(${row.start}px)`,
                }}
              >
                <button
                  type="button"
                  onClick={() => setExpanded(expanded === op.id ? null : op.id)}
                  className="mb-1 w-full rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-2 py-1.5 text-left transition-colors hover:border-teal/30"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-2xs text-muted">{op.id}</span>
                    <div className="flex items-center gap-1">
                      {op.response_source === "llm" && (
                        <span className="rounded-sm border border-teal/40 px-1 font-mono text-2xs text-teal">
                          LLM
                        </span>
                      )}
                      <span
                        className="rounded-sm px-1.5 py-0.5 font-mono text-2xs uppercase"
                        style={{ color: accent, backgroundColor: `${accent}22` }}
                      >
                        {op.cluster_label}
                      </span>
                    </div>
                  </div>
                  <p className="mt-0.5 truncate font-mono text-xs italic text-data">
                    &ldquo;{op.comment || op.behavioral_intent}&rdquo;
                  </p>
                  <div className="mt-1">
                    <OceanScoreStrip ocean={op.ocean} />
                  </div>
                  {facets.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1 overflow-hidden">
                      {facets.map((f) => (
                        <FacetChip key={f.name} facet={f} />
                      ))}
                    </div>
                  )}
                  {expanded === op.id && (
                    <div className="mt-1.5 space-y-1 border-t border-[color:var(--hairline)] pt-1.5">
                      <p className="font-mono text-2xs text-muted">
                        {op.emotional_state} · sentiment {op.sentiment >= 0 ? "+" : ""}
                        {op.sentiment.toFixed(2)} · action {(op.action_likelihood * 100).toFixed(0)}%
                        {op.stance_confidence != null && (
                          <> · confidence {(op.stance_confidence * 100).toFixed(0)}%</>
                        )}
                      </p>
                      {op.behavioral_intent && (
                        <p className="font-mono text-2xs text-muted/80">intent: {op.behavioral_intent}</p>
                      )}
                      {op.key_concerns.length > 0 && (
                        <p className="font-mono text-2xs text-orange">{op.key_concerns.join(" · ")}</p>
                      )}
                      {op.active_biases && op.active_biases.length > 0 && (
                        <p className="font-mono text-2xs text-teal/80">
                          biases: {op.active_biases.slice(0, 4).join(", ")}
                        </p>
                      )}
                    </div>
                  )}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** World environment: 1,500 IPIP personas population feed. */
export function PersonaOpinionsFeed() {
  const opinions = useSessionStore((s) => s.personaOpinions);
  const population = useSessionStore((s) => s.personasSimulated);
  const target = useSessionStore((s) => s.personaTarget);
  const [filter, setFilter] = useState("all");

  if (!opinions.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        booting synthetic world…
      </div>
    );
  }

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-1 overflow-hidden p-1">
      <PopulationHeader population={population || opinions.length} target={target} />
      <PopulationFeed opinions={opinions} filter={filter} onFilter={setFilter} />
    </div>
  );
}
