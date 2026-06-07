import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";

/** Round timeline, narrative chips, and polarization sparkline from social simulation. */
export function SocialInteractionPanel() {
  const ticks = useSessionStore((s) => s.socialTicks);
  const simulation = useSessionStore((s) => s.socialSimulation);

  const sparkData = useMemo<Data[]>(() => {
    if (ticks.length < 1) return [];
    return [
      {
        x: ticks.map((t) => `R${t.round}`),
        y: ticks.map((t) => t.polarization_index * 100),
        type: "scatter" as const,
        mode: "lines+markers" as const,
        line: { color: COLORS.orange, width: 1.6 },
        marker: { size: 5, color: COLORS.orange },
        hovertemplate: "Round %{x}<br>Polarization %{y:.0f}%<extra></extra>",
      },
    ];
  }, [ticks]);

  const narratives = simulation?.final_narratives ?? ticks[ticks.length - 1]?.narratives ?? [];

  if (ticks.length === 0 && !simulation) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting social interaction layer
      </div>
    );
  }

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_auto_minmax(0,1fr)] gap-1 overflow-hidden p-1">
      <div className="grid shrink-0 grid-cols-3 gap-1 font-mono text-2xs">
        <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-2 py-1">
          <span className="text-muted">Rounds</span>
          <p className="text-data">{simulation?.rounds_completed ?? ticks.length}</p>
        </div>
        <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-2 py-1">
          <span className="text-muted">Contagion</span>
          <p className="text-teal">
            {((simulation?.contagion_index ?? 0) * 100).toFixed(0)}%
          </p>
        </div>
        <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-2 py-1">
          <span className="text-muted">Polarization</span>
          <p className="text-orange">
            {((simulation?.polarization_index ?? ticks[ticks.length - 1]?.polarization_index ?? 0) * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {narratives.length > 0 && (
        <div className="shrink-0">
          <p className="mb-1 font-mono text-2xs uppercase tracking-wider text-muted">Narratives</p>
          <div className="flex flex-wrap gap-1">
            {narratives.slice(0, 6).map((n) => (
              <span
                key={n.narrative_id}
                className="rounded-sm border border-[color:var(--hairline)] bg-bg/50 px-1.5 py-0.5 font-mono text-2xs"
                title={`Sentiment ${n.sentiment.toFixed(2)}`}
              >
                {n.label}{" "}
                <span className="text-teal">{n.adoption_pct.toFixed(0)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {ticks.length > 0 && (
        <div className="min-h-0 flex-1">
          <Plot
            data={sparkData}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
            layout={{
              ...basePlotlyLayout(),
              margin: { l: 36, r: 8, t: 4, b: 28 },
              yaxis: {
                gridcolor: COLORS.grid,
                tickfont: { size: 7, color: COLORS.muted },
                title: { text: "Polarization %", font: { size: 8, color: COLORS.muted } },
              },
              xaxis: {
                gridcolor: COLORS.grid,
                tickfont: { size: 7, color: COLORS.muted },
              },
            }}
            config={PLOTLY_CONFIG}
          />
        </div>
      )}
    </div>
  );
}
