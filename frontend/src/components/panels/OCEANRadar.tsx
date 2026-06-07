import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, OCEAN_LABELS, PLOTLY_CONFIG } from "@/lib/theme";

const DIMS = ["O", "C", "E", "A", "N"] as const;

/** Aggregate OCEAN distribution across the simulated population (radar). */
export function OCEANRadar() {
  const oceanMean = useSessionStore((s) => s.oceanMean);
  const personas = useSessionStore((s) => s.personasSimulated);

  const data = useMemo<Data[]>(() => {
    if (!oceanMean) return [];
    const r = DIMS.map((d) => oceanMean[d]);
    const theta = DIMS.map((d) => OCEAN_LABELS[d]);
    return [
      {
        type: "scatterpolar",
        r: [...r, r[0]],
        theta: [...theta, theta[0]],
        fill: "toself",
        fillcolor: "rgba(0,180,216,0.18)",
        line: { color: COLORS.teal, width: 2 },
        marker: { color: COLORS.teal, size: 6 },
        hovertemplate: "%{theta}: %{r:.1f}<extra></extra>",
      },
    ];
  }, [oceanMean]);

  if (!oceanMean) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting OCEAN aggregation
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <Plot
        data={data}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        layout={{
          ...basePlotlyLayout(),
          margin: { l: 30, r: 30, t: 24, b: 24 },
          polar: {
            bgcolor: "rgba(0,0,0,0)",
            radialaxis: {
              range: [0, 100],
              gridcolor: COLORS.grid,
              linecolor: COLORS.grid,
              tickfont: { size: 8, color: COLORS.muted },
              angle: 90,
            },
            angularaxis: {
              gridcolor: COLORS.grid,
              linecolor: COLORS.grid,
              tickfont: { size: 9, color: COLORS.data },
            },
          },
        }}
        config={PLOTLY_CONFIG}
      />
      <span className="pointer-events-none absolute bottom-1 right-2 font-mono text-2xs text-muted">
        n={personas}
      </span>
    </div>
  );
}
