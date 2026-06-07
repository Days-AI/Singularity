import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { useSessionStore } from "@/store/sessionStore";
import {
  basePlotlyLayout,
  COLORS,
  OCEAN_COLORS,
  OCEAN_LABELS,
  PLOTLY_CONFIG,
} from "@/lib/theme";

const DIMS = ["O", "C", "E", "A", "N"] as const;

function bandLabel(score: number): string {
  if (score >= 66) return "high";
  if (score <= 33) return "low";
  return "mod";
}

function DimensionChip({
  dim,
  score,
}: {
  dim: (typeof DIMS)[number];
  score: number;
}) {
  const color = OCEAN_COLORS[dim];
  return (
    <div
      className="flex min-h-[2.5rem] flex-col justify-center rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-1 py-0.5"
      style={{ borderTopColor: color, borderTopWidth: 2 }}
      title={`${OCEAN_LABELS[dim]} — ${bandLabel(score)}`}
    >
      <span className="font-mono text-[10px] font-semibold leading-none" style={{ color }}>
        {dim}
      </span>
      <p className="font-mono text-xs font-medium leading-tight text-data">{score.toFixed(0)}</p>
    </div>
  );
}

/** Aggregate OCEAN distribution across the simulated population (radar + dimension strip). */
export function OCEANRadar() {
  const oceanMean = useSessionStore((s) => s.oceanMean);
  const personas = useSessionStore((s) => s.personasSimulated);
  const personaTarget = useSessionStore((s) => s.personaTarget);

  const data = useMemo<Data[]>(() => {
    if (!oceanMean) return [];
    const r = DIMS.map((d) => oceanMean[d]);
    const theta = [...DIMS];
    return [
      {
        type: "scatterpolar",
        r: [...r, r[0]],
        theta: [...theta, theta[0]],
        fill: "toself",
        fillcolor: "rgba(0,180,216,0.15)",
        line: { color: COLORS.teal, width: 1.8 },
        marker: { color: COLORS.teal, size: 5 },
        hovertemplate: "%{theta}: %{r:.0f}<extra>%{customdata}</extra>",
        customdata: [
          ...DIMS.map((d) => OCEAN_LABELS[d]),
          OCEAN_LABELS[theta[0]],
        ],
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
    <div className="grid h-full min-h-0 grid-rows-[auto_auto_1fr] gap-1 overflow-hidden p-1">
      <div className="flex shrink-0 items-center justify-between gap-2 font-mono text-[10px]">
        <span className="truncate uppercase tracking-wider text-muted">Population mean</span>
        <span className="shrink-0 text-muted">
          n={personas.toLocaleString()}
          {personaTarget > 0 && personas < personaTarget
            ? ` / ${personaTarget.toLocaleString()}`
            : ""}
        </span>
      </div>

      <div className="grid shrink-0 grid-cols-5 gap-0.5">
        {DIMS.map((d) => (
          <DimensionChip key={d} dim={d} score={oceanMean[d]} />
        ))}
      </div>

      <div className="relative min-h-0 min-w-0">
        <Plot
          data={data}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          layout={{
            ...basePlotlyLayout(),
            margin: { l: 12, r: 12, t: 8, b: 8 },
            polar: {
              bgcolor: "rgba(0,0,0,0)",
              radialaxis: {
                range: [0, 100],
                gridcolor: COLORS.grid,
                linecolor: COLORS.grid,
                tickfont: { size: 7, color: COLORS.muted },
                angle: 90,
                nticks: 4,
                showline: false,
              },
              angularaxis: {
                gridcolor: COLORS.grid,
                linecolor: COLORS.grid,
                tickfont: { size: 9, color: COLORS.data },
                direction: "clockwise",
                rotation: 90,
              },
            },
          }}
          config={PLOTLY_CONFIG}
        />
      </div>
    </div>
  );
}
