import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";

const SOURCE_PALETTE = [
  COLORS.teal,
  COLORS.orange,
  COLORS.positive,
  "#9B6DFF",
  COLORS.alert,
  "#FFD54F",
];

/** Donut chart of evidence items grouped by web/financial source. */
export function SourceBreakdownChart() {
  const evidence = useSessionStore((s) => s.evidence);

  const { data, labels } = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of evidence) {
      counts.set(e.source, (counts.get(e.source) ?? 0) + 1);
    }
    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    return {
      labels: entries.map(([k]) => k),
      data: entries.map(([, v]) => v),
    };
  }, [evidence]);

  const plotData = useMemo<Data[]>(() => {
    if (!data.length) return [];
    return [
      {
        type: "pie",
        labels,
        values: data,
        hole: 0.55,
        marker: {
          colors: labels.map((_, i) => SOURCE_PALETTE[i % SOURCE_PALETTE.length]),
        },
        textfont: { color: COLORS.data, size: 9 },
        hovertemplate: "%{label}<br>%{value} items<extra></extra>",
      },
    ];
  }, [data, labels]);

  if (!evidence.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting evidence sources
      </div>
    );
  }

  return (
    <div className="h-full min-h-0 p-1">
      <Plot
        data={plotData}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        layout={{
          ...basePlotlyLayout(),
          margin: { l: 8, r: 8, t: 8, b: 8 },
          showlegend: true,
          legend: {
            font: { size: 8, color: COLORS.muted },
            bgcolor: "rgba(0,0,0,0)",
          },
        }}
        config={PLOTLY_CONFIG}
      />
    </div>
  );
}
