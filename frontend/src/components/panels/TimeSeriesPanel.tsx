import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";

/**
 * Multi-segment forecast: historical actuals, point forecast, and a shaded 90%
 * prediction interval band (drawn as an upper trace + filled lower trace).
 */
export function TimeSeriesPanel() {
  const forecast = useSessionStore((s) => s.forecast);

  const data = useMemo<Data[]>(() => {
    if (!forecast) return [];
    const histX = forecast.history.map((p) => p.date);
    const histY = forecast.history.map((p) => p.value);
    const predX = forecast.predictions.map((p) => p.date);
    const predY = forecast.predictions.map((p) => p.value);
    const bandX = forecast.intervals.map((p) => p.date);
    const upper = forecast.intervals.map((p) => p.upper);
    const lower = forecast.intervals.map((p) => p.lower);

    return [
      {
        x: bandX,
        y: upper,
        type: "scatter",
        mode: "lines",
        line: { width: 0 },
        hoverinfo: "skip",
        showlegend: false,
      },
      {
        x: bandX,
        y: lower,
        type: "scatter",
        mode: "lines",
        line: { width: 0 },
        fill: "tonexty",
        fillcolor: "rgba(245,166,35,0.16)",
        hoverinfo: "skip",
        showlegend: false,
      },
      {
        x: histX,
        y: histY,
        type: "scatter",
        mode: "lines",
        name: "actual",
        line: { color: COLORS.teal, width: 1.6 },
        hovertemplate: "%{x}<br>%{y:.1f}<extra>actual</extra>",
      },
      {
        x: [histX[histX.length - 1], ...predX],
        y: [histY[histY.length - 1], ...predY],
        type: "scatter",
        mode: "lines",
        name: "forecast",
        line: { color: COLORS.orange, width: 1.8, dash: "dot" },
        hovertemplate: "%{x}<br>%{y:.1f}<extra>forecast</extra>",
      },
    ];
  }, [forecast]);

  if (!forecast) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting forecast
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
          margin: { l: 36, r: 12, t: 8, b: 28 },
          xaxis: {
            gridcolor: COLORS.grid,
            linecolor: COLORS.grid,
            zeroline: false,
            tickfont: { size: 8, color: COLORS.muted },
            nticks: 6,
          },
          yaxis: {
            gridcolor: COLORS.grid,
            linecolor: COLORS.grid,
            zeroline: false,
            tickfont: { size: 8, color: COLORS.muted },
          },
          shapes: [
            {
              type: "line",
              x0: forecast.history[forecast.history.length - 1]?.date,
              x1: forecast.history[forecast.history.length - 1]?.date,
              y0: 0,
              y1: 1,
              yref: "paper",
              line: { color: COLORS.muted, width: 1, dash: "dash" },
            },
          ],
        }}
        config={PLOTLY_CONFIG}
      />
      <div className="pointer-events-none absolute right-2 top-1 flex gap-3 font-mono text-2xs">
        <span className="text-teal">actual</span>
        <span className="text-orange">forecast +90d</span>
        <span className="text-muted">MASE {forecast.mase_score.toFixed(2)}</span>
      </div>
    </div>
  );
}
