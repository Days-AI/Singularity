import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import Plot from "@/lib/plotly";
import { basePlotlyLayout, COLORS, OCEAN_LABELS, PLOTLY_CONFIG } from "@/lib/theme";
import { OutcomeGaugeSvg } from "@/lib/gauge";
import type { ReportModel } from "@/lib/reportAnalytics";
import { CaptureCard, ChartEmpty } from "./CaptureCard";

const PLOT_STYLE = { width: "100%", height: "100%" } as const;
const CLUSTER_PALETTE = [COLORS.teal, COLORS.orange, COLORS.positive, "#9B6DFF", COLORS.alert];

function layout(extra: Partial<Layout> = {}): Partial<Layout> {
  return { ...basePlotlyLayout(), ...extra };
}

/* ---- Behavioral / sentiment ------------------------------------------------ */

export function SentimentDistributionChart({ model }: { model: ReportModel }) {
  const values = model.personaPoints.map((p) => p.sentiment);
  const data = useMemo<Data[]>(
    () => [
      {
        type: "histogram",
        x: values,
        nbinsx: 21,
        marker: { color: COLORS.teal, line: { color: COLORS.bg, width: 1 } },
        hovertemplate: "sentiment %{x:.2f}<br>n=%{y}<extra></extra>",
      },
    ],
    [values]
  );
  return (
    <CaptureCard title="Sentiment Distribution" subtitle={`n=${model.personaPoints.length}`}>
      {values.length ? (
        <Plot
          data={data}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 36, r: 12, t: 8, b: 30 },
            bargap: 0.05,
            xaxis: { title: { text: "sentiment (-1..1)" }, gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } },
            yaxis: { gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } },
          })}
        />
      ) : (
        <ChartEmpty label="awaiting persona responses" />
      )}
    </CaptureCard>
  );
}

export function OceanRadarChart({ model }: { model: ReportModel }) {
  const dims = ["O", "C", "E", "A", "N"] as const;
  const om = model.oceanMean;
  const data = useMemo<Data[]>(() => {
    if (!om) return [];
    const r = dims.map((d) => om[d]);
    const theta = dims.map((d) => OCEAN_LABELS[d]);
    return [
      {
        type: "scatterpolar",
        r: [...r, r[0]],
        theta: [...theta, theta[0]],
        fill: "toself",
        fillcolor: "rgba(0,180,216,0.18)",
        line: { color: COLORS.teal, width: 2 },
        hovertemplate: "%{theta}: %{r:.1f}<extra></extra>",
      },
    ];
  }, [om]);
  return (
    <CaptureCard title="OCEAN Personality Profile">
      {om ? (
        <Plot
          data={data}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 40, r: 40, t: 20, b: 24 },
            polar: {
              bgcolor: "rgba(0,0,0,0)",
              radialaxis: { range: [0, 100], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } },
              angularaxis: { gridcolor: COLORS.grid, tickfont: { size: 9, color: COLORS.data } },
            },
          })}
        />
      ) : (
        <ChartEmpty label="awaiting OCEAN aggregation" />
      )}
    </CaptureCard>
  );
}

export function ClusterDonutChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "pie",
        hole: 0.55,
        labels: model.clusters.map((c) => c.label),
        values: model.clusters.map((c) => c.size),
        marker: { colors: CLUSTER_PALETTE },
        textfont: { color: COLORS.data, size: 10 },
        hovertemplate: "%{label}<br>%{value} (%{percent})<extra></extra>",
      },
    ],
    [model.clusters]
  );
  return (
    <CaptureCard title="Population Segmentation">
      {model.clusters.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 8, r: 8, t: 8, b: 8 }, showlegend: true, legend: { font: { size: 9, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="awaiting clusters" />
      )}
    </CaptureCard>
  );
}

export function ClusterSentimentChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        x: model.clusters.map((c) => c.label),
        y: model.clusters.map((c) => c.meanSentiment),
        marker: { color: model.clusters.map((c) => (c.meanSentiment >= 0 ? COLORS.positive : COLORS.alert)) },
        hovertemplate: "%{x}<br>sentiment %{y:.2f}<extra></extra>",
      },
    ],
    [model.clusters]
  );
  return (
    <CaptureCard title="Sentiment by Segment">
      {model.clusters.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 36, r: 12, t: 8, b: 36 }, xaxis: { tickfont: { size: 8, color: COLORS.muted } }, yaxis: { range: [-1, 1], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="awaiting clusters" />
      )}
    </CaptureCard>
  );
}

export function AdoptionFunnelChart({ model }: { model: ReportModel }) {
  const aware = 1;
  const interested = model.positiveShare || 0;
  const intent = (model.adoptionRate + (model.positiveShare || 0)) / 2;
  const likely = model.adoptionRate;
  const data = useMemo<Data[]>(
    () =>
      [
        {
          type: "funnel",
          y: ["Aware", "Interested", "Intent", "Likely to Act"],
          x: [aware, interested, intent, likely].map((v) => Math.round(v * 100)),
          marker: { color: [COLORS.teal, "#1f9bbf", COLORS.orange, COLORS.positive] },
          textinfo: "value+percent initial",
          hovertemplate: "%{y}: %{x}%<extra></extra>",
        } as unknown as Data,
      ],
    [aware, interested, intent, likely]
  );
  return (
    <CaptureCard title="Adoption Funnel" subtitle="% of audience">
      {model.sampleResponses ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 80, r: 16, t: 8, b: 16 } })} />
      ) : (
        <ChartEmpty label="awaiting responses" />
      )}
    </CaptureCard>
  );
}

export function ConcernsChart({ model }: { model: ReportModel }) {
  const sorted = [...model.topConcerns].reverse();
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: sorted.map((c) => c.weight),
        y: sorted.map((c) => c.label),
        marker: { color: COLORS.orange },
        hovertemplate: "%{y}: %{x}<extra></extra>",
      },
    ],
    [sorted]
  );
  return (
    <CaptureCard title="Top Audience Concerns">
      {model.topConcerns.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 120, r: 12, t: 8, b: 24 }, xaxis: { gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } }, yaxis: { tickfont: { size: 9, color: COLORS.data } } })} />
      ) : (
        <ChartEmpty label="no concerns recorded" />
      )}
    </CaptureCard>
  );
}

/* ---- Evidence / market ----------------------------------------------------- */

export function EvidenceSourceChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        x: model.evidenceBySource.map((e) => e.source),
        y: model.evidenceBySource.map((e) => e.count),
        marker: {
          color: model.evidenceBySource.map((e) =>
            e.meanSentiment > 0.1 ? COLORS.positive : e.meanSentiment < -0.1 ? COLORS.alert : COLORS.teal
          ),
        },
        hovertemplate: "%{x}<br>%{y} items<extra></extra>",
      },
    ],
    [model.evidenceBySource]
  );
  return (
    <CaptureCard title="Evidence by Source">
      {model.evidenceBySource.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 36, r: 12, t: 8, b: 60 }, xaxis: { tickangle: -35, tickfont: { size: 8, color: COLORS.muted } }, yaxis: { gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="no evidence collected" />
      )}
    </CaptureCard>
  );
}

export function EvidenceSentimentChart({ model }: { model: ReportModel }) {
  const data = useMemo(() => {
    const findings = model.evidenceFindings.filter((f) => f.sentiment != null).slice(0, 10);
    return {
      findings,
      plot: findings.length
        ? ([
            {
              type: "bar" as const,
              orientation: "h" as const,
              y: findings.map((f) => (f.title.length > 32 ? `${f.title.slice(0, 30)}…` : f.title)),
              x: findings.map((f) => f.sentiment ?? 0),
              marker: {
                color: findings.map((f) =>
                  (f.sentiment ?? 0) > 0.1 ? COLORS.positive : (f.sentiment ?? 0) < -0.1 ? COLORS.alert : COLORS.teal
                ),
              },
              hovertemplate: "%{y}<br>sentiment %{x:.2f}<extra></extra>",
            },
          ] satisfies Data[])
        : ([] as Data[]),
    };
  }, [model.evidenceFindings]);
  return (
    <CaptureCard title="Finding Sentiment" subtitle="top evidence by polarity">
      {data.findings.length ? (
        <Plot
          data={data.plot}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 120, r: 12, t: 8, b: 28 },
            xaxis: {
              range: [-1, 1],
              gridcolor: COLORS.grid,
              tickfont: { size: 8, color: COLORS.muted },
            },
            yaxis: { tickfont: { size: 7, color: COLORS.muted }, automargin: true },
          })}
        />
      ) : (
        <ChartEmpty label="no sentiment-tagged evidence" />
      )}
    </CaptureCard>
  );
}

/* ---- Simulation intelligence ------------------------------------------------ */

export function CausalOutcomeChart({ model }: { model: ReportModel }) {
  return (
    <CaptureCard title="Causal Outcome" subtitle="overall prediction">
      {model.causalOutcome != null ? (
        <OutcomeGaugeSvg value={model.causalOutcome} label="OUTCOME" />
      ) : (
        <ChartEmpty label="awaiting causal graph" />
      )}
    </CaptureCard>
  );
}

export function DeliberationMetricsChart({ model }: { model: ReportModel }) {
  const metrics = model.deliberationMetrics;
  const data = useMemo<Data[]>(
    () =>
      metrics
        ? [
            {
              type: "bar",
              x: metrics.map((m) => m.label),
              y: metrics.map((m) => m.value),
              marker: {
                color: metrics.map((m) =>
                  m.label === "Polarization" || m.label === "Entropy"
                    ? COLORS.orange
                    : COLORS.teal
                ),
              },
              hovertemplate: "%{x}<br>%{y:.0f}%<extra></extra>",
            },
          ]
        : [],
    [metrics]
  );
  return (
    <CaptureCard title="Deliberation Metrics">
      {metrics?.length ? (
        <Plot
          data={data}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 36, r: 12, t: 8, b: 40 },
            yaxis: { range: [0, 100], gridcolor: COLORS.grid, ticksuffix: "%", tickfont: { size: 8, color: COLORS.muted } },
            xaxis: { tickfont: { size: 8, color: COLORS.muted } },
          })}
        />
      ) : (
        <ChartEmpty label="awaiting deliberation" />
      )}
    </CaptureCard>
  );
}

export function ConsensusGaugeChart({ model }: { model: ReportModel }) {
  const scores = model.consensusScores;
  const data = useMemo<Data[]>(
    () =>
      scores
        ? [
            {
              type: "bar",
              x: ["Agreement", "Council align."],
              y: [scores.agreement, scores.councilAlignment],
              marker: { color: [COLORS.teal, COLORS.positive] },
              hovertemplate: "%{x}<br>%{y:.0f}%<extra></extra>",
            },
          ]
        : [],
    [scores]
  );
  return (
    <CaptureCard
      title="Consensus Score"
      subtitle={
        scores?.recommendedAction
          ? scores.recommendedAction.slice(0, 48) + (scores.recommendedAction.length > 48 ? "…" : "")
          : undefined
      }
    >
      {scores ? (
        <Plot
          data={data}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 36, r: 12, t: 8, b: 36 },
            yaxis: { range: [0, 100], gridcolor: COLORS.grid, ticksuffix: "%", tickfont: { size: 8, color: COLORS.muted } },
            xaxis: { tickfont: { size: 8, color: COLORS.muted } },
          })}
        />
      ) : (
        <ChartEmpty label="awaiting consensus" />
      )}
    </CaptureCard>
  );
}

export function SocialNarrativeChart({ model }: { model: ReportModel }) {
  const rows = model.socialNarratives.slice(0, 8);
  const data = useMemo<Data[]>(
    () =>
      rows.length
        ? [
            {
              type: "bar",
              orientation: "h",
              y: rows.map((r) => r.label),
              x: rows.map((r) => r.adoptionPct),
              marker: {
                color: rows.map((r) =>
                  r.sentiment > 0.1 ? COLORS.positive : r.sentiment < -0.1 ? COLORS.alert : COLORS.teal
                ),
              },
              hovertemplate: "%{y}<br>%{x:.0f}% adoption<extra></extra>",
            },
          ]
        : [],
    [rows]
  );
  return (
    <CaptureCard title="Social Narrative Adoption">
      {rows.length ? (
        <Plot
          data={data}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 100, r: 12, t: 8, b: 28 },
            xaxis: { range: [0, 100], gridcolor: COLORS.grid, ticksuffix: "%", tickfont: { size: 8, color: COLORS.muted } },
            yaxis: { tickfont: { size: 8, color: COLORS.muted }, automargin: true },
          })}
        />
      ) : (
        <ChartEmpty label="awaiting social simulation" />
      )}
    </CaptureCard>
  );
}

export function CouncilConfidenceChart({ model }: { model: ReportModel }) {
  const rows = model.councilConfidence;
  const data = useMemo<Data[]>(
    () =>
      rows.length
        ? [
            {
              type: "bar",
              x: rows.map((r) => r.role),
              y: rows.map((r) => r.confidence),
              marker: { color: COLORS.teal },
              hovertemplate: "%{x}<br>%{y:.0f}% confidence<extra></extra>",
            },
          ]
        : [],
    [rows]
  );
  return (
    <CaptureCard title="Council Confidence">
      {rows.length ? (
        <Plot
          data={data}
          useResizeHandler
          style={PLOT_STYLE}
          config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 36, r: 12, t: 8, b: 72 },
            yaxis: { range: [0, 100], gridcolor: COLORS.grid, ticksuffix: "%", tickfont: { size: 8, color: COLORS.muted } },
            xaxis: { tickangle: -25, tickfont: { size: 7, color: COLORS.muted } },
          })}
        />
      ) : (
        <ChartEmpty label="awaiting council" />
      )}
    </CaptureCard>
  );
}

export function MarketSizingChart({ model }: { model: ReportModel }) {
  const m = model.marketSizing;
  const data = useMemo<Data[]>(
    () =>
      [
        {
          type: "funnel",
          y: ["TAM", "SAM", "SOM"],
          x: [m.tam, m.sam, m.som],
          marker: { color: [COLORS.teal, COLORS.orange, COLORS.positive] },
          textinfo: "value+percent initial",
          hovertemplate: "%{y}: %{x:,}<extra></extra>",
        } as unknown as Data,
      ],
    [m]
  );
  return (
    <CaptureCard title="Market Sizing (TAM / SAM / SOM)" subtitle="illustrative · model-derived">
      <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
        layout={layout({ margin: { l: 60, r: 16, t: 8, b: 16 } })} />
    </CaptureCard>
  );
}

/* ---- Forecast / scenarios -------------------------------------------------- */

export function ForecastChart({ model }: { model: ReportModel }) {
  const f = model.forecast;
  const data = useMemo<Data[]>(() => {
    if (!f) return [];
    const histX = f.history.map((p) => p.date);
    const histY = f.history.map((p) => p.value);
    const predX = f.predictions.map((p) => p.date);
    const predY = f.predictions.map((p) => p.value);
    const bandX = f.intervals.map((p) => p.date);
    const lastHistX = histX.length ? histX[histX.length - 1] : predX[0];
    const lastHistY = histY.length ? histY[histY.length - 1] : predY[0];
    return [
      { x: bandX, y: f.intervals.map((p) => p.upper), type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: bandX, y: f.intervals.map((p) => p.lower), type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(245,166,35,0.16)", hoverinfo: "skip", showlegend: false },
      { x: histX, y: histY, type: "scatter", mode: "lines", name: "actual", line: { color: COLORS.teal, width: 1.6 } },
      { x: [lastHistX, ...predX], y: [lastHistY, ...predY], type: "scatter", mode: "lines", name: "forecast", line: { color: COLORS.orange, width: 1.8, dash: "dot" } },
    ];
  }, [f]);
  return (
    <CaptureCard title="Forecast Trajectory" subtitle={f ? `${f.model} · MASE ${f.mase_score.toFixed(2)}` : undefined} span={2}>
      {f ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 40, r: 16, t: 8, b: 30 }, showlegend: true, legend: { font: { size: 9, color: COLORS.muted } }, xaxis: { gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted }, nticks: 8 }, yaxis: { gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="awaiting forecast" />
      )}
    </CaptureCard>
  );
}

export function ScenarioChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        x: model.scenarios.map((s) => s.name),
        y: model.scenarios.map((s) => s.endValue),
        marker: { color: [COLORS.alert, COLORS.teal, COLORS.positive] },
        text: model.scenarios.map((s) => `${s.pctChange >= 0 ? "+" : ""}${s.pctChange.toFixed(1)}%`),
        textposition: "outside",
        textfont: { color: COLORS.data, size: 10 },
        hovertemplate: "%{x}<br>end %{y:.1f}<extra></extra>",
      },
    ],
    [model.scenarios]
  );
  return (
    <CaptureCard title="Scenario Planning" subtitle="horizon end-state">
      {model.scenarios.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 40, r: 16, t: 20, b: 28 }, xaxis: { tickfont: { size: 9, color: COLORS.data } }, yaxis: { gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="awaiting forecast" />
      )}
    </CaptureCard>
  );
}

/* ---- Causal / strategy ----------------------------------------------------- */

export function SensitivityTornadoChart({ model }: { model: ReportModel }) {
  const sorted = [...model.sensitivity].reverse();
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: sorted.map((s) => s.weight),
        y: sorted.map((s) => s.label),
        marker: { color: COLORS.teal },
        hovertemplate: "%{y}<br>weight %{x:.2f}<extra></extra>",
      },
    ],
    [sorted]
  );
  return (
    <CaptureCard title="Driver Sensitivity (Tornado)">
      {model.sensitivity.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 130, r: 12, t: 8, b: 24 }, xaxis: { range: [0, 1], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } }, yaxis: { tickfont: { size: 9, color: COLORS.data } } })} />
      ) : (
        <ChartEmpty label="awaiting causal graph" />
      )}
    </CaptureCard>
  );
}

function quadrantShapes(): Partial<Layout> {
  return {
    shapes: [
      { type: "line", x0: 50, x1: 50, y0: 0, y1: 100, line: { color: COLORS.grid, width: 1, dash: "dot" } },
      { type: "line", x0: 0, x1: 100, y0: 50, y1: 50, line: { color: COLORS.grid, width: 1, dash: "dot" } },
    ],
  };
}

export function GrowthMatrixChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "scatter",
        mode: "text+markers",
        x: model.growthMatrix.map((p) => p.x),
        y: model.growthMatrix.map((p) => p.y),
        text: model.growthMatrix.map((p) => p.label),
        textposition: "top center",
        textfont: { size: 8, color: COLORS.muted },
        marker: { size: model.growthMatrix.map((p) => 8 + p.size / 6), color: COLORS.teal, opacity: 0.8, line: { color: COLORS.bg, width: 1 } },
        hovertemplate: "%{text}<br>criticality %{x:.0f} · prediction %{y:.0f}<extra></extra>",
      },
    ],
    [model.growthMatrix]
  );
  return (
    <CaptureCard title="Growth Opportunity Matrix" subtitle="criticality x likelihood">
      {model.growthMatrix.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 44, r: 16, t: 12, b: 36 }, ...quadrantShapes(), xaxis: { title: { text: "Criticality" }, range: [0, 100], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } }, yaxis: { title: { text: "Predicted success" }, range: [0, 100], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="awaiting causal graph" />
      )}
    </CaptureCard>
  );
}

export function RiskMatrixChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "scatter",
        mode: "text+markers",
        x: model.riskMatrix.map((p) => p.x),
        y: model.riskMatrix.map((p) => p.y),
        text: model.riskMatrix.map((p) => p.label),
        textposition: "top center",
        textfont: { size: 8, color: COLORS.muted },
        marker: {
          size: model.riskMatrix.map((p) => 10 + p.size / 6),
          color: model.riskMatrix.map((p) => (p.x * p.y > 0.25 ? COLORS.alert : COLORS.orange)),
          opacity: 0.85,
          line: { color: COLORS.bg, width: 1 },
        },
        hovertemplate: "%{text}<br>likelihood %{x:.2f} · impact %{y:.2f}<extra></extra>",
      },
    ],
    [model.riskMatrix]
  );
  return (
    <CaptureCard title="Risk vs Impact Matrix">
      {model.riskMatrix.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 44, r: 16, t: 12, b: 36 },
            shapes: [
              { type: "line", x0: 0.5, x1: 0.5, y0: 0, y1: 1, line: { color: COLORS.grid, width: 1, dash: "dot" } },
              { type: "line", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: COLORS.grid, width: 1, dash: "dot" } },
            ],
            xaxis: { title: { text: "Likelihood" }, range: [0, 1], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } },
            yaxis: { title: { text: "Impact" }, range: [0, 1], gridcolor: COLORS.grid, tickfont: { size: 8, color: COLORS.muted } },
          })} />
      ) : (
        <ChartEmpty label="awaiting causal graph" />
      )}
    </CaptureCard>
  );
}

export function SentimentHeatmapChart({ model }: { model: ReportModel }) {
  const stimuli = ["Price", "Trust", "Risk", "Social", "Access", "Ethics", "Novelty", "Commit"];
  const data = useMemo<Data[]>(() => {
    if (!model.heatmap.length) return [];
    return [
      {
        type: "heatmap",
        z: model.heatmap.map((r) => r.values),
        y: model.heatmap.map((r) => r.facet),
        x: stimuli.slice(0, model.heatmap[0]?.values.length ?? 0),
        colorscale: [
          [0, COLORS.alert],
          [0.5, COLORS.panel],
          [1, COLORS.positive],
        ],
        zmid: 0,
        hovertemplate: "%{y} · %{x}: %{z:.2f}<extra></extra>",
      },
    ];
  }, [model.heatmap]);
  return (
    <CaptureCard title="Facet x Stimulus Sentiment Heatmap" span={2} height={360}>
      {model.heatmap.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({ margin: { l: 120, r: 16, t: 8, b: 40 }, xaxis: { tickfont: { size: 8, color: COLORS.muted } }, yaxis: { tickfont: { size: 7, color: COLORS.muted } } })} />
      ) : (
        <ChartEmpty label="awaiting heatmap" />
      )}
    </CaptureCard>
  );
}

export function PersonaScatter3DChart({ model }: { model: ReportModel }) {
  const data = useMemo<Data[]>(() => {
    if (!model.personaPoints.length) return [];
    return [
      {
        type: "scatter3d",
        mode: "markers",
        x: model.personaPoints.map((p) => p.pca[0]),
        y: model.personaPoints.map((p) => p.pca[1]),
        z: model.personaPoints.map((p) => p.pca[2]),
        marker: {
          size: 2.6,
          color: model.personaPoints.map((p) => p.sentiment),
          colorscale: [
            [0, COLORS.alert],
            [0.5, COLORS.muted],
            [1, COLORS.positive],
          ],
          cmin: -1,
          cmax: 1,
          opacity: 0.8,
        },
        hovertemplate: "sentiment %{marker.color:.2f}<extra></extra>",
      } as unknown as Data,
    ];
  }, [model.personaPoints]);
  const axis = { gridcolor: COLORS.grid, color: COLORS.muted, showspikes: false, title: { text: "" } };
  return (
    <CaptureCard title="Persona Space (PCA, 3D)" subtitle={`${model.personaPoints.length} sampled`} span={2} height={420}>
      {model.personaPoints.length ? (
        <Plot data={data} useResizeHandler style={PLOT_STYLE} config={PLOTLY_CONFIG}
          layout={layout({
            margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: {
              xaxis: axis,
              yaxis: axis,
              zaxis: axis,
              bgcolor: "rgba(0,0,0,0)",
            },
          } as Partial<Layout>)} />
      ) : (
        <ChartEmpty label="awaiting persona cloud" />
      )}
    </CaptureCard>
  );
}
