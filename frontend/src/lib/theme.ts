import type Plotly from "plotly.js";

/**
 * Runtime theme constants for imperative renderers (D3, Plotly) that cannot
 * consume Tailwind classes. Mirrors tokens.css / tailwind.config.ts.
 */
export const COLORS = {
  bg: "#0A0E1A",
  panel: "#0F1923",
  panelRaised: "#13212E",
  teal: "#00B4D8",
  orange: "#F5A623",
  positive: "#00E676",
  alert: "#FF4C4C",
  data: "#E8ECF0",
  muted: "#546E7A",
  grid: "#1B2A38",
} as const;

export const FONT_MONO = "'IBM Plex Mono', ui-monospace, monospace";

/** OCEAN dimension -> accent color, used across radar / heatmap / 3D scatter. */
export const OCEAN_COLORS: Record<string, string> = {
  O: "#00B4D8",
  C: "#00E676",
  E: "#F5A623",
  A: "#9B6DFF",
  N: "#FF4C4C",
};

export const OCEAN_LABELS: Record<string, string> = {
  O: "Openness",
  C: "Conscientiousness",
  E: "Extraversion",
  A: "Agreeableness",
  N: "Neuroticism",
};

/** Shared Plotly layout fragment for the dark terminal look. */
export function basePlotlyLayout(): Partial<Plotly.Layout> {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: FONT_MONO, color: COLORS.data, size: 10 },
    margin: { l: 40, r: 16, t: 16, b: 32 },
    showlegend: false,
    hoverlabel: {
      bgcolor: COLORS.panelRaised,
      bordercolor: COLORS.teal,
      font: { family: FONT_MONO, color: COLORS.data, size: 11 },
    },
  };
}

export const PLOTLY_CONFIG: Partial<Plotly.Config> = {
  displaylogo: false,
  responsive: true,
  displayModeBar: false,
};
