/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "react-plotly.js/factory" {
  import type Plotly from "plotly.js";
  import type { PlotParams } from "react-plotly.js";
  import type * as React from "react";
  export default function createPlotlyComponent(
    plotly: typeof Plotly
  ): React.ComponentType<PlotParams>;
}

declare module "plotly.js-dist-min";
