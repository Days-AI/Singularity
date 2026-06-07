import { useCallback, useState } from "react";
import type { Layout, Layouts } from "react-grid-layout";

/**
 * Stable identifiers for every dashboard tile. Used both as the react-grid-layout
 * item key (`i`) and to look up the panel's default geometry.
 */
export const PANEL_IDS = [
  "agents",
  "prediction",
  "forecast",
  "ocean",
  "scatter",
  "heatmap",
  "evidence",
  "sources",
  "causal",
  "social",
  "council",
  "report",
] as const;

export type PanelId = (typeof PANEL_IDS)[number];

export const GRID_COLS = 12;
export const GRID_ROW_HEIGHT = 110;
export const GRID_MARGIN: [number, number] = [8, 8];

/**
 * Desktop mosaic on a 12-column grid.
 * Bands: agents rail (0–4) | center analytics (4–7) | right charts (8–9) | report rail (10–11).
 * Row baselines align at y=4 (prediction), y=7 (ocean/scatter/heatmap), y=10 (causal), y=13 (social/council).
 */
const LG_LAYOUT: Layout[] = [
  { i: "agents", x: 0, y: 0, w: 4, h: 5, minW: 3, minH: 3 },
  { i: "prediction", x: 4, y: 0, w: 4, h: 4, minW: 3, minH: 3 },
  { i: "forecast", x: 8, y: 0, w: 2, h: 2, minW: 2, minH: 2 },
  { i: "heatmap", x: 8, y: 2, w: 2, h: 5, minW: 2, minH: 2 },
  { i: "report", x: 10, y: 0, w: 2, h: 13, minW: 2, minH: 4 },
  { i: "ocean", x: 4, y: 4, w: 2, h: 3, minW: 2, minH: 2 },
  { i: "scatter", x: 6, y: 4, w: 2, h: 3, minW: 2, minH: 2 },
  { i: "evidence", x: 0, y: 5, w: 2, h: 2, minW: 2, minH: 2 },
  { i: "sources", x: 2, y: 5, w: 2, h: 2, minW: 2, minH: 2 },
  { i: "causal", x: 0, y: 7, w: 10, h: 3, minW: 4, minH: 2 },
  { i: "social", x: 0, y: 10, w: 5, h: 3, minW: 3, minH: 2 },
  { i: "council", x: 5, y: 10, w: 5, h: 3, minW: 3, minH: 2 },
];

/** Single-column stack for narrow screens. */
const XXS_LAYOUT: Layout[] = PANEL_IDS.map((id, idx) => ({
  i: id,
  x: 0,
  y: idx * 3,
  w: 1,
  h: 3,
  minW: 1,
  minH: 2,
}));

export const DEFAULT_LAYOUTS: Layouts = {
  lg: LG_LAYOUT,
  md: LG_LAYOUT,
  sm: LG_LAYOUT,
  xs: XXS_LAYOUT,
  xxs: XXS_LAYOUT,
};

export const GRID_BREAKPOINTS = { lg: 1024, md: 768, sm: 640, xs: 480, xxs: 0 };
export const GRID_COLS_MAP = { lg: 12, md: 12, sm: 12, xs: 1, xxs: 1 };

const STORAGE_KEY = "singularity.panel-layouts.v7";

function loadLayouts(): Layouts | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Layouts;
    // Basic shape guard: require the desktop breakpoint with at least one item.
    if (!parsed?.lg || !Array.isArray(parsed.lg) || parsed.lg.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

function saveLayouts(layouts: Layouts): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts));
  } catch {
    /* ignore quota / privacy-mode errors */
  }
}

/**
 * Persisted, resettable react-grid-layout state. Layout edits (drag / resize)
 * are stored in localStorage so the user's arrangement survives reloads.
 */
export function usePanelLayouts() {
  const [layouts, setLayouts] = useState<Layouts>(() => loadLayouts() ?? DEFAULT_LAYOUTS);

  const handleLayoutChange = useCallback((_current: Layout[], all: Layouts) => {
    setLayouts(all);
    saveLayouts(all);
  }, []);

  const resetLayouts = useCallback(() => {
    setLayouts(DEFAULT_LAYOUTS);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  return { layouts, handleLayoutChange, resetLayouts };
}
