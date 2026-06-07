import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { downsample } from "@/lib/downsample";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";
import type { PersonaPoint } from "@/types/events";

const CLUSTER_COLORS = [COLORS.alert, COLORS.orange, COLORS.positive];
const CLUSTER_LABELS = ["Skeptics", "Pragmatists", "Enthusiasts"];
const DISPLAY_CAP_STREAMING = 250;
const DISPLAY_CAP_IDLE = 600;

function buildTraces(points: PersonaPoint[]): Data[] {
  if (!points.length) return [];
  const groups = new Map<number, PersonaPoint[]>();
  for (const p of points) {
    const arr = groups.get(p.cluster) ?? [];
    arr.push(p);
    groups.set(p.cluster, arr);
  }
  return [...groups.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([cluster, pts]) => ({
      type: "scatter3d",
      mode: "markers",
      name: CLUSTER_LABELS[cluster] ?? `C${cluster}`,
      x: pts.map((p) => p.pca[0]),
      y: pts.map((p) => p.pca[1]),
      z: pts.map((p) => p.pca[2]),
      marker: {
        size: 2.6,
        color: CLUSTER_COLORS[cluster] ?? COLORS.teal,
        opacity: 0.75,
      },
      hovertemplate:
        `${CLUSTER_LABELS[cluster] ?? "cluster"}<br>` +
        "sentiment %{customdata:.2f}<extra></extra>",
      customdata: pts.map((p) => p.sentiment),
    })) as Data[];
}

/** PCA projection of persona OCEAN vectors. Plotly is deferred while streaming. */
export function PersonaScatter3D() {
  const points = useSessionStore((s) => s.personaPoints);
  const connection = useSessionStore((s) => s.connection);
  const personasSimulated = useSessionStore((s) => s.personasSimulated);
  const personaTarget = useSessionStore((s) => s.personaTarget);

  const isStreaming = connection === "streaming" || connection === "connecting";

  const displayPoints = useMemo(() => {
    const cap = isStreaming ? DISPLAY_CAP_STREAMING : DISPLAY_CAP_IDLE;
    return downsample(points, cap);
  }, [points, isStreaming]);

  const data = useMemo(() => buildTraces(displayPoints), [displayPoints]);

  if (!points.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting persona vectors
      </div>
    );
  }

  if (isStreaming) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center">
        <span className="font-mono text-xs text-teal animate-pulse-stream">
          building persona space
        </span>
        <span className="font-mono text-2xs text-muted">
          {personasSimulated.toLocaleString()} / {personaTarget.toLocaleString()} profiles
        </span>
        <span className="font-mono text-2xs text-muted/70">
          3D plot loads when the psychometric phase completes
        </span>
      </div>
    );
  }

  const axis = {
    backgroundcolor: "rgba(0,0,0,0)",
    gridcolor: COLORS.grid,
    zerolinecolor: COLORS.grid,
    showbackground: true,
    tickfont: { size: 7, color: COLORS.muted },
    title: { text: "" },
  };

  return (
    <div className="relative h-full min-h-0 w-full p-1">
      <Plot
        data={data}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        layout={{
          ...basePlotlyLayout(),
          margin: { l: 0, r: 0, t: 0, b: 0 },
          scene: {
            xaxis: axis,
            yaxis: axis,
            zaxis: axis,
            camera: { eye: { x: 1.6, y: 1.5, z: 1.1 } },
            aspectmode: "cube",
          },
        }}
        config={PLOTLY_CONFIG}
      />
      <div className="pointer-events-none absolute left-2 top-1 flex gap-3 font-mono text-2xs">
        {CLUSTER_LABELS.map((l, i) => (
          <span key={l} style={{ color: CLUSTER_COLORS[i] }}>
            {l}
          </span>
        ))}
      </div>
      {points.length > displayPoints.length && (
        <div className="pointer-events-none absolute bottom-1 right-2 font-mono text-2xs text-muted">
          showing {displayPoints.length} of {points.length}
        </div>
      )}
    </div>
  );
}
