import { useMemo } from "react";
import type { Data } from "plotly.js";
import Plot from "@/lib/plotly";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";

const CLUSTER_COLORS = [COLORS.alert, COLORS.orange, COLORS.positive];
const CLUSTER_LABELS = ["Skeptics", "Pragmatists", "Enthusiasts"];

/** PCA projection of the 1500-agent OCEAN vectors, colored by latent cluster. */
export function PersonaScatter3D() {
  const points = useSessionStore((s) => s.personaPoints);

  const data = useMemo<Data[]>(() => {
    if (!points.length) return [];
    const groups = new Map<number, typeof points>();
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
  }, [points]);

  if (!points.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting persona vectors
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
    <div className="relative h-full w-full">
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
    </div>
  );
}
