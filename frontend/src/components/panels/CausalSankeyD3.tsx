import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useSessionStore } from "@/store/sessionStore";
import { useResize } from "@/hooks/useResize";
import { COLORS } from "@/lib/theme";
import type { CausalNode } from "@/types/events";

type Positioned = CausalNode & { x: number; y: number };

const COLUMN: Record<CausalNode["kind"], number> = {
  cause: 0,
  mediator: 1,
  effect: 2,
};

/**
 * Causal flow map. Nodes are arranged in cause -> mediator -> effect columns;
 * edges are drawn as curved links whose width encodes Hawkes excitation weight
 * and whose color encodes Granger significance (teal if p < 0.05, muted dashed
 * otherwise). p-values are rendered as edge labels.
 */
export function CausalSankeyD3() {
  const causal = useSessionStore((s) => s.causal);
  const [wrapRef, size] = useResize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    if (!causal || size.width === 0 || size.height === 0) return;

    const margin = { top: 16, right: 90, bottom: 16, left: 90 };
    const w = size.width - margin.left - margin.right;
    const h = size.height - margin.top - margin.bottom;

    const byColumn = d3.group(causal.nodes, (n) => COLUMN[n.kind]);
    const positioned = new Map<string, Positioned>();
    for (const [col, nodes] of byColumn) {
      const colX = (w / 2) * col;
      nodes.forEach((n, i) => {
        const step = h / (nodes.length + 1);
        positioned.set(n.id, { ...n, x: colX, y: step * (i + 1) });
      });
    }

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const wScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([1, 9]);

    // links
    const linkG = g.append("g");
    causal.edges.forEach((e) => {
      const s = positioned.get(e.source);
      const t = positioned.get(e.target);
      if (!s || !t) return;
      const sig = e.p_value < 0.05;
      const midX = (s.x + t.x) / 2;
      const path = `M${s.x},${s.y} C${midX},${s.y} ${midX},${t.y} ${t.x},${t.y}`;

      linkG
        .append("path")
        .attr("d", path)
        .attr("fill", "none")
        .attr("stroke", sig ? COLORS.teal : COLORS.muted)
        .attr("stroke-opacity", sig ? 0.7 : 0.35)
        .attr("stroke-width", wScale(e.weight))
        .attr("stroke-dasharray", sig ? null : "4 3")
        .attr("stroke-linecap", "round")
        .style("filter", sig ? "drop-shadow(0 0 4px rgba(0,180,216,0.4))" : "none");

      linkG
        .append("text")
        .attr("x", midX)
        .attr("y", (s.y + t.y) / 2 - 4)
        .attr("text-anchor", "middle")
        .attr("fill", sig ? COLORS.teal : COLORS.muted)
        .attr("font-family", "'IBM Plex Mono', monospace")
        .attr("font-size", 8)
        .text(`p=${e.p_value.toFixed(3)}`);
    });

    // nodes
    const nodeG = g
      .selectAll("g.cnode")
      .data([...positioned.values()])
      .join("g")
      .attr("class", "cnode")
      .attr("transform", (d) => `translate(${d.x},${d.y})`);

    const KIND_COLOR: Record<CausalNode["kind"], string> = {
      cause: COLORS.orange,
      mediator: COLORS.teal,
      effect: COLORS.positive,
    };

    nodeG
      .append("rect")
      .attr("x", -52)
      .attr("y", -13)
      .attr("width", 104)
      .attr("height", 26)
      .attr("rx", 3)
      .attr("fill", COLORS.panelRaised)
      .attr("stroke", (d) => KIND_COLOR[d.kind])
      .attr("stroke-width", 1.4);

    nodeG
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.32em")
      .attr("fill", COLORS.data)
      .attr("font-family", "'IBM Plex Mono', monospace")
      .attr("font-size", 9)
      .text((d) => d.label);
  }, [causal, size]);

  return (
    <div ref={wrapRef} className="h-full w-full">
      {!causal ? (
        <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
          awaiting causal inference
        </div>
      ) : (
        <svg ref={svgRef} width={size.width} height={size.height} className="block" />
      )}
    </div>
  );
}
