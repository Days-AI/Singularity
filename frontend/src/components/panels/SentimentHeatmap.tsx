import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useSessionStore } from "@/store/sessionStore";
import { useResize } from "@/hooks/useResize";
import { COLORS } from "@/lib/theme";

/**
 * Facet x stimulus sentiment matrix. Diverging color scale from alert-red
 * (negative) through panel-dark (neutral) to positive-green.
 */
export function SentimentHeatmap() {
  const heatmap = useSessionStore((s) => s.heatmap);
  const [wrapRef, size] = useResize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    if (!heatmap.length || size.width === 0 || size.height === 0) return;

    const margin = { top: 8, right: 8, bottom: 8, left: 96 };
    const w = size.width - margin.left - margin.right;
    const h = size.height - margin.top - margin.bottom;
    const cols = heatmap[0].values.length;

    const x = d3
      .scaleBand<number>()
      .domain(d3.range(cols))
      .range([0, w])
      .padding(0.06);
    const y = d3
      .scaleBand<string>()
      .domain(heatmap.map((r) => r.facet))
      .range([0, h])
      .padding(0.08);

    const color = d3
      .scaleLinear<string>()
      .domain([-1, 0, 1])
      .range([COLORS.alert, "#0E1A24", COLORS.positive])
      .clamp(true);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    g.selectAll("g.row")
      .data(heatmap)
      .join("g")
      .attr("class", "row")
      .attr("transform", (d) => `translate(0,${y(d.facet)})`)
      .each(function (row) {
        const rowG = d3.select(this);
        rowG
          .append("text")
          .attr("x", -8)
          .attr("y", y.bandwidth() / 2)
          .attr("dy", "0.32em")
          .attr("text-anchor", "end")
          .attr("fill", COLORS.muted)
          .attr("font-family", "'IBM Plex Mono', monospace")
          .attr("font-size", 9)
          .text(row.facet);

        rowG
          .selectAll("rect")
          .data(row.values.map((v, c) => ({ v, c })))
          .join("rect")
          .attr("x", (d) => x(d.c) ?? 0)
          .attr("width", x.bandwidth())
          .attr("height", y.bandwidth())
          .attr("rx", 1)
          .attr("fill", "#0E1A24")
          .append("title")
          .text((d) => `${row.facet} / stim ${d.c + 1}: ${d.v.toFixed(2)}`);

        rowG
          .selectAll("rect")
          .data(row.values.map((v, c) => ({ v, c })))
          .transition()
          .duration(500)
          .attr("fill", (d) => color(d.v));
      });
  }, [heatmap, size]);

  return (
    <div ref={wrapRef} className="h-full min-h-0 w-full p-1">
      {heatmap.length === 0 ? (
        <Empty />
      ) : (
        <svg ref={svgRef} width={size.width} height={size.height} className="block" />
      )}
    </div>
  );
}

function Empty() {
  return (
    <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
      awaiting persona batches
    </div>
  );
}
