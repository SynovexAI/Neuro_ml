"use client";

import { useMemo } from "react";
import Plot from "@/components/Plot";
import { plotlyTheme } from "@/lib/edaCharts";
import { forceLayout, type KnowledgeGraph } from "@/lib/kgUtils";

/**
 * Interactive knowledge-graph view on Plotly — scroll to zoom, drag to pan,
 * hover a node for its type/frequency/source chunks, hover an edge for the
 * relation and the chunk it came from.
 *
 * `highlight` dims everything outside the set (used to show a retrieved
 * subgraph against the whole graph), and `layers` labels how many hops away
 * each highlighted node was reached.
 */
export default function KnowledgeGraphPlot({
  graph,
  height = 460,
  highlight,
  seeds,
  layers,
}: {
  graph: KnowledgeGraph;
  height?: number;
  highlight?: Set<string>;
  seeds?: string[];
  layers?: Record<string, number>;
}) {
  const t = plotlyTheme();
  const pos = useMemo(() => forceLayout(graph), [graph]);

  const { data, layout, config } = useMemo(() => {
    const label = (id: string) => graph.nodes.find((n) => n.id === id)?.label || id;
    const on = (id: string) => !highlight || highlight.has(id);

    // one line trace per edge so each can carry its own hover text
    const edgeTraces = graph.edges.map((e) => {
      const a = pos[e.s], b = pos[e.o];
      if (!a || !b) return null;
      const lit = on(e.s) && on(e.o);
      return {
        type: "scatter", mode: "lines",
        x: [a.x, b.x], y: [a.y, b.y],
        line: { color: lit ? t.line : t.grid, width: Math.min(3.5, 1 + (e.weight || 1) * 0.5) },
        opacity: lit ? 0.85 : 0.18,
        hoverinfo: "text",
        hovertext: `${label(e.s)} —${e.rel}→ ${label(e.o)}<br>from chunk ${e.chunks.map((c) => c + 1).join(", ")}`,
        showlegend: false,
      } as Record<string, unknown>;
    }).filter(Boolean) as Record<string, unknown>[];

    // 40 labels will always collide. Label the nodes that carry the structure —
    // highlighted ones, seeds, and the best-connected — and leave the rest to hover.
    const degree = new Map<string, number>();
    graph.nodes.forEach((n) => degree.set(n.id, 0));
    graph.edges.forEach((e) => {
      degree.set(e.s, (degree.get(e.s) || 0) + 1);
      degree.set(e.o, (degree.get(e.o) || 0) + 1);
    });
    const budget = Math.max(8, Math.round(28 - graph.nodes.length * 0.15));
    const labelled = new Set(
      [...graph.nodes]
        .sort((a, b) => (degree.get(b.id)! - degree.get(a.id)!) || (b.freq - a.freq))
        .slice(0, budget)
        .map((n) => n.id),
    );
    const showLabel = (id: string) =>
      seeds?.includes(id) || (highlight ? highlight.has(id) : labelled.has(id));

    const mk = (want: "proper" | "concept") => {
      const ns = graph.nodes.filter((n) => n.type === want);
      const colour = want === "proper" ? t.accent : t.colorway[1];
      return {
        type: "scatter", mode: "markers+text",
        name: want === "proper" ? "proper noun" : "concept",
        x: ns.map((n) => pos[n.id]?.x ?? 0.5),
        y: ns.map((n) => pos[n.id]?.y ?? 0.5),
        text: ns.map((n) => (showLabel(n.id) ? n.label : "")),
        textposition: "top center",
        textfont: { size: 10, color: ns.map((n) => (on(n.id) ? t.muted : t.grid)) },
        marker: {
          size: ns.map((n) => 12 + Math.min(20, n.freq * 3)),
          color: ns.map((n) => (on(n.id) ? colour : t.grid)),
          opacity: ns.map((n) => (on(n.id) ? 0.9 : 0.25)),
          line: {
            width: ns.map((n) => (seeds?.includes(n.id) ? 3 : 1.5)),
            color: ns.map((n) => (seeds?.includes(n.id) ? t.colorway[3] : colour)),
          },
        },
        customdata: ns.map((n) => n.id),
        hovertemplate: ns.map((n) => {
          const hop = layers?.[n.id];
          return [
            `<b>${n.label}</b>`,
            `type: ${n.type}`,
            `appears ${n.freq}× in ${n.chunks.length} chunk(s): ${n.chunks.map((c) => c + 1).join(", ")}`,
            `connections: ${graph.edges.filter((e) => e.s === n.id || e.o === n.id).length}`,
            seeds?.includes(n.id) ? "<b>seed — matched the query</b>" : hop !== undefined ? `reached in ${hop} hop(s)` : "",
          ].filter(Boolean).join("<br>") + "<extra></extra>";
        }),
        showlegend: true,
      } as Record<string, unknown>;
    };

    return {
      data: [...edgeTraces, mk("concept"), mk("proper")],
      layout: {
        height,
        margin: { l: 8, r: 8, t: 8, b: 8 },
        paper_bgcolor: t.paper,
        plot_bgcolor: t.plot,
        font: { color: t.text, size: 11 },
        xaxis: { visible: false, range: [-0.06, 1.06], fixedrange: false },
        yaxis: { visible: false, range: [-0.06, 1.06], scaleanchor: "x", scaleratio: 1, fixedrange: false },
        hovermode: "closest",
        hoverlabel: { bgcolor: t.paper, bordercolor: t.line, font: { color: t.text, size: 11 } },
        dragmode: "pan",
        legend: { orientation: "h", y: -0.02, x: 0, font: { size: 10, color: t.muted } },
        showlegend: true,
      } as Record<string, unknown>,
      config: { displaylogo: false, scrollZoom: true, responsive: true,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] } as Record<string, unknown>,
    };
  }, [graph, pos, t, height, highlight, seeds, layers]);

  if (!graph.nodes.length) return null;

  return <Plot data={data} layout={layout} config={config} style={{ width: "100%" }} />;
}
