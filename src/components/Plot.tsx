"use client";

import { useEffect, useRef } from "react";

// Plotly's dist bundle touches `window`/`document` at import time, so it can only
// be loaded in the browser. We import it lazily (once) inside an effect and reuse
// the promise across every <Plot/> instance.
type PlotlyModule = typeof import("plotly.js-dist-min");
let plotlyPromise: Promise<PlotlyModule["default"]> | null = null;
function loadPlotly(): Promise<PlotlyModule["default"]> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-dist-min").then(
      (m) => (m.default ?? (m as unknown as PlotlyModule["default"]))
    );
  }
  return plotlyPromise;
}

type Fig = Record<string, unknown>;

export default function Plot({
  data,
  layout,
  config,
  className,
  style,
  onClick,
}: {
  data: Fig[];
  layout?: Fig;
  config?: Fig;
  className?: string;
  style?: React.CSSProperties;
  onClick?: (event: any) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    if (!el) return;
    loadPlotly().then((Plotly) => {
      if (cancelled || !ref.current) return;
      Plotly.react(
        ref.current,
        data,
        layout ?? {},
        config ?? { responsive: true, displaylogo: false, displayModeBar: "hover", scrollZoom: false }
      );
      if (onClick && ref.current) {
        (ref.current as any).removeAllListeners?.('plotly_click');
        (ref.current as any).on?.('plotly_click', onClick);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [data, layout, config, onClick]);

  // Tear down the Plotly instance (removes listeners / WebGL contexts) on unmount.
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) loadPlotly().then((Plotly) => Plotly.purge(el));
    };
  }, []);

  return <div ref={ref} className={className} style={{ width: "100%", ...style }} />;
}
