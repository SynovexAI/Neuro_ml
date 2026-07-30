// Builds Plotly figures for the ML Lab's EDA explorer. Everything is derived from
// the in-browser Dataset — no server, no mocks. Covers univariate (single column)
// and bivariate / multivariate (compare columns) views, themed for light & dark.

import type { Dataset, Column } from "./mlUtils";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Trace = Record<string, any>;
type Layout = Record<string, any>;
export interface ChartFig { data: Trace[]; layout: Layout; }

export interface Theme {
  text: string; muted: string; grid: string; line: string;
  paper: string; plot: string; accent: string; colorway: string[]; heat: [number, string][];
}

const FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif";

// Read the live CSS custom properties so charts match the app's light/dark theme.
export function plotlyTheme(): Theme {
  const g = (v: string, fb: string) =>
    (typeof document !== "undefined"
      ? getComputedStyle(document.documentElement).getPropertyValue(v).trim()
      : "") || fb;
  const accent = g("--accent", "#2b50e0");
  const sky = g("--sky", "#0891b2");
  const good = g("--good", "#15803d");
  const warn = g("--warn", "#b45309");
  const crit = g("--crit", "#c0392b");
  return {
    text: g("--text", "#0d1117"),
    muted: g("--muted", "#5c6470"),
    grid: g("--border", "#e6e8ee"),
    line: g("--border-strong", "#d6dae2"),
    paper: g("--surface", "#ffffff"),
    plot: g("--panel", "#f3f4f7"),
    accent,
    colorway: [accent, sky, good, warn, crit, "#a855f7", "#ec4899", "#0ea5e9", "#f59e0b", "#14b8a6"],
    heat: [[0, sky], [0.5, g("--panel-2", "#eceef2")], [1, accent]],
  };
}

// ── column helpers ──
export const findCol = (ds: Dataset, name: string): Column | undefined => ds.columns.find((c) => c.name === name);
export const numCols = (ds: Dataset) => ds.columns.filter((c) => c.type === "num").map((c) => c.name);
export const catCols = (ds: Dataset) => ds.columns.filter((c) => c.type === "cat").map((c) => c.name);

function numVals(ds: Dataset, name: string): number[] {
  const c = findCol(ds, name);
  if (!c) return [];
  return c.values.filter((v) => v != null).map((v) => Number(v));
}
function pairs(ds: Dataset, xn: string, yn: string): { x: number[]; y: number[] } {
  const cx = findCol(ds, xn), cy = findCol(ds, yn);
  const x: number[] = [], y: number[] = [];
  if (!cx || !cy) return { x, y };
  for (let i = 0; i < ds.nrows; i++) {
    const a = cx.values[i], b = cy.values[i];
    if (a == null || b == null) continue;
    x.push(Number(a)); y.push(Number(b));
  }
  return { x, y };
}
function catCounts(col: Column, max = 30): { labels: string[]; counts: number[] } {
  const m = new Map<string, number>();
  col.values.forEach((v) => { if (v != null) { const k = String(v); m.set(k, (m.get(k) || 0) + 1); } });
  const sorted = [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, max);
  return { labels: sorted.map((e) => e[0]), counts: sorted.map((e) => e[1]) };
}

// ── stats ──
export function pearson(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / (Math.sqrt(da * db) || 1);
}
function skewness(v: number[]): number {
  const n = v.length; if (n < 3) return 0;
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  return v.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n;
}
function gaussianKDE(values: number[], gridN = 72): { x: number[]; y: number[] } {
  const n = values.length; if (!n) return { x: [], y: [] };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  const bw = 1.06 * std * Math.pow(n, -1 / 5) || 1;
  const min = Math.min(...values), max = Math.max(...values), pad = ((max - min) || 1) * 0.12;
  const lo = min - pad, hi = max + pad, span = hi - lo || 1;
  const x: number[] = [], y: number[] = [];
  const c = 1 / (n * bw * Math.sqrt(2 * Math.PI));
  for (let i = 0; i < gridN; i++) {
    const xi = lo + (span * i) / (gridN - 1);
    let s = 0;
    for (const v of values) { const u = (xi - v) / bw; s += Math.exp(-0.5 * u * u); }
    x.push(xi); y.push(c * s);
  }
  return { x, y };
}

// ── layout scaffold ──
function base(theme: Theme, title: string, extra: Layout = {}): Layout {
  const axis = { gridcolor: theme.grid, zerolinecolor: theme.grid, linecolor: theme.line, tickfont: { size: 10, color: theme.muted }, titlefont: { size: 11, color: theme.muted } };
  return {
    paper_bgcolor: theme.paper,
    plot_bgcolor: theme.plot,
    font: { family: FONT, color: theme.muted, size: 11 },
    colorway: theme.colorway,
    margin: { l: 54, r: 20, t: 40, b: 48 },
    title: { text: title, font: { size: 13, color: theme.text }, x: 0, xanchor: "left", xref: "paper" },
    xaxis: { ...axis },
    yaxis: { ...axis },
    legend: { font: { size: 10, color: theme.muted }, orientation: "h", yanchor: "bottom", y: 1.02, x: 0 },
    hoverlabel: { font: { family: FONT, size: 11 }, bordercolor: theme.line },
    bargap: 0.05,
    ...extra,
  };
}

// ── spec ──
export interface EdaSpec {
  mode: "single" | "compare";
  // single-column
  uniCol?: string;
  uniChart?: string;
  bins?: number;
  groupBy?: string; // optional categorical to split a numeric view by
  // compare
  cmpChart?: string;
  x?: string;
  y?: string;
  color?: string;
  cols?: string[];
  trend?: boolean;
}

export const SINGLE_NUM = ["Histogram", "KDE (density)", "Box", "Violin", "ECDF", "Strip / jitter"];
export const SINGLE_CAT = ["Bar (counts)", "Horizontal bar", "Pie", "Donut"];
export const COMPARE_CHARTS = [
  "Scatter", "Line", "2D density", "Scatter matrix", "Correlation heatmap",
  "Box by group", "Violin by group", "Grouped bar", "Stacked bar", "Parallel coordinates",
];

// Build the Plotly figure for a spec. Returns null-ish empty fig if inputs are missing.
export function buildFigure(ds: Dataset, spec: EdaSpec, theme: Theme): ChartFig {
  return spec.mode === "single" ? buildSingle(ds, spec, theme) : buildCompare(ds, spec, theme);
}

function empty(theme: Theme, msg: string): ChartFig {
  return { data: [], layout: base(theme, "", { annotations: [{ text: msg, showarrow: false, font: { size: 12, color: theme.muted }, xref: "paper", yref: "paper", x: 0.5, y: 0.5 }], xaxis: { visible: false }, yaxis: { visible: false } }) };
}

// ── SINGLE COLUMN ──
function buildSingle(ds: Dataset, spec: EdaSpec, theme: Theme): ChartFig {
  const col = spec.uniCol ? findCol(ds, spec.uniCol) : undefined;
  if (!col) return empty(theme, "Pick a column.");
  const chart = spec.uniChart || (col.type === "num" ? "Histogram" : "Bar (counts)");
  const grp = spec.groupBy && spec.groupBy !== "(none)" ? findCol(ds, spec.groupBy) : undefined;

  // categorical column
  if (col.type === "cat") {
    const { labels, counts } = catCounts(col);
    if (chart === "Pie" || chart === "Donut")
      return { data: [{ type: "pie", labels, values: counts, hole: chart === "Donut" ? 0.5 : 0, textinfo: "label+percent", marker: { colors: theme.colorway } }], layout: base(theme, `${col.name} — share`, { margin: { l: 10, r: 10, t: 40, b: 10 } }) };
    const horizontal = chart === "Horizontal bar";
    return {
      data: [horizontal
        ? { type: "bar", orientation: "h", y: labels, x: counts, marker: { color: theme.accent } }
        : { type: "bar", x: labels, y: counts, marker: { color: theme.accent } }],
      layout: base(theme, `${col.name} — counts (${labels.length} categories)`, horizontal ? { yaxis: { automargin: true, gridcolor: theme.grid }, margin: { l: 120 } } : { xaxis: { automargin: true } }),
    };
  }

  // numeric column
  const v = col.values.map((x) => (x == null ? null : Number(x)));
  const present = v.filter((x): x is number => x != null);

  if (chart === "Box" || chart === "Violin") {
    const isViolin = chart === "Violin";
    if (grp) {
      // split by group category, one distribution per group
      const gx: string[] = [], gy: number[] = [];
      col.values.forEach((val, i) => { const gv = grp.values[i]; if (val != null && gv != null) { gx.push(String(gv)); gy.push(Number(val)); } });
      const t: Trace = isViolin
        ? { type: "violin", x: gx, y: gy, box: { visible: true }, meanline: { visible: true }, points: false, transforms: undefined }
        : { type: "box", x: gx, y: gy, boxpoints: "outliers" };
      t.marker = { color: theme.accent };
      return { data: [t], layout: base(theme, `${col.name} by ${grp.name}`, { xaxis: { automargin: true, gridcolor: theme.grid } }) };
    }
    const t: Trace = isViolin
      ? { type: "violin", y: present, box: { visible: true }, meanline: { visible: true }, points: "outliers", name: col.name, line: { color: theme.accent } }
      : { type: "box", y: present, boxpoints: "outliers", name: col.name, marker: { color: theme.accent } };
    return { data: [t], layout: base(theme, `${col.name} — ${isViolin ? "violin" : "box"} plot`) };
  }

  if (chart === "KDE (density)") {
    const k = gaussianKDE(present);
    return { data: [{ type: "scatter", x: k.x, y: k.y, mode: "lines", fill: "tozeroy", line: { color: theme.accent, width: 2 }, fillcolor: theme.accent + "33", name: "density" }], layout: base(theme, `${col.name} — kernel density (KDE)`, { yaxis: { title: "density", gridcolor: theme.grid } }) };
  }

  if (chart === "ECDF") {
    const s = [...present].sort((a, b) => a - b);
    const y = s.map((_, i) => (i + 1) / s.length);
    return { data: [{ type: "scatter", x: s, y, mode: "lines", line: { color: theme.accent, width: 2, shape: "hv" } }], layout: base(theme, `${col.name} — empirical CDF`, { yaxis: { title: "cumulative proportion", range: [0, 1.02], gridcolor: theme.grid } }) };
  }

  if (chart === "Strip / jitter") {
    const y = present.map(() => Math.random());
    return { data: [{ type: "scatter", x: present, y, mode: "markers", marker: { color: theme.accent, size: 5, opacity: 0.5 } }], layout: base(theme, `${col.name} — strip plot`, { yaxis: { visible: false, range: [-0.2, 1.2] } }) };
  }

  // Histogram (default), optionally overlaid by group
  if (grp) {
    const groups = new Map<string, number[]>();
    col.values.forEach((val, i) => { const gv = grp.values[i]; if (val != null && gv != null) { const key = String(gv); if (!groups.has(key)) groups.set(key, []); groups.get(key)!.push(Number(val)); } });
    const data: Trace[] = [...groups.entries()].slice(0, 8).map(([name, arr]) => ({ type: "histogram", x: arr, name, opacity: 0.6, nbinsx: spec.bins || 20 }));
    return { data, layout: base(theme, `${col.name} by ${grp.name}`, { barmode: "overlay" }) };
  }
  return { data: [{ type: "histogram", x: present, nbinsx: spec.bins || 20, marker: { color: theme.accent, line: { color: theme.paper, width: 1 } } }], layout: base(theme, `${col.name} — distribution (${spec.bins || 20} bins)`, { yaxis: { title: "count", gridcolor: theme.grid } }) };
}

// ── COMPARE COLUMNS ──
function buildCompare(ds: Dataset, spec: EdaSpec, theme: Theme): ChartFig {
  const chart = spec.cmpChart || "Scatter";
  const nums = numCols(ds);

  if (chart === "Correlation heatmap") {
    const cols = (spec.cols && spec.cols.length >= 2 ? spec.cols : nums).filter((c) => nums.includes(c));
    if (cols.length < 2) return empty(theme, "Select at least 2 numeric columns.");
    const vecs = cols.map((c) => numVals(ds, c));
    const z: number[][] = [], text: string[][] = [];
    for (let i = 0; i < cols.length; i++) {
      z.push([]); text.push([]);
      for (let j = 0; j < cols.length; j++) {
        const r = i === j ? 1 : pearson(vecs[i], vecs[j]);
        z[i].push(r); text[i].push(r.toFixed(2));
      }
    }
    return {
      data: [{ type: "heatmap", z, x: cols, y: cols, zmin: -1, zmax: 1, colorscale: "RdBu", reversescale: true, text, texttemplate: "%{text}", textfont: { size: 9 }, colorbar: { thickness: 12, len: 0.8 } }],
      layout: base(theme, "Correlation matrix (Pearson r)", { xaxis: { automargin: true, side: "bottom", gridcolor: theme.grid }, yaxis: { automargin: true, autorange: "reversed", gridcolor: theme.grid }, margin: { l: 80, r: 20, t: 40, b: 80 } }),
    };
  }

  if (chart === "Scatter matrix") {
    const cols = (spec.cols && spec.cols.length >= 2 ? spec.cols : nums.slice(0, 4)).filter((c) => nums.includes(c));
    if (cols.length < 2) return empty(theme, "Select at least 2 numeric columns.");
    const dims = cols.map((c) => ({ label: c.length > 10 ? c.slice(0, 10) + "…" : c, values: findCol(ds, c)!.values.map((x) => (x == null ? null : Number(x))) }));
    const t: Trace = { type: "splom", dimensions: dims, marker: { size: 3.5, opacity: 0.55, color: theme.accent, line: { width: 0 } }, diagonal: { visible: true }, showupperhalf: true };
    const colc = spec.color && spec.color !== "(none)" ? findCol(ds, spec.color) : undefined;
    if (colc && colc.type === "cat") {
      const cats = Array.from(new Set(colc.values.filter((x) => x != null).map(String)));
      const code = new Map(cats.map((c, i) => [c, i]));
      t.marker.color = colc.values.map((x) => (x == null ? 0 : code.get(String(x)) ?? 0));
      t.marker.colorscale = "Viridis";
    }
    return { data: [t], layout: base(theme, `Scatter matrix · ${cols.length} vars`, { margin: { l: 50, r: 20, t: 40, b: 40 }, dragmode: "select", height: undefined }) };
  }

  if (chart === "Parallel coordinates") {
    const cols = (spec.cols && spec.cols.length >= 2 ? spec.cols : nums.slice(0, 5)).filter((c) => nums.includes(c));
    if (cols.length < 2) return empty(theme, "Select at least 2 numeric columns.");
    const dims = cols.map((c) => { const vals = findCol(ds, c)!.values.map((x) => (x == null ? 0 : Number(x))); return { label: c, values: vals, range: [Math.min(...vals), Math.max(...vals)] }; });
    const line: Trace = { color: theme.colorway[0] };
    const colc = spec.color && spec.color !== "(none)" ? findCol(ds, spec.color) : undefined;
    if (colc) {
      if (colc.type === "num") line.color = colc.values.map((x) => (x == null ? 0 : Number(x)));
      else { const cats = Array.from(new Set(colc.values.filter((x) => x != null).map(String))); const code = new Map(cats.map((c, i) => [c, i])); line.color = colc.values.map((x) => (x == null ? 0 : code.get(String(x)) ?? 0)); }
      line.colorscale = "Viridis"; line.showscale = true; line.colorbar = { thickness: 12, len: 0.7 };
    }
    return { data: [{ type: "parcoords", dimensions: dims, line }], layout: base(theme, `Parallel coordinates · ${cols.length} vars`, { margin: { l: 60, r: 40, t: 60, b: 30 } }) };
  }

  if (chart === "Box by group" || chart === "Violin by group") {
    const yn = spec.y && nums.includes(spec.y) ? spec.y : nums[0];
    const xn = spec.x && catCols(ds).includes(spec.x) ? spec.x : catCols(ds)[0];
    if (!yn || !xn) return empty(theme, "Need a numeric column and a categorical group.");
    const cx = findCol(ds, xn)!, cy = findCol(ds, yn)!;
    const gx: string[] = [], gy: number[] = [];
    for (let i = 0; i < ds.nrows; i++) { if (cx.values[i] != null && cy.values[i] != null) { gx.push(String(cx.values[i])); gy.push(Number(cy.values[i])); } }
    const isV = chart === "Violin by group";
    const t: Trace = isV
      ? { type: "violin", x: gx, y: gy, box: { visible: true }, meanline: { visible: true }, points: false }
      : { type: "box", x: gx, y: gy, boxpoints: "outliers" };
    return { data: [t], layout: base(theme, `${yn} by ${xn}`, { xaxis: { automargin: true, gridcolor: theme.grid }, yaxis: { title: yn, gridcolor: theme.grid } }) };
  }

  if (chart === "Grouped bar" || chart === "Stacked bar") {
    const xn = spec.x && catCols(ds).includes(spec.x) ? spec.x : catCols(ds)[0];
    if (!xn) return empty(theme, "Need a categorical column for the x-axis.");
    const cx = findCol(ds, xn)!;
    const colorSel = spec.color && spec.color !== "(none)" ? findCol(ds, spec.color) : undefined;
    const ynum = spec.y && nums.includes(spec.y) ? spec.y : undefined; // if set, aggregate mean; else count
    const cats = catCounts(cx, 30).labels;
    const barmode = chart === "Stacked bar" ? "stack" : "group";
    if (colorSel && colorSel.type === "cat") {
      const groups = catCounts(colorSel, 12).labels;
      const data: Trace[] = groups.map((gname) => {
        const y = cats.map((cat) => {
          let sum = 0, cnt = 0;
          for (let i = 0; i < ds.nrows; i++) {
            if (String(cx.values[i]) === cat && String(colorSel.values[i]) === gname) {
              if (ynum) { const yv = findCol(ds, ynum)!.values[i]; if (yv != null) { sum += Number(yv); cnt++; } } else cnt++;
            }
          }
          return ynum ? (cnt ? sum / cnt : 0) : cnt;
        });
        return { type: "bar", name: gname, x: cats, y };
      });
      return { data, layout: base(theme, `${xn}${ynum ? ` · mean ${ynum}` : " · counts"} by ${colorSel.name}`, { barmode, xaxis: { automargin: true, gridcolor: theme.grid } }) };
    }
    // single series
    const y = cats.map((cat) => {
      let sum = 0, cnt = 0;
      for (let i = 0; i < ds.nrows; i++) if (String(cx.values[i]) === cat) { if (ynum) { const yv = findCol(ds, ynum)!.values[i]; if (yv != null) { sum += Number(yv); cnt++; } } else cnt++; }
      return ynum ? (cnt ? sum / cnt : 0) : cnt;
    });
    return { data: [{ type: "bar", x: cats, y, marker: { color: theme.accent } }], layout: base(theme, `${xn}${ynum ? ` · mean ${ynum}` : " · counts"}`, { barmode, xaxis: { automargin: true, gridcolor: theme.grid } }) };
  }

  if (chart === "2D density") {
    const xn = spec.x && nums.includes(spec.x) ? spec.x : nums[0];
    const yn = spec.y && nums.includes(spec.y) ? spec.y : nums[1] || nums[0];
    if (!xn || !yn) return empty(theme, "Need two numeric columns.");
    const { x, y } = pairs(ds, xn, yn);
    return {
      data: [
        { type: "histogram2dcontour", x, y, colorscale: "Blues", reversescale: false, showscale: true, colorbar: { thickness: 12, len: 0.7 }, ncontours: 18 },
        { type: "scatter", x, y, mode: "markers", marker: { color: theme.accent, size: 3, opacity: 0.35 }, name: "points" },
      ],
      layout: base(theme, `${xn} vs ${yn} — 2D density`, { xaxis: { title: xn, gridcolor: theme.grid }, yaxis: { title: yn, gridcolor: theme.grid } }),
    };
  }

  // Scatter / Line (num vs num), optional color-by
  const xn = spec.x && nums.includes(spec.x) ? spec.x : nums[0];
  const yn = spec.y && nums.includes(spec.y) ? spec.y : nums[1] || nums[0];
  if (!xn || !yn) return empty(theme, "Need two numeric columns.");
  const isLine = chart === "Line";
  const colc = spec.color && spec.color !== "(none)" ? findCol(ds, spec.color) : undefined;
  const cx = findCol(ds, xn)!, cy = findCol(ds, yn)!;

  if (colc && colc.type === "cat") {
    const groups = catCounts(colc, 10).labels;
    const data: Trace[] = groups.map((gname) => {
      const xs: number[] = [], ys: number[] = [];
      for (let i = 0; i < ds.nrows; i++) if (String(colc.values[i]) === gname && cx.values[i] != null && cy.values[i] != null) { xs.push(Number(cx.values[i])); ys.push(Number(cy.values[i])); }
      if (isLine) { const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]); return { type: "scatter", mode: "lines+markers", name: gname, x: order.map((i) => xs[i]), y: order.map((i) => ys[i]), marker: { size: 5 } }; }
      return { type: "scatter", mode: "markers", name: gname, x: xs, y: ys, marker: { size: 6, opacity: 0.7 } };
    });
    return { data, layout: base(theme, `${xn} vs ${yn} · by ${colc.name}`, { xaxis: { title: xn, gridcolor: theme.grid }, yaxis: { title: yn, gridcolor: theme.grid } }) };
  }

  const { x, y } = pairs(ds, xn, yn);
  const r = pearson(x, y);
  const data: Trace[] = [];
  if (colc && colc.type === "num") {
    const cv: number[] = []; const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < ds.nrows; i++) if (cx.values[i] != null && cy.values[i] != null && colc.values[i] != null) { xs.push(Number(cx.values[i])); ys.push(Number(cy.values[i])); cv.push(Number(colc.values[i])); }
    data.push({ type: "scatter", mode: "markers", x: xs, y: ys, marker: { size: 7, color: cv, colorscale: "Viridis", showscale: true, colorbar: { title: colc.name, thickness: 12, len: 0.7 }, opacity: 0.8 } });
  } else {
    data.push({ type: "scatter", mode: isLine ? "lines+markers" : "markers", x: isLine ? [...x].sort((a, b) => a - b) : x, y: isLine ? y.map((_, i) => y[x.map((_, j) => j).sort((a, b) => x[a] - x[b])[i]]) : y, marker: { size: 6, color: theme.accent, opacity: 0.65 }, line: { color: theme.accent } });
  }
  // OLS trendline
  if (spec.trend && !isLine && x.length > 1) {
    const n = x.length; const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
    const slope = sxy / (sxx || 1), intercept = my - slope * mx;
    const xmin = Math.min(...x), xmax = Math.max(...x);
    data.push({ type: "scatter", mode: "lines", x: [xmin, xmax], y: [slope * xmin + intercept, slope * xmax + intercept], line: { color: theme.colorway[4], width: 2, dash: "dash" }, name: `trend (r=${r.toFixed(2)})` });
  }
  return { data, layout: base(theme, `${xn} vs ${yn}${!isLine ? `  ·  r = ${r.toFixed(2)}` : ""}`, { xaxis: { title: xn, gridcolor: theme.grid }, yaxis: { title: yn, gridcolor: theme.grid }, showlegend: spec.trend && !isLine }) };
}

// ── auto insights: quick "understand this dataset" bullets ──
export interface Insight { kind: "info" | "warn" | "good"; text: string; }
export function datasetInsights(ds: Dataset, target: string): Insight[] {
  const out: Insight[] = [];
  const n = ds.nrows;
  // missing
  ds.columns.forEach((c) => {
    const miss = c.values.filter((v) => v == null).length;
    if (miss / n > 0.2) out.push({ kind: "warn", text: `“${c.name}” is ${((miss / n) * 100).toFixed(0)}% missing — consider imputing or dropping.` });
  });
  // constant / high-cardinality
  ds.columns.forEach((c) => {
    const uniq = new Set(c.values.filter((v) => v != null).map(String)).size;
    if (uniq <= 1) out.push({ kind: "warn", text: `“${c.name}” is constant (1 unique value) — no signal, safe to drop.` });
    else if (c.type === "cat" && uniq > 0.5 * n && uniq > 20) out.push({ kind: "info", text: `“${c.name}” is high-cardinality (${uniq} categories) — likely an ID; one-hot will explode dimensions.` });
  });
  // skew
  ds.columns.filter((c) => c.type === "num").forEach((c) => {
    const v = c.values.filter((x): x is number => x != null).map(Number);
    const s = skewness(v);
    if (Math.abs(s) > 1.5) out.push({ kind: "info", text: `“${c.name}” is ${s > 0 ? "right" : "left"}-skewed (skew ${s.toFixed(1)}) — a log/sqrt transform may help.` });
  });
  // correlations
  const nums = ds.columns.filter((c) => c.type === "num");
  for (let i = 0; i < nums.length; i++)
    for (let j = i + 1; j < nums.length; j++) {
      const a = nums[i].values.filter((x): x is number => x != null).map(Number);
      const b = nums[j].values.filter((x): x is number => x != null).map(Number);
      const r = pearson(a, b);
      if (Math.abs(r) > 0.85) out.push({ kind: "info", text: `“${nums[i].name}” & “${nums[j].name}” are strongly correlated (r=${r.toFixed(2)}) — possible redundancy.` });
    }
  // target balance
  const tcol = findCol(ds, target);
  if (tcol && tcol.type === "cat") {
    const { labels, counts } = catCounts(tcol);
    const total = counts.reduce((a, b) => a + b, 0);
    const maj = counts[0] / total;
    if (maj > 0.7) out.push({ kind: "warn", text: `Target “${target}” is imbalanced — “${labels[0]}” is ${(maj * 100).toFixed(0)}% of rows. Watch accuracy vs. F1.` });
    else out.push({ kind: "good", text: `Target “${target}” is reasonably balanced across ${labels.length} classes.` });
  }
  if (!out.length) out.push({ kind: "good", text: "No obvious data-quality issues detected — clean dataset." });
  return out.slice(0, 8);
}
