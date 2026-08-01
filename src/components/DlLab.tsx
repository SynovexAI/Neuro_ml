"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  genDataset, genSeries, windowSeries, initNet, newOpt, trainEpoch, fullEval, predictVec, predictClass,
  fitScaler, applyScaler, scaleRow, dlSurface, classWeights,
  type Net, type OptState, type DlTask, type Optimizer, type DlEval, type ScaleMethod,
} from "@/lib/dlUtils";
import { parseCSV, buildMatrix, type Dataset, type PrepStep } from "@/lib/mlUtils";
import { pca2 } from "@/lib/ragUtils";
import { plotlyTheme } from "@/lib/edaCharts";
import Plot from "@/components/Plot";

type Step = "data" | "explore" | "prep" | "arch" | "train" | "test";
const STEPS: { k: Step; n: number; label: string }[] = [
  { k: "data", n: 1, label: "Data" }, { k: "explore", n: 2, label: "Explore" }, { k: "prep", n: 3, label: "Preprocess" },
  { k: "arch", n: 4, label: "Architecture" }, { k: "train", n: 5, label: "Train" }, { k: "test", n: 6, label: "Test & Export" },
];
const SAMPLES = [
  { k: "spiral", l: "Spiral", t: "binary", d: "Two interleaving spirals", why: "Needs a curved boundary — a linear model fails, a 2-layer net succeeds. A great first lesson." },
  { k: "circles", l: "Circles", t: "binary", d: "A ring inside a ring", why: "Concentric classes — impossible to split with a straight line, easy for a small net." },
  { k: "xor", l: "XOR", t: "binary", d: "Diagonal quadrants", why: "The classic non-separable case that stumped the single-layer perceptron." },
  { k: "moons", l: "Moons", t: "binary", d: "Two crescents", why: "Mildly non-linear — a small hidden layer separates them cleanly." },
  { k: "blobs3", l: "3 Blobs", t: "multiclass", d: "Three clusters → softmax", why: "Three classes — the output layer uses softmax + cross-entropy." },
  { k: "sine", l: "Sine", t: "regression", d: "1 feature → continuous target", why: "Regression — the net fits a smooth curve with a linear output and MSE loss." },
];
const PAL = ["#5b7cff", "#f59e0b", "#3ecf7f", "#ef4444", "#a855f7", "#22b8cf", "#ec4899", "#84cc16"];
const TS = [
  { k: "air", l: "Airline passengers", d: "Trend + yearly seasonality", n: 144, unit: "monthly" },
  { k: "temp", l: "Daily temperature", d: "Seasonal cycle + noise", n: 240, unit: "daily" },
  { k: "traffic", l: "Web traffic", d: "Trend + weekly cycle", n: 200, unit: "daily" },
];

type Resolved = { X: number[][]; y: number[]; task: DlTask; classes: string[]; featNames: string[]; source: string; series?: { t: number[]; v: number[] }; chronological?: boolean; win?: number };

// deterministic split — chronological (no shuffle) for time series, else seeded random
function splitData(X: number[][], y: number[], testFrac: number, chronological = false) {
  const n = X.length; const nTe = Math.max(1, Math.round(n * testFrac));
  if (chronological) { const cut = n - nTe; return { Xtr: X.slice(0, cut), ytr: y.slice(0, cut), Xte: X.slice(cut), yte: y.slice(cut) }; }
  const idx = [...Array(n).keys()];
  let a = 12345; for (let i = n - 1; i > 0; i--) { a = (a * 1103515245 + 12345) & 0x7fffffff; const j = a % (i + 1); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const te = new Set(idx.slice(0, nTe));
  const Xtr: number[][] = [], ytr: number[] = [], Xte: number[][] = [], yte: number[] = [];
  X.forEach((r, i) => { if (te.has(i)) { Xte.push(r); yte.push(y[i]); } else { Xtr.push(r); ytr.push(y[i]); } });
  return { Xtr, ytr, Xte, yte };
}

export default function DlLab() {
  const [step, setStep] = useState<Step>("data");
  const [source, setSource] = useState<"sample" | "csv" | "ts">("sample");
  const [tsKind, setTsKind] = useState("air");
  const [winSize, setWinSize] = useState(12);
  const [toy, setToy] = useState("spiral");
  const [noise, setNoise] = useState(0.12);
  const [ds, setDs] = useState<Dataset | null>(null);
  const [dsName, setDsName] = useState("");
  const [feats, setFeats] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const [data, setData] = useState<Resolved | null>(null);
  const [testFrac, setTestFrac] = useState(0.25);
  const [scaleMethod, setScaleMethod] = useState<ScaleMethod>("standard");
  const [imputeMethod, setImputeMethod] = useState<"Mean" | "Median" | "Most frequent" | "Constant">("Mean");
  const [encMethod, setEncMethod] = useState<"One-Hot" | "Ordinal" | "Frequency" | "Binary">("One-Hot");
  const [balanceClasses, setBalanceClasses] = useState(false);
  const [exFx, setExFx] = useState(0);
  const [exFy, setExFy] = useState(1);
  const [exMode, setExMode] = useState<"scatter" | "pca" | "dist" | "corr">("scatter");

  const [hidden, setHidden] = useState<number[]>([8, 6]);
  const [act, setAct] = useState("tanh");
  const [optimizer, setOptimizer] = useState<Optimizer>("adam");
  const [lr, setLr] = useState(0.02);
  const [l2, setL2] = useState(0);
  const [batchSize, setBatchSize] = useState(16);
  const [epochsTarget, setEpochsTarget] = useState(300);

  const [epoch, setEpoch] = useState(0);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<{ ep: number; loss: number; acc: number; vloss: number; vacc: number }[]>([]);
  const [surface, setSurface] = useState<{ xs: number[]; ys: number[]; z: number[][] } | null>(null);
  const [evalR, setEvalR] = useState<DlEval | null>(null);
  const [tick, setTick] = useState(0);
  const netRef = useRef<Net | null>(null); const optRef = useRef<OptState | null>(null);
  const epochRef = useRef(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const scRef = useRef<{ mean: number[]; std: number[] } | null>(null);
  const splitRef = useRef<{ Xtr: number[][]; ytr: number[]; Xte: number[][]; yte: number[] } | null>(null);
  const rangeRef = useRef<{ lo: number[]; hi: number[] } | null>(null);
  const [testInput, setTestInput] = useState<number[]>([]);
  const [predResult, setPredResult] = useState<{ kind: "cls"; label: string; conf: number; probs: { name: string; p: number }[] } | { kind: "reg"; value: number } | null>(null);
  const [copied, setCopied] = useState(false);
  const th = plotlyTheme();

  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);

  // ── resolve dataset (sample, CSV, or time series) ──
  function resolveSample() {
    const d = genDataset(toy, 260, noise, 3);
    setData({ X: d.X, y: d.y, task: d.task, classes: d.classes, featNames: d.featNames, source: SAMPLES.find((s) => s.k === toy)?.l || toy });
    resetTraining(); setStep("explore");
  }
  function loadCSV(text: string, name: string) {
    try { const parsed = parseCSV(text); setDs(parsed); setDsName(name); const cols = parsed.columns.map((c) => c.name); setFeats(cols.slice(0, -1)); setTarget(cols[cols.length - 1]); setMsg(""); }
    catch (e) { setMsg("Parse error: " + (e as Error).message); }
  }
  function onFile(f: File | null) { if (!f) return; const r = new FileReader(); r.onload = () => loadCSV(String(r.result), f.name); r.readAsText(f); }
  function resolveSeries() {
    const s = genSeries(tsKind); const w = Math.min(winSize, s.v.length - 2); const win = windowSeries(s.v, w);
    setData({ X: win.X, y: win.y, task: "regression", classes: [], featNames: win.featNames, source: TS.find((t) => t.k === tsKind)?.l || tsKind, series: s, chronological: true, win: w });
    resetTraining(); setStep("explore");
  }
  function detectTask(d: Dataset, tgt: string): DlTask {
    const col = d.columns.find((c) => c.name === tgt); if (!col) return "binary";
    const vals = col.values.filter((v) => v != null); const uniq = new Set(vals.map(String)).size;
    if (col.type === "num" && uniq > 12) return "regression";
    return uniq <= 2 ? "binary" : "multiclass";
  }
  // Impute (chosen method) + encode (chosen method) steps for the selected feature columns.
  function prepSteps(d: Dataset, fcols: string[]): PrepStep[] {
    const steps: PrepStep[] = [];
    d.columns.filter((c) => fcols.includes(c.name) && c.name !== target).forEach((c) => {
      steps.push({ op: "Impute missing", cols: [c.name], method: imputeMethod });
      if (c.type === "cat") steps.push({ op: "Encode categorical", cols: [c.name], method: encMethod });
    });
    return steps;
  }
  function deriveCsv(): Resolved | null {
    if (!ds || !feats.length || !target) return null;
    const fcols = feats.filter((f) => f !== target);
    const dlTask = detectTask(ds, target);
    const b = buildMatrix(ds, fcols, target, dlTask === "regression" ? "regression" : "classification", prepSteps(ds, fcols));
    if (!b.X.length) return null;
    const classes = b.classes ?? [];
    const task: DlTask = dlTask === "regression" ? "regression" : (classes.length <= 2 ? "binary" : "multiclass");
    return { X: b.X, y: b.y, task, classes, featNames: b.featureNames, source: dsName || "uploaded.csv" };
  }
  function buildFromCsv() {
    if (!ds || !feats.length || !target) { setMsg("Pick at least one feature and a target."); return; }
    try {
      const d = deriveCsv();
      if (!d) { setMsg("No usable rows — pick at least one feature and a target."); return; }
      setData(d); resetTraining(); setStep("explore"); setMsg("");
    } catch (e) { setMsg("Build error: " + (e as Error).message); }
  }
  // Changing imputation or encoding re-runs buildMatrix → new fill values / input columns → reset training.
  useEffect(() => {
    if (!data || source !== "csv" || !ds) return;
    try { const d = deriveCsv(); if (d) { setData(d); resetTraining(); } } catch { /* keep prior matrix */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [encMethod, imputeMethod]);
  // Scaling method, split ratio, and class-balancing all change the matrix/loss → any trained net is stale.
  useEffect(() => {
    if (data && netRef.current) resetTraining();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scaleMethod, testFrac, balanceClasses]);
  function resetTraining() { stopTrain(); setEpoch(0); epochRef.current = 0; setHistory([]); setSurface(null); setEvalR(null); netRef.current = null; optRef.current = null; scRef.current = null; splitRef.current = null; }

  const outDim = data ? (data.task === "multiclass" ? data.classes.length : 1) : 1;
  const K = data ? (data.task === "binary" ? 2 : data.classes.length) : 0;

  // ── training ──
  function prepareSplit() {
    if (!data) return null;
    // Split raw first, then fit the scaler on TRAIN ONLY (no leakage), apply to both.
    const raw = splitData(data.X, data.y, testFrac, !!data.chronological);
    const sc = fitScaler(raw.Xtr, scaleMethod); scRef.current = sc;
    const sp = { Xtr: applyScaler(raw.Xtr, sc), ytr: raw.ytr, Xte: applyScaler(raw.Xte, sc), yte: raw.yte }; splitRef.current = sp;
    // raw feature ranges (first two) for boundary axes
    const f0 = data.X.map((r) => r[0]); const f1 = data.featNames.length > 1 ? data.X.map((r) => r[1]) : f0;
    rangeRef.current = { lo: [Math.min(...f0), Math.min(...f1)], hi: [Math.max(...f0), Math.max(...f1)] };
    return sp;
  }
  function startTrain() {
    if (!data) return; stopTrain();
    const sp = prepareSplit(); if (!sp) return;
    const net = initNet(data.X[0].length, hidden, outDim, act, data.task); netRef.current = net; optRef.current = newOpt(net, optimizer);
    const cw = balanceClasses && data.task !== "regression" ? classWeights(sp.ytr, K) : undefined;
    setHistory([]); setEpoch(0); setEvalR(null); setRunning(true); epochRef.current = 0;
    timer.current = setInterval(() => {
      const n = netRef.current!, o = optRef.current!, s = splitRef.current!;
      const stat = trainEpoch(n, s.Xtr, s.ytr, { lr, l2, batchSize }, o, cw);
      const veval = fullEval(n, s.Xte, s.yte, data.classes);
      const ep = ++epochRef.current;
      setEpoch(ep);
      setHistory((h) => [...h, { ep, loss: stat.loss, acc: stat.acc, vloss: veval.loss, vacc: veval.acc }]);
      if (data.featNames.length === 2 && data.task !== "regression" && (ep % 4 === 0 || ep === 1)) setSurface(dlSurface(n, scRef.current!, rangeRef.current!.lo, rangeRef.current!.hi, 44));
      setTick((t) => t + 1);
      if (ep >= epochsTarget) { stopTrain(); setEvalR(fullEval(n, s.Xte, s.yte, data.classes)); }
    }, 40);
  }
  function stopTrain() { if (timer.current) { clearInterval(timer.current); timer.current = null; } setRunning(false); }
  function finishNow() { if (netRef.current && splitRef.current && data) { stopTrain(); if (data.featNames.length === 2 && data.task !== "regression") setSurface(dlSurface(netRef.current, scRef.current!, rangeRef.current!.lo, rangeRef.current!.hi, 44)); setEvalR(fullEval(netRef.current, splitRef.current.Xte, splitRef.current.yte, data.classes)); } }

  // ── explore figures ──
  const exploreFig = useMemo(() => {
    if (!data) return null; const { X, y, task, classes, featNames } = data;
    // 1-feature regression → feature vs target
    if (task === "regression" && featNames.length === 1) return { data: [{ type: "scatter", mode: "markers", x: X.map((r) => r[0]), y, marker: { color: "#5b7cff", size: 6, opacity: 0.7 } }], title: `${featNames[0]} vs target`, xl: featNames[0], yl: "target" };
    // distribution of one feature, split by class
    if (exMode === "dist") { const fx = Math.min(exFx, featNames.length - 1);
      if (task === "regression") return { data: [{ type: "histogram", x: X.map((r) => r[fx]), marker: { color: "#5b7cff" }, opacity: 0.85 }], title: `distribution of ${featNames[fx]}`, xl: featNames[fx], yl: "count", legend: false };
      const traces = [...Array(K).keys()].map((c) => ({ type: "histogram", name: classes[c] ?? `class ${c}`, x: X.map((r, i) => (y[i] === c ? r[fx] : null)).filter((v) => v != null), marker: { color: PAL[c % PAL.length] }, opacity: 0.6 }));
      return { data: traces, title: `${featNames[fx]} by class`, xl: featNames[fx], yl: "count", barmode: "overlay" };
    }
    // scatter of two chosen features, or PCA-2
    const usePca = exMode === "pca" || featNames.length < 2;
    const fx = Math.min(exFx, featNames.length - 1), fy = Math.min(exFy, featNames.length - 1);
    const pts = usePca ? pca2(X) : X.map((r) => ({ x: r[fx], y: r[fy] }));
    const xl = usePca ? "PC1" : featNames[fx], yl = usePca ? "PC2" : featNames[fy];
    if (task === "regression") return { data: [{ type: "scatter", mode: "markers", x: pts.map((p) => p.x), y: pts.map((p) => p.y), marker: { color: y, colorscale: "Viridis", size: 7, showscale: true, colorbar: { title: { text: "target" } } } }], title: usePca ? "PCA projection (colour = target)" : "features coloured by target", xl, yl, legend: false };
    const traces = [...Array(K).keys()].map((c) => ({ type: "scatter", mode: "markers", name: classes[c] ?? `class ${c}`, x: pts.map((p, i) => (y[i] === c ? p.x : null)), y: pts.map((p, i) => (y[i] === c ? p.y : null)), marker: { color: PAL[c % PAL.length], size: 7, opacity: 0.8 } }));
    return { data: traces, title: usePca ? "PCA projection by class" : "data by class", xl, yl };
  }, [data, K, exFx, exFy, exMode]);
  const classCounts = useMemo(() => { if (!data || data.task === "regression") return null; const c = new Array(K).fill(0); data.y.forEach((v) => { c[v]++; }); return c; }, [data, K]);

  // preview table (first rows of the loaded CSV or the resolved matrix)
  function previewTable() {
    if (source === "csv" && ds) { const cols = ds.columns; const rows = Math.min(8, ds.nrows);
      return <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 8, marginTop: 10 }}><table className="dtable"><tbody>
        <tr><th>#</th>{cols.map((c) => <th key={c.name}>{c.name} <span style={{ color: c.name === target ? "var(--accent)" : "var(--faint)" }}>{c.type}{c.name === target ? "·target" : feats.includes(c.name) ? "·in" : ""}</span></th>)}</tr>
        {[...Array(rows).keys()].map((r) => <tr key={r}><td style={{ color: "var(--faint)" }}>{r}</td>{cols.map((c) => <td key={c.name}>{c.values[r] ?? "—"}</td>)}</tr>)}
      </tbody></table></div>;
    }
    return null;
  }
  const prf = (cm: number[][]) => { const K = cm.length; let p = 0, r = 0; for (let k = 0; k < K; k++) { const tp = cm[k][k]; const fp = cm.reduce((a, row) => a + row[k], 0) - tp; const fn = cm[k].reduce((a, b) => a + b, 0) - tp; p += tp / (tp + fp || 1); r += tp / (tp + fn || 1); } p /= K; r /= K; return { precision: p, recall: r, f1: 2 * p * r / (p + r || 1) }; };
  const lay = (title: string, xl: string, yl: string, extra: Record<string, unknown> = {}) => ({ title: { text: title, font: { size: 13, color: th.text } }, paper_bgcolor: th.paper, plot_bgcolor: th.plot, font: { color: th.muted, size: 11 }, margin: { l: 46, r: 16, t: 38, b: 46 }, xaxis: { title: { text: xl }, gridcolor: th.grid, zerolinecolor: th.grid }, yaxis: { title: { text: yl }, gridcolor: th.grid, zerolinecolor: th.grid }, colorway: PAL, ...extra });
  const discrete = (k: number): [number, string][] => { const cs: [number, string][] = []; for (let i = 0; i < k; i++) { const c = PAL[i % PAL.length]; cs.push([k <= 1 ? 0 : i / k, c]); cs.push([k <= 1 ? 1 : (i + 1) / k, c]); } return cs; };

  // ── network diagram (SVG) ──
  function netDiagram() {
    const sizes = [data ? data.X[0].length : (feats.length || 2), ...hidden, outDim];
    const W = 520, H = 300, padX = 60, padY = 30; const layerX = sizes.map((_, l) => padX + (W - 2 * padX) * (sizes.length === 1 ? 0.5 : l / (sizes.length - 1)));
    const nodeY = (l: number, i: number) => { const cnt = Math.min(sizes[l], 8); const yy = padY + (H - 2 * padY) * (cnt === 1 ? 0.5 : i / (cnt - 1)); return yy; };
    const net = netRef.current; const els: React.ReactNode[] = [];
    for (let l = 0; l < sizes.length - 1; l++) { const c0 = Math.min(sizes[l], 8), c1 = Math.min(sizes[l + 1], 8);
      for (let i = 0; i < c0; i++) for (let j = 0; j < c1; j++) { let col = "var(--border-strong)", op = 0.25, wd = 1; if (net && net.W[l] && net.W[l][j] && net.W[l][j][i] != null) { const w = net.W[l][j][i]; col = w >= 0 ? "#3ecf7f" : "#ef4444"; op = Math.min(0.85, 0.15 + Math.abs(w) * 0.5); wd = Math.min(3, 0.5 + Math.abs(w)); } els.push(<line key={`e${l}-${i}-${j}`} x1={layerX[l]} y1={nodeY(l, i)} x2={layerX[l + 1]} y2={nodeY(l + 1, j)} stroke={col} strokeWidth={wd} opacity={op} />); } }
    sizes.forEach((cnt, l) => { const shown = Math.min(cnt, 8); const kind = l === 0 ? "in" : l === sizes.length - 1 ? "out" : "hid"; const fill = kind === "in" ? "#5b7cff" : kind === "out" ? "#f59e0b" : "#a855f7";
      for (let i = 0; i < shown; i++) els.push(<circle key={`n${l}-${i}`} cx={layerX[l]} cy={nodeY(l, i)} r={9} fill={fill} stroke={th.paper} strokeWidth={1.5} />);
      if (cnt > 8) els.push(<text key={`m${l}`} x={layerX[l]} y={H - 8} textAnchor="middle" fill="var(--faint)" fontSize={10} fontFamily="monospace">+{cnt - 8}</text>);
      els.push(<text key={`lb${l}`} x={layerX[l]} y={16} textAnchor="middle" fill="var(--faint)" fontSize={10} fontFamily="monospace">{l === 0 ? `${cnt} in` : l === sizes.length - 1 ? `${cnt} out` : `h${l}·${cnt}`}</text>);
    });
    return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 560, borderRadius: 8, background: "var(--panel)", border: "1px solid var(--border)" }}>{els}</svg>;
  }

  // ── train visualization ──
  function trainViz() {
    if (!data) return null; const H = 300;
    if (data.featNames.length === 2 && data.task !== "regression" && surface) {
      const heat = { type: "heatmap", x: surface.xs, y: surface.ys, z: surface.z, showscale: false, colorscale: discrete(K), zmin: -0.5, zmax: K - 0.5, opacity: 0.35, hoverinfo: "skip" };
      const traces = [...Array(K).keys()].map((c) => ({ type: "scatter", mode: "markers", name: data.classes[c] ?? `class ${c}`, x: data.X.map((r, i) => (data.y[i] === c ? r[0] : null)), y: data.X.map((r, i) => (data.y[i] === c ? r[1] : null)), marker: { color: PAL[c % PAL.length], size: 6, line: { width: 1, color: th.paper } } }));
      return <Plot data={[heat, ...traces] as never} layout={lay(`decision boundary — epoch ${epoch}`, data.featNames[0], data.featNames[1], { showlegend: true, legend: { orientation: "h", y: -0.18 }, height: H }) as never} style={{ height: H, width: "100%" }} />;
    }
    if (data.task === "regression" && data.featNames.length === 1 && netRef.current && scRef.current) {
      const xs: number[] = [], ys: number[] = []; const lo = Math.min(...data.X.map((r) => r[0])), hi = Math.max(...data.X.map((r) => r[0]));
      for (let i = 0; i <= 60; i++) { const x = lo + (hi - lo) * i / 60; xs.push(x); ys.push(predictVec(netRef.current, scaleRow([x], scRef.current))[0]); }
      return <Plot data={[{ type: "scatter", mode: "markers", name: "data", x: data.X.map((r) => r[0]), y: data.y, marker: { color: "#5b7cff", size: 6, opacity: 0.55 } }, { type: "scatter", mode: "lines", name: "network fit", x: xs, y: ys, line: { color: "#f59e0b", width: 3 } }] as never} layout={lay(`fit — epoch ${epoch}`, data.featNames[0], "target", { showlegend: true, legend: { orientation: "h", y: -0.18 }, height: H }) as never} style={{ height: H, width: "100%" }} />;
    }
    // >2 features or multiclass with many features → predicted-vs-actual / correctness scatter on PCA
    if (netRef.current && scRef.current) {
      const pts = pca2(data.X); const correct = data.X.map((r, i) => (data.task === "regression" ? 0 : (predictClass(netRef.current!, scaleRow(r, scRef.current!)) === data.y[i] ? 1 : 0)));
      if (data.task === "regression") return <Plot data={[{ type: "scatter", mode: "markers", x: data.y, y: data.X.map((r) => predictVec(netRef.current!, scaleRow(r, scRef.current!))[0]), marker: { color: "#5b7cff", size: 6, opacity: 0.6 } }, { type: "scatter", mode: "lines", x: [Math.min(...data.y), Math.max(...data.y)], y: [Math.min(...data.y), Math.max(...data.y)], line: { color: th.muted, dash: "dash" }, hoverinfo: "skip" }] as never} layout={lay(`predicted vs actual — epoch ${epoch}`, "actual", "predicted", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />;
      return <Plot data={[{ type: "scatter", mode: "markers", name: "correct", x: pts.map((p, i) => (correct[i] ? p.x : null)), y: pts.map((p, i) => (correct[i] ? p.y : null)), marker: { color: "#3ecf7f", size: 6 } }, { type: "scatter", mode: "markers", name: "wrong", x: pts.map((p, i) => (!correct[i] ? p.x : null)), y: pts.map((p, i) => (!correct[i] ? p.y : null)), marker: { color: "#ef4444", size: 7 } }] as never} layout={lay(`predictions on PCA — epoch ${epoch}`, "PC1", "PC2", { showlegend: true, legend: { orientation: "h", y: -0.18 }, height: H }) as never} style={{ height: H, width: "100%" }} />;
    }
    return <div className="note">Train to see the network fit your data.</div>;
  }
  void tick;

  // weight-distribution histogram (all network weights) — spreads from ~0 as it learns
  function weightHist() {
    const n = netRef.current; if (!n) return <div className="note" style={{ padding: "50px 0", textAlign: "center" }}>weight distribution appears once training starts</div>;
    const w: number[] = []; n.W.forEach((l) => l.forEach((r) => r.forEach((v) => w.push(v)))); const H = 280;
    const mad = w.reduce((a, v) => a + Math.abs(v), 0) / (w.length || 1);
    return <Plot data={[{ type: "histogram", x: w, marker: { color: "#a855f7" }, opacity: 0.85, nbinsx: 30 }] as never} layout={lay(`weight distribution — ${w.length} weights, mean |w| = ${mad.toFixed(2)}`, "weight value", "count", { height: H, showlegend: false, bargap: 0.03 }) as never} style={{ height: H, width: "100%" }} />;
  }
  // per-epoch learning log — how loss / metrics move each epoch (newest first)
  function learningLog() {
    const metric = data?.task === "regression" ? "R²" : "acc";
    if (!history.length) return <div className="note" style={{ padding: "50px 12px", textAlign: "center" }}>the learning log fills in as it trains — one row per epoch, newest on top.</div>;
    const total = history.length; const stepN = Math.max(1, Math.ceil(total / 60));
    const sampled = history.filter((_, i) => i % stepN === 0 || i === total - 1);
    const rows = sampled.slice().reverse();
    return <div><div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}><h4 className="fld" style={{ margin: 0 }}>Learning log</h4><span className="note">{total} epoch{total === 1 ? "" : "s"}{stepN > 1 ? ` · every ${stepN}` : ""}</span></div>
      <div style={{ maxHeight: 300, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 8 }}>
        <table className="dtable" style={{ width: "100%", fontFamily: "var(--mono)", fontSize: 11.5 }}><tbody>
          <tr><th style={{ position: "sticky", top: 0, background: "var(--panel)" }}>epoch</th><th style={{ position: "sticky", top: 0, background: "var(--panel)" }}>loss</th><th style={{ position: "sticky", top: 0, background: "var(--panel)" }}>train {metric}</th><th style={{ position: "sticky", top: 0, background: "var(--panel)" }}>val {metric}</th><th style={{ position: "sticky", top: 0, background: "var(--panel)" }} /></tr>
          {rows.map((r, i) => { const older = rows[i + 1]; const down = older ? r.loss < older.loss : true; const latest = i === 0; return <tr key={r.ep} style={latest ? { background: "var(--panel-2)" } : undefined}><td style={{ color: latest ? "var(--accent)" : undefined, fontWeight: latest ? 600 : undefined }}>{r.ep}</td><td>{r.loss.toFixed(4)}</td><td style={{ color: "var(--good)" }}>{r.acc.toFixed(3)}</td><td style={{ color: "#a855f7" }}>{r.vacc.toFixed(3)}</td><td style={{ color: down ? "var(--good)" : "var(--crit)" }}>{down ? "▼" : "▲"}</td></tr>; })}
        </tbody></table>
      </div>
      <div className="note" style={{ marginTop: 6 }}>▼ loss fell vs the previous shown epoch · ▲ it rose. A steady ▼ that flattens = the model has learned what it can.</div>
    </div>;
  }
  // curves
  const curveFig = () => { if (!history.length) return null; const H = 280; const metric = data?.task === "regression" ? "R²" : "accuracy";
    return { loss: <Plot data={[{ type: "scatter", mode: "lines", name: "train", x: history.map((h) => h.ep), y: history.map((h) => h.loss), line: { color: "#5b7cff", width: 2 } }, { type: "scatter", mode: "lines", name: "val", x: history.map((h) => h.ep), y: history.map((h) => h.vloss), line: { color: "#f59e0b", width: 2, dash: "dot" } }] as never} layout={lay("loss ↓", "epoch", "loss", { showlegend: true, legend: { orientation: "h", y: -0.3 }, height: H }) as never} style={{ height: H, width: "100%" }} />,
      acc: <Plot data={[{ type: "scatter", mode: "lines", name: "train", x: history.map((h) => h.ep), y: history.map((h) => h.acc), line: { color: "#3ecf7f", width: 2 } }, { type: "scatter", mode: "lines", name: "val", x: history.map((h) => h.ep), y: history.map((h) => h.vacc), line: { color: "#a855f7", width: 2, dash: "dot" } }] as never} layout={lay(`${metric} ↑`, "epoch", metric, { showlegend: true, legend: { orientation: "h", y: -0.3 }, height: H, yaxis: { title: { text: metric } } }) as never} style={{ height: H, width: "100%" }} /> };
  };

  // pytorch export
  const pyCode = useMemo(() => {
    if (!data) return "";
    const layers: string[] = []; const sizes = [data.X[0].length, ...hidden];
    for (let l = 0; l < hidden.length; l++) { layers.push(`    nn.Linear(${sizes[l]}, ${sizes[l + 1]}),`); layers.push(`    nn.${act === "relu" ? "ReLU" : act === "sigmoid" ? "Sigmoid" : "Tanh"}(),`); }
    layers.push(`    nn.Linear(${sizes[sizes.length - 1]}, ${outDim}),`);
    const loss = data.task === "regression" ? "nn.MSELoss()" : data.task === "binary" ? "nn.BCEWithLogitsLoss()" : "nn.CrossEntropyLoss()";
    const optLine = optimizer === "adam" ? `torch.optim.Adam(model.parameters(), lr=${lr})` : optimizer === "momentum" ? `torch.optim.SGD(model.parameters(), lr=${lr}, momentum=0.9)` : `torch.optim.SGD(model.parameters(), lr=${lr})`;
    return `import torch, torch.nn as nn\n\nmodel = nn.Sequential(\n${layers.join("\n")}\n)\ncriterion = ${loss}\noptimizer = ${optLine}\n\n# X: (n, ${data.X[0].length}) standardized features   y: ${data.task === "regression" ? "(n,) float" : data.task === "binary" ? "(n,) 0/1" : "(n,) class index"}\nfor epoch in range(${epochsTarget}):\n    optimizer.zero_grad()\n    out = model(X)\n    loss = criterion(out${data.task === "binary" ? ".squeeze(1)" : ""}, y${data.task === "regression" ? ".unsqueeze(1)" : ""})\n    loss.backward(); optimizer.step()`;
  }, [data, hidden, act, outDim, optimizer, lr, epochsTarget]);

  // ── test-page helpers ──
  function doPredict() {
    if (!netRef.current || !scRef.current || !data) return;
    const x = data.featNames.map((_, j) => testInput[j] ?? 0); const o = predictVec(netRef.current, scaleRow(x, scRef.current));
    if (data.task === "regression") setPredResult({ kind: "reg", value: o[0] });
    else { const c = o.indexOf(Math.max(...o)); const probs = o.map((p, i) => ({ name: String(data.classes[i] ?? i), p })).sort((a, b) => b.p - a.p).slice(0, 5); setPredResult({ kind: "cls", label: String(data.classes[c] ?? c), conf: Math.max(...o), probs }); }
  }
  function copyCode() { navigator.clipboard?.writeText(pyCode).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1400); }).catch(() => {}); }
  function downloadFile(name: string, content: string, mime: string) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([content], { type: mime })); a.download = name; a.click(); URL.revokeObjectURL(a.href); }
  function modelJson() { const n = netRef.current; if (!n || !data) return "{}"; return JSON.stringify({ task: data.task, classes: data.classes, featNames: data.featNames, sizes: n.sizes, activation: n.act, scaler: scRef.current, weights: n.W, biases: n.b }, null, 2); }
  const mcard = (v: string, k: string, accent = false, badge?: [string, string]) => <div key={k} style={{ background: "linear-gradient(160deg, var(--panel), var(--surface))", border: `1px solid ${accent ? "var(--accent)" : "var(--border)"}`, borderRadius: 12, padding: "14px 16px", position: "relative", overflow: "hidden" }}><div style={{ fontSize: 28, fontWeight: 600, color: accent ? "var(--accent)" : "var(--text)", lineHeight: 1 }}>{v}</div><div className="note" style={{ marginTop: 6, textTransform: "uppercase", letterSpacing: ".04em" }}>{k}</div>{badge && <span style={{ position: "absolute", right: 10, top: 12, fontSize: 11, color: badge[1], fontFamily: "var(--mono)" }}>{badge[0]}</span>}</div>;
  const panelSt: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 12, padding: 16, background: "var(--panel)" };
  // shared premium panel primitives
  const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 14, background: "var(--panel)", overflow: "hidden" };
  const pnlBody: React.CSSProperties = { padding: 16 };
  const secHead = (dot: string, title: string, right?: React.ReactNode) => <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "11px 16px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}><div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} /><span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--muted)" }}>{title}</span></div>{right}</div>;
  const statCard = (v: React.ReactNode, k: string, color?: string) => <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 11, padding: "11px 13px" }}><div style={{ fontSize: 20, fontWeight: 600, color: color || "var(--text)" }}>{v}</div><div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--faint)", marginTop: 3 }}>{k}</div></div>;
  const canExplore = !!data, best = history.length ? history[history.length - 1] : null;

  return (
    <>
      <div className="lab-head">
        <div><h1 className="page-h">Deep Learning Lab</h1><p className="page-sub" style={{ margin: 0 }}>Design a neural network, then train it — for real, in your browser — on a built-in set or your own CSV. Classification or regression on tabular data (no GPU needed).</p></div>
      </div>

      <div className="stepper" style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "14px 0" }}>
        {STEPS.map((s) => { const enabled = s.k === "data" || canExplore; return <button key={s.k} className={`stepbtn ${step === s.k ? "on" : ""}`} disabled={!enabled} onClick={() => enabled && setStep(s.k)} style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 12px", borderRadius: 8, border: "1px solid var(--border)", background: step === s.k ? "var(--accent)" : "var(--surface)", color: step === s.k ? "#fff" : "var(--text)", opacity: enabled ? 1 : 0.4, cursor: enabled ? "pointer" : "default", fontSize: 13 }}><b>{s.n}</b>{s.label}</button>; })}
      </div>
      {msg && <div className="warnbar">{msg}</div>}

      {step === "data" && (() => {
        const taskBadge = (t: string) => { const m = t === "multiclass" ? ["var(--purple)", "rgba(168,85,247,.13)", "multiclass"] : t === "regression" ? ["#3ecf7f", "rgba(62,207,127,.13)", "regression"] : ["var(--accent)", "rgba(91,124,255,.13)", "binary"]; return <span style={{ fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".04em", padding: "2px 7px", borderRadius: 20, color: m[0], background: m[1] }}>{m[2]}</span>; };
        const scatter = (kind: string, w: number, h: number, r: number, n = 120, nz = 0.08) => {
          const d = genDataset(kind, n, nz, 7); const reg = d.task === "regression" && d.featNames.length === 1;
          const ys = reg ? d.y : d.X.map((p) => p[1]); const xs = d.X.map((p) => p[0]);
          const xmn = Math.min(...xs), xmx = Math.max(...xs), ymn = Math.min(...ys), ymx = Math.max(...ys); const pad = 6;
          const px = (v: number) => pad + ((v - xmn) / ((xmx - xmn) || 1)) * (w - 2 * pad); const py = (v: number) => h - pad - ((v - ymn) / ((ymx - ymn) || 1)) * (h - 2 * pad);
          return <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} style={{ display: "block", background: "var(--panel-2)", borderRadius: 8 }}>{d.X.map((p, i) => <circle key={i} cx={px(p[0])} cy={py(reg ? d.y[i] : p[1])} r={r} fill={reg ? "#3ecf7f" : PAL[d.y[i] % PAL.length]} opacity={0.8} />)}</svg>;
        };
        const seg = (v: "sample" | "csv" | "ts", label: string) => <button onClick={() => setSource(v)} style={{ padding: "7px 14px", borderRadius: 8, border: "none", background: source === v ? "var(--accent)" : "transparent", color: source === v ? "#fff" : "var(--muted)", fontSize: 12.5, fontWeight: 500, cursor: "pointer" }}>{label}</button>;
        const lineSvg = (vals: number[], w: number, h: number, color = "#3ecf7f") => { if (!vals.length) return null; const mn = Math.min(...vals), mx = Math.max(...vals), pad = 5; let dd = ""; vals.forEach((v, i) => { const x = pad + i / (vals.length - 1) * (w - 2 * pad); const y = h - pad - ((v - mn) / ((mx - mn) || 1)) * (h - 2 * pad); dd += (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1) + " "; }); return <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} style={{ display: "block", background: "var(--panel-2)", borderRadius: 8 }}><path d={dd} fill="none" stroke={color} strokeWidth={1.6} /></svg>; };
        const sel = SAMPLES.find((s) => s.k === toy) || SAMPLES[0]; const prev = genDataset(toy, 260, noise, 3);
        const tsData = genSeries(tsKind);
        const typePill = (t: string) => <span style={{ marginLeft: "auto", fontSize: 9, padding: "1px 6px", borderRadius: 20, color: t === "cat" ? "var(--purple)" : "var(--accent)", background: t === "cat" ? "rgba(168,85,247,.12)" : "rgba(91,124,255,.12)" }}>{t}</span>;
        const missTotal = ds ? ds.columns.reduce((a, c) => a + c.values.filter((v) => v == null).length, 0) : 0;
        return <div className="card">
          <div className="card-h"><span className="t">Choose data</span></div>
          <div className="card-b">
            <div className="note" style={{ marginTop: -4, marginBottom: 14 }}>Learn the flow on a built-in shape, forecast a time series, or upload your own CSV.</div>
            <div style={{ display: "inline-flex", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: 3, marginBottom: 18 }}>{seg("sample", "◆ Built-in shapes")}{seg("ts", "⏱ Time series")}{seg("csv", "⬆ Upload CSV")}</div>

            {source === "ts" ? <>
              <div style={{ fontSize: 10.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--faint)", marginBottom: 10 }}>Forecasting datasets — predict the next value from recent history</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                {TS.map((s) => <div key={s.k} onClick={() => setTsKind(s.k)} style={{ background: "var(--panel)", border: `1px solid ${tsKind === s.k ? "var(--accent)" : "var(--border)"}`, boxShadow: tsKind === s.k ? "0 0 0 1px var(--accent)" : "none", borderRadius: 12, padding: 12, cursor: "pointer" }}>
                  {lineSvg(genSeries(s.k).v, 150, 56)}
                  <div style={{ fontWeight: 600, fontSize: 13, marginTop: 9, display: "flex", alignItems: "center", justifyContent: "space-between" }}>{s.l}<span style={{ fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".04em", padding: "2px 7px", borderRadius: 20, color: "#3ecf7f", background: "rgba(62,207,127,.13)" }}>forecast</span></div>
                  <div className="note" style={{ marginTop: 2 }}>{s.d} · {s.n} pts</div>
                </div>)}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 18, alignItems: "center", marginTop: 14, background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: 14 }}>
                <div>{lineSvg(tsData.v, 340, 120)}</div>
                <div>
                  <div className="knob"><div className="kr"><span>Window size (lags)</span><b>{Math.min(winSize, tsData.v.length - 2)}</b></div><input type="range" min={3} max={24} step={1} value={winSize} onChange={(e) => setWinSize(+e.target.value)} /></div>
                  <div className="row" style={{ gap: 16, marginTop: 12 }}>
                    <div><div style={{ fontSize: 16, fontWeight: 600 }}>{tsData.v.length}</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>points</div></div>
                    <div><div style={{ fontSize: 16, fontWeight: 600 }}>{Math.max(0, tsData.v.length - 1 - Math.min(winSize, tsData.v.length - 2))}</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>windows</div></div>
                    <div><div style={{ fontSize: 16, fontWeight: 600, color: "var(--accent)" }}>{Math.min(winSize, tsData.v.length - 2)}</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>features</div></div>
                  </div>
                  <div style={{ fontSize: 11.5, color: "var(--faint)", marginTop: 10 }}>The lab differences the series (predicts each <b style={{ color: "var(--text)" }}>change</b> from the last <b style={{ color: "var(--text)" }}>{Math.min(winSize, tsData.v.length - 2)}</b> changes, then reconstructs the level) so the trend can&apos;t run off-scale. Split is chronological — past trains, future tests.</div>
                </div>
              </div>
              <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={resolveSeries}>Use this series →</button></div>
            </> : source === "sample" ? <>
              <div style={{ fontSize: 10.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--faint)", marginBottom: 10 }}>Pick a shape — each teaches a different decision boundary</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                {SAMPLES.map((s) => <div key={s.k} onClick={() => setToy(s.k)} style={{ background: "var(--panel)", border: `1px solid ${toy === s.k ? "var(--accent)" : "var(--border)"}`, boxShadow: toy === s.k ? "0 0 0 1px var(--accent)" : "none", borderRadius: 12, padding: 12, cursor: "pointer" }}>
                  {scatter(s.k, 150, 74, 2.4)}
                  <div style={{ fontWeight: 600, fontSize: 13, marginTop: 9, display: "flex", alignItems: "center", justifyContent: "space-between" }}>{s.l}{taskBadge(s.t)}</div>
                  <div className="note" style={{ marginTop: 2 }}>{s.d}</div>
                </div>)}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "210px 1fr auto", gap: 18, alignItems: "center", marginTop: 14, background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: 14 }}>
                <div>{scatter(toy, 210, 120, 2.8, 260, noise)}</div>
                <div>
                  <div className="knob" style={{ maxWidth: 280 }}><div className="kr"><span>Noise</span><b>{noise.toFixed(2)}</b></div><input type="range" min={0} max={0.4} step={0.02} value={noise} onChange={(e) => setNoise(+e.target.value)} /></div>
                  <div className="row" style={{ gap: 18, marginTop: 12 }}>
                    <div><div style={{ fontSize: 16, fontWeight: 600 }}>260</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>samples</div></div>
                    <div><div style={{ fontSize: 16, fontWeight: 600 }}>{prev.X[0].length}</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>features</div></div>
                    <div><div style={{ fontSize: 16, fontWeight: 600, color: "var(--accent)" }}>{prev.task === "regression" ? "cont." : prev.classes.length}</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>{prev.task === "regression" ? "target" : "classes"}</div></div>
                  </div>
                  <div style={{ fontSize: 11.5, color: "var(--faint)", marginTop: 10, maxWidth: 380 }}>{sel.why}</div>
                </div>
                <button className="btn" onClick={resolveSample}>Use this dataset →</button>
              </div>
            </> : <>
              <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" hidden onChange={(e) => onFile(e.target.files?.[0] || null)} />
              <div onClick={() => fileRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files?.[0] || null); }} style={{ border: "1.5px dashed var(--border-strong)", borderRadius: 12, padding: 22, textAlign: "center", cursor: "pointer", background: "var(--panel)" }}>
                <div style={{ fontSize: 22, marginBottom: 4 }}>⬆</div><b>{ds ? dsName : "Drop a CSV here"}</b> {ds ? <span className="note">· {ds.nrows} rows · {ds.columns.length} columns — click to replace</span> : <span>or click to browse</span>}
                <div className="note" style={{ marginTop: 5 }}>First row = column names · classification or regression auto-detected</div>
              </div>
              {msg && <div className="note" style={{ marginTop: 8, color: "var(--crit)" }}>{msg}</div>}
              {ds && <>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16, alignItems: "start" }}>
                  <div style={pnl}>{secHead("var(--accent)", `Feature columns · ${feats.filter((f) => f !== target).length} selected`)}
                    <div style={{ maxHeight: 210, overflowY: "auto", padding: 8 }}>{ds.columns.filter((c) => c.name !== target).map((c) => <label key={c.name} style={{ display: "flex", gap: 8, alignItems: "center", padding: "6px 7px", borderRadius: 7, fontSize: 12.5, cursor: "pointer" }}><input type="checkbox" checked={feats.includes(c.name)} onChange={() => setFeats((f) => f.includes(c.name) ? f.filter((x) => x !== c.name) : [...f, c.name])} />{c.name}{typePill(c.type)}</label>)}</div>
                  </div>
                  <div style={pnl}>{secHead("#a855f7", "Target column")}
                    <div style={pnlBody}>
                      <select value={target} onChange={(e) => { setTarget(e.target.value); setFeats((f) => f.filter((x) => x !== e.target.value)); }} style={{ width: "100%" }}>{ds.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select>
                      <div style={{ marginTop: 12, padding: "9px 11px", borderRadius: 9, background: "var(--panel-2)", border: "1px solid var(--border)", fontSize: 11.5, color: "var(--muted)" }}>Detected task: <b style={{ color: "var(--text)" }}>{ds && target ? detectTask(ds, target) : "—"}</b> — auto from the target. <span style={{ color: "var(--faint)" }}>2 → binary, 3+ → multiclass, many numeric → regression.</span></div>
                      <div className="row" style={{ gap: 10, marginTop: 12 }}>
                        {statCard(ds.nrows.toLocaleString(), "rows")}
                        {statCard(ds.columns.length, "columns")}
                        {statCard(missTotal, "missing", missTotal ? "var(--orange)" : "var(--good)")}
                      </div>
                    </div>
                  </div>
                </div>
                <div style={{ fontSize: 10.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--faint)", margin: "16px 0 6px" }}>Preview · first 8 rows</div>
                {previewTable()}
                <div className="row" style={{ marginTop: 14, justifyContent: "flex-end" }}><button className="btn" onClick={buildFromCsv} disabled={!feats.filter((f) => f !== target).length}>Build & continue →</button></div>
              </>}
            </>}
          </div>
        </div>;
      })()}

      {step === "explore" && data && (() => {
        const isCsv = source === "csv" && !!ds;
        const rows = data.X.length; const H = 340;
        const featCount = isCsv && ds ? feats.filter((f) => f !== target).length : data.featNames.length;
        const inputs = data.featNames.length;
        const multiFeat = data.featNames.length > 1;
        const cnt = classCounts; // per-class counts (null for regression)
        // per-feature columns for the summary + correlation source
        const sumCols: { name: string; type: "num" | "cat"; values: (number | string | null)[] }[] = isCsv && ds
          ? ds.columns.filter((c) => feats.includes(c.name) && c.name !== target).map((c) => ({ name: c.name, type: c.type as "num" | "cat", values: c.values }))
          : data.featNames.map((n, j) => ({ name: n, type: "num" as const, values: data.X.map((r) => r[j]) as (number | string | null)[] }));
        // missing
        let missCells = 0; const missByCol: { name: string; n: number }[] = [];
        sumCols.forEach((c) => { const m = c.values.filter((v) => v == null).length; missCells += m; if (m) missByCol.push({ name: c.name, n: m }); });
        missByCol.sort((a, b) => b.n - a.n);
        const missPct = rows * featCount ? (missCells / (rows * featCount)) * 100 : 0;
        // correlation of model inputs with the target (real Pearson)
        const pearson = (a: number[], b: number[]) => { const n = a.length || 1; const ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n; let nu = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; nu += x * y; da += x * x; db += y * y; } return nu / (Math.sqrt(da * db) || 1); };
        const cols = data.featNames.map((_, j) => data.X.map((r) => r[j]));
        const ty = data.y.map(Number);
        const ftCorr = data.featNames.map((n, j) => ({ name: n, r: pearson(cols[j], ty) })).filter((c) => Number.isFinite(c.r));
        const topCorr = [...ftCorr].sort((a, b) => Math.abs(b.r) - Math.abs(a.r))[0];
        // stat helpers
        const stat = (a: number[]) => { const n = a.length || 1; const mean = a.reduce((x, y) => x + y, 0) / n; return { mean, std: Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / n), min: Math.min(...a), max: Math.max(...a) }; };
        const numBins = (a: number[], k = 9) => { if (!a.length) return new Array(k).fill(0); const mn = Math.min(...a), mx = Math.max(...a), sp = (mx - mn) || 1; const b = new Array(k).fill(0); a.forEach((v) => { let i = Math.floor((v - mn) / sp * k); if (i >= k) i = k - 1; if (i < 0) i = 0; b[i]++; }); return b; };
        const spark = (bins: number[], color: string) => { const mx = Math.max(...bins, 1); return <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 22 }}>{bins.map((v, i) => <span key={i} style={{ width: 5, height: Math.max(2, (v / mx) * 22), background: color, borderRadius: 1, opacity: 0.65 }} />)}</div>; };
        // insights
        const insights: { ic: string; txt: React.ReactNode }[] = [];
        if (cnt) { const mx = Math.max(...cnt), mn = Math.min(...cnt); const ratio = mn ? mx / mn : 1; insights.push({ ic: "⚖️", txt: ratio >= 1.5 ? <><b>Imbalanced</b> — {data.classes[cnt.indexOf(mn)]} is {ratio.toFixed(ratio < 10 ? 1 : 0)}× rarer than {data.classes[cnt.indexOf(mx)]}. Balance it in Preprocess.</> : <><b>Balanced</b> — classes within {ratio.toFixed(1)}× of each other.</> }); }
        else { const s = stat(ty); insights.push({ ic: "🎯", txt: <>Target spans <b>{s.min.toFixed(1)}–{s.max.toFixed(1)}</b>, mean {s.mean.toFixed(1)}.</> }); }
        if (topCorr) insights.push({ ic: "🔗", txt: <><b>{topCorr.name}</b> {cnt ? "associates with" : "↔"} target r={topCorr.r.toFixed(2)} — strongest linear signal.</> });
        insights.push({ ic: "🩹", txt: missCells ? <><b>{missCells} missing</b> cell{missCells === 1 ? "" : "s"}{missByCol.length ? <> — mostly {missByCol.slice(0, 2).map((m) => m.name).join(", ")}</> : null}. Imputed in Preprocess.</> : <><b>No missing values</b> — data is complete.</> });
        // view chips (need ≥2 features for scatter/pca/corr)
        const VIEWS: [typeof exMode, string][] = multiFeat ? [["scatter", "Scatter"], ["pca", "PCA 2-D"], ["dist", "Distribution"], ["corr", "Correlation"]] : [["dist", "Distribution"]];
        const curMode = VIEWS.some((v) => v[0] === exMode) ? exMode : "dist";
        const viewChips = <div className="chips">{VIEWS.map(([m, l]) => <button key={m} className={`chip ${curMode === m ? "on" : ""}`} onClick={() => setExMode(m)}>{l}</button>)}</div>;
        // correlation heatmap fig
        const corr = curMode === "corr" ? cols.map((ci) => cols.map((cj) => pearson(ci, cj))) : null;
        const corrScale: [number, string][] = [[0, "#e5484d"], [0.5, "#0d1117"], [1, "#5b7cff"]];
        return <div className="card"><div className="card-h"><span className="t">Explore — {data.source}</span><span className="mono r">{rows} rows · {featCount} features · {data.task}</span></div>
          <div className="card-b">
            {data.series && <div style={{ ...pnl, marginBottom: 16 }}>{secHead("#3ecf7f", "Series over time", <span className="note" style={{ fontSize: 10 }}>{data.win} lags → next value</span>)}<div style={pnlBody}><Plot data={[{ type: "scatter", mode: "lines", x: data.series.t, y: data.series.v, line: { color: "#3ecf7f", width: 1.6 } }] as never} layout={lay("", "time step", "value", { height: 230, showlegend: false }) as never} style={{ height: 230, width: "100%" }} /><div className="note" style={{ marginTop: 4 }}>The raw signal. To remove the trend, the lab trains on <b>first differences</b> (step-to-step changes): each row is a window of the previous {data.win} changes, and the model predicts the next change — then adds it back onto the last value to forecast the level.</div></div></div>}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10, marginBottom: 16 }}>
              {statCard(rows.toLocaleString(), "rows")}
              {statCard(inputs !== featCount ? <>{featCount} <span style={{ fontSize: 12, color: "var(--faint)" }}>→ {inputs} in</span></> : featCount, "features")}
              {statCard(data.task === "binary" ? "Binary" : data.task === "multiclass" ? "Multiclass" : "Regression", "task", "var(--purple)")}
              {cnt ? statCard(data.classes.length, `classes · ${target || "target"}`) : statCard(`${stat(ty).min.toFixed(0)}–${stat(ty).max.toFixed(0)}`, `target · ${target || "y"}`)}
              {statCard(`${missPct.toFixed(missPct && missPct < 0.1 ? 2 : 1)}%`, "missing cells", missCells ? "var(--orange)" : "var(--good)")}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 16, alignItems: "stretch" }}>
              <div style={pnl}>
                {secHead("var(--accent)", "Visualize", viewChips)}
                <div style={pnlBody}>
                  {(curMode === "scatter" || curMode === "dist") && <div className="row" style={{ gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
                    {curMode === "scatter" && <><label className="note">X</label><select value={exFx} onChange={(e) => setExFx(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select><label className="note">Y</label><select value={exFy} onChange={(e) => setExFy(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select></>}
                    {curMode === "dist" && <><label className="note">feature</label><select value={exFx} onChange={(e) => setExFx(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select></>}
                  </div>}
                  {curMode === "corr" && corr
                    ? <Plot data={[{ z: corr, x: data.featNames, y: data.featNames, type: "heatmap", zmin: -1, zmax: 1, colorscale: corrScale, colorbar: { title: { text: "r" }, thickness: 12, len: 0.9 } }] as never} layout={lay("feature correlation (Pearson)", "", "", { height: H, xaxis: { tickangle: -40, automargin: true }, yaxis: { automargin: true } }) as never} style={{ height: H, width: "100%" }} />
                    : exploreFig && <><Plot data={exploreFig.data as never} layout={lay(exploreFig.title, exploreFig.xl, exploreFig.yl, { showlegend: (exploreFig as { legend?: boolean }).legend !== false, legend: { orientation: "h", y: -0.2 }, height: H, barmode: (exploreFig as { barmode?: string }).barmode }) as never} style={{ height: H, width: "100%" }} /></>}
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div style={pnl}>
                  {secHead("#3ecf7f", cnt ? "Class balance" : "Target distribution", <span className="note" style={{ fontSize: 10 }}>{target || (cnt ? "class" : "y")}</span>)}
                  <div style={pnlBody}>
                    {cnt ? (() => { const mx = Math.max(...cnt, 1); return <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>{cnt.map((c, i) => <div key={i} className="row" style={{ gap: 10, alignItems: "center", fontSize: 11.5 }}>
                      <span style={{ flex: "0 0 64px", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{data.classes[i] ?? `class ${i}`}</span>
                      <div style={{ flex: 1, height: 16, background: "var(--panel-2)", borderRadius: 5, overflow: "hidden" }}><div style={{ width: `${(c / mx) * 100}%`, height: "100%", background: PAL[i % PAL.length], borderRadius: 5 }} /></div>
                      <span className="mono" style={{ flex: "0 0 38px", textAlign: "right", color: "var(--faint)" }}>{c}</span>
                    </div>)}</div>; })()
                    : <Plot data={[{ type: "histogram", x: ty, marker: { color: "#3ecf7f" }, opacity: 0.85 }] as never} layout={lay("", target || "target", "count", { height: 150, showlegend: false }) as never} style={{ height: 150, width: "100%" }} />}
                  </div>
                </div>
                <div style={{ ...pnl, flex: 1 }}>
                  {secHead("var(--orange)", "At a glance")}
                  <div style={{ ...pnlBody, paddingTop: 4 }}>{insights.map((s, i) => <div key={i} className="row" style={{ gap: 9, alignItems: "start", fontSize: 11.5, padding: "9px 0", borderTop: i ? "1px solid var(--border)" : "none", lineHeight: "16px" }}><span style={{ flex: "0 0 auto" }}>{s.ic}</span><span style={{ color: "var(--muted)" }}>{s.txt}</span></div>)}</div>
                </div>
              </div>
            </div>

            <div style={{ ...pnl, marginTop: 16 }}>
              {secHead("var(--purple)", "Feature summary", <span className="note" style={{ fontSize: 10 }}>{sumCols.length} column{sumCols.length === 1 ? "" : "s"}</span>)}
              <div style={{ overflowX: "auto" }}>
                <table className="dtable" style={{ width: "100%" }}><tbody>
                  <tr><th style={{ textAlign: "left" }}>feature</th><th>type</th><th>mean / mode</th><th>std</th><th>range</th><th>missing</th><th style={{ textAlign: "left" }}>distribution</th></tr>
                  {sumCols.map((c, i) => { const miss = c.values.filter((v) => v == null).length;
                    if (c.type === "num") { const nums = c.values.filter((v): v is number => v != null).map(Number); const s = stat(nums);
                      return <tr key={i}><td style={{ color: "var(--text)", fontWeight: 500 }}>{c.name}</td><td><span style={{ fontSize: 9.5, padding: "1px 7px", borderRadius: 20, color: "var(--accent)", background: "rgba(91,124,255,.12)", border: "1px solid var(--border)" }}>num</span></td><td>{s.mean.toFixed(2)}</td><td>{s.std.toFixed(2)}</td><td>{s.min.toFixed(1)} – {s.max.toFixed(1)}</td><td style={{ color: miss ? "var(--orange)" : "var(--good)" }}>{miss}</td><td>{spark(numBins(nums), "#5b7cff")}</td></tr>;
                    }
                    const strs = c.values.filter((v): v is string => v != null).map(String); const cm = new Map<string, number>(); strs.forEach((v) => cm.set(v, (cm.get(v) || 0) + 1)); const mode = [...cm.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";
                    return <tr key={i}><td style={{ color: "var(--text)", fontWeight: 500 }}>{c.name}</td><td><span style={{ fontSize: 9.5, padding: "1px 7px", borderRadius: 20, color: "var(--purple)", background: "rgba(168,85,247,.12)", border: "1px solid var(--border)" }}>cat</span></td><td>{mode}</td><td style={{ color: "var(--faint)" }}>—</td><td>{cm.size} cats</td><td style={{ color: miss ? "var(--orange)" : "var(--good)" }}>{miss}</td><td>{spark([...cm.values()].slice(0, 9), "#a855f7")}</td></tr>;
                  })}
                </tbody></table>
              </div>
            </div>

            <div className="stepnav" style={{ marginTop: 16 }}><button className="btn ghost" onClick={() => setStep("data")}>← Back</button><button className="btn" onClick={() => setStep("prep")}>Next: Preprocess →</button></div>
          </div>
        </div>;
      })()}

      {step === "prep" && data && (() => {
        const nf = data.featNames.length; const H = 210;
        const isCsv = source === "csv" && !!ds;
        const scViz = fitScaler(data.X, scaleMethod);
        const scaleName = ({ standard: "Standard z-score", minmax: "Min-Max [0,1]", robust: "Robust (median/IQR)", none: "None (raw)" } as const)[scaleMethod];
        const nTest = Math.max(1, Math.round(data.X.length * testFrac)); const nTrain = data.X.length - nTest;
        // real facts about what the auto-pipeline did to the loaded data
        let imputed = 0; const catInfo: { name: string; n: number }[] = [];
        if (isCsv && ds) { const fcols = feats.filter((f) => f !== target); ds.columns.filter((c) => fcols.includes(c.name)).forEach((c) => { imputed += c.values.filter((v) => v == null).length; if (c.type === "cat") catInfo.push({ name: c.name, n: data.featNames.filter((fn) => fn === c.name || fn.startsWith(c.name + "=") || fn.startsWith(c.name + "_b")).length }); }); }
        const rawCols = isCsv ? feats.filter((f) => f !== target).length : nf;
        const hasCat = catInfo.length > 0;
        const cnt = data.task !== "regression" ? (() => { const c = new Array(K).fill(0); data.y.forEach((v) => c[v]++); return c; })() : null;
        const flowSteps: { l: string; s: string; on: boolean }[] = [
          { l: "Raw", s: `${rawCols} col${rawCols === 1 ? "" : "s"}`, on: true },
          { l: "Impute", s: imputed ? `${imputeMethod} · ${imputed}` : "none", on: imputed > 0 },
          { l: "Encode", s: hasCat ? encMethod : "none", on: hasCat },
          { l: "Scale", s: scaleMethod === "none" ? "off" : scaleName, on: scaleMethod !== "none" },
          { l: "Split", s: `${nTrain}/${nTest}`, on: true },
          { l: "Network", s: `${nf} in`, on: true },
        ];
        const summary: { ic: string; txt: React.ReactNode }[] = [];
        summary.push({ ic: "🧮", txt: <><b>{nf}</b> model input{nf === 1 ? "" : "s"} from <b>{rawCols}</b> {isCsv ? "selected" : "raw"} column{rawCols === 1 ? "" : "s"}{data.task !== "regression" ? <> · target <b>{data.classes.length}</b> classes</> : <> · numeric target</>}</> });
        if (isCsv) {
          summary.push({ ic: "🩹", txt: imputed ? <>Imputed <b>{imputed}</b> missing value{imputed === 1 ? "" : "s"} with <b>{imputeMethod}</b></> : <>No missing values found — nothing imputed</> });
          if (hasCat) catInfo.forEach((ci) => summary.push({ ic: "🔤", txt: <><b>{encMethod}</b> encoded <b>{ci.name}</b> → <b>{ci.n}</b> column{ci.n === 1 ? "" : "s"}</> }));
          else summary.push({ ic: "🔤", txt: <>No categorical columns — no encoding needed</> });
        } else summary.push({ ic: "🔢", txt: <>Built-in <b>{data.source}</b> — all numeric, no missing values or categoricals</> });
        summary.push({ ic: "📏", txt: <>Scaling: <b>{scaleName}</b>{scaleMethod === "none" ? " — features keep their raw magnitudes" : ", fit on the training split only"}</> });
        if (data.chronological) summary.push({ ic: "⏱️", txt: <>Split is <b>chronological</b> — trained on the earliest rows, tested on the latest. No shuffling, so the future never leaks into training.</> });
        if (cnt) summary.push({ ic: "⚖️", txt: balanceClasses ? <>Class weights <b>on</b> — rarer classes are upweighted in the loss</> : <>Class weights off — classes train in their natural proportion</> });
        const boxRaw = data.featNames.map((f, j) => ({ type: "box", name: f, y: data.X.map((r) => r[j]), marker: { color: "#5b7cff" }, boxpoints: false }));
        const boxScaled = data.featNames.map((f, j) => ({ type: "box", name: f, y: data.X.map((r) => (r[j] - scViz.mean[j]) / scViz.std[j]), marker: { color: "#3ecf7f" }, boxpoints: false }));
        // ── per-column inspect (missing → impute, categorical → encode, numeric → scale) ──
        const inspectCols: { name: string; type: "num" | "cat"; values: (number | string | null)[] }[] = isCsv && ds
          ? ds.columns.filter((c) => feats.includes(c.name) && c.name !== target).map((c) => ({ name: c.name, type: c.type as "num" | "cat", values: c.values }))
          : data.featNames.map((n, j) => ({ name: n, type: "num" as const, values: data.X.map((r) => r[j]) as (number | string | null)[] }));
        const iIdx = Math.min(exFx, Math.max(0, inspectCols.length - 1)); const iSel = inspectCols[iIdx];
        const iMissing = iSel ? iSel.values.filter((v) => v == null).length : 0;
        const numPresent = iSel && iSel.type === "num" ? iSel.values.filter((v): v is number => v != null).map(Number) : [];
        const fillVal = (() => { if (!numPresent.length) return 0; if (imputeMethod === "Median") { const s = [...numPresent].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; } if (imputeMethod === "Constant") return 0; if (imputeMethod === "Most frequent") { const m = new Map<number, number>(); numPresent.forEach((v) => m.set(v, (m.get(v) || 0) + 1)); return [...m.entries()].sort((a, b) => b[1] - a[1])[0][0]; } return numPresent.reduce((a, b) => a + b, 0) / numPresent.length; })();
        const numJ = iSel && iSel.type === "num" ? data.featNames.indexOf(iSel.name) : -1;
        const numScaled = numJ >= 0 ? data.X.map((r) => (r[numJ] - scViz.mean[numJ]) / scViz.std[numJ]) : [];
        // categorical: replay impute+encode for the mapping table
        const catData = iSel && iSel.type === "cat" ? (() => {
          const raw0 = iSel.values.map((v) => (v == null ? null : String(v)));
          const present = raw0.filter((v): v is string => v != null);
          const modeC = (() => { const m = new Map<string, number>(); present.forEach((v) => m.set(v, (m.get(v) || 0) + 1)); return [...m.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? ""; })();
          const filled = raw0.map((v) => (v == null ? (imputeMethod === "Constant" ? "missing" : modeC) : v)) as string[];
          const cm = new Map<string, number>(); filled.forEach((v) => cm.set(v, (cm.get(v) || 0) + 1));
          const cats = [...cm.keys()]; const bits = Math.max(1, Math.ceil(Math.log2(cats.length || 1)));
          const map = cats.map((cat, i) => { let enc: string; if (encMethod === "Ordinal") enc = String(i); else if (encMethod === "Frequency") enc = (cm.get(cat)! / filled.length).toFixed(3); else if (encMethod === "Binary") { let s = ""; for (let b = bits - 1; b >= 0; b--) s += (i >> b) & 1; enc = s; } else enc = `${iSel.name}=${cat}`; return { cat, count: cm.get(cat)!, enc }; });
          return { cats, counts: cats.map((c) => cm.get(c)!), map };
        })() : null;
        const SCALES: [ScaleMethod, string, string][] = [["standard", "Standard", "z=(x−μ)/σ"], ["minmax", "Min-Max", "→[0,1]"], ["robust", "Robust", "median/IQR"], ["none", "None", "raw"]];
        const cwShown = cnt ? (balanceClasses ? classWeights(data.y, K) : cnt.map(() => 1)) : null;
        const wBadge = (w: number) => { const up = w > 1.02; return <span className="mono" style={{ fontSize: 10.5, fontWeight: 600, padding: "1px 7px", borderRadius: 20, color: up ? "#3ecf7f" : "var(--faint)", background: up ? "rgba(62,207,127,.14)" : "var(--panel-2)", border: `1px solid ${up ? "rgba(62,207,127,.32)" : "var(--border)"}` }}>×{w.toFixed(2)}</span>; };
        const minC = cnt ? Math.min(...cnt) : 0, maxC = cnt ? Math.max(...cnt) : 0; const ratio = minC ? maxC / minC : 1;
        const insight = cnt ? (ratio < 1.25
          ? <>Classes are fairly even ({ratio.toFixed(1)}× spread) — balancing has little effect here.</>
          : <>Rarest <b>{data.classes[cnt.indexOf(minC)]}</b> is <b>{ratio.toFixed(1)}×</b> under the largest class. {balanceClasses ? <>Weights rebalance the loss so it isn&apos;t ignored.</> : <>Enable <b>Balance classes</b> to stop the model favouring the majority.</>}</>) : null;
        return <div className="card"><div className="card-h"><span className="t">Preprocess — turn raw data into a training matrix</span></div>
          <div className="card-b">
            <div className="teach-note"><span className="ic">🛠️</span><span>Every choice here <b>actually rebuilds the matrix the network trains on</b>: encoding sets the input width, scaling rescales each feature (fit on the train split only, so no leakage), the split holds out a test set, and class weights reshape the loss. Change any of them and the model must retrain.</span></div>
            <div className="row" style={{ gap: 6, alignItems: "center", flexWrap: "wrap", margin: "14px 0 4px" }}>{flowSteps.map((f, i) => <span key={f.l} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ padding: "5px 11px", borderRadius: 8, border: `1px solid ${f.on ? "var(--border-strong)" : "var(--border)"}`, background: f.on ? "var(--surface)" : "var(--panel)", opacity: f.on ? 1 : 0.5, minWidth: 62, textAlign: "center" }}><span style={{ fontWeight: 600, fontSize: 12 }}>{f.l}</span><br /><span className="note" style={{ fontFamily: "var(--mono)", fontSize: 10 }}>{f.s}</span></span>{i < flowSteps.length - 1 && <span style={{ color: "var(--faint)" }}>→</span>}</span>)}</div>

            <div className="split col-2e" style={{ gap: 16, alignItems: "stretch", marginTop: 14 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={pnl}>
                  {secHead("var(--accent)", "What happened to your data", <span className="mono note" style={{ fontSize: 10.5 }}>{nf} inputs</span>)}
                  <div style={{ ...pnlBody, display: "flex", flexDirection: "column", gap: 9 }}>{summary.map((s, i) => <div key={i} className="row" style={{ gap: 9, alignItems: "start", fontSize: 12.5 }}><span style={{ flex: "0 0 auto", fontSize: 13, lineHeight: "18px" }}>{s.ic}</span><span style={{ color: "var(--muted)", lineHeight: "18px" }}>{s.txt}</span></div>)}</div>
                </div>
                <div style={{ ...pnl, flex: 1 }}>
                  {secHead("#a855f7", "Options — rebuild the matrix", <span className="note" style={{ fontSize: 10 }}>retrains on change</span>)}
                  <div style={pnlBody}>
                    <label className="fld">Missing values — imputation {isCsv ? (imputed ? <span className="note">— {imputed} to fill</span> : <span className="note">— none in this data</span>) : <span className="note">— none in built-in data</span>}</label>
                    <select value={imputeMethod} onChange={(e) => setImputeMethod(e.target.value as typeof imputeMethod)} disabled={!isCsv || imputed === 0} style={{ width: "100%", marginBottom: 14, opacity: isCsv && imputed ? 1 : 0.5 }}>
                      {["Mean", "Median", "Most frequent", "Constant"].map((m) => <option key={m} value={m}>{m}{m === "Constant" ? " (0 / \"missing\")" : m === "Mean" || m === "Median" ? " (numeric)" : ""}</option>)}
                    </select>
                    <label className="fld">Categorical encoding {!hasCat && <span className="note">— none in this data</span>}</label>
                    <select value={encMethod} onChange={(e) => setEncMethod(e.target.value as typeof encMethod)} disabled={!hasCat} style={{ width: "100%", marginBottom: 14, opacity: hasCat ? 1 : 0.5 }}>
                      {["One-Hot", "Ordinal", "Frequency", "Binary"].map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                    <label className="fld">Feature scaling</label>
                    <div className="chips" style={{ marginBottom: 14 }}>{SCALES.map(([m, l, sub]) => <button key={m} className={`chip ${scaleMethod === m ? "on" : ""}`} onClick={() => setScaleMethod(m)} title={sub}>{l}</button>)}</div>
                    <label className="fld">Test split — <b style={{ color: "var(--text)" }}>{Math.round(testFrac * 100)}%</b> held out ({nTest} rows)</label>
                    <input type="range" min={0.1} max={0.4} step={0.05} value={testFrac} onChange={(e) => setTestFrac(+e.target.value)} style={{ width: "100%", marginBottom: 14 }} />
                    <label className="row" style={{ gap: 9, alignItems: "center", cursor: data.task === "regression" ? "not-allowed" : "pointer", opacity: data.task === "regression" ? 0.5 : 1, padding: "9px 11px", border: "1px solid var(--border)", borderRadius: 9, background: balanceClasses ? "rgba(62,207,127,.08)" : "var(--panel-2)" }}>
                      <input type="checkbox" checked={balanceClasses} disabled={data.task === "regression"} onChange={(e) => setBalanceClasses(e.target.checked)} />
                      <span style={{ fontSize: 12.5 }}>Balance classes <span className="note">— upweight rare classes in the loss{data.task === "regression" ? " (classification only)" : ""}</span></span>
                    </label>
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={pnl}>
                  {secHead("#5b7cff", "Feature scales", <span className="note" style={{ fontSize: 10 }}>raw → {scaleName}{scaleMethod === "none" && " (off)"}</span>)}
                  <div style={{ ...pnlBody, paddingTop: 6 }}>
                    <div className="split col-2e" style={{ gap: 10 }}>
                      <Plot data={boxRaw as never} layout={lay("raw ranges", "", "value", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
                      <Plot data={boxScaled as never} layout={lay(scaleName, "", "scaled", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
                    </div>
                  </div>
                </div>
                <div style={{ ...pnl, flex: 1 }}>
                  {secHead("#3ecf7f", cnt ? "Split & class balance" : "Train / test split", <span className="mono note" style={{ fontSize: 10.5 }}>{Math.round((1 - testFrac) * 100)} / {Math.round(testFrac * 100)}</span>)}
                  <div style={pnlBody}>
                    <div style={{ display: "flex", height: 30, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }}>
                      <div style={{ width: `${(1 - testFrac) * 100}%`, background: "var(--accent)", color: "#fff", fontSize: 11, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", minWidth: 0 }}>train · {nTrain}</div>
                      <div style={{ width: `${testFrac * 100}%`, background: "var(--panel-2)", color: "var(--muted)", fontSize: 11, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", minWidth: 0 }}>test · {nTest}</div>
                    </div>
                    {data.chronological && <div className="note" style={{ marginTop: 8, fontSize: 11 }}>⏱️ Chronological — earliest {nTrain} rows train, latest {nTest} forecast. Past → future, never shuffled.</div>}
                    {cnt && (() => { const mx = Math.max(...cnt, 1); return <div style={{ marginTop: 16 }}>
                      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline", marginBottom: 9 }}><span className="fld" style={{ margin: 0 }}>Class balance</span>{cwShown && <span className="note" style={{ fontSize: 10 }}>{balanceClasses ? "training weight" : "count"}</span>}</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>{cnt.map((c, i) => <div key={i} className="row" style={{ gap: 10, alignItems: "center", fontSize: 11.5 }}>
                        <span style={{ flex: "0 0 72px", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{data.classes[i] ?? `class ${i}`}</span>
                        <div style={{ flex: 1, height: 16, background: "var(--panel-2)", borderRadius: 5, overflow: "hidden" }}><div style={{ width: `${(c / mx) * 100}%`, height: "100%", background: PAL[i % PAL.length], borderRadius: 5 }} /></div>
                        <span className="mono" style={{ flex: "0 0 40px", textAlign: "right", color: "var(--faint)" }}>{c}</span>
                        {cwShown && <span style={{ flex: "0 0 54px", textAlign: "right" }}>{wBadge(cwShown[i])}</span>}
                      </div>)}</div>
                      <div className="note" style={{ marginTop: 14, padding: "9px 11px", borderRadius: 9, background: "var(--panel-2)", border: "1px solid var(--border)", display: "flex", gap: 8, alignItems: "start", lineHeight: "17px" }}><span style={{ flex: "0 0 auto" }}>💡</span><span>{insight}</span></div>
                    </div>; })()}
                  </div>
                </div>
              </div>
            </div>

            <div className="row" style={{ gap: 10, alignItems: "center", margin: "18px 0 10px", flexWrap: "wrap" }}>
              <label className="note" style={{ margin: 0 }}>Inspect a column through the pipeline:</label>
              <select value={iIdx} onChange={(e) => setExFx(+e.target.value)}>{inspectCols.map((c, i) => <option key={i} value={i}>{c.name}{c.type === "cat" ? "  (categorical)" : ""}</option>)}</select>
              {iSel && <span className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>{iSel.type === "cat" ? `impute(${imputeMethod})  →  ${encMethod}  →  ${catData?.map.length ?? 0} col${(catData?.map.length ?? 0) === 1 ? "" : "s"}` : `impute(${imputeMethod}${iMissing ? ` = ${fillVal.toFixed(2)}` : ""})  →  ${scaleName}`}</span>}
            </div>
            <div className="split col-2e" style={{ gap: 12, alignItems: "start" }}>
              <div>
                <div className="fld">Before — raw{iMissing ? <span className="note"> · {iMissing} missing</span> : null}</div>
                {iSel && iSel.type === "cat"
                  ? <Plot data={[{ type: "bar", x: catData!.cats, y: catData!.counts, marker: { color: "#5b7cff" } }] as never} layout={lay(`${iSel.name} — categories`, "", "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
                  : <Plot data={[{ type: "histogram", x: numPresent, marker: { color: "#5b7cff" }, opacity: 0.85 }] as never} layout={lay(`${iSel?.name ?? ""} — raw values`, iSel?.name ?? "", "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />}
              </div>
              <div>
                <div className="fld">After — {iSel && iSel.type === "cat" ? `${encMethod} encoded columns` : "model input (imputed + scaled)"}</div>
                {iSel && iSel.type === "cat"
                  ? <div style={{ height: H, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10 }}><table className="dtable" style={{ width: "100%" }}><tbody>
                      <tr><th style={{ textAlign: "left" }}>category</th><th>rows</th><th style={{ textAlign: "left" }}>encoded</th></tr>
                      {catData!.map.map((m, i) => <tr key={i}><td>{m.cat}</td><td style={{ textAlign: "center", color: "var(--faint)" }}>{m.count}</td><td className="mono" style={{ color: "#3ecf7f" }}>{m.enc}</td></tr>)}
                    </tbody></table></div>
                  : <Plot data={[{ type: "histogram", x: numScaled, marker: { color: "#3ecf7f" }, opacity: 0.85 }] as never} layout={lay(`${iSel?.name ?? ""} — ${scaleName}`, "scaled value", "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />}
              </div>
            </div>
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("explore")}>← Back</button><button className="btn" onClick={() => setStep("arch")}>Next: Architecture →</button></div>
          </div>
        </div>;
      })()}

      {step === "arch" && data && (() => {
        const inDim = data.X[0].length; const sizes = [inDim, ...hidden, outDim];
        const params = sizes.slice(0, -1).reduce((a, s, l) => a + s * sizes[l + 1] + sizes[l + 1], 0);
        const neurons = sizes.reduce((a, b) => a + b, 0);
        const outAct = data.task === "binary" ? "sigmoid" : data.task === "multiclass" ? "softmax" : "linear";
        const outLoss = data.task === "binary" ? "binary cross-entropy" : data.task === "multiclass" ? "cross-entropy" : "MSE";
        const blk = (extra: React.CSSProperties): React.CSSProperties => ({ border: "1px solid var(--border)", borderRadius: 10, padding: "11px 14px", display: "flex", alignItems: "center", gap: 10, background: "var(--panel)", ...extra });
        const conn = (label: string, key: string) => <div key={key} style={{ height: 22, position: "relative", margin: "0 auto", width: 2, background: "var(--border-strong)" }}><span style={{ position: "absolute", left: 8, top: 2, fontSize: 9, fontFamily: "var(--mono)", color: "var(--muted)", whiteSpace: "nowrap" }}>{label}</span></div>;
        const dot = (c: string) => <span style={{ width: 9, height: 9, borderRadius: "50%", background: c, flex: "0 0 auto" }} />;
        const lab = (t: string) => <div className="note" style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em" }}>{t}</div>;
        const stack: React.ReactNode[] = [<div key="in" style={blk({})}>{dot("#5b7cff")}<div>{lab("input")}<div style={{ fontWeight: 500, fontSize: 13 }}>{inDim} feature{inDim === 1 ? "" : "s"} → {inDim} neuron{inDim === 1 ? "" : "s"}</div></div></div>];
        hidden.forEach((h, i) => {
          stack.push(conn(act, `c${i}`));
          stack.push(<div key={`d${i}`} style={blk({ background: "var(--surface)", borderColor: "var(--border-strong)" })}>{dot("#a855f7")}<div>{lab(`dense ${i + 1}`)}<div style={{ fontWeight: 500, fontSize: 13 }}>hidden</div></div>
            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}><button className="btn ghost sm" onClick={() => setHidden((hs) => hs.map((x, j) => (j === i ? Math.max(1, x - 1) : x)))}>−</button><b className="mono" style={{ minWidth: 64, textAlign: "center", fontSize: 12 }}>{h} neuron{h === 1 ? "" : "s"}</b><button className="btn ghost sm" onClick={() => setHidden((hs) => hs.map((x, j) => (j === i ? Math.min(16, x + 1) : x)))}>+</button></div>
            <button className="btn ghost sm" onClick={() => setHidden((hs) => hs.filter((_, j) => j !== i))} disabled={hidden.length <= 1} style={{ marginLeft: 4 }}>×</button></div>);
        });
        stack.push(conn(outAct, "cout"));
        stack.push(<div key="out" style={blk({})}>{dot("#f59e0b")}<div>{lab("output")}<div style={{ fontWeight: 500, fontSize: 13 }}>{outDim} neuron{outDim === 1 ? "" : "s"} · {outAct} · {outLoss}</div></div></div>);
        const scard = (v: string, k: string, color?: string) => <div style={{ background: "var(--panel)", borderRadius: 9, padding: "10px 12px" }}><div style={{ fontSize: 19, fontWeight: 600, color: color || "var(--text)" }}>{v}</div><div className="note" style={{ marginTop: 3, textTransform: "uppercase" }}>{k}</div></div>;
        return <div className="card"><div className="card-h"><span className="t">Design the network</span></div>
          <div className="card-b">
            <div className="row" style={{ gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
              <div style={{ flex: 1, minWidth: 160, ...blk({ display: "block" }), borderColor: "var(--accent)" }}><div style={{ fontWeight: 500 }}>Dense <span className="note" style={{ background: "var(--good)", color: "#052", borderRadius: 20, padding: "1px 7px", marginLeft: 4 }}>used</span></div><div className="note" style={{ marginTop: 4 }}>Fully-connected — every input reaches every neuron. Right for tabular data.</div></div>
              <div style={{ flex: 1, minWidth: 160, ...blk({ display: "block" }), opacity: 0.6 }}><div style={{ fontWeight: 500 }}>Conv <span className="note">images · GPU</span></div><div className="note" style={{ marginTop: 4 }}>Slides filters over an image to detect local patterns. Out of scope here.</div></div>
              <div style={{ flex: 1, minWidth: 160, ...blk({ display: "block" }), opacity: 0.6 }}><div style={{ fontWeight: 500 }}>Pooling <span className="note">images · GPU</span></div><div className="note" style={{ marginTop: 4 }}>Downsamples feature maps to shrink them. Pairs with Conv for images.</div></div>
            </div>
            <div className="split col-2e" style={{ gap: 18, alignItems: "start" }}>
              <div>
                <label className="fld">Layer stack — read top to bottom</label>
                <div style={{ display: "flex", flexDirection: "column" }}>{stack}</div>
                <button className="btn ghost sm" onClick={() => setHidden((hs) => [...hs, 6])} disabled={hidden.length >= 4} style={{ marginTop: 10, width: "100%", borderStyle: "dashed" }}>+ add hidden Dense layer</button>
                <label className="fld" style={{ marginTop: 14 }}>Activation (non-linearity between layers)</label>
                <div className="chips">{["tanh", "relu", "sigmoid"].map((a) => <button key={a} className={`chip ${act === a ? "on" : ""}`} onClick={() => setAct(a)}>{a}</button>)}</div>
              </div>
              <div>
                <label className="fld">Network diagram — each column a layer, each dot a neuron</label>
                <div style={panelSt}>{netDiagram()}<div className="note" style={{ marginTop: 6 }}>{netRef.current ? "Edge colour = learned weight (green +, red −), thickness = magnitude." : "Edges are connections; train to see the learned weights."}</div></div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginTop: 12 }}>{scard(String(hidden.length + 1), "layers", "var(--accent)")}{scard(String(neurons), "neurons")}{scard(params.toLocaleString(), "parameters")}</div>
              </div>
            </div>
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("prep")}>← Back</button><button className="btn" onClick={() => setStep("train")}>Next: Train →</button></div>
          </div>
        </div>;
      })()}

      {step === "train" && data && (() => {
        const metricName = data.task === "regression" ? "R²" : "accuracy";
        const pct = Math.min(100, epochsTarget ? (epoch / epochsTarget) * 100 : 0);
        const stat = (v: string, k: string, color?: string) => <div style={{ background: "linear-gradient(160deg, var(--panel), var(--surface))", border: "1px solid var(--border)", borderRadius: 12, padding: "12px 14px" }}><div style={{ fontSize: 22, fontWeight: 600, lineHeight: 1, color: color || "var(--text)" }}>{v}</div><div className="note" style={{ marginTop: 5, textTransform: "uppercase", letterSpacing: ".04em" }}>{k}</div></div>;
        const c = history.length ? curveFig() : null;
        return <div className="card"><div className="card-h"><span className="t">Train — fit the network on your data</span><span className="mono r">{running ? "training…" : epoch ? "done" : "not started"}</span></div>
          <div className="card-b">
            <div className="teach-note" style={{ marginBottom: 12 }}><span className="ic">🎛️</span><span><b>Learning rate</b> = step size per update. <b>Optimizer</b> = how steps are computed (Adam adapts per-weight, usually fastest). <b>L2</b> = weight penalty that fights overfitting. <b>Batch size</b> = rows averaged per update. An <b>epoch</b> = one full pass over the training data.</span></div>
            {/* control panel */}
            <div style={panelSt}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 16 }}>
                <div className="knob" style={{ margin: 0 }}><div className="kr"><span>Learning rate (η)</span><b>{lr}</b></div><input type="range" min={0.001} max={0.2} step={0.001} value={lr} onChange={(e) => setLr(+e.target.value)} /></div>
                <div><label className="note" style={{ display: "block", marginBottom: 5 }}>Optimizer</label><select value={optimizer} onChange={(e) => setOptimizer(e.target.value as Optimizer)} style={{ width: "100%" }}><option value="adam">Adam</option><option value="momentum">SGD + momentum</option><option value="sgd">SGD</option></select></div>
                <div className="knob" style={{ margin: 0 }}><div className="kr"><span>L2 (weight decay)</span><b>{l2}</b></div><input type="range" min={0} max={0.01} step={0.0005} value={l2} onChange={(e) => setL2(+e.target.value)} /></div>
                <div className="knob" style={{ margin: 0 }}><div className="kr"><span>Batch size (rows)</span><b>{batchSize}</b></div><input type="range" min={1} max={64} step={1} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} /></div>
                <div className="knob" style={{ margin: 0 }}><div className="kr"><span>Epochs</span><b>{epochsTarget}</b></div><input type="range" min={50} max={1000} step={25} value={epochsTarget} onChange={(e) => setEpochsTarget(+e.target.value)} disabled={running} /></div>
              </div>
              <div className="row" style={{ gap: 12, alignItems: "center", marginTop: 14 }}>
                <button className="btn" onClick={running ? stopTrain : startTrain}>{running ? "⏸ Pause" : epoch ? "↻ Restart" : "▶ Train"}</button>
                {running && <button className="btn ghost" onClick={finishNow}>⏭ Finish now</button>}
                <div style={{ flex: 1, minWidth: 120, height: 8, background: "var(--panel-2)", borderRadius: 4, overflow: "hidden" }}><div style={{ height: "100%", width: `${pct}%`, background: "linear-gradient(90deg, var(--accent), var(--good))", borderRadius: 4, transition: "width .2s" }} /></div>
                <span className="note mono" style={{ whiteSpace: "nowrap" }}>epoch {epoch} / {epochsTarget}</span>
              </div>
            </div>
            {/* live stat cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, margin: "14px 0" }}>
              {stat(String(epoch), "epoch", "var(--accent)")}
              {stat(best ? best.acc.toFixed(3) : "—", `train ${metricName}`, "var(--good)")}
              {stat(best ? best.vacc.toFixed(3) : "—", `validation ${metricName}`, "#a855f7")}
              {stat(best ? best.loss.toFixed(3) : "—", "loss")}
            </div>
            {/* live network + learning log */}
            <div style={{ ...panelSt, marginBottom: 14 }}>
              <div className="split col-2e" style={{ gap: 18, alignItems: "start" }}>
                <div>
                  <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                    <h4 className="fld" style={{ margin: 0 }}>Network — weights forming live</h4>
                    <span className="note">green + · red − · thickness = |w|</span>
                  </div>
                  {netDiagram()}
                  <div className="note" style={{ marginTop: 6 }}>{epoch ? `epoch ${epoch}: each connection carries a weight the network keeps adjusting — strong edges (thick, saturated) emerge as it separates the classes.` : "press ▶ Train — the grey connections light up as the network learns."}</div>
                </div>
                <div>{learningLog()}</div>
              </div>
            </div>
            {/* balanced 2×2 graph grid: boundary + loss / weights + accuracy */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 14, alignItems: "start" }}>
              <div style={panelSt}>{trainViz()}</div>
              <div style={panelSt}>{c ? c.loss : <div className="note" style={{ padding: "50px 0", textAlign: "center" }}>loss curve appears once training starts</div>}</div>
              <div style={panelSt}>{weightHist()}</div>
              <div style={panelSt}>{c ? c.acc : <div className="note" style={{ padding: "50px 0", textAlign: "center" }}>{metricName} curve appears once training starts</div>}</div>
            </div>
            {best && best.vacc < best.acc - 0.12 && <div className="teach-note" style={{ marginTop: 12 }}><span className="ic">⚠️</span><span><b>Overfitting</b> — train {metricName} is well above validation. Add L2, shrink the network, or get more data.</span></div>}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("arch")}>← Back</button><button className="btn" onClick={() => { finishNow(); setStep("test"); }} disabled={!epoch}>Next: Test →</button></div>
          </div>
        </div>;
      })()}

      {step === "test" && data && (
        <div className="card"><div className="card-h"><span className="t">Test & Export</span></div>
          <div className="card-b">
            {!evalR ? <div className="note">Train the network first (step 5).</div> : (() => {
              const isReg = evalR.task === "regression";
              const m = !isReg && evalR.confusion ? prf(evalR.confusion) : { precision: 0, recall: 0, f1: 0 };
              // Time series: reconstruct LEVELS over the test horizon (v̂ₜ = vₜ₋₁ + Δ̂) and score those instead of the raw changes.
              const seriesPairs = isReg && data.series && netRef.current && scRef.current ? (() => {
                const net = netRef.current!, sc = scRef.current!, win = data.win || data.featNames.length; const rows = data.X.length; const cut = rows - Math.max(1, Math.round(rows * testFrac)); const pairs: [number, number][] = [];
                for (let j = cut; j < rows; j++) { const pd = predictVec(net, scaleRow(data.X[j], sc))[0]; pairs.push([data.series!.v[j + win + 1], data.series!.v[j + win] + pd]); }
                return pairs;
              })() : null;
              let rAcc = evalR.acc, rRmse = Math.sqrt(evalR.loss), rMae = isReg && evalR.predActual ? evalR.predActual.reduce((a, p) => a + Math.abs(p[0] - p[1]), 0) / evalR.predActual.length : 0;
              if (seriesPairs && seriesPairs.length) { const acts = seriesPairs.map((p) => p[0]); const mean = acts.reduce((a, b) => a + b, 0) / acts.length; let ssr = 0, sst = 0, ae = 0; seriesPairs.forEach(([a, p]) => { ssr += (p - a) ** 2; sst += (a - mean) ** 2; ae += Math.abs(p - a); }); rAcc = 1 - ssr / (sst || 1); rRmse = Math.sqrt(ssr / seriesPairs.length); rMae = ae / seriesPairs.length; }
              const badge: [string, string] = isReg ? (rAcc >= 0.8 ? ["▲ strong", "#3ecf7f"] : rAcc >= 0.5 ? ["! fair", "#f59e0b"] : ["✕ weak", "#ef4444"]) : (evalR.acc >= 0.9 ? ["▲ strong", "#3ecf7f"] : evalR.acc >= 0.75 ? ["! fair", "#f59e0b"] : ["✕ weak", "#ef4444"]);
              return <>
                {/* metric cards */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 20 }}>
                  {isReg
                    ? [mcard(rAcc.toFixed(3), seriesPairs ? "R² (forecast)" : "R² (test)", true, badge), mcard(rRmse.toFixed(2), "RMSE"), mcard(rMae.toFixed(2), "MAE")]
                    : [mcard(`${(evalR.acc * 100).toFixed(1)}%`, "accuracy", true, badge), mcard(m.precision.toFixed(2), "precision"), mcard(m.recall.toFixed(2), "recall"), mcard(m.f1.toFixed(2), "F1 score")]}
                </div>
                {/* matrix / scatter + breakdown */}
                {isReg && data.series && netRef.current && scRef.current && (() => {
                  const net = netRef.current!, sc = scRef.current!, win = data.win || data.featNames.length; const series = data.series!;
                  // one-step-ahead reconstructed level for every window: v̂[j+win+1] = v[j+win] + Δ̂
                  const predX = data.X.map((_, j) => j + win + 1); const predLvl = data.X.map((r, j) => series.v[j + win] + predictVec(net, scaleRow(r, sc))[0]);
                  const boundary = (data.X.length - Math.max(1, Math.round(data.X.length * testFrac))) + win + 1;
                  return <div style={{ ...panelSt, marginBottom: 20 }}><h4 className="fld" style={{ margin: "0 0 8px" }}>Forecast — one-step-ahead predicted vs actual</h4>
                    <Plot data={[{ type: "scatter", mode: "lines", name: "actual", x: series.t, y: series.v, line: { color: th.muted, width: 1.6 } }, { type: "scatter", mode: "lines", name: "predicted", x: predX, y: predLvl, line: { color: "#3ecf7f", width: 2 } }] as never} layout={{ ...lay("", "time step", "value", { showlegend: true, legend: { orientation: "h", y: -0.22 }, height: 340, margin: { l: 46, r: 12, t: 12, b: 46 } }), shapes: [{ type: "line", x0: boundary, x1: boundary, yref: "paper", y0: 0, y1: 1, line: { color: th.muted, dash: "dot", width: 1.4 } }], annotations: [{ x: boundary, y: 1, yref: "paper", yanchor: "bottom", text: "train ↔ test", showarrow: false, font: { color: th.muted, size: 10 } }] } as never} style={{ height: 340, width: "100%" }} />
                    <div className="note" style={{ marginTop: 6 }}>Left of the dotted line the model trained on; right is pure forecast on unseen future steps. The net predicts each step&apos;s <b>change</b>, added back onto the last value to reconstruct the level.</div>
                  </div>;
                })()}
                {isReg
                  ? evalR.predActual && <div style={{ ...panelSt, marginBottom: 20, maxWidth: 560 }}><h4 className="fld" style={{ margin: "0 0 8px" }}>Predicted vs actual (held-out test set)</h4><Plot data={[{ type: "scatter", mode: "markers", name: "test rows", x: evalR.predActual.map((p) => p[0]), y: evalR.predActual.map((p) => p[1]), marker: { color: "#5b7cff", size: 7, opacity: 0.75 } }, { type: "scatter", mode: "lines", name: "ŷ = y", x: [Math.min(...evalR.predActual.map((p) => p[0])), Math.max(...evalR.predActual.map((p) => p[0]))], y: [Math.min(...evalR.predActual.map((p) => p[0])), Math.max(...evalR.predActual.map((p) => p[0]))], line: { color: th.muted, dash: "dash" }, hoverinfo: "skip" }] as never} layout={lay("", "actual", "predicted", { showlegend: true, legend: { orientation: "h", y: -0.22 }, height: 340, margin: { l: 46, r: 12, t: 12, b: 46 } }) as never} style={{ height: 340, width: "100%" }} /></div>
                  : evalR.confusion && (() => { const cls = (evalR.classes ?? []).map(String); const cm = evalR.confusion; const ann: Record<string, unknown>[] = []; for (let a = 0; a < cls.length; a++) for (let p = 0; p < cls.length; p++) ann.push({ x: cls[p], y: cls[a], text: String(cm[a][p]), showarrow: false, font: { color: a === p ? "#eafff2" : "#ffe9e9", size: 16 } }); const sz = Math.min(380, 150 + cls.length * 72);
                    return <div className="split col-2e" style={{ gap: 16, alignItems: "start", marginBottom: 20 }}>
                      <div style={panelSt}><h4 className="fld" style={{ margin: "0 0 4px" }}>Confusion matrix</h4><div className="note" style={{ marginBottom: 6 }}>green = correct · amber/red = mistakes</div><Plot data={[{ type: "heatmap", x: cls, y: cls, z: cm, xgap: 4, ygap: 4, colorscale: [[0, th.plot], [0.5, "#f59e0b"], [1, "#3ecf7f"]], showscale: false, hoverinfo: "skip" }] as never} layout={{ ...lay("", "predicted →", "actual ↓", { height: sz, margin: { l: 42, r: 10, t: 8, b: 42 } }), annotations: ann, yaxis: { title: { text: "actual ↓" }, scaleanchor: "x", autorange: "reversed", gridcolor: th.grid } } as never} style={{ height: sz, width: "100%" }} /></div>
                      <div style={panelSt}><h4 className="fld" style={{ margin: "0 0 8px" }}>Per-class accuracy</h4><div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                        <tr><th>class</th><th>rows</th><th>correct</th><th>recall</th></tr>
                        {cls.map((c, k) => { const tot = cm[k].reduce((a, b) => a + b, 0); const cor = cm[k][k]; const rc = cor / (tot || 1); const col = rc >= 0.8 ? "var(--good)" : rc >= 0.5 ? "var(--warn)" : "var(--crit)"; return <tr key={k}><td>{c}</td><td className="mono">{tot}</td><td className="mono">{cor}</td><td className="mono" style={{ color: col }}><span style={{ display: "inline-block", height: 6, borderRadius: 3, width: Math.round(rc * 36), background: col, verticalAlign: "middle", marginRight: 6 }} />{rc.toFixed(2)}</td></tr>; })}
                      </tbody></table></div><div className="note" style={{ marginTop: 8, lineHeight: 1.6 }}>Recall = of the rows that truly belong to a class, the fraction predicted right. Low-recall rows are where the model struggles.</div></div>
                    </div>; })()}
                {/* predict panel */}
                <div style={{ ...panelSt, marginBottom: 20 }}>
                  <h4 className="fld" style={{ margin: "0 0 10px" }}>Predict a new sample <span className="note">(raw feature values)</span></h4>
                  <div className="row" style={{ gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
                    {data.featNames.slice(0, 8).map((f, j) => <div key={f} style={{ display: "flex", flexDirection: "column", gap: 4 }}><label className="note">{f}</label><input type="number" step="any" value={testInput[j] ?? ""} onChange={(e) => setTestInput((t) => { const n = [...t]; n[j] = +e.target.value; return n; })} style={{ width: 96 }} /></div>)}
                    {data.featNames.length > 8 && <span className="note">+{data.featNames.length - 8} more (default 0)</span>}
                    <button className="btn" onClick={doPredict}>Predict ↗</button>
                  </div>
                  {predResult && (predResult.kind === "reg"
                    ? <div style={{ marginTop: 14, border: "1px solid var(--accent)", borderRadius: 12, padding: "14px 16px", background: "var(--surface)" }}><div className="note" style={{ textTransform: "uppercase", letterSpacing: ".04em" }}>predicted value</div><div style={{ fontSize: 26, fontWeight: 600, color: "var(--accent)" }}>{predResult.value.toFixed(3)}</div></div>
                    : <div style={{ marginTop: 14, border: "1px solid var(--good)", borderRadius: 12, padding: "14px 16px", background: "var(--surface)" }}><div className="note" style={{ textTransform: "uppercase", letterSpacing: ".04em" }}>predicted class</div><div style={{ fontSize: 24, fontWeight: 600, color: "var(--good)" }}>{predResult.label} <span style={{ fontSize: 14, color: "var(--muted)", fontWeight: 400 }}>· {(predResult.conf * 100).toFixed(0)}% confident</span></div>
                      <div style={{ marginTop: 10 }}>{predResult.probs.map((pb) => <div key={pb.name} style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontFamily: "var(--mono)", fontSize: 12 }}><span style={{ width: 30, color: "var(--muted)", textAlign: "right" }}>{pb.name}</span><span style={{ flex: 1, height: 8, background: "var(--panel-2)", borderRadius: 4, overflow: "hidden" }}><span style={{ display: "block", height: "100%", width: `${pb.p * 100}%`, background: pb.name === predResult.label ? "var(--good)" : "var(--accent)", borderRadius: 4 }} /></span><span style={{ width: 34 }}>{pb.p.toFixed(2)}</span></div>)}</div>
                    </div>)}
                  <div className="note" style={{ marginTop: 8 }}>Inputs are standardized with the training μ/σ before the network sees them.</div>
                </div>
                {/* export */}
                <div style={panelSt}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
                    <h4 className="fld" style={{ margin: 0 }}>Export — equivalent PyTorch</h4>
                    <div className="row" style={{ gap: 8 }}>
                      <button className="btn ghost sm" onClick={copyCode}>{copied ? "✓ Copied" : "⧉ Copy"}</button>
                      <button className="btn ghost sm" onClick={() => downloadFile("model.py", pyCode, "text/x-python")}>⬇ Download .py</button>
                      <button className="btn ghost sm" onClick={() => downloadFile("model.json", modelJson(), "application/json")}>⬇ Weights .json</button>
                    </div>
                  </div>
                  <pre className="codebox" style={{ maxHeight: 280, overflow: "auto", fontSize: 11.5, margin: 0 }}>{pyCode}</pre>
                </div>
              </>;
            })()}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("train")}>← Back</button></div>
          </div>
        </div>
      )}
    </>
  );
}
