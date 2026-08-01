"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  genDataset, initNet, newOpt, trainEpoch, fullEval, predictVec, predictClass,
  fitScaler, applyScaler, scaleRow, dlSurface,
  type Net, type OptState, type DlTask, type Optimizer, type DlEval,
} from "@/lib/dlUtils";
import { parseCSV, buildMatrix, type Dataset } from "@/lib/mlUtils";
import { pca2 } from "@/lib/ragUtils";
import { plotlyTheme } from "@/lib/edaCharts";
import Plot from "@/components/Plot";

type Step = "data" | "explore" | "prep" | "arch" | "train" | "test";
const STEPS: { k: Step; n: number; label: string }[] = [
  { k: "data", n: 1, label: "Data" }, { k: "explore", n: 2, label: "Explore" }, { k: "prep", n: 3, label: "Preprocess" },
  { k: "arch", n: 4, label: "Architecture" }, { k: "train", n: 5, label: "Train" }, { k: "test", n: 6, label: "Test & Export" },
];
const SAMPLES = [
  { k: "spiral", l: "Spiral", t: "binary" }, { k: "circles", l: "Circles", t: "binary" }, { k: "xor", l: "XOR", t: "binary" },
  { k: "moons", l: "Moons", t: "binary" }, { k: "blobs3", l: "3 Blobs (multiclass)", t: "multiclass" }, { k: "sine", l: "Sine (regression)", t: "regression" },
];
const PAL = ["#5b7cff", "#f59e0b", "#3ecf7f", "#ef4444", "#a855f7", "#22b8cf", "#ec4899", "#84cc16"];

type Resolved = { X: number[][]; y: number[]; task: DlTask; classes: string[]; featNames: string[]; source: string };

// deterministic split
function splitData(X: number[][], y: number[], testFrac: number) {
  const n = X.length; const idx = [...Array(n).keys()];
  let a = 12345; for (let i = n - 1; i > 0; i--) { a = (a * 1103515245 + 12345) & 0x7fffffff; const j = a % (i + 1); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTe = Math.max(1, Math.round(n * testFrac)); const te = new Set(idx.slice(0, nTe));
  const Xtr: number[][] = [], ytr: number[] = [], Xte: number[][] = [], yte: number[] = [];
  X.forEach((r, i) => { if (te.has(i)) { Xte.push(r); yte.push(y[i]); } else { Xtr.push(r); ytr.push(y[i]); } });
  return { Xtr, ytr, Xte, yte };
}

export default function DlLab() {
  const [step, setStep] = useState<Step>("data");
  const [source, setSource] = useState<"sample" | "csv">("sample");
  const [toy, setToy] = useState("spiral");
  const [noise, setNoise] = useState(0.12);
  const [ds, setDs] = useState<Dataset | null>(null);
  const [dsName, setDsName] = useState("");
  const [feats, setFeats] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const [data, setData] = useState<Resolved | null>(null);
  const [testFrac] = useState(0.25);
  const [exFx, setExFx] = useState(0);
  const [exFy, setExFy] = useState(1);
  const [exMode, setExMode] = useState<"scatter" | "pca" | "dist">("scatter");

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
  const th = plotlyTheme();

  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);

  // ── resolve dataset (sample or CSV) ──
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
  function detectTask(d: Dataset, tgt: string): DlTask {
    const col = d.columns.find((c) => c.name === tgt); if (!col) return "binary";
    const vals = col.values.filter((v) => v != null); const uniq = new Set(vals.map(String)).size;
    if (col.type === "num" && uniq > 12) return "regression";
    return uniq <= 2 ? "binary" : "multiclass";
  }
  function buildFromCsv() {
    if (!ds || !feats.length || !target) { setMsg("Pick at least one feature and a target."); return; }
    try {
      const dlTask = detectTask(ds, target);
      const b = buildMatrix(ds, feats.filter((f) => f !== target), target, dlTask === "regression" ? "regression" : "classification", []);
      if (!b.X.length) { setMsg("No usable rows."); return; }
      const classes = b.classes ?? [];
      const task: DlTask = dlTask === "regression" ? "regression" : (classes.length <= 2 ? "binary" : "multiclass");
      setData({ X: b.X, y: b.y, task, classes, featNames: b.featureNames, source: dsName || "uploaded.csv" });
      resetTraining(); setStep("explore"); setMsg("");
    } catch (e) { setMsg("Build error: " + (e as Error).message); }
  }
  function resetTraining() { stopTrain(); setEpoch(0); epochRef.current = 0; setHistory([]); setSurface(null); setEvalR(null); netRef.current = null; optRef.current = null; scRef.current = null; splitRef.current = null; }

  const outDim = data ? (data.task === "multiclass" ? data.classes.length : 1) : 1;
  const K = data ? (data.task === "binary" ? 2 : data.classes.length) : 0;

  // ── training ──
  function prepareSplit() {
    if (!data) return null;
    const sc = fitScaler(data.X); scRef.current = sc;
    const Xs = applyScaler(data.X, sc);
    const sp = splitData(Xs, data.y, testFrac); splitRef.current = sp;
    // raw feature ranges (first two) for boundary axes
    const f0 = data.X.map((r) => r[0]); const f1 = data.featNames.length > 1 ? data.X.map((r) => r[1]) : f0;
    rangeRef.current = { lo: [Math.min(...f0), Math.min(...f1)], hi: [Math.max(...f0), Math.max(...f1)] };
    return sp;
  }
  function startTrain() {
    if (!data) return; stopTrain();
    const sp = prepareSplit(); if (!sp) return;
    const net = initNet(data.X[0].length, hidden, outDim, act, data.task); netRef.current = net; optRef.current = newOpt(net, optimizer);
    setHistory([]); setEpoch(0); setEvalR(null); setRunning(true); epochRef.current = 0;
    timer.current = setInterval(() => {
      const n = netRef.current!, o = optRef.current!, s = splitRef.current!;
      const stat = trainEpoch(n, s.Xtr, s.ytr, { lr, l2, batchSize }, o);
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
    if (!data) return null; const H = 360;
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

  // curves
  const curveFig = () => { if (!history.length) return null; const H = 220; const metric = data?.task === "regression" ? "R²" : "accuracy";
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

      {step === "data" && (
        <div className="card">
          <div className="card-h"><span className="t">Choose data</span></div>
          <div className="card-b">
            <div className="chips" style={{ marginBottom: 12 }}>
              <button className={`chip ${source === "sample" ? "on" : ""}`} onClick={() => setSource("sample")}>Built-in dataset</button>
              <button className={`chip ${source === "csv" ? "on" : ""}`} onClick={() => setSource("csv")}>Upload CSV</button>
            </div>
            {source === "sample" ? <>
              <label className="fld">Dataset</label>
              <div className="chips" style={{ marginBottom: 10 }}>{SAMPLES.map((s) => <button key={s.k} className={`chip ${toy === s.k ? "on" : ""}`} onClick={() => setToy(s.k)}>{s.l}</button>)}</div>
              <div className="knob" style={{ maxWidth: 260 }}><div className="kr"><span>Noise</span><b>{noise.toFixed(2)}</b></div><input type="range" min={0} max={0.4} step={0.02} value={noise} onChange={(e) => setNoise(+e.target.value)} /></div>
              <div className="note" style={{ marginTop: 8 }}>{SAMPLES.find((s) => s.k === toy)?.t === "regression" ? "1 feature → continuous target (regression)." : SAMPLES.find((s) => s.k === toy)?.t === "multiclass" ? "2 features → 3 classes (softmax)." : "2 features → 2 classes (binary)."}</div>
              <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={resolveSample}>Use this dataset →</button></div>
            </> : <>
              <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" hidden onChange={(e) => onFile(e.target.files?.[0] || null)} />
              <div className="row" style={{ gap: 10, alignItems: "center", marginBottom: 10 }}><button className="btn ghost" onClick={() => fileRef.current?.click()}>⬆ Upload CSV</button>{ds && <span className="note">{dsName} · {ds.nrows} rows · {ds.columns.length} columns</span>}</div>
              {ds && <>
                <div className="split col-2e" style={{ gap: 14 }}>
                  <div><label className="fld">Feature columns</label><div style={{ maxHeight: 180, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 8, padding: 8 }}>{ds.columns.map((c) => c.name).filter((c) => c !== target).map((c) => <label key={c} className="note" style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0", cursor: "pointer" }}><input type="checkbox" checked={feats.includes(c)} onChange={() => setFeats((f) => f.includes(c) ? f.filter((x) => x !== c) : [...f, c])} />{c}</label>)}</div></div>
                  <div><label className="fld">Target column</label><select value={target} onChange={(e) => { setTarget(e.target.value); setFeats((f) => f.filter((x) => x !== e.target.value)); }}>{ds.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select>
                    <div className="note" style={{ marginTop: 8 }}>Detected task: <b>{ds && target ? detectTask(ds, target) : "—"}</b> (auto from the target: 2 classes → binary, 3+ → multiclass, many numeric values → regression).</div>
                  </div>
                </div>
                <label className="fld" style={{ marginTop: 12 }}>Data preview (first 8 rows)</label>
                {previewTable()}
                <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={buildFromCsv} disabled={!feats.length}>Build & continue →</button></div>
              </>}
              {!ds && <div className="note">Upload a CSV — first row = column names. Then pick which columns are inputs and which is the target.</div>}
            </>}
          </div>
        </div>
      )}

      {step === "explore" && data && (
        <div className="card"><div className="card-h"><span className="t">Explore — {data.source}</span><span className="mono r">{data.X.length} rows · {data.featNames.length} features · {data.task}</span></div>
          <div className="card-b">
            {data.featNames.length > 1 && <div className="row" style={{ gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
              <div className="chips"><button className={`chip ${exMode === "scatter" ? "on" : ""}`} onClick={() => setExMode("scatter")}>2-feature scatter</button><button className={`chip ${exMode === "pca" ? "on" : ""}`} onClick={() => setExMode("pca")}>PCA projection</button><button className={`chip ${exMode === "dist" ? "on" : ""}`} onClick={() => setExMode("dist")}>distribution</button></div>
              {(exMode === "scatter") && <><label className="note">X</label><select value={exFx} onChange={(e) => setExFx(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select><label className="note">Y</label><select value={exFy} onChange={(e) => setExFy(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select></>}
              {exMode === "dist" && <><label className="note">feature</label><select value={exFx} onChange={(e) => setExFx(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select></>}
            </div>}
            {exploreFig && <Plot data={exploreFig.data as never} layout={lay(exploreFig.title, exploreFig.xl, exploreFig.yl, { showlegend: (exploreFig as { legend?: boolean }).legend !== false, legend: { orientation: "h", y: -0.18 }, height: 380, barmode: (exploreFig as { barmode?: string }).barmode }) as never} style={{ height: 380, width: "100%" }} />}
            {classCounts && <div className="note" style={{ marginTop: 8 }}>class balance: {classCounts.map((c, i) => `${data.classes[i] ?? i}=${c}`).join("  ·  ")}{Math.max(...classCounts) / Math.min(...classCounts, 1) > 2 ? " — imbalanced; watch precision/recall, not just accuracy" : ""}</div>}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("data")}>← Back</button><button className="btn" onClick={() => setStep("prep")}>Next: Preprocess →</button></div>
          </div>
        </div>
      )}

      {step === "prep" && data && (() => {
        const sc = fitScaler(data.X); const nf = data.featNames.length; const H = 240;
        const flow = [`${nf} feature${nf === 1 ? "" : "s"}`, "standardize  z = (x−μ)/σ", `split  ${Math.round(data.X.length * (1 - testFrac))} train / ${Math.round(data.X.length * testFrac)} test`, "→ network"];
        const fj = Math.min(exFx, nf - 1); const raw = data.X.map((r) => r[fj]); const scv = raw.map((v) => (v - sc.mean[fj]) / sc.std[fj]);
        const boxRaw = data.featNames.map((f, j) => ({ type: "box", name: f, y: data.X.map((r) => r[j]), marker: { color: "#5b7cff" }, boxpoints: false }));
        const boxScaled = data.featNames.map((f, j) => ({ type: "box", name: f, y: data.X.map((r) => (r[j] - sc.mean[j]) / sc.std[j]), marker: { color: "#3ecf7f" }, boxpoints: false }));
        return <div className="card"><div className="card-h"><span className="t">Preprocess — standardize &amp; split</span></div>
          <div className="card-b">
            <div className="teach-note"><span className="ic">📏</span><span>Neural nets train far better on <b>standardized</b> inputs — each feature is rescaled to mean&nbsp;0, std&nbsp;1 with <b>z = (x − μ) / σ</b> — so no large-scale feature dominates the gradients. Then a test set is held out and never trained on.</span></div>
            <div className="row" style={{ gap: 6, alignItems: "center", flexWrap: "wrap", margin: "12px 0" }}>{flow.map((s, i) => <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ padding: "6px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--panel)", fontFamily: "var(--mono)", fontSize: 12 }}>{s}</span>{i < flow.length - 1 && <span style={{ color: "var(--faint)" }}>→</span>}</span>)}</div>
            <label className="fld">Feature scales — raw (blue) vs standardized (green). Standardizing pulls every feature onto the same scale.</label>
            <div className="split col-2e" style={{ gap: 12 }}>
              <Plot data={boxRaw as never} layout={lay("raw feature ranges", "", "value", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
              <Plot data={boxScaled as never} layout={lay("standardized (z-score)", "", "z", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
            </div>
            <div className="row" style={{ gap: 8, alignItems: "center", margin: "12px 0 4px" }}><label className="note">one feature, before/after:</label><select value={exFx} onChange={(e) => setExFx(+e.target.value)}>{data.featNames.map((f, i) => <option key={i} value={i}>{f}</option>)}</select></div>
            <div className="split col-2e" style={{ gap: 12 }}>
              <Plot data={[{ type: "histogram", x: raw, marker: { color: "#5b7cff" }, opacity: 0.85 }] as never} layout={lay(`${data.featNames[fj]} — raw`, data.featNames[fj], "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
              <Plot data={[{ type: "histogram", x: scv, marker: { color: "#3ecf7f" }, opacity: 0.85 }] as never} layout={lay(`${data.featNames[fj]} — standardized`, "z-score", "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
            </div>
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("explore")}>← Back</button><button className="btn" onClick={() => setStep("arch")}>Next: Architecture →</button></div>
          </div>
        </div>;
      })()}

      {step === "arch" && data && (
        <div className="card"><div className="card-h"><span className="t">Design the network</span></div>
          <div className="card-b">
            <div className="teach-note" style={{ marginBottom: 12 }}><span className="ic">🧱</span><span><b>Layer types:</b> <b>Dense</b> (fully-connected) — every input connects to every neuron; the right choice for <b>tabular</b> data, so it&apos;s what this lab uses. <b>Conv</b> (convolution) &amp; <b>Pooling</b> layers scan/​downsample <b>images</b> and need a GPU to train — out of scope here. So below, each hidden layer is a <b>Dense</b> layer and you choose how many <b>neurons</b> it has.</span></div>
            <div className="split col-2e" style={{ gap: 16 }}>
              <div>
                <label className="fld">Hidden Dense layers ({hidden.length}) — neurons per layer</label>
                {hidden.map((h, i) => <div key={i} className="row" style={{ gap: 8, alignItems: "center", marginBottom: 6 }}><span className="note" style={{ width: 96 }}>Dense {i + 1}</span><input type="range" min={1} max={16} value={h} onChange={(e) => setHidden((hs) => hs.map((x, j) => (j === i ? +e.target.value : x)))} style={{ flex: 1 }} /><b className="mono" style={{ width: 62 }}>{h} neuron{h === 1 ? "" : "s"}</b><button className="btn ghost sm" onClick={() => setHidden((hs) => hs.filter((_, j) => j !== i))} disabled={hidden.length <= 1}>×</button></div>)}
                <button className="btn ghost sm" onClick={() => setHidden((hs) => [...hs, 6])} disabled={hidden.length >= 4}>+ add layer</button>
                <label className="fld" style={{ marginTop: 12 }}>Activation (non-linearity between layers)</label>
                <div className="chips">{["tanh", "relu", "sigmoid"].map((a) => <button key={a} className={`chip ${act === a ? "on" : ""}`} onClick={() => setAct(a)}>{a}</button>)}</div>
                <label className="fld" style={{ marginTop: 12 }}>Input &amp; output (set by your data)</label>
                <div className="note">Input: <b>{data.X[0].length} neurons</b> (one per feature). Output: {data.task === "binary" ? "1 neuron · sigmoid · binary cross-entropy" : data.task === "multiclass" ? `${outDim} neurons · softmax · cross-entropy (one per class)` : "1 neuron · linear · mean squared error"}.</div>
              </div>
              <div><label className="fld">Network — each column is a layer, each dot a neuron</label>{netDiagram()}<div className="note" style={{ marginTop: 6 }}>Labels: <b>2 in</b> = input neurons (features) · <b>h1·8</b> = hidden layer 1 with 8 neurons · <b>1 out</b> = output. {netRef.current ? "Edge colour = learned weight (green +, red −), thickness = magnitude." : "Edges are connections; train to see the learned weights."}</div></div>
            </div>
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("prep")}>← Back</button><button className="btn" onClick={() => setStep("train")}>Next: Train →</button></div>
          </div>
        </div>
      )}

      {step === "train" && data && (
        <div className="card"><div className="card-h"><span className="t">Train</span><span className="mono r">{running ? "training…" : epoch ? `epoch ${epoch}` : "not started"}</span></div>
          <div className="card-b">
            <div className="teach-note" style={{ marginBottom: 12 }}><span className="ic">🎛️</span><span><b>Learning rate</b> = step size per update (too high → unstable, too low → slow). <b>Optimizer</b> = how steps are computed (Adam adapts per-weight; usually converges fastest). <b>L2</b> = weight penalty that fights overfitting. <b>Batch size</b> = rows averaged before each update. An <b>epoch</b> = one full pass over the training data.</span></div>
            <div className="row" style={{ gap: 14, flexWrap: "wrap", marginBottom: 12, alignItems: "center" }}>
              <div className="knob" style={{ margin: 0, minWidth: 160 }}><div className="kr"><span>Learning rate (η)</span><b>{lr}</b></div><input type="range" min={0.001} max={0.2} step={0.001} value={lr} onChange={(e) => setLr(+e.target.value)} /></div>
              <div><label className="note">Optimizer</label><select value={optimizer} onChange={(e) => setOptimizer(e.target.value as Optimizer)}><option value="adam">Adam</option><option value="momentum">SGD + momentum</option><option value="sgd">SGD</option></select></div>
              <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>L2 (weight decay)</span><b>{l2}</b></div><input type="range" min={0} max={0.01} step={0.0005} value={l2} onChange={(e) => setL2(+e.target.value)} /></div>
              <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Batch size (rows)</span><b>{batchSize}</b></div><input type="range" min={1} max={64} step={1} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} /></div>
              <div className="knob" style={{ margin: 0, minWidth: 160 }}><div className="kr"><span>Epochs (how long to train)</span><b>{epochsTarget}</b></div><input type="range" min={50} max={1000} step={25} value={epochsTarget} onChange={(e) => setEpochsTarget(+e.target.value)} disabled={running} /></div>
            </div>
            <div className="row" style={{ gap: 8, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
              <button className="btn" onClick={running ? stopTrain : startTrain}>{running ? "⏸ Pause" : epoch ? "↻ Restart" : "▶ Train"}</button>
              {running && <button className="btn ghost" onClick={finishNow}>⏭ Finish now</button>}
              <span className="note">epoch <b>{epoch}</b> / {epochsTarget}</span>
              {best && <span className="note">· train {data.task === "regression" ? "R²" : "accuracy"} <b>{best.acc.toFixed(3)}</b> · validation <b>{best.vacc.toFixed(3)}</b> · loss <b>{best.loss.toFixed(3)}</b></span>}
            </div>
            {trainViz()}
            {history.length > 0 && (() => { const c = curveFig(); return c ? <div className="split col-2e" style={{ gap: 12, marginTop: 12 }}>{c.loss}{c.acc}</div> : null; })()}
            {best && best.vacc < best.acc - 0.12 && <div className="teach-note" style={{ marginTop: 10 }}><span className="ic">⚠️</span><span><b>Overfitting</b> — train {data.task === "regression" ? "R²" : "accuracy"} is well above validation. Add L2, shrink the network, or get more data.</span></div>}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("arch")}>← Back</button><button className="btn" onClick={() => { finishNow(); setStep("test"); }} disabled={!epoch}>Next: Test →</button></div>
          </div>
        </div>
      )}

      {step === "test" && data && (
        <div className="card"><div className="card-h"><span className="t">Test & Export</span></div>
          <div className="card-b">
            {!evalR ? <div className="note">Train the network first (step 5).</div> : <>
              {evalR.task === "regression"
                ? <>
                  <div className="split" style={{ gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 14, maxWidth: 420 }}><div className="metric"><span className="v">{evalR.acc.toFixed(3)}</span><span className="k">R² (test)</span></div><div className="metric"><span className="v">{Math.sqrt(evalR.loss).toFixed(2)}</span><span className="k">RMSE</span></div><div className="metric"><span className="v">{evalR.predActual ? (evalR.predActual.reduce((a, p) => a + Math.abs(p[0] - p[1]), 0) / evalR.predActual.length).toFixed(2) : "—"}</span><span className="k">MAE</span></div></div>
                  {evalR.predActual && <div style={{ maxWidth: 460 }}><Plot data={[{ type: "scatter", mode: "markers", name: "test rows", x: evalR.predActual.map((p) => p[0]), y: evalR.predActual.map((p) => p[1]), marker: { color: "#5b7cff", size: 7, opacity: 0.75 } }, { type: "scatter", mode: "lines", name: "ŷ = y (perfect)", x: [Math.min(...evalR.predActual.map((p) => p[0])), Math.max(...evalR.predActual.map((p) => p[0]))], y: [Math.min(...evalR.predActual.map((p) => p[0])), Math.max(...evalR.predActual.map((p) => p[0]))], line: { color: th.muted, dash: "dash" }, hoverinfo: "skip" }] as never} layout={lay("predicted vs actual (held-out test set)", "actual", "predicted", { showlegend: true, legend: { orientation: "h", y: -0.22 }, height: 360 }) as never} style={{ height: 360, width: "100%" }} /></div>}
                </>
                : <>
                  {(() => { const m = evalR.confusion ? prf(evalR.confusion) : { precision: 0, recall: 0, f1: 0 }; return <div className="split" style={{ gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 14, maxWidth: 520 }}><div className="metric"><span className="v">{(evalR.acc * 100).toFixed(1)}%</span><span className="k">accuracy</span></div><div className="metric"><span className="v">{m.precision.toFixed(2)}</span><span className="k">precision</span></div><div className="metric"><span className="v">{m.recall.toFixed(2)}</span><span className="k">recall</span></div><div className="metric"><span className="v">{m.f1.toFixed(2)}</span><span className="k">F1</span></div></div>; })()}
                  {evalR.confusion && (() => { const cls = (evalR.classes ?? []).map(String); const cm = evalR.confusion; const ann: Record<string, unknown>[] = []; for (let a = 0; a < cls.length; a++) for (let p = 0; p < cls.length; p++) ann.push({ x: cls[p], y: cls[a], text: String(cm[a][p]), showarrow: false, font: { color: a === p ? "#eafff2" : "#ffe9e9", size: 16 } }); const sz = Math.min(400, 150 + cls.length * 78);
                    return <div className="split col-2e" style={{ gap: 16, alignItems: "start" }}>
                      <div style={{ maxWidth: sz }}>
                        <label className="fld">Confusion matrix (test) — diagonal = correct, off-diagonal = mistakes</label>
                        <Plot data={[{ type: "heatmap", x: cls, y: cls, z: cm, xgap: 4, ygap: 4, colorscale: [[0, th.plot], [0.5, "#f59e0b"], [1, "#3ecf7f"]], showscale: false, hoverinfo: "skip" }] as never} layout={{ ...lay("", "predicted →", "actual ↓", { height: sz, margin: { l: 46, r: 12, t: 12, b: 46 } }), annotations: ann, yaxis: { title: { text: "actual ↓" }, scaleanchor: "x", autorange: "reversed", gridcolor: th.grid } } as never} style={{ height: sz, width: "100%" }} />
                      </div>
                      <div>
                        <label className="fld">Per-class accuracy (test)</label>
                        <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                          <tr><th>class</th><th>rows</th><th>correct</th><th>recall</th></tr>
                          {cls.map((c, k) => { const tot = cm[k].reduce((a, b) => a + b, 0); const cor = cm[k][k]; const rc = cor / (tot || 1); return <tr key={k}><td>{c}</td><td className="mono">{tot}</td><td className="mono">{cor}</td><td className="mono" style={{ color: rc >= 0.8 ? "var(--good)" : rc >= 0.5 ? "var(--warn)" : "var(--crit)" }}>{rc.toFixed(2)}</td></tr>; })}
                        </tbody></table></div>
                        <div className="note" style={{ marginTop: 8, lineHeight: 1.6 }}>Recall = of the rows that truly belong to a class, the fraction the model got right (the diagonal ÷ its row). Low-recall classes are where the model struggles.</div>
                      </div>
                    </div>; })()}
                </>}
              <div className="split col-2e" style={{ gap: 16, marginTop: 18 }}>
                <div>
                  <label className="fld">Predict one input (raw feature values)</label>
                  <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
                    {data.featNames.slice(0, 6).map((f, j) => <div key={f}><label className="note">{f}</label><input type="number" step="any" value={testInput[j] ?? ""} onChange={(e) => setTestInput((t) => { const n = [...t]; n[j] = +e.target.value; return n; })} style={{ width: 88 }} /></div>)}
                    {data.featNames.length > 6 && <span className="note">+{data.featNames.length - 6} more (defaults to 0)</span>}
                    <button className="btn sm" onClick={() => { const x = data.featNames.map((_, j) => testInput[j] ?? 0); if (netRef.current && scRef.current) { const o = predictVec(netRef.current, scaleRow(x, scRef.current)); if (data.task === "regression") setMsg(`Prediction: ${o[0].toFixed(3)}`); else { const c = o.indexOf(Math.max(...o)); setMsg(`Prediction: ${data.classes[c] ?? c} (${(Math.max(...o) * 100).toFixed(0)}% confident)`); } } }}>Predict</button>
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>Values are standardized with the training μ/σ before the network sees them.</div>
                </div>
                <div>
                  <label className="fld">Equivalent PyTorch</label>
                  <pre className="codebox" style={{ maxHeight: 240, overflow: "auto", fontSize: 11.5 }}>{pyCode}</pre>
                </div>
              </div>
            </>}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("train")}>← Back</button></div>
          </div>
        </div>
      )}
    </>
  );
}
