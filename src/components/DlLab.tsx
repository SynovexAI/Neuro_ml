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
const MAX_EPOCHS = 500;

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

  const [hidden, setHidden] = useState<number[]>([8, 6]);
  const [act, setAct] = useState("tanh");
  const [optimizer, setOptimizer] = useState<Optimizer>("adam");
  const [lr, setLr] = useState(0.02);
  const [l2, setL2] = useState(0);
  const [batchSize, setBatchSize] = useState(16);

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
      if (ep >= MAX_EPOCHS) { stopTrain(); setEvalR(fullEval(n, s.Xte, s.yte, data.classes)); }
    }, 40);
  }
  function stopTrain() { if (timer.current) { clearInterval(timer.current); timer.current = null; } setRunning(false); }
  function finishNow() { if (netRef.current && splitRef.current && data) { stopTrain(); if (data.featNames.length === 2 && data.task !== "regression") setSurface(dlSurface(netRef.current, scRef.current!, rangeRef.current!.lo, rangeRef.current!.hi, 44)); setEvalR(fullEval(netRef.current, splitRef.current.Xte, splitRef.current.yte, data.classes)); } }

  // ── explore figures ──
  const exploreFig = useMemo(() => {
    if (!data) return null; const { X, y, task, classes, featNames } = data;
    if (task === "regression" && featNames.length === 1) return { data: [{ type: "scatter", mode: "markers", x: X.map((r) => r[0]), y, marker: { color: "#5b7cff", size: 6, opacity: 0.7 } }], title: `${featNames[0]} vs target`, xl: featNames[0], yl: "target" };
    const pts = featNames.length === 2 ? X.map((r) => ({ x: r[0], y: r[1] })) : pca2(X);
    const xl = featNames.length === 2 ? featNames[0] : "PC1", yl = featNames.length === 2 ? featNames[1] : "PC2";
    if (task === "regression") return { data: [{ type: "scatter", mode: "markers", x: pts.map((p) => p.x), y: pts.map((p) => p.y), marker: { color: y, colorscale: "Viridis", size: 7, showscale: true, colorbar: { title: { text: "target" } } } }], title: featNames.length === 2 ? "features coloured by target" : "PCA projection (colour = target)", xl, yl };
    const traces = [...Array(K).keys()].map((c) => ({ type: "scatter", mode: "markers", name: classes[c] ?? `class ${c}`, x: pts.map((p, i) => (y[i] === c ? p.x : null)), y: pts.map((p, i) => (y[i] === c ? p.y : null)), marker: { color: PAL[c % PAL.length], size: 7, opacity: 0.8 } }));
    return { data: traces, title: featNames.length === 2 ? "data by class" : "PCA projection by class", xl, yl };
  }, [data, K]);
  const classCounts = useMemo(() => { if (!data || data.task === "regression") return null; const c = new Array(K).fill(0); data.y.forEach((v) => { c[v]++; }); return c; }, [data, K]);

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
    return `import torch, torch.nn as nn\n\nmodel = nn.Sequential(\n${layers.join("\n")}\n)\ncriterion = ${loss}\noptimizer = ${optLine}\n\n# X: (n, ${data.X[0].length}) standardized features   y: ${data.task === "regression" ? "(n,) float" : data.task === "binary" ? "(n,) 0/1" : "(n,) class index"}\nfor epoch in range(${MAX_EPOCHS}):\n    optimizer.zero_grad()\n    out = model(X)\n    loss = criterion(out${data.task === "binary" ? ".squeeze(1)" : ""}, y${data.task === "regression" ? ".unsqueeze(1)" : ""})\n    loss.backward(); optimizer.step()`;
  }, [data, hidden, act, outDim, optimizer, lr]);

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
            {exploreFig && <Plot data={exploreFig.data as never} layout={lay(exploreFig.title, exploreFig.xl, exploreFig.yl, { showlegend: data.task !== "regression", legend: { orientation: "h", y: -0.18 }, height: 380 }) as never} style={{ height: 380, width: "100%" }} />}
            {classCounts && <div className="note" style={{ marginTop: 8 }}>class balance: {classCounts.map((c, i) => `${data.classes[i] ?? i}=${c}`).join("  ·  ")}{Math.max(...classCounts) / Math.min(...classCounts, 1) > 2 ? " — imbalanced; watch precision/recall, not just accuracy" : ""}</div>}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("data")}>← Back</button><button className="btn" onClick={() => setStep("prep")}>Next: Preprocess →</button></div>
          </div>
        </div>
      )}

      {step === "prep" && data && (
        <div className="card"><div className="card-h"><span className="t">Preprocess — standardize + split</span></div>
          <div className="card-b">
            <div className="teach-note"><span className="ic">📏</span><span>Neural nets train far better on <b>standardized</b> inputs (each feature → mean 0, std 1), so gradients don’t get dominated by large-scale features. We do this automatically and hold out a test set.</span></div>
            {(() => { const sc = fitScaler(data.X); const j = 0; const raw = data.X.map((r) => r[j]); const scv = raw.map((v) => (v - sc.mean[j]) / sc.std[j]); const H = 240;
              return <div className="split col-2e" style={{ gap: 12, marginTop: 12 }}>
                <Plot data={[{ type: "histogram", x: raw, marker: { color: "#5b7cff" }, opacity: 0.85 }] as never} layout={lay(`${data.featNames[0]} — raw`, data.featNames[0], "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
                <Plot data={[{ type: "histogram", x: scv, marker: { color: "#3ecf7f" }, opacity: 0.85 }] as never} layout={lay(`${data.featNames[0]} — standardized`, "z-score", "count", { height: H, showlegend: false }) as never} style={{ height: H, width: "100%" }} />
              </div>; })()}
            <div className="note" style={{ marginTop: 10 }}>{Math.round(data.X.length * (1 - testFrac))} train rows · {Math.round(data.X.length * testFrac)} test rows (held out, never trained on).</div>
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("explore")}>← Back</button><button className="btn" onClick={() => setStep("arch")}>Next: Architecture →</button></div>
          </div>
        </div>
      )}

      {step === "arch" && data && (
        <div className="card"><div className="card-h"><span className="t">Design the network</span></div>
          <div className="card-b">
            <div className="split col-2e" style={{ gap: 16 }}>
              <div>
                <label className="fld">Hidden layers ({hidden.length})</label>
                {hidden.map((h, i) => <div key={i} className="row" style={{ gap: 8, alignItems: "center", marginBottom: 6 }}><span className="note" style={{ width: 54 }}>layer {i + 1}</span><input type="range" min={1} max={16} value={h} onChange={(e) => setHidden((hs) => hs.map((x, j) => (j === i ? +e.target.value : x)))} style={{ flex: 1 }} /><b className="mono" style={{ width: 24 }}>{h}</b><button className="btn ghost sm" onClick={() => setHidden((hs) => hs.filter((_, j) => j !== i))} disabled={hidden.length <= 1}>×</button></div>)}
                <button className="btn ghost sm" onClick={() => setHidden((hs) => [...hs, 6])} disabled={hidden.length >= 4}>+ add layer</button>
                <label className="fld" style={{ marginTop: 12 }}>Activation</label>
                <div className="chips">{["tanh", "relu", "sigmoid"].map((a) => <button key={a} className={`chip ${act === a ? "on" : ""}`} onClick={() => setAct(a)}>{a}</button>)}</div>
                <label className="fld" style={{ marginTop: 12 }}>Output head (auto)</label>
                <div className="note">{data.task === "binary" ? "1 neuron · sigmoid · binary cross-entropy" : data.task === "multiclass" ? `${outDim} neurons · softmax · cross-entropy` : "1 neuron · linear · mean squared error"}</div>
              </div>
              <div><label className="fld">Network</label>{netDiagram()}<div className="note" style={{ marginTop: 6 }}>{netRef.current ? "edge colour = learned weight (green +, red −); thickness = magnitude" : "edges show connections; train to see learned weights"}</div></div>
            </div>
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("prep")}>← Back</button><button className="btn" onClick={() => setStep("train")}>Next: Train →</button></div>
          </div>
        </div>
      )}

      {step === "train" && data && (
        <div className="card"><div className="card-h"><span className="t">Train</span><span className="mono r">{running ? "training…" : epoch ? `epoch ${epoch}` : "not started"}</span></div>
          <div className="card-b">
            <div className="row" style={{ gap: 14, flexWrap: "wrap", marginBottom: 12, alignItems: "center" }}>
              <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Learning rate</span><b>{lr}</b></div><input type="range" min={0.001} max={0.2} step={0.001} value={lr} onChange={(e) => setLr(+e.target.value)} /></div>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value as Optimizer)}><option value="adam">Adam</option><option value="momentum">SGD + momentum</option><option value="sgd">SGD</option></select>
              <div className="knob" style={{ margin: 0, minWidth: 140 }}><div className="kr"><span>L2</span><b>{l2}</b></div><input type="range" min={0} max={0.01} step={0.0005} value={l2} onChange={(e) => setL2(+e.target.value)} /></div>
              <div className="knob" style={{ margin: 0, minWidth: 140 }}><div className="kr"><span>Batch size</span><b>{batchSize}</b></div><input type="range" min={1} max={64} step={1} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} /></div>
            </div>
            <div className="row" style={{ gap: 8, marginBottom: 12 }}>
              <button className="btn" onClick={running ? stopTrain : startTrain}>{running ? "⏸ Pause" : epoch ? "↻ Restart" : "▶ Train"}</button>
              {running && <button className="btn ghost" onClick={finishNow}>⏭ Finish now</button>}
              {best && <span className="note">train {data.task === "regression" ? "R²" : "acc"} {best.acc.toFixed(3)} · val {best.vacc.toFixed(3)} · loss {best.loss.toFixed(3)}</span>}
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
                ? <><div className="split" style={{ gridTemplateColumns: "repeat(2,1fr)", gap: 10, marginBottom: 14, maxWidth: 320 }}><div className="metric"><span className="v">{evalR.acc.toFixed(3)}</span><span className="k">R² (test)</span></div><div className="metric"><span className="v">{Math.sqrt(evalR.loss).toFixed(2)}</span><span className="k">RMSE</span></div></div>
                  {evalR.predActual && <Plot data={[{ type: "scatter", mode: "markers", name: "test", x: evalR.predActual.map((p) => p[0]), y: evalR.predActual.map((p) => p[1]), marker: { color: "#5b7cff", size: 7, opacity: 0.75 } }, { type: "scatter", mode: "lines", name: "ŷ=y", x: [Math.min(...evalR.predActual.map((p) => p[0])), Math.max(...evalR.predActual.map((p) => p[0]))], y: [Math.min(...evalR.predActual.map((p) => p[0])), Math.max(...evalR.predActual.map((p) => p[0]))], line: { color: th.muted, dash: "dash" }, hoverinfo: "skip" }] as never} layout={lay("predicted vs actual (test)", "actual", "predicted", { showlegend: true, legend: { orientation: "h", y: -0.2 }, height: 340 }) as never} style={{ height: 340, width: "100%" }} />}</>
                : <><div className="metric" style={{ maxWidth: 160, marginBottom: 14 }}><span className="v">{(evalR.acc * 100).toFixed(1)}%</span><span className="k">test accuracy</span></div>
                  {evalR.confusion && (() => { const cls = evalR.classes ?? []; const ann: Record<string, unknown>[] = []; for (let a = 0; a < cls.length; a++) for (let p = 0; p < cls.length; p++) ann.push({ x: cls[p], y: cls[a], text: String(evalR.confusion![a][p]), showarrow: false, font: { color: a === p ? "#eafff2" : "#ffe9e9", size: 15 } }); return <Plot data={[{ type: "heatmap", x: cls, y: cls, z: evalR.confusion, xgap: 4, ygap: 4, colorscale: [[0, th.plot], [1, "#3ecf7f"]], showscale: true, hoverinfo: "skip" }] as never} layout={{ ...lay("confusion matrix (test)", "predicted →", "actual ↓", { height: 340 }), annotations: ann } as never} style={{ height: 340, width: "100%" }} />; })()}</>}
              <label className="fld" style={{ marginTop: 16 }}>Predict one input</label>
              <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
                {data.featNames.slice(0, 6).map((f, j) => <div key={f}><label className="note">{f}</label><input type="number" step="any" value={testInput[j] ?? ""} onChange={(e) => setTestInput((t) => { const n = [...t]; n[j] = +e.target.value; return n; })} style={{ width: 90 }} /></div>)}
                <button className="btn sm" onClick={() => { const x = data.featNames.map((_, j) => testInput[j] ?? 0); if (netRef.current && scRef.current) { const o = predictVec(netRef.current, scaleRow(x, scRef.current)); if (data.task === "regression") setMsg(`Prediction: ${o[0].toFixed(3)}`); else { const c = o.indexOf(Math.max(...o)); setMsg(`Prediction: ${data.classes[c] ?? c} (${(Math.max(...o) * 100).toFixed(0)}% confident)`); } } }}>Predict</button>
              </div>
              <label className="fld" style={{ marginTop: 16 }}>Equivalent PyTorch</label>
              <pre className="codebox" style={{ maxHeight: 260, overflow: "auto", fontSize: 11.5 }}>{pyCode}</pre>
            </>}
            <div className="stepnav" style={{ marginTop: 14 }}><button className="btn ghost" onClick={() => setStep("train")}>← Back</button></div>
          </div>
        </div>
      )}
    </>
  );
}
