"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  parseCSV, colStats, buildMatrix, split, makeModel, predict,
  featureImportance, classificationMetrics, regressionMetrics, crossVal, crossValDetailed,
  decisionSurface, learningCurve, gdTrace, gdAnim, rootSplitMath,
  splitCounts, mean, std, treeDepth, countNodes, describe, applyStepsSnapshots, prepColTrace, scaleNumCol,
  type Dataset, type Task, type PrepStep, type TrainConfig, type ClsMetrics, type RegMetrics, type Snapshot,
  type FoldResult, type TreeNode, type BuiltData, type Model,
} from "@/lib/mlUtils";
import { sampleDatasets } from "@/lib/mlDatasets";
import { pickle } from "@/lib/pickle";
import Plot from "@/components/Plot";
import Katex from "@/components/Katex";
import {
  buildFigure, plotlyTheme, datasetInsights,
  SINGLE_NUM, SINGLE_CAT, COMPARE_CHARTS, type EdaSpec,
} from "@/lib/edaCharts";

type Step = "data" | "eda" | "prep" | "model" | "validation" | "train" | "deploy";
type ParamSpec = { name: string; type: "num" | "sel"; def: number | string; min?: number; max?: number; step?: number; opts?: string[] };

const TREE_PARAMS: ParamSpec[] = [{ name: "max_depth", type: "num", def: 5, min: 1, max: 20, step: 1 }, { name: "min_samples_split", type: "num", def: 2, min: 2, max: 40, step: 1 }];
const FOREST_PARAMS: ParamSpec[] = [{ name: "n_estimators", type: "num", def: 25, min: 5, max: 100, step: 5 }, ...TREE_PARAMS];
const MODELS: Record<Task, Record<string, ParamSpec[]>> = {
  classification: {
    LogisticRegression: [{ name: "C", type: "num", def: 1.0, min: 0.01, max: 10, step: 0.01 }, { name: "max_iter", type: "num", def: 300, min: 50, max: 1000, step: 50 }, { name: "learning_rate", type: "num", def: 0.2, min: 0.01, max: 1, step: 0.01 }],
    KNeighborsClassifier: [{ name: "n_neighbors", type: "num", def: 5, min: 1, max: 20, step: 1 }, { name: "weights", type: "sel", def: "uniform", opts: ["uniform", "distance"] }],
    GaussianNB: [],
    DecisionTree: TREE_PARAMS,
    RandomForest: FOREST_PARAMS,
  },
  regression: {
    LinearRegression: [{ name: "fit_intercept", type: "sel", def: "True", opts: ["True", "False"] }],
    Ridge: [{ name: "alpha", type: "num", def: 1.0, min: 0, max: 10, step: 0.1 }],
    KNeighborsRegressor: [{ name: "n_neighbors", type: "num", def: 5, min: 1, max: 20, step: 1 }, { name: "weights", type: "sel", def: "uniform", opts: ["uniform", "distance"] }],
    DecisionTree: TREE_PARAMS,
    RandomForest: FOREST_PARAMS,
  },
};
const MODEL_INFO: Record<string, { label: string; family: string; desc: string }> = {
  LogisticRegression: { label: "Logistic Regression", family: "Linear", desc: "Learns class boundaries via gradient descent. Fast, interpretable coefficients." },
  KNeighborsClassifier: { label: "K-Nearest Neighbors", family: "Instance", desc: "Predicts the majority label of the k closest points. No training — all work at predict time." },
  GaussianNB: { label: "Gaussian Naive Bayes", family: "Probabilistic", desc: "Assumes each feature is an independent Gaussian per class. Very fast baseline." },
  LinearRegression: { label: "Linear Regression", family: "Linear", desc: "Fits a line/plane by least squares via gradient descent." },
  Ridge: { label: "Ridge Regression", family: "Linear", desc: "Linear regression with an L2 penalty to curb overfitting." },
  KNeighborsRegressor: { label: "K-Nearest Neighbors", family: "Instance", desc: "Averages the target of the k nearest points." },
  DecisionTree: { label: "Decision Tree", family: "Tree", desc: "Recursively splits on feature thresholds into if/else rules. Highly interpretable." },
  RandomForest: { label: "Random Forest", family: "Ensemble", desc: "Many decision trees on bootstrapped samples, votes averaged. Robust & accurate." },
};
const OPS: Record<string, { methods: string[]; type: "any" | "num" | "cat"; hint: string }> = {
  "Impute missing": { methods: ["Mean", "Median", "Most frequent", "Constant", "Min", "Max", "Forward fill", "Backward fill", "Interpolate"], type: "any", hint: "Fill gaps (nulls) so no value is missing." },
  "Scale / normalize": { methods: ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileUniform", "None"], type: "num", hint: "Put numeric features on a comparable scale." },
  "Encode categorical": { methods: ["One-Hot", "Ordinal", "Frequency", "Count", "Binary"], type: "cat", hint: "Turn text categories into numbers a model can use." },
  "Handle outliers": { methods: ["IQR clip", "IQR replace", "Z-score clip", "Winsorize 5%"], type: "num", hint: "Tame extreme values that distort the model." },
  "Transform": { methods: ["Log", "Log1p", "Sqrt", "Square", "Cube root", "Reciprocal", "Absolute"], type: "num", hint: "Reshape a distribution (e.g. reduce skew)." },
  "Bin / discretize": { methods: ["Equal-width (5)", "Equal-freq (5)", "Equal-width (10)"], type: "num", hint: "Bucket a numeric column into ordered bins." },
  "Drop column": { methods: ["drop"], type: "any", hint: "Remove a column from the dataset entirely." },
};
const METHOD_DESC: Record<string, string> = {
  Mean: "replaces missing values with the column mean", Median: "replaces missing values with the median (robust to outliers)",
  "Most frequent": "replaces missing values with the most common value", Constant: "replaces missing values with a constant (0 / “missing”)",
  Min: "replaces missing values with the column minimum", Max: "replaces missing values with the column maximum",
  "Forward fill": "carries the last valid value forward into gaps", "Backward fill": "carries the next valid value backward into gaps",
  Interpolate: "linearly interpolates between known values",
  StandardScaler: "centers to mean 0, scales to unit variance (z-score)", MinMaxScaler: "rescales values into the 0–1 range",
  RobustScaler: "scales using median & IQR (robust to outliers)", MaxAbsScaler: "scales by the max absolute value into −1…1",
  QuantileUniform: "maps values to their uniform quantile rank (0–1)", None: "leaves values unchanged",
  "One-Hot": "creates a 0/1 column per category", Ordinal: "maps each category to an integer code",
  Frequency: "replaces each category with its proportion", Count: "replaces each category with its count",
  Binary: "encodes the category index in binary across bit-columns",
  "IQR clip": "clips values to the 1.5×IQR whiskers", "IQR replace": "replaces outliers with the median",
  "Z-score clip": "clips values beyond ±3 standard deviations", "Winsorize 5%": "clips to the 5th & 95th percentiles",
  Log: "applies log(1+|x|)", Log1p: "applies log(1+x)", Sqrt: "applies √|x|", Square: "squares each value",
  "Cube root": "applies the cube root", Reciprocal: "applies 1/x", Absolute: "takes the absolute value",
  "Equal-width (5)": "buckets into 5 equal-width bins", "Equal-freq (5)": "buckets into 5 equal-frequency (quantile) bins",
  "Equal-width (10)": "buckets into 10 equal-width bins", drop: "removes the column",
};
function describeStep(op: string, method: string): string {
  const d = METHOD_DESC[method];
  if (!d) return OPS[op]?.hint ?? op;
  return d.charAt(0).toUpperCase() + d.slice(1) + ".";
}
// Always returns a string so a stray NaN/Infinity never reaches the DOM as a raw number.
const fmtNum = (v: number): string => (Number.isFinite(v) ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : "—");
// Shared Plotly layout for the small train-step line/scatter charts.
/* eslint-disable @typescript-eslint/no-explicit-any */
function chartLayout(t: ReturnType<typeof plotlyTheme>, title: string, xlab: string, ylab: string): Record<string, any> {
  const axis = (lab: string) => ({ title: { text: lab, font: { size: 11, color: t.muted } }, gridcolor: t.grid, zerolinecolor: t.grid, linecolor: t.line, tickfont: { size: 10, color: t.muted } });
  return {
    paper_bgcolor: t.paper, plot_bgcolor: t.plot, autosize: true, showlegend: false,
    font: { family: "ui-sans-serif, system-ui, sans-serif", color: t.muted, size: 11 },
    margin: { l: 56, r: 20, t: 38, b: 46 },
    title: { text: title, font: { size: 13, color: t.text }, x: 0, xanchor: "left", xref: "paper" },
    xaxis: axis(xlab), yaxis: axis(ylab),
    hoverlabel: { font: { family: "ui-sans-serif, system-ui, sans-serif", size: 11 }, bordercolor: t.line },
  };
}
/* eslint-enable @typescript-eslint/no-explicit-any */

// Renders a fitted decision tree as an SVG diagram (nodes reveal top-down).
function TreeDiagram({ root, featureNames, classes, task }: { root: TreeNode; featureNames: string[]; classes: string[]; task: Task }) {
  const levelH = 68, leafW = 78;
  const nodes: { id: number; x: number; y: number; node: TreeNode; depth: number }[] = [];
  const edges: { fx: number; fy: number; tx: number; ty: number; label: string }[] = [];
  let leafX = 0, nextId = 0;
  const rec = (node: TreeNode, depth: number): { x: number; y: number } => {
    const id = nextId++;
    if (node.leaf) { const x = leafX * leafW + leafW / 2; leafX++; const nd = { id, x, y: depth * levelH + 24, node, depth }; nodes.push(nd); return nd; }
    const L = rec(node.left, depth + 1), R = rec(node.right, depth + 1);
    const x = (L.x + R.x) / 2, y = depth * levelH + 24; nodes.push({ id, x, y, node, depth });
    edges.push({ fx: x, fy: y, tx: L.x, ty: L.y, label: "≤" }); edges.push({ fx: x, fy: y, tx: R.x, ty: R.y, label: ">" });
    return { x, y };
  };
  rec(root, 0);
  const W = Math.max(leafX * leafW, 220), H = treeDepth(root) * levelH + 24;
  const label = (n: TreeNode) => n.leaf
    ? (task === "regression" ? Number(n.value).toFixed(2) : (classes[n.value] ?? String(n.value)))
    : `${(featureNames[n.feat] ?? "f" + n.feat).slice(0, 11)} ≤ ${n.thr.toFixed(2)}`;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={W} height={H} className="tree-svg" role="img" aria-label="decision tree">
        {edges.map((e, i) => (<line key={`e${i}`} x1={e.fx} y1={e.fy + 13} x2={e.tx} y2={e.ty - 13} className="tree-edge" />))}
        {edges.map((e, i) => (<text key={`t${i}`} x={(e.fx + e.tx) / 2 + (e.label === "≤" ? -6 : 6)} y={(e.fy + e.ty) / 2} className="tree-elab">{e.label}</text>))}
        {nodes.map((nd) => (
          <g key={nd.id} className="tree-node-g" style={{ animationDelay: `${nd.depth * 0.18}s` }}>
            <rect x={nd.x - 36} y={nd.y - 13} width={72} height={26} rx={6} className={nd.node.leaf ? "tree-leaf" : "tree-inner"} />
            <text x={nd.x} y={nd.y + 4} className="tree-lab">{label(nd.node)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

// Tiny live loss sparkline drawn up to `upto` points.
function Sparkline({ values, upto }: { values: number[]; upto: number }) {
  if (!values.length) return null;
  const W = 240, H = 58, n = values.length;
  const mn = Math.min(...values), mx = Math.max(...values), sp = (mx - mn) || 1;
  const k = Math.max(1, Math.min(upto, n));
  const pts = values.slice(0, k).map((v, i) => `${(i / (n - 1 || 1)) * W},${H - ((v - mn) / sp) * (H - 8) - 4}`).join(" ");
  return (<svg width="100%" viewBox={`0 0 ${W} ${H}`} className="spark" preserveAspectRatio="none"><polyline points={pts} className="spark-line" /></svg>);
}

export default function MlLab() {
  const [step, setStep] = useState<Step>("data");
  const [ds, setDs] = useState<Dataset | null>(null);
  const [dsName, setDsName] = useState("");
  const [dataTab, setDataTab] = useState<"sample" | "upload" | "db">("sample");
  const [sampleKey, setSampleKey] = useState("churn");
  const [dbUrl, setDbUrl] = useState("mysql://user:pass@host:3306/db");
  const [dbQ, setDbQ] = useState("SELECT * FROM customers LIMIT 500;");
  const [dbBusy, setDbBusy] = useState(false);
  const [msg, setMsg] = useState("");

  // data viewer
  const [viewMode, setViewMode] = useState<"head" | "tail" | "range">("head");
  const [nRows, setNRows] = useState(8);
  const [rFrom, setRFrom] = useState(0);
  const [rTo, setRTo] = useState(10);

  const [target, setTarget] = useState("");
  const [task, setTask] = useState<Task>("classification");
  const [features, setFeatures] = useState<string[]>([]);
  const [steps, setSteps] = useState<PrepStep[]>([]);
  const [algo, setAlgo] = useState("LogisticRegression");
  const [params, setParams] = useState<Record<string, string>>({});
  const [testSize, setTestSize] = useState(0.2);
  const [cvFolds, setCvFolds] = useState(5);

  // validation
  const [valSize, setValSize] = useState(0.2);
  const [valMethod, setValMethod] = useState<"kfold" | "holdout">("kfold");
  const [cvResult, setCvResult] = useState<FoldResult[]>([]);
  const [cvRunning, setCvRunning] = useState(false);
  const [cvShown, setCvShown] = useState(0);
  const [treeViz, setTreeViz] = useState<{ root: TreeNode; depth: number; nodes: number; nTrees?: number } | null>(null);

  // training playback (data → model, sample by sample)
  const [tpPlaying, setTpPlaying] = useState(false);
  const [tpIdx, setTpIdx] = useState(0);
  const [tpSpeed, setTpSpeed] = useState(500);
  // test the trained model
  const [trained, setTrained] = useState<{ featureNames: string[]; classes: string[] | null; algo: string; task: Task } | null>(null);
  const [testTab, setTestTab] = useState<"manual" | "sample">("manual");
  const [testInputs, setTestInputs] = useState<Record<string, string>>({});
  const [testOut, setTestOut] = useState<{ pred: string; actual?: string; ok?: boolean } | null>(null);
  const modelRef = useRef<Model | null>(null);

  // EDA explorer
  const [edaMode, setEdaMode] = useState<"single" | "compare">("single");
  const [uniCol, setUniCol] = useState("");
  const [uniChart, setUniChart] = useState("Histogram");
  const [bins, setBins] = useState(20);
  const [groupBy, setGroupBy] = useState("(none)");
  const [cmpChart, setCmpChart] = useState("Scatter");
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");
  const [colorCol, setColorCol] = useState("(none)");
  const [multiCols, setMultiCols] = useState<string[]>([]);
  const [trend, setTrend] = useState(true);

  // preprocessing player
  const [snaps, setSnaps] = useState<Snapshot[]>([]);
  const [ppIdx, setPpIdx] = useState(0);
  const [ppPlaying, setPpPlaying] = useState(false);
  const [ppSpeed, setPpSpeed] = useState(1500);
  const [processedCols, setProcessedCols] = useState<number>(0);

  const [result, setResult] = useState<{ metrics: ClsMetrics | RegMetrics; importance: { name: string; w: number }[] | null; cv: number[]; predActual?: [number, number][]; loss?: number[]; ms: number } | null>(null);
  const [training, setTraining] = useState(false);
  const [flowStep, setFlowStep] = useState(-1);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [mlMode, setMlMode] = useState<"package" | "maths">("package");
  const [animIdx, setAnimIdx] = useState(0);
  const [animPlaying, setAnimPlaying] = useState(false);
  const [sigZ, setSigZ] = useState(1);
  const [summaryCol, setSummaryCol] = useState("");
  const [prepCol, setPrepCol] = useState("");
  const [prepStepIdx, setPrepStepIdx] = useState(1);
  const [prepImputeMethod, setPrepImputeMethod] = useState("Mean");
  const [prepScaleMethod, setPrepScaleMethod] = useState("StandardScaler");
  const [prepEncodeMethod, setPrepEncodeMethod] = useState("One-Hot");
  const [prepStage, setPrepStage] = useState(0);
  const [prepStagePlaying, setPrepStagePlaying] = useState(false);
  const stageTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [modelStage, setModelStage] = useState(0);
  const [modelPlaying, setModelPlaying] = useState(false);
  const modelTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // decision boundary + learning curve + editable code
  const [dbF1, setDbF1] = useState("");
  const [dbF2, setDbF2] = useState("");
  const [dbSurf, setDbSurf] = useState<ReturnType<typeof decisionSurface>>(null);
  const [lcData, setLcData] = useState<{ n: number; train: number; test: number }[]>([]);
  const [lcFeats, setLcFeats] = useState<string[]>([]);
  const [codeDraft, setCodeDraft] = useState("");
  const [codeDirty, setCodeDirty] = useState(false);
  const [savedMsg, setSavedMsg] = useState("");

  const fileRef = useRef<HTMLInputElement>(null);
  const flowTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const desc = useMemo(() => (ds ? describe(ds) : []), [ds]);
  const insights = useMemo(() => (ds ? datasetInsights(ds, target) : []), [ds, target]);
  const edaFig = useMemo(() => {
    if (!ds) return null;
    const spec: EdaSpec = edaMode === "single"
      ? { mode: "single", uniCol, uniChart, bins, groupBy }
      : { mode: "compare", cmpChart, x: xCol, y: yCol, color: colorCol, cols: multiCols, trend };
    return buildFigure(ds, spec, plotlyTheme());
  }, [ds, edaMode, uniCol, uniChart, bins, groupBy, cmpChart, xCol, yCol, colorCol, multiCols, trend]);

  // Preprocessed numeric matrix — what actually goes into the model. Shown in the
  // Model & Validation steps and reused for cross-validation.
  const built = useMemo<BuiltData | null>(() => {
    if (!ds || !target || !features.length) return null;
    try { return buildMatrix(ds, features, target, task, steps); } catch { return null; }
  }, [ds, target, features, task, steps]);
  const modelKind = ["LogisticRegression", "LinearRegression", "Ridge"].includes(algo) ? "gd"
    : algo === "DecisionTree" ? "tree" : algo === "RandomForest" ? "forest"
    : algo === "GaussianNB" ? "gnb" : "knn";

  // ── load ──
  function loadCSV(text: string, name: string, tgt?: string, tk?: Task) {
    const parsed = parseCSV(text);
    if (parsed.columns.length < 2 || parsed.nrows < 4) { setMsg("Need at least 2 columns and a few rows."); return; }
    setDs(parsed); setDsName(name); setMsg(""); setResult(null); setSnaps([]);
    const t = tgt && parsed.columns.some((c) => c.name === tgt) ? tgt : parsed.columns[parsed.columns.length - 1].name;
    const tcol = parsed.columns.find((c) => c.name === t)!;
    const theTask: Task = tk ?? (tcol.type === "cat" ? "classification" : "regression");
    setTarget(t); setTask(theTask);
    setFeatures(parsed.columns.filter((c) => c.name !== t).map((c) => c.name));
    const nums = parsed.columns.filter((c) => c.type === "num" && c.name !== t).map((c) => c.name);
    const cats = parsed.columns.filter((c) => c.type === "cat" && c.name !== t).map((c) => c.name);
    const def: PrepStep[] = [];
    if (nums.length) { def.push({ op: "Impute missing", cols: nums, method: "Mean" }); def.push({ op: "Scale / normalize", cols: nums, method: "StandardScaler" }); }
    if (cats.length) { def.push({ op: "Impute missing", cols: cats, method: "Most frequent" }); def.push({ op: "Encode categorical", cols: cats, method: "One-Hot" }); }
    setSteps(def);
    const a = Object.keys(MODELS[theTask])[0]; setAlgo(a); setParamsFor(theTask, a);
    // EDA defaults
    const first = parsed.columns[0];
    setEdaMode("single"); setUniCol(first.name); setUniChart(first.type === "num" ? "Histogram" : "Bar (counts)");
    setGroupBy("(none)"); setColorCol("(none)"); setBins(20); setCmpChart("Scatter");
    const nc = parsed.columns.filter((c) => c.type === "num");
    setXCol(nc[0]?.name || first.name); setYCol(nc[1]?.name || nc[0]?.name || first.name);
    setMultiCols(nc.slice(0, 4).map((c) => c.name));
    setRFrom(0); setRTo(Math.min(10, parsed.nrows));
  }
  function setParamsFor(tk: Task, a: string) { const p: Record<string, string> = {}; MODELS[tk][a].forEach((s) => { p[s.name] = String(s.def); }); setParams(p); }
  // Change the target column: re-infer the task, reset features + downstream results.
  function pickTarget(name: string) {
    if (!ds) return;
    const col = ds.columns.find((c) => c.name === name); if (!col) return;
    const uniq = new Set(col.values.filter((v) => v != null).map(String)).size;
    const tk: Task = (col.type === "cat" || uniq <= 12) ? "classification" : "regression";
    setTarget(name); setTask(tk);
    const a = Object.keys(MODELS[tk])[0]; setAlgo(a); setParamsFor(tk, a);
    setFeatures(ds.columns.filter((c) => c.name !== name).map((c) => c.name));
    setResult(null); setDbSurf(null); setLcData([]); setLcFeats([]);
  }
  useEffect(() => { loadCSV(sampleDatasets().find((d) => d.key === "churn")!.csv, "churn sample", "churn", "classification"); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // Load a saved build when opened from My Projects (?project=<id>). Only the
  // config is stored (not raw data), so we re-seed a sample by name and re-apply.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("project");
    if (!id) return;
    fetch(`/api/projects?id=${id}`).then((r) => r.json()).then(({ project }) => {
      const c = project?.config; if (!c) return;
      const s = sampleDatasets().find((d) => d.label === c.dsName || d.key === c.dsName);
      if (!s) { setMsg(`Loaded settings for "${project.name}" (${c.algo} · target "${c.target}"). This build used custom-uploaded data — re-load your file in step 1 to run it.`); return; }
      loadCSV(s.csv, s.label, c.target, c.task);
      if (Array.isArray(c.features)) setFeatures(c.features);
      if (Array.isArray(c.steps)) setSteps(c.steps);
      if (c.algo) setAlgo(c.algo);
      if (c.params) setParams(c.params);
      if (c.testSize != null) setTestSize(c.testSize);
      if (c.cvFolds != null) setCvFolds(c.cvFolds);
    }).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  function loadSample() { const s = sampleDatasets().find((d) => d.key === sampleKey)!; loadCSV(s.csv, s.label, s.target, s.task); }
  function onFile(e: React.ChangeEvent<HTMLInputElement>) { const f = e.target.files?.[0]; if (!f) return; const r = new FileReader(); r.onload = () => loadCSV(String(r.result || ""), f.name); r.readAsText(f); e.target.value = ""; }
  async function runDbQuery() { setDbBusy(true); setMsg(""); try { const r = await fetch("/api/ml/query", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: dbUrl, query: dbQ }) }); const j = await r.json(); if (!r.ok) throw new Error(j.error || "query failed"); loadCSV(j.csv, "db query result"); } catch (e) { setMsg((e as Error).message); } finally { setDbBusy(false); } }

  // rows to display
  const viewRows = useMemo(() => {
    if (!ds) return [] as number[];
    if (viewMode === "head") return Array.from({ length: Math.min(nRows, ds.nrows) }, (_, i) => i);
    if (viewMode === "tail") return Array.from({ length: Math.min(nRows, ds.nrows) }, (_, i) => ds.nrows - Math.min(nRows, ds.nrows) + i);
    const a = Math.max(0, Math.min(rFrom, ds.nrows - 1)), b = Math.max(a + 1, Math.min(rTo, ds.nrows));
    return Array.from({ length: b - a }, (_, i) => a + i);
  }, [ds, viewMode, nRows, rFrom, rTo]);

  // ── EDA chart selection helpers ──
  function pickUniCol(name: string) {
    const c = ds?.columns.find((x) => x.name === name); if (!c) return;
    setUniCol(name); setUniChart(c.type === "num" ? "Histogram" : "Bar (counts)"); setGroupBy("(none)");
  }
  function pickCmpChart(v: string) {
    if (!ds) { setCmpChart(v); return; }
    const nums = ds.columns.filter((c) => c.type === "num").map((c) => c.name);
    const cats = ds.columns.filter((c) => c.type === "cat").map((c) => c.name);
    setCmpChart(v);
    if (["Box by group", "Violin by group"].includes(v)) { setYCol(nums[0] || ""); setXCol(cats[0] || ""); }
    else if (["Grouped bar", "Stacked bar"].includes(v)) { setXCol(cats[0] || ""); setYCol("(count)"); setColorCol("(none)"); }
    else if (["Scatter", "Line", "2D density"].includes(v)) { setXCol(nums[0] || ""); setYCol(nums[1] || nums[0] || ""); }
  }

  // ── preprocessing ──
  const [prepOp, setPrepOp] = useState("Impute missing");
  const [prepMethod, setPrepMethod] = useState("Mean");
  const [prepCols, setPrepCols] = useState<string[]>([]);
  const opSpec = OPS[prepOp];
  const eligibleCols = ds ? ds.columns.filter((c) => c.name !== target && (opSpec.type === "any" || c.type === opSpec.type)).map((c) => c.name) : [];
  function addStep() { if (!prepCols.length) return; setSteps((s) => [...s, { op: prepOp, cols: prepCols, method: prepMethod }]); setPrepCols([]); }
  function runPreprocessing() {
    if (!ds) return;
    const { snapshots, finalColumns } = applyStepsSnapshots(ds, steps.filter((s) => !s.cols.includes(target)), 6);
    setSnaps(snapshots); setProcessedCols(finalColumns.length); setPpIdx(0); setPpPlaying(snapshots.length > 1);
  }
  // Step-by-step player: advance one snapshot every `ppSpeed` ms while playing.
  useEffect(() => {
    if (!ppPlaying || snaps.length === 0) return;
    if (ppIdx >= snaps.length - 1) { setPpPlaying(false); return; }
    const t = setTimeout(() => setPpIdx((i) => Math.min(i + 1, snaps.length - 1)), ppSpeed);
    return () => clearTimeout(t);
  }, [ppPlaying, ppIdx, ppSpeed, snaps.length]);
  useEffect(() => () => { if (flowTimer.current) clearInterval(flowTimer.current); }, []);

  // ── validation: run k-fold cross-validation, then reveal each fold in turn ──
  function runCrossVal() {
    if (!built) { setMsg("Pick a target & at least one feature first (Model step)."); return; }
    setMsg(""); setCvRunning(true); setCvResult([]); setCvShown(0);
    setTimeout(() => {
      try {
        const cfg: TrainConfig = { task, algo, params, testSize, cvFolds };
        const res = crossValDetailed(cfg, built.X, built.y, built.classes?.length || 0);
        setCvResult(res); setCvRunning(false);
        let i = 0; const iv = setInterval(() => { i++; setCvShown(i); if (i >= res.length) clearInterval(iv); }, 480);
      } catch (e) { setMsg("Cross-validation error: " + (e as Error).message); setCvRunning(false); }
    }, 60);
  }
  const cvAllScores = cvResult.map((f) => f.score);

  // ── train ──
  function buildCode(): string {
    const num = ds ? ds.columns.filter((c) => c.type === "num" && features.includes(c.name) && c.name !== target).map((c) => c.name) : [];
    const cat = ds ? ds.columns.filter((c) => c.type === "cat" && features.includes(c.name) && c.name !== target).map((c) => c.name) : [];
    const scaleM = steps.find((s) => s.op === "Scale / normalize")?.method || "StandardScaler";
    const scaleMap: Record<string, string> = { StandardScaler: "StandardScaler()", MinMaxScaler: "MinMaxScaler()", RobustScaler: "RobustScaler()", MaxAbsScaler: "MaxAbsScaler()", QuantileUniform: "QuantileTransformer(output_distribution='uniform')", None: "'passthrough'" };
    const scaler = scaleMap[scaleM] || "StandardScaler()";
    const encM = steps.find((s) => s.op === "Encode categorical")?.method;
    const encoder = encM === "Ordinal" ? "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)" : "OneHotEncoder(handle_unknown='ignore')";
    const impM = ({ Mean: "mean", Median: "median", "Most frequent": "most_frequent", Constant: "constant", Min: "median", Max: "median", "Forward fill": "median", "Backward fill": "median", Interpolate: "median" } as Record<string, string>)[steps.find((s) => s.op === "Impute missing" && ds?.columns.find((c) => c.name === s.cols[0])?.type === "num")?.method || "Mean"] || "mean";
    const advanced = steps.filter((s) => ["Handle outliers", "Transform", "Bin / discretize"].includes(s.op));
    const sk: Record<string, { imp: string; cls: string }> = {
      LogisticRegression: { imp: "from sklearn.linear_model import LogisticRegression", cls: "LogisticRegression" },
      KNeighborsClassifier: { imp: "from sklearn.neighbors import KNeighborsClassifier", cls: "KNeighborsClassifier" },
      GaussianNB: { imp: "from sklearn.naive_bayes import GaussianNB", cls: "GaussianNB" },
      LinearRegression: { imp: "from sklearn.linear_model import LinearRegression", cls: "LinearRegression" },
      Ridge: { imp: "from sklearn.linear_model import Ridge", cls: "Ridge" },
      KNeighborsRegressor: { imp: "from sklearn.neighbors import KNeighborsRegressor", cls: "KNeighborsRegressor" },
      DecisionTree: { imp: `from sklearn.tree import DecisionTree${task === "classification" ? "Classifier" : "Regressor"}`, cls: `DecisionTree${task === "classification" ? "Classifier" : "Regressor"}` },
      RandomForest: { imp: `from sklearn.ensemble import RandomForest${task === "classification" ? "Classifier" : "Regressor"}`, cls: `RandomForest${task === "classification" ? "Classifier" : "Regressor"}` },
    };
    const m = sk[algo] || sk.LogisticRegression;
    const skParam: Record<string, string> = { max_iter: "max_iter", C: "C", alpha: "alpha", n_neighbors: "n_neighbors", weights: "weights", max_depth: "max_depth", min_samples_split: "min_samples_split", n_estimators: "n_estimators", fit_intercept: "fit_intercept" };
    const p = Object.entries(params).filter(([k]) => skParam[k]).map(([k, v]) => `${skParam[k]}=${/^-?[0-9.]+$/.test(v) ? v : (["True", "False", "None"].includes(v) ? v : `"${v}"`)}`).concat(algo === "DecisionTree" || algo === "RandomForest" ? ["random_state=42"] : []).join(", ");
    const scoring = task === "classification" ? "accuracy" : "r2";
    const metricsImp = task === "classification"
      ? "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix"
      : "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error";
    const evalBlock = task === "classification"
      ? `pred = pipe.predict(X_test)
print("accuracy :", round(accuracy_score(y_test, pred), 3))
print("precision:", round(precision_score(y_test, pred, average="macro", zero_division=0), 3))
print("recall   :", round(recall_score(y_test, pred, average="macro", zero_division=0), 3))
print("f1       :", round(f1_score(y_test, pred, average="macro", zero_division=0), 3))
print("confusion matrix:\\n", confusion_matrix(y_test, pred))`
      : `pred = pipe.predict(X_test)
print("R2  :", round(r2_score(y_test, pred), 3))
print("MAE :", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3))`;
    const advNote = advanced.length ? `\n# NOTE: your pipeline also has advanced steps handled in-lab:\n${advanced.map((s) => `#   • ${s.op} (${s.method}) on ${s.cols.join(", ")}`).join("\n")}\n#   In sklearn add e.g. PowerTransformer / KBinsDiscretizer / a custom clipper.` : "";
    return `# ══ AI Workbench · complete ML workflow (data → result) ══
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import ${scaleM === "None" ? "OneHotEncoder, OrdinalEncoder" : `${scaleM === "QuantileUniform" ? "QuantileTransformer" : scaleM}, OneHotEncoder, OrdinalEncoder`}
${m.imp}
${metricsImp}

# 1) Load data
df = pd.read_csv("data.csv")
num = ${JSON.stringify(num)}
cat = ${JSON.stringify(cat)}
X, y = df[num + cat], df["${target}"]

# 2) Hold-out split (test set kept untouched until the very end)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=${testSize}, random_state=42${task === "classification" ? ", stratify=y" : ""})
${advNote}
# 3) Preprocess (impute → scale numeric · impute → encode categorical)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="${impM}")),
                      ("sc", ${scaler})]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", ${encoder})]), cat),
])

# 4) Model + full pipeline
model = ${m.cls}(${p})
pipe = Pipeline([("pre", pre), ("model", model)])

# 5) Cross-validate (${cvFolds}-fold) BEFORE the final fit
cv = cross_val_score(pipe, X_train, y_train, cv=${cvFolds}, scoring="${scoring}")
print("CV ${scoring}:", np.round(cv, 3), "| mean:", round(cv.mean(), 3), "± ", round(cv.std(), 3))

# 6) Fit on the training set
pipe.fit(X_train, y_train)

# 7) Evaluate on the held-out test set
${evalBlock}`;
  }
  function runTrain() {
    if (!ds) return; setTraining(true); setStep("train"); setResult(null); setFlowStep(0); setTreeViz(null);
    if (flowTimer.current) clearInterval(flowTimer.current);
    let fs = 0; flowTimer.current = setInterval(() => { fs++; setFlowStep(fs); if (fs >= 6) { if (flowTimer.current) clearInterval(flowTimer.current); } }, 420);
    setTimeout(() => {
      try {
        const b = buildMatrix(ds, features, target, task, steps);
        const nClasses = b.classes?.length || 0;
        const { Xtr, ytr, Xte, yte } = split(b.X, b.y, testSize);
        const cfg: TrainConfig = { task, algo, params, testSize, cvFolds };
        const t0 = performance.now(); const model = makeModel(cfg, Xtr, ytr, nClasses); const pred = predict(model, Xte); const ms = Math.round(performance.now() - t0);
        const importance = featureImportance(model, b.featureNames); const cv = crossVal(cfg, b.X, b.y, nClasses);
        const loss = (model.kind === "logreg" || model.kind === "linear") ? model.loss : undefined;
        modelRef.current = model; setTrained({ featureNames: b.featureNames, classes: b.classes ?? null, algo, task });
        if (model.kind === "tree") setTreeViz({ root: model.root, depth: treeDepth(model.root), nodes: countNodes(model.root) });
        else if (model.kind === "forest") setTreeViz({ root: model.trees[0], depth: treeDepth(model.trees[0]), nodes: countNodes(model.trees[0]), nTrees: model.nTrees });
        if (task === "classification") setResult({ metrics: classificationMetrics(yte, pred.map((p) => Math.round(p)), b.classes!), importance, cv, loss, ms });
        else setResult({ metrics: regressionMetrics(yte, pred), importance, cv, predActual: yte.map((a, i) => [a, pred[i]] as [number, number]), loss, ms });
      } catch (e) { setMsg("Training error: " + (e as Error).message); }
      setTraining(false); setTimeout(() => setFlowStep(7), 500);
    }, 60);
  }
  // On a fresh result: start the data→model playback and seed the tester fields.
  useEffect(() => {
    if (step !== "train" || !result || !ds) return;
    setTpIdx(0); setTpPlaying(true);
    const init: Record<string, string> = {};
    features.forEach((f) => {
      const c = ds.columns.find((x) => x.name === f); if (!c) return;
      if (c.type === "num") { const v = c.values.filter((x): x is number => x != null).map(Number).sort((a, b) => a - b); init[f] = String(v.length ? v[Math.floor(v.length / 2)] : 0); }
      else { const s = colStats(c) as { top: [string, number][] }; init[f] = String(s.top[0]?.[0] ?? ""); }
    });
    setTestInputs(init); setTestOut(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);
  // training-playback ticker
  const TP_SHOWN = ds ? Math.min(18, ds.nrows) : 0;
  const TP_EPOCHS = result?.loss ? Math.min(8, result.loss.length) : 3;
  const TP_TOTAL = Math.max(1, TP_SHOWN * TP_EPOCHS);
  useEffect(() => {
    if (!tpPlaying) return;
    if (tpIdx >= TP_TOTAL - 1) { setTpPlaying(false); return; }
    const t = setTimeout(() => setTpIdx((i) => i + 1), tpSpeed);
    return () => clearTimeout(t);
  }, [tpPlaying, tpIdx, tpSpeed, TP_TOTAL]);
  // Once the learning curve has been computed, toggling the feature subset
  // recomputes it live so the numbers + verdict update as you pick columns.
  useEffect(() => { if (lcData.length && ds && result) runLC(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lcFeats]);
  // Frames for the gradient-descent animation (only for GD models, in maths mode).
  const gdAnimData = useMemo(() => {
    if (mlMode !== "maths" || step !== "train" || !ds || !result) return null;
    if (!["LogisticRegression", "LinearRegression", "Ridge"].includes(algo)) return null;
    try {
      const nums = features.filter((f) => ds.columns.find((c) => c.name === f)?.type === "num");
      const a = dbF1 && nums.includes(dbF1) ? dbF1 : nums[0];
      if (task === "classification") { if (nums.length < 2) return null; const b = dbF2 && nums.includes(dbF2) && dbF2 !== a ? dbF2 : (nums.find((x) => x !== a) || nums[1]); return gdAnim(ds, target, a, b, cfgNow()); }
      return nums.length ? gdAnim(ds, target, a, "", cfgNow()) : null;
    } catch { return null; }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mlMode, step, ds, target, features, task, algo, params, dbF1, dbF2, result]);
  useEffect(() => { setAnimIdx(0); setAnimPlaying(false); }, [gdAnimData]);
  const resetStage = () => { setPrepStage(0); setPrepStagePlaying(false); if (stageTimer.current) { clearInterval(stageTimer.current); stageTimer.current = null; } };
  // When you move to a different step/column, restart the animation and default the method from that step.
  useEffect(() => { resetStage(); const st = steps[Math.min(Math.max(1, prepStepIdx), steps.length || 1) - 1]; if (st && st.op === "Impute missing") setPrepImputeMethod(st.method); else if (st && st.op === "Scale / normalize") setPrepScaleMethod(st.method); else if (st && st.op === "Encode categorical") setPrepEncodeMethod(st.method); }, [prepStepIdx, prepCol, steps]);
  // When you pick a different method, restart the animation (but keep the selection).
  useEffect(() => { resetStage(); }, [prepImputeMethod, prepScaleMethod, prepEncodeMethod]);
  useEffect(() => () => { if (stageTimer.current) clearInterval(stageTimer.current); }, []);
  // Model-step walkthrough resets when the algorithm / task / features change.
  useEffect(() => { setModelStage(0); setModelPlaying(false); if (modelTimer.current) { clearInterval(modelTimer.current); modelTimer.current = null; } }, [algo, task, features, target]);
  useEffect(() => () => { if (modelTimer.current) clearInterval(modelTimer.current); }, []);
  useEffect(() => {
    if (!animPlaying || !gdAnimData) return;
    if (animIdx >= gdAnimData.frames.length - 1) { setAnimPlaying(false); return; }
    const t = setTimeout(() => setAnimIdx((i) => i + 1), 520);
    return () => clearTimeout(t);
  }, [animPlaying, animIdx, gdAnimData]);

  // Run the fitted model on a single raw record (same preprocessing → predict).
  function predictRow(inputs: Record<string, string>, actual?: string) {
    if (!ds || !modelRef.current || !trained) return;
    const cols = ds.columns.map((c) => {
      let val: number | string | null;
      if (c.name === target) val = null;
      else { const raw = inputs[c.name]; val = raw === undefined || raw === "" ? null : (c.type === "num" ? Number(raw) : raw); }
      return { name: c.name, type: c.type, values: [...c.values, val] };
    });
    const aug: Dataset = { nrows: ds.nrows + 1, columns: cols };
    const b2 = buildMatrix(aug, features, target, task, steps);
    const p = predict(modelRef.current, [b2.X[b2.X.length - 1]])[0];
    const pred = trained.classes ? (trained.classes[Math.round(p)] ?? "?") : p.toFixed(3);
    const ok = actual != null && trained.classes ? String(pred) === String(actual) : undefined;
    setTestOut({ pred, actual, ok });
  }
  function randomizeInputs() {
    if (!ds) return;
    const r = Math.floor(Math.random() * ds.nrows);
    const init: Record<string, string> = {};
    ds.columns.forEach((c) => { if (c.name !== target) { const v = c.values[r]; init[c.name] = v == null ? "" : String(v); } });
    setTestInputs(init);
    const actual = ds.columns.find((c) => c.name === target)?.values[r];
    predictRow(init, actual == null ? undefined : String(actual));
  }
  // loss curve + predicted-vs-actual as clean Plotly figures (theme-aware)
  const lossFig = useMemo(() => {
    if (!result?.loss?.length) return null; const t = plotlyTheme(); const L = result.loss;
    return { data: [{ type: "scatter", mode: "lines", x: L.map((_, i) => i), y: L, line: { color: t.accent, width: 2.5, shape: "spline" }, fill: "tozeroy", fillcolor: t.accent + "22", hovertemplate: "epoch %{x}<br>loss %{y:.4f}<extra></extra>" }], layout: chartLayout(t, "Training loss per epoch", "recorded epoch", "loss") };
  }, [result]);
  const pvaFig = useMemo(() => {
    if (!result?.predActual) return null; const t = plotlyTheme();
    const xs = result.predActual.map((p) => p[0]), ys = result.predActual.map((p) => p[1]);
    const all = [...xs, ...ys], mn = Math.min(...all), mx = Math.max(...all);
    return { data: [
      { type: "scatter", mode: "lines", x: [mn, mx], y: [mn, mx], line: { color: t.muted, width: 1.5, dash: "dash" }, hoverinfo: "skip", name: "ideal" },
      { type: "scatter", mode: "markers", x: xs, y: ys, marker: { color: t.accent, size: 7, opacity: 0.6 }, hovertemplate: "actual %{x:.2f}<br>pred %{y:.2f}<extra></extra>", name: "points" },
    ], layout: chartLayout(t, "Predicted vs actual", "actual", "predicted") };
  }, [result]);

  // ════════════ FROM-SCRATCH (MATHS) MODE ════════════
  const colNumVals = (name: string) => (ds?.columns.find((c) => c.name === name)?.values.filter((v) => v != null).map(Number) ?? []);
  const mCard = (title: string, body: React.ReactNode) => <div className="card math-card" style={{ marginBottom: 16 }}><div className="card-h"><span className="t">🧮 The maths — {title}</span></div><div className="card-b">{body}</div></div>;
  const PAL_ML = ["#5b7cff", "#f59e0b", "#3ecf7f", "#ef4444", "#a855f7", "#22b8cf"];
  // What each symbol in a formula means + the value it takes on this data.
  const varLegend = (items: { sym: string; desc: string; how?: string; val?: string }[]) => (
    <div className="var-legend"><div className="vl-title">what each symbol means · from your data</div>{items.map((it, i) => <div key={i} className="vl-row"><span className="vl-sym"><Katex tex={it.sym} /></span><span className="vl-desc">{it.desc}{it.how ? <span className="vl-how"> — {it.how}</span> : null}</span>{it.val != null && <span className="vl-val mono">{it.val}</span>}</div>)}</div>
  );
  // Discrete step controls (formula line → then one cell at a time) so it's easy to follow.
  function stagePlay(maxStage: number) {
    if (stageTimer.current) { clearInterval(stageTimer.current); stageTimer.current = null; setPrepStagePlaying(false); return; }
    setPrepStagePlaying(true);
    stageTimer.current = setInterval(() => { setPrepStage((s) => { if (s >= maxStage) { if (stageTimer.current) { clearInterval(stageTimer.current); stageTimer.current = null; } setPrepStagePlaying(false); return s; } return s + 1; }); }, 950);
  }
  const stageStop = () => { if (stageTimer.current) { clearInterval(stageTimer.current); stageTimer.current = null; } setPrepStagePlaying(false); };
  const stageControls = (maxStage: number) => (
    <div className="prep-ctl">
      <button className="btn ghost sm" disabled={prepStage <= 0} onClick={() => { stageStop(); setPrepStage((s) => Math.max(0, s - 1)); }}>← back</button>
      <button className="btn sm" disabled={prepStage >= maxStage} onClick={() => { stageStop(); setPrepStage((s) => Math.min(maxStage, s + 1)); }}>step →</button>
      <button className="btn ghost sm" onClick={() => stagePlay(maxStage)}>{prepStagePlaying ? "⏸ pause" : "▶ auto"}</button>
      <button className="btn ghost sm" onClick={() => { stageStop(); setPrepStage(0); }}>↺ reset</button>
      <span className="note mono" style={{ marginLeft: "auto" }}>stage {Math.min(prepStage, maxStage) + 1} / {maxStage + 1}</span>
    </div>
  );

  // Step controls for the model walkthrough (own timer so it never fights the prep card).
  function modelStagePlay(maxStage: number) {
    if (modelTimer.current) { clearInterval(modelTimer.current); modelTimer.current = null; setModelPlaying(false); return; }
    setModelPlaying(true);
    modelTimer.current = setInterval(() => { setModelStage((s) => { if (s >= maxStage) { if (modelTimer.current) { clearInterval(modelTimer.current); modelTimer.current = null; } setModelPlaying(false); return s; } return s + 1; }); }, 1100);
  }
  const modelStageStop = () => { if (modelTimer.current) { clearInterval(modelTimer.current); modelTimer.current = null; } setModelPlaying(false); };
  const modelStageControls = (maxStage: number) => (
    <div className="prep-ctl">
      <button className="btn ghost sm" disabled={modelStage <= 0} onClick={() => { modelStageStop(); setModelStage((s) => Math.max(0, s - 1)); }}>← back</button>
      <button className="btn sm" disabled={modelStage >= maxStage} onClick={() => { modelStageStop(); setModelStage((s) => Math.min(maxStage, s + 1)); }}>step →</button>
      <button className="btn ghost sm" onClick={() => modelStagePlay(maxStage)}>{modelPlaying ? "⏸ pause" : "▶ auto"}</button>
      <button className="btn ghost sm" onClick={() => { modelStageStop(); setModelStage(0); }}>↺ reset</button>
      <span className="note mono" style={{ marginLeft: "auto" }}>stage {Math.min(modelStage, maxStage) + 1} / {maxStage + 1}</span>
    </div>
  );
  // Animated gradient descent: play the boundary/line improving as the loss drops.
  function gdAnimView() {
    if (!gdAnimData) return null;
    const A = gdAnimData; const fr = A.frames[Math.min(animIdx, A.frames.length - 1)]; const t = plotlyTheme();
    let data: Record<string, unknown>[]; let title: string;
    if (!A.reg) {
      const K = A.classes.length; const colorscale: [number, string][] = [];
      for (let i = 0; i < K; i++) { const c = PAL_ML[i % PAL_ML.length]; colorscale.push([i / K, c]); colorscale.push([(i + 1) / K, c]); }
      data = [
        { type: "heatmap", x: A.xs, y: A.ys, z: fr.z, showscale: false, colorscale, zmin: -0.5, zmax: K - 0.5, opacity: 0.4, hoverinfo: "skip" },
        ...A.classes.map((cl, ci) => ({ type: "scatter", mode: "markers", name: cl, x: A.points.filter((p) => p.c === ci).map((p) => p.x), y: A.points.filter((p) => p.c === ci).map((p) => p.y), marker: { size: 6, color: PAL_ML[ci % PAL_ML.length], line: { width: 1, color: t.paper } } })),
      ];
      title = `epoch ${fr.ep} · loss ${fr.loss.toFixed(3)}`;
    } else {
      data = [
        { type: "scatter", mode: "markers", name: "data", x: A.points.map((p) => p.x), y: A.points.map((p) => p.y), marker: { size: 6, color: "#5b7cff", opacity: 0.55 } },
        { type: "scatter", mode: "lines", name: "fit", x: A.xs, y: fr.line, line: { color: "#f59e0b", width: 3 } },
      ];
      title = `epoch ${fr.ep} · MSE ${fr.loss.toFixed(2)}`;
    }
    const layout = { ...chartLayout(t, title, dbF1 || "x", A.reg ? target : (dbF2 || "y")), showlegend: false };
    return <div style={{ marginTop: 10 }}>
      <Plot data={data} layout={layout} style={{ height: 340 }} />
      <div className="row" style={{ gap: 10, alignItems: "center", marginTop: 8 }}>
        <button className="btn sm" onClick={() => { if (animIdx >= A.frames.length - 1) setAnimIdx(0); setAnimPlaying((p) => !p); }}>{animPlaying ? "⏸ Pause" : "▶ Play training"}</button>
        <input type="range" min={0} max={A.frames.length - 1} value={Math.min(animIdx, A.frames.length - 1)} onChange={(e) => { setAnimPlaying(false); setAnimIdx(+e.target.value); }} style={{ flex: 1 }} />
        <span className="mono note">epoch {fr.ep}</span>
      </div>
      <div className="note" style={{ marginTop: 4 }}>{A.reg ? "Watch the line shift & tilt to minimise squared error as w updates each epoch." : "Watch the coloured regions bend to separate the classes as the weights update — gradient descent minimising the loss, live."}</div>
    </div>;
  }
  // Interactive sigmoid: drag z, see the probability.
  function sigmoidView() {
    const t = plotlyTheme(); const xs: number[] = [], ys: number[] = [];
    for (let z = -6; z <= 6.001; z += 0.2) { xs.push(z); ys.push(1 / (1 + Math.exp(-z))); }
    const pr = 1 / (1 + Math.exp(-sigZ));
    const data = [
      { type: "scatter", mode: "lines", x: xs, y: ys, line: { color: "#5b7cff", width: 2.5 }, hoverinfo: "skip" },
      { type: "scatter", mode: "markers", x: [sigZ], y: [pr], marker: { size: 11, color: "#f59e0b" } },
    ];
    const layout = { ...chartLayout(t, "σ(z) squashes the score into a probability", "z = w·x + b", "probability"), showlegend: false, shapes: [{ type: "line", x0: -6, x1: 6, y0: 0.5, y1: 0.5, line: { color: t.grid, dash: "dot", width: 1 } }] };
    return <div style={{ marginTop: 12 }}>
      <label className="fld">Interactive — drag the score z</label>
      <Plot data={data} layout={layout} style={{ height: 230 }} />
      <div className="row" style={{ gap: 10, alignItems: "center", marginTop: 6 }}>
        <span className="note">z</span><input type="range" min={-6} max={6} step={0.1} value={sigZ} onChange={(e) => setSigZ(+e.target.value)} style={{ flex: 1 }} />
        <span className="mono"><Katex tex={`\\sigma(${sigZ.toFixed(1)}) = ${pr.toFixed(3)}`} /></span>
      </div>
      <div className="note" style={{ marginTop: 4 }}>Big positive score → probability near 1, big negative → near 0, z = 0 → exactly 0.5 (the decision line).</div>
    </div>;
  }
  const quart = (bn: number[]) => { const s = [...bn].sort((a, b) => a - b); const q = (p: number) => { const i = (s.length - 1) * p, lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); }; return { med: q(0.5), q1: q(0.25), q3: q(0.75) }; };
  function scaleStages(s: PrepStep, bn: number[]): { tex: string; at: number }[] {
    if (!bn.length) return [];
    const mu = mean(bn), sd = std(bn) || 1, mn = Math.min(...bn), mx = Math.max(...bn), x0 = bn[0];
    const { med, q1, q3 } = quart(bn); const iqr = (q3 - q1) || 1, maxabs = Math.max(...bn.map(Math.abs)) || 1;
    if (s.op === "Scale / normalize") {
      if (s.method === "StandardScaler") return [{ tex: `z=\\frac{x-\\mu}{\\sigma}`, at: 0 }, { tex: `z=\\frac{${x0.toFixed(1)}-${mu.toFixed(1)}}{${sd.toFixed(1)}}`, at: 0.15 }, { tex: `z=\\frac{${(x0 - mu).toFixed(1)}}{${sd.toFixed(1)}}`, at: 0.45 }, { tex: `z=${((x0 - mu) / sd).toFixed(2)}`, at: 0.75 }];
      if (s.method === "MinMaxScaler") return [{ tex: `z=\\frac{x-\\min}{\\max-\\min}`, at: 0 }, { tex: `z=\\frac{${x0.toFixed(1)}-${mn.toFixed(1)}}{${(mx - mn).toFixed(1)}}`, at: 0.3 }, { tex: `z=${((x0 - mn) / ((mx - mn) || 1)).toFixed(2)}`, at: 0.7 }];
      if (s.method === "RobustScaler") return [{ tex: `z=\\frac{x-\\text{median}}{\\text{IQR}}`, at: 0 }, { tex: `z=\\frac{${x0.toFixed(1)}-${med.toFixed(1)}}{${iqr.toFixed(1)}}`, at: 0.3 }, { tex: `z=${((x0 - med) / iqr).toFixed(2)}`, at: 0.7 }];
      if (s.method === "MaxAbsScaler") return [{ tex: `z=\\frac{x}{\\max|x|}`, at: 0 }, { tex: `z=\\frac{${x0.toFixed(1)}}{${maxabs.toFixed(1)}}=${(x0 / maxabs).toFixed(2)}`, at: 0.5 }];
      return [{ tex: `z=\\text{${s.method.replace(/[^a-zA-Z0-9 ]/g, " ")}}(x)`, at: 0 }];
    }
    if (s.op === "Transform") { const m = s.method.toLowerCase(); if (m.includes("log")) return [{ tex: `x\\to\\log(1+x)`, at: 0 }, { tex: `${x0.toFixed(1)}\\to${Math.log(1 + Math.max(0, x0)).toFixed(2)}`, at: 0.5 }]; if (m.includes("sqrt")) return [{ tex: `x\\to\\sqrt{x}`, at: 0 }, { tex: `${x0.toFixed(1)}\\to${Math.sqrt(Math.max(0, x0)).toFixed(2)}`, at: 0.5 }]; return [{ tex: `x\\to\\text{${s.method.replace(/[^a-zA-Z0-9 ]/g, " ")}}(x)`, at: 0 }]; }
    if (s.op === "Handle outliers") return [{ tex: `\\text{clip to } [Q_1-1.5\\,\\text{IQR},\\ Q_3+1.5\\,\\text{IQR}]`, at: 0 }, { tex: `=[${(q1 - 1.5 * iqr).toFixed(1)},\\ ${(q3 + 1.5 * iqr).toFixed(1)}]`, at: 0.4 }];
    if (s.op === "Bin / discretize") return [{ tex: `x\\to\\left\\lfloor\\frac{x-\\min}{\\max-\\min}\\times k\\right\\rfloor`, at: 0 }];
    return [];
  }
  function subsamplePairs(before: (number | string | null)[], after: (number | string | null)[], max = 36): [number, number][] {
    const all: [number, number][] = [];
    for (let i = 0; i < before.length; i++) { if (typeof before[i] === "number" && typeof after[i] === "number") all.push([before[i] as number, after[i] as number]); }
    if (all.length <= max) return all;
    const out: [number, number][] = []; const stp = all.length / max; for (let k = 0; k < max; k++) out.push(all[Math.floor(k * stp)]); return out;
  }
  // Numeric transform (scale / transform / bin / outliers): staged formula + points sliding raw → transformed.
  function numTransformView(colName: string, before: (number | string | null)[], after: (number | string | null)[], s: PrepStep): React.ReactNode {
    const isScale = s.op === "Scale / normalize";
    const scaleMethods = ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"];
    const method = isScale ? (scaleMethods.includes(prepScaleMethod) ? prepScaleMethod : (scaleMethods.includes(s.method) ? s.method : "StandardScaler")) : s.method;
    const bn = before.filter((v) => typeof v === "number") as number[];
    const eff: (number | string | null)[] = isScale ? scaleNumCol(before as (number | null)[], method) : after;
    const pairs = subsamplePairs(before, eff);
    if (!pairs.length) return <div className="note">Nothing numeric to animate here.</div>;
    const rb = pairs.map((p) => p[0]), ra = pairs.map((p) => p[1]);
    // Shared absolute axis over BOTH raw and scaled values, so the shift/squeeze is visible
    // (affine scalers preserve relative order, so per-range normalisation would show no motion).
    const lo = Math.min(...rb, ...ra), hi = Math.max(...rb, ...ra), span = (hi - lo) || 1;
    const pos = (x: number) => 4 + 92 * (x - lo) / span;
    const rpx = rb.map(pos), apx = ra.map(pos);
    const zeroPct = lo <= 0 && hi >= 0 ? pos(0) : null;
    const stages = scaleStages({ ...s, method }, bn);
    const nFormula = stages.length; const applySteps = 4; const maxStage = nFormula - 1 + applySteps;
    const tt = Math.max(0, Math.min(1, (prepStage - (nFormula - 1)) / applySteps));
    const items: { sym: string; desc: string; how?: string; val?: string }[] = [{ sym: "x", desc: "each raw value" }];
    if (bn.length) {
      const mu = mean(bn), sd = std(bn) || 1, mn = Math.min(...bn), mx = Math.max(...bn); const { med, q1, q3 } = quart(bn); const iqr = (q3 - q1) || 1, maxabs = Math.max(...bn.map(Math.abs)) || 1;
      if (isScale) {
        if (method === "StandardScaler") items.push({ sym: "\\mu", desc: "mean", how: "average of x", val: mu.toFixed(2) }, { sym: "\\sigma", desc: "std deviation", how: "spread of x", val: sd.toFixed(2) });
        else if (method === "MinMaxScaler") items.push({ sym: "\\min", desc: "smallest value", val: mn.toFixed(2) }, { sym: "\\max", desc: "largest value", val: mx.toFixed(2) });
        else if (method === "RobustScaler") items.push({ sym: "\\text{median}", desc: "middle value", val: med.toFixed(2) }, { sym: "\\text{IQR}", desc: "Q3 − Q1", how: "robust spread", val: iqr.toFixed(2) });
        else if (method === "MaxAbsScaler") items.push({ sym: "\\max|x|", desc: "largest magnitude", val: maxabs.toFixed(2) });
        items.push({ sym: "z", desc: "the scaled output value" });
      } else if (s.op === "Handle outliers") items.push({ sym: "Q_1,\\ Q_3", desc: "quartiles", val: `${q1.toFixed(1)}, ${q3.toFixed(1)}` }, { sym: "\\text{IQR}", desc: "Q3 − Q1", val: iqr.toFixed(2) });
      else if (s.op === "Bin / discretize") items.push({ sym: "\\min,\\max", desc: "column range", val: `${mn.toFixed(1)}, ${mx.toFixed(1)}` }, { sym: "k", desc: "number of bins" });
    }
    return <>
      {isScale && <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
        <label className="fld" style={{ margin: 0 }}>Method</label>
        <select value={method} onChange={(e) => setPrepScaleMethod(e.target.value)} style={{ maxWidth: 190 }}>{scaleMethods.map((m) => <option key={m}>{m}</option>)}</select>
      </div>}
      {varLegend(items)}
      <div className="prep-2col">
        <div className="prep-col">
          <div className="prep-col-h">the formula, step by step</div>
          {stages.map((l, i) => <div key={i} className={`fx-line ${prepStage >= i ? "on" : ""}`}><Katex block tex={l.tex} /></div>)}
        </div>
        <div className="prep-col">
          <div className="prep-col-h">values slide raw → transformed (same axis)</div>
          <div className="num-line">
            {zeroPct != null && <span className="num-zero" style={{ left: `${zeroPct}%` }} data-lab="0" />}
            {pairs.map((_, i) => <span key={i} className="num-dot" style={{ left: `${rpx[i] + (apx[i] - rpx[i]) * tt}%` }} />)}
          </div>
          <div className="num-axis"><span>{lo.toFixed(1)}</span><span>{hi.toFixed(1)}</span></div>
          <div className="note" style={{ marginTop: 8 }}>{tt < 1 ? `raw dots slide toward their ${isScale ? "scaled" : "transformed"} positions on one shared axis — step to move them` : "every point is now at its transformed value"}</div>
        </div>
      </div>
      {stageControls(maxStage)}
      <div className="note" style={{ marginTop: 8, lineHeight: 1.7 }}>before → {prepStat(before)}<br />after&nbsp;&nbsp;→ {prepStat(eff)}</div>
    </>;
  }
  const prepStat = (vs: (number | string | null)[]) => { const a = vs.filter((v) => typeof v === "number") as number[]; return a.length ? `μ=${mean(a).toFixed(2)}, σ=${std(a).toFixed(2)}, min=${Math.min(...a).toFixed(1)}, max=${Math.max(...a).toFixed(1)}` : "—"; };

  // Imputation: pick a method → its formula computes, then every ∅ fills with it.
  function imputeView(colName: string, before: (number | string | null)[], s: PrepStep, numeric: boolean): React.ReactNode {
    const methods = numeric ? ["Mean", "Median", "Most frequent", "Constant"] : ["Most frequent", "Constant"];
    const method = methods.includes(prepImputeMethod) ? prepImputeMethod : (methods.includes(s.method) ? s.method : methods[0]);
    const present = before.filter((v) => v != null);
    let fill = "0"; let stages: { tex: string; at: number }[] = []; let impItems: { sym: string; desc: string; how?: string; val?: string }[] = [];
    if (numeric) {
      const nums = present as number[]; const sum = nums.reduce((a, b) => a + b, 0); const { med } = quart(nums.length ? nums : [0]);
      const cnt = new Map<number, number>(); nums.forEach((x) => cnt.set(x, (cnt.get(x) || 0) + 1)); const mode = [...cnt.entries()].sort((a, b) => b[1] - a[1])[0];
      if (method === "Mean") { fill = (sum / (nums.length || 1)).toFixed(2); stages = [{ tex: `\\mu=\\frac{\\sum x}{n}`, at: 0 }, { tex: `\\mu=\\frac{${sum.toFixed(1)}}{${nums.length}}`, at: 0.2 }, { tex: `\\mu=${fill}`, at: 0.45 }]; impItems = [{ sym: "x", desc: "each present value" }, { sym: "n", desc: "count of present values", val: String(nums.length) }, { sym: "\\textstyle\\sum x", desc: "sum of them", val: sum.toFixed(1) }, { sym: "\\mu", desc: "mean = the fill value", how: "sum ÷ n", val: fill }]; }
      else if (method === "Median") { fill = String(Math.round(med * 100) / 100); stages = [{ tex: `\\text{sort the values, take the middle}`, at: 0 }, { tex: `\\text{median}=${fill}`, at: 0.4 }]; impItems = [{ sym: "\\text{median}", desc: "middle value when sorted", how: "robust to outliers", val: fill }]; }
      else if (method === "Most frequent") { fill = String(mode ? mode[0] : 0); stages = [{ tex: `\\text{mode}=${fill}\\ (\\text{appears}\\ ${mode ? mode[1] : 0}\\times)`, at: 0 }]; impItems = [{ sym: "\\text{mode}", desc: "most common value", how: `appears ${mode ? mode[1] : 0} times`, val: fill }]; }
      else { fill = "0"; stages = [{ tex: `\\text{fill}=0\\ (\\text{constant})`, at: 0 }]; impItems = [{ sym: "0", desc: "a fixed constant you choose" }]; }
    } else {
      const strs = present as string[]; const cnt = new Map<string, number>(); strs.forEach((x) => cnt.set(x, (cnt.get(x) || 0) + 1)); const mode = [...cnt.entries()].sort((a, b) => b[1] - a[1])[0];
      if (method === "Most frequent") { fill = mode ? mode[0] : "missing"; stages = [{ tex: `\\text{mode}=\\text{${String(fill).replace(/[^a-zA-Z0-9 ]/g, " ")}}\\ (\\text{appears}\\ ${mode ? mode[1] : 0}\\times)`, at: 0 }]; impItems = [{ sym: "\\text{mode}", desc: "most common category", how: `appears ${mode ? mode[1] : 0} times`, val: fill }]; }
      else { fill = "missing"; stages = [{ tex: `\\text{fill}=\\text{missing (constant)}`, at: 0 }]; impItems = [{ sym: "\\text{\"missing\"}", desc: "a new placeholder category" }]; }
    }
    const missIdx = before.map((v, i) => (v == null ? i : -1)).filter((i) => i >= 0);
    const showIdx = Array.from(new Set([0, 1, 2, 3, ...missIdx.slice(0, 4)])).filter((i) => i < before.length).sort((a, b) => a - b).slice(0, 8);
    const nFormula = stages.length;
    const missVisible = showIdx.filter((i) => before[i] == null);
    const maxStage = nFormula - 1 + missVisible.length;
    return <>
      <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
        <label className="fld" style={{ margin: 0 }}>Method</label>
        <select value={method} onChange={(e) => setPrepImputeMethod(e.target.value)} style={{ maxWidth: 170 }}>{methods.map((m) => <option key={m}>{m}</option>)}</select>
        <span className="note">{missIdx.length} missing in “{colName}” → fill each with the {method.toLowerCase()} = <b>{fill}</b></span>
      </div>
      {varLegend(impItems)}
      <div className="prep-2col">
        <div className="prep-col">
          <div className="prep-col-h">the formula, step by step</div>
          {stages.map((l, i) => <div key={i} className={`fx-line ${prepStage >= i ? "on" : ""}`}><Katex block tex={l.tex} /></div>)}
          {prepStage >= nFormula - 1 && <div className="note" style={{ marginTop: 8 }}>→ fill value = <b style={{ color: "var(--good)" }}>{fill}</b></div>}
        </div>
        <div className="prep-col">
          <div className="prep-col-h">{missIdx.length ? `then fill each ∅ (${missVisible.length} shown)` : "no missing values"}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {showIdx.map((i) => { const miss = before[i] == null; const filled = miss && prepStage >= nFormula + missVisible.indexOf(i);
              return <div key={i} className="icell"><span className="irow-lab">row {i}</span><span className={`ibox ${miss ? "miss" : ""} ${filled ? "filled" : ""}`}>{miss ? (filled ? <b style={{ color: "var(--good)" }}>{fill}</b> : <span style={{ color: "var(--crit)" }}>∅</span>) : String(before[i])}</span></div>; })}
          </div>
        </div>
      </div>
      {stageControls(maxStage)}
    </>;
  }
  // One-hot: the indicator matrix builds row by row as the animation advances.
  function catOneHotView(colName: string): React.ReactNode {
    if (!ds) return null;
    const col = ds.columns.find((c) => c.name === colName)!;
    const methods = ["One-Hot", "Ordinal", "Frequency", "Count", "Binary"];
    const method = methods.includes(prepEncodeMethod) ? prepEncodeMethod : "One-Hot";
    const cats = Array.from(new Set(col.values.filter((v) => v != null).map(String)));
    const catsShown = cats.slice(0, 6);
    const idxMap = new Map(cats.map((c, i) => [c, i]));
    const counts = new Map<string, number>(); let n = 0; col.values.forEach((v) => { if (v != null) { const k = String(v); counts.set(k, (counts.get(k) || 0) + 1); n++; } });
    const rowIdx = col.values.map((v, i) => (v != null ? i : -1)).filter((i) => i >= 0).slice(0, 5);
    let headers: string[]; let valsOf: (v: string) => string[]; let note: React.ReactNode; let oneHot = false;
    if (method === "Ordinal") { headers = [`${colName}_idx`]; valsOf = (v) => [String(idxMap.get(v) ?? 0)]; note = <>Each category → an integer <b>index</b> (compact — but implies a fake order).</>; }
    else if (method === "Frequency") { headers = [`${colName}_freq`]; valsOf = (v) => [((counts.get(v) || 0) / (n || 1)).toFixed(2)]; note = <>Each category → how often it appears: <b>count ÷ n</b>.</>; }
    else if (method === "Count") { headers = [`${colName}_count`]; valsOf = (v) => [String(counts.get(v) || 0)]; note = <>Each category → its raw <b>count</b> in the data.</>; }
    else if (method === "Binary") { const bits = Math.max(1, Math.ceil(Math.log2(cats.length || 1))); headers = Array.from({ length: bits }, (_, b) => `b${b}`); valsOf = (v) => { const i = idxMap.get(v) ?? 0; return Array.from({ length: bits }, (_, b) => String((i >> b) & 1)); }; note = <>Each category’s index written in <b>binary bits</b> — fewer columns than one-hot.</>; }
    else { headers = catsShown; valsOf = (v) => catsShown.map((c) => (c === v ? "1" : "0")); note = <>A column per category; each row puts a <b>1</b> in its own column, 0 elsewhere.</>; oneHot = true; }
    const maxStage = Math.max(0, rowIdx.length - 1);
    const cells: React.ReactNode[] = [<div key="corner" className="oh-lab" style={{ color: "var(--faint)" }}>{colName}</div>, ...headers.map((h) => <div key={`h${h}`} className="oh-h">{h}</div>)];
    rowIdx.forEach((ri, r) => {
      const val = String(col.values[ri]); const on = prepStage >= r; const vals = valsOf(val);
      cells.push(<div key={`l${ri}`} className="oh-lab" style={{ opacity: on ? 1 : 0.4 }}>“{val}”</div>);
      vals.forEach((vv, ci) => { const hit = oneHot && vv === "1"; cells.push(<div key={`${ri}-${ci}`} className={`oh-c ${hit && on ? "on" : ""}`} style={{ opacity: on ? 1 : 0.4, color: on && !oneHot ? "var(--text)" : undefined, fontWeight: on && !oneHot ? 600 : undefined }}>{on ? vv : (oneHot ? "0" : "·")}</div>); });
    });
    return <>
      <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
        <label className="fld" style={{ margin: 0 }}>Method</label>
        <select value={method} onChange={(e) => setPrepEncodeMethod(e.target.value)} style={{ maxWidth: 150 }}>{methods.map((m) => <option key={m}>{m}</option>)}</select>
        <span className="note">{cats.length} categories</span>
      </div>
      <div className="note" style={{ marginBottom: 10 }}>{note}</div>
      <div className="prep-col-h">the encoded output, one row at a time</div>
      <div className="oh-grid" style={{ gridTemplateColumns: `auto repeat(${headers.length}, minmax(58px, 1fr))` }}>{cells}</div>
      {stageControls(maxStage)}
    </>;
  }

  // "How the summary statistics come from the data" — formulas applied + visualized.
  function mathSummary() {
    if (!ds) return null;
    const nums = ds.columns.filter((c) => c.type === "num");
    if (!nums.length) return null;
    const name = nums.find((c) => c.name === summaryCol)?.name || nums[0].name;
    const v = colNumVals(name).slice().sort((a, b) => a - b); const n = v.length;
    if (!n) return null;
    const sum = v.reduce((a, b) => a + b, 0), mu = sum / n;
    const ss = v.reduce((a, x) => a + (x - mu) ** 2, 0), sd = Math.sqrt(ss / n);
    const qtile = (p: number) => { const idx = (n - 1) * p, lo = Math.floor(idx), hi = Math.ceil(idx); return v[lo] + (v[hi] - v[lo]) * (idx - lo); };
    const median = qtile(0.5), q1 = qtile(0.25), q3 = qtile(0.75), mn = v[0], mx = v[n - 1];
    const t = plotlyTheme();
    return <div className="card math-card" style={{ marginTop: 16 }}>
      <div className="card-h"><span className="t">🧮 How each summary statistic is computed</span><select value={name} onChange={(e) => setSummaryCol(e.target.value)} className="r" style={{ maxWidth: 190 }}>{nums.map((c) => <option key={c.name}>{c.name}</option>)}</select></div>
      <div className="card-b">
        <div className="note" style={{ marginBottom: 8 }}>Each number in the table above is a formula run over the {n} values of “{name}”. Here they are, <b>applied to your data</b>:</div>
        <div className="mathrow"><Katex block tex={`\\mu = \\tfrac{1}{n}\\sum_i x_i = \\tfrac{${sum.toFixed(1)}}{${n}} = ${mu.toFixed(3)}`} /></div>
        <div className="mathrow"><Katex block tex={`\\sigma = \\sqrt{\\tfrac{1}{n}\\sum_i (x_i-\\mu)^2} = \\sqrt{\\tfrac{${ss.toFixed(1)}}{${n}}} = ${sd.toFixed(3)}`} /></div>
        <div className="note" style={{ margin: "4px 0" }}>Sort the values, then read off the positions (min, 25%, 50%, 75%, max):</div>
        <div className="mathrow"><Katex tex={`\\min=${mn.toFixed(2)},\\ \\ Q_1=${q1.toFixed(2)},\\ \\ \\text{median}=${median.toFixed(2)},\\ \\ Q_3=${q3.toFixed(2)},\\ \\ \\max=${mx.toFixed(2)}`} /></div>
        {varLegend([
          { sym: "x_i", desc: "each value in the column", how: `${n} of them` },
          { sym: "n", desc: "count of values", how: "how many rows", val: String(n) },
          { sym: "\\textstyle\\sum x", desc: "sum of all values", how: "add them up", val: sum.toFixed(1) },
          { sym: "\\mu", desc: "mean", how: "sum ÷ n", val: mu.toFixed(3) },
          { sym: "\\sigma", desc: "standard deviation", how: "spread around the mean", val: sd.toFixed(3) },
          { sym: "Q_1,\\ Q_3", desc: "lower / upper quartiles", how: "25% and 75% marks", val: `${q1.toFixed(1)}, ${q3.toFixed(1)}` },
        ])}
        <label className="fld" style={{ marginTop: 12 }}>Box plot — the box is Q1→Q3, the line is the median, whiskers reach min/max, dashed = mean</label>
        <Plot data={[{ type: "box", x: v, name, boxmean: true, marker: { color: "#5b7cff" }, line: { color: "#5b7cff" } }]} layout={{ ...chartLayout(t, "", name, ""), showlegend: false }} style={{ height: 170 }} />
        <label className="fld" style={{ marginTop: 6 }}>Distribution — where the mean (orange) and median (green) fall</label>
        <Plot data={[{ type: "histogram", x: colNumVals(name), marker: { color: "#5b7cff" }, opacity: 0.8 }]} layout={{ ...chartLayout(t, "", name, "count"), showlegend: false, shapes: [{ type: "line", x0: mu, x1: mu, y0: 0, y1: 1, yref: "paper", line: { color: "#f59e0b", width: 2 } }, { type: "line", x0: median, x1: median, y0: 0, y1: 1, yref: "paper", line: { color: "#3ecf7f", width: 2, dash: "dot" } }] }} style={{ height: 220 }} />
        <div className="note" style={{ marginTop: 6 }}>When the mean sits far from the median the data is <b>skewed</b> — that&apos;s why we report both.</div>
      </div>
    </div>;
  }
  function mathPrep() {
    if (!ds) return null;
    const feats = features;
    if (!feats.length) return mCard("preprocessing", <div className="note">No features.</div>);
    const colName = feats.includes(prepCol) ? prepCol : feats[0];
    const numeric = ds.columns.find((c) => c.name === colName)?.type === "num";
    const trace = prepColTrace(ds, steps, colName);
    const stepCount = trace.length - 1;
    const selRow = <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
      <label className="fld" style={{ margin: 0 }}>Watch column</label>
      <select value={colName} onChange={(e) => { setPrepCol(e.target.value); setPrepStepIdx(1); }} style={{ maxWidth: 200 }}>{feats.map((f) => <option key={f} value={f}>{f}{ds.columns.find((c) => c.name === f)?.type === "cat" ? " (text)" : ""}</option>)}</select>
    </div>;
    if (stepCount === 0) return mCard("preprocessing — step by step", <>{selRow}<div className="note">Add a preprocessing step (Impute, Scale, Encode, …) that touches “{colName}”, then step through it here.</div></>);
    const idx = Math.min(Math.max(1, prepStepIdx), stepCount);
    const s = steps[idx - 1];
    const before = trace[idx - 1].values, after = trace[idx].values;
    const touches = s.cols.includes(colName);

    let body: React.ReactNode;
    if (s.op === "Impute missing" && touches) body = imputeView(colName, before, s, numeric);
    else if (s.op === "Encode categorical" && touches && !numeric) body = catOneHotView(colName);
    else if (numeric && trace[idx].changed) body = numTransformView(colName, before, after, s);
    else body = <div className="note">Step “{s.op} · {s.method}” doesn’t transform “{colName}”. Pick a step that affects it, or switch columns.</div>;

    return mCard("preprocessing — step by step", <>
      {selRow}
      <div className="chips" style={{ marginBottom: 10 }}>
        <span className="chip" style={{ cursor: "default", opacity: 0.55 }}>raw</span>
        {steps.map((st, i) => <span key={i} className={`chip ${i + 1 === idx ? "on" : ""}`} style={{ cursor: "pointer", opacity: trace[i + 1].changed ? 1 : 0.4 }} onClick={() => setPrepStepIdx(i + 1)}>{i + 1}. {st.op}</span>)}
      </div>
      <div className="note" style={{ marginBottom: 6 }}>Step {idx}/{stepCount}: <b>{s.op} · {s.method}</b> on {s.cols.join(", ")}</div>
      {body}
      <div className="row" style={{ gap: 8, marginTop: 12 }}>
        <button className="btn ghost sm" disabled={idx <= 1} onClick={() => setPrepStepIdx(idx - 1)}>← Prev step</button>
        <button className="btn ghost sm" disabled={idx >= stepCount} onClick={() => setPrepStepIdx(idx + 1)}>Next step →</button>
      </div>
    </>);
  }
  // Model step: for every classifier/regressor, show the variable legend (symbol · meaning ·
  // how it's derived from your data · its value), the formula flow, and a stepped SVG animation
  // of the mechanism running on the real rows.
  function mathModel() {
    const label = MODEL_INFO[algo]?.label ?? algo;
    const isReg = task === "regression";
    let b: ReturnType<typeof buildMatrix> | null = null;
    try { if (ds) b = buildMatrix(ds, features, target, task, steps); } catch { b = null; }
    if (!ds || !b || !b.X.length || !b.featureNames.length) return mCard(`${label} — how it learns`, <div className="note">Pick your features &amp; target first — then this walks the model through your actual rows.</div>);
    const X = b.X, Y = b.y, names = b.featureNames, classes = b.classes;
    const n = X.length, nf = names.length, K = classes?.length ?? 0;
    // plot geometry
    const PX0 = 30, PX1 = 344, PYb = 212, PYt = 20;
    const sx = (t: number) => PX0 + (PX1 - PX0) * Math.max(0, Math.min(1, t));
    const sy = (t: number) => PYb + (PYt - PYb) * Math.max(0, Math.min(1, t));
    const pick = (m: number, mx: number) => (m <= mx ? [...Array(m).keys()] : [...Array(mx).keys()].map((k) => Math.floor(k * (m / mx))));
    const idx = pick(n, 46);
    const colj = (j: number) => X.map((r) => r[j] ?? 0);
    const nrm = (arr: number[]) => { const mn = Math.min(...arr), mx = Math.max(...arr), d = (mx - mn) || 1; return { mn, mx, z: arr.map((v) => (v - mn) / d) }; };
    const idxF = names.indexOf(dbF1) >= 0 ? names.indexOf(dbF1) : 0;
    const idxG = names.indexOf(dbF2) >= 0 && names.indexOf(dbF2) !== idxF ? names.indexOf(dbF2) : Math.min(nf - 1, idxF + 1);
    const clsColor = (i: number) => PAL_ML[Y[i] % PAL_ML.length];
    const st = modelStage;
    const frame = <rect key="fr" x={PX0} y={PYt} width={PX1 - PX0} height={PYb - PYt} fill="none" stroke="var(--border)" />;
    const svg = (kids: React.ReactNode) => <svg viewBox="0 0 360 240" style={{ width: "100%", borderRadius: 8, background: "var(--panel)", border: "1px solid var(--border)" }}>{frame}{kids}</svg>;
    const metric = (ids: number[]): number => { if (!ids.length) return 0; if (isReg) { const ys = ids.map((i) => Y[i]); const mu = ys.reduce((a, c) => a + c, 0) / ys.length; return ys.reduce((a, c) => a + (c - mu) ** 2, 0) / ys.length; } const c: Record<number, number> = {}; ids.forEach((i) => { c[Y[i]] = (c[Y[i]] || 0) + 1; }); let s = 1; for (const k in c) { const p = c[k] / ids.length; s -= p * p; } return s; };
    const bestSplit = (ids: number[], feats: number[]) => {
      const parent = metric(ids); let best = { gain: -1, feat: feats[0], thr: 0, gl: 0, gr: 0, nl: 0, nr: 0 };
      for (const f of feats) { const vals = [...new Set(ids.map((i) => X[i][f]))].sort((a, c) => a - c); for (let t = 0; t < vals.length - 1; t++) { const thr = (vals[t] + vals[t + 1]) / 2; const L = ids.filter((i) => X[i][f] <= thr), R = ids.filter((i) => X[i][f] > thr); if (!L.length || !R.length) continue; const gl = metric(L), gr = metric(R); const gain = parent - (L.length / ids.length) * gl - (R.length / ids.length) * gr; if (gain > best.gain) best = { gain, feat: f, thr, gl, gr, nl: L.length, nr: R.length }; } }
      return { ...best, parent };
    };

    let intro = ""; let legend: { sym: string; desc: string; how?: string; val?: string }[] = [];
    let formulas: [string, string][] = []; let caps: string[] = []; let viz: (s: number) => React.ReactNode = () => null;
    const kind = modelKind; // "gd" | "tree" | "forest" | "gnb" | "knn"

    if (kind === "gd" && !isReg) { // Logistic Regression
      let gw: number[] | null = null;
      try { const gt = gdTrace(cfgNow(), X, Y, K); const last = gt?.snaps?.[(gt?.snaps?.length ?? 0) - 1]; gw = last?.w || null; } catch { /* ignore */ }
      const z = (r: number[]) => gw ? gw[0] + r.reduce((a, v, j) => a + (gw![j + 1] || 0) * v, 0) : 0;
      const sig = (v: number) => 1 / (1 + Math.exp(-v));
      const f0 = nrm(colj(idxF)); const pz = X.map((r) => sig(z(r)));
      const z0 = z(X[0]), p0 = sig(z0);
      const order = [...idx].sort((a, c) => f0.z[a] - f0.z[c]);
      intro = "A weighted sum of your features becomes a score z; the sigmoid squashes it to a probability, and 0.5 is the cut between the two classes.";
      legend = [
        { sym: "x", desc: `a feature`, how: `e.g. ${names[idxF]}, row 0`, val: X[0][idxF]?.toFixed(2) },
        { sym: "\\mathbf w, b", desc: "weights & bias", how: "learned by gradient descent", val: gw ? `b=${gw[0].toFixed(2)}, w₁=${(gw[1] ?? 0).toFixed(2)}` : "—" },
        { sym: "z", desc: "linear score", how: "w·x + b for a row", val: z0.toFixed(2) },
        { sym: "\\hat y", desc: `P(${classes?.[1] ?? "class 1"})`, how: "σ(z)", val: p0.toFixed(2) },
      ];
      formulas = [["z = \\mathbf{w}\\cdot\\mathbf{x} + b", "weighted sum → a score"], ["\\hat y = \\sigma(z) = \\tfrac{1}{1+e^{-z}}", "squash into 0…1"], [`\\hat y = \\sigma(${z0.toFixed(2)}) = ${p0.toFixed(2)}`, "apply to row 0"], ["\\hat y \\ge 0.5 \\Rightarrow " + `\\text{${(classes?.[1] ?? "class 1").replace(/[^a-zA-Z0-9 ]/g, " ")}}`, "threshold the probability"]];
      caps = [`your rows along ${names[idxF]}, labelled by class`, "the fitted probability curve", "each row drops to its predicted ŷ", "ŷ ≥ 0.5 splits the class"];
      viz = (s) => svg([
        ...idx.map((i, k) => <circle key={"p" + k} cx={sx(f0.z[i])} cy={s >= 2 ? sy(pz[i]) : sy(Y[i] ? 0.86 : 0.14)} r={4} fill={clsColor(i)} opacity={0.85} style={{ transition: "cy .4s" }} />),
        s >= 1 ? <polyline key="curve" points={order.map((i) => `${sx(f0.z[i])},${sy(pz[i])}`).join(" ")} fill="none" stroke="#3ecf7f" strokeWidth={2} /> : null,
        s >= 3 ? <line key="thr" x1={PX0} y1={sy(0.5)} x2={PX1} y2={sy(0.5)} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="5 4" /> : null,
        s >= 3 ? <text key="thrl" x={PX0 + 3} y={sy(0.5) - 4} fill="var(--faint)" fontSize={11} fontFamily="monospace">0.5</text> : null,
      ]);
    } else if (kind === "gd" && isReg) { // Linear / Ridge Regression
      let gw: number[] | null = null;
      try { const gt = gdTrace(cfgNow(), X, Y, 0); const last = gt?.snaps?.[(gt?.snaps?.length ?? 0) - 1]; gw = last?.w || null; } catch { /* ignore */ }
      const pred = (r: number[]) => gw ? gw[0] + r.reduce((a, v, j) => a + (gw![j + 1] || 0) * v, 0) : 0;
      const f0 = nrm(colj(idxF)); const yn = nrm(Y); const yh = X.map(pred); const yMin = Math.min(...Y), yRange = (Math.max(...Y) - yMin) || 1; const yhn = { z: yh.map((v) => (v - yMin) / yRange) };
      const mse = idx.reduce((a, i) => a + (yh[i] - Y[i]) ** 2, 0) / idx.length;
      const order = [...idx].sort((a, c) => f0.z[a] - f0.z[c]);
      intro = "Fit a straight line so the average squared vertical gap (residual) between the line and your points is as small as possible.";
      legend = [
        { sym: "x", desc: "a feature", how: `${names[idxF]}, row 0`, val: X[0][idxF]?.toFixed(2) },
        { sym: "\\mathbf w, b", desc: "slope & intercept", how: "learned by least squares", val: gw ? `b=${gw[0].toFixed(2)}, w₁=${(gw[1] ?? 0).toFixed(2)}` : "—" },
        { sym: "\\hat y", desc: "prediction", how: "w·x + b", val: gw ? pred(X[0]).toFixed(2) : "—" },
        { sym: "\\mathcal L", desc: "mean squared error", how: "avg of (ŷ−y)²", val: mse.toFixed(2) },
      ];
      formulas = [["\\hat y = \\mathbf{w}\\cdot\\mathbf{x} + b", "a line through the cloud"], ["r_i = \\hat y_i - y_i", "residual: the miss"], [`\\mathcal L = \\tfrac1n\\sum r_i^2 = ${mse.toFixed(2)}`, "mean squared error"]].concat(algo === "Ridge" ? [["\\mathcal L \\mathrel{+}= \\alpha\\lVert\\mathbf w\\rVert^2", "Ridge adds an L2 penalty"]] : []) as [string, string][];
      caps = ["your points: feature vs target", "the fitted line", "residuals = vertical gaps", "MSE = mean of the squares"];
      viz = (s) => svg([
        ...idx.map((i, k) => <circle key={"p" + k} cx={sx(f0.z[i])} cy={sy(yn.z[i])} r={4} fill="#5b7cff" opacity={0.7} />),
        s >= 1 ? <polyline key="ln" points={order.map((i) => `${sx(f0.z[i])},${sy(yhn.z[i])}`).join(" ")} fill="none" stroke="#f59e0b" strokeWidth={2.5} /> : null,
        ...(s >= 2 ? idx.map((i, k) => <line key={"r" + k} x1={sx(f0.z[i])} y1={sy(yn.z[i])} x2={sx(f0.z[i])} y2={sy(yhn.z[i])} stroke="#ef4444" strokeWidth={1} opacity={0.7} />) : []),
        s >= 3 ? <text key="mse" x={150} y={34} fill="var(--faint)" fontSize={12} fontFamily="monospace">MSE = {mse.toFixed(2)}</text> : null,
      ]);
    } else if (kind === "knn") {
      const kk = Math.max(1, Math.round(Number(params.n_neighbors) || 5));
      const qi = idx[Math.floor(idx.length / 2)] ?? 0; const q = X[qi];
      const dist = (r: number[]) => Math.hypot(...r.map((v, j) => v - q[j]));
      const others = idx.filter((i) => i !== qi);
      const near = [...others].sort((a, c) => dist(X[a]) - dist(X[c])).slice(0, kk);
      const f0 = nrm(colj(idxF)), f1 = nrm(colj(idxG));
      const dNear = dist(X[near[0]]);
      const votes = near.filter((i) => Y[i] === 1).length;
      const yhat = isReg ? (near.reduce((a, i) => a + Y[i], 0) / near.length).toFixed(2) : (classes?.[votes * 2 >= kk ? 1 : 0] ?? "—");
      intro = isReg ? "No training. For a new point, measure the distance to every stored point, keep the k nearest, and average their target." : "No training. For a new point, measure the distance to every stored point, keep the k nearest, and let them vote.";
      legend = [
        { sym: "d", desc: "Euclidean distance", how: "√Σ(aⱼ−bⱼ)² across features", val: dNear.toFixed(2) + " (nearest)" },
        { sym: "k", desc: "neighbours kept", how: "your n_neighbors setting", val: String(kk) },
        { sym: "\\hat y", desc: isReg ? "average of k" : "majority of k", how: isReg ? "mean target of neighbours" : "most common class", val: String(yhat) },
      ];
      formulas = [["d(\\mathbf a,\\mathbf b) = \\sqrt{\\textstyle\\sum_j (a_j-b_j)^2}", "distance to every point"], [`\\text{keep the } k=${kk} \\text{ smallest } d`, "the nearest neighbours"], isReg ? ["\\hat y = \\tfrac1k\\textstyle\\sum_{i\\in kNN} y_i", "average their target"] : ["\\hat y = \\text{mode of their labels}", "they vote"]];
      caps = ["stored rows + a new ★ point", "distance to all of them", `circle the k=${kk} nearest`, isReg ? "average → ŷ" : `vote → ${yhat}`];
      viz = (s) => svg([
        ...(s >= 1 ? others.map((i, k) => <line key={"l" + k} x1={sx(f0.z[qi])} y1={sy(f1.z[qi])} x2={sx(f0.z[i])} y2={sy(f1.z[i])} stroke="var(--border-strong)" strokeWidth={0.6} opacity={0.5} />) : []),
        ...others.map((i, k) => <circle key={"p" + k} cx={sx(f0.z[i])} cy={sy(f1.z[i])} r={s >= 2 && near.includes(i) ? 6 : 4} fill={isReg ? "#5b7cff" : clsColor(i)} opacity={s >= 2 && !near.includes(i) ? 0.35 : 0.85} />),
        s >= 2 ? <circle key="ring" cx={sx(f0.z[qi])} cy={sy(f1.z[qi])} r={Math.max(...near.map((i) => Math.hypot(sx(f0.z[i]) - sx(f0.z[qi]), sy(f1.z[i]) - sy(f1.z[qi]))))} fill="none" stroke="#a855f7" strokeWidth={1.5} /> : null,
        <text key="star" x={sx(f0.z[qi]) - 6} y={sy(f1.z[qi]) + 5} fill="var(--text)" fontSize={16} fontFamily="monospace">★</text>,
        s >= 3 ? <text key="vt" x={116} y={PYb + 2} fill="var(--faint)" fontSize={11} fontFamily="monospace">{isReg ? `mean of ${kk} → ${yhat}` : `votes → ${yhat}`}</text> : null,
      ]);
    } else if (kind === "gnb") {
      const cA = idx.filter((i) => Y[i] === 0), cB = idx.filter((i) => Y[i] === 1);
      const colF = colj(idxF); const nf0 = nrm(colF);
      const mV = (ids: number[]) => { const v = ids.map((i) => colF[i]); const mu = v.reduce((a, c) => a + c, 0) / (v.length || 1); const va = v.reduce((a, c) => a + (c - mu) ** 2, 0) / (v.length || 1) || 1; return { mu, va }; };
      const sA = mV(cA.length ? cA : idx), sB = mV(cB.length ? cB : idx);
      const priorA = (cA.length || 1) / idx.length, priorB = (cB.length || 1) / idx.length;
      const xq = X[0][idxF]; const xqn = nf0.z[0];
      const bell = (x: number, mu: number, va: number) => Math.exp(-((x - mu) ** 2) / (2 * va));
      const likeA = bell(xq, sA.mu, sA.va), likeB = bell(xq, sB.mu, sB.va);
      const range = [Math.min(...colF), Math.max(...colF)];
      const curve = (mu: number, va: number) => { const pts: string[] = []; for (let t = 0; t <= 60; t++) { const x = range[0] + (range[1] - range[0]) * (t / 60); pts.push(`${sx(t / 60)},${sy(bell(x, mu, va) * 0.9)}`); } return pts.join(" "); };
      intro = "Learn one bell curve (mean & variance) per class. A point's class score is that class's prior times the bell heights at the point.";
      legend = [
        { sym: "\\mu", desc: `class mean (${names[idxF]})`, how: `avg of ${names[idxF]} in class ${classes?.[0] ?? 0}`, val: sA.mu.toFixed(2) },
        { sym: "\\sigma^2", desc: "class variance", how: "spread within the class", val: sA.va.toFixed(2) },
        { sym: "P(y)", desc: "class prior", how: "class count ÷ n", val: priorA.toFixed(2) },
        { sym: "P(x\\mid y)", desc: "bell height at x", how: "Gaussian at the point", val: likeA.toFixed(2) },
      ];
      formulas = [["P(x\\mid y) = \\tfrac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}", "a bell per class"], ["P(y\\mid x) \\propto P(y)\\,\\textstyle\\prod_j P(x_j\\mid y)", "prior × likelihoods"], [`P(${(classes?.[0] ?? "A").toString().replace(/[^a-zA-Z0-9 ]/g, " ")})\\!\\propto\\!${(priorA * likeA).toFixed(2)},\\ P(${(classes?.[1] ?? "B").toString().replace(/[^a-zA-Z0-9 ]/g, " ")})\\!\\propto\\!${(priorB * likeB).toFixed(2)}`, "apply → argmax wins"]];
      caps = ["a bell curve for each class", `drop your point x = ${xq.toFixed(2)}`, "read each bell's height", `×prior → ${((priorA * likeA) >= (priorB * likeB) ? (classes?.[0] ?? "A") : (classes?.[1] ?? "B"))} wins`];
      viz = (s) => svg([
        <polyline key="bA" points={curve(sA.mu, sA.va)} fill="none" stroke={PAL_ML[0]} strokeWidth={2} />,
        <polyline key="bB" points={curve(sB.mu, sB.va)} fill="none" stroke={PAL_ML[1]} strokeWidth={2} />,
        s >= 1 ? <line key="xq" x1={sx(xqn)} y1={PYt} x2={sx(xqn)} y2={PYb} stroke="var(--faint)" strokeDasharray="4 4" /> : null,
        s >= 1 ? <text key="xql" x={sx(xqn) - 4} y={PYb + 2} fill="var(--text)" fontSize={11} fontFamily="monospace">x</text> : null,
        s >= 2 ? <circle key="dA" cx={sx(xqn)} cy={sy(likeA * 0.9)} r={6} fill={PAL_ML[0]} /> : null,
        s >= 2 ? <circle key="dB" cx={sx(xqn)} cy={sy(likeB * 0.9)} r={6} fill={PAL_ML[1]} /> : null,
        s >= 3 ? <text key="win" x={96} y={34} fill="var(--faint)" fontSize={11} fontFamily="monospace">{(priorA * likeA) >= (priorB * likeB) ? `P(${classes?.[0] ?? "A"}) wins` : `P(${classes?.[1] ?? "B"}) wins`}</text> : null,
      ]);
    } else { // tree & forest
      const rowsAll = pick(n, 260);
      const sp = bestSplit(rowsAll, [idxF, idxG]);
      const leftIds = rowsAll.filter((i) => X[i][sp.feat] <= sp.thr);
      const sp2 = leftIds.length > 3 ? bestSplit(leftIds, [idxF, idxG]) : null;
      const f0 = nrm(colj(idxF)), f1 = nrm(colj(idxG));
      const thrPosOf = (feat: number, thr: number) => { const raw = colj(feat); const mn = Math.min(...raw), mx = Math.max(...raw); return (thr - mn) / ((mx - mn) || 1); };
      const spAxis = sp.feat === idxF ? "x" : "y";
      const thrPos = thrPosOf(sp.feat, sp.thr);
      const metricName = isReg ? "variance" : "Gini";
      if (kind === "tree") {
        intro = "Greedily test every feature threshold, keep the split that removes the most impurity, then recurse into if/else rules on each side.";
        legend = [
          { sym: "G", desc: `${metricName} impurity`, how: isReg ? "target spread in a node" : "1 − Σ pₖ² at a node", val: sp.parent.toFixed(3) },
          { sym: "n_L, n_R", desc: "rows each side", how: "counts after the split", val: `${sp.nl}, ${sp.nr}` },
          { sym: "\\text{gain}", desc: "impurity removed", how: "parent − weighted children", val: sp.gain.toFixed(3) },
          { sym: "\\text{thr}", desc: "chosen threshold", how: `best cut on ${names[sp.feat]}`, val: sp.thr.toFixed(2) },
        ];
        formulas = [[`G = ${isReg ? "\\text{var}(y)" : "1 - \\sum_k p_k^2"}`, "how mixed a node is"], ["\\text{gain} = G - \\tfrac{n_L}{n}G_L - \\tfrac{n_R}{n}G_R", "impurity a split removes"], [`\\text{gain} = ${sp.parent.toFixed(2)} - \\tfrac{${sp.nl}}{${sp.nl + sp.nr}}(${sp.gl.toFixed(2)}) - \\tfrac{${sp.nr}}{${sp.nl + sp.nr}}(${sp.gr.toFixed(2)}) = ${sp.gain.toFixed(2)}`, "apply at the root"]];
        caps = [`mixed data (${metricName} = ${sp.parent.toFixed(2)})`, `best cut: ${names[sp.feat]} ≤ ${sp.thr.toFixed(2)}`, `two purer sides (gain ${sp.gain.toFixed(2)})`, sp2 ? "recurse → cut again" : "leaves are pure enough"];
        viz = (s) => svg([
          ...rowsAll.filter((i) => idx.includes(i)).map((i, k) => <circle key={"p" + k} cx={sx(f0.z[i])} cy={sy(f1.z[i])} r={4} fill={isReg ? "#5b7cff" : clsColor(i)} opacity={s >= 2 ? (X[i][sp.feat] <= sp.thr ? 0.9 : 0.5) : 0.8} />),
          s >= 1 ? (spAxis === "x" ? <line key="cut" x1={sx(thrPos)} y1={PYt} x2={sx(thrPos)} y2={PYb} stroke="#5b7cff" strokeWidth={2} /> : <line key="cut" x1={PX0} y1={sy(thrPos)} x2={PX1} y2={sy(thrPos)} stroke="#5b7cff" strokeWidth={2} />) : null,
          s >= 1 ? <text key="cutl" x={spAxis === "x" ? sx(thrPos) + 4 : PX0 + 4} y={PYt + 12} fill="var(--faint)" fontSize={11} fontFamily="monospace">{names[sp.feat].slice(0, 10)} ≤ {sp.thr.toFixed(1)}</text> : null,
          s >= 3 && sp2 ? (sp2.feat === idxF
            ? <line key="cut2" x1={sx(thrPosOf(sp2.feat, sp2.thr))} y1={PYt} x2={sx(thrPosOf(sp2.feat, sp2.thr))} y2={PYb} stroke="#3ecf7f" strokeWidth={1.5} strokeDasharray="5 3" />
            : <line key="cut2" x1={PX0} y1={sy(thrPosOf(sp2.feat, sp2.thr))} x2={PX1} y2={sy(thrPosOf(sp2.feat, sp2.thr))} stroke="#3ecf7f" strokeWidth={1.5} strokeDasharray="5 3" />) : null,
        ]);
      } else { // forest
        const m = Math.max(1, Math.round(Number(params.nTrees) || 3));
        const shown = Math.min(3, m); const tx = [110, 200, 290].slice(0, shown);
        const votes = tx.map((_, i) => (i % 3 === 2 ? 0 : 1));
        const winner = votes.filter((v) => v === 1).length * 2 >= votes.length ? (classes?.[1] ?? "B") : (classes?.[0] ?? "A");
        intro = "Train many trees, each on a bootstrap resample with a random feature subset, then average their votes to cut variance.";
        legend = [
          { sym: "m", desc: "number of trees", how: "your n_estimators", val: String(m) },
          { sym: "\\text{bootstrap}", desc: "resample rows", how: "sample n rows with replacement", val: `${n} → ${n}` },
          { sym: "T_i", desc: "the i-th tree", how: "fit on its own sample", val: `${shown} shown` },
          { sym: "\\hat y", desc: isReg ? "mean of trees" : "majority vote", how: "aggregate the m outputs", val: isReg ? "avg" : String(winner) },
        ];
        formulas = [["\\text{each } T_i:\\ \\text{bootstrap} + \\text{random features}", "de-correlate the trees"], ["\\text{every tree predicts}", `${shown} independent votes`], isReg ? ["\\hat y = \\tfrac1m\\textstyle\\sum_i T_i(\\mathbf x)", "average the trees"] : ["\\hat y = \\text{mode}\\{T_1,\\dots,T_m\\}", "majority wins"]];
        caps = [`${shown} of ${m} trees, each its own sample`, "each tree casts a vote", "tally the votes", isReg ? "average → ŷ" : `majority → ${winner}`];
        viz = (s) => svg(tx.map((x, i) => [
          <line key={"a" + i} x1={x} y1={70} x2={x - 18} y2={112} stroke="var(--faint)" strokeWidth={1.5} />,
          <line key={"b" + i} x1={x} y1={70} x2={x + 18} y2={112} stroke="var(--faint)" strokeWidth={1.5} />,
          <circle key={"n" + i} cx={x} cy={70} r={5} fill="#a855f7" />,
          <text key={"t" + i} x={x - 8} y={58} fill="var(--faint)" fontSize={11} fontFamily="monospace">T{i + 1}</text>,
          s >= 1 ? <circle key={"vl" + i} cx={x - 18} cy={124} r={6} fill={PAL_ML[votes[i]]} /> : null,
          s >= 1 ? <circle key={"vr" + i} cx={x + 18} cy={124} r={6} fill={PAL_ML[votes[i]]} /> : null,
          s >= 1 ? <text key={"vt" + i} x={x - 10} y={150} fill={PAL_ML[votes[i]]} fontSize={11} fontFamily="monospace">→ {classes?.[votes[i]] ?? (votes[i] ? "B" : "A")}</text> : null,
          s >= 3 && i === 1 ? <text key="win" x={x - 44} y={196} fill="#f59e0b" fontSize={13} fontFamily="monospace">ŷ = {isReg ? "mean" : winner} (majority)</text> : null,
        ]));
      }
    }

    const maxStage = caps.length - 1;
    return mCard(`${label} — how it learns`, <>
      <div className="note" style={{ marginBottom: 10, lineHeight: 1.6 }}>{intro}</div>
      {varLegend(legend)}
      <div className="prep-2col">
        <div className="prep-col">
          <div className="prep-col-h">the formula, step by step</div>
          {formulas.map(([tex, cap], i) => <div key={i} className={`fx-line ${st >= i ? "on" : ""}`}><Katex block tex={tex} /><div className="note" style={{ marginTop: 2 }}>{cap}</div></div>)}
        </div>
        <div className="prep-col">
          <div className="prep-col-h">on your data</div>
          {viz(st)}
          <div className="note" style={{ marginTop: 8 }}>{caps[Math.min(st, maxStage)]}</div>
        </div>
      </div>
      {modelStageControls(maxStage)}
      {kind === "gd" && !isReg && sigmoidView()}
    </>);
  }
  function mathValidation() {
    return mCard(`${valMethod === "kfold" ? `${cvFolds}-fold cross-validation` : "hold-out validation"}`, <>
      {valMethod === "kfold" ? <>
        <div className="note" style={{ marginBottom: 8 }}>Split the data into {cvFolds} folds; train on {cvFolds - 1}, score on the held-out one, rotate, then average:</div>
        <div className="mathrow"><Katex block tex={`\\text{CV} = \\tfrac{1}{k}\\sum_{f=1}^{k} \\text{score}_f`} /></div>
        {cvResult.length > 0 && <div className="mathrow"><Katex tex={`\\text{CV} = \\tfrac{1}{${cvResult.length}}(${cvResult.map((r) => r.score.toFixed(2)).join("+")}) = ${(cvResult.reduce((a, r) => a + r.score, 0) / cvResult.length).toFixed(3)}`} /></div>}
      </> : <>
        <div className="mathrow"><Katex block tex={`n_{\\text{test}} = ${Math.round((ds?.nrows || 0) * testSize)}, \\quad n_{\\text{train}} = ${(ds?.nrows || 0) - Math.round((ds?.nrows || 0) * testSize)}`} /></div>
        <div className="note">A fixed {(testSize * 100).toFixed(0)}% of rows is held out and never seen during training.</div>
      </>}
    </>);
  }
  function mathTrain() {
    if (!ds || !result) return <div className="note">Train the model to see the numbers.</div>;
    let gt = null, sp = null, X0: number[] | undefined;
    try { const b = buildMatrix(ds, features, target, task, steps); const nc = b.classes?.length || 0; gt = gdTrace(cfgNow(), b.X, b.y, nc); sp = modelRef.current ? rootSplitMath(modelRef.current, b.X, b.y, nc) : null; X0 = b.X[0]; } catch { /* ignore */ }
    void X0;
    if (gt && gt.snaps.length) {
      const last = gt.snaps[gt.snaps.length - 1];
      return mCard("gradient descent — watch w update", <>
        <div className="note" style={{ marginBottom: 8 }}>Each epoch nudges every weight <b>downhill</b> on the loss by the gradient, scaled by the learning rate <Katex tex={`\\eta=${gt.lr}`} />:</div>
        <div className="mathrow"><Katex block tex={`\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta\\,\\nabla_{\\mathbf w}\\mathcal{L}`} /></div>
        {gdAnimView()}
        <div className="mathrow" style={{ marginTop: 10 }}><Katex tex={`\\text{first step: } w_1: 0 - ${gt.lr}\\times(${gt.grad0[1]?.toFixed(3) ?? "0"}) = ${(-gt.lr * (gt.grad0[1] ?? 0)).toFixed(4)}`} /></div>
        <label className="fld" style={{ marginTop: 12 }}>Per-epoch (loss ↓ as the gradient shrinks)</label>
        <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
          <tr><th>epoch</th><th>loss</th><th>‖∇‖</th><th>b</th><th>w₁</th><th>w₂</th></tr>
          {gt.snaps.map((s) => <tr key={s.ep}><td>{s.ep}</td><td className="mono">{s.loss.toFixed(4)}</td><td className="mono">{s.gnorm.toFixed(4)}</td><td className="mono">{s.w[0]?.toFixed(3)}</td><td className="mono">{s.w[1]?.toFixed(3)}</td><td className="mono">{s.w[2]?.toFixed(3)}</td></tr>)}
        </tbody></table></div>
        <div className="note" style={{ marginTop: 8 }}>After {gt.snaps[gt.snaps.length - 1].ep + 1} epochs the loss settled at <b>{last.loss.toFixed(4)}</b> and the gradient is tiny ({last.gnorm.toFixed(4)}) — it has converged.</div>
      </>);
    }
    if (sp) {
      return mCard("the best split (greedy)", <>
        <div className="note" style={{ marginBottom: 8 }}>At the root the tree tries every feature/threshold and keeps the one that reduces impurity most:</div>
        <div className="mathrow"><Katex block tex={`\\text{gain} = ${sp.parent.toFixed(3)} - \\tfrac{${sp.nL}}{${sp.nL + sp.nR}}(${sp.left.toFixed(3)}) - \\tfrac{${sp.nR}}{${sp.nL + sp.nR}}(${sp.right.toFixed(3)}) = ${sp.gain.toFixed(3)}`} /></div>
        {(() => { const t = plotlyTheme(); return <Plot data={[{ type: "bar", x: ["parent", `left (n=${sp.nL})`, `right (n=${sp.nR})`], y: [sp.parent, sp.left, sp.right], marker: { color: ["#5b7cff", "#3ecf7f", "#f59e0b"] } }]} layout={{ ...chartLayout(t, `${sp.metric} impurity — the split makes each side purer`, "", sp.metric), showlegend: false }} style={{ height: 220 }} />; })()}
        <div className="note">Chosen split: feature <b>#{sp.feat}</b> ≤ {sp.thr.toFixed(3)} · {sp.metric} drops from {sp.parent.toFixed(3)} to a weighted {(sp.parent - sp.gain).toFixed(3)}. This repeats recursively down the tree.</div>
      </>);
    }
    if (modelRef.current?.kind === "gnb") {
      const m = modelRef.current;
      return mCard("class statistics learned", <>
        <div className="note" style={{ marginBottom: 8 }}>“Training” is just measuring each class’s prior and each feature’s mean &amp; variance:</div>
        <div style={{ overflowX: "auto" }}><table className="dtable"><tbody><tr><th>class</th><th>prior P(y)</th><th>μ (feat 1)</th><th>σ² (feat 1)</th></tr>
          {m.priors.map((pr, k) => <tr key={k}><td>{trained?.classes?.[k] ?? k}</td><td className="mono">{pr.toFixed(3)}</td><td className="mono">{m.means[k][0]?.toFixed(3)}</td><td className="mono">{m.vars[k][0]?.toFixed(3)}</td></tr>)}
        </tbody></table></div>
      </>);
    }
    return mCard("lazy learner (KNN)", <div className="note">KNN does <b>no training</b> — it just stores the {ds.nrows} rows. All the work happens at prediction time: compute the distance to every stored point, keep the {params.n_neighbors || 5} closest, and vote.</div>);
  }
  function mathPredict() {
    if (!ds || !result || !modelRef.current) return null;
    let X0: number[] = [], names: string[] = [];
    try { const b = buildMatrix(ds, features, target, task, steps); X0 = b.X[0]; names = b.featureNames; } catch { return null; }
    const m = modelRef.current;
    if (m.kind === "logreg" || m.kind === "linear") {
      const w = m.kind === "logreg" ? m.W[Math.min(1, m.classes - 1)] : m.w;
      const terms = X0.slice(0, 3).map((x, j) => `${w[j + 1]?.toFixed(2)}\\!\\times\\!${x.toFixed(2)}`).join(" + ");
      const z = w[0] + X0.reduce((a, x, j) => a + (w[j + 1] || 0) * x, 0);
      return mCard("predict one row — the arithmetic", <>
        <div className="mathrow"><Katex block tex={`z = b + \\sum_j w_j x_j = ${w[0]?.toFixed(2)} + ${terms} + \\dots = ${z.toFixed(3)}`} /></div>
        {m.kind === "logreg"
          ? <div className="mathrow"><Katex block tex={`\\hat{y} = \\sigma(${z.toFixed(2)}) = \\frac{1}{1+e^{-(${z.toFixed(2)})}} = ${(1 / (1 + Math.exp(-z))).toFixed(3)}`} /></div>
          : <div className="mathrow"><Katex block tex={`\\hat{y} = ${z.toFixed(3)}`} /></div>}
        <div className="note">Using the trained weights on the first row’s {names.length} features (first 3 terms shown). That’s the whole prediction — a weighted sum, then a squash for classification.</div>
      </>);
    }
    return null;
  }

  // Pure-Python (stdlib only) from-scratch implementation of the current model.
  function buildScratchCode(): string {
    const num = ds ? ds.columns.filter((c) => c.type === "num" && features.includes(c.name) && c.name !== target).map((c) => c.name) : [];
    const cat = ds ? ds.columns.filter((c) => c.type === "cat" && features.includes(c.name) && c.name !== target).map((c) => c.name) : [];
    const cls = task === "classification";
    const pre = `# AI Workbench · ${MODEL_INFO[algo]?.label ?? algo} — FROM SCRATCH (pure Python, stdlib only)
import csv, math, random
random.seed(42)

NUM = ${JSON.stringify(num)}
CAT = ${JSON.stringify(cat)}
TARGET = ${JSON.stringify(target)}

rows = list(csv.DictReader(open("data.csv")))   # <-- only "package" used: read the file

# ── standardize numeric: z = (x - mean) / std ──
means, stds = {}, {}
for c in NUM:
    xs = [float(r[c]) for r in rows if r[c] not in ("", None)]
    m = sum(xs) / len(xs); s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
    means[c], stds[c] = m, s
cats = {c: sorted({r[c] for r in rows}) for c in CAT}   # one-hot categories

def featurize(r):
    v = []
    for c in NUM:
        x = float(r[c]) if r[c] not in ("", None) else means[c]
        v.append((x - means[c]) / stds[c])
    for c in CAT:
        v += [1.0 if r[c] == cat else 0.0 for cat in cats[c]]
    return v

X = [featurize(r) for r in rows]
${cls ? `classes = sorted({r[TARGET] for r in rows}); cidx = {c: i for i, c in enumerate(classes)}
y = [cidx[r[TARGET]] for r in rows]` : `y = [float(r[TARGET]) for r in rows]`}

idx = list(range(len(X))); random.shuffle(idx)
cut = int(len(idx) * ${1 - testSize}); tr, te = idx[:cut], idx[cut:]
d = len(X[0])`;

    const p = (k: string, def: number) => Number(params[k]) || def;
    let body = "";
    if (algo === "LogisticRegression") body = `
# ── logistic regression via gradient descent ──
K = len(classes); W = [[0.0] * (d + 1) for _ in range(K)]
def softmax(z):
    mx = max(z); e = [math.exp(v - mx) for v in z]; s = sum(e); return [v / s for v in e]
def fwd(x):
    xb = [1.0] + x; return softmax([sum(w[j] * xb[j] for j in range(d + 1)) for w in W])
lr, L2 = ${p("learning_rate", 0.2)}, ${(0.01 / p("C", 1)).toFixed(4)}
for ep in range(${p("max_iter", 300)}):
    G = [[0.0] * (d + 1) for _ in range(K)]
    for i in tr:
        xb = [1.0] + X[i]; pr = fwd(X[i])
        for k in range(K):
            err = pr[k] - (1.0 if y[i] == k else 0.0)
            for j in range(d + 1): G[k][j] += err * xb[j]
    for k in range(K):
        for j in range(d + 1):
            g = G[k][j] / len(tr) + (L2 * W[k][j] if j > 0 else 0.0)
            W[k][j] -= lr * g
def predict(x): p = fwd(x); return p.index(max(p))
acc = sum(1 for i in te if predict(X[i]) == y[i]) / len(te)
print("accuracy:", round(acc, 3))`;
    else if (algo === "LinearRegression" || algo === "Ridge") body = `
# ── linear regression via gradient descent ──
w = [0.0] * (d + 1); lr, alpha = 0.05, ${(algo === "Ridge" ? 0.01 * p("alpha", 1) : 0).toFixed(4)}
for ep in range(400):
    G = [0.0] * (d + 1)
    for i in tr:
        xb = [1.0] + X[i]; err = sum(w[j] * xb[j] for j in range(d + 1)) - y[i]
        for j in range(d + 1): G[j] += err * xb[j]
    for j in range(d + 1):
        g = G[j] / len(tr) + (alpha * w[j] if j > 0 else 0.0); w[j] -= lr * g
def predict(x): xb = [1.0] + x; return sum(w[j] * xb[j] for j in range(d + 1))
mte = sum(y[i] for i in te) / len(te)
ss = sum((y[i] - predict(X[i])) ** 2 for i in te); st = sum((y[i] - mte) ** 2 for i in te)
print("R2:", round(1 - ss / st, 3))`;
    else if (algo.startsWith("KNeighbors")) body = `
# ── k-nearest neighbours (no training) ──
k = ${p("n_neighbors", 5)}
def predict(x):
    nn = sorted((sum((X[i][j] - x[j]) ** 2 for j in range(d)) ** 0.5, y[i]) for i in tr)[:k]
    ${cls ? `labels = [lab for _, lab in nn]; return max(set(labels), key=labels.count)` : `return sum(lab for _, lab in nn) / k`}
${cls ? `acc = sum(1 for i in te if predict(X[i]) == y[i]) / len(te); print("accuracy:", round(acc, 3))` : `mte = sum(y[i] for i in te)/len(te); ss=sum((y[i]-predict(X[i]))**2 for i in te); st=sum((y[i]-mte)**2 for i in te); print("R2:", round(1-ss/st,3))`}`;
    else if (algo === "GaussianNB") body = `
# ── Gaussian Naive Bayes (measure class stats) ──
K = len(classes)
means_ = [[0.0] * d for _ in range(K)]; varr = [[0.0] * d for _ in range(K)]; cnt = [0] * K
for i in tr:
    cnt[y[i]] += 1
    for j in range(d): means_[y[i]][j] += X[i][j]
for k in range(K):
    for j in range(d): means_[k][j] /= cnt[k] or 1
for i in tr:
    for j in range(d): varr[y[i]][j] += (X[i][j] - means_[y[i]][j]) ** 2
for k in range(K):
    for j in range(d): varr[k][j] = varr[k][j] / (cnt[k] or 1) + 1e-6
prior = [c / len(tr) for c in cnt]
def predict(x):
    best = (-1e18, 0)
    for k in range(K):
        ll = math.log(prior[k] or 1e-9)
        for j in range(d): ll += -0.5 * math.log(2 * math.pi * varr[k][j]) - (x[j] - means_[k][j]) ** 2 / (2 * varr[k][j])
        if ll > best[0]: best = (ll, k)
    return best[1]
acc = sum(1 for i in te if predict(X[i]) == y[i]) / len(te)
print("accuracy:", round(acc, 3))`;
    else body = `
# ── CART decision tree${algo === "RandomForest" ? " / random forest" : ""} from scratch ──
K = ${cls ? "len(classes)" : "1"}
def impurity(ys):
    ${cls ? `if not ys: return 0.0
    c = [0] * K
    for v in ys: c[v] += 1
    return 1 - sum((ci / len(ys)) ** 2 for ci in c)` : `if not ys: return 0.0
    m = sum(ys) / len(ys); return sum((v - m) ** 2 for v in ys) / len(ys)`}
def leaf(ys):
    ${cls ? `c = [0] * K
    for v in ys: c[v] += 1
    return c.index(max(c))` : `return sum(ys) / len(ys)`}
def build(rowsi, depth, feats):
    ys = [y[i] for i in rowsi]
    if depth >= ${p("max_depth", algo === "RandomForest" ? 6 : 5)} or len(rowsi) < ${p("min_samples_split", 2)} or len(set(ys)) <= 1:
        return ("leaf", leaf(ys))
    base = impurity(ys); best = (1e-9, None, None)
    for f in feats:
        vals = sorted(set(X[i][f] for i in rowsi))
        for a, b in zip(vals, vals[1:]):
            thr = (a + b) / 2
            L = [i for i in rowsi if X[i][f] <= thr]; R = [i for i in rowsi if X[i][f] > thr]
            if not L or not R: continue
            gain = base - len(L)/len(rowsi)*impurity([y[i] for i in L]) - len(R)/len(rowsi)*impurity([y[i] for i in R])
            if gain > best[0]: best = (gain, f, thr)
    if best[1] is None: return ("leaf", leaf(ys))
    _, f, thr = best
    L = [i for i in rowsi if X[i][f] <= thr]; R = [i for i in rowsi if X[i][f] > thr]
    return ("node", f, thr, build(L, depth+1, feats), build(R, depth+1, feats))
def pred_tree(t, x):
    while t[0] == "node":
        t = t[3] if x[t[1]] <= t[2] else t[4]
    return t[1]
${algo === "RandomForest" ? `import random as _r
mfeat = max(1, int(d ** 0.5)); trees = []
for _ in range(${p("n_estimators", 25)}):
    samp = [_r.choice(tr) for _ in tr]; fs = _r.sample(range(d), mfeat)
    trees.append(build(samp, 0, fs))
def predict(x):
    votes = [pred_tree(t, x) for t in trees]
    ${cls ? `return max(set(votes), key=votes.count)` : `return sum(votes) / len(votes)`}` : `tree = build(tr, 0, list(range(d)))
def predict(x): return pred_tree(tree, x)`}
${cls ? `acc = sum(1 for i in te if predict(X[i]) == y[i]) / len(te); print("accuracy:", round(acc, 3))` : `mte = sum(y[i] for i in te)/len(te); ss=sum((y[i]-predict(X[i]))**2 for i in te); st=sum((y[i]-mte)**2 for i in te); print("R2:", round(1-ss/st,3))`}`;
    return pre + "\n" + body + "\n";
  }
  function copyScratch() { navigator.clipboard.writeText(buildScratchCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  function downloadScratch() { const blob = new Blob([buildScratchCode()], { type: "text/x-python" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ml_from_scratch.py"; a.click(); URL.revokeObjectURL(a.href); }

  function download() { const blob = new Blob([buildCode()], { type: "text/x-python" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ml_workflow.py"; a.click(); URL.revokeObjectURL(a.href); }
  function copyCode() { navigator.clipboard.writeText(buildCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  function downloadBlob2(data: BlobPart, filename: string, mime: string) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([data], { type: mime })); a.download = filename; a.click(); URL.revokeObjectURL(a.href); }

  const cfgNow = (): TrainConfig => ({ task, algo, params, testSize, cvFolds });
  const PAL = ["#5b7cff", "#f59e0b", "#3ecf7f", "#ef4444", "#a855f7", "#22b8cf", "#ec4899", "#84cc16"];
  const numFeats = () => (ds ? features.filter((f) => ds.columns.find((c) => c.name === f)?.type === "num" && f !== target) : []);

  function runBoundary() {
    if (!ds) return;
    const nums = numFeats();
    if (nums.length < 2) { setMsg("Need at least two numeric features for a decision boundary."); return; }
    const a = dbF1 && nums.includes(dbF1) ? dbF1 : nums[0];
    const b = dbF2 && nums.includes(dbF2) && dbF2 !== a ? dbF2 : (nums.find((n) => n !== a) || nums[1]);
    setDbF1(a); setDbF2(b);
    const surf = decisionSurface(ds, target, a, b, cfgNow());
    setDbSurf(surf);
    if (!surf) setMsg("Boundary needs two different numeric features and a categorical target (2–8 classes).");
  }
  function boundaryFig() {
    if (!dbSurf) return null;
    const t = plotlyTheme(); const K = dbSurf.classes.length;
    const colorscale: [number, string][] = [];
    for (let i = 0; i < K; i++) { const c = PAL[i % PAL.length]; colorscale.push([i / K, c]); colorscale.push([(i + 1) / K, c]); }
    const data: Record<string, unknown>[] = [
      { type: "heatmap", x: dbSurf.xs, y: dbSurf.ys, z: dbSurf.z, showscale: false, colorscale, zmin: -0.5, zmax: K - 0.5, opacity: 0.4, hoverinfo: "skip" },
      ...dbSurf.classes.map((cl, ci) => ({ type: "scatter", mode: "markers", name: cl, x: dbSurf.points.filter((p) => p.c === ci).map((p) => p.x), y: dbSurf.points.filter((p) => p.c === ci).map((p) => p.y), marker: { size: 7, color: PAL[ci % PAL.length], line: { width: 1, color: t.paper } } })),
    ];
    const layout = { ...chartLayout(t, `Decision boundary · ${MODEL_INFO[algo]?.label ?? algo}`, dbF1, dbF2), showlegend: true, legend: { font: { color: t.text, size: 10 }, orientation: "h" } };
    return { data, layout };
  }
  function runLC() {
    if (!ds) return;
    const use = lcFeats.length ? lcFeats : features;
    if (!use.length) { setMsg("Pick at least one feature for the learning curve."); return; }
    try { const b = buildMatrix(ds, use, target, task, steps); setLcData(learningCurve(b.X, b.y, cfgNow(), b.classes?.length || 0)); }
    catch (e) { setMsg("Learning curve error: " + (e as Error).message); }
  }
  // Auto-diagnose the learning curve → good fit / overfit / underfit, with the
  // live numbers + the exact rule that fired (updates whenever lcData changes).
  const lcFmt = (v: number) => (task === "classification" ? `${(v * 100).toFixed(0)}%` : v.toFixed(2));
  function lcDiagnosis() {
    if (!lcData.length) return null;
    const last = lcData[lcData.length - 1];
    const gap = last.train - last.test;
    const cls_ = task === "classification";
    const goodThresh = cls_ ? 0.7 : 0.5;
    const gapT = cls_ ? "15%" : "0.15", goodT = cls_ ? "70%" : "0.50";
    const base = { train: last.train, test: last.test, gap };
    if (gap > 0.15) return { ...base, label: "Overfitting", cls: "bad", rule: `gap ${lcFmt(gap)} > ${gapT}`, why: `train (${lcFmt(last.train)}) is much higher than validation (${lcFmt(last.test)}) — it memorises the training rows. Try more data, fewer features, or stronger regularisation (lower C / higher alpha, less depth).` };
    if (last.test < goodThresh && last.train < goodThresh + 0.1) return { ...base, label: "Underfitting", cls: "warn", rule: `validation ${lcFmt(last.test)} < ${goodT} and both lines flat`, why: `even on training data the score is low — the model is too simple. Try a more powerful model, more/better features, or less regularisation.` };
    return { ...base, label: "Good fit", cls: "ok", rule: `gap ${lcFmt(gap)} ≤ ${gapT} and validation ${lcFmt(last.test)} ≥ ${goodT}`, why: `it does about as well on unseen data as on training data — it learned the real pattern and generalises.` };
  }
  function lcFig() {
    if (!lcData.length) return null; const t = plotlyTheme();
    const xs = lcData.map((d) => d.n), tr = lcData.map((d) => d.train), te = lcData.map((d) => d.test);
    const d = lcDiagnosis();
    const clr = d?.cls === "bad" ? "#ef4444" : d?.cls === "warn" ? "#f59e0b" : "#3ecf7f";
    const gapFill = d?.cls === "bad" ? "rgba(239,68,68,0.16)" : d?.cls === "warn" ? "rgba(245,158,11,0.14)" : "rgba(62,207,127,0.12)";
    const data: Record<string, unknown>[] = [
      { x: xs, y: tr, name: "train", mode: "lines+markers", line: { color: "#5b7cff", width: 2 } },
      { x: xs, y: te, name: "validation", mode: "lines+markers", line: { color: "#f59e0b", width: 2 }, fill: "tonexty", fillcolor: gapFill }, // shades the train↔validation gap
    ];
    const layout = {
      ...chartLayout(t, "Learning curve", "training examples", task === "classification" ? "accuracy" : "R²"),
      showlegend: true, legend: { font: { color: t.text, size: 10 }, orientation: "h" },
      annotations: d ? [{
        x: xs[xs.length - 1], y: (tr[tr.length - 1] + te[te.length - 1]) / 2, xref: "x", yref: "y",
        xanchor: "right", yanchor: "middle", text: `◀ ${d.label}<br>gap ${(task === "classification" ? `${Math.round((tr[tr.length - 1] - te[te.length - 1]) * 100)}%` : (tr[tr.length - 1] - te[te.length - 1]).toFixed(2))}`,
        align: "right", showarrow: false, font: { size: 11, color: clr }, bgcolor: t.paper, bordercolor: clr, borderwidth: 1, borderpad: 4,
      }] : [],
    };
    return { data, layout };
  }

  function setStepMethod(op: string, method: string) { setSteps((ss) => { const i = ss.findIndex((s) => s.op === op); if (i < 0) return ss; const n = [...ss]; n[i] = { ...n[i], method }; return n; }); }
  function applyCodeToFlow() {
    const code = codeDirty ? codeDraft : buildCode(); const applied: string[] = [];
    const ts = code.match(/test_size\s*=\s*([0-9.]+)/); if (ts) { setTestSize(Number(ts[1])); applied.push("test_size"); }
    const cvm = code.match(/\bcv\s*=\s*(\d+)/); if (cvm) { setCvFolds(Number(cvm[1])); applied.push("cv folds"); }
    const algoMap: Record<string, string> = { LogisticRegression: "LogisticRegression", LinearRegression: "LinearRegression", Ridge: "Ridge", KNeighborsClassifier: "KNeighborsClassifier", KNeighborsRegressor: "KNeighborsRegressor", GaussianNB: "GaussianNB", DecisionTreeClassifier: "DecisionTree", DecisionTreeRegressor: "DecisionTree", RandomForestClassifier: "RandomForest", RandomForestRegressor: "RandomForest" };
    for (const [cls, key] of Object.entries(algoMap)) { if (new RegExp(`\\b${cls}\\s*\\(`).test(code)) { if (MODELS[task][key]) { setAlgo(key); setParamsFor(task, key); applied.push("algorithm→" + key); } break; } }
    const pnum = (re: RegExp, name: string) => { const m = code.match(re); if (m) { setParams((p) => ({ ...p, [name]: m[1] })); applied.push(name); } };
    pnum(/n_estimators\s*=\s*(\d+)/, "n_estimators"); pnum(/max_depth\s*=\s*(\d+)/, "max_depth"); pnum(/min_samples_split\s*=\s*(\d+)/, "min_samples_split"); pnum(/n_neighbors\s*=\s*(\d+)/, "n_neighbors"); pnum(/max_iter\s*=\s*(\d+)/, "max_iter"); pnum(/\bC\s*=\s*([0-9.]+)/, "C"); pnum(/\balpha\s*=\s*([0-9.]+)/, "alpha");
    const scMap: Record<string, string> = { StandardScaler: "StandardScaler", MinMaxScaler: "MinMaxScaler", RobustScaler: "RobustScaler", MaxAbsScaler: "MaxAbsScaler", QuantileTransformer: "QuantileUniform" };
    for (const [cls, method] of Object.entries(scMap)) { if (new RegExp(`\\b${cls}\\s*\\(`).test(code)) { setStepMethod("Scale / normalize", method); applied.push("scaler→" + method); break; } }
    if (/OrdinalEncoder\s*\(/.test(code)) { setStepMethod("Encode categorical", "Ordinal"); applied.push("encoder→Ordinal"); }
    else if (/OneHotEncoder\s*\(/.test(code)) { setStepMethod("Encode categorical", "One-Hot"); applied.push("encoder→One-Hot"); }
    setCodeDirty(false);
    setMsg(applied.length ? `Applied to flow: ${applied.join(", ")}. Re-run training to see the effect.` : "No recognized settings found. Syncable knobs: algorithm, hyperparameters, test_size, cv, scaler, encoder.");
  }
  function modelBundle() { const m = modelRef.current; if (!m || !trained) return null; return { ...m, _meta: { algo: trained.algo, task: trained.task, features: trained.featureNames, classes: trained.classes, testSize, params, source: "AI Workbench ML Lab" } }; }
  function exportPkl() { const obj = modelBundle(); if (!obj) { setMsg("Train a model first."); return; } downloadBlob2(pickle(obj) as BlobPart, "model.pkl", "application/octet-stream"); }
  function exportJson() { const obj = modelBundle(); if (!obj) { setMsg("Train a model first."); return; } downloadBlob2(JSON.stringify(obj, null, 2), "model.json", "application/json"); }
  async function saveProject() {
    const config = { dsName, target, task, features, steps, algo, params, testSize, cvFolds };
    try { const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "ml", name: dsName || "ML build", config }) }); setSavedMsg(r.ok ? "Saved ✓" : "Save failed"); }
    catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2000);
  }

  const stepBtn = (k: Step, n: number, label: string) => (<button className={step === k ? "on" : ""} disabled={!ds} onClick={() => setStep(k)}><b>{n}</b>{label}</button>);
  const cvMean = result ? result.cv.reduce((a, b) => a + b, 0) / (result.cv.length || 1) : 0;
  const cur = snaps[Math.min(ppIdx, snaps.length - 1)];
  const FLOW = [
    { w: "Load preprocessed matrix", d: "the encoded/scaled feature matrix enters the trainer" },
    { w: "Split train / test", d: `hold out ${Math.round(testSize * 100)}% as an unseen test set` },
    { w: `Initialise ${algo}`, d: "set up parameters & (for GD models) zero the weights" },
    { w: `Fit ${algo} on train`, d: modelKind === "gd" ? "gradient descent minimises the loss over epochs" : modelKind === "tree" ? "recursively split features to reduce impurity" : modelKind === "forest" ? "grow many trees on bootstrapped samples" : "store the training data / class statistics" },
    { w: "Predict on test set", d: "run the fitted model on the held-out rows" },
    { w: "Evaluate metrics", d: task === "classification" ? "accuracy · precision · recall · F1" : "R² · MAE · RMSE" },
    { w: "Done", d: "results & feature importance ready" },
  ];

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 04 · runs in your browser</div><h2 className="page-h">ML Lab</h2><p className="page-sub" style={{ margin: 0 }}>Connect data, explore, preprocess (with a live pipeline animation), tune a model, and watch it train — all real, in your browser.</p></div>
        <div className="acts"><button className="btn ghost sm" onClick={saveProject}>{savedMsg || "💾 Save"}</button><button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code</button></div>
      </div>
      {msg && <div className="err">{msg}</div>}
      <div className="row" style={{ alignItems: "center", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
        <div className="seg" style={{ width: 300 }}>
          <button className={mlMode === "package" ? "on" : ""} onClick={() => setMlMode("package")}>📦 Package (sklearn)</button>
          <button className={mlMode === "maths" ? "on" : ""} onClick={() => setMlMode("maths")}>🧮 From scratch (maths)</button>
        </div>
        <span className="note" style={{ flex: 1, minWidth: 240 }}>{mlMode === "maths"
          ? "Same 7 steps — but every step shows the actual formulas with your numbers, no libraries. Code export is pure Python."
          : "The production workflow with scikit-learn. Switch to “From scratch” to see the maths under every step."}</span>
      </div>
      <div className="teach-note"><span className="ic">🎓</span><span><b>Teaching engine.</b> {mlMode === "maths" ? <>Every model here is implemented <b>from scratch</b> — these cards show the exact equations and intermediate numbers behind each step.</> : <>Models train from scratch in your browser so every step is inspectable. The <b>Get code</b> export uses scikit-learn.</>}</span></div>
      <div className="stepper"><button className={step === "data" ? "on" : ""} onClick={() => setStep("data")}><b>1</b>Data</button>{stepBtn("eda", 2, "EDA")}{stepBtn("prep", 3, "Preprocessing")}{stepBtn("model", 4, "Model")}{stepBtn("validation", 5, "Validation")}{stepBtn("train", 6, "Train")}<button className={step === "deploy" ? "on" : ""} disabled={!result} onClick={() => setStep("deploy")}><b>7</b>Test &amp; Export</button></div>

      {mlMode === "maths" && step === "prep" && mathPrep()}
      {mlMode === "maths" && step === "model" && mathModel()}
      {mlMode === "maths" && step === "validation" && mathValidation()}
      {mlMode === "maths" && step === "train" && mathTrain()}
      {mlMode === "maths" && step === "deploy" && mathPredict()}

      {/* STEP 1 DATA */}
      {step === "data" && (
        <div className="card">
          <div className="card-h"><span className="t">Connect your data</span><div className="tabs"><button className={dataTab === "sample" ? "on" : ""} onClick={() => setDataTab("sample")}>Sample</button><button className={dataTab === "upload" ? "on" : ""} onClick={() => setDataTab("upload")}>Upload CSV</button><button className={dataTab === "db" ? "on" : ""} onClick={() => setDataTab("db")}>Database</button></div></div>
          <div className="card-b">
            {dataTab === "sample" && (<><label className="fld">Sample dataset</label><div className="row" style={{ gap: 8 }}><select value={sampleKey} onChange={(e) => setSampleKey(e.target.value)}>{sampleDatasets().map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}</select><button className="btn sm" onClick={loadSample}>Load</button></div></>)}
            {dataTab === "upload" && (<><div className="dropzone" onClick={() => fileRef.current?.click()}>Click to upload a <b>CSV</b> file (parsed &amp; trained locally)</div><input ref={fileRef} type="file" accept=".csv,.tsv,text/csv" onChange={onFile} style={{ display: "none" }} /></>)}
            {dataTab === "db" && (<><label className="fld">MySQL / TiDB connection URL</label><input type="text" value={dbUrl} onChange={(e) => setDbUrl(e.target.value)} /><label className="fld" style={{ marginTop: 10 }}>Query</label><textarea rows={2} value={dbQ} onChange={(e) => setDbQ(e.target.value)} /><div style={{ marginTop: 10 }}><button className="btn sm" onClick={runDbQuery} disabled={dbBusy}>{dbBusy ? "Running…" : "Run query & load"}</button></div></>)}
            {ds && (<>
              <div className="row" style={{ gap: 10, flexWrap: "wrap", margin: "16px 0 12px", alignItems: "center" }}>
                <span className="pill"><span className="dot" />{dsName}</span><span className="pill">rows {ds.nrows}</span><span className="pill">cols {ds.columns.length}</span>
                <div className="rangebtns" style={{ marginLeft: "auto" }}><button className={viewMode === "head" ? "on" : ""} onClick={() => setViewMode("head")}>Head</button><button className={viewMode === "tail" ? "on" : ""} onClick={() => setViewMode("tail")}>Tail</button><button className={viewMode === "range" ? "on" : ""} onClick={() => setViewMode("range")}>Range</button></div>
                {viewMode !== "range" ? <span className="row" style={{ gap: 6 }}><span className="note">rows</span><input type="number" value={nRows} min={1} max={ds.nrows} onChange={(e) => setNRows(Math.max(1, +e.target.value))} style={{ width: 70 }} /></span>
                  : <span className="row" style={{ gap: 6 }}><span className="note">from</span><input type="number" value={rFrom} min={0} max={ds.nrows} onChange={(e) => setRFrom(+e.target.value)} style={{ width: 70 }} /><span className="note">to</span><input type="number" value={rTo} min={1} max={ds.nrows} onChange={(e) => setRTo(+e.target.value)} style={{ width: 70 }} /></span>}
              </div>
              <div className="row" style={{ gap: 10, alignItems: "center", marginBottom: 12, flexWrap: "wrap", background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, padding: "9px 12px" }}>
                <label className="fld" style={{ margin: 0 }}>🎯 Target column <span className="note">— what the model predicts</span></label>
                <select value={target} onChange={(e) => pickTarget(e.target.value)} style={{ maxWidth: 220 }}>{ds.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}</select>
                <span className="badge">{task}</span>
                <span className="note">{features.length} feature{features.length === 1 ? "" : "s"} · task auto-detected from the column</span>
              </div>
              {task === "classification" && (() => {
                const col = ds.columns.find((c) => c.name === target); if (!col) return null;
                const vals = col.values.filter((v) => v != null).map(String);
                const counts = new Map<string, number>(); vals.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
                const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]); const total = vals.length || 1;
                const maxPct = entries[0] ? entries[0][1] / total : 0;
                const PAL = ["#5b7cff", "#f59e0b", "#3ecf7f", "#ef4444", "#a855f7", "#22b8cf", "#ec4899", "#84cc16"];
                return (
                  <div style={{ marginBottom: 14 }}>
                    <label className="fld">Class balance · {target} ({entries.length} classes)</label>
                    <div style={{ display: "flex", height: 16, borderRadius: 5, overflow: "hidden", border: "1px solid var(--border)" }}>{entries.slice(0, 8).map(([k, c], i) => <div key={k} title={`${k}: ${c}`} style={{ width: `${(c / total) * 100}%`, background: PAL[i % PAL.length] }} />)}</div>
                    <div className="chips" style={{ marginTop: 6 }}>{entries.slice(0, 8).map(([k, c], i) => <span key={k} className="chip" style={{ cursor: "default" }}><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: PAL[i % PAL.length], marginRight: 6, verticalAlign: "middle" }} />{k} · {((c / total) * 100).toFixed(0)}%</span>)}</div>
                    {maxPct > 0.65 && <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">⚠️</span><span><b>Imbalanced</b> — {(maxPct * 100).toFixed(0)}% is one class. A model can score high accuracy just by predicting the majority, so judge it on <b>precision / recall / F1</b> and the confusion matrix instead.</span></div>}
                  </div>
                );
              })()}
              <label className="fld">Column overview</label>
              <div className="col-ov">{ds.columns.map((c) => { const s = colStats(c); return (
                <div key={c.name} className={`col-ov-card ${c.name === target ? "tgt" : ""}`}>
                  <div className="row" style={{ justifyContent: "space-between", alignItems: "center", gap: 6 }}><b style={{ fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={c.name}>{c.name}</b><span className="badge">{c.type}</span></div>
                  {s.type === "num"
                    ? <div className="note" style={{ marginTop: 4 }}>{s.min.toFixed(1)} – {s.max.toFixed(1)} · μ {s.mean.toFixed(2)}</div>
                    : <div className="note" style={{ marginTop: 4 }}>{s.unique} unique · top “{s.top[0]?.[0] ?? "—"}”</div>}
                  <div className="note" style={{ marginTop: 2, color: s.missing ? "var(--crit)" : "var(--faint)" }}>{s.missing} missing{c.name === target ? " · 🎯 target" : ""}</div>
                </div>
              ); })}</div>
              <div style={{ overflowX: "auto" }}><table className="dtable"><tbody><tr><th>#</th>{ds.columns.map((c) => <th key={c.name}>{c.name} <span style={{ color: c.name === target ? "var(--accent)" : "var(--faint)" }}>{c.type}{c.name === target ? "·target" : ""}</span></th>)}</tr>{viewRows.map((r) => <tr key={r}><td style={{ color: "var(--faint)" }}>{r}</td>{ds.columns.map((c) => <td key={c.name}>{c.values[r] ?? "—"}</td>)}</tr>)}</tbody></table></div>
              <label className="fld" style={{ marginTop: 16 }}>Summary statistics (numeric columns · like pandas .describe())</label>
              <div style={{ overflowX: "auto" }}><table className="dtable"><tbody><tr><th>column</th><th>count</th><th>missing</th><th>mean</th><th>std</th><th>min</th><th>25%</th><th>50%</th><th>75%</th><th>max</th></tr>{desc.map((d) => <tr key={d.name}><td>{d.name}</td><td>{d.count}</td><td>{d.missing}</td><td>{d.mean.toFixed(2)}</td><td>{d.std.toFixed(2)}</td><td>{d.min.toFixed(1)}</td><td>{d.q25.toFixed(1)}</td><td>{d.q50.toFixed(1)}</td><td>{d.q75.toFixed(1)}</td><td>{d.max.toFixed(1)}</td></tr>)}</tbody></table></div>
              {mlMode === "maths" && mathSummary()}
              <div className="stepnav"><button className="btn" onClick={() => setStep("eda")}>Next: EDA →</button></div>
            </>)}
          </div>
        </div>
      )}

      {/* STEP 2 EDA */}
      {step === "eda" && ds && (() => {
        const numericNames = ds.columns.filter((c) => c.type === "num").map((c) => c.name);
        const catNames = ds.columns.filter((c) => c.type === "cat").map((c) => c.name);
        const uniColType = ds.columns.find((c) => c.name === uniCol)?.type ?? "num";
        const missingCells = ds.columns.reduce((a, c) => a + c.values.filter((v) => v == null).length, 0);
        let dupRows = 0; { const seen = new Set<string>(); for (let i = 0; i < ds.nrows; i++) { const key = ds.columns.map((c) => String(c.values[i])).join("¦"); if (seen.has(key)) dupRows++; else seen.add(key); } }
        const plotHeight = cmpChart === "Scatter matrix" && edaMode === "compare" ? 540 : (cmpChart === "Parallel coordinates" && edaMode === "compare" ? 470 : 430);
        const needsXY = ["Scatter", "Line", "2D density"].includes(cmpChart);
        const needsGroupNum = ["Box by group", "Violin by group"].includes(cmpChart);
        const needsBar = ["Grouped bar", "Stacked bar"].includes(cmpChart);
        const needsMulti = ["Scatter matrix", "Correlation heatmap", "Parallel coordinates"].includes(cmpChart);
        return (
          <div className="split" style={{ gridTemplateColumns: "244px 1fr" }}>
            {/* LEFT — overview + column list */}
            <div className="eda-side">
              <div className="card">
                <div className="card-h"><span className="t">Dataset overview</span></div>
                <div className="card-b" style={{ padding: 12 }}>
                  <div className="ov-grid">
                    <div className="ov"><b>{ds.nrows}</b><span>rows</span></div>
                    <div className="ov"><b>{ds.columns.length}</b><span>columns</span></div>
                    <div className="ov"><b>{numericNames.length}</b><span>numeric</span></div>
                    <div className="ov"><b>{catNames.length}</b><span>categorical</span></div>
                    <div className="ov"><b>{missingCells}</b><span>missing cells</span></div>
                    <div className="ov"><b>{dupRows}</b><span>duplicate rows</span></div>
                  </div>
                </div>
              </div>
              <div className="card" style={{ marginTop: 14 }}>
                <div className="card-h"><span className="t">Columns</span><span className="mono r">click to plot</span></div>
                <div className="card-b" style={{ maxHeight: 430, overflow: "auto", padding: 8 }}>
                  {ds.columns.map((c) => { const s = colStats(c); return (
                    <div key={c.name} className={`col-item ${edaMode === "single" && uniCol === c.name ? "on" : ""}`} onClick={() => { setEdaMode("single"); pickUniCol(c.name); }}>
                      <span>{c.name}</span>
                      {s.missing > 0 && <span className="note" style={{ color: "var(--crit)" }}>{s.missing}na</span>}
                      <span className={`ty ${c.type}`}>{c.type}{c.name === target ? "·t" : ""}</span>
                    </div>); })}
                </div>
              </div>
            </div>

            {/* RIGHT — explorer */}
            <div>
              <div className="card">
                <div className="card-h">
                  <span className="t">Explore</span>
                  <div className="tabs">
                    <button className={edaMode === "single" ? "on" : ""} onClick={() => setEdaMode("single")}>Single column</button>
                    <button className={edaMode === "compare" ? "on" : ""} onClick={() => setEdaMode("compare")}>Compare columns</button>
                  </div>
                </div>
                <div className="card-b">
                  {edaMode === "single" ? (
                    <div className="eda-controls">
                      <div className="ec"><label className="fld">Column</label>
                        <select value={uniCol} onChange={(e) => pickUniCol(e.target.value)}>{ds.columns.map((c) => <option key={c.name} value={c.name}>{c.name} · {c.type}</option>)}</select>
                      </div>
                      <div className="ec"><label className="fld">Chart type</label>
                        <select value={uniChart} onChange={(e) => setUniChart(e.target.value)}>{(uniColType === "num" ? SINGLE_NUM : SINGLE_CAT).map((c) => <option key={c}>{c}</option>)}</select>
                      </div>
                      {uniColType === "num" && uniChart === "Histogram" && (
                        <div className="ec"><label className="fld">Bins · {bins}</label><input type="range" min={5} max={60} value={bins} onChange={(e) => setBins(+e.target.value)} /></div>
                      )}
                      {uniColType === "num" && ["Histogram", "Box", "Violin"].includes(uniChart) && catNames.length > 0 && (
                        <div className="ec"><label className="fld">Split by (group)</label>
                          <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)}><option>(none)</option>{catNames.map((c) => <option key={c}>{c}</option>)}</select>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="eda-controls">
                      <div className="ec"><label className="fld">Chart type</label>
                        <select value={cmpChart} onChange={(e) => pickCmpChart(e.target.value)}>{COMPARE_CHARTS.map((c) => <option key={c}>{c}</option>)}</select>
                      </div>
                      {needsXY && (<>
                        <div className="ec"><label className="fld">X (numeric)</label><select value={xCol} onChange={(e) => setXCol(e.target.value)}>{numericNames.map((c) => <option key={c}>{c}</option>)}</select></div>
                        <div className="ec"><label className="fld">Y (numeric)</label><select value={yCol} onChange={(e) => setYCol(e.target.value)}>{numericNames.map((c) => <option key={c}>{c}</option>)}</select></div>
                        {cmpChart !== "2D density" && <div className="ec"><label className="fld">Color / group</label><select value={colorCol} onChange={(e) => setColorCol(e.target.value)}><option>(none)</option>{ds.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select></div>}
                        {cmpChart === "Scatter" && <label className="ec chkline"><input type="checkbox" checked={trend} onChange={(e) => setTrend(e.target.checked)} /> trendline (OLS)</label>}
                      </>)}
                      {needsGroupNum && (<>
                        <div className="ec"><label className="fld">Value (numeric)</label><select value={yCol} onChange={(e) => setYCol(e.target.value)}>{numericNames.map((c) => <option key={c}>{c}</option>)}</select></div>
                        <div className="ec"><label className="fld">Group (categorical)</label><select value={xCol} onChange={(e) => setXCol(e.target.value)}>{catNames.map((c) => <option key={c}>{c}</option>)}</select></div>
                      </>)}
                      {needsBar && (<>
                        <div className="ec"><label className="fld">X (categorical)</label><select value={xCol} onChange={(e) => setXCol(e.target.value)}>{catNames.map((c) => <option key={c}>{c}</option>)}</select></div>
                        <div className="ec"><label className="fld">Split by (categorical)</label><select value={colorCol} onChange={(e) => setColorCol(e.target.value)}><option>(none)</option>{catNames.map((c) => <option key={c}>{c}</option>)}</select></div>
                        <div className="ec"><label className="fld">Bar height</label><select value={yCol} onChange={(e) => setYCol(e.target.value)}><option>(count)</option>{numericNames.map((c) => <option key={c}>mean {c}</option>)}</select></div>
                      </>)}
                      {needsMulti && (
                        <div className="ec ec-wide"><label className="fld">Numeric columns ({multiCols.length} selected)</label>
                          <div className="checklist">{numericNames.map((c) => <span key={c} className={`chk ${multiCols.includes(c) ? "on" : ""}`} onClick={() => setMultiCols((m) => m.includes(c) ? m.filter((x) => x !== c) : [...m, c])}>{c}</span>)}</div>
                        </div>
                      )}
                      {["Scatter matrix", "Parallel coordinates"].includes(cmpChart) && (
                        <div className="ec"><label className="fld">Color by</label><select value={colorCol} onChange={(e) => setColorCol(e.target.value)}><option>(none)</option>{ds.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select></div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="card" style={{ marginTop: 14 }}>
                <div className="card-b" style={{ padding: 10 }}>
                  {edaFig && edaFig.data.length ? <Plot data={edaFig.data} layout={edaFig.layout} style={{ height: plotHeight }} /> : <div className="note" style={{ padding: 30, textAlign: "center" }}>Adjust the controls above to render a chart.</div>}
                </div>
              </div>

              <div className="card" style={{ marginTop: 14 }}>
                <div className="card-h"><span className="t">What to look at</span><span className="mono r">auto-insights</span></div>
                <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {insights.map((ins, i) => <div key={i} className={`insight ${ins.kind}`}><span className="idot" />{ins.text}</div>)}
                </div>
              </div>

              <div className="stepnav"><button className="btn" onClick={() => setStep("prep")}>Next: Preprocess →</button></div>
            </div>
          </div>
        );
      })()}

      {/* STEP 3 PREP */}
      {step === "prep" && ds && (
        <>
          <div className="split col-2e">
            <div className="card"><div className="card-h"><span className="t">Add a preprocessing step</span></div>
              <div className="card-b">
                <label className="fld">Operation</label><select value={prepOp} onChange={(e) => { setPrepOp(e.target.value); setPrepMethod(OPS[e.target.value].methods[0]); setPrepCols([]); }}>{Object.keys(OPS).map((o) => <option key={o}>{o}</option>)}</select>
                <label className="fld" style={{ marginTop: 12 }}>Columns</label><div className="checklist">{eligibleCols.map((c) => <span key={c} className={`chk ${prepCols.includes(c) ? "on" : ""}`} onClick={() => setPrepCols((p) => p.includes(c) ? p.filter((x) => x !== c) : [...p, c])}>{c}</span>)}</div>
                <label className="fld" style={{ marginTop: 12 }}>Method</label><select value={prepMethod} onChange={(e) => setPrepMethod(e.target.value)}>{opSpec.methods.map((m) => <option key={m}>{m}</option>)}</select>
                <div className="op-hint">{opSpec.hint}<span>{describeStep(prepOp, prepMethod)}</span></div>
                <button className="btn" style={{ marginTop: 14 }} onClick={addStep}>+ Add step</button>
              </div>
            </div>
            <div className="card"><div className="card-h"><span className="t">Pipeline</span><span className="mono r">{steps.length} steps</span></div>
              <div className="card-b">{steps.map((s, i) => <div key={i} className="prep-step"><span className="op">{s.op}</span><span className="mono">{s.method} · {s.cols.join(", ")}</span><span className="rm" onClick={() => setSteps((ss) => ss.filter((_, j) => j !== i))}>×</span></div>)}{steps.length === 0 && <div className="note">No steps yet.</div>}
                <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={runPreprocessing}>▶ Run preprocessing</button><span className="note">applies every step &amp; animates the transform</span></div>
              </div>
            </div>
          </div>
          {snaps.length > 0 && cur && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-h"><span className="t">Pipeline — step by step</span><span className="mono r">{cur.nRows} rows × {cur.nCols} cols</span></div>
              <div className="card-b">
                {/* player controls */}
                <div className="pp-player">
                  <button className="pp-ctrl" title="Restart" onClick={() => { setPpPlaying(false); setPpIdx(0); }}>⏮</button>
                  <button className="pp-ctrl" title="Previous step" onClick={() => { setPpPlaying(false); setPpIdx((i) => Math.max(0, i - 1)); }}>‹</button>
                  <button className="pp-ctrl play" onClick={() => { if (ppIdx >= snaps.length - 1) { setPpIdx(0); setPpPlaying(true); } else setPpPlaying((p) => !p); }}>{ppPlaying ? "⏸ Pause" : (ppIdx >= snaps.length - 1 ? "↻ Replay" : "▶ Play")}</button>
                  <button className="pp-ctrl" title="Next step" onClick={() => { setPpPlaying(false); setPpIdx((i) => Math.min(snaps.length - 1, i + 1)); }}>›</button>
                  <span className="pp-count">step {ppIdx} / {snaps.length - 1}</span>
                  <span className="pp-speed">speed
                    <select value={ppSpeed} onChange={(e) => setPpSpeed(+e.target.value)}><option value={2600}>0.5×</option><option value={1500}>1×</option><option value={800}>2×</option></select>
                  </span>
                </div>
                <div className="pp-progress"><i style={{ width: `${(ppIdx / Math.max(1, snaps.length - 1)) * 100}%` }} /></div>

                {/* jump-to-step chips */}
                <div className="pp-chips">{snaps.map((s, i) => <button key={i} className={`chk ${ppIdx === i ? "on" : ""}`} onClick={() => { setPpPlaying(false); setPpIdx(i); }}>{i === 0 ? "raw" : `${i}. ${s.op}`}</button>)}</div>

                {/* current-step explanation */}
                <div className="pp-explain">
                  <div className="pp-badge">{ppIdx === 0 ? "RAW" : `STEP ${ppIdx}`}</div>
                  <div style={{ flex: 1 }}>
                    <div className="pp-title">{cur.op}{cur.method !== "source data" ? <span className="pp-method"> · {cur.method}</span> : null}</div>
                    <div className="pp-sub">{ppIdx === 0 ? "The original dataset before any preprocessing." : describeStep(cur.op, cur.method)}{cur.changedCols.length ? <> Affected columns: <b>{cur.changedCols.join(", ")}</b>.</> : null}</div>
                  </div>
                  <div className="pp-shape">{cur.nRows}×{cur.nCols}</div>
                </div>

                {/* data table — changed columns highlighted */}
                <div style={{ overflowX: "auto" }}><table className="pp-table"><tbody><tr>{cur.colNames.map((n) => <th key={n} className={cur.changedCols.includes(n) ? "hl" : ""}>{n}</th>)}</tr>{cur.sample.map((row, ri) => <tr key={ri}>{row.map((v, ci) => <td key={ci} className={cur.changedCols.includes(cur.colNames[ci]) ? "hl" : ""}>{v}</td>)}</tr>)}</tbody></table></div>

                <div className="split col-2e" style={{ marginTop: 14 }}>
                  <div><label className="fld">Before (raw)</label><div className="note">{snaps[0].nCols} columns, {snaps[0].nRows} rows</div><div style={{ overflowX: "auto" }}><table className="pp-table"><tbody><tr>{snaps[0].colNames.slice(0, 6).map((n) => <th key={n}>{n}</th>)}</tr>{snaps[0].sample.slice(0, 3).map((row, ri) => <tr key={ri}>{row.slice(0, 6).map((v, ci) => <td key={ci}>{v}</td>)}</tr>)}</tbody></table></div></div>
                  <div><label className="fld">After (fully processed)</label><div className="note">{processedCols} columns (all numeric), ready to train</div><div style={{ overflowX: "auto" }}><table className="pp-table"><tbody><tr>{snaps[snaps.length - 1].colNames.slice(0, 6).map((n) => <th key={n}>{n}</th>)}</tr>{snaps[snaps.length - 1].sample.slice(0, 3).map((row, ri) => <tr key={ri}>{row.slice(0, 6).map((v, ci) => <td key={ci}>{v}</td>)}</tr>)}</tbody></table></div></div>
                </div>
                <div className="stepnav"><button className="btn" onClick={() => setStep("model")}>Next: Model →</button></div>
              </div>
            </div>
          )}
          {snaps.length === 0 && <div className="stepnav" style={{ marginTop: 16 }}><button className="btn" onClick={() => setStep("model")}>Next: Model →</button></div>}
        </>
      )}

      {/* STEP 4 MODEL */}
      {step === "model" && ds && (
        <>
          <div className="card">
            <div className="card-h"><span className="t">Choose a model</span><span className="mono r">{Object.keys(MODELS[task]).length} available · {task}</span></div>
            <div className="card-b">
              <div className="row" style={{ marginBottom: 12 }}><span className="note">Task</span>
                <div className="rangebtns"><button className={task === "classification" ? "on" : ""} onClick={() => { setTask("classification"); const a = Object.keys(MODELS.classification)[0]; setAlgo(a); setParamsFor("classification", a); }}>Classification</button><button className={task === "regression" ? "on" : ""} onClick={() => { setTask("regression"); const a = Object.keys(MODELS.regression)[0]; setAlgo(a); setParamsFor("regression", a); }}>Regression</button></div>
              </div>
              <div className="model-grid">
                {Object.keys(MODELS[task]).map((a) => { const info = MODEL_INFO[a]; const np = MODELS[task][a].length; return (
                  <div key={a} className={`model-card ${algo === a ? "on" : ""}`} onClick={() => { setAlgo(a); setParamsFor(task, a); }}>
                    <div className="mc-top"><span className="mc-name">{info?.label ?? a}</span><span className="mc-fam">{info?.family}</span></div>
                    <div className="mc-desc">{info?.desc}</div>
                    <div className="mc-foot">{np ? `${np} hyperparameter${np > 1 ? "s" : ""}` : "no hyperparameters"}{algo === a ? " · selected" : ""}</div>
                  </div>); })}
              </div>
            </div>
          </div>

          <div className="split col-2e" style={{ marginTop: 16 }}>
            <div className="card"><div className="card-h"><span className="t">Target &amp; features</span></div>
              <div className="card-b">
                <label className="fld">Target column</label><select value={target} onChange={(e) => pickTarget(e.target.value)}>{ds.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select>
                <div className="split col-2e" style={{ marginTop: 12 }}><div><label className="fld">Test size</label><input type="text" value={testSize} onChange={(e) => setTestSize(Number(e.target.value) || 0.2)} /></div><div><label className="fld">CV folds</label><input type="text" value={cvFolds} onChange={(e) => setCvFolds(Number(e.target.value) || 5)} /></div></div>
                <label className="fld" style={{ marginTop: 12 }}>Feature columns ({features.length})</label><div className="checklist">{ds.columns.filter((c) => c.name !== target).map((c) => <span key={c.name} className={`chk ${features.includes(c.name) ? "on" : ""}`} onClick={() => setFeatures((f) => f.includes(c.name) ? f.filter((x) => x !== c.name) : [...f, c.name])}>{c.name}</span>)}</div>
              </div>
            </div>
            <div className="card"><div className="card-h"><span className="t">Hyperparameters</span><span className="mono r">{MODEL_INFO[algo]?.label ?? algo}</span></div>
              <div className="card-b"><div className="hyper">{MODELS[task][algo].length === 0 && <div className="note">This model has no tunable hyperparameters.</div>}{MODELS[task][algo].map((sp) => (
                <div key={sp.name} className="hp"><label>{sp.name}</label>{sp.type === "sel"
                  ? <select value={params[sp.name] ?? String(sp.def)} onChange={(e) => setParams((p) => ({ ...p, [sp.name]: e.target.value }))}>{sp.opts!.map((o) => <option key={o}>{o}</option>)}</select>
                  : <span className="slider-row"><input type="range" min={sp.min} max={sp.max} step={sp.step} value={params[sp.name] ?? String(sp.def)} onChange={(e) => setParams((p) => ({ ...p, [sp.name]: e.target.value }))} style={{ width: 96 }} /><input type="text" value={params[sp.name] ?? String(sp.def)} onChange={(e) => setParams((p) => ({ ...p, [sp.name]: e.target.value }))} style={{ width: 62 }} /><span className="rng">{sp.min}–{sp.max}</span></span>}</div>
              ))}</div>
                <div className="note" style={{ marginTop: 12 }}>Defaults shown; each field shows its valid min–max range.</div>
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-h"><span className="t">Preprocessed data → model input</span><span className="mono r">{built ? `${built.X.length} rows × ${built.featureNames.length} features` : "—"}</span></div>
            <div className="card-b">
              {built && built.featureNames.length ? (<>
                <div className="note" style={{ marginBottom: 8 }}>After your preprocessing pipeline every feature is numeric — this exact matrix is what the model trains on.</div>
                <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                  <tr><th>#</th>{built.featureNames.map((n) => <th key={n}>{n}</th>)}<th style={{ color: "var(--accent)" }}>{target} (y)</th></tr>
                  {built.X.slice(0, 6).map((row, ri) => <tr key={ri}><td style={{ color: "var(--faint)" }}>{ri}</td>{row.map((v, ci) => <td key={ci}>{fmtNum(v)}</td>)}<td style={{ color: "var(--accent)" }}>{built.classes ? (built.classes[built.y[ri]] ?? "—") : fmtNum(built.y[ri])}</td></tr>)}
                </tbody></table></div>
              </>) : <div className="note">Pick a target and at least one feature to build the model-input matrix.</div>}
              <div className="stepnav"><button className="btn" onClick={() => setStep("validation")}>Next: Validation →</button></div>
            </div>
          </div>
        </>
      )}

      {/* STEP 5 VALIDATION */}
      {step === "validation" && ds && (
        <>
          <div className="card">
            <div className="card-h"><span className="t">How will you validate the model?</span><span className="mono r">{MODEL_INFO[algo]?.label ?? algo}</span></div>
            <div className="card-b">
              <div className="row" style={{ gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
                <div className="rangebtns"><button className={valMethod === "kfold" ? "on" : ""} onClick={() => setValMethod("kfold")}>K-Fold cross-validation</button><button className={valMethod === "holdout" ? "on" : ""} onClick={() => setValMethod("holdout")}>Hold-out validation</button></div>
              </div>
              {/* split diagram */}
              {(() => { const n = built ? built.X.length : ds.nrows; const sc = valMethod === "holdout" ? splitCounts(n, valSize, testSize) : { train: n - Math.round(n * testSize), val: 0, test: Math.round(n * testSize) }; const pct = (x: number) => `${(x / n) * 100}%`; return (
                <>
                  <label className="fld">Data split ({n} rows)</label>
                  <div className="split-bar">
                    <div className="sb train" style={{ width: pct(sc.train) }}><span>train · {sc.train}</span></div>
                    {valMethod === "holdout" && sc.val > 0 && <div className="sb val" style={{ width: pct(sc.val) }}><span>val · {sc.val}</span></div>}
                    <div className="sb test" style={{ width: pct(sc.test) }}><span>test · {sc.test}</span></div>
                  </div>
                  <div className="row" style={{ gap: 18, flexWrap: "wrap", marginTop: 12 }}>
                    <div className="knob" style={{ margin: 0, minWidth: 190 }}><div className="kr"><span>Test size</span><b>{testSize}</b></div><input type="range" min={0.1} max={0.4} step={0.05} value={testSize} onChange={(e) => setTestSize(+e.target.value)} /></div>
                    {valMethod === "holdout" && <div className="knob" style={{ margin: 0, minWidth: 190 }}><div className="kr"><span>Validation size</span><b>{valSize}</b></div><input type="range" min={0.1} max={0.4} step={0.05} value={valSize} onChange={(e) => setValSize(+e.target.value)} /></div>}
                    {valMethod === "kfold" && <div className="knob" style={{ margin: 0, minWidth: 190 }}><div className="kr"><span>CV folds (k)</span><b>{cvFolds}</b></div><input type="range" min={2} max={10} value={cvFolds} onChange={(e) => setCvFolds(+e.target.value)} /></div>}
                  </div>
                  <div className="note" style={{ marginTop: 10 }}>{valMethod === "kfold" ? `K-fold splits the training data into ${cvFolds} parts; the model trains on ${cvFolds - 1} and validates on the held-out part, rotating through all ${cvFolds}. Every row is validated exactly once — the mean score is a robust estimate.` : "Hold-out keeps a single validation slice aside to tune on, and a separate test slice untouched until the end."}</div>
                </>
              ); })()}
            </div>
          </div>

          {valMethod === "kfold" && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-h"><span className="t">Cross-validation</span><span className="mono r">{cvRunning ? <><span className="busy-dot" />running…</> : cvResult.length ? `${cvResult.length} folds complete` : "not run"}</span></div>
              <div className="card-b">
                <div className="row" style={{ marginBottom: 14 }}><button className="btn" onClick={runCrossVal} disabled={cvRunning || !built}>▶ Run {cvFolds}-fold cross-validation</button><span className="note">trains a fresh model on each fold — {task === "classification" ? "accuracy" : "R²"} per fold</span></div>
                {cvResult.slice(0, cvShown).map((f) => (
                  <div key={f.fold} className="fold-row reveal-in">
                    <span className="fold-lbl">fold {f.fold}</span>
                    <div className="fold-mini"><span className="fm-train" style={{ flex: f.trainN }} title={`train ${f.trainN}`} /><span className="fm-test" style={{ flex: f.testN }} title={`test ${f.testN}`} /></div>
                    <div className="fold-bar"><i style={{ width: `${Math.max(0, Math.min(1, f.score)) * 100}%` }} /></div>
                    <span className="fold-score">{f.score.toFixed(3)}</span>
                  </div>
                ))}
                {cvShown >= cvResult.length && cvResult.length > 0 && (
                  <div className="cv-summary">
                    <div className="metric"><span className="v">{mean(cvAllScores).toFixed(3)}</span><span className="k">mean {task === "classification" ? "accuracy" : "R²"}</span></div>
                    <div className="metric"><span className="v">±{std(cvAllScores).toFixed(3)}</span><span className="k">std dev</span></div>
                    <div className="metric"><span className="v">{Math.min(...cvAllScores).toFixed(3)}</span><span className="k">worst fold</span></div>
                    <div className="metric"><span className="v">{Math.max(...cvAllScores).toFixed(3)}</span><span className="k">best fold</span></div>
                  </div>
                )}
                {cvResult.length === 0 && !cvRunning && <div className="note">Run cross-validation to validate the model across all {cvFolds} folds before final training.</div>}
                <div className="stepnav"><button className="btn ghost" onClick={() => setStep("model")}>← Back</button><button className="btn" onClick={runTrain}>Next: Train →</button></div>
              </div>
            </div>
          )}
          {valMethod === "holdout" && <div className="stepnav" style={{ marginTop: 16 }}><button className="btn ghost" onClick={() => setStep("model")}>← Back</button><button className="btn" onClick={runTrain}>Next: Train →</button></div>}
        </>
      )}

      {/* STEP 6 TRAIN */}
      {step === "train" && ds && (
        <>
          <div className="row" style={{ marginBottom: 14 }}><button className="btn" onClick={runTrain} disabled={training}>▶ Train model</button><span className="mono" style={{ color: "var(--faint)", marginLeft: 10 }}>{training ? "training…" : result ? `trained in ${result.ms}ms` : "not trained"}</span></div>
          <div className="tflow2">{FLOW.map((s, i) => <div key={i} className={`tf2 ${flowStep > i ? "done" : flowStep === i ? "active" : ""}`}><div className="tf2-i">{flowStep > i ? "✓" : flowStep === i ? <span className="busy-dot" /> : i + 1}</div><div className="tf2-t"><b>{s.w}</b><span>{s.d}</span></div></div>)}</div>
          {result && (<>
            {/* data → model, sample by sample */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-h"><span className="t">Watch it train — data fed to the model</span><span className="mono r">sample by sample</span></div>
              <div className="card-b">
                {(() => {
                  const curS = TP_SHOWN ? tpIdx % TP_SHOWN : 0;
                  const epoch = TP_SHOWN ? Math.floor(tpIdx / TP_SHOWN) + 1 : 1;
                  const prog = TP_TOTAL > 1 ? tpIdx / (TP_TOTAL - 1) : 1;
                  const lossVal = result.loss?.length ? result.loss[Math.round(prog * (result.loss.length - 1))] : undefined;
                  const featShown = features.slice(0, 3);
                  return (<>
                    <div className="pp-player">
                      <button className="pp-ctrl" title="Restart" onClick={() => { setTpPlaying(false); setTpIdx(0); }}>⏮</button>
                      <button className="pp-ctrl play" onClick={() => { if (tpIdx >= TP_TOTAL - 1) { setTpIdx(0); setTpPlaying(true); } else setTpPlaying((p) => !p); }}>{tpPlaying ? "⏸ Pause" : (tpIdx >= TP_TOTAL - 1 ? "↻ Replay" : "▶ Play")}</button>
                      <span className="pp-count">epoch {epoch}/{TP_EPOCHS} · sample {curS + 1}/{TP_SHOWN}</span>
                      <span className="pp-speed">speed<select value={tpSpeed} onChange={(e) => setTpSpeed(+e.target.value)}><option value={900}>0.5×</option><option value={500}>1×</option><option value={220}>2×</option></select></span>
                    </div>
                    <div className="pp-progress"><i style={{ width: `${prog * 100}%` }} /></div>
                    <div className="feeder">
                      <div className="feeder-rows">
                        <div className="feed-head"><span className="fr-idx">row</span>{featShown.map((f) => <span key={f} className="fr-cell">{f.slice(0, 8)}</span>)}<span className="fr-y">{target.slice(0, 8)}</span></div>
                        {Array.from({ length: TP_SHOWN }, (_, i) => i).map((i) => (
                          <div key={i} className={`feed-row ${i === curS ? "on" : i < curS ? "past" : ""}`}>
                            <span className="fr-idx">#{i}</span>
                            {featShown.map((f) => { const c = ds.columns.find((x) => x.name === f); const v = c?.values[i]; return <span key={f} className="fr-cell">{v == null ? "∅" : (typeof v === "number" ? (Number.isInteger(v) ? v : Number(v).toFixed(1)) : String(v).slice(0, 8))}</span>; })}
                            <span className="fr-y">{(() => { const v = ds.columns.find((x) => x.name === target)?.values[i]; return v == null ? "?" : String(v).slice(0, 8); })()}</span>
                          </div>
                        ))}
                      </div>
                      <div className={`feeder-arrow ${tpPlaying ? "run" : ""}`}>→</div>
                      <div className="feeder-model">
                        <div className="fm-title">{MODEL_INFO[algo]?.label ?? algo}</div>
                        {modelKind === "gd" && result.loss?.length ? (<>
                          <div className="fm-metric">loss <b>{lossVal!.toFixed(3)}</b></div>
                          <Sparkline values={result.loss} upto={Math.round(prog * result.loss.length)} />
                          <div className="note" style={{ marginTop: 6 }}>each sample nudges the weights ↓ the loss</div>
                        </>) : (modelKind === "tree" || modelKind === "forest") ? (
                          <div className="fm-metric">splitting samples<div className="note" style={{ marginTop: 6 }}>rows partitioned to reduce impurity{modelKind === "forest" ? ` across ${treeViz?.nTrees ?? ""} trees` : ""}</div></div>
                        ) : modelKind === "knn" ? (
                          <div className="fm-metric">memorising<div className="note" style={{ marginTop: 6 }}>KNN is lazy — it just stores each sample</div></div>
                        ) : (
                          <div className="fm-metric">class stats<div className="note" style={{ marginTop: 6 }}>updating mean &amp; variance per class</div></div>
                        )}
                      </div>
                    </div>
                    <div className="note" style={{ marginTop: 10 }}>The model sees the training rows repeatedly — each full pass is one <b>epoch</b>{modelKind === "gd" ? ", adjusting its weights a little each time until the loss flattens." : "."}</div>
                  </>);
                })()}
              </div>
            </div>

            {/* how the model trained — visualization */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-h"><span className="t">How {MODEL_INFO[algo]?.label ?? algo} trained</span><span className="mono r">{modelKind === "gd" ? "gradient descent" : modelKind === "tree" ? "recursive splitting" : modelKind === "forest" ? `${treeViz?.nTrees ?? ""} trees` : modelKind === "gnb" ? "class statistics" : "lazy / instance-based"}</span></div>
              <div className="card-b">
                {modelKind === "gd" && result.loss && (<>
                  <div className="note" style={{ marginBottom: 8 }}>The model starts with zero weights and takes small steps <b>down the loss surface</b> each epoch (gradient descent). Watch the loss fall and flatten as it converges.</div>
                  {lossFig && <Plot data={lossFig.data} layout={lossFig.layout} style={{ height: 260 }} />}
                  <div className="note" style={{ marginTop: 6 }}>start loss {result.loss[0].toFixed(3)} → final {result.loss[result.loss.length - 1].toFixed(3)} over {result.loss.length} recorded epochs.</div>
                </>)}
                {(modelKind === "tree" || modelKind === "forest") && treeViz && (<>
                  <div className="note" style={{ marginBottom: 10 }}>{modelKind === "forest" ? `The forest grew ${treeViz.nTrees} trees on bootstrapped samples and averages their votes. Below is one representative tree` : "The tree repeatedly picks the feature & threshold that best separates the target"} — depth {treeViz.depth}, {treeViz.nodes} nodes. Each internal node splits on <b>feature ≤ threshold</b>; leaves give the prediction.</div>
                  <TreeDiagram root={treeViz.root} featureNames={(built ?? { featureNames: [] as string[] }).featureNames} classes={result.metrics.task === "classification" ? result.metrics.classes : []} task={task} />
                </>)}
                {modelKind === "knn" && <div className="note">K-Nearest Neighbors is a <b>lazy learner</b> — it does no work at &ldquo;fit&rdquo; time beyond memorising the {built?.X.length ?? 0} training rows. At prediction it measures distance to every stored point and lets the {params.n_neighbors ?? 5} nearest vote.</div>}
                {modelKind === "gnb" && <div className="note">Gaussian Naive Bayes learns, per class, the <b>mean &amp; variance</b> of each feature (assuming a bell curve) plus the class priors. Prediction picks the class with the highest combined likelihood.</div>}
              </div>
            </div>

            <div className="split col-2e">
              <div className="card"><div className="card-h"><span className="t">Test-set results</span><span className="mono r">{MODEL_INFO[algo]?.label ?? algo}</span></div>
                <div className="card-b">
                  {result.metrics.task === "classification" ? (<>
                    <div className="split" style={{ gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 16 }}><div className="metric"><span className="v">{result.metrics.accuracy.toFixed(2)}</span><span className="k">accuracy</span></div><div className="metric"><span className="v">{result.metrics.precision.toFixed(2)}</span><span className="k">precision</span></div><div className="metric"><span className="v">{result.metrics.recall.toFixed(2)}</span><span className="k">recall</span></div><div className="metric"><span className="v">{result.metrics.f1.toFixed(2)}</span><span className="k">f1</span></div></div>
                    <label className="fld">Confusion matrix ({result.metrics.classes.join(", ")})</label>
                    <div className="cmx" style={{ gridTemplateColumns: `repeat(${result.metrics.classes.length},1fr)`, maxWidth: 260 }}>{(result.metrics as ClsMetrics).confusion.flatMap((row, i) => row.map((val, j) => { const mx = Math.max(...(result.metrics as ClsMetrics).confusion.flat(), 1); return <div key={`${i}-${j}`} style={{ background: i === j ? "var(--good)" : (val > 0 ? "var(--crit)" : "var(--panel-2)"), color: val === 0 && i !== j ? "var(--muted)" : "#fff", opacity: i === j || val === 0 ? 1 : 0.5 + 0.5 * (val / mx) }}>{val}</div>; }))}</div>
                  </>) : (<>
                    <div className="split" style={{ gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 16 }}><div className="metric"><span className="v">{result.metrics.r2.toFixed(2)}</span><span className="k">R²</span></div><div className="metric"><span className="v">{result.metrics.mae.toFixed(1)}</span><span className="k">MAE</span></div><div className="metric"><span className="v">{result.metrics.rmse.toFixed(1)}</span><span className="k">RMSE</span></div></div>
                    {pvaFig && <Plot data={pvaFig.data} layout={pvaFig.layout} style={{ height: 240 }} />}
                  </>)}
                  <label className="fld" style={{ marginTop: 14 }}>Cross-validation ({result.cv.length}-fold {task === "classification" ? "accuracy" : "R²"})</label>
                  <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{result.cv.map((s) => s.toFixed(3)).join(", ")} · mean {cvMean.toFixed(3)} ± {std(result.cv).toFixed(3)}</div>
                </div>
              </div>
              <div className="card"><div className="card-h"><span className="t">Feature importance</span></div>
                <div className="card-b">
                  {result.importance && result.importance.map((f) => <div key={f.name} style={{ marginBottom: 8, fontSize: 12 }}>{f.name}<div className="impbar"><i style={{ width: `${Math.round(f.w * 100)}%` }} /></div></div>)}
                  {!result.importance && <div className="note">This model exposes no per-feature coefficients (e.g. KNN / Naive Bayes). See the training visualization above for how it works.</div>}
                </div>
              </div>
            </div>

          </>)}
          {result && (<>
            {task === "classification" && numFeats().length >= 2 && (
              <div className="card" style={{ marginTop: 16 }}>
                <div className="card-h"><span className="t">Decision boundary</span><div className="r"><button className="btn sm" onClick={runBoundary}>{dbSurf ? "↻ Redraw" : "▶ Draw boundary"}</button></div></div>
                <div className="card-b">
                  <div className="note" style={{ marginBottom: 10 }}>Trains a 2-feature {MODEL_INFO[algo]?.label ?? algo} and colours the region it assigns to each class, with the data points on top — <b>how the model separates the classes</b>.</div>
                  <div className="split col-2e" style={{ maxWidth: 460, marginBottom: 12 }}>
                    <div><label className="fld">Feature X</label><select value={dbF1 || numFeats()[0] || ""} onChange={(e) => setDbF1(e.target.value)}>{numFeats().map((f) => <option key={f}>{f}</option>)}</select></div>
                    <div><label className="fld">Feature Y</label><select value={dbF2 || numFeats()[1] || ""} onChange={(e) => setDbF2(e.target.value)}>{numFeats().map((f) => <option key={f}>{f}</option>)}</select></div>
                  </div>
                  {(() => { const f = boundaryFig(); return f ? <Plot data={f.data} layout={f.layout} style={{ height: 380 }} /> : <div className="note">Pick two features and draw the boundary.</div>; })()}
                  {dbSurf && (() => { const full = task === "classification" && result ? (result.metrics as ClsMetrics).accuracy : null; return (
                    <div className="teach-note" style={{ marginTop: 10 }}><span className="ic">📉</span><span>This <b>2-feature</b> model scores <b>{(dbSurf.acc * 100).toFixed(0)}%</b> on these points{full != null ? <> vs the <b>full {features.length}-feature</b> model&apos;s <b>{(full * 100).toFixed(0)}%</b></> : ""}. Two features rarely capture everything — more informative features usually lift accuracy.</span></div>
                  ); })()}
                </div>
              </div>
            )}
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-h"><span className="t">Learning curve</span>{(() => { const d = lcDiagnosis(); return d ? <span className="badge" style={{ color: d.cls === "ok" ? "var(--good)" : d.cls === "bad" ? "var(--crit)" : "var(--warn)", borderColor: "currentColor" }}>{d.cls === "ok" ? "✓ " : d.cls === "bad" ? "✕ " : "! "}{d.label}</span> : null; })()}<div className="r"><button className="btn sm" onClick={runLC}>{lcData.length ? "↻ Recompute" : "▶ Compute"}</button></div></div>
              <div className="card-b">
                <div className="note" style={{ marginBottom: 10 }}>Retrains on growing slices of the data. A wide <b>train-vs-validation gap</b> = overfitting; both low &amp; flat = underfitting; converging high = healthy.</div>
                <label className="fld">Features used ({(lcFeats.length ? lcFeats : features).length}/{features.length}) — pick columns to base the curve on, then recompute</label>
                <div className="checklist" style={{ marginBottom: 12 }}>{features.map((f) => { const on = lcFeats.length ? lcFeats.includes(f) : true; return <span key={f} className={`chk ${on ? "on" : ""}`} onClick={() => setLcFeats((prev) => { const cur = prev.length ? prev : [...features]; const next = cur.includes(f) ? cur.filter((x) => x !== f) : [...cur, f]; return next.length === features.length ? [] : next; })}>{f}</span>; })}</div>
                {(() => { const f = lcFig(); return f ? <Plot data={f.data} layout={f.layout} style={{ height: 300 }} /> : <div className="note">Compute to see train vs validation score as the data grows.</div>; })()}
                {(() => { const d = lcDiagnosis(); if (!d) return null; const clr = d.cls === "bad" ? "var(--crit)" : d.cls === "warn" ? "var(--warn)" : "var(--good)"; return (
                  <div style={{ marginTop: 12 }}>
                    <div className="etl-metrics" style={{ marginBottom: 10 }}>
                      <div className="m">train (open-book)<b style={{ color: "#5b7cff" }}>{lcFmt(d.train)}</b></div>
                      <div className="m">validation (unseen)<b style={{ color: "#f59e0b" }}>{lcFmt(d.test)}</b></div>
                      <div className="m">gap<b style={{ color: clr }}>{lcFmt(d.gap)}</b></div>
                      <div className="m">verdict<b style={{ color: clr }}>{d.label}</b></div>
                    </div>
                    <div className="teach-note"><span className="ic">{d.cls === "ok" ? "✅" : d.cls === "bad" ? "⚠️" : "🔎"}</span><span><b>{d.label}</b> — because <b>{d.rule}</b>. {d.why}</span></div>
                    <div className="note" style={{ marginTop: 8, lineHeight: 1.65 }}><b>How to read it:</b> blue = score on rows it trained on (like an open-book test), orange = score on rows it never saw (closed-book). The <b>gap</b> between them is how much worse it does on new data. <b>Big gap</b> → overfitting (memorised) · <b>both low &amp; flat</b> → underfitting (too simple) · <b>small gap &amp; both high</b> → good fit.</div>
                  </div>
                ); })()}
              </div>
            </div>
            <div className="stepnav"><button className="btn ghost" onClick={() => setStep("validation")}>← Validation</button><button className="btn" onClick={() => setStep("deploy")}>Next: Test &amp; Export →</button></div>
          </>)}
        </>
      )}

      {/* STEP 7 — TEST & EXPORT */}
      {step === "deploy" && ds && result && (
        <>
          <div className="teach-note"><span className="ic">📦</span><span><b>Test &amp; export.</b> Try the trained model on new inputs, edit the production code (Apply-to-flow syncs it back), and download the model.</span></div>
          <div className="card">
            <div className="card-h"><span className="t">Test the trained model</span>
              <div className="tabs"><button className={testTab === "manual" ? "on" : ""} onClick={() => setTestTab("manual")}>Manual input</button><button className={testTab === "sample" ? "on" : ""} onClick={() => setTestTab("sample")}>Random sample</button></div>
            </div>
            <div className="card-b">
              {testTab === "sample" && <div className="row" style={{ marginBottom: 12 }}><button className="btn" onClick={randomizeInputs}>🎲 Pick a random row &amp; predict</button><span className="note">fills the fields from a real row, then predicts &amp; compares to the true label</span></div>}
              <div className="test-grid">
                {features.map((f) => { const c = ds.columns.find((x) => x.name === f); if (!c) return null; return (
                  <div key={f} className="tf">
                    <label className="fld">{f} <span style={{ color: "var(--faint)" }}>{c.type}</span></label>
                    {c.type === "cat"
                      ? <select value={testInputs[f] ?? ""} onChange={(e) => setTestInputs((v) => ({ ...v, [f]: e.target.value }))}>{Array.from(new Set(c.values.filter((x) => x != null).map(String))).slice(0, 60).map((o) => <option key={o}>{o}</option>)}</select>
                      : <input type="text" value={testInputs[f] ?? ""} onChange={(e) => setTestInputs((v) => ({ ...v, [f]: e.target.value }))} />}
                  </div>); })}
              </div>
              <div className="row" style={{ marginTop: 14, gap: 12, flexWrap: "wrap" }}>
                <button className="btn" onClick={() => predictRow(testInputs)}>▶ Predict</button>
                {testOut && (
                  <div className={`pred-out ${testOut.ok === true ? "good" : testOut.ok === false ? "bad" : ""}`}>
                    <span className="po-label">prediction</span>
                    <span className="po-val">{trained?.classes ? testOut.pred : `${target} ≈ ${testOut.pred}`}</span>
                    {testOut.actual != null && <span className="po-actual">actual <b>{testOut.actual}</b> {testOut.ok === true ? "✓" : testOut.ok === false ? "✗" : ""}</span>}
                  </div>
                )}
              </div>
              <div className="note" style={{ marginTop: 10 }}>Inputs run through the exact same preprocessing pipeline, then the trained {MODEL_INFO[algo]?.label ?? algo} predicts. Categorical fields are limited to values seen in training.</div>
            </div>
          </div>
          {mlMode === "maths" && (
            <div className="card math-card" style={{ marginTop: 16 }}>
              <div className="card-h"><span className="t">🧮 From-scratch code — pure Python (no libraries)</span><div className="r"><button className="btn ghost sm" onClick={copyScratch}>{copied ? "Copied ✓" : "Copy"}</button><button className="btn sm" onClick={downloadScratch}>Download .py</button></div></div>
              <div className="card-b">
                <div className="note" style={{ marginBottom: 8 }}>The <b>same model</b>, implemented with only <code>csv</code> + <code>math</code> — every formula from the steps above turned into plain loops. No numpy, no sklearn.</div>
                <div className="code" style={{ maxHeight: 360 }}>{buildScratchCode()}</div>
              </div>
            </div>
          )}
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-h"><span className="t">Complete workflow code — {mlMode === "maths" ? "sklearn (for comparison)" : "editable"}</span><div className="r"><button className="btn ghost sm" onClick={applyCodeToFlow}>↥ Apply to flow</button>{codeDirty && <button className="btn ghost sm" onClick={() => setCodeDirty(false)}>Reset</button>}<button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button><button className="btn ghost sm" onClick={download}>.py</button></div></div>
            <div className="card-b">
              <div className="note" style={{ marginBottom: 8 }}>Edit the code, then <b>Apply to flow</b> to sync recognised settings (algorithm, hyperparameters, test_size, cv, scaler, encoder) back into the steps above, then re-train.</div>
              <textarea value={codeDirty ? codeDraft : buildCode()} onChange={(e) => { setCodeDraft(e.target.value); setCodeDirty(true); }} spellCheck={false} style={{ minHeight: 320, fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.5 }} />
              <div className="row" style={{ marginTop: 12, gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                <label className="fld" style={{ margin: 0 }}>Export trained model:</label>
                <button className="btn sm" onClick={exportPkl}>🥒 model.pkl</button>
                <button className="btn ghost sm" onClick={exportJson}>📄 model.json</button>
                <span className="note"><code>pickle.load(open(&quot;model.pkl&quot;,&quot;rb&quot;))</code> → the trained params as a dict.</span>
              </div>
            </div>
          </div>
          <div className="stepnav"><button className="btn ghost" onClick={() => setStep("train")}>← Back to Train</button></div>
        </>
      )}

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal"><div className="mh"><b>Complete workflow code</b><div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8 }}><button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button><button className="btn sm" onClick={download}>Download</button></div><button className="x" onClick={() => setShowCode(false)}>×</button></div><div className="mb"><div className="code">{buildCode()}</div></div></div>
      </div>
    </>
  );
}
