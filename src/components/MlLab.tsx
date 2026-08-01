"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  parseCSV, colStats, buildMatrix, split, makeModel, predict,
  featureImportance, classificationMetrics, regressionMetrics, crossVal, crossValDetailed,
  decisionSurface, learningCurve, gdTrace, gdAnim,
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
  const cfgNow = (): TrainConfig => ({ task, algo, params, testSize, cvFolds });

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
  const modelTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [modelApplied, setModelApplied] = useState(0);
  const [modelApplying, setModelApplying] = useState(false);
  const [modelSpeed, setModelSpeed] = useState(260); // ms per row when applying (bigger = slower)
  const modelApplyTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const modelSpeedRef = useRef(260);
  // Train step: fitting animation → check-metric → verdict.
  const [trainT, setTrainT] = useState(0);
  const [checkT, setCheckT] = useState(0);
  const [trainPhase, setTrainPhase] = useState<"train" | "check">("train");
  const [trainRunning, setTrainRunning] = useState(false);
  const [trainSpeed, setTrainSpeed] = useState(200);
  const trainTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const trainSpeedRef = useRef(200);
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
  const resetStage = () => { setPrepStage(0); setPrepStagePlaying(false); if (stageTimer.current) { clearInterval(stageTimer.current); stageTimer.current = null; } };
  // When you move to a different step/column, restart the animation and default the method from that step.
  useEffect(() => { resetStage(); const st = steps[Math.min(Math.max(1, prepStepIdx), steps.length || 1) - 1]; if (st && st.op === "Impute missing") setPrepImputeMethod(st.method); else if (st && st.op === "Scale / normalize") setPrepScaleMethod(st.method); else if (st && st.op === "Encode categorical") setPrepEncodeMethod(st.method); }, [prepStepIdx, prepCol, steps]);
  // When you pick a different method, restart the animation (but keep the selection).
  useEffect(() => { resetStage(); }, [prepImputeMethod, prepScaleMethod, prepEncodeMethod]);
  useEffect(() => () => { if (stageTimer.current) clearInterval(stageTimer.current); }, []);
  // Model-step walkthrough resets when the algorithm / task / features change.
  useEffect(() => { setModelStage(0); setModelApplied(0); setModelApplying(false); if (modelTimer.current) { clearInterval(modelTimer.current); modelTimer.current = null; } if (modelApplyTimer.current) { clearInterval(modelApplyTimer.current); modelApplyTimer.current = null; } }, [algo, task, features, target]);
  useEffect(() => () => { if (modelTimer.current) clearInterval(modelTimer.current); if (modelApplyTimer.current) clearInterval(modelApplyTimer.current); }, []);
  // Train-step walkthrough resets when the model / task / features / retrain change.
  useEffect(() => { setTrainT(0); setCheckT(0); setTrainPhase("train"); setTrainRunning(false); if (trainTimer.current) { clearInterval(trainTimer.current); trainTimer.current = null; } }, [algo, task, features, target, result]);
  useEffect(() => () => { if (trainTimer.current) clearInterval(trainTimer.current); }, []);
  // Heavy Plotly data for the train step (computed once per settings/retrain, not per tick):
  // GD → evolving boundary/line frames; other classifiers → decision-region surface; other regressors → prediction curve.
  const trainVizData = useMemo((): { kind: "gd"; anim: ReturnType<typeof gdAnim>; f1: string; f2: string } | { kind: "surface"; surf: ReturnType<typeof decisionSurface>; f1: string; f2: string } | { kind: "regcurve"; pts: { x: number; y: number; yh: number }[]; f1: string } | null => {
    if (mlMode !== "maths" || step !== "train" || !ds || !result || !modelRef.current) return null;
    const nums = features.filter((f) => ds.columns.find((c) => c.name === f)?.type === "num");
    if (!nums.length) return null;
    const f1 = dbF1 && nums.includes(dbF1) ? dbF1 : nums[0];
    const f2 = dbF2 && nums.includes(dbF2) && dbF2 !== f1 ? dbF2 : (nums.find((x) => x !== f1) || "");
    const cfg = cfgNow();
    const k = ["LogisticRegression", "LinearRegression", "Ridge"].includes(algo) ? "gd" : algo === "GaussianNB" ? "gnb" : algo === "DecisionTree" ? "tree" : algo === "RandomForest" ? "forest" : "knn";
    try {
      if (k === "gd") return { kind: "gd", anim: f1 ? gdAnim(ds, target, f1, f2, cfg) : null, f1, f2 };
      if (task === "classification") return { kind: "surface", surf: f1 && f2 ? decisionSurface(ds, target, f1, f2, cfg) : null, f1, f2 };
      const b2 = buildMatrix(ds, features, target, task, steps);
      const idxF = b2.featureNames.indexOf(f1) >= 0 ? b2.featureNames.indexOf(f1) : 0;
      const pr = predict(modelRef.current, b2.X);
      const pts = b2.X.map((r, i) => ({ x: r[idxF], y: b2.y[i], yh: pr[i] })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.yh)).sort((a, c) => a.x - c.x);
      return { kind: "regcurve", pts, f1 };
    } catch { return null; }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mlMode, step, ds, target, features, task, algo, params, dbF1, dbF2, result, steps, testSize]);

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
  const modelStageStop = () => { if (modelTimer.current) { clearInterval(modelTimer.current); modelTimer.current = null; } };
  const modelApplyStop = () => { if (modelApplyTimer.current) { clearInterval(modelApplyTimer.current); modelApplyTimer.current = null; } setModelApplying(false); };
  // Slowly apply the (now fully-substituted) formula across every row so the graph fills in.
  function modelApplyStart(total: number) {
    if (modelApplyTimer.current) { clearInterval(modelApplyTimer.current); modelApplyTimer.current = null; }
    modelApplyTimer.current = setInterval(() => { setModelApplied((a) => { if (a >= total) { if (modelApplyTimer.current) { clearInterval(modelApplyTimer.current); modelApplyTimer.current = null; } setModelApplying(false); return a; } return a + 1; }); }, modelSpeedRef.current);
  }
  function modelApplyAll(maxStage: number, total: number) {
    modelStageStop(); setModelStage(maxStage);
    if (modelApplyTimer.current) { modelApplyStop(); return; }
    setModelApplied((a) => (a >= total ? 0 : a)); setModelApplying(true);
    modelApplyStart(total);
  }
  const setSpeed = (ms: number, total: number) => { setModelSpeed(ms); modelSpeedRef.current = ms; if (modelApplyTimer.current) modelApplyStart(total); };
  // Controls: step through the substitution chains, then apply the formula to every row.
  const modelStageControls = (maxStage: number, total: number) => (
    <div className="prep-ctl" style={{ flexWrap: "wrap" }}>
      <button className="btn ghost sm" disabled={modelStage <= 0} onClick={() => { modelStageStop(); modelApplyStop(); setModelApplied(0); setModelStage((s) => Math.max(0, s - 1)); }}>← back</button>
      <button className="btn sm" disabled={modelStage >= maxStage} onClick={() => { modelStageStop(); modelApplyStop(); setModelStage((s) => Math.min(maxStage, s + 1)); }}>step →</button>
      <button className="btn sm" onClick={() => modelApplyAll(maxStage, total)}>{modelApplying ? "⏸ pause" : "▶ apply to all rows"}</button>
      <button className="btn ghost sm" onClick={() => { modelStageStop(); modelApplyStop(); setModelApplied(0); setModelStage(0); }}>↺ reset</button>
      <span className="note mono" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>speed
        <select className="mini-sel" value={modelSpeed} onChange={(e) => setSpeed(+e.target.value, total)}>
          <option value={520}>0.5×</option>
          <option value={260}>1×</option>
          <option value={120}>2×</option>
          <option value={45}>4×</option>
        </select>
      </span>
      <span className="note mono" style={{ marginLeft: "auto" }}>step {Math.min(modelStage, maxStage)} / {maxStage}</span>
    </div>
  );
  // Train-step controls: fit the model (fills the table + animation), then check the metric.
  const trainStop = () => { if (trainTimer.current) { clearInterval(trainTimer.current); trainTimer.current = null; } setTrainRunning(false); };
  function trainRun(kind: "train" | "check", total: number) {
    if (trainTimer.current) { trainStop(); return; } // toggle pause
    setTrainPhase(kind);
    if (kind === "train") setTrainT((a) => (a >= total ? 0 : a)); else setCheckT((a) => (a >= total ? 0 : a));
    setTrainRunning(true);
    trainTimer.current = setInterval(() => {
      const bump = (a: number) => { if (a >= total) { if (trainTimer.current) { clearInterval(trainTimer.current); trainTimer.current = null; } setTrainRunning(false); return a; } return a + 1; };
      if (kind === "train") setTrainT(bump); else setCheckT(bump);
    }, trainSpeedRef.current);
  }
  const setTrainSpeedFn = (ms: number) => { setTrainSpeed(ms); trainSpeedRef.current = ms; };
  const trainReset = () => { trainStop(); setTrainT(0); setCheckT(0); setTrainPhase("train"); };
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
    const fv = (x: number | undefined, dp = 2) => (typeof x === "number" && Number.isFinite(x) ? x.toFixed(dp) : "—");
    const pick = (m: number, mx: number) => (m <= mx ? [...Array(m).keys()] : [...Array(mx).keys()].map((k) => Math.floor(k * (m / mx))));
    const colj = (j: number) => X.map((r) => (Number.isFinite(r[j]) ? r[j] : NaN));
    // Normalise ignoring non-finite values (used to sort rows along a feature).
    const nrm = (arr: number[]) => { const f = arr.filter((v) => Number.isFinite(v)); const mn = f.length ? Math.min(...f) : 0, mx = f.length ? Math.max(...f) : 1, d = (mx - mn) || 1; return { mn, mx, z: arr.map((v) => (Number.isFinite(v) ? (v - mn) / d : 0)) }; };
    const idx = pick(n, 46).filter((i) => Number.isFinite(Y[i]));
    const idxF = names.indexOf(dbF1) >= 0 ? names.indexOf(dbF1) : 0;
    const idxG = names.indexOf(dbF2) >= 0 && names.indexOf(dbF2) !== idxF ? names.indexOf(dbF2) : Math.min(nf - 1, idxF + 1);
    const st = modelStage;
    // Rich Plotly graph for the model mechanism (with legend), matching the train step.
    const th = plotlyTheme();
    const legendLayout = { showlegend: true, legend: { orientation: "h" as const, y: -0.2, font: { size: 11 } } };
    const mPlot = (data: Record<string, unknown>[], title: string, xlab: string, ylab: string, extra: Record<string, unknown> = {}): React.ReactNode => <Plot data={data} layout={{ ...chartLayout(th, title, xlab, ylab), ...legendLayout, height: 400, ...extra }} style={{ height: 400, width: "100%" }} />;
    const xa = (i: number) => X[i]?.[idxF]; const xb = (i: number) => X[i]?.[idxG];
    const metric = (ids: number[]): number => { if (!ids.length) return 0; if (isReg) { const ys = ids.map((i) => Y[i]); const mu = ys.reduce((a, c) => a + c, 0) / ys.length; return ys.reduce((a, c) => a + (c - mu) ** 2, 0) / ys.length; } const c: Record<number, number> = {}; ids.forEach((i) => { c[Y[i]] = (c[Y[i]] || 0) + 1; }); let s = 1; for (const k in c) { const p = c[k] / ids.length; s -= p * p; } return s; };
    const bestSplit = (ids: number[], feats: number[]) => {
      const parent = metric(ids); let best = { gain: -1, feat: feats[0], thr: 0, gl: 0, gr: 0, nl: 0, nr: 0 };
      for (const f of feats) { const vals = [...new Set(ids.map((i) => X[i][f]).filter((v) => Number.isFinite(v)))].sort((a, c) => a - c); for (let t = 0; t < vals.length - 1; t++) { const thr = (vals[t] + vals[t + 1]) / 2; const L = ids.filter((i) => X[i][f] <= thr), R = ids.filter((i) => X[i][f] > thr); if (!L.length || !R.length) continue; const gl = metric(L), gr = metric(R); const gain = parent - (L.length / ids.length) * gl - (R.length / ids.length) * gr; if (gain > best.gain) best = { gain, feat: f, thr, gl, gr, nl: L.length, nr: R.length }; } }
      return { ...best, parent };
    };

    let intro = ""; let legend: { sym: string; desc: string; how?: string; val?: string }[] = [];
    // chainFor(row): the formula for THAT row — symbolic → its real numbers → its result.
    // So single-stepping shows the example row, and apply-to-all re-fills it per row.
    let chainFor: (r: number) => { sym: string; sub: string; res: string }[] = () => [];
    // viz(applied, curRow, stage): applied>0 fills the graph & highlights curRow;
    // applied===0 shows just the one example row executing through the formula stages.
    let viz: (applied: number, curRow: number, stage: number) => React.ReactNode = () => null;
    let total = idx.length; let applyNote = "each row is scored one at a time";
    let exampleRow = idx[0] ?? 0; let orderList: number[] = idx;
    const kind = modelKind; // "gd" | "tree" | "forest" | "gnb" | "knn"
    // b + w₁(x₁) + w₂(x₂) + … for one row, as a KaTeX string with the real numbers.
    const linSub = (w: number[], x: number[]) => `${fv(w[0])} ${x.slice(0, 2).map((xi, j) => `${(w[j + 1] ?? 0) < 0 ? "-" : "+"}\\,${fv(Math.abs(w[j + 1] ?? 0))}\\!\\times\\!${fv(xi)}`).join(" ")}${x.length > 2 ? " + \\dots" : ""}`;

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
        { sym: "x", desc: "a feature", how: `the ${names[idxF]} cell of row 0`, val: fv(X[0]?.[idxF]) },
        { sym: "\\mathbf w, b", desc: "weights & bias", how: `gradient descent over your ${n} rows`, val: gw ? `b=${fv(gw[0])}, w₁=${fv(gw[1])}` : "—" },
        { sym: "z", desc: "linear score", how: "z = b + Σⱼ wⱼ·xⱼ over the row", val: fv(z0) },
        { sym: "\\hat y", desc: `P(${classes?.[1] ?? "class 1"})`, how: "ŷ = 1 / (1 + e⁻ᶻ)", val: fv(p0) },
      ];
      const cls1 = (classes?.[1] ?? "class 1").toString().replace(/[^a-zA-Z0-9 ]/g, " "), cls0 = (classes?.[0] ?? "class 0").toString().replace(/[^a-zA-Z0-9 ]/g, " ");
      exampleRow = idx[0] ?? 0; orderList = order;
      chainFor = (r) => { const zr = z(X[r]), pr = sig(zr); return [
        { sym: "z = b + \\sum_j w_j x_j", sub: gw ? `z = ${linSub(gw, X[r])}` : "z = \\dots", res: `z = ${fv(zr)}` },
        { sym: "\\hat y = \\sigma(z) = \\tfrac{1}{1+e^{-z}}", sub: `\\hat y = \\tfrac{1}{1+e^{-(${fv(zr)})}}`, res: `\\hat y = ${fv(pr)}` },
        { sym: "\\text{class} = 1 \\text{ if } \\hat y \\ge 0.5", sub: `${fv(pr)} \\ge 0.5`, res: `\\Rightarrow \\text{${pr >= 0.5 ? cls1 : cls0}}` },
      ]; };
      applyNote = "each row's score becomes a probability on the curve";
      viz = (applied, curRow, stage) => {
        const shown = order.slice(0, applied);
        const c0 = shown.filter((i) => Y[i] === 0), c1 = shown.filter((i) => Y[i] === 1);
        const data: Record<string, unknown>[] = [
          { type: "scatter", mode: "lines", name: `P(${cls1}) curve`, x: order.map(xa), y: order.map((i) => pz[i]), line: { color: "#3ecf7f", width: 2.5 } },
          { type: "scatter", mode: "markers", name: cls0, x: c0.map(xa), y: c0.map((i) => pz[i]), marker: { size: 7, color: PAL_ML[0], line: { width: 1, color: th.paper } } },
          { type: "scatter", mode: "markers", name: cls1, x: c1.map(xa), y: c1.map((i) => pz[i]), marker: { size: 7, color: PAL_ML[1], line: { width: 1, color: th.paper } } },
        ];
        if (applied === 0) data.push({ type: "scatter", mode: "markers", name: "this row", x: [xa(curRow)], y: [stage >= 2 ? pz[curRow] : (Y[curRow] ? 0.86 : 0.14)], marker: { size: 14, color: PAL_ML[Y[curRow] % PAL_ML.length], line: { width: 2, color: th.text } } });
        return mPlot(data, "each row → its probability; the 0.5 line splits the class", names[idxF], `P(${cls1})`, { shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: "#f59e0b", dash: "dash", width: 1.5 } }] });
      };
    } else if (kind === "gd" && isReg) { // Linear / Ridge Regression
      let gw: number[] | null = null;
      try { const gt = gdTrace(cfgNow(), X, Y, 0); const last = gt?.snaps?.[(gt?.snaps?.length ?? 0) - 1]; gw = last?.w || null; } catch { /* ignore */ }
      const pred = (r: number[]) => gw ? gw[0] + r.reduce((a, v, j) => a + (gw![j + 1] || 0) * v, 0) : 0;
      const f0 = nrm(colj(idxF)); const yh = X.map(pred);
      const msePairs = idx.filter((i) => Number.isFinite(yh[i])); const mse = msePairs.length ? msePairs.reduce((a, i) => a + (yh[i] - Y[i]) ** 2, 0) / msePairs.length : NaN;
      const order = [...idx].sort((a, c) => f0.z[a] - f0.z[c]);
      intro = "Fit a straight line so the average squared vertical gap (residual) between the line and your points is as small as possible.";
      legend = [
        { sym: "x", desc: "a feature", how: `the ${names[idxF]} cell of row 0`, val: fv(X[0]?.[idxF]) },
        { sym: "\\mathbf w, b", desc: "slope & intercept", how: `least-squares fit over your ${n} rows`, val: gw ? `b=${fv(gw[0])}, w₁=${fv(gw[1])}` : "—" },
        { sym: "\\hat y", desc: "prediction", how: "ŷ = b + Σⱼ wⱼ·xⱼ", val: gw ? fv(pred(X[0])) : "—" },
        { sym: "\\mathcal L", desc: "mean squared error", how: `mean of (ŷ−y)² over ${msePairs.length} rows`, val: fv(mse) },
      ];
      exampleRow = idx[0] ?? 0; orderList = order;
      chainFor = (r) => { const yhr = pred(X[r]), rr = yhr - Y[r]; return [
        { sym: "\\hat y = b + \\sum_j w_j x_j", sub: gw ? `\\hat y = ${linSub(gw, X[r])}` : "\\hat y = \\dots", res: `\\hat y = ${fv(yhr)}` },
        { sym: "r = \\hat y - y", sub: `r = ${fv(yhr)} - ${fv(Y[r])}`, res: `r = ${fv(rr)}` },
        { sym: "\\mathcal L = \\tfrac1n\\sum r_i^2", sub: `\\mathcal L = \\tfrac1{${msePairs.length}}(${fv(rr * rr)} + \\dots)`, res: `\\mathcal L = ${fv(mse)}` },
      ].concat(algo === "Ridge" ? [{ sym: "\\mathcal L \\mathrel{+}= \\alpha\\lVert\\mathbf w\\rVert^2", sub: "\\text{add the L2 penalty}", res: "\\text{shrinks the weights}" }] : []); };
      applyNote = "each row's prediction lands on the line";
      viz = (applied, curRow) => {
        const shown = (applied === 0 ? [curRow] : order.slice(0, applied));
        const rx: (number | null)[] = [], ry: (number | null)[] = [];
        shown.forEach((i) => { rx.push(xa(i), xa(i), null); ry.push(Y[i], yh[i], null); });
        const data: Record<string, unknown>[] = [
          { type: "scatter", mode: "lines", name: "fitted line ŷ", x: order.map(xa), y: order.map((i) => yh[i]), line: { color: "#f59e0b", width: 2.5 } },
          { type: "scatter", mode: "lines", name: "residual (ŷ−y)", x: rx, y: ry, line: { color: "#ef4444", width: 1 }, opacity: 0.55, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", name: "actual y", x: shown.map(xa), y: shown.map((i) => Y[i]), marker: { size: 7, color: "#5b7cff", line: { width: 1, color: th.paper } } },
        ];
        if (applied === 0) data.push({ type: "scatter", mode: "markers", name: "this row", x: [xa(curRow)], y: [Y[curRow]], marker: { size: 14, color: "#5b7cff", line: { width: 2, color: th.text } } });
        return mPlot(data, `${label} — least-squares fit · MSE = ${fv(mse)}`, names[idxF], target);
      };
    } else if (kind === "knn") {
      const kk = Math.max(1, Math.round(Number(params.n_neighbors) || 5));
      const qi = idx[Math.floor(idx.length / 2)] ?? 0; const q = X[qi];
      const dist = (r: number[]) => Math.hypot(...r.map((v, j) => v - q[j]));
      const others = idx.filter((i) => i !== qi);
      const near = [...others].sort((a, c) => dist(X[a]) - dist(X[c])).slice(0, kk);
      const dNear = near.length ? dist(X[near[0]]) : NaN;
      const votes = near.filter((i) => Y[i] === 1).length;
      const yhatNum = isReg && near.length ? near.reduce((a, i) => a + Y[i], 0) / near.length : NaN;
      const yhat = isReg ? fv(yhatNum) : String(classes?.[votes * 2 >= kk ? 1 : 0] ?? "—");
      intro = isReg ? "No training. For a new point, measure the distance to every stored point, keep the k nearest, and average their target." : "No training. For a new point, measure the distance to every stored point, keep the k nearest, and let them vote.";
      legend = [
        { sym: "d", desc: "Euclidean distance", how: `√Σⱼ(qⱼ−xⱼ)² across ${nf} features`, val: `${fv(dNear)} (nearest)` },
        { sym: "k", desc: "neighbours kept", how: `n_neighbors setting = ${kk}`, val: String(kk) },
        { sym: "\\hat y", desc: isReg ? "average of k" : "majority of k", how: isReg ? `mean target of the ${kk} nearest` : `most common label among the ${kk}`, val: yhat },
      ];
      const qx1 = X[qi]?.[idxF], qx2 = X[qi]?.[idxG];
      const nearD = near.slice(0, 5).map((i) => fv(dist(X[i]))).join(",\\ ");
      exampleRow = near[0] ?? others[0] ?? qi; orderList = others;
      chainFor = (r) => { const rx1 = X[r]?.[idxF], rx2 = X[r]?.[idxG], dr = dist(X[r]); return [
        { sym: "d = \\sqrt{\\textstyle\\sum_j (q_j - x_j)^2}", sub: `d = \\sqrt{(${fv(qx1)}\\!-\\!${fv(rx1)})^2 + (${fv(qx2)}\\!-\\!${fv(rx2)})^2 + \\dots}`, res: `d = ${fv(dr)}` },
        { sym: `\\text{keep the } k=${kk} \\text{ smallest } d`, sub: `d = [${nearD}]`, res: `${kk} \\text{ neighbours}` },
        isReg
          ? { sym: "\\hat y = \\tfrac1k\\textstyle\\sum_{i\\in kNN} y_i", sub: `\\hat y = \\tfrac1{${kk}}(${near.map((i) => fv(Y[i])).join("+")})`, res: `\\hat y = ${yhat}` }
          : { sym: "\\hat y = \\text{mode of the labels}", sub: `\\text{votes} = ${votes}\\text{ vs }${kk - votes}`, res: `\\Rightarrow \\text{${yhat.replace(/[^a-zA-Z0-9 ]/g, " ")}}` },
      ]; };
      total = others.length; applyNote = "distance measured to each stored row";
      viz = (applied, curRow) => {
        const done = applied >= total;
        const lineRows = applied === 0 ? [curRow] : others.slice(0, applied);
        const lx: (number | null)[] = [], ly: (number | null)[] = [];
        lineRows.forEach((i) => { lx.push(xa(qi), xa(i), null); ly.push(xb(qi), xb(i), null); });
        const data: Record<string, unknown>[] = [
          { type: "scatter", mode: "lines", name: "distance", x: lx, y: ly, line: { color: "#a855f7", width: applied === 0 ? 1.6 : 0.7 }, opacity: 0.5, hoverinfo: "skip" },
          ...(isReg
            ? [{ type: "scatter", mode: "markers", name: "stored rows", x: others.map(xa), y: others.map(xb), marker: { size: 6, color: "#5b7cff", opacity: done ? 0.4 : 0.75 } }]
            : [0, 1].map((cl) => ({ type: "scatter", mode: "markers", name: String(classes?.[cl] ?? cl), x: others.filter((i) => Y[i] === cl).map(xa), y: others.filter((i) => Y[i] === cl).map(xb), marker: { size: 6, color: PAL_ML[cl], opacity: done ? 0.4 : 0.8 } }))),
          { type: "scatter", mode: "markers", name: `the k=${kk} nearest`, x: (done ? near : []).map(xa), y: (done ? near : []).map(xb), marker: { size: 12, color: "rgba(0,0,0,0)", line: { width: 2.5, color: "#a855f7" } } },
          { type: "scatter", mode: "markers", name: "new point ★", x: [xa(qi)], y: [xb(qi)], marker: { size: 16, symbol: "star", color: th.text } },
        ];
        return mPlot(data, done ? `k nearest ${isReg ? "averaged" : "voted"} → ${yhat}` : "distance from ★ to each stored row", names[idxF], names[idxG]);
      };
    } else if (kind === "gnb") {
      const cA = idx.filter((i) => Y[i] === 0), cB = idx.filter((i) => Y[i] === 1);
      const colF = colj(idxF);
      const mV = (ids: number[]) => { const v = ids.map((i) => colF[i]).filter((c) => Number.isFinite(c)); const mu = v.length ? v.reduce((a, c) => a + c, 0) / v.length : 0; const va = (v.length ? v.reduce((a, c) => a + (c - mu) ** 2, 0) / v.length : 1) || 1; return { mu, va }; };
      const sA = mV(cA.length ? cA : idx), sB = mV(cB.length ? cB : idx);
      const priorA = (cA.length || 1) / (idx.length || 1), priorB = (cB.length || 1) / (idx.length || 1);
      const cf = colF.filter((c) => Number.isFinite(c)); const range = [cf.length ? Math.min(...cf) : 0, cf.length ? Math.max(...cf) : 1];
      const xq = Number.isFinite(X[0]?.[idxF]) ? X[0][idxF] : sA.mu;
      const bell = (x: number, mu: number, va: number) => Math.exp(-((x - mu) ** 2) / (2 * va));
      const likeA = bell(xq, sA.mu, sA.va);
      intro = "Learn one bell curve (mean & variance) per class. A point's class score is that class's prior times the bell heights at the point.";
      legend = [
        { sym: "\\mu", desc: `class mean (${names[idxF]})`, how: `sum of ${names[idxF]} ÷ ${cA.length} rows of class ${classes?.[0] ?? 0}`, val: fv(sA.mu) },
        { sym: "\\sigma^2", desc: "class variance", how: `avg of (x−μ)² over those ${cA.length} rows`, val: fv(sA.va) },
        { sym: "P(y)", desc: "class prior", how: `rows in class ÷ total = ${cA.length}/${idx.length}`, val: fv(priorA) },
        { sym: "P(x\\mid y)", desc: "bell height at x", how: `Gaussian at x = ${fv(xq)} (row 0)`, val: fv(likeA) },
      ];
      const clsA = (classes?.[0] ?? "A").toString().replace(/[^a-zA-Z0-9 ]/g, " "), clsB = (classes?.[1] ?? "B").toString().replace(/[^a-zA-Z0-9 ]/g, " ");
      const rowOrder = [...idx].sort((a, c) => colF[a] - colF[c]);
      const xOf = (r: number) => (Number.isFinite(colF[r]) ? colF[r] : sA.mu);
      exampleRow = rowOrder[Math.floor(rowOrder.length / 2)] ?? (idx[0] ?? 0); orderList = rowOrder;
      chainFor = (r) => { const xr = xOf(r); const lA = bell(xr, sA.mu, sA.va), lB = bell(xr, sB.mu, sB.va), pA = priorA * lA, pB = priorB * lB; return [
        { sym: "P(x\\mid y) = \\tfrac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}", sub: `P(${fv(xr)}\\mid ${clsA}) = e^{-\\frac{(${fv(xr)}-${fv(sA.mu)})^2}{2\\cdot${fv(sA.va)}}}`, res: `= ${fv(lA)}` },
        { sym: "P(y\\mid x) \\propto P(y)\\,P(x\\mid y)", sub: `${clsA}: ${fv(priorA)}\\!\\times\\!${fv(lA)},\\ ${clsB}: ${fv(priorB)}\\!\\times\\!${fv(lB)}`, res: `\\propto ${fv(pA)} \\text{ vs } ${fv(pB)}` },
        { sym: "\\hat y = \\arg\\max_y P(y\\mid x)", sub: `\\max(${fv(pA)},\\ ${fv(pB)})`, res: `\\Rightarrow \\text{${pA >= pB ? clsA : clsB}}` },
      ]; };
      total = idx.length; applyNote = "each row scored under its class bell";
      viz = (applied, curRow) => {
        const xs: number[] = []; for (let t = 0; t <= 60; t++) xs.push(range[0] + (range[1] - range[0]) * (t / 60));
        const shown = rowOrder.slice(0, applied); const sA0 = shown.filter((i) => Y[i] === 0), sB0 = shown.filter((i) => Y[i] === 1);
        const data: Record<string, unknown>[] = [
          { type: "scatter", mode: "lines", name: `${clsA} bell`, x: xs, y: xs.map((x) => bell(x, sA.mu, sA.va)), line: { color: PAL_ML[0], width: 2.5 } },
          { type: "scatter", mode: "lines", name: `${clsB} bell`, x: xs, y: xs.map((x) => bell(x, sB.mu, sB.va)), line: { color: PAL_ML[1], width: 2.5 } },
          { type: "scatter", mode: "markers", name: `${clsA} rows`, x: sA0.map(xOf), y: sA0.map((i) => bell(xOf(i), sA.mu, sA.va)), marker: { size: 6, color: PAL_ML[0] } },
          { type: "scatter", mode: "markers", name: `${clsB} rows`, x: sB0.map(xOf), y: sB0.map((i) => bell(xOf(i), sB.mu, sB.va)), marker: { size: 6, color: PAL_ML[1] } },
        ];
        const extra: Record<string, unknown> = {};
        if (applied === 0) { const xr = xOf(curRow); data.push({ type: "scatter", mode: "markers", name: "this row under each bell", x: [xr, xr], y: [bell(xr, sA.mu, sA.va), bell(xr, sB.mu, sB.va)], marker: { size: 13, color: th.text, line: { width: 2, color: th.paper } } }); extra.shapes = [{ type: "line", x0: xr, x1: xr, yref: "paper", y0: 0, y1: 1, line: { color: th.muted, dash: "dot", width: 1 } }]; }
        return mPlot(data, "each row scored under its class bell", names[idxF], "likelihood  P(x | class)", extra);
      };
    } else { // tree & forest
      const rowsAll = pick(n, 260).filter((i) => Number.isFinite(Y[i]) && Number.isFinite(X[i][idxF]) && Number.isFinite(X[i][idxG]));
      const sp = bestSplit(rowsAll, [idxF, idxG]);
      const leftIds = rowsAll.filter((i) => X[i][sp.feat] <= sp.thr);
      const sp2 = leftIds.length > 3 ? bestSplit(leftIds, [idxF, idxG]) : null;
      const metricName = isReg ? "variance" : "Gini";
      if (kind === "tree") {
        intro = "Greedily test every feature threshold, keep the split that removes the most impurity, then recurse into if/else rules on each side.";
        legend = [
          { sym: "G", desc: `${metricName} impurity`, how: isReg ? `variance of the target over ${sp.nl + sp.nr} rows` : `1 − Σₖ pₖ² over ${sp.nl + sp.nr} rows`, val: fv(sp.parent, 3) },
          { sym: "n_L, n_R", desc: "rows each side", how: `count of rows with ${names[sp.feat]} ≤ / > thr`, val: `${sp.nl}, ${sp.nr}` },
          { sym: "\\text{gain}", desc: "impurity removed", how: "G − (n_L/n)G_L − (n_R/n)G_R", val: fv(sp.gain, 3) },
          { sym: "\\text{thr}", desc: "chosen threshold", how: `the ${names[sp.feat]} cut that maximises gain`, val: fv(sp.thr) },
        ];
        const pc: Record<number, number> = {}; rowsAll.forEach((i) => { pc[Y[i]] = (pc[Y[i]] || 0) + 1; }); const shares = Object.values(pc).map((c) => c / (rowsAll.length || 1));
        const featC = names[sp.feat].replace(/[^a-zA-Z0-9 ]/g, " ").slice(0, 12);
        const plotPts = rowsAll.filter((i) => idx.includes(i));
        exampleRow = plotPts[0] ?? (idx[0] ?? 0); orderList = plotPts;
        chainFor = (r) => { const xr = X[r]?.[sp.feat]; const left = xr <= sp.thr; return [
          { sym: `G = ${isReg ? "\\text{var}(y)" : "1 - \\sum_k p_k^2"}`, sub: isReg ? "G = \\text{var}(y)" : `G = 1 - (${shares.map((p) => `${fv(p)}^2`).join(" + ")})`, res: `G = ${fv(sp.parent, 3)}` },
          { sym: "\\text{gain} = G - \\tfrac{n_L}{n}G_L - \\tfrac{n_R}{n}G_R", sub: `= ${fv(sp.parent)} - \\tfrac{${sp.nl}}{${sp.nl + sp.nr}}(${fv(sp.gl)}) - \\tfrac{${sp.nr}}{${sp.nl + sp.nr}}(${fv(sp.gr)})`, res: `\\text{gain} = ${fv(sp.gain)}` },
          { sym: `\\text{route the row: } \\text{${featC}} \\le \\text{thr}?`, sub: `${fv(xr)} \\le ${fv(sp.thr)}`, res: `\\Rightarrow \\text{${left ? "left" : "right"}}` },
        ]; };
        total = plotPts.length; applyNote = "each row falls to its side of the split";
        const cutShape = (feat: number, thr: number, color: string, dash?: string) => (feat === idxF
          ? { type: "line", x0: thr, x1: thr, yref: "paper", y0: 0, y1: 1, line: { color, width: 2, dash } }
          : { type: "line", xref: "paper", x0: 0, x1: 1, y0: thr, y1: thr, line: { color, width: 2, dash } });
        viz = (applied, curRow) => {
          const shown = (applied === 0 ? [curRow] : plotPts.slice(0, applied));
          const leftPts = shown.filter((i) => X[i][sp.feat] <= sp.thr), rightPts = shown.filter((i) => X[i][sp.feat] > sp.thr);
          const shapes: Record<string, unknown>[] = [cutShape(sp.feat, sp.thr, "#5b7cff")];
          if (applied >= total && sp2) shapes.push(cutShape(sp2.feat, sp2.thr, "#3ecf7f", "dash"));
          const data: Record<string, unknown>[] = [
            { type: "scatter", mode: "markers", name: `${featC} ≤ ${fv(sp.thr)} (left)`, x: leftPts.map(xa), y: leftPts.map(xb), marker: { size: 7, color: PAL_ML[0], line: { width: 1, color: th.paper } } },
            { type: "scatter", mode: "markers", name: `${featC} > ${fv(sp.thr)} (right)`, x: rightPts.map(xa), y: rightPts.map(xb), marker: { size: 7, color: PAL_ML[1], opacity: 0.7, line: { width: 1, color: th.paper } } },
          ];
          if (applied === 0) data.push({ type: "scatter", mode: "markers", name: "this row", x: [xa(curRow)], y: [xb(curRow)], marker: { size: 14, color: th.text, line: { width: 2, color: th.paper } } });
          return mPlot(data, `best cut: ${featC} ≤ ${fv(sp.thr)} · gain ${fv(sp.gain)}`, names[idxF], names[idxG], { shapes });
        };
      } else { // forest
        const m = Math.max(1, Math.round(Number(params.nTrees) || 3));
        const shown = Math.min(3, m);
        const votes = [...Array(shown).keys()].map((i) => (i % 3 === 2 ? 0 : 1));
        const winner = votes.filter((v) => v === 1).length * 2 >= votes.length ? (classes?.[1] ?? "B") : (classes?.[0] ?? "A");
        intro = "Train many trees, each on a bootstrap resample with a random feature subset, then average their votes to cut variance.";
        legend = [
          { sym: "m", desc: "number of trees", how: `n_estimators setting = ${m}`, val: String(m) },
          { sym: "\\text{bootstrap}", desc: "resample rows", how: "draw n rows with replacement per tree", val: `${n} → ${n}` },
          { sym: "T_i", desc: "the i-th tree", how: "fit on its own bootstrap sample", val: `${shown} shown` },
          { sym: "\\hat y", desc: isReg ? "mean of trees" : "majority vote", how: isReg ? "ŷ = (1/m) Σᵢ Tᵢ(x)" : "ŷ = mode{T₁…Tₘ}", val: isReg ? "avg" : String(winner) },
        ];
        const winnerC = (isReg ? "mean" : winner).toString().replace(/[^a-zA-Z0-9 ]/g, " ");
        const voteLbl = (t: number) => (classes?.[votes[t]] ?? (votes[t] ? "B" : "A")).toString().replace(/[^a-zA-Z0-9 ]/g, " ");
        exampleRow = 0; orderList = [...Array(shown).keys()]; total = shown;
        chainFor = (t) => [
          { sym: "\\text{each } T_i: \\text{bootstrap} + \\text{random features}", sub: `T_{${t + 1}}: \\text{sample } ${n} \\text{ rows}`, res: `\\text{tree } ${t + 1}` },
          { sym: "\\text{every tree predicts}", sub: `T_{${t + 1}} \\to \\text{${voteLbl(t)}}`, res: `${shown} \\text{ votes}` },
          isReg
            ? { sym: "\\hat y = \\tfrac1m\\textstyle\\sum_i T_i(\\mathbf x)", sub: "\\text{average the tree outputs}", res: "\\hat y = \\text{mean}" }
            : { sym: "\\hat y = \\text{mode}\\{T_1,\\dots,T_m\\}", sub: `\\text{majority of } ${shown}`, res: `\\Rightarrow \\text{${winnerC}}` },
        ];
        applyNote = "each tree casts its vote, then they aggregate";
        viz = (applied, curRow, stage) => {
          const revealed = [...Array(shown).keys()].filter((i) => applied > i || (applied === 0 && i === curRow && stage >= 2));
          const data: Record<string, unknown>[] = [{ type: "bar", orientation: "h", name: "each tree's vote", x: revealed.map(() => 1), y: revealed.map((i) => `T${i + 1}`), marker: { color: revealed.map((i) => PAL_ML[votes[i]]) }, text: revealed.map((i) => `→ ${voteLbl(i)}`), textposition: "inside", insidetextanchor: "middle", hoverinfo: "skip" }];
          return mPlot(data, applied >= total ? `${shown} trees → majority vote = ${isReg ? "mean" : winner}` : "each tree casts its vote (colour = predicted class)", "one vote per tree", "tree", { xaxis: { range: [0, 1.15], showticklabels: false, zeroline: false } });
        };
      }
    }

    const applied = Math.min(modelApplied, total);
    // curRow: which row's numbers the formula shows — the example row while stepping,
    // the row currently being applied once "apply to all rows" is running.
    const curRow = applied > 0 ? (orderList[Math.min(applied, orderList.length) - 1] ?? exampleRow) : exampleRow;
    const chains = chainFor(curRow);
    const maxStage = chains.length;
    const revealAll = applied > 0; // apply-to-all keeps every line visible, updating per row
    const rowTag = kind === "forest" ? "tree" : "row";
    return mCard(`${label} — how it learns`, <>
      <div className="note" style={{ marginBottom: 10, lineHeight: 1.6 }}>{intro}</div>
      {varLegend(legend)}
      <div className="prep-2col">
        <div className="prep-col">
          <div className="prep-col-h">the formula — filled in from {applied > 0 ? `${rowTag} ${applied}` : "one example " + rowTag}</div>
          {chains.map((s, i) => { const on = revealAll || st >= i + 1; return <div key={i} className={`fx-chain ${on ? "on" : ""}`}>
            <div className="fx-cl"><span className="fx-tag">formula</span><Katex tex={s.sym} /></div>
            {on && s.sub ? <div className="fx-cl"><span className="fx-tag">your numbers</span><Katex tex={s.sub} /></div> : null}
            {on && s.res ? <div className="fx-cl"><span className="fx-tag">result</span><span className="fx-res"><Katex tex={s.res} /></span></div> : null}
          </div>; })}
        </div>
        <div className="prep-col">
          <div className="prep-col-h">{applied > 0 ? "applied across every row" : "this row, executed"}</div>
          {viz(applied, curRow, st)}
          <div className="note" style={{ marginTop: 8 }}>{applied >= total ? "applied to every row" : (applied > 0 ? `${applyNote} — ${applied} / ${total} ${rowTag}s` : "step the formula to run this one row, then ▶ apply to all rows")}</div>
        </div>
      </div>
      {modelStageControls(maxStage, total)}
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
    if (!ds || !result || !modelRef.current) return <div className="note">Train the model first (▶ Train model, above) — then this walks through how it was fitted and how well it scores.</div>;
    let b: ReturnType<typeof buildMatrix> | null = null;
    try { b = buildMatrix(ds, features, target, task, steps); } catch { b = null; }
    if (!b || !b.X.length) return <div className="note">Train the model first to see how it learns.</div>;
    const isReg = task === "regression";
    const kind = modelKind;
    const m = modelRef.current;
    const names = b.featureNames, classes = b.classes, K = classes?.length ?? 0;
    const label = MODEL_INFO[algo]?.label ?? algo;
    // geometry (own copy so it's independent of the model card)
    const PX0 = 30, PX1 = 344, PYb = 288, PYt = 22;
    const c01 = (t: number) => (Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0);
    const sx = (t: number) => PX0 + (PX1 - PX0) * c01(t);
    const sy = (t: number) => PYb + (PYt - PYb) * c01(t);
    const fv = (x: number | undefined, dp = 2) => (typeof x === "number" && Number.isFinite(x) ? x.toFixed(dp) : "—");
    const svg = (kids: React.ReactNode) => <svg viewBox="0 0 360 310" style={{ width: "100%", borderRadius: 8, background: "var(--panel)", border: "1px solid var(--border)" }}><rect x={PX0} y={PYt} width={PX1 - PX0} height={PYb - PYt} fill="none" stroke="var(--border)" />{kids}</svg>;
    const idxF = names.indexOf(dbF1) >= 0 ? names.indexOf(dbF1) : 0;
    const idxG = names.indexOf(dbF2) >= 0 && names.indexOf(dbF2) !== idxF ? names.indexOf(dbF2) : Math.min(names.length - 1, idxF + 1);
    const col = (j: number) => b!.X.map((r) => (Number.isFinite(r[j]) ? r[j] : NaN));
    const nrm = (arr: number[]) => { const f = arr.filter((v) => Number.isFinite(v)); const mn = f.length ? Math.min(...f) : 0, mx = f.length ? Math.max(...f) : 1, d = (mx - mn) || 1; return { mn, mx, z: arr.map((v) => (Number.isFinite(v) ? (v - mn) / d : 0)) }; };
    const PAL = PAL_ML;

    // ── test split + predictions (for the check phase) ──
    const { Xte, yte } = split(b.X, b.y, testSize);
    const preds = predict(m, Xte);
    const nTe = Xte.length;

    // ── TRAINING spec ──
    let trainIntro = "", trainTotal = 0, trainUnit = "step", trainNote = "";
    let trainCols: string[] = [], trainRows: (string | number)[][] = [];
    let trainMath: (t: number) => [string, string][] = () => [];
    let trainViz: (t: number) => React.ReactNode = () => svg(null);
    let gdSnaps: { ep: number; loss: number; gnorm: number; w: number[] }[] = [];

    if (kind === "gd") {
      const gt = gdTrace(cfgNow(), b.X, b.y, K);
      const snaps = gt?.snaps ?? []; gdSnaps = snaps;
      trainTotal = snaps.length; trainUnit = "epoch";
      trainIntro = "Training loops over epochs: each one computes the gradient of the loss and steps every weight downhill. Watch the loss and ‖∇‖ shrink until they flatten — that's convergence.";
      trainCols = ["epoch", "loss", "‖∇‖", "b", "w₁", "w₂"];
      trainRows = snaps.map((s) => [s.ep, fv(s.loss, 4), fv(s.gnorm, 4), fv(s.w[0], 3), fv(s.w[1], 3), fv(s.w[2], 3)]);
      trainMath = (t) => { const s = snaps[Math.min(Math.max(t - 1, 0), snaps.length - 1)]; if (!s) return []; return [
        ["update rule", "\\mathbf w \\leftarrow \\mathbf w - \\eta\\,\\nabla_{\\mathbf w}\\mathcal L"],
        [`epoch ${s.ep} (η=${gt!.lr})`, `\\mathcal L = ${fv(s.loss, 4)},\\ \\lVert\\nabla\\rVert = ${fv(s.gnorm, 4)}`],
        ["weights now", `b=${fv(s.w[0], 3)},\\ w_1=${fv(s.w[1], 3)},\\ w_2=${fv(s.w[2], 3)}`],
      ]; };
      const fA = nrm(col(idxF)), fB = nrm(col(idxG)); const yn = nrm(b.y);
      trainNote = isReg ? "loss falls as the line tilts to fit; " : "loss falls as the boundary rotates to separate the classes; ";
      trainViz = (t) => {
        const s = snaps[Math.min(Math.max(t - 1, 0), snaps.length - 1)]; const w = s?.w ?? [];
        const lmax = Math.max(...snaps.map((q) => q.loss), 1e-6), lmin = Math.min(...snaps.map((q) => q.loss));
        const lx = (e: number) => PX0 + (PX1 - PX0) * (e / ((trainTotal - 1) || 1));
        const lyv = (v: number) => 40 + (120 - 40) * (1 - (v - lmin) / ((lmax - lmin) || 1));
        const shown = Math.max(1, Math.min(t, trainTotal));
        // bottom band scatter map (y 160..292)
        const bY = (v01: number) => 292 - v01 * 132;
        const kids: React.ReactNode[] = [
          <text key="lh" x={PX0} y={32} fill="var(--faint)" fontSize={10} fontFamily="monospace">loss ↓ over epochs</text>,
          <polyline key="lc" points={snaps.slice(0, shown).map((q, e) => `${lx(e)},${lyv(q.loss)}`).join(" ")} fill="none" stroke="#3ecf7f" strokeWidth={2} />,
          <line key="sep" x1={PX0} y1={140} x2={PX1} y2={140} stroke="var(--border)" />,
          <text key="bh" x={PX0} y={156} fill="var(--faint)" fontSize={10} fontFamily="monospace">{isReg ? "fitted line vs data" : "decision boundary"}</text>,
        ];
        b!.X.forEach((_, i) => kids.push(<circle key={"p" + i} cx={sx(fA.z[i])} cy={bY(isReg ? yn.z[i] : fB.z[i])} r={3.5} fill={isReg ? "#5b7cff" : PAL[b!.y[i] % PAL.length]} opacity={0.7} />));
        if (w.length) {
          if (isReg) { const line: string[] = []; for (let q = 0; q <= 20; q++) { const xf = fA.mn + (fA.mx - fA.mn) * (q / 20); const yh = w[0] + (w[idxF + 1] || 0) * xf; line.push(`${sx((xf - fA.mn) / ((fA.mx - fA.mn) || 1))},${bY((yh - (nrm(b!.y).mn)) / (((nrm(b!.y).mx) - (nrm(b!.y).mn)) || 1))}`); } kids.push(<polyline key="fit" points={line.join(" ")} fill="none" stroke="#f59e0b" strokeWidth={2.5} />); }
          else { const wF = w[idxF + 1] || 1e-6, wG = w[idxG + 1] || 1e-6, w0 = w[0] || 0; const line: string[] = []; for (let q = 0; q <= 20; q++) { const xf = fA.mn + (fA.mx - fA.mn) * (q / 20); const xg = -(w0 + wF * xf) / wG; line.push(`${sx((xf - fA.mn) / ((fA.mx - fA.mn) || 1))},${bY((xg - fB.mn) / ((fB.mx - fB.mn) || 1))}`); } kids.push(<polyline key="bd" points={line.join(" ")} fill="none" stroke="#a855f7" strokeWidth={2} />); }
        }
        return svg(kids);
      };
    } else if (kind === "tree" || kind === "forest") {
      const roots = kind === "forest" ? ((m as unknown as { trees: unknown[] }).trees || []) : [(m as unknown as { root: unknown }).root];
      type TN = { leaf: boolean; feat?: number; thr?: number; n: number; left?: TN; right?: TN; value?: number };
      const laidOut: { node: TN; x: number; y: number; px?: number; py?: number; ri: number }[] = [];
      const lay = (node: TN | undefined, depth: number, x0: number, x1: number, px?: number, py?: number) => { if (!node || depth > 3) return; const x = (x0 + x1) / 2, y = 40 + depth * 58; laidOut.push({ node, x, y, px, py, ri: 0 }); if (!node.leaf) { lay(node.left, depth + 1, x0, x, x, y); lay(node.right, depth + 1, x, x1, x, y); } };
      lay(roots[0] as TN, 0, PX0 + 8, PX1 - 8);
      laidOut.sort((a, c) => a.y - c.y || a.x - c.x).forEach((nd, i) => { nd.ri = i; });
      trainTotal = kind === "forest" ? Math.min(roots.length, 8) : laidOut.length;
      trainUnit = kind === "forest" ? "tree" : "node";
      trainIntro = kind === "forest"
        ? "Training fits many trees — each on its own bootstrap resample with a random feature subset. More trees cut variance; predictions are averaged/voted."
        : "Training grows the tree greedily: at each node it keeps the split with the highest impurity gain, then recurses on each side until pure or max depth.";
      trainCols = kind === "forest" ? ["tree", "depth", "nodes"] : ["node", "feature", "≤ thr", "samples"];
      const depthOf = (nd: TN): number => (nd.leaf ? 1 : 1 + Math.max(depthOf(nd.left as TN), depthOf(nd.right as TN)));
      const countOf = (nd: TN): number => (nd.leaf ? 1 : 1 + countOf(nd.left as TN) + countOf(nd.right as TN));
      trainRows = kind === "forest"
        ? (roots.slice(0, trainTotal) as TN[]).map((r, i) => [`T${i + 1}`, depthOf(r), countOf(r)])
        : laidOut.map((l, i) => [i === 0 ? "root" : `n${i}`, l.node.leaf ? "leaf" : (names[l.node.feat ?? 0] ?? `#${l.node.feat}`).slice(0, 10), l.node.leaf ? "—" : fv(l.node.thr), l.node.n]);
      trainMath = (t) => kind === "forest"
        ? [["for each tree", "\\text{bootstrap } n \\text{ rows} + \\text{random features}"], [`tree ${Math.min(t, trainTotal) || 1}`, `\\text{fit } T_{${Math.min(t, trainTotal) || 1}} \\text{ independently}`], ["combine", isReg ? "\\hat y = \\tfrac1m\\sum_i T_i" : "\\hat y = \\text{mode}\\{T_i\\}"]]
        : (() => { const l = laidOut[Math.min(Math.max(t - 1, 0), laidOut.length - 1)]; if (!l) return []; return [["at each node", "\\text{gain} = G - \\tfrac{n_L}{n}G_L - \\tfrac{n_R}{n}G_R"], [l.node.leaf ? "leaf" : "split", l.node.leaf ? "\\text{pure / max depth reached}" : `\\text{${(names[l.node.feat ?? 0] ?? "x").replace(/[^a-zA-Z0-9 ]/g, " ")}} \\le ${fv(l.node.thr)}`], ["samples here", `n = ${l.node.n}`]]; })();
      trainNote = kind === "forest" ? "trees fit one at a time; each sees a different resample" : "the tree grows node by node, most-informative split first";
      trainViz = (t) => {
        if (kind === "forest") { const tx = [80, 145, 210, 275]; const kids: React.ReactNode[] = []; for (let i = 0; i < Math.min(4, trainTotal); i++) { const on = t > i; const x = tx[i]; kids.push(<circle key={"n" + i} cx={x} cy={110} r={on ? 7 : 5} fill="#a855f7" opacity={on ? 1 : 0.25} />, <line key={"la" + i} x1={x} y1={110} x2={x - 16} y2={158} stroke="var(--faint)" opacity={on ? 1 : 0.2} />, <line key={"lb" + i} x1={x} y1={110} x2={x + 16} y2={158} stroke="var(--faint)" opacity={on ? 1 : 0.2} />, <circle key={"cl" + i} cx={x - 16} cy={170} r={5} fill={PAL[0]} opacity={on ? 0.9 : 0.2} />, <circle key={"cr" + i} cx={x + 16} cy={170} r={5} fill={PAL[1]} opacity={on ? 0.9 : 0.2} />, <text key={"t" + i} x={x - 8} y={98} fill="var(--faint)" fontSize={11} fontFamily="monospace">T{i + 1}</text>); } kids.push(<text key="cap" x={100} y={240} fill="var(--faint)" fontSize={11} fontFamily="monospace">{Math.min(t, trainTotal)} / {trainTotal} trees fit</text>); return svg(kids); }
        const kids: React.ReactNode[] = [];
        laidOut.forEach((l, i) => { const on = t > l.ri; if (l.px != null && on) kids.push(<line key={"e" + i} x1={l.px} y1={l.py} x2={l.x} y2={l.y} stroke="var(--border-strong)" />); });
        laidOut.forEach((l, i) => { const on = t > l.ri; const cur = t - 1 === l.ri; kids.push(<circle key={"n" + i} cx={l.x} cy={l.y} r={cur ? 9 : 7} fill={l.node.leaf ? "#3ecf7f" : "#a855f7"} opacity={on ? 1 : 0.18} stroke={cur ? "var(--text)" : undefined} strokeWidth={cur ? 1.5 : undefined} />); });
        return svg(kids);
      };
    } else if (kind === "gnb") {
      const gm = m as unknown as { means: number[][]; vars: number[][]; priors: number[] };
      const nf = Math.min(names.length, 3);
      const cells: { k: number; j: number }[] = [];
      for (let k = 0; k < K; k++) for (let j = 0; j < nf; j++) cells.push({ k, j });
      trainTotal = cells.length; trainUnit = "stat";
      trainIntro = "“Training” Gaussian NB is one pass of bookkeeping — no iteration. For each class and feature it measures the mean and variance, and counts rows for the prior.";
      trainCols = ["class", "feature", "μ", "σ²", "prior"];
      trainRows = cells.map(({ k, j }) => [String(classes?.[k] ?? k), (names[j] ?? `#${j}`).slice(0, 10), fv(gm.means[k]?.[j]), fv(gm.vars[k]?.[j]), fv(gm.priors[k])]);
      trainMath = (t) => { const c = cells[Math.min(Math.max(t - 1, 0), cells.length - 1)] || cells[0]; if (!c) return []; return [["accumulate", "\\mu = \\tfrac1n\\sum x,\\quad \\sigma^2 = \\tfrac1n\\sum (x-\\mu)^2"], [`class ${classes?.[c.k] ?? c.k}, ${names[c.j] ?? c.j}`, `\\mu = ${fv(gm.means[c.k]?.[c.j])},\\ \\sigma^2 = ${fv(gm.vars[c.k]?.[c.j])}`], ["prior", `P(y) = ${fv(gm.priors[c.k])}`]]; };
      trainNote = "each class·feature statistic measured in one scan of the rows";
      trainViz = (t) => { const rng = nrm(col(idxF)); const bell = (x: number, mu: number, va: number) => Math.exp(-((x - mu) ** 2) / (2 * (va || 1))); const kids: React.ReactNode[] = []; const seen = Math.min(t, trainTotal); for (let k = 0; k < K; k++) { const revealed = cells.slice(0, seen).some((c) => c.k === k && c.j === idxF); const mu = gm.means[k]?.[idxF] ?? 0, va = gm.vars[k]?.[idxF] ?? 1; const muN = (mu - rng.mn) / ((rng.mx - rng.mn) || 1); const pts: string[] = []; for (let q = 0; q <= 60; q++) { const xn = q / 60; const x = rng.mn + (rng.mx - rng.mn) * xn; pts.push(`${sx(xn)},${sy(bell(x, mu, va) * 0.9)}`); } kids.push(<polyline key={"b" + k} points={pts.join(" ")} fill="none" stroke={PAL[k % PAL.length]} strokeWidth={2} opacity={revealed ? 1 : 0.15} />); if (revealed) kids.push(<line key={"mu" + k} x1={sx(muN)} y1={sy(bell(mu, mu, va) * 0.9)} x2={sx(muN)} y2={PYb} stroke={PAL[k % PAL.length]} strokeDasharray="3 3" opacity={0.5} />); } return svg(kids); };
    } else { // knn
      const store = Math.min(b.X.length, 40);
      trainTotal = store; trainUnit = "row";
      trainIntro = "KNN has no training loop — fitting just stores every row verbatim. All the real work is deferred to prediction time (measure distance, keep k nearest, vote/average).";
      trainCols = ["row", (names[idxF] ?? "f1").slice(0, 8), (names[idxG] ?? "f2").slice(0, 8), "label"];
      trainRows = [...Array(store).keys()].map((i) => [i, fv(b!.X[i]?.[idxF]), fv(b!.X[i]?.[idxG]), String(isReg ? fv(b!.y[i]) : (classes?.[b!.y[i]] ?? b!.y[i]))]);
      trainMath = (t) => [["fit", "\\text{store all rows as-is}"], ["stored", `${Math.min(t, trainTotal)} \\text{ rows kept in memory}`], ["parameters", "\\text{none learned}"]];
      trainNote = "rows are simply memorised — no parameters are fit";
      const fA = nrm(col(idxF)), fB = nrm(col(idxG));
      trainViz = (t) => svg([...Array(Math.min(t, trainTotal)).keys()].map((i) => <circle key={"p" + i} cx={sx(fA.z[i])} cy={sy(fB.z[i])} r={4} fill={isReg ? "#5b7cff" : PAL[b!.y[i] % PAL.length]} opacity={0.8} />));
    }

    // ── CHECK spec ──
    const checkTotal = nTe;
    const finalM = result.metrics;
    let verdict: [string, string, string] = ["", "", "#8a8a86"];
    let checkMath: (t: number) => [string, string][] = () => [];
    let checkTiles: (t: number) => [string, string][] = () => [];
    let checkNote = "", checkH = "";

    if (!isReg) {
      const correct = (upto: number) => { let c = 0; for (let i = 0; i < upto; i++) if (preds[i] === yte[i]) c++; return c; };
      const cm = finalM.task === "classification" ? finalM : null;
      const accF = cm?.accuracy ?? 0;
      verdict = accF >= 0.9 ? ["Excellent — strong model", "well above baseline on held-out data", "#3ecf7f"] : accF >= 0.8 ? ["Good model", "solid accuracy on the test split", "#3ecf7f"] : accF >= 0.65 ? ["Fair — could improve", "try more features, scaling, or tuning", "#f59e0b"] : ["Weak — needs improvement", "near baseline; rethink features or model", "#ef4444"];
      checkH = "confusion matrix (test set)";
      checkMath = (t) => { const c = correct(t); return [["accuracy", "\\text{accuracy} = \\dfrac{\\text{correct}}{\\text{total}}"], ["substitute", `= \\dfrac{${c}}{${t}}`], ["result", `\\text{accuracy} = ${t ? fv(c / t, 3) : "—"}`]]; };
      checkTiles = (t) => cm ? [["accuracy", t ? fv(correct(t) / t) : "—"], ["precision", fv(cm.precision)], ["recall", fv(cm.recall)], ["F1", fv(cm.f1)]] : [];
      checkNote = "each test row is predicted and dropped into the matrix";
    } else {
      const ym = yte.reduce((a, v) => a + v, 0) / (yte.length || 1);
      const ssr = (upto: number) => { let s = 0; for (let i = 0; i < upto; i++) s += (preds[i] - yte[i]) ** 2; return s; };
      const sst = (upto: number) => { let s = 0; for (let i = 0; i < upto; i++) s += (yte[i] - ym) ** 2; return s; };
      const rm = finalM.task === "regression" ? finalM : null;
      const r2F = rm?.r2 ?? 0;
      verdict = r2F >= 0.8 ? ["Excellent — strong fit", "explains most of the variance", "#3ecf7f"] : r2F >= 0.6 ? ["Good fit", "captures the trend well", "#3ecf7f"] : r2F >= 0.3 ? ["Fair — could improve", "sizeable residuals remain", "#f59e0b"] : ["Weak — needs improvement", "barely beats predicting the mean", "#ef4444"];
      checkH = "predicted vs actual (test set)";
      checkMath = (t) => { const ss = ssr(t), mse = t ? ss / t : NaN; return [["MSE", "\\text{MSE} = \\tfrac1n\\sum(\\hat y - y)^2"], ["substitute", `= \\tfrac1{${t}}\\,(${fv(ss)})`], ["result", `\\text{MSE} = ${fv(mse, 3)},\\ \\text{RMSE} = ${fv(Math.sqrt(mse), 3)}`]]; };
      checkTiles = (t) => { const ss = ssr(t), mse = t ? ss / t : NaN, st = sst(t); return rm ? [["MSE", fv(mse, 3)], ["RMSE", fv(Math.sqrt(mse), 3)], ["R²", t ? fv(1 - ss / (st || 1)) : "—"]] : []; };
      checkNote = "each test row's prediction is placed against the ŷ = y line";
    }

    // ── Plotly graphs (rich, wide, with legends) ──
    const th = plotlyTheme();
    const discreteScale = (kk: number): [number, string][] => { const cs: [number, string][] = []; for (let i = 0; i < kk; i++) { const c = PAL[i % PAL.length]; cs.push([kk <= 1 ? 0 : i / kk, c]); cs.push([kk <= 1 ? 1 : (i + 1) / kk, c]); } return cs; };
    const legendLayout = { showlegend: true, legend: { orientation: "h" as const, y: -0.18, font: { size: 11 } } };
    function trainGraph(t: number): React.ReactNode {
      const d = trainVizData; const H = 400;
      if (kind === "gd" && d?.kind === "gd" && d.anim) {
        const anim = d.anim; const shown = Math.max(1, Math.min(t, trainTotal));
        const fi = Math.round((Math.min(Math.max(t - 1, 0), trainTotal - 1) / ((trainTotal - 1) || 1)) * (anim.frames.length - 1));
        const fr = anim.frames[Math.max(0, Math.min(fi, anim.frames.length - 1))] || anim.frames[anim.frames.length - 1];
        const cur = gdSnaps[Math.min(shown - 1, gdSnaps.length - 1)];
        const loss = <Plot data={[
          { type: "scatter", mode: "lines", name: "training loss", x: gdSnaps.slice(0, shown).map((s) => s.ep), y: gdSnaps.slice(0, shown).map((s) => s.loss), line: { color: "#3ecf7f", width: 2.5 } },
          { type: "scatter", mode: "markers", name: "current epoch", x: cur ? [cur.ep] : [], y: cur ? [cur.loss] : [], marker: { size: 10, color: "#3ecf7f", line: { width: 1, color: th.paper } } },
        ]} layout={{ ...chartLayout(th, "loss ↓ over epochs", "epoch", "loss"), ...legendLayout, height: H }} style={{ height: H, width: "100%" }} />;
        const bound = !anim.reg
          ? <Plot data={[
              { type: "heatmap", x: anim.xs, y: anim.ys, z: fr?.z, showscale: false, colorscale: discreteScale(anim.classes.length), zmin: -0.5, zmax: anim.classes.length - 0.5, opacity: 0.32, hoverinfo: "skip", name: "regions" },
              ...anim.classes.map((cl, ci) => ({ type: "scatter" as const, mode: "markers" as const, name: `class ${cl}`, x: anim.points.filter((p) => p.c === ci).map((p) => p.x), y: anim.points.filter((p) => p.c === ci).map((p) => p.y), marker: { size: 7, color: PAL[ci % PAL.length], line: { width: 1, color: th.paper } } })),
            ]} layout={{ ...chartLayout(th, `epoch ${fr?.ep ?? 0} · loss ${fv(fr?.loss, 3)} — shaded regions = what the model predicts there`, d.f1, d.f2), ...legendLayout, height: H }} style={{ height: H, width: "100%" }} />
          : <Plot data={[
              { type: "scatter", mode: "markers", name: "data points", x: anim.points.map((p) => p.x), y: anim.points.map((p) => p.y), marker: { size: 7, color: "#5b7cff", opacity: 0.6 } },
              { type: "scatter", mode: "lines", name: "fitted line", x: anim.xs, y: fr?.line, line: { color: "#f59e0b", width: 3 } },
            ]} layout={{ ...chartLayout(th, `epoch ${fr?.ep ?? 0} · MSE ${fv(fr?.loss, 2)} — the line tilts to minimise squared error`, d.f1, target), ...legendLayout, height: H }} style={{ height: H, width: "100%" }} />;
        return <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12, alignItems: "start" }}>{loss}{bound}</div>;
      }
      if (d?.kind === "surface" && d.surf) {
        const s = d.surf; const kk = s.classes.length; const ratio = trainTotal ? Math.min(t, trainTotal) / trainTotal : 1; const shown = Math.max(1, Math.round(ratio * s.points.length));
        return <Plot data={[
          { type: "heatmap", x: s.xs, y: s.ys, z: s.z, showscale: false, colorscale: discreteScale(kk), zmin: -0.5, zmax: kk - 0.5, opacity: 0.32, hoverinfo: "skip", name: "regions" },
          ...s.classes.map((cl, ci) => ({ type: "scatter" as const, mode: "markers" as const, name: `class ${cl}`, x: s.points.slice(0, shown).filter((p) => p.c === ci).map((p) => p.x), y: s.points.slice(0, shown).filter((p) => p.c === ci).map((p) => p.y), marker: { size: 7, color: PAL[ci % PAL.length], line: { width: 1, color: th.paper } } })),
        ]} layout={{ ...chartLayout(th, `learned decision regions · train acc ${(s.acc * 100).toFixed(0)}% — colour = predicted class`, d.f1, d.f2), ...legendLayout, height: H + 20 }} style={{ height: H + 20, width: "100%" }} />;
      }
      if (d?.kind === "regcurve") {
        const ratio = trainTotal ? Math.min(t, trainTotal) / trainTotal : 1; const shown = Math.max(1, Math.round(ratio * d.pts.length));
        return <Plot data={[
          { type: "scatter", mode: "markers", name: "actual y", x: d.pts.map((p) => p.x), y: d.pts.map((p) => p.y), marker: { size: 6, color: "#5b7cff", opacity: 0.5 } },
          { type: "scatter", mode: "lines", name: "model prediction ŷ", x: d.pts.slice(0, shown).map((p) => p.x), y: d.pts.slice(0, shown).map((p) => p.yh), line: { color: "#f59e0b", width: 3 } },
        ]} layout={{ ...chartLayout(th, `${label} prediction vs ${d.f1}`, d.f1, target), ...legendLayout, height: H + 20 }} style={{ height: H + 20, width: "100%" }} />;
      }
      return trainViz(t); // SVG fallback (e.g. <2 numeric features)
    }
    function checkGraph(t: number): React.ReactNode {
      const H = 400;
      if (!isReg) {
        const cls = (classes ?? []).map(String); const M = Array.from({ length: K }, () => new Array(K).fill(0)); for (let i = 0; i < Math.min(t, nTe); i++) { const a = yte[i], p = preds[i]; if (a < K && p < K) M[a][p]++; }
        const ann = [] as Record<string, unknown>[]; for (let a = 0; a < K; a++) for (let p = 0; p < K; p++) ann.push({ x: cls[p], y: cls[a], text: String(M[a][p]), showarrow: false, font: { color: a === p ? "#eafff2" : "#ffe9e9", size: 17 } });
        return <Plot data={[{ type: "heatmap", x: cls, y: cls, z: M, xgap: 4, ygap: 4, colorscale: [[0, th.plot], [0.5, "#f59e0b"], [1, "#3ecf7f"]], showscale: true, colorbar: { title: { text: "count" } }, hoverinfo: "skip" }]} layout={{ ...chartLayout(th, "confusion matrix — the diagonal is correct, off-diagonal are mistakes", "predicted class →", "actual class ↓"), height: H, annotations: ann as never }} style={{ height: H, width: "100%" }} />;
      }
      const xs: number[] = [], ys: number[] = []; for (let i = 0; i < Math.min(t, nTe); i++) { xs.push(yte[i]); ys.push(preds[i]); }
      const allV = [...preds, ...yte].filter(Number.isFinite); const lo = allV.length ? Math.min(...allV) : 0, hi = allV.length ? Math.max(...allV) : 1;
      return <Plot data={[
        { type: "scatter", mode: "lines", name: "ŷ = y (perfect)", x: [lo, hi], y: [lo, hi], line: { color: th.muted, dash: "dash", width: 1.5 } },
        { type: "scatter", mode: "markers", name: "test rows", x: xs, y: ys, marker: { size: 8, color: "#5b7cff", opacity: 0.8, line: { width: 1, color: th.paper } } },
      ]} layout={{ ...chartLayout(th, "predicted vs actual — the closer to the dashed line, the better", "actual y", "predicted ŷ"), ...legendLayout, height: H }} style={{ height: H, width: "100%" }} />;
    }

    // ── render ──
    const inTrain = trainPhase === "train";
    const tShown = Math.min(trainT, trainTotal);
    const mathLines = inTrain ? trainMath(trainT) : checkMath(checkT);
    const tiles = inTrain ? [] : checkTiles(checkT);
    const checkDone = !inTrain && checkT >= checkTotal;
    const canCheck = trainT >= trainTotal && trainTotal > 0;
    const pill = (n: number, txt: string, on: boolean) => <span key={n} className="fx-tag" style={{ width: "auto", padding: "3px 10px", borderRadius: 20, border: "1px solid var(--border)", background: on ? "var(--accent)" : "transparent", color: on ? "#fff" : "var(--faint)" }}>{txt}</span>;
    return mCard(`${label} — training & evaluation`, <>
      <div className="row" style={{ gap: 6, marginBottom: 10 }}>{pill(1, "1 · train (fit)", inTrain)}{pill(2, "2 · check", !inTrain)}{pill(3, "3 · verdict", checkDone)}</div>
      <div className="note" style={{ marginBottom: 10, lineHeight: 1.6 }}>{inTrain ? trainIntro : (isReg ? "Score the fitted model on the held-out test rows: sum the squared errors into MSE/RMSE, and R² vs just predicting the mean." : "Score the fitted model on the held-out test rows: each prediction is checked against the truth and tallied into accuracy and the confusion matrix.")}</div>
      <div className="prep-col-h">{inTrain ? "how the model is fitted — with your numbers" : "the metric, computed on the test set"}</div>
      <div className="row" style={{ gap: 10, flexWrap: "wrap", marginBottom: 6 }}>{mathLines.map(([lab, tex], i) => <div key={i} style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, padding: "7px 11px", flex: "1 1 210px" }}><div className="note" style={{ marginBottom: 2 }}>{lab}</div><Katex block tex={tex} /></div>)}</div>
      {tiles.length ? <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 4 }}>{tiles.map(([kk, vv], i) => <div key={i} style={{ background: "var(--panel)", borderRadius: 8, padding: "6px 14px", minWidth: 74 }}><div className="note">{kk}</div><div style={{ fontSize: 22, fontWeight: 600 }}>{vv}</div></div>)}</div> : null}
      {checkDone ? <div className="row" style={{ gap: 10, alignItems: "center", marginTop: 12, padding: "12px 14px", borderRadius: 10, background: `${verdict[2]}22`, color: verdict[2] }}><span style={{ fontSize: 22 }}>🏆</span><span><b style={{ fontSize: 15 }}>{verdict[0]}</b><span className="note" style={{ display: "block", color: verdict[2] }}>{verdict[1]}</span></span></div> : null}
      <div className="prep-col-h" style={{ marginTop: 14 }}>{inTrain ? (kind === "gd" ? "loss curve + " + (isReg ? "fitted line" : "decision boundary") : kind === "gnb" || kind === "tree" || kind === "forest" || kind === "knn" ? (isReg ? "model fit on your data" : "decision regions on your data") : "on your data") : checkH}</div>
      {inTrain ? trainGraph(trainT) : checkGraph(checkT)}
      <div className="note" style={{ marginTop: 6 }}>{inTrain ? (tShown >= trainTotal ? "training complete — " + trainNote : `${trainNote} — ${trainUnit} ${tShown} / ${trainTotal}`) : (checkT >= checkTotal ? `all ${checkTotal} test rows scored` : `${checkNote} — ${Math.min(checkT, checkTotal)} / ${checkTotal}`)}</div>
      {trainTotal > 0 && <>
        <div className="prep-col-h" style={{ marginTop: 14 }}>{inTrain ? `the full training log — fills as it ${trainUnit === "epoch" ? "trains" : "builds"}` : "training log (complete)"}</div>
        <div style={{ overflowX: "auto", maxHeight: 190, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 8 }}>
          <table className="dtable"><tbody>
            <tr>{trainCols.map((cc, i) => <th key={i}>{cc}</th>)}</tr>
            {trainRows.slice(0, inTrain ? tShown : trainTotal).map((r, i) => <tr key={i} style={i === tShown - 1 && inTrain ? { background: "var(--accent)", color: "#fff" } : undefined}>{r.map((cc, j) => <td key={j} className="mono">{cc}</td>)}</tr>)}
          </tbody></table>
        </div>
      </>}
      <div className="prep-ctl" style={{ flexWrap: "wrap" }}>
        <button className="btn sm" onClick={() => trainRun("train", trainTotal)}>{trainRunning && inTrain ? "⏸ pause" : (trainT >= trainTotal && trainTotal > 0 ? "↻ re-train" : "▶ train")}</button>
        <button className="btn sm" disabled={!canCheck} onClick={() => trainRun("check", checkTotal)}>{trainRunning && !inTrain ? "⏸ pause" : `✓ check ${isReg ? "error" : "accuracy"}`}</button>
        <button className="btn ghost sm" onClick={trainReset}>↺ reset</button>
        <span className="note mono" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>speed
          <select className="mini-sel" value={trainSpeed} onChange={(e) => setTrainSpeedFn(+e.target.value)}>
            <option value={380}>0.5×</option><option value={200}>1×</option><option value={90}>2×</option><option value={35}>4×</option>
          </select>
        </span>
        <span className="note mono" style={{ marginLeft: "auto" }}>{inTrain ? `${trainUnit} ${tShown} / ${trainTotal}` : `row ${Math.min(checkT, checkTotal)} / ${checkTotal}`}</span>
      </div>
    </>);
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
