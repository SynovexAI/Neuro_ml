// Real in-browser ML: CSV parsing, EDA stats, preprocessing, training, metrics.
// Models are implemented from scratch (logistic/linear regression via gradient
// descent, KNN) so training genuinely runs on the data — no server, no mocks.

export type ColType = "num" | "cat";
export interface Column { name: string; type: ColType; values: (number | string | null)[]; }
export interface Dataset { columns: Column[]; nrows: number; }
export type Task = "classification" | "regression";
export interface PrepStep { op: string; cols: string[]; method: string; }

// ── parsing ──
export function parseCSV(text: string): Dataset {
  const clean = text.replace(/\r\n?/g, "\n").trim();
  const delim = (clean.split("\n")[0].match(/;/g)?.length || 0) > (clean.split("\n")[0].match(/,/g)?.length || 0) ? ";" : (clean.includes("\t") && !clean.split("\n")[0].includes(",") ? "\t" : ",");
  const rows = clean.split("\n").map((line) => splitLine(line, delim));
  const header = rows[0];
  const body = rows.slice(1).filter((r) => r.length > 1 || (r.length === 1 && r[0] !== ""));
  const cols: Column[] = header.map((name, ci) => {
    const raw = body.map((r) => (r[ci] ?? "").trim());
    const nonEmpty = raw.filter((v) => v !== "");
    const numeric = nonEmpty.length > 0 && nonEmpty.every((v) => v !== "" && !isNaN(Number(v)));
    const type: ColType = numeric ? "num" : "cat";
    const values = raw.map((v) => (v === "" ? null : (type === "num" ? Number(v) : v)));
    return { name: name.trim() || `col${ci}`, type, values };
  });
  return { columns: cols, nrows: body.length };
}
function splitLine(line: string, d: string): string[] {
  const out: string[] = []; let cur = ""; let q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
    else if (c === d && !q) { out.push(cur); cur = ""; }
    else cur += c;
  }
  out.push(cur);
  return out;
}

// ── stats ──
export interface NumStats { type: "num"; count: number; missing: number; mean: number; std: number; min: number; max: number; }
export interface CatStats { type: "cat"; count: number; missing: number; unique: number; top: [string, number][]; }
export function colStats(col: Column): NumStats | CatStats {
  const missing = col.values.filter((v) => v == null).length;
  if (col.type === "num") {
    const v = col.values.filter((x) => x != null) as number[];
    const mean = v.reduce((a, b) => a + b, 0) / (v.length || 1);
    const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length || 1));
    return { type: "num", count: v.length, missing, mean, std, min: Math.min(...v), max: Math.max(...v) };
  }
  const v = col.values.filter((x) => x != null) as string[];
  const counts = new Map<string, number>();
  v.forEach((x) => counts.set(String(x), (counts.get(String(x)) || 0) + 1));
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return { type: "cat", count: v.length, missing, unique: counts.size, top };
}
export function histogram(v: number[], bins = 12): { edges: number[]; counts: number[] } {
  if (!v.length) return { edges: [], counts: [] };
  const min = Math.min(...v), max = Math.max(...v), span = (max - min) || 1;
  const counts = new Array(bins).fill(0);
  v.forEach((x) => { let b = Math.floor(((x - min) / span) * bins); if (b >= bins) b = bins - 1; if (b < 0) b = 0; counts[b]++; });
  const edges = Array.from({ length: bins + 1 }, (_, i) => min + (span * i) / bins);
  return { edges, counts };
}
export function pearson(a: number[], b: number[]): number {
  const n = a.length; const ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / (Math.sqrt(da * db) || 1);
}

// ── preprocessing → numeric matrix ──
function modeNum(a: number[]): number { const m = new Map<number, number>(); a.forEach((x) => m.set(x, (m.get(x) || 0) + 1)); return [...m.entries()].sort((x, y) => y[1] - x[1])[0][0]; }
function modeStr(a: string[]): string { const m = new Map<string, number>(); a.forEach((x) => m.set(x, (m.get(x) || 0) + 1)); return [...m.entries()].sort((x, y) => y[1] - x[1])[0]?.[0] ?? ""; }

// ── shared column-level transforms — used by BOTH the animation preview and the
//    real training matrix, so what you watch is exactly what the model trains on. ──
function imputeFill(present: number[], method: string): number {
  if (!present.length) return 0;
  if (method === "Median") { const s = [...present].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }
  if (method === "Most frequent") return modeNum(present);
  if (method === "Constant") return 0;
  if (method === "Min") return Math.min(...present);
  if (method === "Max") return Math.max(...present);
  return present.reduce((a, b) => a + b, 0) / present.length; // Mean
}
function interpolateCol(vals: (number | null)[]): number[] {
  const out = vals.map((v) => (v == null ? null : Number(v))) as (number | null)[];
  const n = out.length;
  const first = out.findIndex((v) => v != null);
  if (first < 0) return out.map(() => 0);
  for (let k = 0; k < first; k++) out[k] = out[first];
  for (let i = first + 1; i < n; i++) {
    if (out[i] != null) continue;
    let j = i + 1; while (j < n && out[j] == null) j++;
    if (j >= n) { for (let k = i; k < n; k++) out[k] = out[i - 1]; break; }
    const a = out[i - 1] as number, b = out[j] as number, span = j - (i - 1);
    for (let k = i; k < j; k++) out[k] = a + (b - a) * ((k - (i - 1)) / span);
    i = j - 1;
  }
  return out.map((v) => Number(v));
}
export function imputeNumCol(vals: (number | null)[], method: string): number[] {
  const present = vals.filter((v): v is number => v != null);
  if (!present.length) return vals.map(() => 0);
  if (method === "Forward fill") { let last = present[0]; return vals.map((v) => (v == null ? last : (last = Number(v)))); }
  if (method === "Backward fill") { const out = vals.map((v) => (v == null ? null : Number(v))); let next = present[present.length - 1]; for (let i = out.length - 1; i >= 0; i--) { if (out[i] == null) out[i] = next; else next = out[i] as number; } return out.map((v) => Number(v)); }
  if (method === "Interpolate") return interpolateCol(vals);
  const fill = imputeFill(present, method);
  return vals.map((v) => (v == null ? fill : Number(v)));
}
export function scaleNumCol(vals: (number | null)[], method: string): (number | null)[] {
  if (method === "None") return vals;
  const present = vals.filter((v): v is number => v != null);
  if (!present.length) return vals;
  if (method === "MinMaxScaler") { const mn = Math.min(...present), mx = Math.max(...present), sp = (mx - mn) || 1; return vals.map((v) => (v == null ? null : (Number(v) - mn) / sp)); }
  if (method === "MaxAbsScaler") { const mx = Math.max(...present.map((x) => Math.abs(x))) || 1; return vals.map((v) => (v == null ? null : Number(v) / mx)); }
  if (method === "RobustScaler") { const s = [...present].sort((a, b) => a - b); const med = quantile(s, 0.5); const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), iqr = (q3 - q1) || 1; return vals.map((v) => (v == null ? null : (Number(v) - med) / iqr)); }
  if (method === "QuantileUniform") { const s = [...present].sort((a, b) => a - b); const nn = s.length; return vals.map((v) => { if (v == null) return null; const x = Number(v); let lo = 0, hi = nn; while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] <= x) lo = m + 1; else hi = m; } return nn > 1 ? (lo - 1) / (nn - 1) : 0.5; }); }
  const mean = present.reduce((a, b) => a + b, 0) / present.length; const std = Math.sqrt(present.reduce((a, b) => a + (b - mean) ** 2, 0) / present.length) || 1;
  return vals.map((v) => (v == null ? null : (Number(v) - mean) / std)); // StandardScaler
}
export function transformNumCol(vals: (number | null)[], method: string): (number | null)[] {
  return vals.map((v) => {
    if (v == null) return null; const x = Number(v);
    switch (method) {
      case "Log": return Number(Math.log1p(Math.abs(x)).toFixed(4));
      case "Log1p": return Number(Math.log1p(Math.max(x, -0.999999)).toFixed(4));
      case "Sqrt": return Number(Math.sqrt(Math.abs(x)).toFixed(4));
      case "Square": return x * x;
      case "Cube root": return Number(Math.cbrt(x).toFixed(4));
      case "Reciprocal": return Number((1 / (x || 1e-9)).toFixed(4));
      case "Absolute": return Math.abs(x);
      default: return x;
    }
  });
}
export function outlierNumCol(vals: (number | null)[], method: string): (number | null)[] {
  const present = vals.filter((v): v is number => v != null);
  if (!present.length) return vals;
  const s = [...present].sort((a, b) => a - b);
  const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), iqr = q3 - q1, med = quantile(s, 0.5);
  const mean = present.reduce((a, b) => a + b, 0) / present.length, std = Math.sqrt(present.reduce((a, b) => a + (b - mean) ** 2, 0) / present.length) || 1;
  let lo: number, hi: number;
  if (method === "Z-score clip") { lo = mean - 3 * std; hi = mean + 3 * std; }
  else if (method === "Winsorize 5%") { lo = quantile(s, 0.05); hi = quantile(s, 0.95); }
  else { lo = q1 - 1.5 * iqr; hi = q3 + 1.5 * iqr; } // IQR clip / IQR replace
  return vals.map((v) => { if (v == null) return null; const x = Number(v); if (method === "IQR replace") return (x < lo || x > hi ? med : x); return Math.min(hi, Math.max(lo, x)); });
}
export function binNumCol(vals: (number | null)[], method: string): (number | null)[] {
  const k = method.includes("10") ? 10 : 5;
  const present = vals.filter((v): v is number => v != null);
  if (!present.length) return vals;
  if (method.startsWith("Equal-freq")) {
    const s = [...present].sort((a, b) => a - b);
    const edges = Array.from({ length: k - 1 }, (_, i) => quantile(s, (i + 1) / k));
    return vals.map((v) => { if (v == null) return null; const x = Number(v); let b = 0; while (b < edges.length && x > edges[b]) b++; return b; });
  }
  const mn = Math.min(...present), mx = Math.max(...present), sp = (mx - mn) || 1; // Equal-width
  return vals.map((v) => { if (v == null) return null; let b = Math.floor(((Number(v) - mn) / sp) * k); if (b >= k) b = k - 1; if (b < 0) b = 0; return b; });
}
// Fixed, sklearn-style order in which per-column ops are applied for training.
const NUM_OP_ORDER = ["Impute missing", "Handle outliers", "Transform", "Bin / discretize", "Scale / normalize"];

export interface BuiltData { X: number[][]; y: number[]; featureNames: string[]; classes?: string[]; }
export function buildMatrix(ds: Dataset, featureCols: string[], targetName: string, task: Task, steps: PrepStep[]): BuiltData {
  const last = (op: string, name: string) => [...steps].reverse().find((s) => s.op === op && s.cols.includes(name));
  const dropped = new Set(steps.filter((s) => s.op === "Drop column").flatMap((s) => s.cols));
  const feats = featureCols.filter((c) => !dropped.has(c) && c !== targetName);
  const outCols: { name: string; values: number[] }[] = [];
  for (const name of feats) {
    const col = ds.columns.find((c) => c.name === name); if (!col) continue;
    if (col.type === "num") {
      let arr: (number | null)[] = col.values.map((v) => (v == null ? null : Number(v)));
      for (const op of NUM_OP_ORDER) {
        const m = last(op, name)?.method;
        if (op === "Impute missing") arr = imputeNumCol(arr, m || "Mean"); // always ensure no nulls
        else if (!m) continue;
        else if (op === "Handle outliers") arr = outlierNumCol(arr, m);
        else if (op === "Transform") arr = transformNumCol(arr, m);
        else if (op === "Bin / discretize") arr = binNumCol(arr, m);
        else if (op === "Scale / normalize") arr = scaleNumCol(arr, m);
      }
      outCols.push({ name, values: arr.map((v) => (v == null ? 0 : Number(v))) });
    } else {
      const raw = col.values.map((v) => (v == null ? null : String(v)));
      const mode = modeStr(raw.filter((v): v is string => v != null));
      const impM = last("Impute missing", name)?.method;
      const filled = raw.map((v) => (v == null ? (impM === "Constant" ? "missing" : mode) : v)) as string[];
      const enc = last("Encode categorical", name)?.method || "One-Hot";
      const cats = Array.from(new Set(filled));
      if (enc === "Ordinal") { const map = new Map(cats.map((c, i) => [c, i])); outCols.push({ name, values: filled.map((v) => map.get(v)!) }); }
      else if (enc === "Frequency") { const f = new Map<string, number>(); filled.forEach((v) => f.set(v, (f.get(v) || 0) + 1)); outCols.push({ name, values: filled.map((v) => f.get(v)! / filled.length) }); }
      else if (enc === "Count") { const f = new Map<string, number>(); filled.forEach((v) => f.set(v, (f.get(v) || 0) + 1)); outCols.push({ name, values: filled.map((v) => f.get(v)!) }); }
      else if (enc === "Binary") { const map = new Map(cats.map((c, i) => [c, i])); const bits = Math.max(1, Math.ceil(Math.log2(cats.length || 1))); for (let b = 0; b < bits; b++) outCols.push({ name: `${name}_b${b}`, values: filled.map((v) => (map.get(v)! >> b) & 1) }); }
      else for (const cat of cats) outCols.push({ name: `${name}=${cat}`, values: filled.map((v) => (v === cat ? 1 : 0)) }); // One-Hot
    }
  }
  const X = Array.from({ length: ds.nrows }, (_, r) => outCols.map((c) => c.values[r]));
  const featureNames = outCols.map((c) => c.name);
  const target = ds.columns.find((c) => c.name === targetName)!;
  let y: number[]; let classes: string[] | undefined;
  if (task === "classification") {
    const tv = target.values.map((v) => (v == null ? "?" : String(v)));
    classes = Array.from(new Set(tv));
    const cmap = new Map(classes.map((c, i) => [c, i]));
    y = tv.map((v) => cmap.get(v)!);
  } else {
    y = target.values.map((v) => (v == null ? 0 : Number(v)));
  }
  return { X, y, featureNames, classes };
}

// ── split ──
function seededShuffle<T>(arr: T[], seed = 42): T[] {
  const a = [...arr]; let s = seed;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
export function split(X: number[][], y: number[], testSize = 0.2) {
  const idx = seededShuffle(X.map((_, i) => i));
  const nTest = Math.max(1, Math.round(idx.length * testSize));
  const testI = new Set(idx.slice(0, nTest));
  const Xtr: number[][] = [], ytr: number[] = [], Xte: number[][] = [], yte: number[] = [];
  X.forEach((row, i) => { if (testI.has(i)) { Xte.push(row); yte.push(y[i]); } else { Xtr.push(row); ytr.push(y[i]); } });
  return { Xtr, ytr, Xte, yte };
}

// ── math ──
const dot = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function softmax(z: number[]): number[] { const m = Math.max(...z); const e = z.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); }

// ── models ──
export type TreeNode =
  | { leaf: true; value: number; n: number }
  | { leaf: false; feat: number; thr: number; n: number; left: TreeNode; right: TreeNode };
export type Model =
  | { kind: "logreg"; W: number[][]; classes: number; loss?: number[] }
  | { kind: "linear"; w: number[]; loss?: number[] }
  | { kind: "knn"; X: number[][]; y: number[]; k: number; weights: string; task: Task; classes: number }
  | { kind: "gnb"; means: number[][]; vars: number[][]; priors: number[]; classes: number }
  | { kind: "tree"; root: TreeNode; task: Task; classes: number; importance: number[] }
  | { kind: "forest"; trees: TreeNode[]; task: Task; classes: number; importance: number[]; nTrees: number };

export function trainLogReg(X: number[][], y: number[], nClasses: number, p: { lr: number; epochs: number; l2: number }): Model {
  const n = X.length, d = X[0]?.length || 0;
  const Xb = X.map((r) => [1, ...r]);
  const W = Array.from({ length: nClasses }, () => new Array(d + 1).fill(0));
  const loss: number[] = []; const rec = Math.max(1, Math.floor(p.epochs / 50));
  for (let ep = 0; ep < p.epochs; ep++) {
    const grad = Array.from({ length: nClasses }, () => new Array(d + 1).fill(0));
    for (let i = 0; i < n; i++) {
      const probs = softmax(W.map((w) => dot(w, Xb[i])));
      for (let k = 0; k < nClasses; k++) { const err = probs[k] - (y[i] === k ? 1 : 0); for (let j = 0; j <= d; j++) grad[k][j] += err * Xb[i][j]; }
    }
    for (let k = 0; k < nClasses; k++) for (let j = 0; j <= d; j++) { let g = grad[k][j] / n; if (j > 0) g += p.l2 * W[k][j]; W[k][j] -= p.lr * g; }
    if (ep % rec === 0 || ep === p.epochs - 1) { let L = 0; for (let i = 0; i < n; i++) { const pr = softmax(W.map((w) => dot(w, Xb[i]))); L += -Math.log(Math.max(1e-9, pr[y[i]])); } loss.push(L / n); }
  }
  return { kind: "logreg", W, classes: nClasses, loss };
}
export function trainGNB(X: number[][], y: number[], nClasses: number): Model {
  const d = X[0]?.length || 0;
  const means = Array.from({ length: nClasses }, () => new Array(d).fill(0));
  const vars = Array.from({ length: nClasses }, () => new Array(d).fill(0));
  const counts = new Array(nClasses).fill(0);
  X.forEach((row, i) => { counts[y[i]]++; row.forEach((v, j) => (means[y[i]][j] += v)); });
  for (let k = 0; k < nClasses; k++) for (let j = 0; j < d; j++) means[k][j] /= counts[k] || 1;
  X.forEach((row, i) => row.forEach((v, j) => (vars[y[i]][j] += (v - means[y[i]][j]) ** 2)));
  for (let k = 0; k < nClasses; k++) for (let j = 0; j < d; j++) vars[k][j] = vars[k][j] / (counts[k] || 1) + 1e-6;
  const priors = counts.map((c) => c / X.length);
  return { kind: "gnb", means, vars, priors, classes: nClasses };
}
function predictGNBOne(m: Extract<Model, { kind: "gnb" }>, x: number[]): number {
  let best = 0, bestLL = -Infinity;
  for (let k = 0; k < m.classes; k++) {
    let ll = Math.log(m.priors[k] || 1e-9);
    for (let j = 0; j < x.length; j++) { const v = m.vars[k][j]; ll += -0.5 * Math.log(2 * Math.PI * v) - ((x[j] - m.means[k][j]) ** 2) / (2 * v); }
    if (ll > bestLL) { bestLL = ll; best = k; }
  }
  return best;
}
export function trainLinear(X: number[][], y: number[], p: { lr: number; epochs: number; alpha: number }): Model {
  const n = X.length, d = X[0]?.length || 0;
  const Xb = X.map((r) => [1, ...r]);
  const w = new Array(d + 1).fill(0);
  const loss: number[] = []; const rec = Math.max(1, Math.floor(p.epochs / 50));
  for (let ep = 0; ep < p.epochs; ep++) {
    const grad = new Array(d + 1).fill(0);
    for (let i = 0; i < n; i++) { const pred = dot(w, Xb[i]); const err = pred - y[i]; for (let j = 0; j <= d; j++) grad[j] += err * Xb[i][j]; }
    for (let j = 0; j <= d; j++) { let g = grad[j] / n; if (j > 0) g += p.alpha * w[j]; w[j] -= p.lr * g; }
    if (ep % rec === 0 || ep === p.epochs - 1) { let L = 0; for (let i = 0; i < n; i++) L += (dot(w, Xb[i]) - y[i]) ** 2; loss.push(L / n); }
  }
  return { kind: "linear", w, loss };
}
export function trainKNN(X: number[][], y: number[], k: number, weights: string, task: Task, classes: number): Model {
  return { kind: "knn", X, y, k, weights, task, classes };
}
function knnPredictOne(m: Extract<Model, { kind: "knn" }>, x: number[]): number {
  const dists = m.X.map((row, i) => ({ d: Math.sqrt(row.reduce((a, v, j) => a + (v - x[j]) ** 2, 0)), y: m.y[i] }));
  dists.sort((a, b) => a.d - b.d);
  const near = dists.slice(0, m.k);
  if (m.task === "regression") { const wts = near.map((p) => (m.weights === "distance" ? 1 / (p.d + 1e-6) : 1)); return near.reduce((a, p, i) => a + p.y * wts[i], 0) / wts.reduce((a, b) => a + b, 0); }
  const votes = new Array(m.classes).fill(0); near.forEach((p) => { votes[p.y] += m.weights === "distance" ? 1 / (p.d + 1e-6) : 1; });
  return votes.indexOf(Math.max(...votes));
}
// ── decision tree (CART) + random forest — real recursive fits, no libraries ──
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gini(y: number[], nClasses: number): number { if (!y.length) return 0; const c = new Array(nClasses).fill(0); y.forEach((v) => c[v]++); let s = 1; for (let k = 0; k < nClasses; k++) { const p = c[k] / y.length; s -= p * p; } return s; }
function variance(y: number[]): number { if (!y.length) return 0; const m = y.reduce((a, b) => a + b, 0) / y.length; return y.reduce((a, b) => a + (b - m) ** 2, 0) / y.length; }
function leafValue(y: number[], task: Task, nClasses: number): number { if (task === "regression") return y.reduce((a, b) => a + b, 0) / (y.length || 1); const c = new Array(nClasses).fill(0); y.forEach((v) => c[v]++); return c.indexOf(Math.max(...c)); }
interface TreeOpts { maxDepth: number; minSplit: number; maxFeatures: number; }
function buildTree(X: number[][], y: number[], task: Task, nClasses: number, opts: TreeOpts, imp: number[], depth: number, rnd: () => number): TreeNode {
  const n = y.length;
  const nodeImp = task === "regression" ? variance(y) : gini(y, nClasses);
  const pure = task === "regression" ? nodeImp < 1e-9 : new Set(y).size <= 1;
  if (depth >= opts.maxDepth || n < opts.minSplit || pure) return { leaf: true, value: leafValue(y, task, nClasses), n };
  const d = X[0]?.length || 0;
  let feats = Array.from({ length: d }, (_, i) => i);
  if (opts.maxFeatures < d) { for (let i = feats.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [feats[i], feats[j]] = [feats[j], feats[i]]; } feats = feats.slice(0, opts.maxFeatures); }
  let best = { gain: 1e-9, feat: -1, thr: 0 };
  for (const f of feats) {
    const vals = X.map((r) => r[f]);
    const uniq = Array.from(new Set(vals)).sort((a, b) => a - b);
    if (uniq.length < 2) continue;
    const stepc = Math.max(1, Math.floor((uniq.length - 1) / 12));
    for (let u = stepc; u < uniq.length; u += stepc) {
      const thr = (uniq[u - 1] + uniq[u]) / 2;
      const yl: number[] = [], yr: number[] = [];
      for (let i = 0; i < n; i++) (vals[i] <= thr ? yl : yr).push(y[i]);
      if (!yl.length || !yr.length) continue;
      const il = task === "regression" ? variance(yl) : gini(yl, nClasses);
      const ir = task === "regression" ? variance(yr) : gini(yr, nClasses);
      const gain = nodeImp - (yl.length / n) * il - (yr.length / n) * ir;
      if (gain > best.gain) best = { gain, feat: f, thr };
    }
  }
  if (best.feat < 0) return { leaf: true, value: leafValue(y, task, nClasses), n };
  imp[best.feat] += best.gain * n;
  const Xl: number[][] = [], yl: number[] = [], Xr: number[][] = [], yr: number[] = [];
  for (let i = 0; i < n; i++) { if (X[i][best.feat] <= best.thr) { Xl.push(X[i]); yl.push(y[i]); } else { Xr.push(X[i]); yr.push(y[i]); } }
  return { leaf: false, feat: best.feat, thr: best.thr, n, left: buildTree(Xl, yl, task, nClasses, opts, imp, depth + 1, rnd), right: buildTree(Xr, yr, task, nClasses, opts, imp, depth + 1, rnd) };
}
function predictTreeOne(node: TreeNode, x: number[]): number { let cur = node; while (!cur.leaf) cur = x[cur.feat] <= cur.thr ? cur.left : cur.right; return cur.value; }
export function trainTree(X: number[][], y: number[], task: Task, nClasses: number, p: { maxDepth: number; minSplit: number }): Model {
  const d = X[0]?.length || 0; const imp = new Array(d).fill(0);
  const root = buildTree(X, y, task, nClasses, { maxDepth: p.maxDepth, minSplit: Math.max(2, p.minSplit), maxFeatures: d }, imp, 0, mulberry32(42));
  return { kind: "tree", root, task, classes: nClasses, importance: imp };
}
export function trainForest(X: number[][], y: number[], task: Task, nClasses: number, p: { nTrees: number; maxDepth: number; minSplit: number }): Model {
  const d = X[0]?.length || 0, imp = new Array(d).fill(0), trees: TreeNode[] = [], rnd = mulberry32(123);
  const maxFeatures = Math.max(1, Math.round(Math.sqrt(d)));
  for (let t = 0; t < p.nTrees; t++) {
    const Xs: number[][] = [], ys: number[] = [];
    for (let i = 0; i < X.length; i++) { const j = Math.floor(rnd() * X.length); Xs.push(X[j]); ys.push(y[j]); }
    trees.push(buildTree(Xs, ys, task, nClasses, { maxDepth: p.maxDepth, minSplit: Math.max(2, p.minSplit), maxFeatures }, imp, 0, rnd));
  }
  return { kind: "forest", trees, task, classes: nClasses, importance: imp, nTrees: p.nTrees };
}
function predictForestOne(m: Extract<Model, { kind: "forest" }>, x: number[]): number {
  if (m.task === "regression") { let s = 0; for (const t of m.trees) s += predictTreeOne(t, x); return s / (m.trees.length || 1); }
  const votes = new Array(m.classes).fill(0); for (const t of m.trees) votes[predictTreeOne(t, x)]++; return votes.indexOf(Math.max(...votes));
}
export function treeDepth(node: TreeNode): number { return node.leaf ? 1 : 1 + Math.max(treeDepth(node.left), treeDepth(node.right)); }
export function countNodes(node: TreeNode): number { return node.leaf ? 1 : 1 + countNodes(node.left) + countNodes(node.right); }

export function predict(m: Model, X: number[][]): number[] {
  if (m.kind === "logreg") { const Xb = X.map((r) => [1, ...r]); return Xb.map((r) => { const pr = softmax(m.W.map((w) => dot(w, r))); return pr.indexOf(Math.max(...pr)); }); }
  if (m.kind === "linear") { const Xb = X.map((r) => [1, ...r]); return Xb.map((r) => dot(m.w, r)); }
  if (m.kind === "gnb") return X.map((x) => predictGNBOne(m, x));
  if (m.kind === "tree") return X.map((x) => predictTreeOne(m.root, x));
  if (m.kind === "forest") return X.map((x) => predictForestOne(m, x));
  return X.map((x) => knnPredictOne(m, x));
}
export function featureImportance(m: Model, names: string[]): { name: string; w: number }[] | null {
  if (m.kind === "logreg") { const imp = names.map((_, j) => m.W.reduce((a, w) => a + Math.abs(w[j + 1]), 0)); const mx = Math.max(...imp, 1e-9); return names.map((n, j) => ({ name: n, w: imp[j] / mx })).sort((a, b) => b.w - a.w).slice(0, 8); }
  if (m.kind === "linear") { const imp = names.map((_, j) => Math.abs(m.w[j + 1])); const mx = Math.max(...imp, 1e-9); return names.map((n, j) => ({ name: n, w: imp[j] / mx })).sort((a, b) => b.w - a.w).slice(0, 8); }
  if (m.kind === "tree" || m.kind === "forest") { const mx = Math.max(...m.importance, 1e-9); return names.map((n, j) => ({ name: n, w: (m.importance[j] || 0) / mx })).sort((a, b) => b.w - a.w).slice(0, 8); }
  return null;
}

// ── metrics ──
export interface ClsMetrics { task: "classification"; accuracy: number; precision: number; recall: number; f1: number; confusion: number[][]; classes: string[]; }
export interface RegMetrics { task: "regression"; r2: number; mae: number; rmse: number; }
export function classificationMetrics(yTrue: number[], yPred: number[], classes: string[]): ClsMetrics {
  const K = classes.length; const cm = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < yTrue.length; i++) cm[yTrue[i]][yPred[i]]++;
  const acc = yTrue.reduce((a, t, i) => a + (t === yPred[i] ? 1 : 0), 0) / yTrue.length;
  let pr = 0, rc = 0;
  for (let k = 0; k < K; k++) { const tp = cm[k][k]; const fp = cm.reduce((a, r) => a + r[k], 0) - tp; const fn = cm[k].reduce((a, b) => a + b, 0) - tp; pr += tp / (tp + fp || 1); rc += tp / (tp + fn || 1); }
  pr /= K; rc /= K; const f1 = 2 * pr * rc / (pr + rc || 1);
  return { task: "classification", accuracy: acc, precision: pr, recall: rc, f1, confusion: cm, classes };
}
export function regressionMetrics(yTrue: number[], yPred: number[]): RegMetrics {
  const n = yTrue.length; const mean = yTrue.reduce((a, b) => a + b, 0) / n;
  let ssRes = 0, ssTot = 0, mae = 0;
  for (let i = 0; i < n; i++) { ssRes += (yTrue[i] - yPred[i]) ** 2; ssTot += (yTrue[i] - mean) ** 2; mae += Math.abs(yTrue[i] - yPred[i]); }
  return { task: "regression", r2: 1 - ssRes / (ssTot || 1), mae: mae / n, rmse: Math.sqrt(ssRes / n) };
}

// ── trainer + CV ──
export interface TrainConfig { task: Task; algo: string; params: Record<string, string>; testSize: number; cvFolds: number; }
export function makeModel(cfg: TrainConfig, X: number[][], y: number[], nClasses: number): Model {
  const num = (k: string, def: number) => { const v = Number(cfg.params[k]); return isFinite(v) ? v : def; };
  // L2 penalty in this averaged-gradient solver must be small; C=1 → 0.01 (light).
  // The old mapping (1/C → 1.0 at C=1) over-regularised and collapsed to the majority class.
  if (cfg.algo === "LogisticRegression") return trainLogReg(X, y, nClasses, { lr: num("learning_rate", 0.2), epochs: num("max_iter", 300), l2: 0.01 / (num("C", 1) || 1) });
  if (cfg.algo === "KNeighborsClassifier") return trainKNN(X, y, num("n_neighbors", 5), cfg.params.weights || "uniform", "classification", nClasses);
  if (cfg.algo === "GaussianNB") return trainGNB(X, y, nClasses);
  if (cfg.algo === "LinearRegression") return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0 });
  // Same rescaling as logreg: the raw alpha (default 1) over-regularised this
  // averaged-gradient solver and underfit; 0.01× keeps it mild but effective.
  if (cfg.algo === "Ridge") return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0.01 * num("alpha", 1) });
  if (cfg.algo === "KNeighborsRegressor") return trainKNN(X, y, num("n_neighbors", 5), cfg.params.weights || "uniform", "regression", 0);
  if (cfg.algo === "DecisionTree") return trainTree(X, y, cfg.task, nClasses, { maxDepth: num("max_depth", 5), minSplit: num("min_samples_split", 2) });
  if (cfg.algo === "RandomForest") return trainForest(X, y, cfg.task, nClasses, { nTrees: num("n_estimators", 25), maxDepth: num("max_depth", 6), minSplit: num("min_samples_split", 2) });
  return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0 });
}

// k-fold cross-validation. Returns EVERY fold with its score and split sizes, so the
// UI can show that all folds actually ran (nothing silently dropped).
export interface FoldResult { fold: number; score: number; trainN: number; testN: number; }
export function crossValDetailed(cfg: TrainConfig, X: number[][], y: number[], nClasses: number): FoldResult[] {
  const folds = Math.max(2, Math.min(10, Math.min(cfg.cvFolds, X.length)));
  const idx = seededShuffle(X.map((_, i) => i), 7);
  const out: FoldResult[] = [];
  for (let f = 0; f < folds; f++) {
    const testI = new Set(idx.filter((_, i) => i % folds === f));
    const Xtr: number[][] = [], ytr: number[] = [], Xte: number[][] = [], yte: number[] = [];
    X.forEach((row, i) => { if (testI.has(i)) { Xte.push(row); yte.push(y[i]); } else { Xtr.push(row); ytr.push(y[i]); } });
    if (!Xtr.length || !Xte.length) continue;
    const m = makeModel(cfg, Xtr, ytr, nClasses);
    const pred = predict(m, Xte);
    const score = cfg.task === "classification"
      ? yte.reduce((a, t, i) => a + (t === Math.round(pred[i]) ? 1 : 0), 0) / yte.length
      : regressionMetrics(yte, pred).r2;
    out.push({ fold: f + 1, score, trainN: Xtr.length, testN: Xte.length });
  }
  return out;
}
export function crossVal(cfg: TrainConfig, X: number[][], y: number[], nClasses: number): number[] {
  return crossValDetailed(cfg, X, y, nClasses).map((f) => f.score);
}
// Decision boundary over TWO numeric features (trains a 2-feature model of the
// same type, predicts a grid, returns it + the data points). Classification only.
export function decisionSurface(ds: Dataset, targetName: string, f1: string, f2: string, cfg: TrainConfig, res = 46):
  { xs: number[]; ys: number[]; z: number[][]; points: { x: number; y: number; c: number }[]; classes: string[]; acc: number } | null {
  const c1 = ds.columns.find((c) => c.name === f1), c2 = ds.columns.find((c) => c.name === f2), ty = ds.columns.find((c) => c.name === targetName);
  if (!c1 || !c2 || !ty || c1.type !== "num" || c2.type !== "num" || f1 === f2) return null;
  const raw: [number, number, string][] = [];
  for (let i = 0; i < ds.nrows; i++) { const a = c1.values[i], b = c2.values[i], t = ty.values[i]; if (a == null || b == null || t == null) continue; raw.push([Number(a), Number(b), String(t)]); }
  if (raw.length < 4) return null;
  const classes = Array.from(new Set(raw.map((r) => r[2])));
  if (classes.length < 2 || classes.length > 8) return null;
  const cmap = new Map(classes.map((c, i) => [c, i]));
  const xs0 = raw.map((r) => r[0]), ys0 = raw.map((r) => r[1]);
  const m1 = mean(xs0), s1 = std(xs0) || 1, m2 = mean(ys0), s2 = std(ys0) || 1;
  const X = raw.map((r) => [(r[0] - m1) / s1, (r[1] - m2) / s2]);
  const y = raw.map((r) => cmap.get(r[2])!);
  const model = makeModel({ ...cfg, task: "classification" }, X, y, classes.length);
  const min1 = Math.min(...xs0), max1 = Math.max(...xs0), min2 = Math.min(...ys0), max2 = Math.max(...ys0);
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < res; i++) xs.push(min1 + ((max1 - min1) * i) / (res - 1));
  for (let j = 0; j < res; j++) ys.push(min2 + ((max2 - min2) * j) / (res - 1));
  const flat: number[][] = [];
  for (const gy of ys) for (const gx of xs) flat.push([(gx - m1) / s1, (gy - m2) / s2]);
  const preds = predict(model, flat);
  const z: number[][] = [];
  for (let j = 0; j < res; j++) z.push(preds.slice(j * res, j * res + res).map((p) => Math.round(p)));
  const points = raw.map((r, i) => ({ x: r[0], y: r[1], c: y[i] }));
  const yp = predict(model, X);
  const acc = y.reduce((a, t, i) => a + (Math.round(yp[i]) === t ? 1 : 0), 0) / y.length;
  return { xs, ys, z, points, classes, acc };
}

// Learning curve: train on growing fractions of the training set; return train vs
// hold-out score at each size (teaches over/under-fitting).
export function learningCurve(X: number[][], y: number[], cfg: TrainConfig, nClasses: number):
  { n: number; train: number; test: number }[] {
  const { Xtr, ytr, Xte, yte } = split(X, y, cfg.testSize || 0.2);
  const acc = (yt: number[], yp: number[]) => yt.reduce((a, t, i) => a + (Math.round(yp[i]) === t ? 1 : 0), 0) / (yt.length || 1);
  const r2 = (yt: number[], yp: number[]) => { const m = mean(yt); let ss = 0, st = 0; for (let i = 0; i < yt.length; i++) { ss += (yt[i] - yp[i]) ** 2; st += (yt[i] - m) ** 2; } return st ? 1 - ss / st : 0; };
  const score = cfg.task === "classification" ? acc : r2;
  const fracs = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1];
  return fracs.map((f) => {
    const n = Math.max(2, Math.round(Xtr.length * f));
    const xs = Xtr.slice(0, n), ys = ytr.slice(0, n);
    const model = makeModel(cfg, xs, ys, nClasses);
    return { n, train: score(ys, predict(model, xs)), test: score(yte, predict(model, Xte)) };
  });
}

// ── maths-mode instrumentation: expose the intermediate numbers the training math
// produces (a faithful mirror of trainLogReg / trainLinear, recording snapshots) ──
export interface GdTrace { kind: "logreg" | "linear"; d: number; lr: number; grad0: number[]; snaps: { ep: number; loss: number; gnorm: number; w: number[] }[]; }
export function gdTrace(cfg: TrainConfig, X: number[][], y: number[], nClasses: number): GdTrace | null {
  const isLog = cfg.algo === "LogisticRegression";
  const isLin = cfg.algo === "LinearRegression" || cfg.algo === "Ridge";
  if (!isLog && !isLin) return null;
  const n = X.length, d = X[0]?.length || 0;
  const Xb = X.map((r) => [1, ...r]);
  const epochs = isLog ? Math.max(1, Math.round(Number(cfg.params.max_iter) || 300)) : 400;
  const lr = isLog ? (Number(cfg.params.learning_rate) || 0.2) : 0.05;
  const want = new Set<number>([0, 1, 2]); for (let i = 0; i <= 10; i++) want.add(Math.round((epochs - 1) * i / 10));
  const snaps: GdTrace["snaps"] = []; let grad0: number[] = [];
  if (isLin) {
    const w = new Array(d + 1).fill(0); const alpha = cfg.algo === "Ridge" ? 0.01 * (Number(cfg.params.alpha) || 1) : 0;
    for (let ep = 0; ep < epochs; ep++) {
      const grad = new Array(d + 1).fill(0);
      for (let i = 0; i < n; i++) { const pred = dot(w, Xb[i]); const err = pred - y[i]; for (let j = 0; j <= d; j++) grad[j] += err * Xb[i][j]; }
      const gn = grad.map((g) => g / n);
      for (let j = 0; j <= d; j++) { let g = gn[j]; if (j > 0) g += alpha * w[j]; w[j] -= lr * g; }
      if (ep === 0) grad0 = gn;
      if (want.has(ep)) { let L = 0; for (let i = 0; i < n; i++) L += (dot(w, Xb[i]) - y[i]) ** 2; snaps.push({ ep, loss: L / n, gnorm: Math.sqrt(gn.reduce((a, g) => a + g * g, 0)), w: [...w] }); }
    }
    return { kind: "linear", d, lr, grad0, snaps };
  }
  const W = Array.from({ length: nClasses }, () => new Array(d + 1).fill(0)); const l2 = 0.01 / (Number(cfg.params.C) || 1);
  const cls = Math.min(1, nClasses - 1);
  for (let ep = 0; ep < epochs; ep++) {
    const grad = Array.from({ length: nClasses }, () => new Array(d + 1).fill(0));
    for (let i = 0; i < n; i++) { const probs = softmax(W.map((w) => dot(w, Xb[i]))); for (let k = 0; k < nClasses; k++) { const err = probs[k] - (y[i] === k ? 1 : 0); for (let j = 0; j <= d; j++) grad[k][j] += err * Xb[i][j]; } }
    const gn = grad[cls].map((g) => g / n);
    for (let k = 0; k < nClasses; k++) for (let j = 0; j <= d; j++) { let g = grad[k][j] / n; if (j > 0) g += l2 * W[k][j]; W[k][j] -= lr * g; }
    if (ep === 0) grad0 = gn;
    if (want.has(ep)) { let L = 0; for (let i = 0; i < n; i++) { const pr = softmax(W.map((w) => dot(w, Xb[i]))); L += -Math.log(Math.max(1e-9, pr[y[i]])); } snaps.push({ ep, loss: L / n, gnorm: Math.sqrt(gn.reduce((a, g) => a + g * g, 0)), w: [...W[cls]] }); }
  }
  return { kind: "logreg", d, lr, grad0, snaps };
}
// Animated gradient descent on a *visualisable* problem: 2 features → a decision
// boundary that improves per epoch (classification), or 1 feature → a line that
// fits (regression). Returns frames so the UI can play the training.
export interface GdAnim {
  reg: boolean; xs: number[]; ys?: number[]; classes: string[];
  points: { x: number; y: number; c: number }[];
  frames: { ep: number; loss: number; z?: number[][]; line?: number[] }[];
}
export function gdAnim(ds: Dataset, targetName: string, f1: string, f2: string, cfg: TrainConfig, res = 34): GdAnim | null {
  const ty = ds.columns.find((c) => c.name === targetName);
  const c1 = ds.columns.find((c) => c.name === f1);
  if (!ty || !c1 || c1.type !== "num") return null;
  const reg = cfg.task === "regression";
  const epList = [0, 1, 2, 3, 5, 8, 12, 18, 26, 40, 70, 120, 200, 299];

  if (!reg) {
    const c2 = ds.columns.find((c) => c.name === f2);
    if (!c2 || c2.type !== "num" || f1 === f2) return null;
    const raw: [number, number, string][] = [];
    for (let i = 0; i < ds.nrows; i++) { const a = c1.values[i], b = c2.values[i], t = ty.values[i]; if (a == null || b == null || t == null) continue; raw.push([Number(a), Number(b), String(t)]); }
    if (raw.length < 6) return null;
    const classes = Array.from(new Set(raw.map((r) => r[2]))); if (classes.length < 2 || classes.length > 6) return null;
    const cmap = new Map(classes.map((c, i) => [c, i]));
    const xs0 = raw.map((r) => r[0]), ys0 = raw.map((r) => r[1]);
    const m1 = mean(xs0), s1 = std(xs0) || 1, m2 = mean(ys0), s2 = std(ys0) || 1;
    const X = raw.map((r) => [(r[0] - m1) / s1, (r[1] - m2) / s2]); const y = raw.map((r) => cmap.get(r[2])!);
    const K = classes.length, n = X.length, Xb = X.map((r) => [1, ...r]);
    const W = Array.from({ length: K }, () => [0, 0, 0]); const lr = Number(cfg.params.learning_rate) || 0.3, l2 = 0.01 / (Number(cfg.params.C) || 1);
    const mn1 = Math.min(...xs0), mx1 = Math.max(...xs0), mn2 = Math.min(...ys0), mx2 = Math.max(...ys0);
    const gx: number[] = [], gy: number[] = [];
    for (let i = 0; i < res; i++) gx.push(mn1 + ((mx1 - mn1) * i) / (res - 1));
    for (let j = 0; j < res; j++) gy.push(mn2 + ((mx2 - mn2) * j) / (res - 1));
    const frames: GdAnim["frames"] = [];
    const snapSet = new Set(epList);
    const total = 300;
    for (let ep = 0; ep < total; ep++) {
      if (snapSet.has(ep)) {
        const z: number[][] = [];
        for (const vy of gy) { const rowz: number[] = []; for (const vx of gx) { const xb = [1, (vx - m1) / s1, (vy - m2) / s2]; const sc = W.map((w) => w[0] * xb[0] + w[1] * xb[1] + w[2] * xb[2]); const p = softmax(sc); rowz.push(p.indexOf(Math.max(...p))); } z.push(rowz); }
        let L = 0; for (let i = 0; i < n; i++) { const p = softmax(W.map((w) => dot(w, Xb[i]))); L += -Math.log(Math.max(1e-9, p[y[i]])); }
        frames.push({ ep, loss: L / n, z });
      }
      const G = Array.from({ length: K }, () => [0, 0, 0]);
      for (let i = 0; i < n; i++) { const p = softmax(W.map((w) => dot(w, Xb[i]))); for (let k = 0; k < K; k++) { const e = p[k] - (y[i] === k ? 1 : 0); for (let j = 0; j < 3; j++) G[k][j] += e * Xb[i][j]; } }
      for (let k = 0; k < K; k++) for (let j = 0; j < 3; j++) { let g = G[k][j] / n; if (j > 0) g += l2 * W[k][j]; W[k][j] -= lr * g; }
    }
    return { reg: false, xs: gx, ys: gy, classes, points: raw.map((r, i) => ({ x: r[0], y: r[1], c: y[i] })), frames };
  }

  // regression: fit y = w0 + w1·x on ONE feature (f1 vs target)
  const raw: [number, number][] = [];
  for (let i = 0; i < ds.nrows; i++) { const a = c1.values[i], t = ty.values[i]; if (a == null || t == null) continue; raw.push([Number(a), Number(t)]); }
  if (raw.length < 6) return null;
  const xs0 = raw.map((r) => r[0]); const m1 = mean(xs0), s1 = std(xs0) || 1;
  const X = raw.map((r) => (r[0] - m1) / s1), y = raw.map((r) => r[1]);
  const n = X.length; let w0 = 0, w1 = 0; const lr = 0.1;
  const mn = Math.min(...xs0), mx = Math.max(...xs0); const gx: number[] = [];
  for (let i = 0; i < res; i++) gx.push(mn + ((mx - mn) * i) / (res - 1));
  const frames: GdAnim["frames"] = []; const snapSet = new Set(epList);
  for (let ep = 0; ep < 300; ep++) {
    if (snapSet.has(ep)) { const line = gx.map((xv) => w0 + w1 * ((xv - m1) / s1)); let L = 0; for (let i = 0; i < n; i++) L += (w0 + w1 * X[i] - y[i]) ** 2; frames.push({ ep, loss: L / n, line }); }
    let g0 = 0, g1 = 0; for (let i = 0; i < n; i++) { const e = w0 + w1 * X[i] - y[i]; g0 += e; g1 += e * X[i]; }
    w0 -= lr * g0 / n; w1 -= lr * g1 / n;
  }
  return { reg: true, xs: gx, classes: [], points: raw.map((r) => ({ x: r[0], y: r[1], c: 0 })), frames };
}
export interface SplitMath { feat: number; thr: number; parent: number; left: number; right: number; gain: number; nL: number; nR: number; metric: "gini" | "variance"; }
export function rootSplitMath(model: Model, X: number[][], y: number[], nClasses: number): SplitMath | null {
  if (model.kind !== "tree" && model.kind !== "forest") return null;
  const root = model.kind === "tree" ? model.root : model.trees[0];
  if (root.leaf) return null;
  const reg = model.task === "regression";
  const imp = (yy: number[]) => (reg ? variance(yy) : gini(yy, nClasses));
  const yl: number[] = [], yr: number[] = [];
  for (let i = 0; i < X.length; i++) (X[i][root.feat] <= root.thr ? yl : yr).push(y[i]);
  const parent = imp(y), left = imp(yl), right = imp(yr);
  return { feat: root.feat, thr: root.thr, parent, left, right, gain: parent - (yl.length / y.length) * left - (yr.length / y.length) * right, nL: yl.length, nR: yr.length, metric: reg ? "variance" : "gini" };
}

// Full numeric-column values after EACH preprocessing step (cumulative) — powers
// the maths-mode step-by-step animation of a numeric feature transforming.
export type PrepCell = number | string | null;
export interface PrepColStep { label: string; op: string; method: string; changed: boolean; values: PrepCell[]; }
export function prepColTrace(ds: Dataset, steps: PrepStep[], colName: string): PrepColStep[] {
  const col = ds.columns.find((c) => c.name === colName);
  if (!col) return [];
  const num = col.type === "num";
  let cur: PrepCell[] = num ? col.values.map((v) => (v == null ? null : Number(v))) : col.values.map((v) => (v == null ? null : String(v)));
  const out: PrepColStep[] = [{ label: "raw", op: "raw", method: "source", changed: true, values: [...cur] }];
  for (const s of steps) {
    let changed = false;
    if (s.cols.includes(colName)) {
      if (num) {
        const n = cur as (number | null)[];
        if (s.op === "Impute missing") { cur = imputeNumCol(n, s.method); changed = true; }
        else if (s.op === "Scale / normalize") { cur = scaleNumCol(n, s.method); changed = true; }
        else if (s.op === "Handle outliers") { cur = outlierNumCol(n, s.method); changed = true; }
        else if (s.op === "Transform") { cur = transformNumCol(n, s.method); changed = true; }
        else if (s.op === "Bin / discretize") { cur = binNumCol(n, s.method); changed = true; }
      } else if (s.op === "Impute missing") {
        const present = (cur as (string | null)[]).filter((v): v is string => v != null);
        const fill = s.method === "Constant" ? "missing" : modeStr(present);
        cur = (cur as (string | null)[]).map((v) => (v == null ? fill : v)); changed = true;
      }
    }
    out.push({ label: `${s.op} · ${s.method}`, op: s.op, method: s.method, changed, values: [...cur] });
  }
  return out;
}
// Per-method fill values for an Impute step (what each method WOULD fill).
export function imputeMethodFills(before: PrepCell[], numeric: boolean): { name: string; fill: string }[] {
  const present = before.filter((v) => v != null);
  if (numeric) {
    const nums = present as number[]; if (!nums.length) return [];
    const sorted = [...nums].sort((a, b) => a - b); const med = sorted[Math.floor((sorted.length - 1) / 2)];
    const counts = new Map<number, number>(); nums.forEach((x) => counts.set(x, (counts.get(x) || 0) + 1));
    const modeN = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
    return [
      { name: "Mean", fill: (nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(2) },
      { name: "Median", fill: String(med) },
      { name: "Most frequent", fill: String(modeN) },
      { name: "Constant", fill: "0" },
    ];
  }
  const strs = present as string[]; if (!strs.length) return [];
  const counts = new Map<string, number>(); strs.forEach((x) => counts.set(x, (counts.get(x) || 0) + 1));
  const modeS = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  return [{ name: "Most frequent", fill: modeS }, { name: "Constant", fill: "missing" }];
}

export function mean(a: number[]): number { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
export function std(a: number[]): number { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); }

// Three-way hold-out split sizes for the validation-step diagram.
export interface Split3 { train: number; val: number; test: number; }
export function splitCounts(n: number, valSize: number, testSize: number): Split3 {
  const test = Math.round(n * testSize), val = Math.round(n * valSize);
  return { test, val, train: Math.max(0, n - test - val) };
}

// ── describe / quantiles / box ──
export function quantile(sortedAsc: number[], q: number): number {
  if (!sortedAsc.length) return 0;
  const pos = (sortedAsc.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? sortedAsc[lo] : sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * (pos - lo);
}
export interface DescribeRow { name: string; count: number; missing: number; mean: number; std: number; min: number; q25: number; q50: number; q75: number; max: number; }
export function describe(ds: Dataset): DescribeRow[] {
  return ds.columns.filter((c) => c.type === "num").map((c) => {
    const v = c.values.filter((x): x is number => x != null); const s = [...v].sort((a, b) => a - b);
    const mean = v.reduce((a, b) => a + b, 0) / (v.length || 1);
    const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length || 1));
    return { name: c.name, count: v.length, missing: c.values.length - v.length, mean, std, min: s[0] ?? 0, q25: quantile(s, 0.25), q50: quantile(s, 0.5), q75: quantile(s, 0.75), max: s[s.length - 1] ?? 0 };
  });
}
export function boxStats(v: number[]) {
  const s = [...v].sort((a, b) => a - b);
  const q1 = quantile(s, 0.25), med = quantile(s, 0.5), q3 = quantile(s, 0.75), iqr = q3 - q1, lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;
  const inLo = s.filter((x) => x >= lo), inHi = s.filter((x) => x <= hi);
  return { min: s[0] ?? 0, q1, med, q3, max: s[s.length - 1] ?? 0, whiskLo: inLo[0] ?? s[0] ?? 0, whiskHi: inHi[inHi.length - 1] ?? s[s.length - 1] ?? 0, outliers: s.filter((x) => x < lo || x > hi) };
}

// ── preprocessing with per-step snapshots (for run + before/after + animation) ──
export interface Snapshot { op: string; method: string; cols: string[]; colNames: string[]; changedCols: string[]; sample: string[][]; nRows: number; nCols: number; }
export function applyStepsSnapshots(ds: Dataset, steps: PrepStep[], sampleK = 6): { snapshots: Snapshot[]; finalColumns: Column[] } {
  let working: Column[] = ds.columns.map((c) => ({ name: c.name, type: c.type, values: [...c.values] }));
  const K = Math.min(sampleK, ds.nrows);
  const disp = (cols: Column[]): string[][] => Array.from({ length: K }, (_, r) => cols.map((c) => { const v = c.values[r]; return v == null ? "∅" : (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v)); }));
  const snaps: Snapshot[] = [{ op: "raw", method: "source data", cols: [], colNames: working.map((c) => c.name), changedCols: [], sample: disp(working), nRows: ds.nrows, nCols: working.length }];
  for (const s of steps) {
    const changed: string[] = [];
    const asNum = (col: Column) => col.values.map((v) => (v == null ? null : Number(v)));
    if (s.op === "Impute missing") {
      for (const name of s.cols) { const col = working.find((c) => c.name === name); if (!col) continue;
        if (col.type === "num") col.values = imputeNumCol(asNum(col), s.method);
        else { const present = col.values.filter((v): v is string => v != null); const fill = s.method === "Constant" ? "missing" : modeStr(present); col.values = col.values.map((v) => (v == null ? fill : v)); }
        changed.push(name); }
    } else if (s.op === "Scale / normalize") {
      for (const name of s.cols) { const col = working.find((c) => c.name === name); if (!col || col.type !== "num") continue; col.values = scaleNumCol(asNum(col), s.method); changed.push(name); }
    } else if (s.op === "Handle outliers") {
      for (const name of s.cols) { const col = working.find((c) => c.name === name); if (!col || col.type !== "num") continue; col.values = outlierNumCol(asNum(col), s.method); changed.push(name); }
    } else if (s.op === "Transform") {
      for (const name of s.cols) { const col = working.find((c) => c.name === name); if (!col || col.type !== "num") continue; col.values = transformNumCol(asNum(col), s.method); changed.push(name); }
    } else if (s.op === "Bin / discretize") {
      for (const name of s.cols) { const col = working.find((c) => c.name === name); if (!col || col.type !== "num") continue; col.values = binNumCol(asNum(col), s.method); changed.push(name); }
    } else if (s.op === "Encode categorical") {
      for (const name of s.cols) { const idx = working.findIndex((c) => c.name === name); if (idx < 0) continue; const col = working[idx]; const vals = col.values.map((v) => (v == null ? "missing" : String(v)));
        if (s.method === "One-Hot") { const cats = Array.from(new Set(vals)); const nc: Column[] = cats.map((cat) => ({ name: `${name}=${cat}`, type: "num", values: vals.map((v) => (v === cat ? 1 : 0)) })); working.splice(idx, 1, ...nc); changed.push(...nc.map((c) => c.name)); }
        else if (s.method === "Binary") { const cats = Array.from(new Set(vals)); const map = new Map(cats.map((c, i) => [c, i])); const bits = Math.max(1, Math.ceil(Math.log2(cats.length || 1))); const nc: Column[] = Array.from({ length: bits }, (_, b) => ({ name: `${name}_b${b}`, type: "num", values: vals.map((v) => (map.get(v)! >> b) & 1) })); working.splice(idx, 1, ...nc); changed.push(...nc.map((c) => c.name)); }
        else if (s.method === "Frequency") { const freq = new Map<string, number>(); vals.forEach((v) => freq.set(v, (freq.get(v) || 0) + 1)); col.values = vals.map((v) => freq.get(v)! / vals.length); col.type = "num"; changed.push(name); }
        else if (s.method === "Count") { const freq = new Map<string, number>(); vals.forEach((v) => freq.set(v, (freq.get(v) || 0) + 1)); col.values = vals.map((v) => freq.get(v)!); col.type = "num"; changed.push(name); }
        else { const cats = Array.from(new Set(vals)); const map = new Map(cats.map((c, i) => [c, i])); col.values = vals.map((v) => map.get(v)!); col.type = "num"; changed.push(name); } // Ordinal
      }
    } else if (s.op === "Drop column") {
      working = working.filter((c) => !s.cols.includes(c.name)); changed.push(...s.cols);
    }
    snaps.push({ op: s.op, method: s.method, cols: s.cols, colNames: working.map((c) => c.name), changedCols: changed, sample: disp(working), nRows: working[0]?.values.length || ds.nrows, nCols: working.length });
  }
  return { snapshots: snaps, finalColumns: working };
}
