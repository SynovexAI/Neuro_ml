// Real in-browser ML: CSV parsing, EDA stats, preprocessing, training, metrics.
// Models are implemented from scratch (logistic/linear regression via gradient
// descent, KNN) so training genuinely runs on the data — no server, no mocks.

export type ColType = "num" | "cat";
export interface Column { name: string; type: ColType; values: (number | string | null)[]; }
export interface Dataset { columns: Column[]; nrows: number; }
export type Task = "classification" | "regression";
export interface PrepStep { op: string; cols: string[]; method: string; }

// ── parsing ──
export function parseCSV(text: string, opts: { header?: boolean } = {}): Dataset {
  const hasHeader = opts.header !== false; // default: first row is a header
  const clean = text.replace(/\r\n?/g, "\n").trim();
  const line0 = clean.split("\n")[0] || "";
  const nSemi = (line0.match(/;/g)?.length || 0), nComma = (line0.match(/,/g)?.length || 0), nTab = (line0.match(/\t/g)?.length || 0);
  // delimiter: prefer explicit separators, fall back to whitespace (many UCI .data files)
  let delim: string;
  if (nSemi > nComma && nSemi > 0) delim = ";";
  else if (nComma > 0) delim = ",";
  else if (nTab > 0) delim = "\t";
  else if (line0.trim().split(/\s+/).length > 1) delim = " "; // whitespace sentinel
  else delim = ",";
  const rows = clean.split("\n").map((line) => (delim === " " ? line.trim().split(/\s+/) : splitLine(line, delim)));
  const ncol = rows.reduce((m, r) => Math.max(m, r.length), 0);
  const header = hasHeader ? rows[0] : Array.from({ length: ncol }, (_, i) => `col${i + 1}`);
  const body = (hasHeader ? rows.slice(1) : rows).filter((r) => r.length > 1 || (r.length === 1 && r[0] !== ""));
  const cols: Column[] = Array.from({ length: ncol }, (_, ci) => {
    const name = (header[ci] ?? "").trim() || `col${ci + 1}`;
    const raw = body.map((r) => (r[ci] ?? "").trim());
    const nonEmpty = raw.filter((v) => v !== "");
    const numeric = nonEmpty.length > 0 && nonEmpty.every((v) => v !== "" && !isNaN(Number(v)));
    const type: ColType = numeric ? "num" : "cat";
    const values = raw.map((v) => (v === "" ? null : (type === "num" ? Number(v) : v)));
    return { name, type, values };
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
    const min = v.length ? v.reduce((a, b) => Math.min(a, b), v[0]) : 0;
    const max = v.length ? v.reduce((a, b) => Math.max(a, b), v[0]) : 0;
    return { type: "num", count: v.length, missing, mean, std, min, max };
  }
  const v = col.values.filter((x) => x != null) as string[];
  const counts = new Map<string, number>();
  v.forEach((x) => counts.set(String(x), (counts.get(String(x)) || 0) + 1));
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return { type: "cat", count: v.length, missing, unique: counts.size, top };
}
export function histogram(v: number[], bins = 12): { edges: number[]; counts: number[] } {
  if (!v.length) return { edges: [], counts: [] };
  const min = v.reduce((a, b) => Math.min(a, b), v[0]);
  const max = v.reduce((a, b) => Math.max(a, b), v[0]);
  const span = (max - min) || 1;
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
  if (method === "Min") return present.reduce((a, b) => Math.min(a, b), present[0]);
  if (method === "Max") return present.reduce((a, b) => Math.max(a, b), present[0]);
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
  if (method === "KNN impute") {
    const out = vals.map((v) => v == null ? null : Number(v)) as (number | null)[];
    const valid: number[] = [];
    for (let i = 0; i < out.length; i++) if (out[i] != null) valid.push(i);
    if (!valid.length) return out.map(() => 0);
    for (let i = 0; i < out.length; i++) {
      if (out[i] != null) continue;
      let pos = valid.findIndex((j) => j >= i);
      if (pos < 0) pos = valid.length;
      const candidates: number[] = [];
      for (let d = 0; candidates.length < 5 && (pos - d >= 0 || pos + d < valid.length); d++) {
        const left = pos - d - 1, right = pos + d;
        if (left >= 0) candidates.push(valid[left]);
        if (right < valid.length && candidates.length < 5) candidates.push(valid[right]);
      }
      let sum = 0; for (const j of candidates) sum += Number(out[j]);
      out[i] = candidates.length ? sum / candidates.length : 0;
    }
    return out.map((v) => Number(v));
  }
  if (method === "Iterative impute") { let fill=present.reduce((a,b)=>a+b,0)/(present.length||1); for(let it=0;it<3;it++){ const out=vals.map(v=>v==null?fill:Number(v)); fill=out.reduce((a,b)=>a+b,0)/out.length; } return vals.map(v=>v==null?fill:Number(v)); }
  const fill = imputeFill(present, method);
  return vals.map((v) => (v == null ? fill : Number(v)));
}
export function scaleNumCol(vals: (number | null)[], method: string): (number | null)[] {
  if (method === "None") return vals;
  const present = vals.filter((v): v is number => v != null);
  if (!present.length) return vals;
  if (method === "MinMaxScaler") { const mn = present.reduce((a, b) => Math.min(a, b), present[0]), mx = present.reduce((a, b) => Math.max(a, b), present[0]), sp = (mx - mn) || 1; return vals.map((v) => (v == null ? null : (Number(v) - mn) / sp)); }
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
      case "Yeo-Johnson": { const l=0.5; return x>=0?(((x+1)**l)-1)/l: -(((((-x)+1)**(2-l))-1)/(2-l)); }
      case "Box-Cox": { const z=Math.max(x,1e-9), l=0.5; return (((z**l)-1)/l); }
      case "Rank": return x;
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
  const mn = present.reduce((a, b) => Math.min(a, b), present[0]), mx = present.reduce((a, b) => Math.max(a, b), present[0]), sp = (mx - mn) || 1; // Equal-width
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
  // Additional feature engineering exposed through the preprocessing step.
  for (const st of steps.filter(s=>["Polynomial features","Interaction features","Missing indicators","Date/time features","Text features"].includes(s.op))) {
    if(st.op==="Polynomial features") for(const name of st.cols){const c=ds.columns.find(x=>x.name===name); if(!c||c.type!=="num") continue; const v=c.values.map(x=>x==null?0:Number(x)); if(st.method==="degree 2"||st.method==="degree 3") outCols.push({name:`${name}²`,values:v.map(x=>x*x)}); if(st.method==="degree 3") outCols.push({name:`${name}³`,values:v.map(x=>x*x*x)});}
    if(st.op==="Interaction features" && st.cols.length>1){for(let a=0;a<st.cols.length;a++) for(let b=a+1;b<st.cols.length;b++){const ca=ds.columns.find(x=>x.name===st.cols[a]),cb=ds.columns.find(x=>x.name===st.cols[b]); if(!ca||!cb||ca.type!=="num"||cb.type!=="num") continue; outCols.push({name:`${st.cols[a]}×${st.cols[b]}`,values:Array.from({length:ds.nrows},(_,i)=>(ca.values[i]==null?0:Number(ca.values[i]))*(cb.values[i]==null?0:Number(cb.values[i])))});}}
    if(st.op==="Missing indicators") for(const name of st.cols){const c=ds.columns.find(x=>x.name===name); if(c) outCols.push({name:`${name}__missing`,values:c.values.map(v=>v==null?1:0)});}
    if(st.op==="Date/time features") for(const name of st.cols){const c=ds.columns.find(x=>x.name===name); if(!c) continue; const vals=c.values.map(v=>v==null?null:new Date(String(v))); const method=st.method; const nums=vals.map(d=>{if(!d||Number.isNaN(d.getTime())) return 0; if(method==="Year") return d.getFullYear(); if(method==="Month") return d.getMonth()+1; if(method==="Day") return d.getDate(); if(method==="Day of week") return d.getDay(); if(method==="Quarter") return Math.floor(d.getMonth()/3)+1; return d.getDay()===0||d.getDay()===6?1:0;}); outCols.push({name:`${name}_${method.replace(/ /g,"_")}`,values:nums});}
    if(st.op==="Text features") for(const name of st.cols){const c=ds.columns.find(x=>x.name===name); if(!c) continue; const vals=c.values.map(v=>String(v??"")); const nums=vals.map(v=>st.method==="Word count"?v.trim().split(/\s+/).filter(Boolean).length:st.method==="Character count"?v.length:new Set(v.toLowerCase().trim().split(/\s+/).filter(Boolean)).size); outCols.push({name:`${name}_${st.method.replace(/ /g,"_")}`,values:nums});}
  }
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
      // Build the vocabulary/statistics from rows with a known target. This
      // keeps prediction-time unknown categories from changing the feature
      // dimensionality when a new row is appended to the dataset.
      const knownRows = filled.filter((_, i) => targetName ? ds.columns.find((c) => c.name === targetName)?.values[i] != null : true);
      const cats = Array.from(new Set(knownRows.length ? knownRows : filled));
      if (enc === "Ordinal") {
        const map = new Map(cats.map((c, i) => [c, i]));
        outCols.push({ name, values: filled.map((v) => map.has(v) ? map.get(v)! : -1) });
      } else if (enc === "Frequency") {
        const f = new Map<string, number>(); knownRows.forEach((v) => f.set(v, (f.get(v) || 0) + 1));
        outCols.push({ name, values: filled.map((v) => f.get(v) ? f.get(v)! / knownRows.length : 0) });
      } else if (enc === "Count") {
        const f = new Map<string, number>(); knownRows.forEach((v) => f.set(v, (f.get(v) || 0) + 1));
        outCols.push({ name, values: filled.map((v) => f.get(v) || 0) });
      } else if (enc === "Label") { const map=new Map(cats.map((c,i)=>[c,i])); outCols.push({ name, values: filled.map(v=>map.get(v)??-1) });
      } else if (enc === "Binary") {
        const map = new Map(cats.map((c, i) => [c, i])); const bits = Math.max(1, Math.ceil(Math.log2(cats.length || 1)));
        for (let b = 0; b < bits; b++) outCols.push({ name: `${name}_b${b}`, values: filled.map((v) => map.has(v) ? ((map.get(v)! >> b) & 1) : 0) });
      } else {
        // Prevent a high-cardinality categorical column from exploding the
        // browser-side feature matrix. Keep the 100 most frequent categories
        // and map the remainder to a stable __other bucket.
        const freq = new Map<string, number>();
        knownRows.forEach(v => freq.set(v, (freq.get(v) || 0) + 1));
        const kept = [...cats].sort((a,b)=>(freq.get(b)||0)-(freq.get(a)||0)).slice(0, 100);
        const keptSet = new Set(kept);
        const values = filled.map(v => keptSet.has(v) ? v : "__other");
        const vocab = keptSet.size < cats.length ? [...kept, "__other"] : kept;
        for (const cat of vocab) outCols.push({ name: `${name}=${cat}`, values: values.map(v => (v === cat ? 1 : 0)) });
      } // One-Hot
    }
  }
  const X = Array.from({ length: ds.nrows }, (_, r) => outCols.map((c) => c.values[r]));
  const featureNames = outCols.map((c) => c.name);
  const target = targetName ? ds.columns.find((c) => c.name === targetName) : undefined;
  // Unsupervised workflows (clustering) do not require a target column.
  if (!targetName) return { X, y: new Array(ds.nrows).fill(0), featureNames };
  if (!target) throw new Error(`Target column "${targetName}" not found.`);
  let y: number[]; let classes: string[] | undefined;
  if (task === "classification") {
    // Missing targets are not a real class. Keep their row encoded as 0 for
    // inference compatibility, but never include "?" in the class vocabulary.
    const tv = target.values.map((v) => (v == null || String(v).trim() === "" ? null : String(v)));
    classes = Array.from(new Set(tv.filter((v): v is string => v != null)));
    const cmap = new Map(classes.map((c, i) => [c, i]));
    y = tv.map((v) => (v == null ? 0 : (cmap.get(v) ?? 0)));
  } else {
    y = target.values.map((v) => (v == null || !Number.isFinite(Number(v)) ? 0 : Number(v)));
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
  | { leaf: true; value: number; n: number; dist?: number[] }
  | { leaf: false; feat: number; thr: number; n: number; left: TreeNode; right: TreeNode };
export type Model =
  | { kind: "logreg"; W: number[][]; classes: number; loss?: number[]; mu: number[]; sd: number[] }
  | { kind: "linear"; w: number[]; loss?: number[]; mu: number[]; sd: number[] }
  | { kind: "knn"; X: number[][]; y: number[]; k: number; weights: string; task: Task; classes: number }
  | { kind: "gnb"; means: number[][]; vars: number[][]; priors: number[]; classes: number }
  | { kind: "tree"; root: TreeNode; task: Task; classes: number; importance: number[] }
  | { kind: "forest"; trees: TreeNode[]; task: Task; classes: number; importance: number[]; nTrees: number }
  | { kind: "svm"; w: number[]; b: number; mu: number[]; sd: number[]; classes: number }
  | { kind: "svr"; w: number[]; b: number; mu: number[]; sd: number[]; epsilon: number }
  | { kind: "boost"; trees: TreeNode[]; weights: number[]; base: number; task: Task; classes: number; importance: number[] };

function standardizeForGD(X: number[][]): { Xs: number[][]; mu: number[]; sd: number[] } {
  const d = X[0]?.length || 0;
  const mu = Array.from({ length: d }, (_, j) => {
    const v = X.map(r => Number(r[j])).filter(Number.isFinite);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
  });
  const sd = Array.from({ length: d }, (_, j) => {
    const v = X.map(r => Number(r[j])).filter(Number.isFinite);
    const m = mu[j];
    const q = v.length ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) : 1;
    return Number.isFinite(q) && q > 1e-12 ? q : 1;
  });
  return { Xs: X.map(r => r.map((v, j) => (Number(v) - mu[j]) / sd[j])), mu, sd };
}
function applyGDScale(X: number[][], mu: number[], sd: number[]): number[][] {
  return X.map(r => r.map((v, j) => (Number(v) - (mu[j] ?? 0)) / (sd[j] || 1)));
}

export function trainLogReg(X: number[][], y: number[], nClasses: number, p: { lr: number; epochs: number; l2: number }): Model {
  const scaled = standardizeForGD(X);
  const Xs = scaled.Xs;
  const n = Xs.length, d = Xs[0]?.length || 0;
  const Xb = Xs.map((r) => [1, ...r]);
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
  return { kind: "logreg", W, classes: nClasses, loss, mu: scaled.mu, sd: scaled.sd };
}
export function trainSVM(X: number[][], y: number[], nClasses: number, p: { lr: number; epochs: number; C: number }): Model {
  const scaled=standardizeForGD(X), Xs=scaled.Xs, n=Xs.length, d=Xs[0]?.length||0;
  const lr=p.lr, C=p.C;
  // One-vs-rest linear hinge-loss SVM; for multiclass, train a shared score per class via softmax-like margins.
  const W=Array.from({length:nClasses},()=>new Array(d).fill(0)), B=new Array(nClasses).fill(0);
  for(let ep=0;ep<p.epochs;ep++){ for(let i=0;i<n;i++){ for(let k=0;k<nClasses;k++){ const sign=y[i]===k?1:-1; const margin=sign*(dot(W[k],Xs[i])+B[k]); if(margin<1){ for(let j=0;j<d;j++) W[k][j]+=lr*(C*sign*Xs[i][j]-0.001*W[k][j]); B[k]+=lr*C*sign; } } } }
  return {kind:"svm",w:W.flat(),b:0,mu:scaled.mu,sd:scaled.sd,classes:nClasses};
}

export function trainSVR(X:number[][],y:number[],p:{lr:number;epochs:number;C:number;epsilon:number}):Model{
  const scaled=standardizeForGD(X), Xs=scaled.Xs,n=Xs.length,d=Xs[0]?.length||0,w=new Array(d).fill(0); let b=mean(y);
  for(let ep=0;ep<p.epochs;ep++){ const gw=new Array(d).fill(0); let gb=0; for(let i=0;i<n;i++){const e=dot(w,Xs[i])+b-y[i],a=Math.abs(e)-p.epsilon;if(a>0){const sign=e>0?1:-1;for(let j=0;j<d;j++)gw[j]+=sign*Xs[i][j];gb+=sign;}} for(let j=0;j<d;j++)w[j]-=p.lr*(gw[j]/n+0.001*w[j]); b-=p.lr*gb/n; }
  return {kind:"svr",w,b,mu:scaled.mu,sd:scaled.sd,epsilon:p.epsilon};
}

export function trainGradientBoosting(X:number[][],y:number[],task:Task,nClasses:number,p:{nTrees:number;maxDepth:number;learningRate:number}):Model{
  const n=X.length,d=X[0]?.length||0,imp=new Array(d).fill(0),trees:TreeNode[]=[],weights:number[]=[],base=task==="regression"?mean(y):0;
  let pred=new Array(n).fill(base);
  for(let t=0;t<p.nTrees;t++){ const residual=task==="regression"?y.map((v,i)=>v-pred[i]):y.map((v,i)=>v===pred[i]?0: (v===0?1:-1)); const yy=task==="regression"?residual:residual.map(v=>v>0?1:0); const tr=trainTree(X,yy,task==="regression"?"regression":"classification",task==="regression"?1:2,{maxDepth:Math.max(1,p.maxDepth),minSplit:2}); if(tr.kind!=="tree") break; trees.push(tr.root); weights.push(p.learningRate); const step=predict(tr,X); pred=pred.map((v,i)=>v+p.learningRate*step[i]); }
  return {kind:"boost",trees,weights,base,task,classes:nClasses,importance:imp};
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
export function trainLinear(X: number[][], y: number[], p: { lr: number; epochs: number; alpha: number; fitIntercept?: boolean }): Model {
  const scaled = standardizeForGD(X);
  const Xs = scaled.Xs;
  const n = Xs.length, d = Xs[0]?.length || 0;
  const Xb = Xs.map((r) => [p.fitIntercept === false ? 0 : 1, ...r]);
  const w = new Array(d + 1).fill(0);
  const loss: number[] = []; const rec = Math.max(1, Math.floor(p.epochs / 50));
  for (let ep = 0; ep < p.epochs; ep++) {
    const grad = new Array(d + 1).fill(0);
    for (let i = 0; i < n; i++) { const pred = dot(w, Xb[i]); const err = pred - y[i]; for (let j = 0; j <= d; j++) grad[j] += err * Xb[i][j]; }
    for (let j = 0; j <= d; j++) { let g = grad[j] / n; if (j > 0) g += p.alpha * w[j]; w[j] -= p.lr * g; }
    if (ep % rec === 0 || ep === p.epochs - 1) { let L = 0; for (let i = 0; i < n; i++) L += (dot(w, Xb[i]) - y[i]) ** 2; loss.push(L / n); }
  }
  return { kind: "linear", w, loss, mu: scaled.mu, sd: scaled.sd };
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
export function variance(y: number[]): number { if (!y.length) return 0; const m = y.reduce((a, b) => a + b, 0) / y.length; return y.reduce((a, b) => a + (b - m) ** 2, 0) / y.length; }
function leafValue(y: number[], task: Task, nClasses: number): number { if (task === "regression") return y.reduce((a, b) => a + b, 0) / (y.length || 1); const c = new Array(nClasses).fill(0); y.forEach((v) => c[v]++); return c.indexOf(Math.max(...c)); }
interface TreeOpts { maxDepth: number; minSplit: number; maxFeatures: number; }
function leafDist(y: number[], nClasses: number): number[] { const c = new Array(nClasses).fill(0); y.forEach((v) => c[v]++); const n = y.length || 1; return c.map((x) => x / n); }
function buildTree(X: number[][], y: number[], task: Task, nClasses: number, opts: TreeOpts, imp: number[], depth: number, rnd: () => number): TreeNode {
  const n = y.length;
  const mkLeaf = (yy: number[]): TreeNode => (task === "regression" ? { leaf: true, value: leafValue(yy, task, nClasses), n: yy.length } : { leaf: true, value: leafValue(yy, task, nClasses), n: yy.length, dist: leafDist(yy, nClasses) });
  const nodeImp = task === "regression" ? variance(y) : gini(y, nClasses);
  const pure = task === "regression" ? nodeImp < 1e-9 : new Set(y).size <= 1;
  if (depth >= opts.maxDepth || n < opts.minSplit || pure) return mkLeaf(y);
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
  if (best.feat < 0) return mkLeaf(y);
  imp[best.feat] += best.gain * n;
  const Xl: number[][] = [], yl: number[] = [], Xr: number[][] = [], yr: number[] = [];
  for (let i = 0; i < n; i++) { if (X[i][best.feat] <= best.thr) { Xl.push(X[i]); yl.push(y[i]); } else { Xr.push(X[i]); yr.push(y[i]); } }
  return { leaf: false, feat: best.feat, thr: best.thr, n, left: buildTree(Xl, yl, task, nClasses, opts, imp, depth + 1, rnd), right: buildTree(Xr, yr, task, nClasses, opts, imp, depth + 1, rnd) };
}
function predictTreeOne(node: TreeNode, x: number[]): number {
  let cur: TreeNode = node;
  while (!cur.leaf) {
    const branch = cur as Extract<TreeNode, { leaf: false }>;
    cur = x[branch.feat] <= branch.thr ? branch.left : branch.right;
  }
  return cur.value;
}
function treeProbaOne(node: TreeNode, x: number[], nClasses: number): number[] {
  let cur: TreeNode = node;
  while (!cur.leaf) {
    const branch = cur as Extract<TreeNode, { leaf: false }>;
    cur = x[branch.feat] <= branch.thr ? branch.left : branch.right;
  }
  if (cur.dist) return cur.dist;
  const one = new Array(nClasses).fill(0); one[cur.value] = 1; return one;
}
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
export function treeDepth(node: TreeNode): number {
  if (node.leaf) return 1;
  const branch = node as Extract<TreeNode, { leaf: false }>;
  return 1 + Math.max(treeDepth(branch.left), treeDepth(branch.right));
}
export function countNodes(node: TreeNode): number {
  if (node.leaf) return 1;
  const branch = node as Extract<TreeNode, { leaf: false }>;
  return 1 + countNodes(branch.left) + countNodes(branch.right);
}

export function predict(m: Model, X: number[][]): number[] {
  if (m.kind === "logreg") { const Xb = applyGDScale(X, m.mu, m.sd).map((r) => [1, ...r]); return Xb.map((r) => { const pr = softmax(m.W.map((w) => dot(w, r))); return pr.indexOf(Math.max(...pr)); }); }
  if (m.kind === "linear") { const Xb = applyGDScale(X, m.mu, m.sd).map((r) => [1, ...r]); return Xb.map((r) => dot(m.w, r)); }
  if (m.kind === "gnb") return X.map((x) => predictGNBOne(m, x));
  if (m.kind === "tree") return X.map((x) => predictTreeOne(m.root, x));
  if (m.kind === "forest") return X.map((x) => predictForestOne(m, x));
  if (m.kind === "svm") { const W=Array.from({length:m.classes},(_,k)=>m.w.slice(k*(m.w.length/m.classes),(k+1)*(m.w.length/m.classes))); const Xs=applyGDScale(X,m.mu,m.sd); return Xs.map(x=>{const scores=W.map(w=>dot(w,x)); return scores.indexOf(Math.max(...scores));}); }
  if (m.kind === "svr") { const Xs=applyGDScale(X,m.mu,m.sd); return Xs.map(x=>dot(m.w,x)+m.b); }
  if (m.kind === "boost") { return X.map(x=>{let s=m.base; for(let i=0;i<m.trees.length;i++) s+=m.weights[i]*predictTreeOne(m.trees[i],x); return m.task==="classification"?Math.max(0,Math.min(m.classes-1,Math.round(s))):s;}); }
  return X.map((x) => knnPredictOne(m, x));
}
// Per-class probabilities (predict_proba) for classifiers — used for ROC/PR and threshold tuning.
export function predictProba(m: Model, X: number[][]): number[][] {
  if (m.kind === "logreg") { const Xb = applyGDScale(X, m.mu, m.sd).map((r) => [1, ...r]); return Xb.map((r) => softmax(m.W.map((w) => dot(w, r)))); }
  if (m.kind === "gnb") return X.map((x) => { const lls: number[] = []; for (let k = 0; k < m.classes; k++) { let ll = Math.log(m.priors[k] || 1e-9); for (let j = 0; j < x.length; j++) { const v = m.vars[k][j]; ll += -0.5 * Math.log(2 * Math.PI * v) - ((x[j] - m.means[k][j]) ** 2) / (2 * v); } lls.push(ll); } const mx = Math.max(...lls); const ex = lls.map((l) => Math.exp(l - mx)); const s = ex.reduce((a, b) => a + b, 0) || 1; return ex.map((e) => e / s); });
  if (m.kind === "knn") return X.map((x) => { const dists = m.X.map((row, i) => ({ d: Math.sqrt(row.reduce((a, v, j) => a + (v - x[j]) ** 2, 0)), y: m.y[i] })); dists.sort((a, b) => a.d - b.d); const near = dists.slice(0, m.k); const votes = new Array(m.classes).fill(0); near.forEach((p) => { votes[p.y] += m.weights === "distance" ? 1 / (p.d + 1e-6) : 1; }); const s = votes.reduce((a: number, b: number) => a + b, 0) || 1; return votes.map((v: number) => v / s); });
  if (m.kind === "tree") return X.map((x) => treeProbaOne(m.root, x, m.classes));
  if (m.kind === "forest") return X.map((x) => { const acc = new Array(m.classes).fill(0); for (const t of m.trees) { const pr = treeProbaOne(t, x, m.classes); for (let k = 0; k < m.classes; k++) acc[k] += pr[k]; } return acc.map((v) => v / (m.trees.length || 1)); });
  if (m.kind === "svm") { const Xs=applyGDScale(X,m.mu,m.sd), d=m.w.length/m.classes; return Xs.map(x=>{const z=Array.from({length:m.classes},(_,k)=>dot(m.w.slice(k*d,(k+1)*d),x)); return softmax(z);}); }
  if (m.kind === "boost" && m.task === "classification") return X.map(x=>{const scores=new Array(m.classes).fill(0); let s=m.base; for(let i=0;i<m.trees.length;i++) s+=m.weights[i]*predictTreeOne(m.trees[i],x); scores[Math.max(0,Math.min(m.classes-1,Math.round(s)))] = 1; return scores;});
  return X.map(() => []); // linear regression has no class probabilities
}

// ── threshold / ROC / PR (one-vs-rest for a chosen positive class) ──
export interface RocPoint { fpr: number; tpr: number; }
export function rocCurve(yTrue01: number[], scores: number[]): { points: RocPoint[]; auc: number } {
  const pairs = scores.map((s, i) => ({ s, y: yTrue01[i] })).sort((a, b) => b.s - a.s);
  const P = yTrue01.reduce((a, y) => a + y, 0), N = yTrue01.length - P;
  const points: RocPoint[] = [{ fpr: 0, tpr: 0 }]; let tp = 0, fp = 0, auc = 0, pf = 0, pt = 0;
  for (const p of pairs) { if (p.y === 1) tp++; else fp++; const tpr = tp / (P || 1), fpr = fp / (N || 1); points.push({ fpr, tpr }); auc += (fpr - pf) * (tpr + pt) / 2; pf = fpr; pt = tpr; }
  return { points, auc };
}
export interface PrPoint { recall: number; precision: number; }
export function prCurve(yTrue01: number[], scores: number[]): { points: PrPoint[]; ap: number } {
  const pairs = scores.map((s, i) => ({ s, y: yTrue01[i] })).sort((a, b) => b.s - a.s);
  const P = yTrue01.reduce((a, y) => a + y, 0);
  const points: PrPoint[] = []; let tp = 0, fp = 0, ap = 0, prevRec = 0;
  for (const p of pairs) { if (p.y === 1) tp++; else fp++; const recall = tp / (P || 1), precision = tp / (tp + fp || 1); points.push({ recall, precision }); ap += (recall - prevRec) * precision; prevRec = recall; }
  return { points, ap };
}
export function metricsAtThreshold(yTrue01: number[], scores: number[], thr: number): { tp: number; fp: number; fn: number; tn: number; precision: number; recall: number; f1: number; accuracy: number } {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < yTrue01.length; i++) { const pred = scores[i] >= thr ? 1 : 0; if (yTrue01[i] === 1) { if (pred) tp++; else fn++; } else { if (pred) fp++; else tn++; } }
  const precision = tp / (tp + fp || 1), recall = tp / (tp + fn || 1);
  return { tp, fp, fn, tn, precision, recall, f1: 2 * precision * recall / (precision + recall || 1), accuracy: (tp + tn) / (yTrue01.length || 1) };
}
export function featureImportance(m: Model, names: string[]): { name: string; w: number }[] | null {
  if (m.kind === "logreg") { const imp = names.map((_, j) => m.W.reduce((a, w) => a + Math.abs(w[j + 1]), 0)); const mx = Math.max(...imp, 1e-9); return names.map((n, j) => ({ name: n, w: imp[j] / mx })).sort((a, b) => b.w - a.w).slice(0, 8); }
  if (m.kind === "linear") { const imp = names.map((_, j) => Math.abs(m.w[j + 1])); const mx = Math.max(...imp, 1e-9); return names.map((n, j) => ({ name: n, w: imp[j] / mx })).sort((a, b) => b.w - a.w).slice(0, 8); }
  if (m.kind === "svm") { const d=m.w.length/m.classes; const imp=names.map((_,j)=>m.w.reduce((a,_,k)=>a+Math.abs(m.w[k*d+j]),0)); const mx=Math.max(...imp,1e-9); return names.map((n,j)=>({name:n,w:imp[j]/mx})).sort((a,b)=>b.w-a.w).slice(0,8); }
  if (m.kind === "tree" || m.kind === "forest" || m.kind === "boost") { const mx = Math.max(...m.importance, 1e-9); return names.map((n, j) => ({ name: n, w: (m.importance[j] || 0) / mx })).sort((a, b) => b.w - a.w).slice(0, 8); }
  return null;
}

// ── unsupervised clustering ──
export type ClusterAlgorithm = "KMeans" | "DBSCAN" | "Agglomerative";
export interface ClusterResult { algorithm: ClusterAlgorithm; labels: number[]; centers?: number[][]; k?: number; epsilon?: number; minSamples?: number; silhouette: number; sizes: { cluster: number; size: number }[]; iterations?: number; }
function rowDistance(a: number[], b: number[]): number { let s=0; for(let j=0;j<a.length;j++){const d=(a[j]||0)-(b[j]||0); s+=d*d;} return Math.sqrt(s); }
function safeClusterMatrix(X: number[][]): number[][] { if(!X.length||!X[0]?.length) throw new Error("Clustering requires at least one numeric feature."); const d=X[0].length; if(X.some(r=>r.length!==d||r.some(v=>!Number.isFinite(v)))) throw new Error("Clustering requires finite numeric feature values. Run preprocessing first."); return X.map(r=>r.map(Number)); }
function initKMeansCenters(X:number[][],k:number):number[][] { const centers=[X[0].slice()]; while(centers.length<k){let best=0,bestD=-Infinity; for(let i=0;i<X.length;i++){const d=Math.min(...centers.map(c=>rowDistance(X[i],c))); if(d>bestD){bestD=d;best=i;}} centers.push(X[best].slice());} return centers; }
export function kMeans(X0:number[][],k=3,maxIter=100):ClusterResult { const X=safeClusterMatrix(X0),kk=Math.max(2,Math.min(Math.floor(k),X.length)); let centers=initKMeansCenters(X,kk); const labels=new Array(X.length).fill(-1); let iterations=0; for(let it=0;it<Math.max(1,maxIter);it++){iterations=it+1;let changed=false; for(let i=0;i<X.length;i++){let bi=0,bd=Infinity;for(let c=0;c<centers.length;c++){const d=rowDistance(X[i],centers[c]);if(d<bd){bd=d;bi=c;}}if(labels[i]!==bi){labels[i]=bi;changed=true;}} const next=Array.from({length:kk},()=>new Array(X[0].length).fill(0));const counts=new Array(kk).fill(0);X.forEach((r,i)=>{const c=labels[i];counts[c]++;r.forEach((v,j)=>next[c][j]+=v);});for(let c=0;c<kk;c++){if(!counts[c]){let far=0,farD=-Infinity;for(let i=0;i<X.length;i++){const d=Math.min(...centers.map(cc=>rowDistance(X[i],cc)));if(d>farD){farD=d;far=i;}}next[c]=X[far].slice();}else next[c]=next[c].map(v=>v/counts[c]);} centers=next;if(!changed)break;} return {algorithm:"KMeans",labels,centers,k:kk,silhouette:silhouetteScore(X,labels),sizes:centers.map((_,c)=>({cluster:c,size:labels.filter(x=>x===c).length})),iterations}; }
export function dbscan(X0:number[][],epsilon=0.5,minSamples=5):ClusterResult { const X=safeClusterMatrix(X0),n=X.length,eps=Math.max(1e-9,epsilon),minS=Math.max(2,Math.floor(minSamples));const labels=new Array(n).fill(-1),visited=new Array(n).fill(false);const region=(i:number)=>{const out:number[]=[];for(let j=0;j<n;j++)if(rowDistance(X[i],X[j])<=eps)out.push(j);return out;};let cluster=0;for(let i=0;i<n;i++){if(visited[i])continue;visited[i]=true;const neigh=region(i);if(neigh.length<minS){labels[i]=-1;continue;}labels[i]=cluster;const queue=[...neigh];for(let q=0;q<queue.length;q++){const j=queue[q];if(!visited[j]){visited[j]=true;const n2=region(j);if(n2.length>=minS)for(const z of n2)if(!queue.includes(z))queue.push(z);}if(labels[j]===-1)labels[j]=cluster;}cluster++;}const unique=[...new Set(labels)].sort((a,b)=>a-b);return {algorithm:"DBSCAN",labels,epsilon:eps,minSamples:minS,silhouette:silhouetteScore(X,labels),sizes:unique.map(c=>({cluster:c,size:labels.filter(x=>x===c).length}))}; }
export function agglomerativeClustering(X0:number[][],k=3):ClusterResult { const X=safeClusterMatrix(X0),kk=Math.max(2,Math.min(Math.floor(k),X.length));const groups=X.map((_,i)=>[i]);const distGroups=(a:number[],b:number[])=>{let best=Infinity;for(const i of a)for(const j of b)best=Math.min(best,rowDistance(X[i],X[j]));return best;};while(groups.length>kk){let ai=0,bi=1,bd=Infinity;for(let i=0;i<groups.length;i++)for(let j=i+1;j<groups.length;j++){const d=distGroups(groups[i],groups[j]);if(d<bd){bd=d;ai=i;bi=j;}}groups[ai]=groups[ai].concat(groups[bi]);groups.splice(bi,1);}const labels=new Array(X.length).fill(-1);groups.forEach((g,c)=>g.forEach(i=>labels[i]=c));const centers=groups.map(g=>X[0].map((_,j)=>g.reduce((a,i)=>a+X[i][j],0)/g.length));return {algorithm:"Agglomerative",labels,centers,k:kk,silhouette:silhouetteScore(X,labels),sizes:groups.map((g,c)=>({cluster:c,size:g.length}))}; }
export function silhouetteScore(X0:number[][],labels:number[]):number { const X=safeClusterMatrix(X0),valid=labels.map((l,i)=>l>=0?i:-1).filter(i=>i>=0),clusters=[...new Set(valid.map(i=>labels[i]))];if(clusters.length<2||valid.length<3)return 0;let total=0,count=0;for(const i of valid){const own=labels[i],same=valid.filter(j=>labels[j]===own&&j!==i);if(!same.length)continue;const a=same.reduce((s,j)=>s+rowDistance(X[i],X[j]),0)/same.length;let b=Infinity;for(const c of clusters)if(c!==own){const other=valid.filter(j=>labels[j]===c);if(other.length)b=Math.min(b,other.reduce((s,j)=>s+rowDistance(X[i],X[j]),0)/other.length);}total+=(Number.isFinite(b)?(b-a)/Math.max(a,b,1e-12):0);count++;}return count?total/count:0;}

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
  if (cfg.algo === "SVMClassifier") return trainSVM(X, y, nClasses, { lr: num("learning_rate", 0.05), epochs: num("max_iter", 300), C: num("C", 1) });
  if (cfg.algo === "LinearRegression") return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0, fitIntercept: cfg.params.fit_intercept !== "False" });
  // Same rescaling as logreg: the raw alpha (default 1) over-regularised this
  // averaged-gradient solver and underfit; 0.01× keeps it mild but effective.
  if (cfg.algo === "Ridge") return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0.01 * num("alpha", 1) });
  if (cfg.algo === "Lasso") return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0.01 * num("alpha", 0.1) });
  if (cfg.algo === "SVR") return trainSVR(X, y, { lr: 0.03, epochs: 400, C: num("C", 1), epsilon: num("epsilon", 0.1) });
  if (cfg.algo === "KNeighborsRegressor") return trainKNN(X, y, num("n_neighbors", 5), cfg.params.weights || "uniform", "regression", 0);
  if (cfg.algo === "DecisionTree") return trainTree(X, y, cfg.task, nClasses, { maxDepth: num("max_depth", 5), minSplit: num("min_samples_split", 2) });
  if (cfg.algo === "RandomForest") return trainForest(X, y, cfg.task, nClasses, { nTrees: num("n_estimators", 25), maxDepth: num("max_depth", 6), minSplit: num("min_samples_split", 2) });
  if (cfg.algo === "GradientBoosting") return trainGradientBoosting(X, y, cfg.task, nClasses, { nTrees: num("n_estimators", 25), maxDepth: num("max_depth", 2), learningRate: num("learning_rate", 0.1) });
  return trainLinear(X, y, { lr: 0.05, epochs: 400, alpha: 0 });
}

// k-fold cross-validation. Returns EVERY fold with its score and split sizes, so the
// UI can show that all folds actually ran (nothing silently dropped).
export interface FoldResult { fold: number; score: number; trainN: number; testN: number; }
function groupsForFold(groupIds: (string|number)[], folds: number, seed: number): Map<string,number> { const gs=Array.from(new Set(groupIds.map(String))); const sh=seededShuffle(gs, 97+seed); return new Map(sh.map((g,i)=>[g,i%folds])); }


export function crossValDetailed(cfg: TrainConfig, X: number[][], y: number[], nClasses: number, strategy: "kfold"|"stratified"|"repeated"|"timeseries" = "kfold", repeats = 3, groupIds?: (string|number)[]): FoldResult[] {
  const folds = Math.max(2, Math.min(10, Math.min(cfg.cvFolds, X.length)));
  const out: FoldResult[] = [];
  const rounds = strategy === "repeated" ? Math.max(2, repeats) : 1;
  for (let rep = 0; rep < rounds; rep++) {
    let idx: number[];
    if (groupIds?.length && strategy === "kfold") { const groups=Array.from(new Set(groupIds.map(String))); const gm=new Map(groups.map((g,i)=>[g,i])); idx=seededShuffle(X.map((_,i)=>i), 31+rep).sort((a,b)=>(gm.get(String(groupIds[a]))??0)-(gm.get(String(groupIds[b]))??0)); }
    else if (strategy === "timeseries") idx = X.map((_,i)=>i);
    else if (strategy === "stratified" && cfg.task === "classification") {
      const buckets=Array.from({length:nClasses},()=>[] as number[]); y.forEach((v,i)=>buckets[v]?.push(i)); idx=[]; const shuffled=buckets.map((b,k)=>seededShuffle(b,7+k+rep)); for(let k=0;k<folds;k++) for(const b of shuffled) if(b[k]!=null) idx.push(b[k]);
    } else idx = seededShuffle(X.map((_, i) => i), 7 + rep);
    for (let f = 0; f < folds; f++) {
      const testI = new Set(strategy === "timeseries" ? idx.slice(Math.floor(X.length*(f/folds)), Math.floor(X.length*((f+1)/folds))) : groupIds?.length ? idx.filter(i => ((groupsForFold(groupIds, folds, rep).get(String(groupIds[i])) ?? 0) === f)) : idx.filter((_, i) => i % folds === f));
    const Xtr: number[][] = [], ytr: number[] = [], Xte: number[][] = [], yte: number[] = [];
    X.forEach((row, i) => { if (testI.has(i)) { Xte.push(row); yte.push(y[i]); } else { Xtr.push(row); ytr.push(y[i]); } });
    if (!Xtr.length || !Xte.length) continue;
    const m = makeModel(cfg, Xtr, ytr, nClasses);
    const pred = predict(m, Xte);
    const score = cfg.task === "classification"
      ? yte.reduce((a, t, i) => a + (t === Math.round(pred[i]) ? 1 : 0), 0) / yte.length
      : regressionMetrics(yte, pred).r2;
    out.push({ fold: rep * folds + f + 1, score, trainN: Xtr.length, testN: Xte.length });
    }
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
  const rootBranch = root as Extract<TreeNode, { leaf: false }>;
  const reg = model.task === "regression";
  const imp = (yy: number[]) => (reg ? variance(yy) : gini(yy, nClasses));
  const yl: number[] = [], yr: number[] = [];
  for (let i = 0; i < X.length; i++) (X[i][rootBranch.feat] <= rootBranch.thr ? yl : yr).push(y[i]);
  const parent = imp(y), left = imp(yl), right = imp(yr);
  return { feat: rootBranch.feat, thr: rootBranch.thr, parent, left, right, gain: parent - (yl.length / y.length) * left - (yr.length / y.length) * right, nL: yl.length, nR: yr.length, metric: reg ? "variance" : "gini" };
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
        else if (s.method === "Label") { const cats=Array.from(new Set(vals)); const map=new Map(cats.map((c,i)=>[c,i])); working.splice(idx,1,{name,type:"num",values:vals.map(v=>map.get(v)??-1)}); changed.push(name); }
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

// ──────────────────────────────────────────────────────────────────────────
// ── Feature Engineering: derive new numeric columns from existing ones ──
// ──────────────────────────────────────────────────────────────────────────
export const ENGINEER_OPS = ["Square (x²)", "Square root (√x)", "Interaction (A×B)", "Ratio (A÷B)", "Sum (A+B)", "Difference (A−B)"] as const;
export type EngineerOp = (typeof ENGINEER_OPS)[number];
// Which engineered ops need a second source column.
export const ENGINEER_NEEDS_B: Record<EngineerOp, boolean> = {
  "Square (x²)": false, "Square root (√x)": false,
  "Interaction (A×B)": true, "Ratio (A÷B)": true, "Sum (A+B)": true, "Difference (A−B)": true,
};
// Builds one new derived Column from one or two existing numeric columns. Pure
// function — the caller (UI) decides whether/how to append it to the dataset.
export function engineerColumn(ds: Dataset, colA: string, colB: string | null, op: EngineerOp): Column {
  const a = ds.columns.find((c) => c.name === colA);
  if (!a) throw new Error(`Column not found: ${colA}`);
  const av = a.values.map((v) => (v == null ? null : Number(v)));
  let bv: (number | null)[] | null = null;
  if (ENGINEER_NEEDS_B[op]) {
    if (!colB) throw new Error("Pick a second column for this operation.");
    const b = ds.columns.find((c) => c.name === colB);
    if (!b) throw new Error(`Column not found: ${colB}`);
    bv = b.values.map((v) => (v == null ? null : Number(v)));
  }
  const out: (number | null)[] = av.map((x, i) => {
    if (x == null || !Number.isFinite(x)) return null;
    if (op === "Square (x²)") return x * x;
    if (op === "Square root (√x)") return x < 0 ? null : Math.sqrt(x);
    const y = bv![i];
    if (y == null || !Number.isFinite(y)) return null;
    if (op === "Interaction (A×B)") return x * y;
    if (op === "Ratio (A÷B)") return y === 0 ? null : x / y;
    if (op === "Sum (A+B)") return x + y;
    return x - y; // "Difference (A−B)"
  });
  const name = op === "Square (x²)" ? `${colA}^2`
    : op === "Square root (√x)" ? `sqrt(${colA})`
    : op === "Interaction (A×B)" ? `${colA}×${colB}`
    : op === "Ratio (A÷B)" ? `${colA}÷${colB}`
    : op === "Sum (A+B)" ? `${colA}+${colB}`
    : `${colA}−${colB}`;
  return { name, type: "num", values: out };
}

// ──────────────────────────────────────────────────────────────────────────
// ── Feature Selection: rank the encoded feature matrix for a model ──
// ──────────────────────────────────────────────────────────────────────────
export interface FeatureScore { name: string; score: number; }
// Univariate filter: |Pearson correlation| between each (already-encoded, numeric)
// feature column and the target. Higher = more linearly associated with the target.
// A standard, fast filter-method feature-selection technique.
export function rankFeaturesByCorrelation(X: number[][], y: number[], featureNames: string[]): FeatureScore[] {
  const out: FeatureScore[] = featureNames.map((name, j) => {
    const col = X.map((row) => row[j]);
    const r = pearson(col, y);
    return { name, score: Number.isFinite(r) ? Math.abs(r) : 0 };
  });
  return out.sort((a, b) => b.score - a.score);
}
// Variance filter: near-constant columns carry little information. Best read
// on the RAW (pre-scaling) matrix, since scaling normalises variance to ~1.
export function rankFeaturesByVariance(X: number[][], featureNames: string[]): FeatureScore[] {
  const out: FeatureScore[] = featureNames.map((name, j) => ({ name, score: variance(X.map((row) => row[j])) }));
  return out.sort((a, b) => b.score - a.score);
}
// Maps an ENCODED column name (e.g. "city=Paris" from One-Hot, or "age_b0" from
// Binary encoding) back to the raw dataset column it came from, so a per-encoded-
// column score can be rolled up into a per-raw-feature decision.
export function rawFeatureOf(encodedName: string): string {
  if (encodedName.includes("=")) return encodedName.split("=")[0];
  const m = encodedName.match(/^(.*)_b\d+$/);
  return m ? m[1] : encodedName;
}
// Aggregates encoded-column scores back to raw feature columns (max over each
// raw column's encoded pieces) and returns them ranked, highest first.
export function aggregateFeatureScores(scores: FeatureScore[]): FeatureScore[] {
  const best = new Map<string, number>();
  for (const s of scores) { const raw = rawFeatureOf(s.name); best.set(raw, Math.max(best.get(raw) ?? -Infinity, s.score)); }
  return Array.from(best.entries()).map(([name, score]) => ({ name, score })).sort((a, b) => b.score - a.score);
}

// ──────────────────────────────────────────────────────────────────────────
// ── Hyperparameter Tuning: grid search over cross-validated score ──
// ──────────────────────────────────────────────────────────────────────────
export interface GridSearchResult { params: Record<string, string>; meanScore: number; foldScores: number[]; method?: string; }
export interface SearchRun { method: "Grid Search" | "Random Search" | "Bayesian Optimization"; best: GridSearchResult | null; results: GridSearchResult[]; tested: number; }
// Exhaustive grid search: tries every combination in `grid`, scores each with
// the same k-fold CV used elsewhere in the app, and returns results best-first.
export function gridSearch(base: TrainConfig, grid: Record<string, string[]>, X: number[][], y: number[], nClasses: number): GridSearchResult[] {
  const keys = Object.keys(grid).filter((k) => grid[k]?.length);
  let combos: Record<string, string>[] = [{}];
  for (const k of keys) {
    const next: Record<string, string>[] = [];
    for (const c of combos) for (const v of grid[k]) next.push({ ...c, [k]: v });
    combos = next;
  }
  return combos.map((combo) => {
    const cfg: TrainConfig = { ...base, params: { ...base.params, ...combo } };
    const foldScores = crossVal(cfg, X, y, nClasses);
    const meanScore = foldScores.length ? foldScores.reduce((a, b) => a + b, 0) / foldScores.length : 0;
    return { params: combo, meanScore, foldScores };
  }).sort((a, b) => b.meanScore - a.meanScore);
}

export function randomSearch(base: TrainConfig, grid: Record<string, string[]>, X: number[][], y: number[], nClasses: number, iterations = 20, seed = 17): GridSearchResult[] {
  const keys = Object.keys(grid).filter((k) => grid[k]?.length);
  const maxCombos = keys.reduce((n, k) => n * Math.max(1, grid[k].length), 1);
  const target = Math.min(Math.max(1, iterations), maxCombos);
  let state = seed >>> 0;
  const rnd = () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; };
  const seen = new Set<string>(), combos: Record<string,string>[] = [];
  while (combos.length < target) {
    const c: Record<string,string> = {}; keys.forEach((k) => { const vals = grid[k]; c[k] = vals[Math.floor(rnd() * vals.length)]; });
    const key = JSON.stringify(c); if (!seen.has(key)) { seen.add(key); combos.push(c); }
  }
  return scoreSearchCombos(base, combos, X, y, nClasses).sort((a,b)=>b.meanScore-a.meanScore);
}

// Lightweight Bayesian-style optimization: a surrogate based on inverse-distance weighted
// scores proposes candidates near promising numeric settings, while retaining exploration.
// This is deterministic, bounded, and suitable for an in-browser ML workbench.
export function bayesianSearch(base: TrainConfig, grid: Record<string, string[]>, X: number[][], y: number[], nClasses: number, iterations = 20): GridSearchResult[] {
  const keys = Object.keys(grid).filter((k) => grid[k]?.length);
  const all: Record<string,string>[] = []; let combos: Record<string,string>[] = [{}];
  for (const k of keys) { const next: Record<string,string>[]=[]; for (const c of combos) for (const v of grid[k]) next.push({...c,[k]:v}); combos=next; if(combos.length>500) break; }
  all.push(...combos);
  if (!all.length) return [];
  const tested: GridSearchResult[] = []; const seen = new Set<string>();
  const take = (c: Record<string,string>) => { const key=JSON.stringify(c); if(seen.has(key)) return false; seen.add(key); const r=scoreSearchCombos(base,[c],X,y,nClasses)[0]; if(r) tested.push(r); return true; };
  take(all[Math.floor(all.length/2)]);
  while (tested.length < Math.min(Math.max(1,iterations), all.length)) {
    let bestCandidate = all.find(c=>!seen.has(JSON.stringify(c))) || all[0], bestAcq=-Infinity;
    for (const c of all) { const key=JSON.stringify(c); if(seen.has(key)) continue; let pred=0, wsum=0; for(const r of tested){ let d=0; for(const k of keys){ const vals=grid[k]; const ai=vals.indexOf(c[k]), bi=vals.indexOf(r.params[k]); d += Math.abs(ai-bi)/(Math.max(1,vals.length-1)); } const w=1/(0.05+d); pred+=w*r.meanScore; wsum+=w; } pred/=wsum||1; const exploration=0.1/Math.sqrt(tested.length+1); const acq=pred+exploration; if(acq>bestAcq){bestAcq=acq;bestCandidate=c;} }
    take(bestCandidate);
  }
  return tested.sort((a,b)=>b.meanScore-a.meanScore);
}

function scoreSearchCombos(base: TrainConfig, combos: Record<string,string>[], X: number[][], y: number[], nClasses: number): GridSearchResult[] {
  return combos.map(combo=>{ const cfg: TrainConfig={...base,params:{...base.params,...combo}}; const foldScores=crossVal(cfg,X,y,nClasses); const meanScore=foldScores.length?foldScores.reduce((a,b)=>a+b,0)/foldScores.length:0; return {params:combo,meanScore,foldScores}; });
}

// Small, sensible default search grids per algorithm — used to auto-populate
// the "Auto-tune" UI so the user isn't stuck defining a grid from scratch.
export function defaultGrid(algo: string): Record<string, string[]> {
  const g: Record<string, Record<string, string[]>> = {
    LogisticRegression: { C: ["0.01", "0.1", "1", "10"], learning_rate: ["0.01", "0.05", "0.2", "0.5"], max_iter: ["200", "500", "1000"] },
    KNeighborsClassifier: { n_neighbors: ["3", "5", "9", "15"], weights: ["uniform", "distance"] },
    KNeighborsRegressor: { n_neighbors: ["3", "5", "9", "15"], weights: ["uniform", "distance"] },
    Ridge: { alpha: ["0.01", "0.1", "1", "5", "10"] },
    Lasso: { alpha: ["0.001", "0.01", "0.1", "1"] },
    DecisionTree: { max_depth: ["3", "5", "8", "12"], min_samples_split: ["2", "5", "10"] },
    LinearRegression: { fit_intercept: ["True", "False"] },
    RandomForest: { n_estimators: ["10", "25", "50", "100"], max_depth: ["3", "4", "6", "10"], min_samples_split: ["2", "5", "10"] },
    SVMClassifier: { C: ["0.01", "0.1", "1", "10"], learning_rate: ["0.01", "0.05", "0.1"] },
    GradientBoosting: { n_estimators: ["10", "25", "50"], max_depth: ["1", "2", "3"], learning_rate: ["0.03", "0.1", "0.2"] },
    SVR: { C: ["0.1", "1", "10"], epsilon: ["0.01", "0.1", "0.2"] },
  };
  return g[algo] || {};
}

// ──────────────────────────────────────────────────────────────────────────
// ── Explainability (XAI): permutation importance — model-agnostic, works ──
// ── for every algorithm in this app, unlike the tree/linear-only          ──
// ── `featureImportance` above. Same idea SHAP/LIME are built on: how much ──
// ── does the model's score drop when a feature's signal is destroyed?    ──
// ──────────────────────────────────────────────────────────────────────────
export interface PermImportance { name: string; importance: number; }
export function permutationImportance(m: Model, X: number[][], y: number[], featureNames: string[], task: Task, seed = 13): PermImportance[] {
  const scoreOf = (Xin: number[][]) => {
    const pred = predict(m, Xin);
    return task === "classification"
      ? y.reduce((a, t, i) => a + (t === Math.round(pred[i]) ? 1 : 0), 0) / (y.length || 1)
      : regressionMetrics(y, pred).r2;
  };
  const base = scoreOf(X);
  const out: PermImportance[] = featureNames.map((name, j) => {
    const perm = seededShuffle(X.map((_, i) => i), seed + j);
    const Xp = X.map((row) => row.slice());
    const col = X.map((row) => row[j]);
    Xp.forEach((row, i) => { row[j] = col[perm[i]]; });
    const s = scoreOf(Xp);
    return { name, importance: base - s };
  });
  return out.sort((a, b) => b.importance - a.importance);
}

// ──────────────────────────────────────────────────────────────────────────
// ── AI-suggested feature engineering: rule-based, explainable candidates ──
// Not an LLM call — a deterministic statistical scan (correlation + skew
// checks) that proposes engineered columns and explains why each one might
// help, exactly like the manual "Feature engineering" card but auto-searched.
// ──────────────────────────────────────────────────────────────────────────
export interface FeatureSuggestion {
  id: string; op: EngineerOp; colA: string; colB: string | null; name: string;
  reason: string; benefit: string; risk: "Low" | "Medium" | "High"; score: number;
}
function skewness(vals: number[]): number {
  const n = vals.length; if (n < 3) return 0;
  const m = vals.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1;
  return vals.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n;
}
export function suggestFeatures(ds: Dataset, target: string, maxSuggestions = 8): FeatureSuggestion[] {
  const numCols = ds.columns.filter((c) => c.type === "num" && c.name !== target);
  const tcol = ds.columns.find((c) => c.name === target);
  if (!tcol || numCols.length === 0) return [];
  let tnum: number[];
  if (tcol.type === "cat") {
    const cats = Array.from(new Set(tcol.values.filter((v): v is string => v != null).map(String)));
    const map = new Map(cats.map((c, i) => [c, i]));
    tnum = tcol.values.map((v) => (v == null ? NaN : (map.get(String(v)) ?? NaN)));
  } else tnum = tcol.values.map((v) => (v == null ? NaN : Number(v)));

  const corrWithTarget = (vals: (number | null)[]): number => {
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < vals.length; i++) { const x = vals[i], y = tnum[i]; if (x != null && Number.isFinite(x) && Number.isFinite(y)) { xs.push(x); ys.push(y); } }
    if (xs.length < 5) return 0;
    const r = pearson(xs, ys);
    return Number.isFinite(r) ? r : 0;
  };
  const stats = numCols.map((c) => {
    const vals = c.values.map((v) => (v == null ? null : Number(v)));
    const finite = vals.filter((v): v is number => v != null && Number.isFinite(v));
    return { col: c, vals, corr: corrWithTarget(vals), skew: skewness(finite), allNonNeg: vals.every((v) => v == null || v >= 0) };
  });

  const suggestions: FeatureSuggestion[] = [];
  for (const s of stats) {
    if (Math.abs(s.skew) > 1 && s.allNonNeg) {
      suggestions.push({
        id: `sqrt-${s.col.name}`, op: "Square root (√x)", colA: s.col.name, colB: null, name: `sqrt(${s.col.name})`,
        reason: `"${s.col.name}" is heavily skewed (skewness ${s.skew.toFixed(2)}).`,
        benefit: "A square-root transform compresses the long tail — often makes linear-model relationships more linear.",
        risk: "Low", score: Math.abs(s.skew) * 0.5 + Math.abs(s.corr),
      });
    }
  }
  for (const s of stats) {
    if (Math.abs(s.corr) > 0.05 && Math.abs(s.corr) < 0.4) {
      const sq = s.vals.map((v) => (v == null ? null : v * v));
      const sqCorr = corrWithTarget(sq);
      if (Math.abs(sqCorr) > Math.abs(s.corr) + 0.05) {
        suggestions.push({
          id: `sq-${s.col.name}`, op: "Square (x²)", colA: s.col.name, colB: null, name: `${s.col.name}^2`,
          reason: `Squaring "${s.col.name}" correlates more strongly with the target (|r|=${Math.abs(sqCorr).toFixed(2)}) than the raw column (|r|=${Math.abs(s.corr).toFixed(2)}).`,
          benefit: "Captures a U-shaped or accelerating relationship a linear model would otherwise miss.",
          risk: "Low", score: Math.abs(sqCorr) - Math.abs(s.corr),
        });
      }
    }
  }
  const ranked = [...stats].sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr)).slice(0, 6);
  for (let i = 0; i < ranked.length; i++) for (let j = i + 1; j < ranked.length; j++) {
    const a = ranked[i], b = ranked[j];
    if (Math.abs(a.corr) < 0.1 || Math.abs(b.corr) < 0.1) continue;
    const interaction = a.vals.map((v, k) => (v == null || b.vals[k] == null ? null : v * (b.vals[k] as number)));
    const iCorr = corrWithTarget(interaction);
    if (Math.abs(iCorr) > Math.max(Math.abs(a.corr), Math.abs(b.corr)) + 0.03) {
      suggestions.push({
        id: `int-${a.col.name}-${b.col.name}`, op: "Interaction (A×B)", colA: a.col.name, colB: b.col.name, name: `${a.col.name}×${b.col.name}`,
        reason: `Both "${a.col.name}" (|r|=${Math.abs(a.corr).toFixed(2)}) and "${b.col.name}" (|r|=${Math.abs(b.corr).toFixed(2)}) individually associate with the target, and their product correlates even more strongly (|r|=${Math.abs(iCorr).toFixed(2)}).`,
        benefit: "Combined effects like this often beat either raw feature alone in linear/gradient-descent models.",
        risk: "Medium", score: Math.abs(iCorr),
      });
    }
    if (b.allNonNeg && b.vals.every((v) => v == null || v > 0)) {
      const ratio = a.vals.map((v, k) => (v == null || b.vals[k] == null || b.vals[k] === 0 ? null : v / (b.vals[k] as number)));
      const rCorr = corrWithTarget(ratio);
      if (Math.abs(rCorr) > Math.max(Math.abs(a.corr), Math.abs(b.corr)) + 0.03) {
        suggestions.push({
          id: `ratio-${a.col.name}-${b.col.name}`, op: "Ratio (A÷B)", colA: a.col.name, colB: b.col.name, name: `${a.col.name}÷${b.col.name}`,
          reason: `"${a.col.name}" relative to "${b.col.name}" correlates more strongly with the target (|r|=${Math.abs(rCorr).toFixed(2)}) than either column alone.`,
          benefit: "Ratios often represent a rate or proportion that's more meaningful than either raw quantity.",
          risk: "Medium", score: Math.abs(rCorr),
        });
      }
    }
  }
  return suggestions.filter((s) => s.score > 0).sort((a, b) => b.score - a.score).slice(0, maxSuggestions);
}

// ──────────────────────────────────────────────────────────────────────────
// ── Auto pipeline detection: one-click target/task/features/preprocessing ──
// Heuristic, transparent, and always shown as a preview before it's applied —
// never silently overwrites the user's current setup.
// ──────────────────────────────────────────────────────────────────────────
export interface AutoPipelinePlan {
  target: string; task: Task; features: string[]; steps: PrepStep[];
  excludedIdLike: string[]; excludedConstant: string[];
  targetCandidates: { name: string; score: number; reason: string }[];
  warnings: string[];
}
export function autoDetectPipeline(ds: Dataset): AutoPipelinePlan {
  const n = ds.nrows;
  const TARGET_HINTS = ["target", "label", "class", "outcome", "churn", "y", "result", "response", "default", "fraud", "survived", "price", "score", "status", "diagnosis", "approved", "purchase", "sales", "revenue"];
  const candidates = ds.columns.map((c, idx) => {
    const nonNull = c.values.filter((v) => v != null);
    const uniq = new Set(nonNull.map(String)).size;
    const uniquePct = nonNull.length ? uniq / nonNull.length : 0;
    let score = 0; const reasons: string[] = [];
    const lower = c.name.toLowerCase();
    if (TARGET_HINTS.some((h) => lower === h || lower.includes(h))) { score += 3; reasons.push("name matches a common target keyword"); }
    if (c.type === "cat" && uniq >= 2 && uniq <= 12) { score += 2; reasons.push("low-cardinality categorical — looks like a class label"); }
    if (c.type === "num" && uniq >= 2 && uniq <= 10) { score += 1; reasons.push("small set of numeric values — could be an encoded class"); }
    if (uniquePct > 0.95 && nonNull.length > 20) { score -= 3; reasons.push("near-unique — looks like an identifier, not a label"); }
    if (idx === ds.columns.length - 1) { score += 0.5; reasons.push("last column (common dataset convention)"); }
    return { name: c.name, score, reason: reasons.join("; ") || "no strong signal" };
  }).sort((a, b) => b.score - a.score);
  const target = candidates[0]?.name ?? ds.columns[0].name;
  const tcol = ds.columns.find((c) => c.name === target)!;
  const uniqT = new Set(tcol.values.filter((v) => v != null).map(String)).size;
  const task: Task = (tcol.type === "cat" || uniqT <= 12) ? "classification" : "regression";

  const excludedIdLike: string[] = [], excludedConstant: string[] = [];
  const candidateCols = ds.columns.filter((c) => c.name !== target).filter((c) => {
    const nonNull = c.values.filter((v) => v != null);
    const uniq = new Set(nonNull.map(String)).size;
    if (uniq <= 1) { excludedConstant.push(c.name); return false; }
    const uniquePct = nonNull.length ? uniq / nonNull.length : 0;
    if (c.type === "cat" && uniquePct > 0.95 && nonNull.length > 20) { excludedIdLike.push(c.name); return false; }
    return true;
  });
  const features = candidateCols.map((c) => c.name);
  const nums = candidateCols.filter((c) => c.type === "num").map((c) => c.name);
  const cats = candidateCols.filter((c) => c.type === "cat").map((c) => c.name);
  const hasMissing = (name: string) => ds.columns.find((c) => c.name === name)!.values.some((v) => v == null);
  const steps: PrepStep[] = [];
  const numsWithMissing = nums.filter(hasMissing);
  if (numsWithMissing.length) steps.push({ op: "Impute missing", cols: numsWithMissing, method: "Median" });
  if (nums.length) steps.push({ op: "Scale / normalize", cols: nums, method: "StandardScaler" });
  const catsWithMissing = cats.filter(hasMissing);
  if (catsWithMissing.length) steps.push({ op: "Impute missing", cols: catsWithMissing, method: "Most frequent" });
  const lowCardCats: string[] = [], highCardCats: string[] = [];
  for (const name of cats) {
    const uniq = new Set(ds.columns.find((c) => c.name === name)!.values.filter((v) => v != null).map(String)).size;
    (uniq <= 12 ? lowCardCats : highCardCats).push(name);
  }
  if (lowCardCats.length) steps.push({ op: "Encode categorical", cols: lowCardCats, method: "One-Hot" });
  if (highCardCats.length) steps.push({ op: "Encode categorical", cols: highCardCats, method: "Ordinal" });

  const warnings: string[] = [];
  if (highCardCats.length) warnings.push(`High-cardinality column(s) ${highCardCats.join(", ")} were ordinal- rather than one-hot-encoded to avoid a column explosion — consider frequency/target encoding instead.`);
  if (excludedIdLike.length) warnings.push(`Excluded likely identifier column(s): ${excludedIdLike.join(", ")}.`);
  if (excludedConstant.length) warnings.push(`Excluded constant column(s): ${excludedConstant.join(", ")}.`);
  if (n < 50) warnings.push(`Only ${n} rows — validation results will be noisy; treat any model choice as tentative.`);

  return { target, task, features, steps, excludedIdLike, excludedConstant, targetCandidates: candidates.slice(0, 5), warnings };
}
