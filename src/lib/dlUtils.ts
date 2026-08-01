// A real from-scratch multilayer perceptron — forward + backprop + mini-batch
// (S)GD/Momentum/Adam — for tabular binary / multi-class / regression tasks.
// No libraries. Trains fully in the browser (no GPU); scope is tabular data, not
// images/text (those need heavy compute).

export type DlTask = "binary" | "multiclass" | "regression";
export interface Net { sizes: number[]; W: number[][][]; b: number[][]; act: string; task: DlTask; outDim: number; }
export interface DataSet { X: number[][]; y: number[]; task: DlTask; classes: string[]; featNames: string[] }
export type Optimizer = "sgd" | "momentum" | "adam";
export interface OptState { optimizer: Optimizer; t: number; mW: number[][][]; mb: number[][]; vW: number[][][]; vb: number[][] }

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const R = 5; // toy data lives in roughly [-R, R]^2

// ── built-in 2-D toy datasets (binary) ──
export function genDataset(kind: string, n = 240, noise = 0.15, seed = 3): DataSet {
  const rng = mulberry32(seed); const X: number[][] = [], y: number[] = [];
  const jit = () => (rng() * 2 - 1) * noise * R;
  if (kind === "xor") { for (let i = 0; i < n; i++) { const x = (rng() * 2 - 1) * R, z = (rng() * 2 - 1) * R; X.push([x + jit(), z + jit()]); y.push((x > 0) === (z > 0) ? 1 : 0); } }
  else if (kind === "circles") { for (let i = 0; i < n; i++) { const inner = i < n / 2; const r = inner ? rng() * R * 0.45 : R * 0.6 + rng() * R * 0.4; const t = rng() * Math.PI * 2; X.push([r * Math.cos(t) + jit(), r * Math.sin(t) + jit()]); y.push(inner ? 0 : 1); } }
  else if (kind === "moons") { for (let i = 0; i < n; i++) { const up = i < n / 2; const t = rng() * Math.PI; if (up) X.push([R * Math.cos(t) + jit(), R * Math.sin(t) - R * 0.3 + jit()]); else X.push([R - R * Math.cos(t) - R * 0.7 + jit(), -R * Math.sin(t) + R * 0.3 + jit()]); y.push(up ? 0 : 1); } }
  else if (kind === "blobs3") { const centers = [[-3, 3], [3, 3], [0, -3]]; for (let i = 0; i < n; i++) { const c = i % 3; X.push([centers[c][0] + (rng() * 2 - 1) * 1.6, centers[c][1] + (rng() * 2 - 1) * 1.6]); y.push(c); } return { X, y, task: "multiclass", classes: ["A", "B", "C"], featNames: ["x1", "x2"] }; }
  else if (kind === "sine") { for (let i = 0; i < n; i++) { const x = (rng() * 2 - 1) * R; X.push([x]); y.push(Math.sin(x) * 3 + (rng() * 2 - 1) * noise * 3); } return { X, y, task: "regression", classes: [], featNames: ["x"] }; }
  else { const per = Math.floor(n / 2); for (let c = 0; c < 2; c++) for (let i = 0; i < per; i++) { const r = (i / per) * R; const t = c * Math.PI + (i / per) * 3.2; X.push([r * Math.sin(t) + jit(), r * Math.cos(t) + jit()]); y.push(c); } }
  return { X, y, task: "binary", classes: ["0", "1"], featNames: ["x1", "x2"] };
}

// ── standardization (neural nets need scaled inputs) ──
export function fitScaler(X: number[][]): { mean: number[]; std: number[] } {
  const d = X[0]?.length || 0, n = X.length || 1; const mean = new Array(d).fill(0), std = new Array(d).fill(0);
  X.forEach((r) => r.forEach((v, j) => { mean[j] += v / n; }));
  X.forEach((r) => r.forEach((v, j) => { std[j] += (v - mean[j]) ** 2 / n; }));
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]) || 1;
  return { mean, std };
}
export const applyScaler = (X: number[][], s: { mean: number[]; std: number[] }): number[][] => X.map((r) => r.map((v, j) => (v - s.mean[j]) / s.std[j]));
export const scaleRow = (x: number[], s: { mean: number[]; std: number[] }): number[] => x.map((v, j) => (v - s.mean[j]) / s.std[j]);

const actF = (name: string, z: number) => name === "relu" ? Math.max(0, z) : name === "sigmoid" ? 1 / (1 + Math.exp(-z)) : Math.tanh(z);
const actD = (name: string, a: number) => name === "relu" ? (a > 0 ? 1 : 0) : name === "sigmoid" ? a * (1 - a) : 1 - a * a;
function softmax(z: number[]): number[] { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0) || 1; return e.map((v) => v / s); }

export function initNet(inputDim: number, hidden: number[], outDim: number, act: string, task: DlTask, seed = 42): Net {
  const sizes = [inputDim, ...hidden, outDim]; const rng = mulberry32(seed); const W: number[][][] = [], b: number[][] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fin = sizes[l], fout = sizes[l + 1]; const scale = act === "relu" ? Math.sqrt(2 / fin) : Math.sqrt(1 / fin);
    W.push(Array.from({ length: fout }, () => Array.from({ length: fin }, () => (rng() * 2 - 1) * scale)));
    b.push(new Array(fout).fill(0));
  }
  return { sizes, W, b, act, task, outDim };
}

// Forward pass; the last layer applies sigmoid (binary) / softmax (multiclass) / identity (regression).
export function forward(net: Net, x: number[]): { as: number[][]; out: number[] } {
  const as: number[][] = [x.slice()];
  for (let l = 0; l < net.W.length; l++) {
    const zs = new Array(net.W[l].length); const last = l === net.W.length - 1;
    for (let i = 0; i < net.W[l].length; i++) { let z = net.b[l][i]; const row = net.W[l][i]; for (let j = 0; j < row.length; j++) z += row[j] * as[l][j]; zs[i] = last ? z : actF(net.act, z); }
    as.push(zs);
  }
  const zL = as[as.length - 1];
  const out = net.task === "multiclass" ? softmax(zL) : net.task === "binary" ? [1 / (1 + Math.exp(-zL[0]))] : zL.slice();
  as[as.length - 1] = out;
  return { as, out };
}
export const predictVec = (net: Net, x: number[]): number[] => forward(net, x).out;
export const predictClass = (net: Net, x: number[]): number => { const o = predictVec(net, x); return net.task === "regression" ? o[0] : (net.task === "binary" ? (o[0] > 0.5 ? 1 : 0) : o.indexOf(Math.max(...o))); };

function target(net: Net, y: number): number[] { if (net.task === "multiclass") { const t = new Array(net.outDim).fill(0); t[y] = 1; return t; } return [y]; }

export function newOpt(net: Net, optimizer: Optimizer): OptState {
  const mW = net.W.map((l) => l.map((r) => r.map(() => 0))), vW = net.W.map((l) => l.map((r) => r.map(() => 0)));
  const mb = net.b.map((l) => l.map(() => 0)), vb = net.b.map((l) => l.map(() => 0));
  return { optimizer, t: 0, mW, mb, vW, vb };
}

// One mini-batch: accumulate gradients over the batch, then apply the optimizer update.
function trainBatch(net: Net, X: number[][], y: number[], batch: number[], lr: number, l2: number, opt: OptState) {
  const gW = net.W.map((l) => l.map((r) => r.map(() => 0))), gb = net.b.map((l) => l.map(() => 0));
  for (const bi of batch) {
    const { as } = forward(net, X[bi]); const L = net.W.length; const t = target(net, y[bi]);
    const deltas: number[][] = new Array(L);
    deltas[L - 1] = as[L].map((o, i) => o - t[i]); // pred - target (matched loss/activation)
    for (let l = L - 2; l >= 0; l--) { const dl = new Array(net.sizes[l + 1]).fill(0); for (let j = 0; j < net.sizes[l + 1]; j++) { let s = 0; for (let i = 0; i < net.sizes[l + 2]; i++) s += net.W[l + 1][i][j] * deltas[l + 1][i]; dl[j] = s * actD(net.act, as[l + 1][j]); } deltas[l] = dl; }
    for (let l = 0; l < L; l++) for (let i = 0; i < net.W[l].length; i++) { const row = net.W[l][i]; for (let j = 0; j < row.length; j++) gW[l][i][j] += deltas[l][i] * as[l][j]; gb[l][i] += deltas[l][i]; }
  }
  const n = batch.length || 1; opt.t++;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const step = (w: number, g: number, m: { v: number }, v: { v: number }) => {
    if (opt.optimizer === "adam") { m.v = b1 * m.v + (1 - b1) * g; v.v = b2 * v.v + (1 - b2) * g * g; const mh = m.v / (1 - Math.pow(b1, opt.t)), vh = v.v / (1 - Math.pow(b2, opt.t)); return w - lr * mh / (Math.sqrt(vh) + eps); }
    if (opt.optimizer === "momentum") { m.v = 0.9 * m.v - lr * g; return w + m.v; }
    return w - lr * g;
  };
  for (let l = 0; l < net.W.length; l++) for (let i = 0; i < net.W[l].length; i++) {
    const row = net.W[l][i];
    for (let j = 0; j < row.length; j++) { const g = gW[l][i][j] / n + l2 * row[j]; const mRef = { v: opt.mW[l][i][j] }, vRef = { v: opt.vW[l][i][j] }; row[j] = step(row[j], g, mRef, vRef); opt.mW[l][i][j] = mRef.v; opt.vW[l][i][j] = vRef.v; }
    const gbv = gb[l][i] / n; const mRef = { v: opt.mb[l][i] }, vRef = { v: opt.vb[l][i] }; net.b[l][i] = step(net.b[l][i], gbv, mRef, vRef); opt.mb[l][i] = mRef.v; opt.vb[l][i] = vRef.v;
  }
}

export interface EpochStat { loss: number; acc: number }
export function trainEpoch(net: Net, X: number[][], y: number[], p: { lr: number; l2: number; batchSize: number }, opt: OptState): EpochStat {
  const n = X.length; const idx = [...Array(n).keys()]; const rng = mulberry32(777 + opt.t);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const B = Math.max(1, Math.min(p.batchSize || n, n));
  for (let s = 0; s < n; s += B) trainBatch(net, X, y, idx.slice(s, s + B), p.lr, p.l2, opt);
  return evalNet(net, X, y);
}

// Loss + a headline metric (accuracy for classification, R² for regression).
export function evalNet(net: Net, X: number[][], y: number[]): EpochStat {
  let loss = 0, correct = 0;
  if (net.task === "regression") { const mean = y.reduce((a, b) => a + b, 0) / (y.length || 1); let ssr = 0, sst = 0; for (let i = 0; i < X.length; i++) { const p = predictVec(net, X[i])[0]; ssr += (p - y[i]) ** 2; sst += (y[i] - mean) ** 2; } return { loss: ssr / (X.length || 1), acc: 1 - ssr / (sst || 1) }; }
  for (let i = 0; i < X.length; i++) { const o = predictVec(net, X[i]); if (net.task === "binary") { const p = o[0]; loss += -(y[i] * Math.log(Math.max(1e-7, p)) + (1 - y[i]) * Math.log(Math.max(1e-7, 1 - p))); if ((p > 0.5 ? 1 : 0) === y[i]) correct++; } else { loss += -Math.log(Math.max(1e-7, o[y[i]])); if (o.indexOf(Math.max(...o)) === y[i]) correct++; } }
  return { loss: loss / (X.length || 1), acc: correct / (X.length || 1) };
}

export interface DlEval { task: DlTask; loss: number; acc: number; confusion?: number[][]; classes?: string[]; predActual?: [number, number][] }
export function fullEval(net: Net, X: number[][], y: number[], classes: string[]): DlEval {
  const base = evalNet(net, X, y);
  if (net.task === "regression") { return { task: "regression", loss: base.loss, acc: base.acc, predActual: X.map((x, i) => [y[i], predictVec(net, x)[0]] as [number, number]) }; }
  const K = net.task === "binary" ? 2 : net.outDim; const cm = Array.from({ length: K }, () => new Array(K).fill(0));
  X.forEach((x, i) => { const p = predictClass(net, x); if (y[i] < K && p < K) cm[y[i]][p]++; });
  return { task: net.task, loss: base.loss, acc: base.acc, confusion: cm, classes };
}

// Decision surface over a 2-D grid (for 2-feature classification), on scaled inputs.
export function dlSurface(net: Net, scaler: { mean: number[]; std: number[] }, lo: number[], hi: number[], res = 40): { xs: number[]; ys: number[]; z: number[][] } {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < res; i++) { xs.push(lo[0] + (hi[0] - lo[0]) * i / (res - 1)); ys.push(lo[1] + (hi[1] - lo[1]) * i / (res - 1)); }
  const z: number[][] = [];
  for (const vy of ys) { const row: number[] = []; for (const vx of xs) { row.push(predictClass(net, scaleRow([vx, vy], scaler))); } z.push(row); }
  return { xs, ys, z };
}
export const DATA_BOUND = R;
