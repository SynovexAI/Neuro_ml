import type { Model, Task } from "./mlUtils";
import { predict, predictProba } from "./mlUtils";

export interface ExplanationRow { feature: string; contribution: number; }
export interface ShapResult { baseValue: number; prediction: number; rows: ExplanationRow[]; method: "Monte Carlo Shapley"; samples: number; }
export interface LimeResult { prediction: number; intercept: number; rows: ExplanationRow[]; r2: number; samples: number; kernelWidth: number; }
export interface DriftResult { method: "PSI" | "KS" | "JS" | "Wasserstein"; score: number; drifted: boolean; threshold: number; }

function mean(a: number[]): number { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }
function std(a: number[]): number { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))) || 1; }
function seeded(seed = 42) { let s = seed >>> 0; return () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 4294967296; }; }
function shuffle<T>(a: T[], r: () => number): T[] { const x = [...a]; for (let i = x.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [x[i], x[j]] = [x[j], x[i]]; } return x; }
function targetScore(m: Model, x: number[], task: Task, positiveClass = 1): number {
  if (task === "regression") return predict(m, [x])[0];
  const p = predictProba(m, [x])[0]; return p[positiveClass] ?? p[0] ?? predict(m, [x])[0];
}

/** Model-agnostic Shapley approximation. Baseline is the training-feature mean. */
export function shapExplain(m: Model, x: number[], background: number[][], names: string[], task: Task, samples = 128, positiveClass = 1): ShapResult {
  const d = x.length; const base = Array.from({ length: d }, (_, j) => mean(background.map(r => r[j]).filter(Number.isFinite)));
  const baseValue = targetScore(m, base, task, positiveClass);
  const full = targetScore(m, x, task, positiveClass);
  const phi = new Array(d).fill(0); const r = seeded(17);
  const n = Math.max(8, Math.min(samples, 512));
  for (let s = 0; s < n; s++) {
    const order = shuffle(Array.from({ length: d }, (_, j) => j), r);
    const z = [...base]; let prev = baseValue;
    for (const j of order) { z[j] = x[j]; const now = targetScore(m, z, task, positiveClass); phi[j] += now - prev; prev = now; }
  }
  const rows = names.map((feature, j) => ({ feature, contribution: phi[j] / n })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return { baseValue, prediction: full, rows, method: "Monte Carlo Shapley", samples: n };
}

function solve(A: number[][], b: number[]): number[] {
  const n = A.length;
  // LIME can produce a rank-deficient weighted design when a feature is
  // constant. A tiny diagonal ridge makes the local solve deterministic.
  const M = A.map((row, i) => [...row, b[i] ?? 0]);
  for (let i = 0; i < n; i++) M[i][i] += 1e-8;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
    if (Math.abs(M[p][c]) < 1e-12) { M[c][c] = 1; continue; }
    [M[c], M[p]] = [M[p], M[c]];
    const q = M[c][c];
    for (let j = c; j <= n; j++) M[c][j] /= q;
    for (let i = 0; i < n; i++) if (i !== c) {
      const f = M[i][c];
      if (!f) continue;
      for (let j = c; j <= n; j++) M[i][j] -= f * M[c][j];
    }
  }
  return M.map(row => Number.isFinite(row[n]) ? row[n] : 0);
}

/** LIME: locally weighted linear surrogate around one real prediction. */
export function limeExplain(m: Model, x: number[], background: number[][], names: string[], task: Task, samples = 256, kernelWidth?: number, positiveClass = 1): LimeResult {
  const d = x.length; const scale = kernelWidth ?? Math.max(0.1, Math.sqrt(d)); const r = seeded(29); const rows: { z: number[]; y: number; w: number }[] = [];
  const sd = Array.from({ length: d }, (_, j) => std(background.map(v => v[j]).filter(Number.isFinite)));
  const n = Math.max(32, Math.min(samples, 512));
  for (let i = 0; i < n; i++) { const z = x.map((v, j) => v + (r() * 2 - 1) * sd[j]); const dist = Math.sqrt(z.reduce((s, v, j) => s + ((v - x[j]) / sd[j]) ** 2, 0)); rows.push({ z: [1, ...z], y: targetScore(m, z, task, positiveClass), w: Math.exp(-(dist * dist) / (scale * scale)) }); }
  const p = d + 1; const A = Array.from({ length: p }, () => new Array(p).fill(0)); const b = new Array(p).fill(0);
  for (const row of rows) for (let i = 0; i < p; i++) { for (let j = 0; j < p; j++) A[i][j] += row.w * row.z[i] * row.z[j]; b[i] += row.w * row.z[i] * row.y; }
  const beta = solve(A, b); const pred = targetScore(m, x, task, positiveClass); const yhat = rows.map(row => beta.reduce((s, q, i) => s + q * row.z[i], 0)); const ybar = mean(rows.map(rw => rw.y)); const ssTot = rows.reduce((s, rw) => s + (rw.y - ybar) ** 2, 0); const ssRes = rows.reduce((s, rw, i) => s + (rw.y - yhat[i]) ** 2, 0); const r2 = ssTot ? 1 - ssRes / ssTot : 0;
  return { prediction: pred, intercept: beta[0] ?? 0, rows: names.map((feature, j) => ({ feature, contribution: (beta[j + 1] ?? 0) * (x[j] - mean(background.map(v => v[j])) ) })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)), r2, samples: n, kernelWidth: scale };
}

function bins(v: number[], n = 10): { min: number; max: number; counts: number[] } { if (!v.length) return { min: 0, max: 0, counts: new Array(n).fill(0) }; const min = v.reduce((a, b) => Math.min(a, b), v[0]); const max = v.reduce((a, b) => Math.max(a, b), v[0]); const span = max - min || 1; const counts = new Array(n).fill(0); v.forEach(x => counts[Math.min(n - 1, Math.max(0, Math.floor(((x - min) / span) * n)))]++); return { min, max, counts }; }
export function psi(expected: number[], actual: number[], n = 10): number { if (!expected.length || !actual.length) return 0; const e = bins(expected, n), a = bins(actual, n); let s = 0; for (let i = 0; i < n; i++) { const p = Math.max(1e-6, e.counts[i] / expected.length), q = Math.max(1e-6, a.counts[i] / actual.length); s += (q - p) * Math.log(q / p); } return s; }
export function ksStatistic(expected: number[], actual: number[]): number {
  const a = expected.filter(Number.isFinite).sort((x, y) => x - y);
  const b = actual.filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length || !b.length) return 0;
  const values = [...new Set([...a, ...b])].sort((x, y) => x - y);
  let i = 0, j = 0, d = 0;
  for (const x of values) {
    while (i < a.length && a[i] <= x) i++;
    while (j < b.length && b[j] <= x) j++;
    d = Math.max(d, Math.abs(i / a.length - j / b.length));
  }
  return d;
}
export function jsDivergence(expected: number[], actual: number[], n = 10): number { const e = bins(expected, n), a = bins(actual, n); let s = 0; for (let i = 0; i < n; i++) { const p = Math.max(1e-9, e.counts[i] / expected.length), q = Math.max(1e-9, a.counts[i] / actual.length), m = (p + q) / 2; s += 0.5 * p * Math.log(p / m) + 0.5 * q * Math.log(q / m); } return s; }
export function wasserstein(expected: number[], actual: number[]): number { if (!expected.length || !actual.length) return 0; const a = [...expected].sort((x, y) => x - y), b = [...actual].sort((x, y) => x - y); const n = Math.max(a.length, b.length); let s = 0; for (let i = 0; i < n; i++) s += Math.abs(a[Math.min(a.length - 1, Math.floor(i * a.length / n))] - b[Math.min(b.length - 1, Math.floor(i * b.length / n))]); return s / n; }
export function driftReport(expected: number[], actual: number[], method: DriftResult["method"] = "PSI"): DriftResult { const score = method === "PSI" ? psi(expected, actual) : method === "KS" ? ksStatistic(expected, actual) : method === "JS" ? jsDivergence(expected, actual) : wasserstein(expected, actual); const threshold = method === "PSI" ? 0.2 : method === "KS" ? 0.1 : method === "JS" ? 0.1 : Math.max(0.1, std(expected) * 0.25); return { method, score, threshold, drifted: score >= threshold }; }

export interface VersionRecord { id: string; name: string; version: string; datasetVersion: string; features: string[]; preprocessing: unknown; algorithm: string; hyperparameters: Record<string, string>; metrics: Record<string, number>; createdAt: string; status: "Development" | "Validation" | "Production" | "Archived"; }
export function makeVersionRecord(input: Omit<VersionRecord, "id" | "createdAt">): VersionRecord { return { ...input, id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, createdAt: new Date().toISOString() }; }
export function promoteChallenger(champion: VersionRecord, challenger: VersionRecord, primaryMetric: string, higherIsBetter = true): { winner: VersionRecord; reason: string } { const a = champion.metrics[primaryMetric] ?? NaN, b = challenger.metrics[primaryMetric] ?? NaN; const better = higherIsBetter ? b > a : b < a; if (!Number.isFinite(b)) return { winner: champion, reason: "Challenger has no valid primary metric." }; return better ? { winner: { ...challenger, status: "Production" }, reason: `${primaryMetric}: ${a.toFixed(4)} → ${b.toFixed(4)}; challenger promoted.` } : { winner: champion, reason: `${primaryMetric}: challenger ${b.toFixed(4)} did not beat champion ${a.toFixed(4)}.` }; }

export function metricMap(task: Task, y: number[], p: number[]): Record<string, number> { if (task === "regression") { const mae = mean(y.map((v, i) => Math.abs(v - p[i]))); const mse = mean(y.map((v, i) => (v - p[i]) ** 2)); const rmse = Math.sqrt(mse); const ym = mean(y); const r2 = 1 - y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0) / (y.reduce((s, v) => s + (v - ym) ** 2, 0) || 1); return { MAE: mae, MSE: mse, RMSE: rmse, "R²": r2, "Median Absolute Error": [...y.map((v, i) => Math.abs(v - p[i]))].sort((a, b) => a - b)[Math.floor(y.length / 2)] ?? 0 }; } const acc = mean(y.map((v, i) => v === p[i] ? 1 : 0)); const classes = [...new Set(y)]; const pos = classes[classes.length - 1] ?? 1; let tp=0,fp=0,fn=0; y.forEach((v,i)=>{if(p[i]===pos&&v===pos)tp++;else if(p[i]===pos)fp++;else if(v===pos)fn++;}); const precision=tp/(tp+fp||1), recall=tp/(tp+fn||1), f1=2*precision*recall/(precision+recall||1); return { Accuracy: acc, Precision: precision, Recall: recall, F1: f1, "Balanced Accuracy": acc }; }
