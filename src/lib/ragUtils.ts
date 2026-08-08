// Client-side retrieval utilities — real BM25 + TF-IDF cosine, no dependencies.
// (Neural embeddings via transformers.js can be swapped in as a "vector" backend later.)

export type Vec = Record<string, number>;
export type RagIndex = { docs: string[][]; df: Record<string, number>; N: number; avgdl: number; vectors: Vec[] };

export function tokenize(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9]+/g) || []);
}

export function chunkText(text: string, size: number, overlap: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length <= size) return words.length ? [words.join(" ")] : [];
  const step = Math.max(1, size - overlap);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += step) {
    chunks.push(words.slice(i, i + size).join(" "));
    if (i + size >= words.length) break;
  }
  return chunks;
}

export function buildIndex(chunks: string[]): RagIndex {
  const docs = chunks.map(tokenize);
  const df: Record<string, number> = {};
  docs.forEach((d) => { new Set(d).forEach((t) => { df[t] = (df[t] || 0) + 1; }); });
  const N = docs.length || 1;
  const avgdl = docs.reduce((a, d) => a + d.length, 0) / N || 1;
  const vectors = docs.map((d) => {
    const tf: Record<string, number> = {};
    d.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
    const v: Vec = {};
    Object.entries(tf).forEach(([t, c]) => { v[t] = (c / d.length) * Math.log(N / (df[t] || 1) + 1); });
    return v;
  });
  return { docs, df, N, avgdl, vectors };
}

export function bm25Scores(idx: RagIndex, query: string): number[] {
  const q = tokenize(query);
  const { docs, df, N, avgdl } = idx;
  const k1 = 1.5, b = 0.75;
  return docs.map((d) => {
    const tf: Record<string, number> = {};
    d.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
    let s = 0;
    q.forEach((t) => {
      if (!tf[t]) return;
      const idf = Math.log(1 + (N - (df[t] || 0) + 0.5) / ((df[t] || 0) + 0.5));
      s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * d.length / avgdl));
    });
    return s;
  });
}

export function queryVector(idx: RagIndex, query: string): Vec {
  const q = tokenize(query);
  const { df, N } = idx;
  const tf: Record<string, number> = {};
  q.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
  const v: Vec = {};
  Object.entries(tf).forEach(([t, c]) => { v[t] = (c / q.length) * Math.log(N / (df[t] || 1) + 1); });
  return v;
}

export function cosine(a: Vec, b: Vec): number {
  let dot = 0, na = 0, nb = 0;
  for (const k in a) { na += a[k] * a[k]; if (b[k]) dot += a[k] * b[k]; }
  for (const k in b) nb += b[k] * b[k];
  return dot / ((Math.sqrt(na) * Math.sqrt(nb)) || 1);
}

// ── selectable similarity metric (the same three Milvus exposes) ──
// cosine = angle · dot = inner product (IP) · euclidean = L2 distance → 1/(1+d) so higher is better.
export type Metric = "cosine" | "dot" | "euclidean";
export const METRIC_LABEL: Record<Metric, string> = { cosine: "cosine", dot: "dot (IP)", euclidean: "euclidean (L2)" };
export const METRIC_MILVUS: Record<Metric, string> = { cosine: "COSINE", dot: "IP", euclidean: "L2" };

// Similarity over sparse TF-IDF vectors. Distance metrics are mapped to a similarity (higher = better).
export function simSparse(a: Vec, b: Vec, metric: Metric = "cosine"): number {
  if (metric === "cosine") return cosine(a, b);
  if (metric === "dot") { let d = 0; for (const k in a) if (b[k]) d += a[k] * b[k]; return d; }
  let s = 0; // euclidean over the union of keys
  for (const k in a) { const diff = a[k] - (b[k] || 0); s += diff * diff; }
  for (const k in b) if (!(k in a)) s += b[k] * b[k];
  return 1 / (1 + Math.sqrt(s));
}
// Similarity over dense (neural) vectors, same three metrics.
export function simDense(a: number[], b: number[], metric: Metric = "cosine"): number {
  if (metric === "cosine") return denseCos(a, b);
  const n = Math.min(a.length, b.length);
  if (metric === "dot") { let d = 0; for (let i = 0; i < n; i++) d += a[i] * b[i]; return d; }
  let s = 0; for (let i = 0; i < n; i++) { const diff = a[i] - b[i]; s += diff * diff; }
  return 1 / (1 + Math.sqrt(s));
}

// Normalize a score array to 0..1 for display / hybrid blending.
function norm(arr: number[]): number[] {
  const max = Math.max(...arr, 1e-9);
  return arr.map((x) => x / max);
}

export type Strategy = "vector" | "keyword" | "hybrid";

export function retrieve(idx: RagIndex, query: string, strategy: Strategy, k: number, metric: Metric = "cosine"): { i: number; score: number }[] {
  const bm = norm(bm25Scores(idx, query));
  const qv = queryVector(idx, query);
  const vec = norm(idx.vectors.map((v) => simSparse(qv, v, metric)));
  let scores: number[];
  if (strategy === "keyword") scores = bm;
  else if (strategy === "vector") scores = vec;
  else scores = bm.map((s, i) => 0.5 * s + 0.5 * vec[i]);
  return scores
    .map((s, i) => ({ i, score: s }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// ── dense (neural) embeddings support ──
export function denseCos(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0; const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
// Retrieve on dense vectors: keyword = BM25, vector = cosine on neural embeddings, hybrid = blend.
export function retrieveDense(idx: RagIndex, query: string, qVec: number[], chunkVecs: number[][], strategy: Strategy, k: number, metric: Metric = "cosine"): { i: number; score: number }[] {
  const bm = norm(bm25Scores(idx, query));
  const vec = norm(chunkVecs.map((v) => simDense(qVec, v, metric)));
  let scores: number[];
  if (strategy === "keyword") scores = bm;
  else if (strategy === "vector") scores = vec;
  else scores = bm.map((s, i) => 0.5 * s + 0.5 * vec[i]);
  return scores.map((s, i) => ({ i, score: s })).sort((a, b) => b.score - a.score).slice(0, k);
}

// ── MMR re-ranking (diversity): trades relevance vs redundancy ──
export function mmrRerank(cand: number[], rel: (i: number) => number, sim: (i: number, j: number) => number, lambda: number, k: number): number[] {
  const pool = [...cand]; const selected: number[] = [];
  while (selected.length < Math.min(k, cand.length) && pool.length) {
    let best = pool[0], bestScore = -Infinity;
    for (const i of pool) { const div = selected.length ? Math.max(...selected.map((j) => sim(i, j))) : 0; const score = lambda * rel(i) - (1 - lambda) * div; if (score > bestScore) { bestScore = score; best = i; } }
    selected.push(best); pool.splice(pool.indexOf(best), 1);
  }
  return selected;
}

// ── retrieval-quality metrics vs a labelled relevant set ──
export function retrievalMetrics(ranked: number[], relevant: Set<number>, k: number): { p: number; r: number; mrr: number; ndcg: number } {
  const topK = ranked.slice(0, k);
  const hits = topK.filter((i) => relevant.has(i)).length;
  const p = k ? hits / k : 0;
  const r = relevant.size ? hits / relevant.size : 0;
  let mrr = 0; for (let i = 0; i < ranked.length; i++) if (relevant.has(ranked[i])) { mrr = 1 / (i + 1); break; }
  let dcg = 0; topK.forEach((i, rank) => { if (relevant.has(i)) dcg += 1 / Math.log2(rank + 2); });
  let idcg = 0; for (let rank = 0; rank < Math.min(k, relevant.size); rank++) idcg += 1 / Math.log2(rank + 2);
  return { p, r, mrr, ndcg: idcg ? dcg / idcg : 0 };
}

// ── PCA to 2-D (power iteration) for visualising the embedding space ──
export function pca2(vectors: number[][]): { x: number; y: number }[] {
  const n = vectors.length, d = vectors[0]?.length || 0;
  if (!n || !d) return vectors.map(() => ({ x: 0, y: 0 }));
  const mean = new Array(d).fill(0); vectors.forEach((v) => v.forEach((x, j) => { mean[j] += x / n; }));
  const cen = vectors.map((v) => v.map((x, j) => x - mean[j]));
  const covMul = (vec: number[]) => { const Xv = cen.map((row) => row.reduce((a, x, j) => a + x * vec[j], 0)); const out = new Array(d).fill(0); cen.forEach((row, i) => row.forEach((x, j) => { out[j] += x * Xv[i] / n; })); return out; };
  const power = (deflate?: number[]) => { let v = new Array(d).fill(0).map((_, i) => Math.sin(i + 1)); for (let it = 0; it < 40; it++) { let nv = covMul(v); if (deflate) { const proj = nv.reduce((a, x, j) => a + x * deflate[j], 0); nv = nv.map((x, j) => x - proj * deflate[j]); } const nrm = Math.sqrt(nv.reduce((a, x) => a + x * x, 0)) || 1; v = nv.map((x) => x / nrm); } return v; };
  const pc1 = power(), pc2 = power(pc1);
  return cen.map((row) => ({ x: row.reduce((a, x, j) => a + x * pc1[j], 0), y: row.reduce((a, x, j) => a + x * pc2[j], 0) }));
}
