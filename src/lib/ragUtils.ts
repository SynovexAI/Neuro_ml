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

// Normalize a score array to 0..1 for display / hybrid blending.
function norm(arr: number[]): number[] {
  const max = Math.max(...arr, 1e-9);
  return arr.map((x) => x / max);
}

export type Strategy = "vector" | "keyword" | "hybrid";

export function retrieve(idx: RagIndex, query: string, strategy: Strategy, k: number): { i: number; score: number }[] {
  const bm = norm(bm25Scores(idx, query));
  const qv = queryVector(idx, query);
  const vec = idx.vectors.map((v) => cosine(qv, v));
  let scores: number[];
  if (strategy === "keyword") scores = bm;
  else if (strategy === "vector") scores = vec;
  else scores = bm.map((s, i) => 0.5 * s + 0.5 * vec[i]);
  return scores
    .map((s, i) => ({ i, score: s }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}
