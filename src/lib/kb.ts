import "server-only";
import { chunkText, buildIndex, queryVector, cosine, type RagIndex, type Vec } from "./ragUtils";

// Best-guess embedding model for an OpenAI-compatible provider base URL.
export function pickEmbModel(baseUrl: string): string {
  if (/generativelanguage|gemini/i.test(baseUrl)) return "text-embedding-004";
  return "text-embedding-3-small";
}

// Call the provider's OpenAI-compatible /embeddings endpoint. Throws on failure
// (the caller falls back to TF-IDF).
export async function embedTexts(baseUrl: string, apiKey: string, model: string, texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  const B = 64;
  for (let i = 0; i < texts.length; i += B) {
    const batch = texts.slice(i, i + B);
    const res = await fetch(baseUrl.replace(/\/$/, "") + "/embeddings", {
      method: "POST",
      headers: { "content-type": "application/json", ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) },
      body: JSON.stringify({ model, input: batch }),
    });
    if (!res.ok) throw new Error(`embeddings ${res.status}`);
    const j = await res.json() as { data?: { embedding: number[] }[] };
    const data = j.data || [];
    if (data.length !== batch.length) throw new Error("embeddings count mismatch");
    for (const d of data) out.push(d.embedding);
  }
  return out;
}

function denseCos(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

export function chunkDocs(docs: { name: string; text: string }[]): { docName: string; text: string }[] {
  const out: { docName: string; text: string }[] = [];
  for (const d of docs) for (const c of chunkText(d.text, 60, 12)) out.push({ docName: d.name, text: c });
  return out;
}

// Vectorize chunks: real embeddings when available, else TF-IDF (always works).
export async function embedChunks(chunkTexts: string[], provider: { baseUrl: string; apiKey: string }): Promise<{ embModel: string; embMeta: unknown; vectors: unknown[] }> {
  const model = pickEmbModel(provider.baseUrl);
  try {
    const vectors = await embedTexts(provider.baseUrl, provider.apiKey, model, chunkTexts);
    return { embModel: model, embMeta: null, vectors };
  } catch {
    const idx = buildIndex(chunkTexts);
    return { embModel: "tfidf", embMeta: { df: idx.df, N: idx.N }, vectors: idx.vectors };
  }
}

// Retrieve the top-k chunk texts for a query from a KB's stored vectors.
export async function retrieve(
  kb: { embModel: string | null; embMeta: unknown },
  chunks: { text: string | null; embedding: unknown }[],
  query: string,
  provider: { baseUrl: string; apiKey: string },
  k = 4,
): Promise<string[]> {
  if (!chunks.length) return [];
  let scored: { text: string; score: number }[];
  if (kb.embModel && kb.embModel !== "tfidf") {
    let qv: number[];
    try { qv = (await embedTexts(provider.baseUrl, provider.apiKey, kb.embModel, [query]))[0]; }
    catch { return []; }
    scored = chunks.map((c) => ({ text: c.text || "", score: denseCos(qv, (c.embedding as number[]) || []) }));
  } else {
    const meta = (kb.embMeta as { df: Record<string, number>; N: number }) || { df: {}, N: 1 };
    const idx = { df: meta.df, N: meta.N, avgdl: 1, docs: [], vectors: [] } as unknown as RagIndex;
    const qv = queryVector(idx, query);
    scored = chunks.map((c) => ({ text: c.text || "", score: cosine(qv, (c.embedding as Vec) || {}) }));
  }
  return scored.sort((a, b) => b.score - a.score).slice(0, k).filter((s) => s.score > 0).map((s) => s.text);
}
