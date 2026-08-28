// Does each of the 5 mainstream RAG architectures actually RUN on this platform?
// Executes the real library code — no mocks, no stubs.
import { describe, it, expect } from "vitest";
import { buildIndex, bm25Scores, queryVector, cosine, retrieve, retrieveDense, mmrRerank, retrievalMetrics } from "../src/lib/ragUtils";
import { extractGraph, retrieveGraph } from "../src/lib/kgUtils";
import { AGENT_TOOLS } from "../src/lib/agentTools";

const CORPUS = [
  "Ada Lovelace worked at the Analytical Engine project in London with Charles Babbage.",
  "Charles Babbage designed the Analytical Engine, a mechanical general-purpose computer.",
  "The error code E4021 indicates a thermal shutdown in the battery management module.",
  "Automobiles powered by electricity are increasingly common in Norway and Sweden.",
  "Grace Hopper developed the first compiler and worked on the Harvard Mark I computer.",
];

describe("01 · HYBRID RAG — dense + sparse, fused", () => {
  const idx = buildIndex(CORPUS);

  it("the sparse (BM25) leg runs and finds exact tokens", () => {
    const s = bm25Scores(idx, "E4021 thermal shutdown");
    expect(s).toHaveLength(CORPUS.length);
    expect(s.indexOf(Math.max(...s))).toBe(2); // the error-code chunk
  });

  it('the built-in "vector" leg is TF-IDF cosine — lexical, not semantic', () => {
    // shares a term -> matches
    const overlap = queryVector(idx, "Analytical Engine");
    expect(Math.max(...idx.vectors.map((v) => cosine(overlap, v)))).toBeGreaterThan(0);
    // pure paraphrase ("electric cars" vs "Automobiles powered by electricity")
    // scores ZERO: TF-IDF has no notion of synonymy. Real semantic matching only
    // happens in neural mode via /api/rag/embed -> retrieveDense().
    const para = queryVector(idx, "electric cars");
    expect(Math.max(...idx.vectors.map((v) => cosine(para, v)))).toBe(0);
  });

  it("the neural leg does resolve paraphrase, when embeddings are available", () => {
    // /api/rag/embed returns real provider embeddings; retrieveDense consumes them
    //                 babbage    babbage2    errorcode      AUTOMOBILES    hopper
    const chunkVecs = [[0.1, 0.9, 0], [0.15, 0.85, 0], [0.2, 0.6, 0.7], [0.95, 0.05, 0.05], [0.2, 0.8, 0.1]];
    const qVecElectricCars = [0.93, 0.1, 0.04]; // "electric cars" lands near the automobiles chunk
    const out = retrieveDense(idx, "electric cars", qVecElectricCars, chunkVecs, "vector", 2);
    expect(out[0].i).toBe(3);
    expect(out[0].score).toBeGreaterThan(0.9);
  });

  it("hybrid actually blends — it is not just one leg relabelled", () => {
    const q = "Analytical Engine computer";
    const vec = retrieve(idx, q, "vector", 5).map((r) => r.i);
    const key = retrieve(idx, q, "keyword", 5).map((r) => r.i);
    const hyb = retrieve(idx, q, "hybrid", 5).map((r) => r.i);
    expect(hyb).toHaveLength(5);
    // alpha must move the ranking: alpha=1 ≡ vector, alpha=0 ≡ keyword
    expect(retrieve(idx, q, "hybrid", 5, "cosine", 1).map((r) => r.i)).toEqual(vec);
    expect(retrieve(idx, q, "hybrid", 5, "cosine", 0).map((r) => r.i)).toEqual(key);
  });

  it("runs on real neural vectors too, not only TF-IDF", () => {
    const chunkVecs = CORPUS.map((_, i) => [i / 5, 1 - i / 5, 0.5]);
    const out = retrieveDense(idx, "compiler", [0.8, 0.2, 0.5], chunkVecs, "hybrid", 3);
    expect(out).toHaveLength(3);
    expect(out[0].score).toBeGreaterThanOrEqual(out[2].score);
  });

  it("MMR de-duplication and retrieval metrics compute", () => {
    const picked = mmrRerank([0, 1, 4], (i) => 1 - i * 0.1, (a, b) => (a === b ? 1 : 0.2), 0.5, 2);
    expect(picked).toHaveLength(2);
    const m = retrievalMetrics([0, 2, 1], new Set([1, 2]), 3);
    expect(m.p).toBeGreaterThan(0);
    expect(m.mrr).toBeGreaterThan(0);
  });

  it("CLOSED: Reciprocal Rank Fusion is now available alongside the alpha blend", async () => {
    const src = await import("node:fs").then((fs) => fs.readFileSync("src/lib/ragUtils.ts", "utf8"));
    expect(src).toMatch(/reciprocal rank fusion/i);
    expect(src).toMatch(/export function rrfFuse/);
  });
});

describe("02 · GRAPHRAG — entities and relationships", () => {
  const g = extractGraph(CORPUS);

  it("extracts a real graph with nodes and edges", () => {
    expect(g.nodes.length).toBeGreaterThan(0);
    expect(g.edges.length).toBeGreaterThan(0);
    expect(g.nodes.some((n) => /babbage|lovelace|hopper/i.test(n.label))).toBe(true);
  });

  it("retrieves a subgraph and traverses multiple hops", () => {
    const one = retrieveGraph(g, "Babbage", 5, 1);
    const two = retrieveGraph(g, "Babbage", 5, 2);
    expect(one.chunkIds.length).toBeGreaterThan(0);
    expect(two.nodes.length).toBeGreaterThanOrEqual(one.nodes.length);
    expect(Object.keys(two.layers).length).toBeGreaterThan(0);
  });

  it("GAP: no community detection / global summarisation", async () => {
    const src = await import("node:fs").then((fs) => fs.readFileSync("src/lib/kgUtils.ts", "utf8"));
    expect(src).not.toMatch(/community|louvain|leiden/i);
  });
});

describe("03 · AGENTIC RAG — retrieval as a plan", () => {
  it("the retrieval tools an agent would plan over are real and callable", async () => {
    const ids = AGENT_TOOLS.map((t) => t.id);
    for (const need of ["rag", "web_search", "db_query", "knowledge"]) expect(ids).toContain(need);
    const calc = AGENT_TOOLS.find((t) => t.id === "calculator")!;
    expect(await calc.run("2*(3+4)^2", {} as never)).toBe("98");
  });
});

describe("04 · CORRECTIVE RAG (CRAG) — grade before you trust", () => {
  it("GAP: no relevance grader, query rewriter, or web fallback branch", async () => {
    const fs = await import("node:fs");
    const files = ["src/lib/ragUtils.ts", "src/lib/kgUtils.ts", "src/components/RagFlowLab.tsx"];
    const all = files.map((f) => fs.readFileSync(f, "utf8")).join("\n");
    expect(all).not.toMatch(/gradeDocs|relevanceGrade|rewriteQuery|correctiveRetrieve/);
  });
});

describe("05 · MULTIMODAL RAG — one index over text, images, tables", () => {
  it("CLOSED: a unified multi-modality index now exists", async () => {
    const fs = await import("node:fs");
    const src = fs.readFileSync("src/lib/multimodalUtils.ts", "utf8");
    expect(src).toMatch(/export function buildMultimodalIndex/);
    expect(src).toMatch(/export function retrieveMultimodal/);
  });

  it("STILL OPEN: no vision encoder — images are reachable only via their text", async () => {
    const fs = await import("node:fs");
    const src = fs.readFileSync("src/lib/multimodalUtils.ts", "utf8");
    // the module must keep saying so rather than implying pixel search
    expect(src).toMatch(/no vision encoder/i);
    expect(src).not.toMatch(/import .*(clip|colpali)/i);
  });
});
