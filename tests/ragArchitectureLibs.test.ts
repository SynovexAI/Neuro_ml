// The two architectures the platform was missing, plus RRF — real algorithms.
import { describe, it, expect } from "vitest";
import { buildIndex, rrfFuse, hybridRetrieve, bm25ScoresTuned, bm25Explain } from "../src/lib/ragUtils";
import { gradeDocs, gradeRetrieval, rewriteQuery, correctiveRetrieve } from "../src/lib/cragUtils";
import { planRetrievers, runAgenticRetrieval, detectSignals, matchedEntities } from "../src/lib/agenticRag";
import { extractGraph, forceLayout, graphComponents, type KnowledgeGraph } from "../src/lib/kgUtils";
import { parseTable, tableToRowTexts, buildMultimodalIndex, retrieveMultimodal, sampleMultimodalCorpus } from "../src/lib/multimodalUtils";

const CORPUS = [
  "The Analytical Engine was designed by Charles Babbage as a mechanical general purpose computer.",
  "Ada Lovelace wrote the first algorithm intended for the Analytical Engine.",
  "Error code E4021 indicates a thermal shutdown in the battery management module.",
  "Grace Hopper built the first compiler and worked on the Harvard Mark I.",
  "Norway leads Europe in electric vehicle adoption per capita.",
];

describe("RRF — rank fusion without score normalisation", () => {
  it("sums 1/(k+rank) and needs no comparable scales", () => {
    const sparse = [{ i: 7, score: 91.4 }, { i: 2, score: 40.1 }]; // BM25, unbounded
    const dense = [{ i: 2, score: 0.81 }, { i: 9, score: 0.42 }];  // cosine, [-1,1]
    const out = rrfFuse([sparse, dense], 60, 5);
    // doc 2 appears in BOTH lists, so it wins despite topping neither
    expect(out[0].i).toBe(2);
    expect(out[0].score).toBeCloseTo(1 / 62 + 1 / 61, 6);
    expect(out[0].ranks).toEqual([2, 1]);
    // a doc in only one list still keeps its single contribution
    expect(out.find((r) => r.i === 7)!.ranks).toEqual([1, null]);
  });

  it("is immune to score scale, unlike the alpha blend", () => {
    const a = [{ i: 1, score: 5000 }, { i: 2, score: 4999 }];
    const b = [{ i: 2, score: 0.9 }, { i: 1, score: 0.1 }];
    const scaled = [{ i: 1, score: 5 }, { i: 2, score: 4.999 }];
    expect(rrfFuse([a, b]).map((r) => r.i)).toEqual(rrfFuse([scaled, b]).map((r) => r.i));
  });

  it("hybridRetrieve exposes both legs and honours the fusion choice", () => {
    const idx = buildIndex(CORPUS);
    const r = hybridRetrieve(idx, "Analytical Engine Babbage", 3, { fusion: "rrf" });
    expect(r.fusion).toBe("rrf");
    expect(r.sparse.length).toBeGreaterThan(0);
    expect(r.dense.length).toBeGreaterThan(0);
    expect(r.fused[0].ranks).toBeDefined();
    expect(hybridRetrieve(idx, "Analytical Engine", 3, { fusion: "alpha", alpha: 0.5 }).fusion).toBe("alpha");
  });
});

describe("CRAG — grade the retrieval before trusting it", () => {
  const idx = buildIndex(CORPUS);

  it("grades a well-supported query CORRECT", () => {
    const g = gradeDocs(idx, "Analytical Engine Babbage", [0, 1, 2]);
    expect(g[0].verdict).toBe("relevant");
    expect(gradeRetrieval(g).grade).toBe("correct");
  });

  it("grades an off-corpus query INCORRECT and refuses to generate", () => {
    const res = correctiveRetrieve(idx, "photosynthesis chlorophyll stomata", 3);
    expect(res.grade).toBe("incorrect");
    expect(res.usedFallback).toBe(true);
    expect(res.finalHits).toEqual([]); // nothing to answer from — better than hallucinating
    expect(res.steps.map((s) => s.stage)).toContain("fallback");
  });

  it("uses the external leg when one is wired", () => {
    const res = correctiveRetrieve(idx, "photosynthesis chlorophyll stomata", 3, { externalSearch: () => [42] });
    expect(res.usedFallback).toBe(true);
    expect(res.finalHits).toEqual([42]);
  });

  it("rewrites a partial query using corpus feedback", () => {
    const rw = rewriteQuery(idx, "Babbage quantumcomputing", [0]);
    expect(rw.kept).toContain("babbage");
    expect(rw.dropped).toContain("quantumcomputing"); // appears nowhere in the corpus
    expect(rw.added.length).toBeGreaterThan(0);
    expect(rw.rewritten).not.toMatch(/quantumcomputing/);
  });

  it("the AMBIGUOUS branch actually re-retrieves", () => {
    const res = correctiveRetrieve(idx, "compiler Hopper spacecraft navigation", 3, { thresholds: { upper: 0.9, lower: 0.05 } });
    expect(res.grade).toBe("ambiguous");
    expect(res.rewritten).toBeTruthy();
    expect(res.steps.map((s) => s.stage)).toEqual(["retrieve", "grade", "rewrite", "re-retrieve", "answer"]);
  });
});

describe("Multimodal RAG — one index over text, tables and images", () => {
  it("serialises table rows with their headers so they are retrievable", () => {
    const rows = parseTable("Quarter,Revenue\nQ4 2024,$2.42M");
    expect(rows).toEqual([["Quarter", "Revenue"], ["Q4 2024", "$2.42M"]]);
    expect(tableToRowTexts(rows, "Results")[0]).toBe("Results · Quarter: Q4 2024 · Revenue: $2.42M");
  });

  it("ranks a table row above prose when the answer is in the table", () => {
    const mi = buildMultimodalIndex(sampleMultimodalCorpus());
    const { hits } = retrieveMultimodal(mi, "Q4 2024 revenue margin", 3);
    expect(hits[0].modality).toBe("table");
    expect(hits[0].item.content).toMatch(/2\.42M/);
  });

  it("reaches an answer that only exists in a figure", () => {
    const mi = buildMultimodalIndex(sampleMultimodalCorpus());
    const { hits } = retrieveMultimodal(mi, "pie chart revenue by region APAC share", 2);
    expect(hits[0].modality).toBe("image");
    expect(hits[0].item.title).toMatch(/regional-split/);
  });

  it("can be filtered to one modality, and reports the mix", () => {
    const mi = buildMultimodalIndex(sampleMultimodalCorpus());
    const all = retrieveMultimodal(mi, "revenue 2024", 5);
    expect(all.perModality.table + all.perModality.text + all.perModality.image).toBe(all.hits.length);
    const only = retrieveMultimodal(mi, "revenue 2024", 5, { only: ["image"] });
    expect(only.hits.every((h) => h.modality === "image")).toBe(true);
  });
});

describe("Agentic RAG — the planner routes on real signals", () => {
  const idx = buildIndex(CORPUS);

  it("routes exact literals to BM25, not embeddings", () => {
    const plan = planRetrievers("what does error E4021 mean");
    expect(plan[0].retriever).toBe("keyword");
    expect(plan[0].why).toMatch(/E4021/);
  });

  it("routes relationship questions to the graph", () => {
    const plan = planRetrievers("who works with Charles Babbage", { hasGraph: true });
    expect(plan[0].retriever).toBe("graph");
  });

  it("routes open questions to vector search", () => {
    expect(planRetrievers("tell me about early computing")[0].retriever).toBe("vector");
  });

  it("stops as soon as evidence clears the bar", () => {
    const res = runAgenticRetrieval(idx, "error E4021 thermal shutdown", 3);
    expect(res.confident).toBe(true);
    expect(res.answeredBy).toBe("keyword");
    expect(res.steps).toHaveLength(1);
    expect(res.steps[0].verdict).toBe("accepted");
  });

  it("tries another retriever when the first is rejected, and admits defeat honestly", () => {
    const res = runAgenticRetrieval(idx, "photosynthesis chlorophyll stomata", 3, { threshold: 0.9 });
    expect(res.confident).toBe(false);
    expect(res.steps.length).toBeGreaterThan(1);
    expect(res.steps.every((s) => s.verdict === "rejected")).toBe(true);
  });
});

describe("BM25 knobs actually change the ranking", () => {
  const docs = ["alpha alpha alpha alpha beta", "alpha beta gamma delta epsilon zeta eta theta iota kappa"];
  const idx2 = buildIndex(docs);

  it("k1 controls term-frequency saturation", () => {
    const low = bm25ScoresTuned(idx2, "alpha", 0.1, 0.75);
    const high = bm25ScoresTuned(idx2, "alpha", 3.0, 0.75);
    // doc 0 repeats "alpha" 4x; raising k1 should widen its lead over doc 1
    expect(high[0] - high[1]).toBeGreaterThan(low[0] - low[1]);
  });

  it("b controls length normalisation", () => {
    const noNorm = bm25ScoresTuned(idx2, "beta", 1.5, 0);
    const fullNorm = bm25ScoresTuned(idx2, "beta", 1.5, 1);
    // both contain "beta" once; with b=0 length is ignored so scores match
    expect(noNorm[0]).toBeCloseTo(noNorm[1], 6);
    // with b=1 the longer doc 1 is penalised
    expect(fullNorm[1]).toBeLessThan(fullNorm[0]);
  });

  it("explain sums to the tuned score", () => {
    const ex = bm25Explain(idx2, "alpha beta", 0, 1.5, 0.75);
    expect(ex.total).toBeCloseTo(bm25ScoresTuned(idx2, "alpha beta", 1.5, 0.75)[0], 6);
    expect(ex.terms.map((t) => t.term)).toEqual(["alpha", "beta"]);
    expect(ex.terms[0].tf).toBe(4);
  });
});

describe("forceLayout — the knowledge graph must be readable", () => {
  const g = extractGraph([
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine project in London.",
    "Charles Babbage designed the Analytical Engine, a mechanical general purpose computer.",
    "Grace Hopper developed the first compiler and worked on the Harvard Mark I computer.",
    "Error code E4021 indicates a thermal shutdown in the battery management module.",
  ]);

  it("places every node inside the frame", () => {
    const pos = forceLayout(g);
    expect(Object.keys(pos)).toHaveLength(g.nodes.length);
    for (const p of Object.values(pos)) {
      expect(p.x).toBeGreaterThanOrEqual(0); expect(p.x).toBeLessThanOrEqual(1);
      expect(p.y).toBeGreaterThanOrEqual(0); expect(p.y).toBeLessThanOrEqual(1);
    }
  });

  it("is deterministic — the same graph lays out identically", () => {
    expect(forceLayout(g)).toEqual(forceLayout(g));
  });

  it("separates nodes instead of stacking them", () => {
    const pos = forceLayout(g);
    const ids = g.nodes.map((n) => n.id);
    let worst = Infinity;
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const a = pos[ids[i]], b = pos[ids[j]];
      worst = Math.min(worst, Math.hypot(a.x - b.x, a.y - b.y));
    }
    // the old ring layout produced near-zero separations for dense graphs
    expect(worst).toBeGreaterThan(0.01);
  });

  it("fills the frame without distorting clusters", () => {
    const pos = forceLayout(g);
    const xs = Object.values(pos).map((p) => p.x), ys = Object.values(pos).map((p) => p.y);
    const spanX = Math.max(...xs) - Math.min(...xs), spanY = Math.max(...ys) - Math.min(...ys);
    // scaling is UNIFORM, so the longer axis fills the frame and the shorter
    // keeps its true proportion — stretching both to 1 would skew every cluster
    expect(Math.max(spanX, spanY)).toBeGreaterThan(0.8);
    expect(Math.min(spanX, spanY)).toBeGreaterThan(0.25);
  });

  it("pulls connected nodes closer than unconnected ones on average", () => {
    const pos = forceLayout(g);
    const d = (a: string, b: string) => Math.hypot(pos[a].x - pos[b].x, pos[a].y - pos[b].y);
    const linked = g.edges.filter((e) => pos[e.s] && pos[e.o]).map((e) => d(e.s, e.o));
    const ids = g.nodes.map((n) => n.id);
    const all: number[] = [];
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) all.push(d(ids[i], ids[j]));
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
    expect(mean(linked)).toBeLessThan(mean(all));
  });

  it("handles the degenerate cases", () => {
    expect(forceLayout({ nodes: [], edges: [] })).toEqual({});
    const one = forceLayout({ nodes: [{ id: "a", label: "A", type: "proper", freq: 1, chunks: [0] }], edges: [] });
    expect(one.a).toEqual({ x: 0.5, y: 0.5 });
  });
});

describe("layout must not collapse onto the frame walls", () => {
  // the real failure: many small disconnected components + a few clusters
  const many: KnowledgeGraph = {
    nodes: Array.from({ length: 40 }, (_, i) => ({ id: `n${i}`, label: `n${i}`, type: (i % 2 ? "proper" : "concept") as "proper" | "concept", freq: 1 + (i % 3), chunks: [i % 5] })),
    edges: [
      // one 6-node cluster, one 3-node cluster, the rest isolated pairs / singletons
      { s: "n0", o: "n1", rel: "r", chunks: [0], weight: 2 }, { s: "n1", o: "n2", rel: "r", chunks: [0], weight: 1 },
      { s: "n2", o: "n3", rel: "r", chunks: [0], weight: 1 }, { s: "n3", o: "n4", rel: "r", chunks: [0], weight: 1 },
      { s: "n4", o: "n5", rel: "r", chunks: [0], weight: 1 },
      { s: "n10", o: "n11", rel: "r", chunks: [1], weight: 1 }, { s: "n11", o: "n12", rel: "r", chunks: [1], weight: 1 },
      { s: "n20", o: "n21", rel: "r", chunks: [2], weight: 1 },
    ],
  };

  it("finds the connected components, largest first", () => {
    const comps = graphComponents(many);
    expect(comps[0]).toHaveLength(6);
    expect(comps[1]).toHaveLength(3);
    expect(comps[2]).toHaveLength(2);
    expect(comps.reduce((a, c) => a + c.length, 0)).toBe(40);
  });

  it("does not pile nodes onto the frame border", () => {
    const pos = forceLayout(many);
    const onEdge = Object.values(pos).filter((p) => p.x <= 0.03 || p.x >= 0.97 || p.y <= 0.03 || p.y >= 0.97);
    // the old global simulation clamped a large share of nodes to the walls
    expect(onEdge.length / 40).toBeLessThan(0.2);
  });

  it("uses the interior of the frame in both directions", () => {
    const pos = forceLayout(many);
    const xs = Object.values(pos).map((p) => p.x), ys = Object.values(pos).map((p) => p.y);
    const spanX = Math.max(...xs) - Math.min(...xs), spanY = Math.max(...ys) - Math.min(...ys);
    expect(Math.max(spanX, spanY)).toBeGreaterThan(0.8);
    expect(Math.min(spanX, spanY)).toBeGreaterThan(0.4);
  });

  it("keeps each cluster together rather than scattering it", () => {
    const pos = forceLayout(many);
    const d = (a: string, b: string) => Math.hypot(pos[a].x - pos[b].x, pos[a].y - pos[b].y);
    // members of the 6-node cluster should sit closer to each other than to a far singleton
    const within = d("n0", "n5");
    const across = d("n0", "n39");
    expect(within).toBeLessThan(across);
  });
});

describe("planner routes an entity name to the graph", () => {
  const g = extractGraph([
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine project in London.",
    "Grace Hopper developed the first compiler and worked on the Harvard Mark I computer.",
    "Error code E4021 indicates a thermal shutdown in the battery management module.",
  ]);
  const labels = g.nodes.map((n) => n.label);

  it("detects a bare entity name with no relationship words", () => {
    // the phrasing-only router ranked graph LAST for this, at 0.25
    const sig = detectSignals("Grace Hopper", labels);
    expect(sig.named).toBe("grace hopper");
    expect(sig.relational).toBeNull();
    expect(sig.exact).toBeNull();
  });

  it("ranks the graph first for a named entity", () => {
    const plan = planRetrievers("Grace Hopper", { hasGraph: true, entityLabels: labels });
    expect(plan[0].retriever).toBe("graph");
    expect(plan[0].why).toMatch(/node in the graph/i);
  });

  it("ranks it highest of all when the query names an entity AND asks about relations", () => {
    const plan = planRetrievers("who worked with Charles Babbage", { hasGraph: true, entityLabels: labels });
    expect(plan[0].retriever).toBe("graph");
    expect(plan[0].strength).toBeGreaterThan(0.9);
  });

  it("still prefers keyword for an exact literal, even beside an entity name", () => {
    expect(planRetrievers("E4021 thermal shutdown", { hasGraph: true, entityLabels: labels })[0].retriever).toBe("keyword");
  });

  it("falls back to vector when nothing is named and nothing is literal", () => {
    const plan = planRetrievers("tell me about early computing", { hasGraph: true, entityLabels: labels });
    expect(plan[0].retriever).toBe("vector");
  });

  it("matches whole entity names only, not fragments", () => {
    expect(matchedEntities("gracefully handling errors", labels)).not.toContain("grace hopper");
    expect(matchedEntities("london", labels)).toContain("london");
  });
});
