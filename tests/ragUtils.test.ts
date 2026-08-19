import { describe, it, expect } from "vitest";
import { chunkText, buildIndex, retrieve, tokenize, simSparse, simDense, type Metric } from "@/lib/ragUtils";

describe("chunkText", () => {
  it("returns a single chunk when text fits in one window", () => {
    expect(chunkText("one two three", 10, 2)).toEqual(["one two three"]);
  });

  it("splits into overlapping windows", () => {
    const words = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9";
    const chunks = chunkText(words, 4, 1); // step = 3
    expect(chunks.length).toBeGreaterThan(1);
    // consecutive chunks share the overlap word
    const last0 = chunks[0].split(" ").at(-1);
    const first1 = chunks[1].split(" ")[0];
    expect(last0).toBe(first1);
  });
});

describe("tokenize", () => {
  it("lowercases and strips punctuation", () => {
    expect(tokenize("Refund, please! (30 days)")).toEqual(["refund", "please", "30", "days"]);
  });
});

describe("buildIndex + retrieve", () => {
  const chunks = [
    "refund policy damaged items within thirty days full refund",
    "store hours nine to six on weekdays closed holidays",
    "shipping is free on orders over fifty dollars",
  ];
  const idx = buildIndex(chunks);

  it("ranks the most relevant chunk first (hybrid)", () => {
    const hits = retrieve(idx, "how do refunds for damaged items work", "hybrid", 1);
    expect(hits[0].i).toBe(0);
  });

  it("finds the shipping chunk for a shipping query", () => {
    const hits = retrieve(idx, "free shipping threshold dollars", "hybrid", 1);
    expect(hits[0].i).toBe(2);
  });

  it("returns at most k hits", () => {
    expect(retrieve(idx, "refund", "keyword", 2).length).toBeLessThanOrEqual(2);
  });

  it("ranks the refund chunk first under every vector metric", () => {
    for (const m of ["cosine", "dot", "euclidean"] as Metric[]) {
      const hits = retrieve(idx, "how do refunds for damaged items work", "vector", 1, m);
      expect(hits[0].i, `metric ${m}`).toBe(0);
    }
  });
});

describe("similarity metrics", () => {
  it("sparse: identical vectors are maximally similar; distance metrics stay finite", () => {
    const a = { x: 1, y: 2 };
    expect(simSparse(a, a, "cosine")).toBeCloseTo(1, 6);
    expect(simSparse(a, a, "euclidean")).toBeCloseTo(1, 6); // 1/(1+0)
    expect(simSparse(a, a, "dot")).toBeCloseTo(5, 6); // 1*1 + 2*2
    expect(simSparse(a, { z: 9 }, "euclidean")).toBeGreaterThan(0); // disjoint keys handled
  });

  it("dense: cosine ignores magnitude while dot rewards it", () => {
    const a = [1, 0], b = [2, 0];
    expect(simDense(a, b, "cosine")).toBeCloseTo(1, 6);
    expect(simDense(a, b, "dot")).toBeCloseTo(2, 6);
    expect(simDense(a, [0, 1], "cosine")).toBeCloseTo(0, 6);
  });
});
