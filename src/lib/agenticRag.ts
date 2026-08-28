// Agentic RAG — retrieval as a plan, not a single step.
//
// Plain RAG runs one retriever once. Agentic RAG puts a planner in front: it
// inspects the question, picks the retriever most likely to answer it, checks
// whether what came back is good enough, and if not tries a different one. The
// loop ends when the evidence clears a confidence bar or the step budget runs out.
//
// The planner here is a real deterministic router rather than an LLM call, so
// the routing decision is inspectable and reproducible — you can see exactly
// which signal in the question sent it to BM25 rather than the graph. A
// production system would swap this for a model; the loop shape is identical.

import { type RagIndex, retrieve } from "./ragUtils";
import { type KnowledgeGraph, retrieveGraph } from "./kgUtils";
import { gradeDocs, gradeRetrieval, type DocGrade } from "./cragUtils";

export type RetrieverId = "keyword" | "vector" | "graph";

export interface RouteSignal {
  retriever: RetrieverId;
  why: string;
  /** 0..1 — how strongly the question matches this retriever's profile. */
  strength: number;
}

const EXACT_TOKEN = /\b(?:[A-Z]{1,4}[-_]?\d{2,}|\d+(?:\.\d+)?%|v?\d+\.\d+(?:\.\d+)?|\$\d)/;
const RELATION_WORD = /\b(who|whom|whose|between|connect(?:s|ed|ion)?|relat(?:e|ed|ionship)|works?\s+(?:at|on|with)|link(?:s|ed)?|associat)/i;

/** Which graph entity labels (if any) the query actually names. */
export function matchedEntities(query: string, labels: string[]): string[] {
  const q = ` ${query.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ")} `;
  return labels
    .filter((l) => l && l.length > 2 && q.includes(` ${l.toLowerCase()} `))
    .sort((a, b) => b.length - a.length);
}

/**
 * The raw signals the planner reads, exposed so the UI can show exactly what
 * fired without re-implementing (and drifting from) these patterns.
 */
export function detectSignals(query: string, entityLabels: string[] = []): {
  named: string | null; relational: string | null; exact: string | null;
} {
  return {
    named: matchedEntities(query, entityLabels)[0] ?? null,
    relational: RELATION_WORD.exec(query)?.[0] ?? null,
    exact: EXACT_TOKEN.exec(query)?.[0] ?? null,
  };
}

/**
 * Rank the available retrievers for this question. Signals, strongest first:
 *   names a graph entity   → graph; its neighbours are one hop away. This is the
 *     signal a phrasing-only router misses: "Grace Hopper" is a graph question
 *     despite containing no relationship words at all.
 *   relationship language  → graph; the answer is an edge, which no single chunk holds
 *   exact tokens (error codes, versions, percentages) → BM25; embeddings blur
 *     rare literals, keyword search does not
 *   none of the above      → vector, for paraphrase
 */
export function planRetrievers(query: string, opts: { hasGraph?: boolean; entityLabels?: string[] } = {}): RouteSignal[] {
  const out: RouteSignal[] = [];
  const exact = EXACT_TOKEN.exec(query);
  const relational = RELATION_WORD.exec(query);
  // Naming a node IS the strongest graph signal — "Grace Hopper" is a graph
  // question even though it contains no relationship words at all.
  const named = matchedEntities(query, opts.entityLabels || []);

  out.push(
    exact
      ? { retriever: "keyword", why: `Contains the literal "${exact[0]}" — an exact token embeddings tend to blur.`, strength: 0.9 }
      : { retriever: "keyword", why: "No exact literals; keyword search is a fallback here.", strength: 0.35 },
  );
  if (opts.hasGraph) {
    if (named.length && relational) {
      out.push({ retriever: "graph", why: `Names the entity "${named[0]}" AND uses relationship phrasing ("${relational[0]}") — the answer is almost certainly an edge.`, strength: 0.95 });
    } else if (named.length) {
      out.push({ retriever: "graph", why: `Names "${named[0]}", which is a node in the graph — its neighbours are reachable in one hop.`, strength: 0.8 });
    } else if (relational) {
      out.push({ retriever: "graph", why: `Relationship phrasing ("${relational[0]}") — the answer is likely an edge, not a chunk.`, strength: 0.7 });
    } else {
      out.push({ retriever: "graph", why: "Names no known entity and uses no relationship phrasing; the graph is unlikely to add anything.", strength: 0.25 });
    }
  }
  out.push({
    retriever: "vector",
    why: exact || relational || named.length
      ? "General semantic match — a reasonable second opinion."
      : "Open-ended question with no literals and no known entity: semantic similarity is the best first attempt.",
    strength: exact || relational || named.length ? 0.5 : 0.8,
  });

  return out.sort((a, b) => b.strength - a.strength);
}

export interface AgenticStep {
  n: number;
  retriever: RetrieverId;
  why: string;
  hits: number[];
  best: number;
  verdict: "accepted" | "rejected";
  note: string;
  grades: DocGrade[];
}

export interface AgenticResult {
  steps: AgenticStep[];
  finalHits: number[];
  confident: boolean;
  /** Which retriever produced the accepted evidence. */
  answeredBy?: RetrieverId;
}

/**
 * Run the plan until the evidence clears `threshold` or the budget is spent.
 * Each attempt is graded with the same CRAG scorer, so "confident" means the
 * retrieved chunks actually cover the question — not merely that something
 * came back.
 */
export function runAgenticRetrieval(
  idx: RagIndex,
  query: string,
  k: number,
  opts: { graph?: KnowledgeGraph; threshold?: number; maxSteps?: number; hops?: number } = {},
): AgenticResult {
  const { graph, threshold = 0.5, maxSteps = 3, hops = 1 } = opts;
  const plan = planRetrievers(query, { hasGraph: !!graph, entityLabels: graph?.nodes.map((n) => n.label) });
  const steps: AgenticStep[] = [];

  for (let n = 0; n < Math.min(maxSteps, plan.length); n++) {
    const { retriever, why } = plan[n];
    let hits: number[] = [];
    if (retriever === "graph" && graph) hits = retrieveGraph(graph, query, k, hops).chunkIds;
    else if (retriever === "keyword") hits = retrieve(idx, query, "keyword", k).map((r) => r.i);
    else hits = retrieve(idx, query, "vector", k).map((r) => r.i);

    const grades = gradeDocs(idx, query, hits);
    const { best } = gradeRetrieval(grades);
    const accepted = best >= threshold;
    steps.push({
      n: n + 1,
      retriever,
      why,
      hits,
      best,
      grades,
      verdict: accepted ? "accepted" : "rejected",
      note: accepted
        ? `Evidence scores ${best.toFixed(2)} ≥ ${threshold} — good enough, stopping.`
        : hits.length === 0
        ? "Returned nothing — trying the next retriever."
        : `Evidence only scores ${best.toFixed(2)} — below the bar, trying the next retriever.`,
    });

    if (accepted) return { steps, finalHits: hits, confident: true, answeredBy: retriever };
  }

  // Nothing cleared the bar. Return the best attempt but say so.
  const bestStep = [...steps].sort((a, b) => b.best - a.best)[0];
  return { steps, finalHits: bestStep?.hits || [], confident: false, answeredBy: bestStep?.retriever };
}
