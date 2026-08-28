// Corrective RAG (CRAG) — grade the retrieval before you trust it.
//
// The failure mode plain RAG has: it retrieves the top-k chunks no matter how
// bad they are, hands them to the LLM, and the LLM answers confidently from
// irrelevant context. CRAG puts an evaluator between retrieval and generation
// and branches on the grade:
//
//   CORRECT   → answer from what was retrieved
//   AMBIGUOUS → rewrite the query, retrieve again, combine
//   INCORRECT → discard the retrieval, fall back to an external source
//
// Everything here is deterministic and runs client-side. No LLM judge — the
// grader is a real lexical/semantic scorer so the decision is inspectable and
// reproducible, which is what makes it teachable.

import { type RagIndex, bm25Scores, queryVector, simSparse, tokenize, retrieve } from "./ragUtils";

export type Grade = "correct" | "ambiguous" | "incorrect";

export interface DocGrade {
  i: number;
  /** 0..1 — how well this chunk supports the query. */
  score: number;
  /** Share of query terms that actually appear in the chunk. */
  coverage: number;
  /** Cosine between query and chunk in the index's vector space. */
  similarity: number;
  verdict: "relevant" | "partial" | "irrelevant";
}

export interface CragThresholds {
  /** ≥ this on the best doc → CORRECT. */
  upper: number;
  /** < this on the best doc → INCORRECT. */
  lower: number;
}

export const DEFAULT_CRAG_THRESHOLDS: CragThresholds = { upper: 0.5, lower: 0.2 };

/**
 * Score each retrieved chunk against the query on two independent axes, so a
 * chunk that merely shares vocabulary can't pass on similarity alone:
 *   coverage   — how much of the question the chunk actually addresses
 *   similarity — how close it sits in vector space
 * The combined score weights coverage higher: a chunk missing most of the
 * query's terms is not evidence, however close its embedding happens to be.
 */
export function gradeDocs(idx: RagIndex, query: string, candidates: number[]): DocGrade[] {
  const qTerms = [...new Set(tokenize(query))];
  const qv = queryVector(idx, query);
  const bm = bm25Scores(idx, query);
  const bmMax = Math.max(...bm, 1e-9);

  return candidates.map((i) => {
    const docTerms = new Set(idx.docs[i] || []);
    const coverage = qTerms.length ? qTerms.filter((t) => docTerms.has(t)).length / qTerms.length : 0;
    const similarity = Math.max(0, simSparse(qv, idx.vectors[i] || {}, "cosine"));
    const lexical = (bm[i] || 0) / bmMax;
    // coverage 0.5 · similarity 0.3 · bm25 0.2
    const score = 0.5 * coverage + 0.3 * similarity + 0.2 * lexical;
    const verdict: DocGrade["verdict"] = score >= 0.5 ? "relevant" : score >= 0.2 ? "partial" : "irrelevant";
    return { i, score: Number(score.toFixed(4)), coverage: Number(coverage.toFixed(4)), similarity: Number(similarity.toFixed(4)), verdict };
  });
}

/** Turn the per-doc grades into the branch decision. */
export function gradeRetrieval(grades: DocGrade[], t: CragThresholds = DEFAULT_CRAG_THRESHOLDS): { grade: Grade; best: number; reason: string } {
  if (grades.length === 0) return { grade: "incorrect", best: 0, reason: "Retrieval returned nothing at all." };
  const best = Math.max(...grades.map((g) => g.score));
  const relevant = grades.filter((g) => g.verdict === "relevant").length;
  if (best >= t.upper) {
    return { grade: "correct", best, reason: `Top chunk scores ${best.toFixed(2)} (≥ ${t.upper}) — ${relevant} chunk(s) directly support the query.` };
  }
  if (best < t.lower) {
    return { grade: "incorrect", best, reason: `Best chunk only scores ${best.toFixed(2)} (< ${t.lower}) — nothing retrieved actually addresses the query.` };
  }
  return { grade: "ambiguous", best, reason: `Best chunk scores ${best.toFixed(2)} — partially on topic, but no chunk clearly answers the query.` };
}

/**
 * Rewrite an under-performing query using the corpus itself: keep the terms the
 * index actually knows, drop the ones that appear nowhere (they can only add
 * noise), and expand with the strongest co-occurring terms from the chunks that
 * did partially match. This is query expansion via pseudo-relevance feedback —
 * the classic non-LLM rewriter, and it makes the "why" visible.
 */
export function rewriteQuery(idx: RagIndex, query: string, partialHits: number[], maxExpansions = 3): { rewritten: string; kept: string[]; dropped: string[]; added: string[] } {
  const qTerms = [...new Set(tokenize(query))];
  const kept = qTerms.filter((t) => (idx.df[t] || 0) > 0);
  const dropped = qTerms.filter((t) => (idx.df[t] || 0) === 0);

  // score candidate expansion terms by how distinctive they are inside the
  // partially-matching chunks (tf) versus the corpus at large (idf)
  const cand: Record<string, number> = {};
  partialHits.forEach((i) => {
    const doc = idx.docs[i] || [];
    const tf: Record<string, number> = {};
    doc.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
    Object.entries(tf).forEach(([t, c]) => {
      if (kept.includes(t) || t.length < 4) return;
      cand[t] = (cand[t] || 0) + (c / doc.length) * Math.log(idx.N / (idx.df[t] || 1) + 1);
    });
  });

  const added = Object.entries(cand).sort((a, b) => b[1] - a[1]).slice(0, maxExpansions).map(([t]) => t);
  const rewritten = [...kept, ...added].join(" ");
  return { rewritten: rewritten || query, kept, dropped, added };
}

export interface CragStep {
  stage: "retrieve" | "grade" | "rewrite" | "re-retrieve" | "fallback" | "answer";
  detail: string;
  grade?: Grade;
  hits?: number[];
}

export interface CragResult {
  grade: Grade;
  finalHits: number[];
  /** Set when the grade was INCORRECT and the corpus was abandoned. */
  usedFallback: boolean;
  rewritten?: string;
  grades: DocGrade[];
  steps: CragStep[];
}

/**
 * The full corrective loop. `externalSearch` stands in for the web-search
 * fallback leg — when the corpus genuinely cannot answer, CRAG is supposed to
 * go outside it rather than answer from bad context.
 */
export function correctiveRetrieve(
  idx: RagIndex,
  query: string,
  k: number,
  opts: { thresholds?: CragThresholds; externalSearch?: (q: string) => number[] } = {},
): CragResult {
  const t = opts.thresholds || DEFAULT_CRAG_THRESHOLDS;
  const steps: CragStep[] = [];

  const first = retrieve(idx, query, "hybrid", k).map((r) => r.i);
  steps.push({ stage: "retrieve", detail: `Retrieved top-${k} for "${query}".`, hits: first });

  const grades = gradeDocs(idx, query, first);
  const { grade, reason } = gradeRetrieval(grades, t);
  steps.push({ stage: "grade", detail: reason, grade });

  if (grade === "correct") {
    steps.push({ stage: "answer", detail: "Retrieval trusted — generating from these chunks." });
    return { grade, finalHits: first, usedFallback: false, grades, steps };
  }

  if (grade === "ambiguous") {
    const partial = grades.filter((g) => g.verdict !== "irrelevant").map((g) => g.i);
    const rw = rewriteQuery(idx, query, partial.length ? partial : first);
    steps.push({
      stage: "rewrite",
      detail: `Rewrote query → "${rw.rewritten}"${rw.dropped.length ? ` · dropped unknown term(s): ${rw.dropped.join(", ")}` : ""}${rw.added.length ? ` · expanded with: ${rw.added.join(", ")}` : ""}`,
    });
    const second = retrieve(idx, rw.rewritten, "hybrid", k).map((r) => r.i);
    const merged = [...new Set([...second, ...partial])].slice(0, k);
    steps.push({ stage: "re-retrieve", detail: `Re-retrieved on the rewritten query and merged with the partial hits.`, hits: merged });
    steps.push({ stage: "answer", detail: "Generating from the corrected retrieval." });
    return { grade, finalHits: merged, usedFallback: false, rewritten: rw.rewritten, grades, steps };
  }

  // INCORRECT — the corpus cannot answer this. Do not generate from it.
  const external = opts.externalSearch ? opts.externalSearch(query) : [];
  steps.push({
    stage: "fallback",
    detail: external.length
      ? `Corpus rejected. Fell back to an external source and pulled ${external.length} result(s).`
      : "Corpus rejected and no external source is wired — answering is refused rather than hallucinated.",
    hits: external,
  });
  return { grade, finalHits: external, usedFallback: true, grades, steps };
}
