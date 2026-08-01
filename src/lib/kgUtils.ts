// Client-side knowledge-graph construction + traversal retrieval for the RAG lab.
// Heuristic extraction (deterministic, no deps); an LLM path can supply cleaner triples.

export type KgNode = { id: string; label: string; type: "proper" | "concept"; freq: number; chunks: number[] };
export type KgEdge = { s: string; o: string; rel: string; chunks: number[]; weight: number };
export type KnowledgeGraph = { nodes: KgNode[]; edges: KgEdge[] };

const STOP = new Set("the a an and or but if then of to in on at for with from by as is are was were be been being this that these those it its they them their you your we our he she his her not no can may might will would should could must do does did has have had which who whom whose what when where why how than into over under out up down off within per each any all some more most other such only also very just been being your our".split(/\s+/));
const REL = /^(is|are|was|were|be|been|has|have|had|may|can|will|would|handled|issued|applies|apply|applied|include|includes|included|takes|take|taken|requires|require|required|covers|cover|covered|returned|refunded|start|arrive|within|for|to|of|by|on|over|under)$/;

const norm = (w: string) => w.toLowerCase().replace(/[^a-z0-9]/g, "");
const sentences = (t: string) => t.split(/(?<=[.!?])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
const wordsOf = (s: string) => s.match(/[A-Za-z][A-Za-z0-9'’-]*/g) || [];

// Heuristic entities (frequent non-stopword tokens + repeated bigrams) linked by intra-sentence co-occurrence.
export function extractGraph(chunkTexts: string[], opts?: { maxNodes?: number }): KnowledgeGraph {
  const maxNodes = opts?.maxNodes ?? 22;
  const freq = new Map<string, number>(), label = new Map<string, string>(), proper = new Set<string>();
  const chunkOf = new Map<string, Set<number>>();
  const add = (key: string, disp: string, ci: number) => {
    freq.set(key, (freq.get(key) || 0) + 1);
    if (!label.has(key)) label.set(key, disp.toLowerCase());
    let s = chunkOf.get(key); if (!s) { s = new Set(); chunkOf.set(key, s); } s.add(ci);
  };
  chunkTexts.forEach((ct, ci) => {
    for (const sent of sentences(ct)) {
      const ws = wordsOf(sent);
      for (let i = 0; i < ws.length; i++) {
        const w = ws[i], nw = norm(w); if (nw.length < 3 || STOP.has(nw)) continue;
        add(nw, w, ci); if (/^[A-Z]/.test(w)) proper.add(nw);
        if (i + 1 < ws.length) { const w2 = ws[i + 1], nw2 = norm(w2); if (nw2.length > 2 && !STOP.has(nw2)) { const key = nw + " " + nw2; add(key, w + " " + w2, ci); if (/^[A-Z]/.test(w)) proper.add(key); } }
      }
    }
  });
  // rank: bigrams boosted; drop single tokens already covered by a kept bigram
  const cand = [...freq.entries()].sort((a, b) => (b[1] * (b[0].includes(" ") ? 1.7 : 1)) - (a[1] * (a[0].includes(" ") ? 1.7 : 1)) || a[0].localeCompare(b[0]));
  const kept: string[] = [];
  for (const [k, c] of cand) { if (kept.length >= maxNodes) break; if (c < 1) continue; if (!k.includes(" ") && kept.some((kk) => kk.includes(" ") && kk.split(" ").includes(k))) continue; kept.push(k); }
  const keptSet = new Set(kept);
  const nodes: KgNode[] = kept.map((k) => ({ id: k, label: label.get(k)!, type: proper.has(k) ? "proper" : "concept", freq: freq.get(k)!, chunks: [...(chunkOf.get(k) || [])] }));
  const edgeMap = new Map<string, KgEdge>();
  chunkTexts.forEach((ct, ci) => {
    for (const sent of sentences(ct)) {
      const ws = wordsOf(sent), nws = ws.map(norm);
      const present: { key: string; pos: number; len: number }[] = [];
      for (let i = 0; i < nws.length; i++) { const bg = nws[i] + " " + (nws[i + 1] || ""); if (keptSet.has(bg)) { present.push({ key: bg, pos: i, len: 2 }); i++; } else if (keptSet.has(nws[i])) present.push({ key: nws[i], pos: i, len: 1 }); }
      for (let a = 0; a < present.length - 1; a++) {
        const A = present[a], B = present[a + 1]; if (A.key === B.key) continue;
        const mid = ws.slice(A.pos + A.len, B.pos).map((x) => x.toLowerCase());
        const rel = mid.filter((x) => REL.test(x)).slice(0, 3).join(" ") || "related to";
        const id = A.key + "→" + B.key; const e = edgeMap.get(id);
        if (e) { e.weight++; if (!e.chunks.includes(ci)) e.chunks.push(ci); } else edgeMap.set(id, { s: A.key, o: B.key, rel, chunks: [ci], weight: 1 });
      }
    }
  });
  return { nodes, edges: [...edgeMap.values()].filter((e) => keptSet.has(e.s) && keptSet.has(e.o)) };
}

// Build a graph from LLM-extracted (subject, relation, object) triples, mapping each to the chunks it appears in.
export function graphFromTriples(triples: { s: string; r: string; o: string }[], chunkTexts: string[]): KnowledgeGraph {
  const lc = chunkTexts.map((c) => c.toLowerCase());
  const nodeMap = new Map<string, KgNode>();
  const ensure = (raw: string) => { const id = norm(raw.trim().split(/\s+/).slice(0, 3).join(" ")) || norm(raw); const label = raw.trim().toLowerCase(); if (!id) return null; let n = nodeMap.get(id); if (!n) { n = { id, label, type: /^[A-Z]/.test(raw.trim()) ? "proper" : "concept", freq: 1, chunks: [] }; nodeMap.set(id, n); } else n.freq++; return n; };
  const chunksFor = (a: string, b: string) => lc.map((c, i) => (c.includes(a.toLowerCase()) || c.includes(b.toLowerCase()) ? i : -1)).filter((i) => i >= 0);
  const edges: KgEdge[] = [];
  for (const t of triples) { const S = ensure(t.s), O = ensure(t.o); if (!S || !O || S.id === O.id) continue; const ch = chunksFor(t.s, t.o); S.chunks = [...new Set([...S.chunks, ...ch])]; O.chunks = [...new Set([...O.chunks, ...ch])]; edges.push({ s: S.id, o: O.id, rel: (t.r || "related to").toLowerCase(), chunks: ch, weight: 1 }); }
  return { nodes: [...nodeMap.values()], edges };
}

// Entity-link the query → expand `hops` → collect chunks attached to the visited subgraph, ranked by coverage.
export function retrieveGraph(g: KnowledgeGraph, query: string, k: number, hops = 1): { chunkIds: number[]; path: KgEdge[]; seeds: string[]; nodes: string[] } {
  const q = new Set((query.toLowerCase().match(/[a-z0-9]+/g) || []));
  const seeds = g.nodes.filter((n) => q.has(n.id) || n.id.split(" ").some((t) => q.has(t))).map((n) => n.id);
  const visited = new Set(seeds), path: KgEdge[] = []; let frontier = new Set(seeds);
  for (let h = 0; h < hops; h++) {
    const next = new Set<string>();
    for (const e of g.edges) { if (frontier.has(e.s) && !visited.has(e.o)) { next.add(e.o); path.push(e); } if (frontier.has(e.o) && !visited.has(e.s)) { next.add(e.s); path.push(e); } }
    next.forEach((n) => visited.add(n)); frontier = next; if (!next.size) break;
  }
  for (const e of g.edges) if (seeds.includes(e.s) && seeds.includes(e.o) && !path.includes(e)) path.push(e);
  const score = new Map<number, number>();
  g.nodes.filter((n) => visited.has(n.id)).forEach((n) => n.chunks.forEach((ci) => score.set(ci, (score.get(ci) || 0) + 1)));
  const chunkIds = [...score.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map((e) => e[0]);
  return { chunkIds, path, seeds, nodes: [...visited] };
}

// Deterministic force-directed layout (seeded on a circle → no randomness) for the graph view.
export function layoutGraph(g: KnowledgeGraph, w: number, h: number): Record<string, { x: number; y: number }> {
  const nodes = g.nodes, n = nodes.length || 1, pos: Record<string, { x: number; y: number }> = {};
  nodes.forEach((nd, i) => { const a = (i / n) * 2 * Math.PI; pos[nd.id] = { x: w / 2 + Math.cos(a) * w * 0.31, y: h / 2 + Math.sin(a) * h * 0.33 }; });
  for (let it = 0; it < 90; it++) {
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) { const a = pos[nodes[i].id], b = pos[nodes[j].id]; let dx = a.x - b.x, dy = a.y - b.y; const d = Math.sqrt(dx * dx + dy * dy) || 1; const f = 1400 / (d * d); dx /= d; dy /= d; a.x += dx * f; a.y += dy * f; b.x -= dx * f; b.y -= dy * f; }
    for (const e of g.edges) { const a = pos[e.s], b = pos[e.o]; if (!a || !b) continue; let dx = b.x - a.x, dy = b.y - a.y; const d = Math.sqrt(dx * dx + dy * dy) || 1; const f = (d - 95) * 0.02; dx /= d; dy /= d; a.x += dx * f; a.y += dy * f; b.x -= dx * f; b.y -= dy * f; }
    for (const nd of nodes) { const p = pos[nd.id]; p.x = Math.max(34, Math.min(w - 34, p.x)); p.y = Math.max(26, Math.min(h - 26, p.y)); }
  }
  return pos;
}
