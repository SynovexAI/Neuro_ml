"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { chunkBy, pca2, CHUNK_STRATEGY_LABEL, type ChunkStrategy } from "@/lib/ragUtils";
import { toast } from "@/lib/toast";

// ─────────────────────────────────────────────────────────────────────────────
// Practice Playground — a teaching workbench that sits beside Steps / Canvas.
//
// Each module lets a student poke one step of the RAG (and training) pipeline on
// their own text. Everything here runs on REAL math — deterministic demo
// embeddings, real mean/max/CLS pooling, L2 normalization, PCA projection, cosine
// similarity, and real gradient descent — so the numbers on screen are the numbers
// the formulas produce. Embeddings are a small deterministic "Demo-Embed" model
// (not a neural network call) so the lab is instant, offline, and reproducible.
// ─────────────────────────────────────────────────────────────────────────────

const ACC = "#7c5cff";       // playground accent (purple, matches the screenshots)
const BLUE = "#5b7cff";
const GREEN = "#3ecf7f";
const AMBER = "#f59e0b";
const PINK = "#e0559f";
const RED = "#f0616d";
const DIM_COLORS = [BLUE, "#22b8cf", GREEN, AMBER, PINK, ACC, RED, "#eab308"];

type ModuleId = "chunk" | "embed" | "pool" | "index" | "retrieve" | "backprop" | "e2e";
type Pooling = "mean" | "max" | "cls";

const MODULES: { id: ModuleId; icon: string; title: string; short: string; sub: string }[] = [
  { id: "chunk", icon: "🧩", title: "Chunk", short: "Chunk", sub: "Split text into chunks" },
  { id: "embed", icon: "🔮", title: "Embedding", short: "Embedding", sub: "Turn chunks into vectors" },
  { id: "pool", icon: "🧬", title: "Pooling", short: "Pooling", sub: "Combine token vectors" },
  { id: "index", icon: "🗄", title: "Indexing", short: "Indexing", sub: "Store & organize vectors" },
  { id: "retrieve", icon: "🔍", title: "Retrieve", short: "Retrieve", sub: "Search similar vectors" },
  { id: "backprop", icon: "🧠", title: "Backpropagation", short: "Backprop", sub: "Training simulator" },
  { id: "e2e", icon: "🌐", title: "End-to-End", short: "End-to-End", sub: "Run full pipeline" },
];

const DEMO_MODELS = [
  { id: "Demo-Embed-8D", dim: 8 },
  { id: "Demo-Embed-16D", dim: 16 },
];
const EMB_EXAMPLES = ["all i need", "Machine learning is fun", "Paris is the capital of France", "AI changes the world"];

// A tiny knowledge corpus for the Indexing / Retrieve / End-to-End modules.
const DEMO_CORPUS = [
  "Artificial intelligence is the simulation of human intelligence by machines.",
  "Deep learning uses neural networks with many layers.",
  "Machine learning is a subset of artificial intelligence.",
  "Transformers are used in modern natural language processing models.",
  "Reinforcement learning learns by trial and error from rewards.",
  "Natural language processing helps computers understand human text.",
  "Computer vision enables machines to interpret images and video.",
  "Gradient descent optimizes model parameters to reduce loss.",
];

// ── the deterministic demo embedding "model" ───────────────────────────────────
// FNV-1a hash → a stable float in [0,1] per (token, dimension). Same word always
// maps to the same vector, so a student can reason about the numbers.
function hashStr(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function embedToken(token: string, dim: number): number[] {
  const t = token.toLowerCase();
  const out: number[] = [];
  for (let j = 0; j < dim; j++) out.push(Math.round((0.08 + (hashStr(t + "#" + j) % 1000) / 1000 * 0.9) * 100) / 100);
  return out;
}
const words = (s: string) => (s.trim().match(/\S+/g) || []);
function poolVecs(vecs: number[][], method: Pooling, dim: number): number[] {
  if (!vecs.length) return new Array(dim).fill(0);
  if (method === "cls") return vecs[0].slice();
  const out = new Array(dim).fill(method === "max" ? -Infinity : 0);
  for (const v of vecs) for (let j = 0; j < dim; j++) out[j] = method === "max" ? Math.max(out[j], v[j]) : out[j] + v[j];
  if (method === "mean") for (let j = 0; j < dim; j++) out[j] = out[j] / vecs.length;
  return out.map((x) => Math.round(x * 100) / 100);
}
function embedText(text: string, dim: number, method: Pooling = "mean"): number[] {
  const toks = words(text);
  if (!toks.length) return new Array(dim).fill(0);
  return poolVecs(toks.map((t) => embedToken(t, dim)), method, dim);
}
const l2 = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
const cosineD = (a: number[], b: number[]) => dot(a, b) / ((l2(a) * l2(b)) || 1);
const euclidD = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0));
// Score two vectors under the selected distance metric (higher = more similar,
// except Euclidean where the raw distance is returned — lower is better).
const metricScore = (metric: string, a: number[], b: number[]) => metric.startsWith("Euclidean") ? euclidD(a, b) : metric.startsWith("Dot") ? dot(a, b) : cosineD(a, b);
const normalize = (v: number[]) => { const n = l2(v) || 1; return v.map((x) => x / n); };
const fmt = (v: number[], n = 8) => "[" + v.slice(0, n).map((x) => x.toFixed(2)).join(", ") + (v.length > n ? ", …" : "") + "]";
const POOL_LABEL: Record<Pooling, string> = { mean: "Mean pooling", max: "Max pooling", cls: "CLS pooling" };

// ── shared styling helpers ──────────────────────────────────────────────────
const panel: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 14, background: "var(--panel)", overflow: "hidden" };
// Each Section is a fixed-size box: its header stays pinned and the body scrolls
// inside (vertically, and horizontally for wide tables) once content passes `maxH`.
function Section({ title, right, children, pad = 16, maxH = 360 }: { title: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; pad?: number; maxH?: number }) {
  return (
    <div style={panel}>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center", padding: "11px 16px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{title}</span>
        {right}
      </div>
      <div style={{ padding: pad, maxHeight: maxH, overflow: "auto" }}>{children}</div>
    </div>
  );
}
function Bar({ v, color }: { v: number; color: string }) {
  return (
    <div style={{ flex: 1, height: 7, borderRadius: 6, background: "var(--panel-2)", overflow: "hidden" }}>
      <div style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%`, height: "100%", background: color, borderRadius: 6 }} />
    </div>
  );
}
function Pill({ color, children }: { color: string; children: React.ReactNode }) {
  return <span style={{ fontSize: 9.5, fontWeight: 700, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".04em", padding: "3px 9px", borderRadius: 20, background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}>{children}</span>;
}
function Btn({ on, onClick, children, color = ACC, title }: { on?: boolean; onClick?: () => void; children: React.ReactNode; color?: string; title?: string }) {
  return <button title={title} onClick={onClick} style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${on ? color : "var(--border)"}`, background: on ? color : "var(--panel-2)", color: on ? "#fff" : "var(--muted)", fontSize: 12.5, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>{children}</button>;
}

// Hover tooltip. Uses position:fixed at the cursor so it is never clipped by the
// fixed-size scroll containers. Wrap any element: <Tip tip="…"><Btn/></Tip>.
function Tip({ tip, children, w = 250 }: { tip: string; children: React.ReactNode; w?: number }) {
  const [p, setP] = useState<{ x: number; y: number } | null>(null);
  const move = (e: React.MouseEvent) => setP({ x: e.clientX, y: e.clientY });
  const vw = typeof window !== "undefined" ? window.innerWidth : 1280;
  const vh = typeof window !== "undefined" ? window.innerHeight : 800;
  return (
    <span style={{ display: "inline-flex", alignItems: "center" }} onMouseEnter={move} onMouseMove={move} onMouseLeave={() => setP(null)}>
      {children}
      {p && (
        <span role="tooltip" style={{ position: "fixed", left: Math.min(p.x + 14, vw - w - 12), top: Math.min(p.y + 18, vh - 90), zIndex: 9999, width: w, background: "var(--panel)", border: "1px solid var(--border-strong)", borderRadius: 9, padding: "9px 11px", fontSize: 11.5, lineHeight: 1.55, fontWeight: 400, color: "var(--text)", textTransform: "none", letterSpacing: 0, boxShadow: "0 10px 30px rgba(0,0,0,.4)", pointerEvents: "none" }}>{tip}</span>
      )}
    </span>
  );
}
// Inline glossary term: dotted underline + hover explanation.
function Info({ children, tip, color = "var(--faint)" }: { children: React.ReactNode; tip: string; color?: string }) {
  return <Tip tip={tip}><span style={{ borderBottom: `1px dotted ${color}`, cursor: "help" }}>{children}</span></Tip>;
}
// Empty-state shown before a gated step has been run.
function Placeholder({ text, icon = "🔮", minH = 240 }: { text: string; icon?: string; minH?: number }) {
  return (
    <div style={{ border: "1px dashed var(--border-strong)", borderRadius: 14, background: "var(--panel)", minHeight: minH, display: "grid", placeItems: "center", padding: 24, textAlign: "center" }}>
      <div>
        <div style={{ fontSize: 30, marginBottom: 10, opacity: 0.5 }}>{icon}</div>
        <div style={{ fontSize: 13, color: "var(--muted)", maxWidth: 380, lineHeight: 1.6 }}>{text}</div>
      </div>
    </div>
  );
}
// Plain-English definitions shown on hover across the playground.
const G = {
  meanPooling: "Mean pooling: average each dimension across all token vectors. Every token contributes equally to the chunk vector.",
  maxPooling: "Max pooling: for each dimension, keep the largest value across all tokens. Captures the strongest signal per feature.",
  clsPooling: "CLS pooling: use the first ([CLS]) token's vector as the whole-chunk vector — common in BERT-style encoders.",
  l2norm: "L2 norm: a vector's length, √(Σ xᵢ²). Vectors are normalized to length 1 so cosine comparisons are fair.",
  normalization: "Normalization: rescale a vector to unit length (÷ its L2 norm) so only its direction matters in similarity search.",
  cosine: "Cosine similarity: cosine of the angle between two vectors (−1…1). Higher = same direction = more similar meaning.",
  pca: "PCA: projects the high-dimensional vectors down to 2D for display, keeping the directions of greatest variance. Positions are illustrative only.",
  hnsw: "HNSW: a graph-based index for fast approximate nearest-neighbour search — it avoids scanning every stored vector.",
  efc: "ef_construction: how many candidate neighbours HNSW explores while building the index. Higher = better recall, slower build.",
  mconn: "M: how many neighbour links each node keeps in the HNSW graph. Higher = better recall, more memory.",
  embeddingModel: "Embedding model: converts text into a fixed-length vector. Here a deterministic demo model, so results are instant and reproducible.",
  learningRate: "Learning rate: the size of each step taken down the gradient. Too high overshoots the target; too low learns slowly.",
  lossMse: "Loss (MSE): mean squared error between the current embedding and the target. Training drives this toward zero.",
  gradient: "Gradient ∂L/∂W: the direction in which the loss rises fastest. Weights are stepped the opposite way to reduce loss.",
  overlap: "Overlap: words repeated from the end of one chunk into the start of the next, so an idea isn't cut in half at a boundary.",
  chunkFixed: "Fixed window: slide a fixed-size word window across the text, repeating `overlap` words between windows.",
  chunkSentence: "Sentence: group whole sentences up to ~size words — never splits in the middle of a sentence.",
  chunkParagraph: "Paragraph: group by blank-line paragraphs (falls back to sentences when there are none).",
  chunkSemantic: "Semantic: start a new chunk when the topic shifts (TF-IDF similarity drops) — variable size, no fixed overlap.",
  topK: "Top-K: how many of the highest-scoring chunks to keep as the retrieved context.",
  vectorNorm: "Vector norm (L2): the length of the final embedding vector, √(Σ xᵢ²).",
};
const POOL_TIP: Record<Pooling, string> = { mean: G.meanPooling, max: G.maxPooling, cls: G.clsPooling };
const CHUNK_TIP: Record<ChunkStrategy, string> = { fixed: G.chunkFixed, sentence: G.chunkSentence, paragraph: G.chunkParagraph, semantic: G.chunkSemantic };

// Click-to-open dialog for "show me the exact calculation" popups. Centered overlay
// (position:fixed) so it is never clipped by the fixed-size scroll containers.
function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.55)", zIndex: 10000, display: "grid", placeItems: "center", padding: 20 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(580px, 94vw)", maxHeight: "85vh", overflow: "auto", background: "var(--panel)", border: "1px solid var(--border-strong)", borderRadius: 14, boxShadow: "0 24px 70px rgba(0,0,0,.55)" }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center", padding: "12px 16px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
          <b style={{ fontSize: 14 }}>{title}</b>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--muted)", fontSize: 17, cursor: "pointer", lineHeight: 1 }}>✕</button>
        </div>
        <div style={{ padding: 16 }}>{children}</div>
      </div>
    </div>
  );
}
// A monospace worked-calculation block: one line per step, last line highlighted.
function CalcBlock({ lines, result, color = ACC }: { lines: string[]; result: string; color?: string }) {
  return (
    <div style={{ fontFamily: "var(--mono)", fontSize: 12.5, color: "var(--muted)", lineHeight: 1.95, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", overflowX: "auto" }}>
      {lines.map((l, i) => <div key={i} style={{ whiteSpace: "nowrap" }}>{l}</div>)}
      <div style={{ color, fontWeight: 700, marginTop: 4, whiteSpace: "nowrap" }}>{result}</div>
    </div>
  );
}

// A radar / spider chart of an embedding's per-dimension values (all in [0,1]).
function Radar({ values, color = ACC, size = 250 }: { values: number[]; color?: string; size?: number }) {
  const n = values.length, cx = size / 2, cy = size / 2, R = size / 2 - 32;
  const ang = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const pt = (i: number, r: number) => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const poly = values.map((v, i) => pt(i, R * Math.max(0.02, Math.min(1, v))).join(",")).join(" ");
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ display: "block" }}>
      {[0.25, 0.5, 0.75, 1].map((f, k) => (
        <polygon key={k} points={values.map((_, i) => pt(i, R * f).join(",")).join(" ")} fill="none" stroke="var(--border)" strokeWidth={0.7} opacity={0.6} />
      ))}
      {values.map((_, i) => { const [x, y] = pt(i, R); return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="var(--border)" strokeWidth={0.7} opacity={0.5} />; })}
      <polygon points={poly} fill={`color-mix(in srgb, ${color} 26%, transparent)`} stroke={color} strokeWidth={1.8} />
      {values.map((v, i) => { const [x, y] = pt(i, R * Math.max(0.02, Math.min(1, v))); return <circle key={i} cx={x} cy={y} r={2.6} fill={color} />; })}
      {values.map((_, i) => { const [x, y] = pt(i, R + 14); return <text key={i} x={x} y={y + 3} fontSize={9} fill="var(--faint)" textAnchor="middle">d{i + 1}</text>; })}
    </svg>
  );
}
// Value distribution histogram over [0,1] (8 bins).
function Histogram({ values, color = ACC }: { values: number[]; color?: string }) {
  const bins = new Array(8).fill(0);
  values.forEach((v) => { const b = Math.min(7, Math.max(0, Math.floor(Math.min(0.999, Math.max(0, v)) * 8))); bins[b]++; });
  const mx = Math.max(1, ...bins);
  const W = 300, H = 130, bw = W / 8;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      {bins.map((c, i) => { const h = (c / mx) * (H - 24); return <rect key={i} x={i * bw + 3} y={H - 18 - h} width={bw - 6} height={h} rx={2} fill={color} opacity={0.85} />; })}
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => <text key={i} x={t * W} y={H - 4} fontSize={8} fill="var(--faint)" textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}>{t.toFixed(2)}</text>)}
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: Chunk
// ═══════════════════════════════════════════════════════════════════════════
const CHUNK_SAMPLE = `Machine learning is a subset of artificial intelligence. It builds models from data instead of relying on explicit rules. Deep learning extends this idea with neural networks that stack many layers.

Retrieval-augmented generation grounds a language model in external documents. First the corpus is split into chunks, then each chunk is embedded into a vector. At query time the closest chunks are retrieved and passed back to the model as context.

Vector databases store these embeddings for fast similarity search. They rely on indexes like HNSW or IVF so they never have to scan every vector. Careful chunking keeps related ideas together, which keeps retrieval accurate.`;

// Map each chunk string back to a [start,end) word range in the source, so we can
// draw which words landed in which chunk (and where chunks overlap). Chunks are
// contiguous slices/joins of the source words, so we locate each chunk's word
// sequence starting from just after the previous chunk (monotonic).
function chunkSpans(source: string, chunks: string[]) {
  const src = words(source).map((w) => w.toLowerCase());
  const spans: { start: number; end: number }[] = [];
  let hint = 0;
  for (const c of chunks) {
    const cw = words(c).map((w) => w.toLowerCase());
    if (!cw.length) { spans.push({ start: hint, end: hint }); continue; }
    const findFrom = (from: number) => {
      for (let i = Math.max(0, from); i + cw.length <= src.length; i++) {
        let ok = true;
        for (let j = 0; j < cw.length; j++) if (src[i + j] !== cw[j]) { ok = false; break; }
        if (ok) return i;
      }
      return -1;
    };
    let start = findFrom(hint);
    if (start < 0) start = findFrom(0);
    if (start < 0) start = hint;
    spans.push({ start, end: start + cw.length });
    hint = start + 1;
  }
  return { total: src.length || 1, spans };
}

// The clearest way to *see* chunking: color the real text. Each word is tinted by
// the chunk it belongs to; words shared by two chunks (overlap) show both colors
// split down the middle. A slim segmented bar on top gives the same at a glance.
const chunkColor = (ci: number) => DIM_COLORS[ci % DIM_COLORS.length];
function wordBg(cs: number[], strong = false): string {
  if (!cs.length) return "transparent";
  const a = chunkColor(cs[0]);
  if (cs.length === 1) return `color-mix(in srgb, ${a} ${strong ? 90 : 34}%, transparent)`;
  const b = chunkColor(cs[1]);
  const pa = strong ? 90 : 55, pb = strong ? 90 : 55;
  return `linear-gradient(105deg, color-mix(in srgb, ${a} ${pa}%, transparent) 0 50%, color-mix(in srgb, ${b} ${pb}%, transparent) 50% 100%)`;
}
function ChunkMap({ text, chunks }: { text: string; chunks: string[] }) {
  const srcWords = words(text);
  const { total, spans } = chunkSpans(text, chunks);
  // which chunk(s) each word belongs to
  const wordChunks: number[][] = Array.from({ length: total }, () => []);
  spans.forEach((s, ci) => { for (let i = s.start; i < Math.min(s.end, total); i++) wordChunks[i].push(ci); });
  // contiguous runs of identical membership → segments for the overview bar
  const sig = (i: number) => wordChunks[i].join(",");
  const runs: { from: number; to: number; cs: number[] }[] = [];
  for (let i = 0, start = 0; i <= total; i++) { if (i === total || (i > 0 && sig(i) !== sig(start))) { runs.push({ from: start, to: i, cs: wordChunks[start] || [] }); start = i; } }

  return (
    <div>
      {/* slim segmented overview bar — the whole document left→right */}
      <div style={{ display: "flex", height: 20, borderRadius: 7, overflow: "hidden", border: "1px solid var(--border)", marginBottom: 4 }}>
        {runs.map((r, i) => <div key={i} title={r.cs.length ? `words ${r.from + 1}–${r.to} · chunk ${r.cs.map((c) => c + 1).join(" & ")}` : ""} style={{ flexGrow: r.to - r.from, flexBasis: 0, background: wordBg(r.cs, true), borderRight: i < runs.length - 1 ? "1px solid var(--surface)" : "none" }} />)}
      </div>
      <div className="row" style={{ justifyContent: "space-between", fontSize: 9, color: "var(--faint)", fontFamily: "var(--mono)", marginBottom: 12 }}><span>word 1</span><span>word {total}</span></div>

      {/* the real text, highlighted by chunk */}
      <div style={{ lineHeight: 2.15, fontSize: 13.5, wordBreak: "break-word" }}>
        {srcWords.map((w, i) => {
          const cs = wordChunks[i] || [];
          const overlap = cs.length > 1;
          return (
            <span key={i} title={overlap ? `shared by chunk ${cs[0] + 1} & ${cs[1] + 1}` : cs.length ? `chunk ${cs[0] + 1}` : "not in any chunk"}
              style={{ background: wordBg(cs), color: "var(--text)", borderRadius: 4, padding: "3px 3px", margin: "0 1px", boxShadow: overlap ? `inset 0 -3px 0 ${AMBER}` : "none", cursor: "default" }}>{w}</span>
          );
        })}
      </div>

      {/* legend: colour per chunk + overlap marker */}
      <div className="row" style={{ gap: 10, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}>
        {chunks.map((_, ci) => <span key={ci} className="row" style={{ gap: 5, alignItems: "center", fontSize: 10.5, color: "var(--muted)" }}><span style={{ width: 12, height: 12, borderRadius: 3, background: `color-mix(in srgb, ${chunkColor(ci)} 60%, transparent)` }} />chunk {ci + 1}</span>)}
        <span className="row" style={{ gap: 5, alignItems: "center", fontSize: 10.5, color: "var(--muted)", marginLeft: 4 }}><span style={{ width: 18, height: 12, borderRadius: 3, background: "var(--panel-2)", boxShadow: `inset 0 -3px 0 ${AMBER}` }} />overlap (shared by two chunks)</span>
      </div>
    </div>
  );
}

function ChunkModule() {
  const [text, setText] = useState(CHUNK_SAMPLE);
  const [size, setSize] = useState(16);
  const [overlap, setOverlap] = useState(4);
  const [strategy, setStrategy] = useState<ChunkStrategy>("fixed");
  const chunks = useMemo(() => chunkBy(text, strategy, size, overlap), [text, strategy, size, overlap]);
  const totalWords = words(text).length;
  const sizes = chunks.map((c) => words(c).length);
  const maxSz = Math.max(1, ...sizes);
  // count words shared between consecutive chunks (real overlap in this strategy)
  const { spans } = chunkSpans(text, chunks);
  const overlapWords = spans.slice(1).reduce((a, s, k) => a + Math.max(0, Math.min(s.end, spans[k].end) - Math.max(s.start, spans[k].start)), 0);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 300px", gap: 16 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title="Your text" maxH={520}>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5} style={{ width: "100%", boxSizing: "border-box", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: 12, color: "var(--text)", fontSize: 13, fontFamily: "inherit", resize: "vertical", lineHeight: 1.5 }} />
          <div className="row" style={{ gap: 16, marginTop: 12, flexWrap: "wrap", alignItems: "center" }}>
            <div style={{ display: "inline-flex", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: 3 }}>
              {(["fixed", "sentence", "paragraph", "semantic"] as ChunkStrategy[]).map((s) => (
                <Tip key={s} tip={CHUNK_TIP[s]}><button onClick={() => setStrategy(s)} style={{ padding: "6px 12px", borderRadius: 8, border: "none", background: strategy === s ? ACC : "transparent", color: strategy === s ? "#fff" : "var(--muted)", fontSize: 12, fontWeight: 600, cursor: "pointer", textTransform: "capitalize" }}>{s}</button></Tip>
              ))}
            </div>
            <label className="row" style={{ gap: 8, alignItems: "center", fontSize: 12, color: "var(--muted)" }}>Size <input type="range" min={4} max={40} value={size} onChange={(e) => setSize(+e.target.value)} /><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{size}</b></label>
            <label className="row" style={{ gap: 8, alignItems: "center", fontSize: 12, color: "var(--muted)" }}><Info tip={G.overlap}>Overlap</Info> <input type="range" min={0} max={12} value={overlap} onChange={(e) => setOverlap(+e.target.value)} disabled={strategy === "semantic"} /><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{overlap}</b></label>
          </div>
          <div className="note" style={{ marginTop: 8, fontSize: 11 }}>{strategy === "fixed" ? "Slides a fixed word window across the text — overlap repeats trailing words." : strategy === "sentence" ? "Groups whole sentences up to ~size words — never splits mid-sentence." : strategy === "paragraph" ? "Groups by blank-line paragraphs (falls back to sentences if none)." : "Starts a new chunk when the topic shifts (TF-IDF drop) — no fixed overlap."}</div>
        </Section>

        <Section title="Chunk map" right={<Pill color={GREEN}>{CHUNK_STRATEGY_LABEL[strategy]}</Pill>} maxH={460}>
          <ChunkMap text={text} chunks={chunks} />
        </Section>

        <Section title={`Chunks (${chunks.length})`} maxH={460}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {chunks.map((c, i) => (
              <div key={i} style={{ border: `1px solid ${DIM_COLORS[i % DIM_COLORS.length]}`, borderLeftWidth: 3, borderRadius: 10, background: "var(--panel-2)", padding: "9px 12px" }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
                  <Pill color={DIM_COLORS[i % DIM_COLORS.length]}>chunk {i + 1}</Pill>
                  <span style={{ fontSize: 10, color: "var(--faint)", fontFamily: "var(--mono)" }}>{words(c).length} words</span>
                </div>
                <div style={{ fontSize: 12.5, lineHeight: 1.5 }}>{c}</div>
              </div>
            ))}
          </div>
        </Section>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title="Summary">
          {[["Words", totalWords], ["Chunks", chunks.length], ["Strategy", CHUNK_STRATEGY_LABEL[strategy]], ["Avg words/chunk", chunks.length ? Math.round(totalWords / chunks.length) : 0], ["Overlap words", overlapWords]].map(([k, v]) => (
            <div key={k as string} className="row" style={{ justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--border)" }}><span style={{ fontSize: 12, color: "var(--muted)" }}>{k}</span><b style={{ fontSize: 13 }}>{v}</b></div>
          ))}
        </Section>
        <Section title="Chunk sizes" maxH={320}>
          {sizes.map((s, i) => (
            <div key={i} style={{ marginBottom: 7 }}>
              <div className="row" style={{ justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}><span style={{ color: "var(--muted)" }}>chunk {i + 1}</span><b style={{ fontFamily: "var(--mono)" }}>{s}w</b></div>
              <Bar v={s / maxSz} color={DIM_COLORS[i % DIM_COLORS.length]} />
            </div>
          ))}
        </Section>
        <div style={{ ...panel, padding: 14, fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>
          <b style={{ color: "var(--text)" }}>💡 Why chunking?</b><br />Retrieval works on chunks, not whole documents. Smaller chunks are more precise; larger chunks keep more context. Overlap stops ideas from being cut in half at a boundary. Switch the strategy and watch the <b style={{ color: "var(--text)" }}>chunk map</b> change.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: Embedding
// ═══════════════════════════════════════════════════════════════════════════
function EmbeddingModule({ onStore }: { onStore: (v: number[], label: string) => void }) {
  const [text, setText] = useState("all i need");
  const [modelId, setModelId] = useState(DEMO_MODELS[0].id);
  const [pooling, setPooling] = useState<Pooling>("max");
  const [activeTok, setActiveTok] = useState(0);
  const dim = DEMO_MODELS.find((m) => m.id === modelId)!.dim;
  const toks = words(text);
  const tokVecs = useMemo(() => toks.map((t) => embedToken(t, dim)), [text, dim]);
  const pooled = useMemo(() => poolVecs(tokVecs, pooling, dim), [tokVecs, pooling, dim]);
  const norm = l2(pooled);
  const [detail, setDetail] = useState<{ title: string; node: React.ReactNode } | null>(null);
  // Output is gated behind "Run embedding" — nothing shows until the user runs it,
  // and editing the text or model invalidates the previous run.
  const [ran, setRan] = useState(false);
  useEffect(() => { setRan(false); }, [text, modelId]);
  const reset = () => { setText("all i need"); setModelId(DEMO_MODELS[0].id); setPooling("max"); setActiveTok(0); setRan(false); };

  // ── click-to-open worked calculations (real numbers) ──
  function showPool(j: number) {
    const vals = tokVecs.map((v) => v[j]); const dl = `d${j + 1}`;
    const winner = vals.reduce((b, v, i) => (v > vals[b] ? i : b), 0);
    const sum = vals.reduce((a, b) => a + b, 0);
    const lines = pooling === "mean"
      ? [`v[${dl}] = (1/n) · Σ vᵢ[${dl}]`, `= ( ${vals.map((v) => v.toFixed(2)).join(" + ")} ) / ${vals.length}`, `= ${sum.toFixed(2)} / ${vals.length}`]
      : pooling === "max"
        ? [`v[${dl}] = max( ${vals.map((v) => v.toFixed(2)).join(", ")} )`]
        : [`v[${dl}] = v₁[${dl}]   (CLS = first token)`];
    const result = pooling === "max" ? `= ${pooled[j].toFixed(2)}   ← from t${winner + 1} · "${toks[winner]}"`
      : pooling === "cls" ? `= ${pooled[j].toFixed(2)}   ← first token "${toks[0]}"`
        : `= ${pooled[j].toFixed(2)}`;
    const node = (
      <div>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>Every token&apos;s value at <b style={{ color: "var(--text)" }}>{dl}</b> — the numbers this operation combines:</div>
        <div style={{ marginBottom: 12 }}>
          {vals.map((v, i) => {
            const hi = (pooling === "max" && i === winner) || (pooling === "cls" && i === 0);
            return <div key={i} className="row" style={{ gap: 8, alignItems: "center", marginBottom: 5 }}>
              <span style={{ width: 80, fontSize: 11, fontFamily: "var(--mono)", color: hi ? ACC : "var(--muted)" }}>t{i + 1} · {toks[i]}</span>
              <Bar v={v} color={hi ? ACC : DIM_COLORS[i % DIM_COLORS.length]} />
              <span style={{ width: 40, textAlign: "right", fontFamily: "var(--mono)", fontSize: 11.5, color: hi ? ACC : "var(--text)", fontWeight: hi ? 700 : 400 }}>{v.toFixed(2)}</span>
            </div>;
          })}
        </div>
        <CalcBlock lines={lines} result={result} />
      </div>
    );
    setDetail({ title: `How ${POOL_LABEL[pooling]} produces ${dl}`, node });
  }
  function showNorm() {
    const sq = pooled.map((x) => x * x); const sumSq = sq.reduce((a, b) => a + b, 0);
    const node = (
      <div>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>The L2 norm is the length of the final vector — square every value, add them up, take the square root.</div>
        <CalcBlock color={GREEN}
          lines={[`‖v‖ = √( Σ v[j]² )`, `= √( ${pooled.map((x) => x.toFixed(2) + "²").join(" + ")} )`, `= √( ${sq.map((s) => s.toFixed(3)).join(" + ")} )`, `= √( ${sumSq.toFixed(3)} )`]}
          result={`= ${Math.sqrt(sumSq).toFixed(2)}`} />
      </div>
    );
    setDetail({ title: "How the L2 vector norm is computed", node });
  }
  function showTokenModel() {
    const node = (
      <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.7 }}>
        <p style={{ marginTop: 0 }}><b style={{ color: "var(--text)" }}>Demo-Embed-{dim}D</b> turns each token into a fixed {dim}-dimensional vector using a deterministic hash of the token text.</p>
        <p>The same word always maps to the same vector — e.g. <span style={{ fontFamily: "var(--mono)", color: ACC }}>&quot;{toks[0]}&quot;</span> → <span style={{ fontFamily: "var(--mono)", fontSize: 11 }}>{fmt(tokVecs[0], dim)}</span>.</p>
        <p style={{ marginBottom: 0 }}>Each bar is one dimension&apos;s value (0–1). Real embedding models <i>learn</i> these numbers from data during training (see the <b style={{ color: "var(--text)" }}>Backpropagation</b> module) — here they&apos;re generated deterministically so the lab is instant and reproducible.</p>
      </div>
    );
    setDetail({ title: "How token embeddings are produced", node });
  }

  const copy = () => { navigator.clipboard?.writeText(JSON.stringify(pooled)); toast("Vector copied", "success"); };
  const download = () => {
    const blob = new Blob([JSON.stringify({ text, model: modelId, pooling, dim, vector: pooled, norm }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "embedding.json"; a.click(); URL.revokeObjectURL(a.href);
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 300px", gap: 16 }}>
      {detail && <Modal title={detail.title} onClose={() => setDetail(null)}>{detail.node}</Modal>}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title="Embedding Playground" right={<div className="row" style={{ gap: 8, alignItems: "center" }}><Pill color={ACC}>{dim}D</Pill><button onClick={reset} style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--muted)", fontSize: 11.5, fontWeight: 600, padding: "5px 10px", cursor: "pointer", fontFamily: "inherit" }}>↻ Reset</button></div>}>
          <div className="row" style={{ gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
            <div style={{ flex: 1, minWidth: 200 }}>
              <label style={lbl}>Your text / chunk</label>
              <input value={text} onChange={(e) => { setText(e.target.value); setActiveTok(0); }} style={inp} />
            </div>
            <div>
              <label style={lbl}><Info tip={G.embeddingModel}>Embedding model</Info></label>
              <select value={modelId} onChange={(e) => setModelId(e.target.value)} style={{ ...inp, width: 170 }}>
                {DEMO_MODELS.map((m) => <option key={m.id} value={m.id}>{m.id}</option>)}
              </select>
            </div>
          </div>
          <div className="row" style={{ gap: 6, marginTop: 10, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 11, color: "var(--faint)" }}>Quick start:</span>
            {EMB_EXAMPLES.map((ex) => <button key={ex} onClick={() => { setText(ex); setActiveTok(0); }} style={{ padding: "5px 10px", borderRadius: 20, border: `1px solid ${text === ex ? ACC : "var(--border)"}`, background: text === ex ? `color-mix(in srgb, ${ACC} 14%, transparent)` : "var(--panel-2)", color: text === ex ? ACC : "var(--muted)", fontSize: 11.5, cursor: "pointer", fontFamily: "inherit" }}>{ex}</button>)}
          </div>
          <div className="row" style={{ gap: 12, marginTop: 14, alignItems: "center", flexWrap: "wrap", justifyContent: "space-between" }}>
            <div className="row" style={{ gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Pooling method</span>
              {(["mean", "max", "cls"] as Pooling[]).map((p) => <Tip key={p} tip={POOL_TIP[p]}><Btn on={pooling === p} onClick={() => setPooling(p)}>{POOL_LABEL[p]}</Btn></Tip>)}
            </div>
            <button onClick={() => { setRan(true); setActiveTok(0); }} style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 18px", borderRadius: 9, border: "none", background: ACC, color: "#fff", fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>▶ Run embedding</button>
          </div>
        </Section>

        {!ran ? (
          <Placeholder minH={320} text="Type your text, pick a pooling method, then press ▶ Run embedding to see the tokens, each token's vector, the pooling operation, and the final embedding." />
        ) : (<>
        <Section title={<>1. Tokens <Pill color={BLUE}>{toks.length} tokens</Pill></>} right={<span style={{ fontSize: 11, color: "var(--faint)" }}>Click a token to inspect its vector</span>}>
          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
            {toks.map((t, i) => (
              <button key={i} onClick={() => setActiveTok(i)} style={{ padding: "7px 12px", borderRadius: 8, border: `1px solid ${activeTok === i ? ACC : "var(--border)"}`, background: activeTok === i ? `color-mix(in srgb, ${ACC} 18%, transparent)` : "var(--panel-2)", color: activeTok === i ? ACC : "var(--text)", fontSize: 12.5, fontFamily: "var(--mono)", cursor: "pointer" }}>t{i + 1} · {t}</button>
            ))}
          </div>
        </Section>

        <Section title={<>2. Token embeddings <span style={{ color: "var(--faint)", fontWeight: 400, fontSize: 11 }}>({dim} dimensions)</span></>} right={<button onClick={showTokenModel} style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 7, color: ACC, fontSize: 11, fontWeight: 600, padding: "4px 10px", cursor: "pointer", fontFamily: "inherit" }}>How are these made? ⓘ</button>}>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(3, toks.length || 1)}, 1fr)`, gap: 12 }}>
            {tokVecs.map((v, ti) => (
              <div key={ti} style={{ border: `1px solid ${activeTok === ti ? ACC : "var(--border)"}`, borderRadius: 10, background: "var(--panel-2)", padding: 12 }}>
                <div style={{ color: DIM_COLORS[ti % DIM_COLORS.length], fontWeight: 700, fontSize: 12.5, fontFamily: "var(--mono)", marginBottom: 8 }}>t{ti + 1} · {toks[ti]}</div>
                {v.map((x, j) => (
                  <div key={j} className="row" style={{ gap: 8, alignItems: "center", marginBottom: 4 }}>
                    <span style={{ width: 22, fontSize: 10, color: "var(--faint)", fontFamily: "var(--mono)" }}>d{j + 1}</span>
                    <Bar v={x} color={DIM_COLORS[ti % DIM_COLORS.length]} />
                    <span style={{ width: 34, fontSize: 10.5, textAlign: "right", fontFamily: "var(--mono)" }}>{x.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Section>

        <Section title="3. Pooling operation" right={<Pill color={ACC}>{pooling} pooling</Pill>}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>
            {pooling === "mean" ? "v[j] = (1/n) · Σ vᵢ[j]" : pooling === "max" ? "v[j] = max(v₁[j], v₂[j], …, vₙ[j])" : "v[j] = v₁[j]  (first / CLS token)"}
          </div>
          <div className="note" style={{ marginBottom: 12, fontSize: 11 }}>👆 Click any dimension below to see the exact calculation.</div>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${dim}, 1fr)`, gap: 6 }}>
            {pooled.map((x, j) => (
              <div key={j} style={{ textAlign: "center" }}>
                <button onClick={() => showPool(j)} title={`How is d${j + 1} computed?`} style={{ width: "100%", border: `1px solid ${ACC}`, borderRadius: 8, padding: "8px 4px", fontFamily: "var(--mono)", fontSize: 12, color: ACC, background: `color-mix(in srgb, ${ACC} 8%, transparent)`, cursor: "pointer" }}>{x.toFixed(2)}</button>
                <div style={{ fontSize: 9, color: "var(--faint)", marginTop: 3 }}>d{j + 1}</div>
              </div>
            ))}
          </div>
        </Section>

        <Section title="4. Final chunk embedding (output)" right={<Pill color={GREEN}>final</Pill>}>
          <div className="row" style={{ gap: 14, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 220, fontFamily: "var(--mono)", fontSize: 13, color: GREEN, border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", background: "var(--panel-2)", overflowX: "auto" }}>{fmt(pooled, dim)}</div>
            <button onClick={showNorm} title="How is this computed?" style={{ textAlign: "center", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 14px", cursor: "pointer", fontFamily: "inherit" }}>
              <div style={{ fontSize: 10, color: "var(--faint)", textTransform: "uppercase", letterSpacing: ".05em" }}>Vector norm (L2) ⓘ</div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 24, fontWeight: 700, color: "var(--text)" }}>{norm.toFixed(2)}</div>
            </button>
          </div>
          <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 10 }}>Dimension: {dim} · Pooling: {POOL_LABEL[pooling]} · Model: {modelId}</div>
        </Section>
        </>)}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {!ran ? (
          <Placeholder minH={220} icon="📊" text="Run the embedding to see the summary, radar chart, value distribution and export actions." />
        ) : (<>
        <Section title="Embedding summary">
          {[["Tokens", toks.length], ["Dimensions", dim], ["Pooling", POOL_LABEL[pooling]], ["Model", modelId], ["Vector norm (L2)", norm.toFixed(2)]].map(([k, v]) => (
            <div key={k as string} className="row" style={{ justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--border)" }}><span style={{ fontSize: 12, color: "var(--muted)" }}>{k}</span><b style={{ fontSize: 12.5 }}>{v}</b></div>
          ))}
        </Section>
        <Section title={<>Final embedding <span style={{ color: "var(--faint)", fontWeight: 400, fontSize: 11 }}>(visual)</span></>}><Radar values={pooled} /></Section>
        <Section title="Value distribution"><Histogram values={pooled} /></Section>
        <Section title="Actions" pad={12}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <button onClick={copy} style={actBtn}>📋 Copy vector</button>
            <button onClick={download} style={actBtn}>⬇ Download JSON</button>
            <button onClick={() => onStore(pooled, text)} style={actBtn}>🗄 Add to vector store</button>
          </div>
        </Section>
        </>)}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: Pooling
// ═══════════════════════════════════════════════════════════════════════════
// Diagram: token nodes on the left wired to the pooled value boxes on the right.
// For max/cls the "winning" wire per dimension is lit; for mean all wires are lit.
function PoolDiagram({ tokVecs, toks, pooled, pooling, winner, activeTok = -1, onToken }: { tokVecs: number[][]; toks: string[]; pooled: number[]; pooling: Pooling; winner: (j: number) => number; activeTok?: number; onToken?: (i: number) => void }) {
  const n = tokVecs.length || 1, D = pooled.length;
  const W = 330, rowH = 24, top = 22, boxW = 46;
  const H = Math.max(n * 34, D * rowH) + top + 8;
  const leftX = 40, poolX = 214;
  const tokY = (i: number) => top + (H - top) * (i + 0.5) / n;
  const dimY = (j: number) => top + j * rowH + rowH / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <text x={poolX + boxW / 2} y={12} fontSize={9} fill="var(--faint)" textAnchor="middle">Pooled</text>
      {tokVecs.map((_, i) => pooled.map((_, j) => {
        const lit = pooling === "max" ? winner(j) === i : pooling === "cls" ? i === 0 : true;
        const opacity = activeTok >= 0 ? (i === activeTok ? (lit ? 0.9 : 0.3) : 0.05) : (lit ? 0.75 : 0.14);
        return <line key={`${i}-${j}`} x1={leftX + 18} y1={tokY(i)} x2={poolX} y2={dimY(j)} stroke={lit ? DIM_COLORS[i % DIM_COLORS.length] : "var(--border)"} strokeWidth={lit ? 1.3 : 0.5} opacity={opacity} strokeDasharray={lit ? "" : "2 3"} />;
      }))}
      {toks.map((_, i) => <g key={i} onClick={() => onToken?.(i)} style={{ cursor: onToken ? "pointer" : "default" }}>
        <rect x={leftX - 17} y={tokY(i) - 10} width={35} height={20} rx={5} fill={`color-mix(in srgb, ${DIM_COLORS[i % DIM_COLORS.length]} ${activeTok === i ? 40 : 22}%, transparent)`} stroke={DIM_COLORS[i % DIM_COLORS.length]} strokeWidth={activeTok === i ? 2 : 1} />
        <text x={leftX + 0.5} y={tokY(i) + 3.5} fontSize={9.5} fill="var(--text)" textAnchor="middle" fontFamily="var(--mono)">t{i + 1}</text>
      </g>)}
      {pooled.map((x, j) => <g key={j}>
        <rect x={poolX} y={dimY(j) - 9} width={boxW} height={18} rx={4} fill={`color-mix(in srgb, ${ACC} 14%, transparent)`} stroke={ACC} />
        <text x={poolX + boxW / 2} y={dimY(j) + 3.5} fontSize={9} fill={ACC} textAnchor="middle" fontFamily="var(--mono)">{x.toFixed(2)}</text>
        <text x={poolX + boxW + 6} y={dimY(j) + 3.5} fontSize={8.5} fill="var(--faint)" textAnchor="start">d{j + 1}</text>
      </g>)}
    </svg>
  );
}
function PoolingModule() {
  const [text, setText] = useState("all i need");
  const [pooling, setPooling] = useState<Pooling>("max");
  const dim = 8;
  const toks = words(text);
  const tokVecs = useMemo(() => toks.map((t) => embedToken(t, dim)), [text]);
  const results: Record<Pooling, number[]> = {
    mean: poolVecs(tokVecs, "mean", dim), max: poolVecs(tokVecs, "max", dim), cls: poolVecs(tokVecs, "cls", dim),
  };
  const pooled = results[pooling];
  // which token "wins" each dimension (for max), or contributes (mean = equal share)
  const winner = (j: number) => tokVecs.reduce((best, v, i) => (v[j] > tokVecs[best][j] ? i : best), 0);
  const [activeTok, setActiveTok] = useState(-1);
  const [detail, setDetail] = useState<{ title: string; node: React.ReactNode } | null>(null);
  const reset = () => { setText("all i need"); setPooling("max"); setActiveTok(-1); };
  const copy = () => { navigator.clipboard?.writeText(JSON.stringify(pooled)); toast("Vector copied", "success"); };
  const pickTok = (i: number) => setActiveTok((a) => (a === i ? -1 : i));
  function showPool(j: number) {
    const vals = tokVecs.map((v) => v[j]); const dl = `d${j + 1}`;
    const win = winner(j); const sum = vals.reduce((a, b) => a + b, 0);
    const lines = pooling === "mean"
      ? [`v[${dl}] = (1/n) · Σ vᵢ[${dl}]`, `= ( ${vals.map((v) => v.toFixed(2)).join(" + ")} ) / ${vals.length}`, `= ${sum.toFixed(2)} / ${vals.length}`]
      : pooling === "max" ? [`v[${dl}] = max( ${vals.map((v) => v.toFixed(2)).join(", ")} )`]
        : [`v[${dl}] = v₁[${dl}]   (CLS = first token)`];
    const result = pooling === "max" ? `= ${pooled[j].toFixed(2)}   ← from t${win + 1} · "${toks[win]}"`
      : pooling === "cls" ? `= ${pooled[j].toFixed(2)}   ← first token "${toks[0]}"` : `= ${pooled[j].toFixed(2)}`;
    setDetail({
      title: `How ${POOL_LABEL[pooling]} produces ${dl}`,
      node: (
        <div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>Every token&apos;s value at <b style={{ color: "var(--text)" }}>{dl}</b>:</div>
          <div style={{ marginBottom: 12 }}>
            {vals.map((v, i) => { const hi = (pooling === "max" && i === win) || (pooling === "cls" && i === 0); return <div key={i} className="row" style={{ gap: 8, alignItems: "center", marginBottom: 5 }}><span style={{ width: 80, fontSize: 11, fontFamily: "var(--mono)", color: hi ? ACC : "var(--muted)" }}>t{i + 1} · {toks[i]}</span><Bar v={v} color={hi ? ACC : DIM_COLORS[i % DIM_COLORS.length]} /><span style={{ width: 40, textAlign: "right", fontFamily: "var(--mono)", fontSize: 11.5, color: hi ? ACC : "var(--text)", fontWeight: hi ? 700 : 400 }}>{v.toFixed(2)}</span></div>; })}
          </div>
          <CalcBlock lines={lines} result={result} />
        </div>
      ),
    });
  }
  const statCard = (label: string, node: React.ReactNode) => <div style={{ border: "1px solid var(--border)", borderRadius: 10, background: "var(--panel-2)", padding: "10px 12px", minWidth: 0 }}><div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginBottom: 4 }}>{label}</div>{node}</div>;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 300px", gap: 16 }}>
      {detail && <Modal title={detail.title} onClose={() => setDetail(null)}>{detail.node}</Modal>}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title="Pooling Playground" right={<button onClick={reset} style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--muted)", fontSize: 11.5, fontWeight: 600, padding: "5px 10px", cursor: "pointer", fontFamily: "inherit" }}>↻ Reset</button>}>
          <div className="note" style={{ marginBottom: 12, fontSize: 12 }}>See how different pooling strategies combine token vectors into a single chunk embedding.</div>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.4fr) repeat(2, minmax(0,0.6fr)) minmax(0,1.6fr)", gap: 10, alignItems: "stretch" }}>
            {statCard("Input text", <input value={text} onChange={(e) => setText(e.target.value)} style={{ ...inp, padding: "4px 8px", fontSize: 12.5 }} />)}
            {statCard("Tokens", <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--mono)" }}>{toks.length}</div>)}
            {statCard("Dimensions", <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--mono)" }}>{dim}D</div>)}
            {statCard("Select pooling method", <div className="row" style={{ gap: 6, marginTop: 2, flexWrap: "wrap" }}>{(["mean", "max", "cls"] as Pooling[]).map((p) => <Tip key={p} tip={POOL_TIP[p]}><Btn on={pooling === p} onClick={() => setPooling(p)}>{POOL_LABEL[p]}</Btn></Tip>)}</div>)}
          </div>
        </Section>
        <Section title="1. Tokens and their embeddings" right={<span style={{ fontSize: 11, color: "var(--faint)" }}>Click a token to highlight its vector</span>}>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(3, toks.length || 1)}, 1fr)`, gap: 12 }}>
            {tokVecs.map((v, ti) => (
              <div key={ti} onClick={() => pickTok(ti)} style={{ border: `1px solid ${activeTok === ti ? DIM_COLORS[ti % DIM_COLORS.length] : "var(--border)"}`, borderRadius: 10, background: activeTok === ti ? `color-mix(in srgb, ${DIM_COLORS[ti % DIM_COLORS.length]} 8%, var(--panel-2))` : "var(--panel-2)", padding: 12, cursor: "pointer", opacity: activeTok < 0 || activeTok === ti ? 1 : 0.5 }}>
                <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}><span style={{ color: DIM_COLORS[ti % DIM_COLORS.length], fontWeight: 700, fontSize: 12.5, fontFamily: "var(--mono)" }}>t{ti + 1} · {toks[ti]}</span><span style={{ fontSize: 9.5, color: "var(--faint)" }}>{dim} dims</span></div>
                {v.map((x, j) => <div key={j} className="row" style={{ gap: 8, alignItems: "center", marginBottom: 4 }}><span style={{ width: 22, fontSize: 10, color: "var(--faint)", fontFamily: "var(--mono)" }}>d{j + 1}</span><Bar v={x} color={DIM_COLORS[ti % DIM_COLORS.length]} /><span style={{ width: 34, fontSize: 10.5, textAlign: "right", fontFamily: "var(--mono)" }}>{x.toFixed(2)}</span></div>)}
              </div>
            ))}
          </div>
        </Section>
        <Section title="2. Pooling operation" right={<Pill color={ACC}>{pooling} pooling</Pill>}>
          <div className="note" style={{ marginBottom: 10, fontSize: 11.5 }}>{pooling === "max" ? "Takes the maximum value for each dimension across all token vectors." : pooling === "mean" ? "Averages each dimension across all token vectors." : "Uses the first (CLS) token's vector as the chunk vector."}</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>{pooling === "mean" ? "v[j] = (1/n) · Σ vᵢ[j]" : pooling === "max" ? "v[j] = max(v₁[j], v₂[j], …, vₙ[j])" : "v[j] = v₁[j]"}</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", width: "100%", fontFamily: "var(--mono)", fontSize: 11.5 }}>
              <thead><tr><th style={th}>Dimension</th>{pooled.map((_, j) => <th key={j} style={th}>d{j + 1}</th>)}</tr></thead>
              <tbody>
                {tokVecs.map((v, ti) => <tr key={ti} onClick={() => pickTok(ti)} style={{ cursor: "pointer", background: activeTok === ti ? `color-mix(in srgb, ${DIM_COLORS[ti % DIM_COLORS.length]} 12%, transparent)` : "transparent", opacity: activeTok < 0 || activeTok === ti ? 1 : 0.5 }}><td style={td}>t{ti + 1} · {toks[ti]}</td>{v.map((x, j) => { const sel = (pooling === "max" && winner(j) === ti) || (pooling === "cls" && ti === 0); return <td key={j} style={{ ...td, textAlign: "center", color: sel ? ACC : "var(--muted)", fontWeight: sel ? 700 : 400 }}>{sel ? "↑ " : ""}{x.toFixed(2)}</td>; })}</tr>)}
                <tr><td style={{ ...td, color: ACC, fontWeight: 700 }}>{pooling.toUpperCase()} (pooled)</td>{pooled.map((x, j) => <td key={j} style={{ ...td, textAlign: "center", padding: 0 }}><button onClick={() => showPool(j)} title={`How is d${j + 1} computed?`} style={{ width: "100%", height: "100%", padding: "7px 8px", background: "transparent", border: "none", color: ACC, fontWeight: 700, fontFamily: "var(--mono)", fontSize: 11.5, cursor: "pointer" }}>{x.toFixed(2)}</button></td>)}</tr>
              </tbody>
            </table>
          </div>
          <div className="row" style={{ gap: 16, marginTop: 8, fontSize: 10.5, color: "var(--muted)", flexWrap: "wrap" }}><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 12, height: 6, borderRadius: 3, background: ACC }} />Selected {pooling === "max" ? "maximum" : pooling === "cls" ? "(CLS token)" : "values"}</span><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 12, height: 6, borderRadius: 3, background: "var(--border-strong)" }} />Other values</span><span style={{ color: "var(--faint)" }}>👆 Click a pooled value for the exact math · click a token to highlight it</span></div>
        </Section>
        <Section title="3. Final pooled vector (chunk embedding)">
          <div className="row" style={{ gap: 14, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 220, fontFamily: "var(--mono)", fontSize: 13, color: ACC, border: `1px solid ${ACC}`, borderRadius: 10, padding: "12px 14px", background: `color-mix(in srgb, ${ACC} 7%, transparent)`, overflowX: "auto" }}>{fmt(pooled)} <span style={{ color: GREEN }}>· {dim}D</span></div>
            <div style={{ textAlign: "center" }}><div style={{ fontSize: 10, color: "var(--faint)", textTransform: "uppercase" }}>Vector norm (L2)</div><div style={{ fontFamily: "var(--mono)", fontSize: 22, fontWeight: 700, color: GREEN }}>{l2(pooled).toFixed(2)}</div><button onClick={copy} style={{ ...actBtn, padding: "5px 10px", marginTop: 6, fontSize: 11 }}>📋 Copy vector</button></div>
          </div>
          <div className="note" style={{ marginTop: 10, fontSize: 11.5 }}>This is the final chunk embedding that represents the input text using {POOL_LABEL[pooling]}. It can now be inserted into the vector store and used during retrieval.</div>
        </Section>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title={`How ${POOL_LABEL[pooling]} works`} pad={14} maxH={460}>
          <ol style={{ margin: "0 0 12px", paddingLeft: 18, fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}>
            {pooling === "max" ? <>
              <li>Look at each dimension (d1…d{dim})</li><li>Find the maximum value among all token vectors</li><li>Build a new vector using those maximum values</li>
            </> : pooling === "mean" ? <>
              <li>Look at each dimension (d1…d{dim})</li><li>Average the value across all token vectors</li><li>Build a new vector from those averages</li>
            </> : <>
              <li>Take the first (CLS / special) token</li><li>Use its vector as the whole-chunk vector</li><li>Common in BERT-style encoders</li>
            </>}
          </ol>
          <PoolDiagram tokVecs={tokVecs} toks={toks} pooled={pooled} pooling={pooling} winner={winner} activeTok={activeTok} onToken={pickTok} />
        </Section>
        <Section title="Token contribution" pad={14}>
          <div style={{ fontSize: 11, color: "var(--faint)", marginBottom: 8 }}>Approx. contribution to the pooled vector</div>
          {toks.map((t, i) => {
            const contrib = pooling === "cls" ? (i === 0 ? 1 : 0) : pooling === "max" ? pooled.filter((_, j) => winner(j) === i).length / dim : 1 / toks.length;
            return <div key={i} style={{ marginBottom: 8 }}><div className="row" style={{ justifyContent: "space-between", fontSize: 11.5, marginBottom: 3 }}><span>{t}</span><b>{Math.round(contrib * 100)}%</b></div><Bar v={contrib} color={ACC} /></div>;
          })}
        </Section>
        <Section title="Compare pooling methods" pad={14}>
          {(["mean", "max", "cls"] as Pooling[]).map((p) => (
            <div key={p} style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: p === pooling ? ACC : "var(--text)" }}>{POOL_LABEL[p]}</div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: p === pooling ? ACC : "var(--faint)", marginTop: 2, wordBreak: "break-all" }}>{fmt(results[p])}</div>
            </div>
          ))}
        </Section>
        <div style={{ ...panel, padding: 14, fontSize: 11.5, color: "var(--muted)", lineHeight: 1.6 }}>💡 <b style={{ color: "var(--text)" }}>Tip:</b> Different pooling methods capture information in different ways. Switch between them and watch the final vector, the diagram and the token contributions change.</div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: Indexing
// ═══════════════════════════════════════════════════════════════════════════
function Breadcrumb({ active }: { active: ModuleId }) {
  const steps: { id: ModuleId; n: number; t: string; s: string }[] = [
    { id: "chunk", n: 1, t: "Chunk", s: "Text chunk" }, { id: "embed", n: 2, t: "Embedding", s: "Chunk vector" },
    { id: "index", n: 3, t: "Indexing", s: "Store vector" }, { id: "retrieve", n: 4, t: "Retrieve", s: "Search vectors" }, { id: "e2e", n: 5, t: "Answer", s: "Generate response" },
  ];
  return (
    <div className="row" style={{ gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
      {steps.map((s, i) => (
        <div key={s.id} className="row" style={{ gap: 8, alignItems: "center" }}>
          <div style={{ border: `1px solid ${s.id === active ? GREEN : "var(--border)"}`, borderRadius: 10, padding: "8px 14px", background: s.id === active ? `color-mix(in srgb, ${GREEN} 10%, transparent)` : "var(--panel)", opacity: steps.findIndex((x) => x.id === active) < i ? 0.5 : 1 }}>
            <div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 20, height: 20, borderRadius: "50%", display: "grid", placeItems: "center", background: s.id === active ? GREEN : "var(--panel-2)", color: s.id === active ? "#fff" : "var(--muted)", fontSize: 11, fontWeight: 700 }}>{s.n}</span><div><div style={{ fontSize: 12, fontWeight: 600 }}>{s.t}</div><div style={{ fontSize: 9.5, color: "var(--faint)" }}>{s.s}</div></div></div>
          </div>
          {i < steps.length - 1 && <span style={{ color: "var(--faint)" }}>›</span>}
        </div>
      ))}
    </div>
  );
}
// Reusable interactive scatter: zoom (buttons + wheel), drag-pan, click a point.
// Point coords are in data space (roughly −1…1); the view maps them to screen and
// recomputes on zoom/pan so markers stay crisp. onPick(i) fires on a click (not drag).
type SPoint = { x: number; y: number; color: string; r?: number; star?: boolean; ring?: boolean; dim?: boolean; label?: string };
function ScatterPlot({ points, links = [], onPick, legend }: { points: SPoint[]; links?: [number, number][]; onPick?: (i: number) => void; legend?: React.ReactNode }) {
  const M = 36, DOM = 1.12;
  const [view, setView] = useState({ cx: 0, cy: 0, zoom: 1 });
  const [hover, setHover] = useState(-1);
  const [panning, setPanning] = useState(false);
  const [cw, setCw] = useState(680);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  // Fill the container width (measured) with a fixed landscape height.
  useEffect(() => { const el = wrapRef.current; if (!el || typeof ResizeObserver === "undefined") return; const ro = new ResizeObserver((es) => { const w = es[0]?.contentRect.width; if (w && w > 0) setCw(w); }); ro.observe(el); return () => ro.disconnect(); }, []);
  const W = Math.max(320, Math.round(cw));
  const H = Math.round(Math.max(280, Math.min(480, W * 0.5)));
  const drag = useRef<{ px: number; py: number; cx: number; cy: number; moved: boolean } | null>(null);
  const movedRef = useRef(false);
  // Keep the view framed on the points: zoom ≥ 1 (can't zoom out past the fit) and
  // pan clamped so the window never leaves the data frame [−DOM, DOM].
  const clampView = (cx: number, cy: number, zoom: number) => { const z = Math.max(1, Math.min(12, zoom)); const h = DOM / z; const m = Math.max(0, DOM - h); return { zoom: z, cx: Math.max(-m, Math.min(m, cx)), cy: Math.max(-m, Math.min(m, cy)) }; };
  const half = DOM / view.zoom;
  const x0 = view.cx - half, x1 = view.cx + half, y0 = view.cy - half, y1 = view.cy + half;
  const sx = (x: number) => M + ((x - x0) / (x1 - x0)) * (W - 2 * M);
  const sy = (y: number) => M + (1 - (y - y0) / (y1 - y0)) * (H - 2 * M);
  const toVb = (e: { clientX: number; clientY: number }): [number, number] => { const r = svgRef.current?.getBoundingClientRect(); if (!r) return [0, 0]; return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H]; };
  useEffect(() => {
    const el = svgRef.current; if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [px, py] = toVb(e); const fx = (px - M) / (W - 2 * M), fy = (py - M) / (H - 2 * M);
      const dx = x0 + fx * (x1 - x0), dy = y0 + (1 - fy) * (y1 - y0);
      const nz = Math.max(1, Math.min(12, view.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); const nh = DOM / nz;
      setView(clampView(dx + nh * (1 - 2 * fx), dy + nh * (2 * fy - 1), nz));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [view, x0, x1, y0, y1, W, H]);
  const onDown = (e: React.PointerEvent) => { const [px, py] = toVb(e); drag.current = { px, py, cx: view.cx, cy: view.cy, moved: false }; movedRef.current = false; setPanning(true); (e.target as Element).setPointerCapture?.(e.pointerId); };
  const onMove = (e: React.PointerEvent) => { if (!drag.current) return; const [px, py] = toVb(e); const dpx = px - drag.current.px, dpy = py - drag.current.py; if (Math.abs(dpx) + Math.abs(dpy) > 2) { drag.current.moved = true; movedRef.current = true; } setView((v) => clampView(drag.current!.cx - (dpx / (W - 2 * M)) * (x1 - x0), drag.current!.cy + (dpy / (H - 2 * M)) * (y1 - y0), v.zoom)); };
  const onUp = () => { drag.current = null; setPanning(false); };
  const pickIf = (i: number) => { if (!movedRef.current) onPick?.(i); };
  const zb = (t: string, fn: () => void) => <button onClick={fn} style={{ width: 24, height: 24, borderRadius: 6, border: "1px solid var(--border)", background: "var(--panel-2)", color: "var(--muted)", cursor: "pointer", fontSize: 13, lineHeight: 1, fontFamily: "inherit" }}>{t}</button>;
  const gridT = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <div style={{ position: "absolute", top: 6, right: 6, zIndex: 2, display: "flex", gap: 4 }}>
        {zb("+", () => setView((v) => clampView(v.cx, v.cy, v.zoom * 1.4)))}
        {zb("−", () => setView((v) => clampView(v.cx, v.cy, v.zoom / 1.4)))}
        {zb("⤢", () => setView({ cx: 0, cy: 0, zoom: 1 }))}
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", cursor: panning ? "grabbing" : "grab", touchAction: "none" }} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
        <defs><clipPath id="spclip"><rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} /></clipPath></defs>
        {gridT.map((f, i) => { const tv = x0 + f * (x1 - x0), X = sx(tv); return <g key={`gx${i}`}><line x1={X} y1={M} x2={X} y2={H - M} stroke="var(--border)" strokeWidth={0.5} opacity={0.4} /><text x={X} y={H - M + 12} fontSize={8} fill="var(--faint)" textAnchor="middle">{tv.toFixed(1)}</text></g>; })}
        {gridT.map((f, i) => { const tv = y0 + f * (y1 - y0), Y = sy(tv); return <g key={`gy${i}`}><line x1={M} y1={Y} x2={W - M} y2={Y} stroke="var(--border)" strokeWidth={0.5} opacity={0.4} /><text x={M - 6} y={Y + 3} fontSize={8} fill="var(--faint)" textAnchor="end">{tv.toFixed(1)}</text></g>; })}
        <g clipPath="url(#spclip)">
          {links.map(([a, b], i) => { const A = points[a], B = points[b]; if (!A || !B) return null; return <line key={`l${i}`} x1={sx(A.x)} y1={sy(A.y)} x2={sx(B.x)} y2={sy(B.y)} stroke={GREEN} strokeWidth={1.3} opacity={0.6} strokeDasharray="3 2" />; })}
          {points.map((p, i) => { const X = sx(p.x), Y = sy(p.y), hov = hover === i;
            if (p.star) return <text key={i} x={X} y={Y + 5} fontSize={hov ? 20 : 16} fill={p.color} textAnchor="middle" style={{ cursor: "pointer" }} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(-1)} onClick={(e) => { e.stopPropagation(); pickIf(i); }}>★</text>;
            return (
              <g key={i} style={{ cursor: "pointer" }} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(-1)} onClick={(e) => { e.stopPropagation(); pickIf(i); }}>
                {p.ring && <circle cx={X} cy={Y} r={(p.r || 6) + 3} fill="none" stroke={p.color} strokeWidth={2.5} />}
                <circle cx={X} cy={Y} r={hov ? (p.r || 4) + 2 : (p.r || 4)} fill={p.color} opacity={p.dim && !hov ? 0.55 : 1} />
                {p.label && hov && <text x={X} y={Y - 9} fontSize={9} fill="var(--text)" textAnchor="middle" fontFamily="var(--mono)" style={{ pointerEvents: "none" }}>{p.label}</text>}
              </g>
            );
          })}
        </g>
      </svg>
      {legend}
      <div className="note" style={{ textAlign: "center", fontSize: 9.5, marginTop: 2 }}>scroll to zoom · drag to pan · click a point for details · ⤢ resets</div>
    </div>
  );
}
const IDX_AGO = ["2m ago", "3m ago", "5m ago", "6m ago", "8m ago", "10m ago", "12m ago", "15m ago"];
// Vector-processing pipeline card + connector + little visuals (reference-style).
function ProcCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ flex: 1, minWidth: 0, border: `1px solid ${GREEN}`, borderRadius: 12, background: `color-mix(in srgb, ${GREEN} 7%, transparent)`, padding: "12px 6px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 5, minHeight: 118, justifyContent: "center" }}>
      <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 2 }}>{title}</div>
      {children}
    </div>
  );
}
const ProcArrow = () => <span style={{ color: GREEN, fontSize: 15, alignSelf: "center", flex: "0 0 auto" }}>→</span>;
function NormRing() {
  return (
    <svg width={40} height={40} viewBox="0 0 44 44" style={{ display: "block" }}>
      <style>{`@keyframes ppspin{to{transform:rotate(360deg)}}`}</style>
      <g style={{ transformOrigin: "22px 22px", animation: "ppspin 4s linear infinite" }}>
        {Array.from({ length: 16 }).map((_, i) => { const a = (i / 16) * 2 * Math.PI; const x = 22 + 16 * Math.cos(a), y = 22 + 16 * Math.sin(a); return <circle key={i} cx={x} cy={y} r={2.2} fill={GREEN} opacity={0.25 + 0.75 * (i / 16)} />; })}
      </g>
    </svg>
  );
}
function DbIcon() {
  return (
    <svg width={30} height={34} viewBox="0 0 30 34" style={{ display: "block" }}>
      {[0, 8, 16].map((oy, i) => <g key={i}><ellipse cx={15} cy={7 + oy} rx={11} ry={4} fill={`color-mix(in srgb, ${GREEN} ${28 - i * 4}%, transparent)`} stroke={GREEN} strokeWidth={1.2} /></g>)}
      <path d={`M4 7 V23`} stroke={GREEN} strokeWidth={1.2} fill="none" /><path d={`M26 7 V23`} stroke={GREEN} strokeWidth={1.2} fill="none" />
    </svg>
  );
}
function IndexingModule({ store }: { store: { v: number[]; label: string }[] }) {
  const dim = 8;
  const [chunk, setChunk] = useState("Machine learning is a subset of artificial intelligence.");
  const [indexType, setIndexType] = useState("HNSW");
  const [metric, setMetric] = useState("Cosine Similarity");
  const [mConn, setMConn] = useState(16);
  const [efc, setEfc] = useState(200);
  const [query, setQuery] = useState("What is machine learning?");
  const vec = useMemo(() => embedText(chunk, dim, "mean"), [chunk]);
  const normVec = normalize(vec);
  // vector store = demo corpus + the current chunk + anything added from Embedding module
  const rows = useMemo(() => {
    const corpus = DEMO_CORPUS.map((t, i) => ({ id: `vec_${String(i + 1).padStart(3, "0")}`, text: t, v: embedText(t, dim, "mean") }));
    const extra = store.map((s, i) => ({ id: `usr_${String(i + 1).padStart(3, "0")}`, text: s.label, v: s.v.length === dim ? s.v : embedText(s.label, dim, "mean") }));
    return [...corpus, ...extra, { id: "vec_new", text: chunk, v: vec }];
  }, [chunk, vec, store]);
  const pts = useMemo(() => pca2(rows.map((r) => r.v)), [rows]);
  const [detail, setDetail] = useState<{ title: string; node: React.ReactNode } | null>(null);
  // Similarity ranking respects the SELECTED distance metric (Euclidean sorts ascending).
  const results = useMemo(() => {
    const q = embedText(query, dim, "mean");
    const asc = metric.startsWith("Euclidean");
    return rows.map((r, i) => ({ i, id: r.id, text: r.text, score: metricScore(metric, q, r.v) }))
      .sort((a, b) => (asc ? a.score - b.score : b.score - a.score)).slice(0, 5);
  }, [query, rows, metric]);
  // "Apply settings" → show HOW this index config searches, with the metric's real math.
  function showApply() {
    const q = embedText(query, dim, "mean");
    const top = results[0]; const tv = top ? rows[top.i].v : vec;
    const dp = dot(q, tv), na = l2(q), nb = l2(tv);
    const metricBlock = metric.startsWith("Cosine")
      ? { formula: "cos(q,v) = (q·v) / (‖q‖·‖v‖)", lines: [`q·v = ${dp.toFixed(3)}`, `‖q‖ = ${na.toFixed(3)} · ‖v‖ = ${nb.toFixed(3)}`], result: `= ${(dp / ((na * nb) || 1)).toFixed(3)}   (higher = closer)` }
      : metric.startsWith("Euclidean")
        ? { formula: "d(q,v) = √( Σ (qᵢ − vᵢ)² )", lines: [`Σ(qᵢ−vᵢ)² = ${q.reduce((s, x, i) => s + (x - tv[i]) ** 2, 0).toFixed(3)}`], result: `d = ${euclidD(q, tv).toFixed(3)}   (lower = closer)` }
        : { formula: "q·v = Σ qᵢ·vᵢ", lines: [`= ${q.map((x, i) => `${x.toFixed(2)}·${tv[i].toFixed(2)}`).slice(0, 3).join(" + ")} + …`], result: `q·v = ${dp.toFixed(3)}   (higher = closer)` };
    const structure = indexType === "HNSW"
      ? `HNSW builds a multi-layer proximity graph. Each vector links to M=${mConn} neighbours; a search greedily hops toward the query, exploring ef_construction=${efc} candidates while building. It finds approximate nearest neighbours in ~log(N) hops instead of scanning all ${rows.length} vectors.`
      : indexType === "IVF_FLAT"
        ? `IVF_FLAT clusters the ${rows.length} vectors into buckets (≈√N). At query time it only searches the nearest bucket(s), skipping the rest — fast, with a small recall trade-off.`
        : `Flat does an exact brute-force scan: it compares the query to all ${rows.length} stored vectors. Slowest but 100% recall.`;
    setDetail({
      title: `How ${indexType} + ${metric.split(" ")[0]} search works`,
      node: (
        <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
          <div style={{ fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>Index structure</div>
          <p style={{ marginTop: 0 }}>{structure}</p>
          <div style={{ fontWeight: 700, color: "var(--text)", margin: "12px 0 6px" }}>Distance metric — query vs nearest vector <span style={{ fontFamily: "var(--mono)", color: GREEN }}>{top?.id}</span></div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)", marginBottom: 8 }}>{metricBlock.formula}</div>
          <CalcBlock color={GREEN} lines={[`q = ${fmt(q, 4)}`, `v = ${fmt(tv, 4)}`, ...metricBlock.lines]} result={metricBlock.result} />
          <p style={{ marginBottom: 0, marginTop: 10 }}>These settings now drive the <b style={{ color: "var(--text)" }}>Test similarity</b> ranking — switch the metric and the Top-5 order changes.</p>
        </div>
      ),
    });
    toast(`Applied ${indexType} · ${metric.split(" ")[0]}`, "success");
  }
  // normalize PCA coords into [−1,1] for the interactive scatter
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const nrm = (v: number, arr: number[]) => { const lo = Math.min(...arr), hi = Math.max(...arr); return ((v - lo) / ((hi - lo) || 1)) * 2 - 1; };
  function showVec(i: number) {
    const r = rows[i]; const qv = embedText(query, dim, "mean");
    setDetail({
      title: `${r.id}${r.id === "vec_new" ? " · this chunk" : ""}`,
      node: (
        <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
          <div style={{ color: "var(--text)", marginBottom: 8 }}>“{r.text}”</div>
          <CalcBlock color={GREEN} lines={[`vector = ${fmt(r.v)}`, `‖v‖ = ${l2(r.v).toFixed(3)}`]} result={`cosine to query = ${cosineD(qv, r.v).toFixed(3)}`} />
        </div>
      ),
    });
  }
  return (
    <div>
      {detail && <Modal title={detail.title} onClose={() => setDetail(null)}>{detail.node}</Modal>}
      <div className="note" style={{ marginBottom: 12, fontSize: 12.5 }}>See how embeddings are prepared and stored in a vector database for fast similarity search.</div>
      <Breadcrumb active="index" />
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 360px", gap: 16 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Section title="Input chunk" right={<button onClick={() => { navigator.clipboard?.writeText(JSON.stringify(vec)); toast("Embedding copied", "success"); }} style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 7, color: "var(--muted)", fontSize: 11, padding: "4px 9px", cursor: "pointer", fontFamily: "inherit" }}>📋 Copy</button>}>
            <input value={chunk} onChange={(e) => setChunk(e.target.value)} style={inp} />
            <div style={{ marginTop: 10, fontFamily: "var(--mono)", fontSize: 12, color: GREEN, border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px", background: "var(--panel-2)", overflowX: "auto" }}>Embedding ({dim}D): {fmt(vec)}</div>
          </Section>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 210px", gap: 16 }}>
            <Section title="1. Vector processing">
              <div className="note" style={{ marginBottom: 12, fontSize: 11 }}>Prepare the embedding before storing.</div>
              <div className="row" style={{ gap: 4, alignItems: "stretch", flexWrap: "nowrap" }}>
                <ProcCard title="Normalization">
                  <NormRing />
                  <div style={{ fontSize: 9.5, color: "var(--faint)", fontFamily: "var(--mono)" }}>L2 normalize vector</div>
                </ProcCard>
                <ProcArrow />
                <ProcCard title="L2 Norm">
                  <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "var(--mono)" }}>{l2(vec).toFixed(2)}</div>
                  <div style={{ fontSize: 9.5, color: "var(--faint)" }}>Normalized</div>
                  <div style={{ fontSize: 10.5, color: GREEN, fontWeight: 700 }}>✓ Yes</div>
                </ProcCard>
                <ProcArrow />
                <ProcCard title="Dimension check">
                  <div style={{ fontSize: 9, color: "var(--faint)" }}>Expected</div>
                  <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--mono)" }}>{dim}D</div>
                  <div style={{ fontSize: 9, color: "var(--faint)" }}>Actual</div>
                  <div style={{ fontSize: 12.5, fontWeight: 700, fontFamily: "var(--mono)", color: GREEN }}>{normVec.length}D ✓</div>
                </ProcCard>
                <ProcArrow />
                <ProcCard title="Vector ready">
                  <DbIcon />
                  <div style={{ fontSize: 9.5, color: "var(--faint)" }}>Ready to store</div>
                </ProcCard>
              </div>
              <div style={{ marginTop: 12, fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", overflowX: "auto" }}>normalized: {fmt(normVec.map((x) => Math.round(x * 100) / 100))}</div>
            </Section>
            <Section title="Index settings" pad={12}>
              {[["Index type", indexType, setIndexType, ["HNSW", "IVF_FLAT", "Flat"]], ["Distance metric", metric, setMetric, ["Cosine Similarity", "Euclidean (L2)", "Dot product"]]].map(([label, val, set, opts]) => (
                <div key={label as string} style={{ marginBottom: 8 }}><label style={{ ...lbl, fontSize: 10 }}>{label as string}</label><select value={val as string} onChange={(e) => (set as (s: string) => void)(e.target.value)} style={{ ...inp, padding: "6px 8px", fontSize: 11.5 }}>{(opts as string[]).map((o) => <option key={o}>{o}</option>)}</select></div>
              ))}
              <div style={{ marginBottom: 8 }}><label style={{ ...lbl, fontSize: 10 }}>Dimensions</label><input value={dim} readOnly style={{ ...inp, padding: "6px 8px", fontSize: 11.5, opacity: 0.7 }} /></div>
              <div style={{ marginBottom: 8 }}><label style={{ ...lbl, fontSize: 10 }}><Info tip={G.mconn}>M (connections)</Info></label><input type="number" value={mConn} onChange={(e) => setMConn(+e.target.value)} style={{ ...inp, padding: "6px 8px", fontSize: 11.5 }} /></div>
              <div style={{ marginBottom: 10 }}><label style={{ ...lbl, fontSize: 10 }}><Info tip={G.efc}>ef_construction</Info></label><input type="number" value={efc} onChange={(e) => setEfc(+e.target.value)} style={{ ...inp, padding: "6px 8px", fontSize: 11.5 }} /></div>
              <button onClick={showApply} style={{ width: "100%", padding: "7px", borderRadius: 8, border: "none", background: GREEN, color: "#04120a", fontSize: 11.5, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>Apply settings — show how it works</button>
            </Section>
          </div>
          <Section title="3. Vector store" right={<Pill color={GREEN}>{rows.length} vectors</Pill>}>
            <div className="note" style={{ marginBottom: 8, fontSize: 11 }}>This vector has been indexed and stored.</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 11.5 }}>
                <thead><tr><th style={th}>ID</th><th style={th}>Chunk (preview)</th><th style={th}>Vector ({dim}D)</th><th style={th}>Norm</th><th style={th}>Added</th></tr></thead>
                <tbody>
                  {rows.map((r, i) => { const isNew = r.id === "vec_new"; return (
                    <tr key={r.id} style={{ background: isNew ? `color-mix(in srgb, ${GREEN} 8%, transparent)` : "transparent" }}>
                      <td style={{ ...td, color: isNew ? GREEN : "var(--muted)", fontFamily: "var(--mono)" }}>{r.id}</td>
                      <td style={{ ...td, maxWidth: 210, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: isNew ? GREEN : "var(--text)" }}>{r.text}</td>
                      <td style={{ ...td, fontFamily: "var(--mono)", color: "var(--faint)" }}>[{r.v.slice(0, 3).map((x: number) => x.toFixed(2)).join(", ")}, …]</td>
                      <td style={{ ...td, fontFamily: "var(--mono)", color: isNew ? GREEN : "var(--text)" }}>{l2(r.v).toFixed(2)}</td>
                      <td style={{ ...td, fontFamily: "var(--mono)", color: isNew ? GREEN : "var(--faint)" }}>{isNew ? "Just now" : IDX_AGO[i % IDX_AGO.length]}</td>
                    </tr>
                  ); })}
                </tbody>
              </table>
            </div>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
              <span style={{ fontSize: 12, color: GREEN, fontWeight: 600 }}>Total vectors: {rows.length}</span>
              <button onClick={() => toast(`Vector store holds ${rows.length} vectors`, "success")} style={{ ...actBtn, padding: "5px 10px", fontSize: 11 }}>⧉ View all vectors</button>
            </div>
          </Section>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Section title={<>2D Vector space <span style={{ color: "var(--faint)", fontWeight: 400, fontSize: 11 }}>(<Info tip={G.pca}>PCA projection</Info>)</span></>}>
            <div className="note" style={{ marginBottom: 6, fontSize: 11 }}>Spatial view of indexed vectors.</div>
            <ScatterPlot
              points={pts.map((p, i) => ({ x: nrm(p.x, xs), y: nrm(p.y, ys), color: rows[i].id === "vec_new" ? GREEN : ACC, ring: rows[i].id === "vec_new", dim: rows[i].id !== "vec_new", label: rows[i].id }))}
              onPick={showVec}
              legend={<div className="row" style={{ gap: 14, justifyContent: "center", fontSize: 10.5, color: "var(--muted)", marginTop: 4 }}><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 9, height: 9, borderRadius: "50%", border: `2px solid ${GREEN}` }} />New vector (this chunk)</span><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 9, height: 9, borderRadius: "50%", background: ACC }} />Existing vectors</span></div>}
            />
          </Section>
          <Section title={<>Test similarity <span style={{ color: "var(--faint)", fontWeight: 400, fontSize: 11 }}>(with a query)</span></>}>
            <div className="note" style={{ marginBottom: 8, fontSize: 11 }}>Enter a query to see nearest vectors.</div>
            <div className="row" style={{ gap: 6 }}>
              <input value={query} onChange={(e) => setQuery(e.target.value)} style={inp} placeholder="Enter a query…" />
              <button onClick={() => toast("Searching…", "success")} style={{ padding: "8px 14px", borderRadius: 8, border: "none", background: GREEN, color: "#04120a", fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>Search</button>
            </div>
            <div className="row" style={{ justifyContent: "space-between", marginTop: 10, fontSize: 10, color: "var(--faint)", textTransform: "uppercase", letterSpacing: ".04em" }}><span>Top 5 results</span><span>Score ({metric.split(" ")[0]})</span></div>
            <div style={{ marginTop: 4 }}>
              {results.map((r, k) => (
                <div key={r.id} style={{ padding: "7px 10px", borderRadius: 8, background: k === 0 ? `color-mix(in srgb, ${GREEN} 9%, transparent)` : "transparent", marginBottom: 4 }}>
                  <div className="row" style={{ justifyContent: "space-between", fontSize: 11 }}><span style={{ fontFamily: "var(--mono)", color: k === 0 ? GREEN : "var(--muted)" }}>{r.id}</span><b style={{ fontFamily: "var(--mono)", color: k === 0 ? GREEN : "var(--text)" }}>{r.score.toFixed(3)}</b></div>
                  <div style={{ fontSize: 11, color: "var(--faint)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.text}</div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      </div>
      {/* explainer row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12, marginTop: 16 }}>
        <div style={{ ...panel, padding: 14 }}><div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 6 }}><span style={{ fontSize: 16 }}>🗄</span><b style={{ fontSize: 12.5 }}>What happens in indexing?</b></div><div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.55 }}>Indexing stores embeddings in a vector database with an index structure (like {indexType}) to enable fast, efficient similarity search.</div></div>
        <div style={{ ...panel, padding: 14 }}><b style={{ fontSize: 12.5 }}>Key points</b><ul style={{ margin: "6px 0 0", paddingLeft: 4, listStyle: "none", fontSize: 11.5, color: "var(--muted)", lineHeight: 1.7 }}><li>✅ Vectors are normalized for consistent similarity</li><li>✅ The index structure speeds up nearest-neighbour search</li><li>✅ Metadata (id, text, source) is stored with each vector</li></ul></div>
        <div style={{ ...panel, padding: 14 }}><div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 6 }}><span style={{ fontSize: 16 }}>💡</span><b style={{ fontSize: 12.5 }}>Tip</b></div><div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.55 }}>Cosine similarity works best with normalized vectors. {indexType} is great for high performance on large datasets.</div></div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: Retrieve
// ═══════════════════════════════════════════════════════════════════════════
const RETR_EXAMPLES = ["How do neural networks learn?", "What is machine learning?", "How are images understood?", "learning from rewards"];
const RETR_METRICS: [string, string][] = [["Cosine Similarity", "Cosine"], ["Euclidean (L2)", "Euclidean"], ["Dot product", "Dot"]];
function RetrieveModule() {
  const dim = 8;
  const [query, setQuery] = useState("How do neural networks learn?");
  const [topK, setTopK] = useState(3);
  const [metric, setMetric] = useState("Cosine Similarity");
  const [detail, setDetail] = useState<{ title: string; node: React.ReactNode } | null>(null);
  const rows = useMemo(() => DEMO_CORPUS.map((t, i) => ({ id: `vec_${String(i + 1).padStart(3, "0")}`, text: t, v: embedText(t, dim, "mean") })), []);
  const q = useMemo(() => embedText(query, dim, "mean"), [query]);
  const asc = metric.startsWith("Euclidean");
  const ranked = useMemo(() => rows.map((r, i) => ({ ...r, i, score: metricScore(metric, q, r.v) })).sort((a, b) => (asc ? a.score - b.score : b.score - a.score)), [rows, q, metric, asc]);
  const scores = ranked.map((r) => r.score);
  const mn = Math.min(...scores), mx = Math.max(...scores);
  const barVal = (s: number) => { const t = (s - mn) / ((mx - mn) || 1); return asc ? 1 - t : t; };
  const mean = scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
  const margin = ranked.length > 1 ? Math.abs(ranked[0].score - ranked[1].score) : 0;
  // PCA of corpus + query (query is the last point)
  const pts = useMemo(() => pca2([...rows.map((r) => r.v), q]), [rows, q]);
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const nrm = (v: number, arr: number[]) => { const lo = Math.min(...arr), hi = Math.max(...arr); return ((v - lo) / ((hi - lo) || 1)) * 2 - 1; };
  const qpt = pts[rows.length];
  const retrievedIdx = new Set(ranked.slice(0, topK).map((r) => r.i));

  function showScore(r: { id: string; text: string; v: number[] }) {
    const dp = dot(q, r.v), na = l2(q), nb = l2(r.v);
    const b = metric.startsWith("Cosine") ? { f: "cos(q,v) = (q·v) / (‖q‖·‖v‖)", lines: [`q·v = ${dp.toFixed(3)}`, `‖q‖ = ${na.toFixed(3)} · ‖v‖ = ${nb.toFixed(3)}`], res: `= ${cosineD(q, r.v).toFixed(3)}   (higher = closer)` }
      : metric.startsWith("Euclidean") ? { f: "d(q,v) = √( Σ (qᵢ − vᵢ)² )", lines: [`Σ (qᵢ−vᵢ)² = ${q.reduce((s, x, i) => s + (x - r.v[i]) ** 2, 0).toFixed(3)}`], res: `d = ${euclidD(q, r.v).toFixed(3)}   (lower = closer)` }
        : { f: "q·v = Σ qᵢ·vᵢ", lines: [`= ${q.map((x, i) => `${x.toFixed(2)}·${r.v[i].toFixed(2)}`).slice(0, 3).join(" + ")} + …`], res: `q·v = ${dp.toFixed(3)}   (higher = closer)` };
    setDetail({
      title: `Score · query ↔ ${r.id}`,
      node: (
        <div>
          <div style={{ fontSize: 12.5, color: "var(--text)", marginBottom: 8 }}>“{r.text}”</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)", marginBottom: 8 }}>{b.f}</div>
          <CalcBlock color={GREEN} lines={[`q = ${fmt(q, 4)}`, `v = ${fmt(r.v, 4)}`, ...b.lines]} result={b.res} />
        </div>
      ),
    });
  }
  const reset = () => { setQuery("How do neural networks learn?"); setTopK(3); setMetric("Cosine Similarity"); };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 300px", gap: 16 }}>
      {detail && <Modal title={detail.title} onClose={() => setDetail(null)}>{detail.node}</Modal>}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title="Retrieve Playground" right={<button onClick={reset} style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--muted)", fontSize: 11.5, fontWeight: 600, padding: "5px 10px", cursor: "pointer", fontFamily: "inherit" }}>↻ Reset</button>}>
          <div className="note" style={{ marginBottom: 10, fontSize: 12 }}>Embed a query, score it against every stored chunk, and keep the closest ones.</div>
          <label style={lbl}>Query</label>
          <input value={query} onChange={(e) => setQuery(e.target.value)} style={inp} />
          <div className="row" style={{ gap: 6, marginTop: 8, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 11, color: "var(--faint)" }}>Try:</span>
            {RETR_EXAMPLES.map((ex) => <button key={ex} onClick={() => setQuery(ex)} style={{ padding: "4px 9px", borderRadius: 20, border: `1px solid ${query === ex ? BLUE : "var(--border)"}`, background: query === ex ? `color-mix(in srgb, ${BLUE} 14%, transparent)` : "var(--panel-2)", color: query === ex ? BLUE : "var(--muted)", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>{ex}</button>)}
          </div>
          <div style={{ marginTop: 10, fontFamily: "var(--mono)", fontSize: 11.5, color: BLUE, border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px", background: "var(--panel-2)", overflowX: "auto" }}>query → {fmt(q)}</div>
          <div className="row" style={{ gap: 14, marginTop: 12, alignItems: "center", flexWrap: "wrap", justifyContent: "space-between" }}>
            <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Distance</span>
              {RETR_METRICS.map(([m, short]) => <Tip key={m} tip={m.startsWith("Cosine") ? G.cosine : m.startsWith("Euclidean") ? "Euclidean distance: straight-line distance √(Σ(qᵢ−vᵢ)²). Lower = closer." : "Dot product: Σ qᵢ·vᵢ — rewards both alignment and magnitude."}><Btn on={metric === m} onClick={() => setMetric(m)}>{short}</Btn></Tip>)}
            </div>
            <label className="row" style={{ gap: 8, alignItems: "center", fontSize: 12, color: "var(--muted)" }}><Info tip={G.topK}>Top-K</Info> <input type="range" min={1} max={6} value={topK} onChange={(e) => setTopK(+e.target.value)} /><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{topK}</b></label>
          </div>
        </Section>

        <Section title={<>Query in vector space <span style={{ color: "var(--faint)", fontWeight: 400, fontSize: 11 }}>(<Info tip={G.pca}>PCA</Info>)</span></>}>
          <div className="note" style={{ marginBottom: 6, fontSize: 11 }}>Lines connect the query ★ to its {topK} nearest chunks. Click any point for details.</div>
          <ScatterPlot
            points={[
              ...rows.map((r, i) => ({ x: nrm(pts[i].x, xs), y: nrm(pts[i].y, ys), color: retrievedIdx.has(i) ? GREEN : ACC, r: retrievedIdx.has(i) ? 6 : 4, dim: !retrievedIdx.has(i), label: r.id })),
              ...(qpt ? [{ x: nrm(qpt.x, xs), y: nrm(qpt.y, ys), color: BLUE, star: true, label: "query" }] : []),
            ]}
            links={ranked.slice(0, topK).map((r) => [rows.length, r.i] as [number, number])}
            onPick={(i) => { if (i < rows.length) showScore(rows[i]); else setDetail({ title: "Query vector", node: <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}><div style={{ color: "var(--text)", marginBottom: 8 }}>“{query}”</div><CalcBlock color={BLUE} lines={[`tokens = ${words(query).length}`, `query vector = ${fmt(q)}`]} result={`‖q‖ = ${l2(q).toFixed(3)}`} /></div> }); }}
            legend={<div className="row" style={{ gap: 14, justifyContent: "center", fontSize: 10.5, color: "var(--muted)", marginTop: 4 }}><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ color: BLUE, fontSize: 13 }}>★</span> query</span><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 9, height: 9, borderRadius: "50%", background: GREEN }} />retrieved</span><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 9, height: 9, borderRadius: "50%", background: ACC, opacity: 0.55 }} />other</span></div>}
          />
        </Section>

        <Section title="Ranked results" right={<Tip tip={metric.startsWith("Cosine") ? G.cosine : metric.startsWith("Euclidean") ? "Lower distance = closer." : "Higher dot = closer."}><Pill color={GREEN}>{metric.split(" ")[0]}</Pill></Tip>}>
          <div className="note" style={{ marginBottom: 8, fontSize: 11 }}>👆 Click a result to see how its score was computed. Green = kept as context (top-{topK}).</div>
          {ranked.map((r, k) => {
            const picked = k < topK;
            return (
              <button key={r.id} onClick={() => showScore(r)} style={{ display: "block", width: "100%", textAlign: "left", border: `1px solid ${picked ? GREEN : "var(--border)"}`, borderRadius: 10, padding: "10px 12px", marginBottom: 8, background: picked ? `color-mix(in srgb, ${GREEN} 7%, transparent)` : "var(--panel-2)", opacity: picked ? 1 : 0.6, cursor: "pointer", fontFamily: "inherit" }}>
                <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                  <span className="row" style={{ gap: 8, alignItems: "center" }}><Pill color={picked ? GREEN : "var(--faint)"}>#{k + 1}</Pill><span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--faint)" }}>{r.id}</span>{picked && <span style={{ fontSize: 10, color: GREEN }}>✓ retrieved</span>}</span>
                  <b style={{ fontFamily: "var(--mono)", fontSize: 13 }}>{r.score.toFixed(3)}</b>
                </div>
                <Bar v={barVal(r.score)} color={picked ? GREEN : "var(--faint)"} />
                <div style={{ fontSize: 12.5, marginTop: 6, color: "var(--text)" }}>{r.text}</div>
              </button>
            );
          })}
        </Section>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Section title="How retrieval works" pad={14}>
          <ol style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}><li>Embed the query into the same vector space</li><li>Score it against every stored chunk vector</li><li>Rank by {metric.split(" ")[0]} {asc ? "distance (lower first)" : "similarity (higher first)"}</li><li>Keep the top-K as retrieved context</li></ol>
        </Section>
        <Section title="Retrieval stats">
          {[["Query tokens", words(query).length], ["Dimensions", dim], ["Metric", metric.split(" ")[0]], ["Corpus size", rows.length], [asc ? "Best (min dist)" : "Best score", ranked[0]?.score.toFixed(3)], ["Mean score", mean.toFixed(3)], ["Top-1 margin", margin.toFixed(3)]].map(([k, v]) => (
            <div key={k as string} className="row" style={{ justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)" }}><span style={{ fontSize: 12, color: "var(--muted)" }}>{k}</span><b style={{ fontSize: 12.5, fontFamily: "var(--mono)" }}>{v}</b></div>
          ))}
        </Section>
        <Section title={<>Query vector <span style={{ color: "var(--faint)", fontWeight: 400, fontSize: 11 }}>(visual)</span></>}><Radar values={q} color={BLUE} /></Section>
        <div style={{ ...panel, padding: 14, fontSize: 11.5, color: "var(--muted)", lineHeight: 1.6 }}>💡 <b style={{ color: "var(--text)" }}>Tip:</b> A high {metric.split(" ")[0]} score means the query and chunk point in similar directions — they mean similar things. Switch the metric and watch the ranking change.</div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: Backpropagation / Training Simulator
// ═══════════════════════════════════════════════════════════════════════════
const BP_EXAMPLES: [string, string][] = [["king", "queen"], ["paris", "france"], ["hello", "hi"], ["cat", "animal"]];
const BP_STEPS = ["Forward Pass", "Calculate Loss", "Backpropagate", "Update Weights", "Next Epoch"];
function BackpropModule() {
  const dim = 8;
  const [pair, setPair] = useState<[string, string]>(["king", "queen"]);
  const [lr, setLr] = useState(0.5);
  const EPOCHS = 10;
  const target = useMemo(() => embedToken(pair[1], dim), [pair]);
  const [weight, setWeight] = useState<number[]>(() => embedToken("king", dim));
  const [prev, setPrev] = useState<number[]>(() => embedToken("king", dim));
  const [epoch, setEpoch] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reset = (p: [string, string] = pair) => {
    if (timer.current) clearTimeout(timer.current);
    const w = embedToken(p[0], dim);
    setWeight(w); setPrev(w); setEpoch(0); setStepIdx(0); setLosses([]); setRunning(false);
  };
  useEffect(() => reset(pair), [pair]);

  const mse = (w: number[]) => w.reduce((s, x, i) => s + (x - target[i]) ** 2, 0) / dim;
  const grad = weight.map((x, i) => (2 / dim) * (x - target[i]));
  const loss = mse(weight);
  const cos = cosineD(weight, target);

  // one full epoch: forward → loss → backprop → update
  const stepEpoch = () => {
    setPrev(weight);
    const g = weight.map((x, i) => (2 / dim) * (x - target[i]));
    const nw = weight.map((x, i) => Math.round((x - lr * g[i]) * 100) / 100);
    setWeight(nw); setEpoch((e) => e + 1); setLosses((L) => [...L, mse(nw)]);
  };
  useEffect(() => {
    if (!running) return;
    if (epoch >= EPOCHS) { setRunning(false); return; }
    timer.current = setTimeout(() => { setStepIdx((s) => (s + 1) % BP_STEPS.length); stepEpoch(); }, 700);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [running, epoch, weight]);

  const l2change = l2(weight.map((x, i) => x - prev[i]));
  // loss chart geometry
  const LW = 320, LH = 150, allL = losses.length ? losses : [loss];
  const maxL = Math.max(...allL, 0.001);
  const lx = (i: number) => 30 + (i / Math.max(1, EPOCHS - 1)) * (LW - 40);
  const ly = (v: number) => 10 + (1 - v / maxL) * (LH - 30);

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
        <div><h2 style={{ margin: 0, fontSize: 18 }}>🧠 Backpropagation / Embedding Training Simulator</h2><div style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 3 }}>Watch an embedding for <b style={{ color: BLUE }}>&quot;{pair[0]}&quot;</b> move toward the target <b style={{ color: AMBER }}>&quot;{pair[1]}&quot;</b> via real gradient descent.</div></div>
        <button onClick={() => reset()} style={actBtn}>↻ Reset Lab</button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "180px minmax(0,1fr) 280px", gap: 16 }}>
        {/* controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Section title="Controls" pad={12}>
            <button onClick={() => { setStepIdx((s) => (s + 1) % BP_STEPS.length); stepEpoch(); }} disabled={epoch >= EPOCHS} style={{ ...actBtn, width: "100%", marginBottom: 8, opacity: epoch >= EPOCHS ? 0.5 : 1 }}>▶ Step epoch</button>
            <button onClick={() => setRunning((r) => !r)} disabled={epoch >= EPOCHS} style={{ ...actBtn, width: "100%", marginBottom: 8, background: running ? RED : ACC, color: "#fff", border: "none", opacity: epoch >= EPOCHS ? 0.5 : 1 }}>{running ? "⏸ Pause" : "⚡ Auto train"}</button>
            <label style={{ ...lbl, marginTop: 6 }}><Info tip={G.learningRate}>Learning rate</Info>: {lr}</label>
            <input type="range" min={0.1} max={1.5} step={0.1} value={lr} onChange={(e) => setLr(+e.target.value)} style={{ width: "100%" }} />
          </Section>
          <Section title="Quick examples" pad={10}>
            {BP_EXAMPLES.map(([a, b]) => <button key={a} onClick={() => setPair([a, b])} style={{ ...actBtn, width: "100%", marginBottom: 6, justifyContent: "flex-start", background: pair[0] === a ? `color-mix(in srgb, ${ACC} 15%, transparent)` : "var(--panel-2)", color: pair[0] === a ? ACC : "var(--text)", border: `1px solid ${pair[0] === a ? ACC : "var(--border)"}` }}>▶ {a} → {b}</button>)}
          </Section>
        </div>
        {/* pipeline + step detail */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Section title="Training pipeline">
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              {BP_STEPS.map((s, i) => (
                <div key={s} className="row" style={{ gap: 6, alignItems: "center" }}>
                  <div style={{ textAlign: "center" }}><div style={{ width: 34, height: 34, borderRadius: "50%", display: "grid", placeItems: "center", margin: "0 auto", border: `2px solid ${stepIdx === i ? ACC : "var(--border)"}`, background: stepIdx === i ? `color-mix(in srgb, ${ACC} 15%, transparent)` : "transparent", color: stepIdx === i ? ACC : "var(--muted)", fontWeight: 700 }}>{i + 1}</div><div style={{ fontSize: 9.5, marginTop: 4, color: stepIdx === i ? ACC : "var(--faint)", width: 62 }}>{s}</div></div>
                  {i < BP_STEPS.length - 1 && <span style={{ color: "var(--faint)" }}>→</span>}
                </div>
              ))}
            </div>
          </Section>
          <Section title={`Step ${stepIdx + 1}: ${BP_STEPS[stepIdx]}`}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: 12, background: "var(--panel-2)" }}>
                <div style={{ fontSize: 11, color: BLUE, fontWeight: 600, marginBottom: 6 }}>Embedding (current) — &quot;{pair[0]}&quot;</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: BLUE, wordBreak: "break-all" }}>{fmt(weight)}</div>
              </div>
              <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: 12, background: "var(--panel-2)" }}>
                <div style={{ fontSize: 11, color: AMBER, fontWeight: 600, marginBottom: 6 }}>Target — &quot;{pair[1]}&quot;</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: AMBER, wordBreak: "break-all" }}>{fmt(target)}</div>
              </div>
            </div>
            <div style={{ marginTop: 12, border: "1px solid var(--border)", borderRadius: 10, padding: 12 }}>
              <div style={{ fontSize: 11, color: RED, fontWeight: 600, marginBottom: 6 }}><Info tip={G.gradient} color={RED}>Gradients ∂L/∂W</Info> = (2/n)(W − target)</div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: RED, wordBreak: "break-all" }}>{fmt(grad.map((x) => Math.round(x * 100) / 100))}</div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 10 }}>Update rule: <span style={{ fontFamily: "var(--mono)" }}>W_new = W_old − lr × ∂L/∂W</span> · Model: <b>Demo-Embed-{dim}D</b> · Learning rate: <b>{lr}</b></div>
            </div>
          </Section>
          <div style={{ ...panel, padding: 14, fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>
            <b style={{ color: "var(--text)" }}>What&apos;s happening?</b> Backpropagation minimizes the difference between the current embedding and the target. The gradient points in the direction of steepest loss increase, so we step the weights the opposite way — scaled by the learning rate — a little each epoch.
          </div>
        </div>
        {/* summary + charts */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Section title="Training summary" pad={12}>
            {[["Epoch", `${epoch} / ${EPOCHS}`], ["Loss (MSE)", loss.toFixed(4)], ["Status", running ? "Running" : epoch >= EPOCHS ? "Converged" : "Ready"], ["Best loss", (losses.length ? Math.min(...losses) : loss).toFixed(4)]].map(([k, v]) => (
              <div key={k as string} className="row" style={{ justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)" }}><span style={{ fontSize: 12, color: "var(--muted)" }}>{k}</span><b style={{ fontSize: 12.5, color: k === "Status" ? (running ? AMBER : epoch >= EPOCHS ? GREEN : "var(--text)") : "var(--text)" }}>{v}</b></div>
            ))}
          </Section>
          <Section title="Loss over epochs">
            <svg viewBox={`0 0 ${LW} ${LH}`} width="100%" style={{ display: "block" }}>
              <line x1={30} y1={LH - 20} x2={LW - 10} y2={LH - 20} stroke="var(--border)" /><line x1={30} y1={10} x2={30} y2={LH - 20} stroke="var(--border)" />
              {losses.length > 1 && <polyline points={losses.map((v, i) => `${lx(i)},${ly(v)}`).join(" ")} fill="none" stroke={RED} strokeWidth={2} />}
              {losses.map((v, i) => <circle key={i} cx={lx(i)} cy={ly(v)} r={2.6} fill={RED} />)}
              <text x={4} y={14} fontSize={8} fill="var(--faint)">{maxL.toFixed(2)}</text><text x={4} y={LH - 22} fontSize={8} fill="var(--faint)">0</text>
            </svg>
          </Section>
          <Section title="Embeddings similarity" pad={14}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}><span style={{ fontSize: 12, color: "var(--muted)" }}><Info tip={G.cosine}>Cosine similarity</Info></span><b style={{ fontFamily: "var(--mono)", color: ACC, fontSize: 15 }}>{cos.toFixed(3)}</b></div>
            <Bar v={(cos + 1) / 2} color={ACC} />
            <div style={{ fontSize: 11, color: GREEN, marginTop: 8 }}>↑ Improving towards target · L2 change this step: {l2change.toFixed(3)}</div>
          </Section>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: End-to-End
// ═══════════════════════════════════════════════════════════════════════════
function EndToEndModule() {
  const dim = 8;
  const [source, setSource] = useState(DEMO_CORPUS.join(" "));
  const [query, setQuery] = useState("What is machine learning?");
  const [size, setSize] = useState(12);
  const [topK, setTopK] = useState(2);
  const chunks = useMemo(() => chunkBy(source, "sentence", size, 0), [source, size]);
  const chunkVecs = useMemo(() => chunks.map((c) => embedText(c, dim, "mean")), [chunks]);
  const ranked = useMemo(() => {
    const q = embedText(query, dim, "mean");
    return chunks.map((c, i) => ({ i, text: c, score: cosineD(q, chunkVecs[i]) })).sort((a, b) => b.score - a.score);
  }, [query, chunks, chunkVecs]);
  const picked = ranked.slice(0, topK);
  const answer = picked.map((p) => p.text).join(" ");

  const stage = (n: number, title: string, body: React.ReactNode, color: string) => (
    <div style={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)", overflow: "hidden" }}>
      <div className="row" style={{ gap: 8, alignItems: "center", padding: "9px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}><span style={{ width: 20, height: 20, borderRadius: "50%", display: "grid", placeItems: "center", background: color, color: "#fff", fontSize: 11, fontWeight: 700 }}>{n}</span><b style={{ fontSize: 13 }}>{title}</b></div>
      <div style={{ padding: 14 }}>{body}</div>
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Section title="End-to-End pipeline — run the whole thing">
        <div className="row" style={{ gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 240 }}><label style={lbl}>Knowledge source</label><textarea value={source} onChange={(e) => setSource(e.target.value)} rows={2} style={{ ...inp, resize: "vertical" }} /></div>
          <div style={{ flex: 1, minWidth: 200 }}><label style={lbl}>Question</label><input value={query} onChange={(e) => setQuery(e.target.value)} style={inp} /></div>
          <label className="row" style={{ gap: 8, alignItems: "center", fontSize: 12, color: "var(--muted)" }}>Top-K <input type="range" min={1} max={4} value={topK} onChange={(e) => setTopK(+e.target.value)} /><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{topK}</b></label>
        </div>
      </Section>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
        {stage(1, "Chunk", <div style={{ fontSize: 12, color: "var(--muted)" }}><b style={{ color: "var(--text)" }}>{chunks.length}</b> chunks (sentence, ~{size} words). First: <span style={{ color: "var(--text)" }}>&quot;{chunks[0]?.slice(0, 60)}…&quot;</span></div>, BLUE)}
        {stage(2, "Embed", <div style={{ fontSize: 12, color: "var(--muted)" }}>Each chunk → {dim}D vector.<div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: ACC, marginTop: 6 }}>{fmt(chunkVecs[0] || [])}</div></div>, ACC)}
        {stage(3, "Index", <div style={{ fontSize: 12, color: "var(--muted)" }}>{chunks.length} normalized vectors stored (cosine · HNSW). Ready for search.</div>, GREEN)}
        {stage(4, "Retrieve", <div style={{ fontSize: 12, color: "var(--muted)" }}>Query embedded, ranked by cosine. Top score: <b style={{ color: "var(--text)", fontFamily: "var(--mono)" }}>{ranked[0]?.score.toFixed(3)}</b></div>, AMBER)}
      </div>
      <Section title="Retrieved context" right={<Pill color={GREEN}>top {topK}</Pill>}>
        {picked.map((p, k) => (
          <div key={k} style={{ border: `1px solid ${GREEN}`, borderRadius: 10, padding: "9px 12px", marginBottom: 8, background: `color-mix(in srgb, ${GREEN} 6%, transparent)` }}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}><Pill color={GREEN}>[{k + 1}] chunk {p.i + 1}</Pill><b style={{ fontFamily: "var(--mono)", fontSize: 12 }}>{p.score.toFixed(3)}</b></div>
            <div style={{ fontSize: 12.5 }}>{p.text}</div>
          </div>
        ))}
      </Section>
      <Section title="5. Grounded answer" right={<Pill color={ACC}>extractive</Pill>}>
        <div style={{ fontSize: 13.5, lineHeight: 1.6 }}>{answer || "No relevant context found."} {picked.map((_, k) => <sup key={k} style={{ color: GREEN, fontWeight: 700 }}>[{k + 1}]</sup>)}</div>
        <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 10 }}>This demo composes the answer from the retrieved chunks (extractive) so it&apos;s always grounded with citations. In the full RAG Lab, a language model writes the answer from this same context.</div>
      </Section>
    </div>
  );
}

// shared input styles
const lbl: React.CSSProperties = { display: "block", fontSize: 11, color: "var(--muted)", marginBottom: 5, fontWeight: 600 };
const inp: React.CSSProperties = { width: "100%", boxSizing: "border-box", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px", color: "var(--text)", fontSize: 13, fontFamily: "inherit" };
const actBtn: React.CSSProperties = { display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--panel-2)", color: "var(--text)", fontSize: 12.5, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" };
const th: React.CSSProperties = { textAlign: "left", padding: "7px 8px", borderBottom: "1px solid var(--border)", color: "var(--faint)", fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".04em" };
const td: React.CSSProperties = { padding: "7px 8px", borderBottom: "1px solid var(--border)" };

// ═══════════════════════════════════════════════════════════════════════════
// Shell
// ═══════════════════════════════════════════════════════════════════════════
export default function RagPracticePlayground() {
  const [mod, setMod] = useState<ModuleId>("embed");
  const [store, setStore] = useState<{ v: number[]; label: string }[]>([]);
  const onStore = (v: number[], label: string) => { setStore((s) => [...s, { v, label }]); toast(`Added to vector store (${store.length + 1})`, "success"); };
  const active = MODULES.find((m) => m.id === mod)!;
  return (
    <div>
      {/* header */}
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
        <div>
          <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 3 }}><span style={{ fontSize: 16 }}>🧪</span><b style={{ fontSize: 15, color: ACC }}>Practice Playground</b></div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>Experiment with each step of the RAG pipeline on your own text.</div>
        </div>
        {store.length > 0 && <div style={{ ...panel, padding: "8px 12px", fontSize: 11, color: "var(--muted)" }}>🗄 <b style={{ color: "var(--text)" }}>{store.length}</b> vector{store.length > 1 ? "s" : ""} in demo store</div>}
      </div>

      {/* horizontal module stepper (image-2 style, full width) */}
      <div className="stepper">
        {MODULES.map((m, i) => (
          <button key={m.id} className={mod === m.id ? "on" : ""} onClick={() => setMod(m.id)}><b>{i + 1}</b>{m.short}</button>
        ))}
      </div>

      {/* active module — full width */}
      <div>
        <div className="row" style={{ gap: 10, alignItems: "center", marginBottom: 12 }}><span style={{ fontSize: 22 }}>{active.icon}</span><div><h2 style={{ margin: 0, fontSize: 19 }}>{active.title} Playground</h2><div style={{ fontSize: 12.5, color: "var(--muted)" }}>{active.sub}</div></div></div>
        {mod === "chunk" && <ChunkModule />}
        {mod === "embed" && <EmbeddingModule onStore={onStore} />}
        {mod === "pool" && <PoolingModule />}
        {mod === "index" && <IndexingModule store={store} />}
        {mod === "retrieve" && <RetrieveModule />}
        {mod === "backprop" && <BackpropModule />}
        {mod === "e2e" && <EndToEndModule />}
      </div>
    </div>
  );
}
