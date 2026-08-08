"use client";
import { useMemo, useState } from "react";
import { pca2, type Metric } from "@/lib/ragUtils";
import Katex from "@/components/Katex";

// Interactive embedding explorer for the RAG Lab. Click any TWO chunks in the
// PCA map and the panel computes cosine similarity / Euclidean distance / dot
// product between them — with the formula, the real numbers, a to-scale
// mini-diagram, and a plain-English interpretation.
//
// HONESTY: the map is a 2-D PCA projection of the high-dimensional vectors, so
// arrows/lines drawn on it are illustrative. Every NUMBER (dot, norms, angle,
// distance) is computed on the FULL vectors — never the 2-D coordinates.

const PALETTE = ["#5b7cff", "#22b8cf", "#3ecf7f", "#f59e0b", "#e0559f", "#a855f7", "#f0616d", "#eab308"];

type Chunk = { text: string; docName: string };
type Tab = Metric; // "cosine" | "euclidean" | "dot"

const dot = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a: number[]) => Math.sqrt(dot(a, a));
const euclid = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); };
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const fmtVec = (v: number[], n = 4) => "[ " + v.slice(0, n).map((x) => x.toFixed(3)).join(", ") + ", … ]";

export default function EmbeddingExplorer({ vectors, chunks, dim, metric, clusterAssign }: {
  vectors: number[][];
  chunks: Chunk[];
  dim: number;
  metric: Metric;
  clusterAssign?: number[];
}) {
  // Pre-select the first two chunks so the compare panel is populated on load.
  const [sel, setSel] = useState<number[]>(vectors.length >= 2 ? [0, 1] : []);
  const [tab, setTab] = useState<Tab>(metric);

  const pts = useMemo(() => pca2(vectors), [vectors]);
  const asg = clusterAssign ?? vectors.map(() => 0);

  // ── scatter geometry ──
  const W = 560, H = 440, M = { l: 46, r: 16, t: 16, b: 40 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const pad = (arr: number[]) => { const lo = Math.min(...arr), hi = Math.max(...arr); const d = (hi - lo) * 0.08 || 1; return [lo - d, hi + d] as const; };
  const [xmin, xmax] = pad(xs), [ymin, ymax] = pad(ys);
  const sx = (x: number) => M.l + ((x - xmin) / (xmax - xmin || 1)) * pw;
  const sy = (y: number) => M.t + (1 - (y - ymin) / (ymax - ymin || 1)) * ph;
  const ticks = (lo: number, hi: number) => { const out: number[] = []; const step = (hi - lo) / 5; for (let i = 0; i <= 5; i++) out.push(lo + step * i); return out; };

  const pick = (i: number) => setSel((s) => s.includes(i) ? s.filter((x) => x !== i) : s.length < 2 ? [...s, i] : [s[1], i]);

  const A = sel[0], B = sel[1];
  const both = sel.length === 2;
  const vA = both ? vectors[A] : null, vB = both ? vectors[B] : null;

  // ── the real metrics (computed on the FULL vectors) ──
  const calc = useMemo(() => {
    if (!vA || !vB) return null;
    const dp = dot(vA, vB), nA = norm(vA), nB = norm(vB);
    const cos = clamp(dp / ((nA * nB) || 1), -1, 1);
    const theta = Math.acos(cos) * 180 / Math.PI;
    const dist = euclid(vA, vB);
    // reference: max distance from A to any other point → a relative sense of "close/far"
    let maxFromA = 0; for (let i = 0; i < vectors.length; i++) if (i !== A) maxFromA = Math.max(maxFromA, euclid(vA, vectors[i]));
    return { dp, nA, nB, cos, theta, dist, ratio: dist / (maxFromA || 1) };
  }, [vA, vB, vectors, A]);

  const stat = (v: string, k: string) => (
    <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: "10px 14px", minWidth: 0 }}>
      <div style={{ fontFamily: "var(--mono)", fontSize: 19, fontWeight: 600, letterSpacing: "-.02em", lineHeight: 1.1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v}</div>
      <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--faint)", marginTop: 3 }}>{k}</div>
    </div>
  );

  const betterBadge = (better: "higher" | "lower") => (
    <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 9px", borderRadius: 20, color: "var(--good)", background: "color-mix(in srgb, var(--good) 14%, transparent)", border: "1px solid color-mix(in srgb, var(--good) 40%, transparent)" }}>{better === "higher" ? "Higher is better" : "Lower is better"}</span>
  );

  // ── mini diagrams (drawn to scale from the REAL values) ──
  const gauge = (theta: number) => {
    const R = 46, cx = 60, cy = 56; const a = Math.PI * (1 - theta / 180); // 0°→right, 180°→left
    const nx = cx + R * Math.cos(a), ny = cy - R * Math.sin(a);
    return (
      <svg viewBox="0 0 120 72" width="120" height="72">
        <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`} fill="none" stroke="var(--border-strong)" strokeWidth={3} />
        <text x={cx - R} y={cy + 12} fontSize="8" fill="var(--faint)" textAnchor="middle">0°</text>
        <text x={cx} y={8} fontSize="8" fill="var(--faint)" textAnchor="middle">90°</text>
        <text x={cx + R} y={cy + 12} fontSize="8" fill="var(--faint)" textAnchor="middle">180°</text>
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#f59e0b" strokeWidth={2.5} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={3} fill="#f59e0b" />
      </svg>
    );
  };
  const arrowsDiagram = (theta: number) => {
    const ox = 14, oy = 60, L = 60; const aRad = theta * Math.PI / 180;
    const ax = ox + L, ay = oy;                         // A along the x-axis
    const bx = ox + L * Math.cos(aRad), by = oy - L * Math.sin(aRad); // B at angle θ
    return (
      <svg viewBox="0 0 150 72" width="150" height="72">
        <defs>
          <marker id="ee-a" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#5b7cff" /></marker>
          <marker id="ee-b" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e0559f" /></marker>
        </defs>
        <line x1={ox} y1={oy} x2={ax} y2={ay} stroke="#5b7cff" strokeWidth={2} markerEnd="url(#ee-a)" />
        <line x1={ox} y1={oy} x2={bx} y2={by} stroke="#e0559f" strokeWidth={2} markerEnd="url(#ee-b)" />
        <text x={ax + 3} y={ay + 3} fontSize="10" fill="#5b7cff" fontStyle="italic">A</text>
        <text x={bx + 2} y={by - 2} fontSize="10" fill="#e0559f" fontStyle="italic">B</text>
        <text x={ox + 24} y={oy - 6} fontSize="9" fill="#f59e0b">θ</text>
      </svg>
    );
  };
  const distanceDiagram = (d: number) => (
    <svg viewBox="0 0 170 44" width="170" height="44">
      <line x1={26} y1={22} x2={144} y2={22} stroke="var(--muted)" strokeWidth={1.5} strokeDasharray="4 3" />
      <circle cx={26} cy={22} r={9} fill="#e0559f" /><text x={26} y={25} fontSize="8" fill="#fff" textAnchor="middle">A</text>
      <circle cx={144} cy={22} r={9} fill="#5b7cff" /><text x={144} y={25} fontSize="8" fill="#fff" textAnchor="middle">B</text>
      <text x={85} y={16} fontSize="10" fill="var(--text)" textAnchor="middle" fontFamily="var(--mono)">{d.toFixed(3)}</text>
    </svg>
  );

  // ── the map overlay when two points are chosen ──
  const overlay = () => {
    if (!both || !calc) return null;
    const Ax = sx(pts[A].x), Ay = sy(pts[A].y), Bx = sx(pts[B].x), By = sy(pts[B].y);
    if (tab === "euclidean") {
      const mx = (Ax + Bx) / 2, my = (Ay + By) / 2;
      return (<>
        <line x1={Ax} y1={Ay} x2={Bx} y2={By} stroke="#f59e0b" strokeWidth={1.6} strokeDasharray="5 4" />
        <text x={mx + 6} y={my - 4} fontSize="11" fill="#f59e0b" fontFamily="var(--mono)">d = {calc.dist.toFixed(3)}</text>
      </>);
    }
    // cosine / dot → arrows from the projected origin to A and B
    const Ox = sx(clamp(0, xmin, xmax)), Oy = sy(clamp(0, ymin, ymax));
    return (<>
      <line x1={Ox} y1={Oy} x2={Ax} y2={Ay} stroke="#5b7cff" strokeWidth={1.8} markerEnd="url(#ee-mA)" />
      <line x1={Ox} y1={Oy} x2={Bx} y2={By} stroke="#e0559f" strokeWidth={1.8} markerEnd="url(#ee-mB)" />
      <text x={(Ox + Ax) / 2} y={(Oy + Ay) / 2 + 12} fontSize="10" fill="#5b7cff" fontStyle="italic">A</text>
      <text x={(Ox + Bx) / 2} y={(Oy + By) / 2 - 6} fontSize="10" fill="#e0559f" fontStyle="italic">B</text>
      <text x={Ox + 10} y={Oy - 6} fontSize="11" fill="#f59e0b" fontFamily="var(--mono)">θ = {calc.theta.toFixed(1)}°</text>
    </>);
  };

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="card-h"><span className="t">🔬 Embedding explorer</span><span className="mono r">click two chunks to compare</span></div>
      <div className="card-b">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 14 }}>
          {stat(String(chunks.length), "chunks")}
          {stat(String(dim), "dimensions")}
          {stat(String(vectors.length), "vectors stored")}
          {stat(metric === "dot" ? "dot (IP)" : metric === "euclidean" ? "euclidean" : "cosine", "similarity")}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.15fr) minmax(0,1fr)", gap: 14, alignItems: "start" }}>
          {/* ── PCA map ── */}
          <div style={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel-2)", padding: 10 }}>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 600 }}>Embedding space <span className="note" style={{ fontWeight: 400 }}>(PCA → 2-D)</span></span>
              {sel.length > 0 && <button className="btn ghost sm" onClick={() => setSel([])}>Clear selection</button>}
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", cursor: "crosshair", userSelect: "none" }}>
              <defs>
                <marker id="ee-mA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#5b7cff" /></marker>
                <marker id="ee-mB" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e0559f" /></marker>
              </defs>
              {/* gridlines + ticks */}
              {ticks(xmin, xmax).map((tv, i) => { const X = sx(tv); return <g key={`x${i}`}><line x1={X} y1={M.t} x2={X} y2={M.t + ph} stroke="var(--border)" strokeWidth={0.5} opacity={0.5} /><text x={X} y={H - 22} fontSize="8.5" fill="var(--faint)" textAnchor="middle">{tv.toFixed(1)}</text></g>; })}
              {ticks(ymin, ymax).map((tv, i) => { const Y = sy(tv); return <g key={`y${i}`}><line x1={M.l} y1={Y} x2={M.l + pw} y2={Y} stroke="var(--border)" strokeWidth={0.5} opacity={0.5} /><text x={M.l - 6} y={Y + 3} fontSize="8.5" fill="var(--faint)" textAnchor="end">{tv.toFixed(1)}</text></g>; })}
              <text x={M.l + pw / 2} y={H - 6} fontSize="9.5" fill="var(--muted)" textAnchor="middle">PCA Component 1</text>
              <text x={12} y={M.t + ph / 2} fontSize="9.5" fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 12 ${M.t + ph / 2})`}>PCA Component 2</text>
              {overlay()}
              {/* points */}
              {pts.map((p, i) => {
                const X = sx(p.x), Y = sy(p.y); const on = sel.includes(i); const color = PALETTE[asg[i] % PALETTE.length];
                return (
                  <g key={i} onClick={() => pick(i)} style={{ cursor: "pointer" }}>
                    {on && <circle cx={X} cy={Y} r={11} fill="none" stroke={i === A ? "#5b7cff" : "#e0559f"} strokeWidth={2.5} />}
                    <circle cx={X} cy={Y} r={on ? 7 : 6} fill={color} opacity={on ? 1 : 0.82} />
                    <text x={X} y={Y - 10} fontSize="8.5" fill="var(--muted)" textAnchor="middle">{i + 1}</text>
                  </g>
                );
              })}
            </svg>
            {/* legend */}
            <div className="row" style={{ gap: 12, flexWrap: "wrap", justifyContent: "center", marginTop: 6 }}>
              {Array.from(new Set(asg)).sort((a, b) => a - b).map((c) => <span key={c} className="row" style={{ gap: 5, alignItems: "center", fontSize: 10, color: "var(--muted)" }}><span style={{ width: 9, height: 9, borderRadius: "50%", background: PALETTE[c % PALETTE.length] }} />cluster {c + 1}</span>)}
            </div>
            <div className="note" style={{ marginTop: 6, fontSize: 10, textAlign: "center", lineHeight: 1.4 }}>PCA is a 2-D projection of the {dim}-D vectors for display only — the metrics on the right are computed on the full vectors.</div>
          </div>

          {/* ── compare panel ── */}
          <div>
            {!both ? (
              <div style={{ border: "1px dashed var(--border-strong)", borderRadius: 12, padding: "30px 18px", textAlign: "center", background: "var(--panel)" }}>
                <div style={{ fontSize: 24, marginBottom: 8, opacity: 0.6 }}>◎ ◎</div>
                <div style={{ fontWeight: 600, fontSize: 13 }}>Select two chunks</div>
                <div className="note" style={{ marginTop: 6, lineHeight: 1.5 }}>Click any two points in the map. You&apos;ll see the cosine similarity, Euclidean distance and dot product between them — with the formula and a worked calculation.</div>
                {sel.length === 1 && <div className="note" style={{ marginTop: 8, color: "var(--accent)" }}>Chunk {sel[0] + 1} selected — pick one more.</div>}
              </div>
            ) : calc && vA && vB && (
              <div style={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)", overflow: "hidden" }}>
                {/* selected header */}
                <div className="row" style={{ justifyContent: "space-between", alignItems: "center", padding: "10px 14px", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--muted)" }}>Selected chunks</span>
                  <span style={{ fontFamily: "var(--mono)", fontWeight: 700 }}><span style={{ color: "#5b7cff" }}>{A + 1}</span> <span style={{ color: "var(--faint)" }}>&amp;</span> <span style={{ color: "#e0559f" }}>{B + 1}</span></span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, padding: "10px 14px", borderBottom: "1px solid var(--border)" }}>
                  {[[A, "#5b7cff", vA], [B, "#e0559f", vB]].map(([idx, col, v]) => (
                    <div key={idx as number} style={{ border: "1px solid var(--border)", borderRadius: 9, padding: "8px 10px", background: "var(--surface)" }}>
                      <div className="row" style={{ gap: 6, alignItems: "center" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: col as string }} /><b style={{ fontSize: 12 }}>Chunk {(idx as number) + 1}</b></div>
                      <div className="note" style={{ fontSize: 9.5, marginTop: 3 }}>PCA: ({pts[idx as number].x.toFixed(2)}, {pts[idx as number].y.toFixed(2)}) · cluster {asg[idx as number] + 1}</div>
                      <div className="note" style={{ fontFamily: "var(--mono)", fontSize: 9.5, marginTop: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={(v as number[]).slice(0, 6).map((x) => x.toFixed(3)).join(", ")}>{fmtVec(v as number[], 4)}</div>
                    </div>
                  ))}
                </div>

                {/* tabs */}
                <div className="row" style={{ borderBottom: "1px solid var(--border)" }}>
                  {([["cosine", "Cosine"], ["euclidean", "Euclidean"], ["dot", "Dot (IP)"]] as [Tab, string][]).map(([k, l]) => (
                    <button key={k} onClick={() => setTab(k)} style={{ flex: 1, padding: "9px 6px", border: "none", background: "transparent", borderBottom: `2px solid ${tab === k ? "var(--accent)" : "transparent"}`, color: tab === k ? "var(--accent)" : "var(--muted)", fontWeight: tab === k ? 600 : 400, fontSize: 11.5, cursor: "pointer" }}>{l}</button>
                  ))}
                </div>

                <div style={{ padding: 14 }}>
                  {tab === "cosine" && (<>
                    <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}><b style={{ fontSize: 13 }}>Cosine similarity</b>{betterBadge("higher")}</div>
                    <div className="note" style={{ marginBottom: 10 }}>Cosine of the angle between the two vectors.</div>
                    <div style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "10px 12px", marginBottom: 10, overflowX: "auto" }}><Katex block tex={"\\cos(\\theta)=\\dfrac{A\\cdot B}{\\lVert A\\rVert\\,\\lVert B\\rVert}"} /></div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)", lineHeight: 1.7 }}>
                      <div>A·B = {calc.dp.toFixed(4)}</div>
                      <div>‖A‖ = {calc.nA.toFixed(4)} · ‖B‖ = {calc.nB.toFixed(4)}</div>
                    </div>
                    <div style={{ background: "color-mix(in srgb, var(--good) 8%, transparent)", border: "1px solid color-mix(in srgb, var(--good) 35%, transparent)", borderRadius: 10, padding: 12, marginTop: 10, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                      <div><div style={{ color: "var(--good)", fontWeight: 700, fontSize: 15 }}>Cosine = {calc.cos.toFixed(4)}</div><div className="note" style={{ marginTop: 2 }}>angle θ = {calc.theta.toFixed(2)}°</div></div>
                      {gauge(calc.theta)}
                    </div>
                    <div className="note" style={{ marginTop: 10, lineHeight: 1.5 }}>{calc.cos >= 0.8 ? "The vectors point in very similar directions — these chunks are semantically very similar." : calc.cos >= 0.5 ? "The vectors point in fairly similar directions — the chunks are related." : calc.cos >= 0.2 ? "Weak similarity — the chunks share a little meaning." : calc.cos >= -0.2 ? "Nearly orthogonal — little semantic relationship." : "Opposite directions — semantically unrelated."}</div>
                  </>)}

                  {tab === "euclidean" && (<>
                    <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}><b style={{ fontSize: 13 }}>Euclidean distance</b>{betterBadge("lower")}</div>
                    <div className="note" style={{ marginBottom: 10 }}>Straight-line distance between the two vectors.</div>
                    <div style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "10px 12px", marginBottom: 10, overflowX: "auto" }}><Katex block tex={"d(A,B)=\\sqrt{\\sum_{i=1}^{n}(A_i-B_i)^2}"} /></div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)", lineHeight: 1.7 }}>
                      <div>d = √Σ<sub>i=1..{dim}</sub> (A<sub>i</sub> − B<sub>i</sub>)²</div>
                      <div>d = {calc.dist.toFixed(4)}</div>
                    </div>
                    <div style={{ background: "color-mix(in srgb, var(--good) 8%, transparent)", border: "1px solid color-mix(in srgb, var(--good) 35%, transparent)", borderRadius: 10, padding: 12, marginTop: 10, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                      <div><div style={{ color: "var(--good)", fontWeight: 700, fontSize: 15 }}>Distance = {calc.dist.toFixed(4)}</div><div className="note" style={{ marginTop: 2 }}>straight-line distance</div></div>
                      {distanceDiagram(calc.dist)}
                    </div>
                    <div className="note" style={{ marginTop: 10, lineHeight: 1.5 }}>{calc.ratio < 0.35 ? "The distance is small — these embeddings are close together in the vector space." : calc.ratio < 0.7 ? "The distance is moderate — the chunks are a medium distance apart." : "The distance is large — these embeddings are far apart."}</div>
                  </>)}

                  {tab === "dot" && (<>
                    <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}><b style={{ fontSize: 13 }}>Dot product (inner product)</b>{betterBadge("higher")}</div>
                    <div className="note" style={{ marginBottom: 10 }}>Similarity using both direction and magnitude.</div>
                    <div style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "10px 12px", marginBottom: 10, overflowX: "auto" }}><Katex block tex={"A\\cdot B=\\sum_{i=1}^{n}A_i B_i"} /></div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)", lineHeight: 1.7 }}>
                      <div>A·B = {calc.dp.toFixed(4)}</div>
                      <div>‖A‖ = {calc.nA.toFixed(4)} · ‖B‖ = {calc.nB.toFixed(4)}</div>
                      <div>θ = {calc.theta.toFixed(2)}° (from cosine)</div>
                    </div>
                    <div style={{ background: "color-mix(in srgb, var(--good) 8%, transparent)", border: "1px solid color-mix(in srgb, var(--good) 35%, transparent)", borderRadius: 10, padding: 12, marginTop: 10, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                      <div><div style={{ color: "var(--good)", fontWeight: 700, fontSize: 15 }}>A·B = {calc.dp.toFixed(4)}</div><div className="note" style={{ marginTop: 2 }}>{calc.dp > 0 ? "vectors point in similar directions" : calc.dp < 0 ? "vectors point in opposing directions" : "vectors are ~orthogonal"}</div></div>
                      {arrowsDiagram(calc.theta)}
                    </div>
                    <div className="note" style={{ marginTop: 10, lineHeight: 1.5 }}>{calc.dp > 0 ? "Positive dot product means the vectors point in similar directions — larger magnitude means stronger similarity." : calc.dp < 0 ? "Negative dot product means the vectors point in opposing directions." : "A near-zero dot product means the vectors are close to orthogonal (unrelated)."}</div>
                  </>)}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
