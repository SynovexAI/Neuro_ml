"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import Plot from "@/components/Plot";
import { plotlyTheme } from "@/lib/edaCharts";
import { pca2, cosine, simDense } from "@/lib/ragUtils";

type Module = "chunk" | "embed" | "pool" | "index" | "retrieve" | "backprop" | "end2end";

// Helper for generating deterministic fake embeddings for visual purposes
function fakeEmbed(text: string, dim: number): number[] {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = Math.imul(31, h) + text.charCodeAt(i) | 0;
  const v = [];
  for (let i = 0; i < dim; i++) {
    const val = (Math.sin(h * 13.37 + i * 2.14) + 1) / 2;
    v.push(val * 0.9 + 0.05); // range [0.05, 0.95]
  }
  return v;
}

// Pooling operation
function meanPool(vecs: number[][]): number[] {
  if (!vecs.length) return [];
  const dim = vecs[0].length;
  const res = new Array(dim).fill(0);
  for (let i = 0; i < vecs.length; i++) {
    for (let j = 0; j < dim; j++) res[j] += vecs[i][j];
  }
  return res.map(v => v / vecs.length);
}

function maxPool(vecs: number[][]): number[] {
  if (!vecs.length) return [];
  const dim = vecs[0].length;
  const res = new Array(dim).fill(-Infinity);
  for (let i = 0; i < vecs.length; i++) {
    for (let j = 0; j < dim; j++) res[j] = Math.max(res[j], vecs[i][j]);
  }
  return res;
}

export default function RagPlayground() {
  const [mod, setMod] = useState<Module>("pool");

  // Common styles to match standard lab UI
  const kgHead = (dot: string, title: string, right?: React.ReactNode) => (
    <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
      <div className="row" style={{ gap: 8, alignItems: "center" }}>
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} />
        <span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--muted)" }}>{title}</span>
      </div>
      {right}
    </div>
  );
  const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 14, background: "var(--panel)", overflow: "hidden" };
  const t = plotlyTheme();
  const pLayout = (title: string, extra: Record<string, unknown> = {}) => ({ title: { text: title, font: { size: 13, color: t.text } }, paper_bgcolor: t.paper, plot_bgcolor: t.plot, font: { color: t.muted, size: 11 }, margin: { l: 40, r: 16, t: 40, b: 40 }, xaxis: { gridcolor: t.grid, zerolinecolor: t.grid }, yaxis: { gridcolor: t.grid, zerolinecolor: t.grid }, colorway: t.colorway, ...extra });

  // --- POOLING PLAYGROUND STATE ---
  const [poolText, setPoolText] = useState("all i need");
  const [poolMethod, setPoolMethod] = useState<"mean" | "max" | "cls">("max");
  const tokens = poolText.split(/\s+/).filter(Boolean);
  const poolDims = 8;
  const tokenVecs = useMemo(() => tokens.map(tk => fakeEmbed(tk, poolDims)), [tokens, poolDims]);
  const pooledVec = useMemo(() => {
    if (poolMethod === "mean") return meanPool(tokenVecs);
    if (poolMethod === "max") return maxPool(tokenVecs);
    return tokenVecs[0] || new Array(poolDims).fill(0); // cls
  }, [tokenVecs, poolMethod]);

  // --- EMBEDDING PLAYGROUND STATE ---
  const [embText, setEmbText] = useState("all i need");

  // --- INDEXING PLAYGROUND STATE ---
  const [idxText, setIdxText] = useState("Machine learning is a subset of artificial intelligence.");
  const [idxVectors, setIdxVectors] = useState<{id: string, chunk: string, vec: number[], norm: number, time: string}[]>([
    { id: "vec_001", chunk: "Artificial intelligence is the simulation...", vec: fakeEmbed("Artificial", poolDims), norm: 1.98, time: "2m ago" },
    { id: "vec_002", chunk: "Deep learning uses neural networks...", vec: fakeEmbed("Deep", poolDims), norm: 2.15, time: "3m ago" },
    { id: "vec_003", chunk: idxText, vec: fakeEmbed(idxText, poolDims), norm: 2.01, time: "Just now" },
    { id: "vec_004", chunk: "Transformers are used in NLP...", vec: fakeEmbed("Transformers", poolDims), norm: 1.92, time: "5m ago" },
    { id: "vec_005", chunk: "Reinforcement learning learns by...", vec: fakeEmbed("Reinforcement", poolDims), norm: 2.08, time: "6m ago" }
  ]);
  const [idxQuery, setIdxQuery] = useState("What is machine learning?");

  // --- BACKPROP PLAYGROUND STATE ---
  const [bpStep, setBpStep] = useState(1);
  const [bpEpoch, setBpEpoch] = useState(3);
  const [bpRunning, setBpRunning] = useState(false);
  const [bpLoss, setBpLoss] = useState(0.42);

  const renderStepper = () => {
    const items: { id: Module; ic: string; t: string; s: string }[] = [
      { id: "chunk", ic: "🧩", t: "Chunk", s: "Split text into chunks" },
      { id: "embed", ic: "🧬", t: "Embedding", s: "Turn chunks into vectors" },
      { id: "pool", ic: "🔮", t: "Pooling", s: "Combine token vectors" },
      { id: "index", ic: "🗄️", t: "Indexing", s: "Store vectors" },
      { id: "retrieve", ic: "🔍", t: "Retrieve", s: "Search vectors" },
      { id: "backprop", ic: "⚙️", t: "Backpropagation", s: "Simulator" },
      { id: "end2end", ic: "🚀", t: "End-to-End", s: "Full pipeline" },
    ];
    return (
      <div className="stepper" style={{ borderBottom: "1px solid var(--border)", padding: "8px 16px", overflowX: "auto", flexShrink: 0 }}>
        {items.map((m, i) => (
          <button 
            key={m.id} 
            className={`step ${mod === m.id ? "on" : ""}`} 
            onClick={() => setMod(m.id)}
            style={{ flexShrink: 0, minWidth: 'auto', padding: '12px 16px' }}
          >
            <span className="num" style={{ width: 24, height: 24, fontSize: 12 }}>{m.ic}</span>
            <span className="lbl">{m.t}</span>
          </button>
        ))}
      </div>
    );
  };

  const renderPooling = () => {
    return (
      <div style={{ padding: 16, flex: 1, overflowY: "auto" }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div>
            <h2 style={{ fontSize: 16, margin: "0 0 4px" }}>Pooling Playground</h2>
            <div className="note">See how different pooling strategies combine token vectors into a single chunk embedding.</div>
          </div>
          <button className="btn ghost sm">↻ Reset</button>
        </div>

        <div className="row" style={{ gap: 20, marginBottom: 24 }}>
          <div style={{ ...pnl, padding: "12px 16px", flex: 1 }}>
            <label className="note" style={{ display: "block", marginBottom: 6 }}>Input text</label>
            <input type="text" value={poolText} onChange={e => setPoolText(e.target.value)} style={{ width: "100%", background: "transparent", border: "none", fontSize: 15, outline: "none", color: "#3ecf7f", fontWeight: 600 }} />
          </div>
          <div style={{ ...pnl, padding: "12px 16px", width: 120 }}>
            <label className="note" style={{ display: "block", marginBottom: 6 }}>Tokens</label>
            <div style={{ fontSize: 15, fontWeight: 600 }}>{tokens.length}</div>
          </div>
          <div style={{ ...pnl, padding: "12px 16px", width: 120 }}>
            <label className="note" style={{ display: "block", marginBottom: 6 }}>Dimensions</label>
            <div style={{ fontSize: 15, fontWeight: 600 }}>{poolDims}D</div>
          </div>
          <div style={{ ...pnl, padding: "12px 16px", flex: 1 }}>
            <label className="note" style={{ display: "block", marginBottom: 6 }}>Select pooling method</label>
            <div className="chips">
              <button className={`chip ${poolMethod === "mean" ? "on" : ""}`} onClick={() => setPoolMethod("mean")}>≡ Mean Pooling</button>
              <button className={`chip ${poolMethod === "max" ? "on" : ""}`} onClick={() => setPoolMethod("max")}>📈 Max Pooling</button>
              <button className={`chip ${poolMethod === "cls" ? "on" : ""}`} onClick={() => setPoolMethod("cls")}>& CLS Pooling</button>
            </div>
          </div>
        </div>

        <div className="split col-2e" style={{ gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ ...pnl }}>
              {kgHead("#a855f7", "1. Tokens and their embeddings", <span className="note" style={{ fontSize: 10 }}>Click a token to highlight its vector</span>)}
              <div style={{ padding: 16 }}>
                <div className="chips" style={{ marginBottom: 16 }}>
                  {tokens.map((tk, i) => (
                    <div key={i} className="chip on" style={{ background: "var(--panel-2)", borderColor: "var(--border)", pointerEvents: "none" }}>
                      <span style={{ color: "#a855f7" }}>t{i+1}</span> • {tk}
                    </div>
                  ))}
                </div>
                <div className="row" style={{ gap: 16, overflowX: "auto", paddingBottom: 8 }}>
                  {tokens.map((tk, i) => (
                    <div key={i} style={{ flex: 1, minWidth: 160, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: 12 }}>
                      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12, fontSize: 12, fontWeight: 600 }}>
                        <span><span style={{ color: "#a855f7" }}>t{i+1}</span> • {tk}</span>
                        <span className="note">{poolDims} dims</span>
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        {tokenVecs[i]?.map((v, d) => (
                          <div key={d} className="row" style={{ gap: 8, alignItems: "center", fontSize: 10, fontFamily: "var(--mono)" }}>
                            <span style={{ width: 14, color: "var(--faint)" }}>d{d+1}</span>
                            <div style={{ flex: 1, height: 6, background: "var(--panel)", borderRadius: 3, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${v * 100}%`, background: "#a855f7", borderRadius: 3 }} />
                            </div>
                            <span style={{ width: 28, textAlign: "right", color: "var(--muted)" }}>{v.toFixed(2)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ ...pnl }}>
              {kgHead("var(--accent)", "2. Pooling operation", <span className="note" style={{ fontSize: 10 }}>Combines multiple token vectors into one</span>)}
              <div style={{ padding: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                  <span style={{ background: "var(--accent)", color: "#fff", padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 700, textTransform: "uppercase" }}>{poolMethod} POOLING</span>
                  <span style={{ fontSize: 12, color: "var(--muted)" }}>
                    {poolMethod === "mean" ? "Takes the average value for each dimension across all token vectors." : 
                     poolMethod === "max" ? "Takes the maximum value for each dimension across all token vectors." : 
                     "Uses the first token's vector (often [CLS]) to represent the entire chunk."}
                  </span>
                </div>
                
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", fontSize: 11, fontFamily: "var(--mono)", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: "left", padding: "6px", color: "var(--faint)", fontWeight: 500, width: 80 }}>Dimension</th>
                        {Array.from({ length: poolDims }).map((_, i) => <th key={i} style={{ padding: "6px", color: "var(--faint)", fontWeight: 500 }}>d{i+1}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {tokens.map((tk, i) => (
                        <tr key={i}>
                          <td style={{ padding: "4px 6px", color: "var(--muted)" }}><span style={{ color: "#a855f7" }}>t{i+1}</span> • {tk}</td>
                          {tokenVecs[i]?.map((v, d) => (
                            <td key={d} style={{ padding: "4px 6px", textAlign: "center", color: (poolMethod === "max" && pooledVec[d] === v) || (poolMethod === "cls" && i === 0) ? "var(--accent)" : "var(--faint)" }}>
                              {v.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                      <tr><td colSpan={poolDims + 1} style={{ padding: 6 }}></td></tr>
                      <tr style={{ background: "color-mix(in srgb, var(--accent) 10%, transparent)", fontWeight: 600 }}>
                        <td style={{ padding: "8px 6px", color: "var(--accent)", textTransform: "uppercase" }}>{poolMethod} (pooled)</td>
                        {pooledVec.map((v, d) => (
                          <td key={d} style={{ padding: "8px 6px", textAlign: "center", color: "var(--accent)" }}>{v.toFixed(2)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div style={{ ...pnl }}>
              {kgHead("#3ecf7f", "3. Final pooled vector (chunk embedding)", <span className="note" style={{ fontSize: 10 }}>Vector norm (L2)</span>)}
              <div style={{ padding: 16 }} className="row">
                <div style={{ flex: 1 }}>
                   <div style={{ fontFamily: "var(--mono)", fontSize: 16, color: "var(--accent)", letterSpacing: 1, padding: "16px 20px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, marginBottom: 12 }}>
                     [ {pooledVec.map(v => v.toFixed(2)).join(", ")} ] <span style={{ float: "right", background: "#3ecf7f", color: "#000", padding: "2px 6px", borderRadius: 4, fontSize: 10, fontWeight: 700 }}>{poolDims}D</span>
                   </div>
                   <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
                     This is the final chunk embedding that represents the input text using {poolMethod === "mean" ? "Mean" : poolMethod === "max" ? "Max" : "CLS"} Pooling.
                     It can now be inserted into the vector store and used during retrieval.
                   </div>
                </div>
                <div style={{ width: 140, marginLeft: 24, display: "flex", flexDirection: "column", gap: 12, alignItems: "center", justifyContent: "center", borderLeft: "1px solid var(--border)", paddingLeft: 24 }}>
                   <div style={{ fontSize: 12, color: "var(--faint)" }}>Vector norm (L2)</div>
                   <div style={{ fontSize: 24, fontWeight: 700, color: "#3ecf7f" }}>
                     {Math.sqrt(pooledVec.reduce((acc, v) => acc + v * v, 0)).toFixed(2)}
                   </div>
                   <button className="btn ghost sm block"><span style={{ fontSize: 14 }}>⎘</span> Copy vector</button>
                </div>
              </div>
            </div>
          </div>
          
          <div style={{ width: 280, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ ...pnl }}>
              {kgHead("var(--faint)", "How it works")}
              <div style={{ padding: 16, fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>
                <ol style={{ paddingLeft: 16, margin: "0 0 16px" }}>
                  <li>Look at each dimension (d1...d{poolDims})</li>
                  <li>Find the {poolMethod === "mean" ? "average" : poolMethod === "max" ? "maximum" : "first"} value among all token vectors</li>
                  <li>Build a new vector using those {poolMethod === "mean" ? "averages" : poolMethod === "max" ? "maximums" : "values"}</li>
                </ol>
                {/* Visualizer diagram placeholder */}
                <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                   <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                     {tokens.map((tk, i) => <div key={i} style={{ padding: "4px 8px", border: "1px solid var(--accent)", borderRadius: 4, fontSize: 11, color: "var(--accent)", textAlign: "center" }}>t{i+1}</div>)}
                   </div>
                   <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, alignItems: "center", color: "var(--border-strong)" }}>
                     <svg width="60" height="100" viewBox="0 0 60 100">
                       <path d="M 0 15 L 60 50" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5"/>
                       <path d="M 0 50 L 60 50" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5"/>
                       <path d="M 0 85 L 60 50" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5"/>
                     </svg>
                   </div>
                   <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <div style={{ fontSize: 9, color: "var(--faint)", textAlign: "center" }}>Pooled</div>
                      {pooledVec.map((v, i) => <div key={i} style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--accent)", textAlign: "right" }}>{v.toFixed(2)} <span style={{ color: "var(--faint)" }}>d{i+1}</span></div>)}
                   </div>
                </div>
              </div>
            </div>

            <div style={{ ...pnl }}>
              {kgHead("var(--faint)", "Token contribution")}
              <div style={{ padding: 16 }}>
                <div className="note" style={{ marginBottom: 12 }}>Approx. contribution to the pooled vector</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {tokens.map((tk, i) => {
                    const contribution = poolMethod === "cls" ? (i === 0 ? 100 : 0) : 
                                         poolMethod === "mean" ? 100 / tokens.length : 
                                         // Mock contribution for max pooling based on how many maxes it provided
                                         (tokenVecs[i].filter((v, d) => v === pooledVec[d]).length / poolDims) * 100;
                    return (
                      <div key={i}>
                        <div className="row" style={{ justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                          <span>{tk}</span>
                          <span className="mono">{Math.round(contribution)}%</span>
                        </div>
                        <div style={{ height: 6, background: "var(--panel)", borderRadius: 3, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${contribution}%`, background: "#a855f7" }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
            
            <div style={{ ...pnl }}>
              {kgHead("var(--faint)", "Compare pooling methods")}
              <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>
                {(["mean", "max", "cls"] as const).map(m => {
                  const vec = m === "mean" ? meanPool(tokenVecs) : m === "max" ? maxPool(tokenVecs) : tokenVecs[0] || [];
                  return (
                    <div key={m}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: poolMethod === m ? "var(--accent)" : "var(--muted)", textTransform: "capitalize", marginBottom: 4 }}>{m} Pooling {m === "cls" && "(t1)"}</div>
                      <div style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--faint)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        [ {vec.map(v => v.toFixed(2)).join(", ")} ]
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderIndexing = () => {
    const currentIdxVectors = idxVectors.map(v => v.id === "vec_003" ? { ...v, chunk: idxText, vec: fakeEmbed(idxText, poolDims) } : v);
    return (
      <div style={{ padding: 16, flex: 1, overflowY: "auto" }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div>
            <h2 style={{ fontSize: 16, margin: "0 0 4px" }}>Indexing Playground</h2>
            <div className="note">See how embeddings are prepared and stored in a vector database for fast similarity search.</div>
          </div>
          <button className="btn ghost sm">↻ Reset</button>
        </div>

        <div className="row" style={{ gap: 20, marginBottom: 24 }}>
          <div style={{ ...pnl, padding: "12px 16px", flex: 1 }}>
            <label className="note" style={{ display: "block", marginBottom: 6 }}>Input chunk</label>
            <input type="text" value={idxText} onChange={e => setIdxText(e.target.value)} style={{ width: "100%", background: "transparent", border: "none", fontSize: 14, outline: "none", color: "var(--text)", padding: "4px 0" }} />
          </div>
          <div style={{ ...pnl, padding: "12px 16px", flex: 1 }}>
            <label className="note" style={{ display: "block", marginBottom: 6 }}>Embedding (8D)</label>
            <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "#3ecf7f", padding: "4px 0" }}>
              [ {fakeEmbed(idxText, poolDims).map(v => v.toFixed(2)).join(", ")} ]
            </div>
          </div>
        </div>
        
        <div className="split col-2e" style={{ gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="row" style={{ gap: 16, alignItems: "stretch" }}>
              <div style={{ ...pnl, flex: 2 }}>
                {kgHead("#3ecf7f", "1. Vector processing", <span className="note" style={{ fontSize: 10 }}>Prepare the embedding before storing</span>)}
                <div style={{ padding: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Normalization</div>
                     <div style={{ width: 40, height: 40, borderRadius: "50%", border: "1px dashed #3ecf7f", margin: "0 auto", display: "grid", placeItems: "center" }}>
                        <span style={{ fontSize: 20, color: "#3ecf7f" }}>✺</span>
                     </div>
                   </div>
                   <div style={{ color: "var(--border-strong)" }}>→</div>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>L2 Norm</div>
                     <div style={{ fontSize: 20, fontWeight: 600, color: "var(--text)" }}>2.01</div>
                     <div style={{ fontSize: 11, color: "#3ecf7f", marginTop: 4 }}>✓ Normalized</div>
                   </div>
                   <div style={{ color: "var(--border-strong)" }}>→</div>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Dimension check</div>
                     <div className="row" style={{ gap: 12, fontSize: 12, justifyContent: "center" }}>
                       <div><div style={{ color: "var(--faint)" }}>Expected</div><div>8D</div></div>
                       <div><div style={{ color: "var(--faint)" }}>Actual</div><div style={{ color: "#3ecf7f" }}>8D ✓</div></div>
                     </div>
                   </div>
                   <div style={{ color: "var(--border-strong)" }}>→</div>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Vector ready</div>
                     <div style={{ fontSize: 24 }}>🛢️</div>
                     <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 4 }}>Ready to store</div>
                   </div>
                </div>
              </div>
              
              <div style={{ ...pnl, flex: 1 }}>
                {kgHead("var(--faint)", "Index settings")}
                <div style={{ padding: 16 }}>
                  <div className="row" style={{ justifyContent: "space-between", marginBottom: 12, fontSize: 12 }}>
                    <span className="note">Index type</span>
                    <select style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", padding: "2px 6px", borderRadius: 4 }}><option>HNSW</option><option>IVF_FLAT</option></select>
                  </div>
                  <div className="row" style={{ justifyContent: "space-between", marginBottom: 12, fontSize: 12 }}>
                    <span className="note">Distance metric</span>
                    <select style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", padding: "2px 6px", borderRadius: 4 }}><option>Cosine</option><option>L2</option><option>IP</option></select>
                  </div>
                  <div className="row" style={{ justifyContent: "space-between", marginBottom: 12, fontSize: 12 }}>
                    <span className="note">Dimensions</span>
                    <span style={{ fontFamily: "var(--mono)" }}>8</span>
                  </div>
                  <button className="btn sm block" style={{ marginTop: 16 }}>Apply settings</button>
                </div>
              </div>
            </div>

            <div style={{ ...pnl }}>
              {kgHead("var(--text)", "3. Vector store", <span className="note" style={{ fontSize: 10 }}>This vector has been indexed and stored</span>)}
              <div style={{ padding: 0, overflowX: "auto" }}>
                <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse", textAlign: "left" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      <th style={{ padding: "12px 16px", color: "var(--faint)", fontWeight: 500 }}>ID</th>
                      <th style={{ padding: "12px 16px", color: "var(--faint)", fontWeight: 500 }}>Chunk (preview)</th>
                      <th style={{ padding: "12px 16px", color: "var(--faint)", fontWeight: 500 }}>Vector (8D)</th>
                      <th style={{ padding: "12px 16px", color: "var(--faint)", fontWeight: 500 }}>Norm</th>
                      <th style={{ padding: "12px 16px", color: "var(--faint)", fontWeight: 500 }}>Added</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentIdxVectors.map((v, i) => (
                      <tr key={v.id} style={{ borderBottom: "1px solid var(--border)", background: v.id === "vec_003" ? "color-mix(in srgb, #3ecf7f 10%, transparent)" : undefined }}>
                        <td style={{ padding: "12px 16px", fontFamily: "var(--mono)", color: "var(--muted)" }}>{v.id}</td>
                        <td style={{ padding: "12px 16px", color: v.id === "vec_003" ? "#3ecf7f" : "var(--text)" }}>{v.chunk.length > 30 ? v.chunk.slice(0, 30) + "..." : v.chunk}</td>
                        <td style={{ padding: "12px 16px", fontFamily: "var(--mono)", color: "var(--faint)" }}>[{v.vec.map(n => n.toFixed(2)).join(", ").slice(0, 16)}...]</td>
                        <td style={{ padding: "12px 16px", fontFamily: "var(--mono)", color: "var(--muted)" }}>{v.norm}</td>
                        <td style={{ padding: "12px 16px", color: v.id === "vec_003" ? "#3ecf7f" : "var(--muted)" }}>{v.time}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ color: "#3ecf7f", fontSize: 12, fontWeight: 600 }}>Total vectors: 5</span>
                <button className="btn ghost sm">⛶ View all vectors</button>
              </div>
            </div>
          </div>
          
          <div style={{ width: 280, display: "flex", flexDirection: "column", gap: 16 }}>
             <div style={{ ...pnl }}>
              {kgHead("var(--faint)", "2D Vector space (PCA projection)", <span className="note" style={{ fontSize: 10 }}>Spatial view</span>)}
              <div style={{ padding: 16 }}>
                <Plot 
                  data={[{
                    x: currentIdxVectors.map((_, i) => Math.sin(i) * 0.8),
                    y: currentIdxVectors.map((_, i) => Math.cos(i) * 0.8),
                    mode: 'markers',
                    type: 'scatter',
                    marker: { size: 8, color: currentIdxVectors.map(v => v.id === "vec_003" ? "#3ecf7f" : "#a855f7") }
                  }]} 
                  layout={{ ...pLayout("", { showlegend: false, height: 240, margin: { l: 30, r: 10, t: 10, b: 30 } }) }} 
                  style={{ height: 240, width: "100%" }} 
                />
                <div className="row" style={{ gap: 16, justifyContent: "center", marginTop: 12, fontSize: 11, color: "var(--muted)" }}>
                  <span className="row" style={{ gap: 6, alignItems: "center" }}><span style={{ width: 8, height: 8, borderRadius: "50%", border: "2px solid #3ecf7f" }} /> New vector (this chunk)</span>
                  <span className="row" style={{ gap: 6, alignItems: "center" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: "#a855f7" }} /> Existing vectors</span>
                </div>
              </div>
            </div>

            <div style={{ ...pnl }}>
              {kgHead("var(--faint)", "Test similarity (with a query)")}
              <div style={{ padding: 16 }}>
                <div className="note" style={{ marginBottom: 12 }}>Enter a query to see nearest vectors.</div>
                <div className="row" style={{ gap: 8, marginBottom: 16 }}>
                  <input type="text" value={idxQuery} onChange={e => setIdxQuery(e.target.value)} style={{ flex: 1, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, padding: "8px 12px", color: "var(--text)" }} />
                  <button className="btn sm">Search</button>
                </div>
                
                <div className="row" style={{ justifyContent: "space-between", fontSize: 11, color: "var(--faint)", marginBottom: 8, borderBottom: "1px solid var(--border)", paddingBottom: 4 }}>
                  <span>Top 5 results</span>
                  <span>Score (Cosine)</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {currentIdxVectors.map((v, i) => (
                    <div key={v.id} className="row" style={{ justifyContent: "space-between", alignItems: "center", fontSize: 11, padding: "6px 8px", borderRadius: 4, background: v.id === "vec_003" ? "color-mix(in srgb, #3ecf7f 10%, transparent)" : undefined }}>
                      <span className="mono" style={{ color: v.id === "vec_003" ? "#3ecf7f" : "var(--muted)" }}>{v.id}</span>
                      <span style={{ flex: 1, padding: "0 12px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", color: v.id === "vec_003" ? "#3ecf7f" : "var(--text)" }}>{v.chunk}</span>
                      <span className="mono" style={{ color: v.id === "vec_003" ? "#3ecf7f" : "var(--muted)" }}>{(0.912 - i * 0.1).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderBackprop = () => {
    return (
      <div style={{ padding: 16, flex: 1, overflowY: "auto" }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div>
            <h2 style={{ fontSize: 16, margin: "0 0 4px", display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ color: "#a855f7" }}>⚔</span> Backpropagation / Embedding Training Simulator
            </h2>
            <div className="note">Learn how embeddings are updated using backpropagation step by step.</div>
          </div>
          <button className="btn ghost sm">↻ Reset Lab</button>
        </div>

        <div className="row" style={{ gap: 16, alignItems: "stretch" }}>
          <div style={{ width: 180, display: "flex", flexDirection: "column", gap: 12 }}>
             <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>Training Controls</div>
             <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
               {[
                 { id: 1, title: "Forward Pass", sub: "Compute prediction", ic: "1" },
                 { id: 2, title: "Calculate Loss", sub: "Compare vs target", ic: "2" },
                 { id: 3, title: "Backpropagate", sub: "Compute gradients", ic: "3" },
                 { id: 4, title: "Update Weights", sub: "Update embeddings", ic: "4" },
                 { id: 5, title: "Next Epoch", sub: "Go to next epoch", ic: "▶" }
               ].map(s => (
                 <button key={s.id} onClick={() => setBpStep(s.id)} style={{ textAlign: "left", padding: "8px 10px", borderRadius: 8, border: `1px solid ${bpStep === s.id ? "var(--accent)" : "var(--border)"}`, background: bpStep === s.id ? "var(--accent-weak)" : "var(--panel)", cursor: "pointer", display: "flex", gap: 10, alignItems: "center" }}>
                   <span style={{ width: 18, height: 18, borderRadius: "50%", background: bpStep === s.id ? "var(--accent)" : "var(--panel-2)", color: bpStep === s.id ? "#fff" : "var(--muted)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700 }}>{s.ic}</span>
                   <div>
                     <div style={{ fontSize: 11.5, fontWeight: 600, color: bpStep === s.id ? "var(--accent)" : "var(--text)" }}>{s.title}</div>
                     <div style={{ fontSize: 9.5, color: "var(--faint)", marginTop: 2 }}>{s.sub}</div>
                   </div>
                 </button>
               ))}
             </div>
             
             <button className="btn block" style={{ marginTop: 8 }} onClick={() => setBpRunning(!bpRunning)}>
               {bpRunning ? "⏸ Pause Training" : "▶ Auto Train (Run all epochs)"}
             </button>
          </div>
          
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
             <div style={{ ...pnl, padding: 16 }}>
               <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 16 }}>Training Pipeline</div>
               <div className="row" style={{ justifyContent: "space-between", alignItems: "center", padding: "0 12px" }}>
                 {[
                   { id: 1, t: "Forward Pass", color: "#5b7cff" },
                   { id: 2, t: "Calculate Loss", color: "#a855f7" },
                   { id: 3, t: "Backpropagate", color: "#f0616d" },
                   { id: 4, t: "Update Weights", color: "#f59e0b" },
                   { id: 5, t: "Next Epoch", color: "#3ecf7f" }
                 ].map((s, i) => (
                   <React.Fragment key={s.id}>
                     <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, opacity: bpStep >= s.id ? 1 : 0.4 }}>
                       <div style={{ width: 32, height: 32, borderRadius: "50%", border: `2px solid ${bpStep === s.id ? s.color : "var(--border-strong)"}`, color: bpStep === s.id ? s.color : "var(--text)", display: "grid", placeItems: "center", fontSize: 14, fontWeight: 600, background: bpStep === s.id ? `color-mix(in srgb, ${s.color} 10%, transparent)` : "transparent" }}>
                         {s.id}
                       </div>
                       <div style={{ fontSize: 10, fontWeight: 500, color: bpStep === s.id ? s.color : "var(--muted)" }}>{s.t}</div>
                     </div>
                     {i < 4 && <div style={{ height: 1, width: 24, background: "var(--border-strong)", opacity: bpStep > s.id ? 1 : 0.3 }} />}
                   </React.Fragment>
                 ))}
               </div>
             </div>

             <div style={{ ...pnl }}>
               {kgHead("#5b7cff", `Step ${bpStep}: ` + (bpStep === 1 ? "Forward Pass" : bpStep === 2 ? "Calculate Loss" : bpStep === 3 ? "Backpropagate" : bpStep === 4 ? "Update Weights" : "Next Epoch"))}
               <div style={{ padding: 20 }}>
                 <div className="row" style={{ justifyContent: "space-between", alignItems: "center", paddingBottom: 24, borderBottom: "1px solid var(--border)" }}>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Input Token</div>
                     <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)" }}>"king"</div>
                   </div>
                   <div style={{ color: "var(--border-strong)" }}>→</div>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Embedding (Current)</div>
                     <div className="mono" style={{ fontSize: 13, color: "#5b7cff" }}>[0.21, 0.53, 0.71, 0.32, 0.11, 0.44, 0.66, 0.25]</div>
                   </div>
                   <div style={{ color: "var(--border-strong)" }}>→</div>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Prediction</div>
                     <div style={{ fontSize: 14, fontWeight: 600, color: "#a855f7" }}>"queen"</div>
                   </div>
                 </div>
                 
                 <div className="row" style={{ justifyContent: "center", alignItems: "center", padding: "24px 0", borderBottom: "1px solid var(--border)", gap: 40 }}>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Target Token (Expected)</div>
                     <div style={{ fontSize: 14, fontWeight: 600, color: "#a855f7" }}>"queen"</div>
                   </div>
                   <div style={{ textAlign: "center" }}>
                     <div className="note" style={{ marginBottom: 8 }}>Target Embedding</div>
                     <div className="mono" style={{ fontSize: 13, color: "#f59e0b" }}>[0.40, 0.48, 0.60, 0.30, 0.20, 0.43, 0.62, 0.35]</div>
                   </div>
                 </div>

                 <div className="row" style={{ gap: 40, paddingTop: 24 }}>
                   <div>
                     <div className="note" style={{ marginBottom: 4 }}>Model</div>
                     <div style={{ fontWeight: 600 }}>Demo-Embed-8D</div>
                   </div>
                   <div>
                     <div className="note" style={{ marginBottom: 4 }}>Embedding Dimensions</div>
                     <div style={{ fontWeight: 600 }}>8D</div>
                   </div>
                   <div>
                     <div className="note" style={{ marginBottom: 4 }}>Learning Rate</div>
                     <div style={{ fontWeight: 600 }}>0.05</div>
                   </div>
                 </div>
               </div>
             </div>
          </div>
          
          <div style={{ width: 240, display: "flex", flexDirection: "column", gap: 16 }}>
             <div className="row" style={{ gap: 12 }}>
               <div style={{ ...pnl, flex: 1, padding: 16 }}>
                 <div className="note" style={{ marginBottom: 4 }}>Training Status</div>
                 <div style={{ color: "#3ecf7f", fontWeight: 600, fontSize: 13 }}>● Running</div>
               </div>
               <div style={{ ...pnl, flex: 1, padding: 16 }}>
                 <div className="note" style={{ marginBottom: 4 }}>Epoch</div>
                 <div style={{ fontWeight: 600, fontSize: 16 }}>{bpEpoch} / 10</div>
               </div>
             </div>
             
             <div style={{ ...pnl }}>
               {kgHead("var(--faint)", "Training Summary")}
               <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12, fontSize: 12 }}>
                 <div className="row" style={{ justifyContent: "space-between" }}><span className="note">Epoch</span><span style={{ fontWeight: 600 }}>{bpEpoch} / 10</span></div>
                 <div className="row" style={{ justifyContent: "space-between" }}><span className="note">Loss (MSE)</span><span style={{ fontWeight: 600 }}>{bpLoss.toFixed(2)}</span></div>
                 <div className="row" style={{ justifyContent: "space-between" }}><span className="note">Status</span><span style={{ fontWeight: 600, color: "#f59e0b" }}>Running</span></div>
                 <div className="row" style={{ justifyContent: "space-between" }}><span className="note">Best Loss</span><span style={{ fontWeight: 600 }}>0.42</span></div>
               </div>
             </div>
             
             <div style={{ ...pnl }}>
               {kgHead("var(--faint)", "Loss Over Epochs")}
               <div style={{ padding: 16 }}>
                 <Plot 
                  data={[{
                    x: [1, 2, 3, 4, 5, 6, 7],
                    y: [1.0, 0.85, 0.65, 0.52, 0.42, 0.35, 0.28],
                    mode: 'lines+markers',
                    type: 'scatter',
                    line: { color: "#f0616d" },
                    name: 'Loss (MSE)'
                  }]} 
                  layout={{ ...pLayout("", { showlegend: false, height: 180, margin: { l: 30, r: 10, t: 10, b: 20 } }) }} 
                  style={{ height: 180, width: "100%" }} 
                />
               </div>
             </div>
             
             <div style={{ ...pnl }}>
               {kgHead("var(--faint)", "Before vs After (Current Step)")}
               <div style={{ padding: 16, fontSize: 11, fontFamily: "var(--mono)" }}>
                 <div className="row" style={{ justifyContent: "space-between", color: "var(--muted)", marginBottom: 8, fontFamily: "var(--sans)" }}>
                   <span>L2 Change</span>
                   <span style={{ color: "#3ecf7f", fontWeight: 600 }}>0.072</span>
                 </div>
                 <div style={{ color: "#5b7cff", marginBottom: 8 }}>[0.21, 0.53, 0.71, 0.32, ...]</div>
                 <div style={{ textAlign: "center", color: "var(--faint)", marginBottom: 8 }}>↓</div>
                 <div style={{ color: "var(--faint)", marginBottom: 4, fontFamily: "var(--sans)" }}>After (Step {bpEpoch})</div>
                 <div style={{ color: "#3ecf7f" }}>[0.24, 0.49, 0.76, 0.29, ...]</div>
               </div>
             </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="card" style={{ height: "calc(100vh - 180px)", display: "flex", flexDirection: "column", overflow: "hidden", padding: 0 }}>
      {renderStepper()}
      {mod === "pool" ? renderPooling() : 
       mod === "index" ? renderIndexing() : 
       mod === "backprop" ? renderBackprop() : 
       <div style={{ padding: 40, flex: 1, display: "grid", placeItems: "center", color: "var(--muted)" }}>
         <div style={{ textAlign: "center" }}>
           <div style={{ fontSize: 40, marginBottom: 16 }}>🚧</div>
           <h3 style={{ margin: "0 0 8px" }}>Module under construction</h3>
           <p className="note">The {mod} module is coming soon.</p>
         </div>
       </div>}
    </div>
  );
}
