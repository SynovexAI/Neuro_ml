"use client";
import { useRef, useState, type ReactNode } from "react";
import AgentOutput from "@/components/AgentOutput";

// Self-contained Agentic RAG panel. Runs a real ReAct loop over /api/chat (injected as `chat`)
// and the parent's retrieval (injected as `retrieve`) — reuses the existing engine, never touches
// the current single-shot path. Supports multi-turn conversation, a streamed final answer,
// controlled+persisted config, and a per-answer retrieval-grounding readout.

export type AgentTool = "vector" | "keyword" | "hybrid" | "kg" | "web";
export type AgentHit = { i: number; score: number };
export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };
export type AgentConfig = { enabled: AgentTool[]; maxSteps: number; topK: number; selfCheck: boolean; maxTokens: number };

export type TraceEvent =
  | { kind: "plan" | "reflect" | "grade"; text: string; step: number }
  | { kind: "tool"; tool: AgentTool; query: string; hits: AgentHit[]; step: number }
  | { kind: "web"; url: string; ok: boolean; step: number }
  | { kind: "answer"; step: number }
  | { kind: "error"; text: string; step: number };

type Turn = { id: number; question: string; events: TraceEvent[]; final: { answer: string; citations: number[] } | null; streaming: boolean; stats: { steps: number; retrievals: number; requeries: number; ms: number } | null };

const TOOL_META: Record<AgentTool, { label: string; icon: string; color: string; desc: string }> = {
  vector: { label: "vector search", icon: "◆", color: "#5b7cff", desc: "meaning match — good for paraphrased questions" },
  keyword: { label: "keyword (BM25)", icon: "🔤", color: "#22b8cf", desc: "exact terms — codes, names, IDs" },
  hybrid: { label: "hybrid", icon: "⚡", color: "#f59e0b", desc: "blends meaning + exact terms (safe default)" },
  kg: { label: "knowledge graph", icon: "🕸", color: "#a855f7", desc: "follows entity → relation links" },
  web: { label: "web fetch", icon: "🌐", color: "#3ecf7f", desc: "fetch a public URL — reaches outside your docs" },
};

type Props = {
  chunks: { text: string; docName: string; docKind?: string }[];
  tools: AgentTool[];
  retrieve: (query: string, tool: AgentTool, k: number) => Promise<AgentHit[]>;
  chat: (messages: ChatMessage[], opts?: { maxTokens?: number }) => Promise<string>;
  chatStream?: (messages: ChatMessage[], onToken: (t: string) => void, opts?: { maxTokens?: number }) => Promise<string>;
  fetchWeb?: (url: string) => Promise<{ title: string; text: string } | null>;
  modelPicker?: ReactNode;
  note?: ReactNode;
  defaultQuestion?: string;
  disabled?: boolean;
  compact?: boolean;
  config?: AgentConfig;
  onConfigChange?: (c: AgentConfig) => void;
};

function parseAction(raw: string): { thought?: string; action?: string; tool?: string; query?: string; url?: string; answer?: string; citations?: number[] } | null {
  if (!raw) return null;
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : raw;
  const start = body.indexOf("{");
  if (start < 0) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < body.length; i++) {
    const ch = body[i];
    if (inStr) { if (esc) esc = false; else if (ch === "\\") esc = true; else if (ch === '"') inStr = false; continue; }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") { depth--; if (depth === 0) { try { return JSON.parse(body.slice(start, i + 1)); } catch { return null; } } }
  }
  return null;
}

const snippet = (s: string, n = 260) => (s.length > n ? s.slice(0, n).trimEnd() + "…" : s);
const DEFAULT_CFG: AgentConfig = { enabled: ["vector", "keyword", "hybrid"], maxSteps: 5, topK: 4, selfCheck: true, maxTokens: 1024 };

const CopySvg = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

const CheckSvg = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

export function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button className="out-copy-btn" onClick={onCopy} title={copied ? "Copied!" : "Copy to clipboard"}>
      {copied ? <CheckSvg /> : <CopySvg />}
    </button>
  );
}


export default function AgenticAnswer({ chunks, tools, retrieve, chat, chatStream, fetchWeb, modelPicker, note, defaultQuestion, disabled, compact, config, onConfigChange }: Props) {
  const [question, setQuestion] = useState(defaultQuestion || "What is the refund policy for damaged items?");
  const [internalCfg, setInternalCfg] = useState<AgentConfig>(config ?? DEFAULT_CFG);
  const controlled = !!config && !!onConfigChange;
  const cfg = controlled ? config! : internalCfg;
  const setCfg = (patch: Partial<AgentConfig>) => { const next = { ...cfg, ...patch }; if (controlled) onConfigChange!(next); else setInternalCfg(next); };

  const [turns, setTurns] = useState<Turn[]>([]);
  const [running, setRunning] = useState(false);
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const cancelRef = useRef(false);
  const turnSeq = useRef(0);

  const activeTools = tools.filter((t) => cfg.enabled.includes(t));
  const toggle = (t: AgentTool) => setCfg({ enabled: cfg.enabled.includes(t) ? cfg.enabled.filter((x) => x !== t) : [...cfg.enabled, t] });

  async function run() {
    const q = question.trim();
    if (running || disabled || !q || activeTools.length === 0) return;
    cancelRef.current = false;
    setRunning(true);
    const turnId = ++turnSeq.current;
    const priorTurns = turns.filter((t) => t.final);
    setTurns((ts) => [...ts, { id: turnId, question: q, events: [], final: null, streaming: false, stats: null }]);
    setQuestion("");
    const t0 = performance.now();
    const trace: TraceEvent[] = [];
    const updateTurn = (patch: Partial<Turn>) => setTurns((ts) => ts.map((t) => (t.id === turnId ? { ...t, ...patch } : t)));
    const push = (e: TraceEvent) => { trace.push(e); updateTurn({ events: [...trace] }); };

    const toolList = activeTools.map((t) => `"${t}" (${TOOL_META[t].desc})`).join(", ");
    const sys = [
      "You are a retrieval agent that answers a question using ONLY a document store you search through tools.",
      "On EACH turn reply with a SINGLE JSON object and nothing else, shaped:",
      `{"thought": string, "action": "search" | "answer", "tool"?: string, "query"?: string, "url"?: string, "citations"?: number[]}`,
      `- action "search": pick a tool from [${toolList}] and a focused "query" to fetch more context. For the "web" tool put a full URL in "url".`,
      `- action "answer": use ONLY when your observations contain enough evidence. Put the chunk numbers you'll rely on in "citations" (e.g. [3, 7]); the prose answer is written separately.`,
      "- Cite chunk numbers exactly as shown in observations, e.g. the number 3 for [chunk 3].",
      cfg.selfCheck ? "- Before answering, in your thought briefly grade whether the retrieved chunks address every part of the question; if a part is unsupported, search again instead of answering." : "",
      "- If, after searching, the answer isn't in the documents, still choose action \"answer\" with empty citations.",
      `You may take at most ${cfg.maxSteps} search steps.`,
    ].filter(Boolean).join("\n");

    // Seed the loop with prior conversation turns so follow-ups have context.
    const messages: ChatMessage[] = [{ role: "system", content: sys }];
    for (const p of priorTurns) { messages.push({ role: "user", content: p.question }); if (p.final) messages.push({ role: "assistant", content: p.final.answer }); }
    messages.push({ role: "user", content: `Question: ${q}` });

    let retrievals = 0, requeries = 0; const usedQueries = new Set<string>();
    const seenIds = new Set<number>();
    let stepNo = 0;

    try {
      for (let iter = 0; iter <= cfg.maxSteps; iter++) {
        if (cancelRef.current) { push({ kind: "error", text: "stopped", step: stepNo + 1 }); break; }
        const forceAnswer = iter === cfg.maxSteps;
        if (forceAnswer) messages.push({ role: "user", content: "You have no search steps left. Choose action \"answer\" now with citations for the evidence gathered (empty if insufficient)." });

        const raw = await chat(messages, { maxTokens: 700 });
        const act = parseAction(raw);
        messages.push({ role: "assistant", content: raw });

        if (act && act.action === "search" && !forceAnswer) {
          stepNo++;
          if (act.thought) push({ kind: iter > 0 ? "reflect" : "plan", text: act.thought, step: stepNo });
          let tool = (act.tool as AgentTool) || activeTools[0];
          if (!activeTools.includes(tool)) tool = activeTools[0];
          stepNo++;
          if (tool === "web") {
            const url = act.url || act.query || "";
            if (fetchWeb && /^https?:\/\//i.test(url)) {
              const doc = await fetchWeb(url).catch(() => null);
              push({ kind: "web", url, ok: !!doc, step: stepNo });
              messages.push({ role: "user", content: doc ? `OBSERVATION (web ${url}):\n${snippet(doc.text, 900)}` : `OBSERVATION (web ${url}): fetch failed.` });
            } else { push({ kind: "web", url, ok: false, step: stepNo }); messages.push({ role: "user", content: "OBSERVATION (web): no valid URL." }); }
            continue;
          }
          const qq = (act.query || q).trim();
          if (usedQueries.has(`${tool}:${qq}`)) requeries++;
          usedQueries.add(`${tool}:${qq}`);
          const hits = await retrieve(qq, tool, cfg.topK);
          retrievals++;
          hits.forEach((h) => seenIds.add(h.i));
          push({ kind: "tool", tool, query: qq, hits, step: stepNo });
          const obs = hits.length ? hits.map((h) => `[chunk ${h.i + 1}] (score ${h.score.toFixed(2)}) ${snippet(chunks[h.i]?.text || "")}`).join("\n") : "(no chunks matched)";
          messages.push({ role: "user", content: `OBSERVATION (tool=${tool}, query="${qq}"):\n${obs}` });
          continue;
        }

        // action === "answer" (or unparseable / forced) → grade, then STREAM the grounded answer.
        stepNo++;
        if (cfg.selfCheck && act?.thought) push({ kind: "grade", text: act.thought, step: stepNo });
        stepNo++;
        push({ kind: "answer", step: stepNo });
        const cites = (act?.citations || []).map((n) => n - 1).filter((i) => i >= 0 && i < chunks.length);
        const useIdx = cites.length ? cites : [...seenIds].slice(0, cfg.topK);
        const ctx = useIdx.length ? useIdx.map((i) => `[chunk ${i + 1} · ${chunks[i]?.docName}] ${chunks[i]?.text}`).join("\n\n") : "(no relevant context retrieved)";
        const finalMsgs: ChatMessage[] = [
          { role: "system", content: "Answer the user's question using ONLY the provided context. Cite sources inline as [chunk N]. Be concise and accurate. If the context doesn't contain the answer, say you don't know." },
          ...priorTurns.flatMap((p) => (p.final ? [{ role: "user" as const, content: p.question }, { role: "assistant" as const, content: p.final.answer }] : [])),
          { role: "user", content: `Context:\n${ctx}\n\nQuestion: ${q}` },
        ];
        updateTurn({ streaming: true, final: { answer: "", citations: useIdx } });
        let answerText = "";
        if (chatStream) {
          await chatStream(finalMsgs, (tok) => { answerText += tok; updateTurn({ final: { answer: answerText, citations: useIdx } }); }, { maxTokens: cfg.maxTokens });
        } else {
          answerText = await chat(finalMsgs, { maxTokens: cfg.maxTokens });
          updateTurn({ final: { answer: answerText, citations: useIdx } });
        }
        updateTurn({ streaming: false });
        break;
      }
    } catch (e) {
      push({ kind: "error", text: (e as Error).message, step: stepNo + 1 });
    }
    updateTurn({ stats: { steps: stepNo, retrievals, requeries, ms: Math.round(performance.now() - t0) } });
    setRunning(false);
  }

  const card = { border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)" } as const;
  const dotFor = (e: TraceEvent) => e.kind === "plan" ? "#a855f7" : e.kind === "reflect" ? "#f59e0b" : e.kind === "tool" ? TOOL_META[e.tool].color : e.kind === "web" ? "#3ecf7f" : e.kind === "grade" ? "#3ecf7f" : e.kind === "answer" ? "var(--good)" : "var(--crit)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* setup */}
      <div style={{ ...card, padding: compact ? 12 : 15 }}>
        <div style={{ fontSize: 10.5, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)", marginBottom: 12, display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#a855f7" }} />agent setup
        </div>
        {modelPicker && (
          <div style={{ marginBottom: 14, paddingBottom: 14, borderBottom: "1px solid var(--border)" }}>
            <div className="note" style={{ marginBottom: 7 }}>Generation model — the LLM that runs the agent loop</div>
            {modelPicker}
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: compact ? "1fr" : "1.1fr 1fr", gap: 16 }}>
          <div>
            <div className="note" style={{ marginBottom: 7 }}>Tools the agent may call</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {tools.map((t) => {
                const on = cfg.enabled.includes(t); const m = TOOL_META[t];
                return (
                  <button key={t} onClick={() => toggle(t)} style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 12, padding: "7px 10px", borderRadius: 8, cursor: "pointer", textAlign: "left", border: `1px solid ${on ? m.color : "var(--border)"}`, background: on ? `color-mix(in srgb, ${m.color} 12%, transparent)` : "var(--surface)", color: on ? "var(--text)" : "var(--muted)" }}>
                    <span style={{ fontSize: 13, flex: "0 0 auto" }}>{m.icon}</span>
                    <span style={{ display: "flex", flexDirection: "column", minWidth: 0, flex: 1, gap: 1 }}>
                      <b style={{ fontWeight: 600 }}>{m.label}</b>
                      <span style={{ fontSize: 10, fontWeight: 400, color: "var(--faint)", lineHeight: 1.3 }}>{m.desc}</span>
                    </span>
                    <span style={{ flex: "0 0 auto", color: on ? m.color : "var(--faint)" }}>{on ? "✓" : "○"}</span>
                  </button>
                );
              })}
            </div>
            {note && <div style={{ marginTop: 8 }}>{note}</div>}
          </div>
          <div>
            <div className="row" style={{ justifyContent: "space-between", fontSize: 12, color: "var(--muted)", marginBottom: 6 }}><span>Max steps</span><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{cfg.maxSteps}</b></div>
            <input type="range" min={1} max={8} step={1} value={cfg.maxSteps} onChange={(e) => setCfg({ maxSteps: +e.target.value })} style={{ width: "100%" }} />
            <div className="row" style={{ justifyContent: "space-between", fontSize: 12, color: "var(--muted)", margin: "12px 0 6px" }}><span>Chunks per retrieval (k)</span><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{cfg.topK}</b></div>
            <input type="range" min={1} max={8} step={1} value={cfg.topK} onChange={(e) => setCfg({ topK: +e.target.value })} style={{ width: "100%" }} />
            <div className="row" style={{ justifyContent: "space-between", fontSize: 12, color: "var(--muted)", margin: "12px 0 6px" }}><span>Max answer tokens</span><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{cfg.maxTokens}</b></div>
            <input type="range" min={256} max={4096} step={256} value={cfg.maxTokens} onChange={(e) => setCfg({ maxTokens: +e.target.value })} style={{ width: "100%" }} />
            <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--muted)", marginTop: 12, cursor: "pointer" }}>
              <input type="checkbox" checked={cfg.selfCheck} onChange={(e) => setCfg({ selfCheck: e.target.checked })} style={{ width: "auto" }} />self-check retrieved chunks before answering
            </label>
          </div>
        </div>
      </div>

      {/* conversation */}
      {turns.map((turn) => {
        const uniqueRetrieved = Array.from(new Set(turn.events.flatMap((e) => (e.kind === "tool" ? e.hits.map((h) => h.i) : []))));
        const cited = turn.final?.citations ?? [];
        return (
          <div key={turn.id} style={{ ...card, padding: compact ? 12 : 15 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--muted)", flex: "0 0 auto" }}>Q</span>
              <span style={{ fontSize: 13.5, fontWeight: 500, color: "var(--text)" }}>{turn.question}</span>
            </div>

            {turn.events.length > 0 && (
              <div style={{ marginBottom: turn.final ? 14 : 0 }}>
                <div className="note" style={{ marginBottom: 9, textTransform: "uppercase", letterSpacing: ".05em", fontSize: 10 }}>reasoning trace{turn.stats ? ` · ${turn.stats.steps} steps` : ""}</div>
                <div style={{ position: "relative", paddingLeft: 22 }}>
                  <div style={{ position: "absolute", left: 6, top: 6, bottom: 6, width: 2, background: "var(--border)" }} />
                  {turn.events.map((e, idx) => (
                    <div key={idx} style={{ position: "relative", paddingBottom: idx === turn.events.length - 1 ? 0 : 12, animation: "reveal-in .3s ease both" }}>
                      <span style={{ position: "absolute", left: -22, top: 2, width: 14, height: 14, borderRadius: "50%", background: `color-mix(in srgb, ${dotFor(e)} 22%, transparent)`, border: `2px solid ${dotFor(e)}` }} />
                      {e.kind === "tool" ? (
                        <div>
                          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                            <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: TOOL_META[e.tool].color, background: `color-mix(in srgb, ${TOOL_META[e.tool].color} 14%, transparent)`, borderRadius: 5, padding: "2px 7px" }}>tool · {TOOL_META[e.tool].label}</span>
                            <span style={{ fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--muted)" }}>“{e.query}”</span>
                          </div>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 7 }}>
                            {e.hits.length ? e.hits.map((h) => {
                              const key = `${turn.id}:${h.i}`;
                              return (
                                <button key={h.i} onClick={() => setOpenChunk(openChunk === key ? null : key)} title="show chunk text" style={{ fontSize: 11, padding: "3px 8px", borderRadius: 6, cursor: "pointer", background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)" }}>
                                  c{h.i + 1} <span style={{ color: TOOL_META[e.tool].color, fontFamily: "var(--mono)" }}>{h.score.toFixed(2)}</span>
                                </button>
                              );
                            }) : <span className="note">no chunks matched</span>}
                          </div>
                          {e.hits.some((h) => `${turn.id}:${h.i}` === openChunk) && (() => { const oi = Number(openChunk!.split(":")[1]); return (
                            <div style={{ marginTop: 7, fontSize: 11.5, color: "var(--muted)", fontFamily: "var(--mono)", lineHeight: 1.5, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 10px" }}>
                              <b style={{ color: "var(--text)" }}>chunk {oi + 1}</b> · {chunks[oi]?.docName}<br />{snippet(chunks[oi]?.text || "", 400)}
                            </div>
                          ); })()}
                        </div>
                      ) : e.kind === "web" ? (
                        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                          <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: "#3ecf7f", background: "rgba(62,207,127,.14)", borderRadius: 5, padding: "2px 7px" }}>web fetch</span>
                          <span style={{ fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--muted)" }}>{e.url}</span>
                          <span style={{ fontSize: 11, color: e.ok ? "var(--good)" : "var(--crit)" }}>{e.ok ? "✓ fetched" : "✗ failed"}</span>
                        </div>
                      ) : e.kind === "answer" ? (
                        <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--good)", background: "color-mix(in srgb, var(--good) 14%, transparent)", borderRadius: 5, padding: "2px 7px" }}>answer</span>
                      ) : e.kind === "error" ? (
                        <div style={{ fontSize: 12, color: "var(--crit)" }}><b>error</b> · {e.text}</div>
                      ) : (
                        <div>
                          <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: dotFor(e), background: `color-mix(in srgb, ${dotFor(e)} 14%, transparent)`, borderRadius: 5, padding: "2px 7px" }}>{e.kind}</span>
                          <div style={{ fontSize: 12.5, marginTop: 6, color: "var(--text)", lineHeight: 1.5 }}>{e.text}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {turn.final && (
              <div style={{ border: "1px solid color-mix(in srgb, var(--good) 45%, var(--border))", borderRadius: 12, background: "color-mix(in srgb, var(--good) 8%, transparent)", padding: "14px 16px" }}>
                <div style={{ fontSize: 10.5, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".06em", color: "var(--good)", marginBottom: 8 }}>◆ answer</div>
                {turn.streaming
                  ? <div style={{ fontSize: 14, lineHeight: 1.65, color: "var(--text)", whiteSpace: "pre-wrap" }}>{turn.final.answer}<span style={{ display: "inline-block", width: 6, height: 14, background: "var(--accent)", verticalAlign: "-2px", marginLeft: 2, animation: "blink 1s steps(2) infinite" }} /></div>
                  : <AgentOutput text={turn.final.answer} />}
                {!turn.streaming && (
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12, paddingTop: 11, borderTop: "1px solid color-mix(in srgb, var(--good) 30%, var(--border))", alignItems: "center" }}>
                    <span className="note">grounding · <b style={{ color: "var(--text)" }}>{cited.length}</b> cited / {uniqueRetrieved.length} retrieved</span>
                    {cited.map((i) => <span key={i} style={{ fontSize: 11, fontFamily: "var(--mono)", padding: "2px 7px", borderRadius: 5, background: "var(--surface)", border: "1px solid var(--good)" }}>{chunks[i]?.docName || "doc"} · c{i + 1}</span>)}
                    {uniqueRetrieved.filter((i) => !cited.includes(i)).map((i) => <span key={i} title="retrieved but not cited" style={{ fontSize: 11, fontFamily: "var(--mono)", padding: "2px 7px", borderRadius: 5, background: "var(--surface)", border: "1px solid var(--border)", color: "var(--faint)" }}>c{i + 1}</span>)}
                    {turn.stats && <span className="note" style={{ marginLeft: "auto" }}>{turn.stats.retrievals} retrieval{turn.stats.retrievals === 1 ? "" : "s"} · {turn.stats.steps} steps{turn.stats.requeries ? ` · ${turn.stats.requeries} re-query` : ""} · {turn.stats.ms}ms</span>}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}

      {/* ask bar */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder={turns.length ? "Ask a follow-up…" : "Ask a question…"} onKeyDown={(e) => { if (e.key === "Enter") run(); }} style={{ flex: 1 }} disabled={running} />
        {running
          ? <button className="btn ghost" onClick={() => { cancelRef.current = true; }}>■ Stop</button>
          : <button className="btn" onClick={run} disabled={disabled || !question.trim() || activeTools.length === 0}>{turns.length ? "▶ Ask" : "▶ Run agent"}</button>}
        {turns.length > 0 && !running && <button className="btn ghost sm" onClick={() => setTurns([])} title="Clear conversation">🗑 Clear</button>}
      </div>
      {activeTools.length === 0 && <div className="note" style={{ color: "var(--warn)" }}>Enable at least one tool above.</div>}
    </div>
  );
}
