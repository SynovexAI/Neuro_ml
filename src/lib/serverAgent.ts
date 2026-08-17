import "server-only";
import {
  AGENT_TOOLS, safeCalc, dateTool, statsTool, unitTool, jsonExtractTool,
  buildKnowledge, reactSystemPrompt, parseReAct, type ToolCtx,
} from "@/lib/agentTools";
import { retrieve, type RagIndex } from "@/lib/ragUtils";
import { resolvesToPrivate } from "@/lib/net";

/* eslint-disable @typescript-eslint/no-explicit-any */

export type Prov = { baseUrl: string; apiKey: string };
export type ServerRun = { answer: string; iterations: number; toolCounts: Record<string, number>; toolCallCount: number; promptTokens: number; completionTokens: number; outcome: string; error?: string };

const est = (s: string) => Math.round((s || "").length / 4);

// One OpenAI-compatible chat completion, server-side (no session / proxy).
async function chatComplete(prov: Prov, model: string, messages: { role: string; content: string }[], temperature: number, maxTokens: number): Promise<{ text: string; promptTokens: number; completionTokens: number }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60_000);
  try {
    const res = await fetch(prov.baseUrl.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", ...(prov.apiKey ? { Authorization: `Bearer ${prov.apiKey}` } : {}) },
      body: JSON.stringify({ model, messages, temperature, max_tokens: maxTokens, stream: false }),
      signal: ctrl.signal,
    });
    const j = await res.json().catch(() => null) as any;
    if (!res.ok) throw new Error(j?.error?.message || j?.error || `provider HTTP ${res.status}`);
    const text = String(j?.choices?.[0]?.message?.content || "").trim();
    const pt = Number(j?.usage?.prompt_tokens) || est(messages.map((m) => m.content).join(" "));
    const ct = Number(j?.usage?.completion_tokens) || est(text);
    return { text, promptTokens: pt, completionTokens: ct };
  } finally { clearTimeout(timer); }
}

// ── server-side tool executors (web_fetch / http_request are SSRF-guarded) ──
async function serverWebFetch(input: string): Promise<string> {
  const url = (input || "").trim().replace(/^["']|["']$/g, "");
  if (!/^https?:\/\//i.test(url)) return "Error: provide a valid http(s) URL.";
  try {
    if (await resolvesToPrivate(new URL(url).hostname)) return "Error: that host is blocked.";
    const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 12_000);
    const r = await fetch(url, { signal: ctrl.signal, headers: { "user-agent": "AIWorkbench/1.0" } }); clearTimeout(t);
    const html = await r.text();
    const text = html.replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/<style[\s\S]*?<\/style>/gi, " ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    return text ? text.slice(0, 1200) + (text.length > 1200 ? " …" : "") : "The page had no extractable text.";
  } catch (e) { return "Error: " + (e as Error).message; }
}
async function serverHttp(input: string): Promise<string> {
  const s = (input || "").trim();
  let spec: { method?: string; url?: string; body?: unknown };
  if (s.startsWith("{")) { try { spec = JSON.parse(s); } catch { return 'Error: input must be a URL or JSON {"method","url","body"}.'; } }
  else spec = { method: "GET", url: s.replace(/^["']|["']$/g, "") };
  if (!spec.url || !/^https?:\/\//i.test(spec.url)) return "Error: provide a valid http(s) URL.";
  try {
    if (await resolvesToPrivate(new URL(spec.url).hostname)) return "Error: that host is blocked.";
    const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 12_000);
    const r = await fetch(spec.url, { method: (spec.method || "GET").toUpperCase(), headers: { "content-type": "application/json" }, body: spec.body ? JSON.stringify(spec.body) : undefined, signal: ctrl.signal }); clearTimeout(t);
    const text = await r.text();
    return `HTTP ${r.status}\n${text.slice(0, 1400)}`;
  } catch (e) { return "Error: " + (e as Error).message; }
}

async function runTool(name: string, input: string, ctx: ToolCtx): Promise<{ obs: string; known: boolean }> {
  const normName = (name || "").toLowerCase().trim();
  if (normName === "human_approval") {
    return { obs: "NOT APPROVED — human approval is required but no approver is available in this runtime. Do not perform the action.", known: true };
  }
  const toolObj = AGENT_TOOLS.find((t) => t.id === normName || t.name.toLowerCase() === normName);
  if (toolObj) {
    try {
      const obs = await toolObj.run(input, ctx);
      return { obs, known: true };
    } catch (e) {
      return { obs: "Error: " + (e as Error).message, known: true };
    }
  }
  return { obs: `Unknown tool "${name}".`, known: false };
}

const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));

// Server-side ReAct loop for a published in-browser agent config.
export async function runReactServer(cfg: any, task: string, prov: Prov, model: string): Promise<ServerRun> {
  const rawEnabled: string[] = Array.isArray(cfg.tools) ? cfg.tools : ["calculator", "datetime", "knowledge", "db_query", "db_schema"];
  const enabled = Array.from(new Set([...rawEnabled, "db_query", "db_schema"]));
  const toolDefs = AGENT_TOOLS.filter((t) => enabled.includes(t.id));
  let index: RagIndex | null = null, chunks: string[] | null = null;
  if (enabled.includes("knowledge") && cfg.knowledge) { const built = buildKnowledge(String(cfg.knowledge)); if (built) { index = built.index; chunks = built.chunks; } }
  const toolCtx: ToolCtx = {
    knowledgeIndex: index,
    knowledgeChunks: chunks || undefined,
    selectedRagId: cfg.selectedRagId,
    dbTable: cfg.dbTable || null,
    dbTableName: cfg.dbTableName,
    dbCustomSchema: cfg.dbCustomSchema,
  };
  const goal = String(cfg.systemPrompt || "");
  const maxIters = clamp(Number(cfg.maxIterations) || 6, 1, 10);
  const temperature = typeof cfg.temperature === "number" ? cfg.temperature : 0.4;
  const maxTokens = clamp(Number(cfg.maxTokens) || 600, 100, 800);
  const messages: { role: string; content: string }[] = [{ role: "system", content: reactSystemPrompt(toolDefs, goal) }, { role: "user", content: task }];

  let pt = 0, ct = 0, toolCallCount = 0, iterations = 0, outcome = "max_iters", answer = "";
  const toolCounts: Record<string, number> = {};
  const seenCalls = new Set<string>();

  for (let i = 0; i < maxIters; i++) {
    iterations = i + 1;
    const step = await chatComplete(prov, model, messages, temperature, maxTokens);
    pt += step.promptTokens; ct += step.completionTokens;
    const p = parseReAct(step.text);
    if (p.final || (!p.action && !p.final)) { answer = p.final || step.text; outcome = "success"; break; }
    
    const callKey = `${(p.action || "").toLowerCase()}:${(p.input || "").trim()}`;
    if (seenCalls.has(callKey)) {
      messages.push({ role: "assistant", content: step.text });
      messages.push({ role: "user", content: "Observation: Tool already called with identical input. Please provide your Final Answer directly based on what you have." });
      const finalPass = await chatComplete(prov, model, messages, temperature, maxTokens).catch(() => ({ promptTokens: 0, completionTokens: 0, text: "" }));
      pt += finalPass.promptTokens; ct += finalPass.completionTokens;
      const parsedFinal = parseReAct(finalPass.text);
      answer = parsedFinal.final || finalPass.text;
      outcome = "success";
      break;
    }
    seenCalls.add(callKey);

    const { obs, known } = await runTool((p.action || "").toLowerCase(), p.input || "", toolCtx);
    if (known) { toolCallCount++; toolCounts[p.action!.toLowerCase()] = (toolCounts[p.action!.toLowerCase()] || 0) + 1; }
    messages.push({ role: "assistant", content: step.text });
    messages.push({ role: "user", content: `Observation: ${obs}` });
  }

  if (!answer) {
    messages.push({ role: "user", content: "Please provide your final answer to the user now based on what is known." });
    const finalPass = await chatComplete(prov, model, messages, temperature, maxTokens).catch(() => ({ promptTokens: 0, completionTokens: 0, text: "" }));
    pt += finalPass.promptTokens; ct += finalPass.completionTokens;
    const parsedFinal = parseReAct(finalPass.text);
    answer = parsedFinal.final || finalPass.text || "No final answer produced.";
    outcome = "success";
  }

  return { answer, iterations, toolCounts, toolCallCount, promptTokens: pt, completionTokens: ct, outcome };
}

// Server-side multi-step workflow for a published in-browser agent config.
export async function runWorkflowServer(cfg: any, task: string, prov: Prov, model: string): Promise<ServerRun> {
  const steps: { name: string; instruction: string }[] = Array.isArray(cfg.steps) ? cfg.steps : [];
  const temperature = typeof cfg.temperature === "number" ? cfg.temperature : 0.4;
  let prev = task, pt = est(task), ct = 0;
  for (const s of steps) {
    const messages = [
      { role: "system", content: `You are executing step "${s.name}" of a workflow. ${s.instruction}` },
      { role: "user", content: `Original request: ${task}\n\nPrevious step output:\n${prev}` },
    ];
    const r = await chatComplete(prov, model, messages, temperature, 800);
    pt += r.promptTokens; ct += r.completionTokens; prev = r.text;
  }
  return { answer: prev, iterations: steps.length, toolCounts: {}, toolCallCount: 0, promptTokens: pt, completionTokens: ct, outcome: "success" };
}
