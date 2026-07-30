// Real agent tooling for the Agent Lab — every tool actually executes (in-browser
// or via the existing server proxies), no mocks. Plus ReAct prompt/parse helpers.

import { chunkText, buildIndex, retrieve, type RagIndex } from "./ragUtils";

export interface ToolCtx {
  knowledgeIndex?: RagIndex | null;
  knowledgeChunks?: string[];
  requestApproval?: (question: string) => Promise<string>;
}
export interface AgentTool {
  id: string;
  name: string;
  desc: string;
  example: string;
  run: (input: string, ctx: ToolCtx) => Promise<string>;
}

// ── safe arithmetic evaluator (recursive descent — no eval/Function) ──
export function safeCalc(expr: string): number {
  const s = expr.trim();
  let i = 0;
  const ws = () => { while (i < s.length && /\s/.test(s[i])) i++; };
  const peek = () => s[i];
  const funcs: Record<string, (...a: number[]) => number> = {
    sqrt: Math.sqrt, abs: Math.abs, round: Math.round, floor: Math.floor, ceil: Math.ceil,
    sin: Math.sin, cos: Math.cos, tan: Math.tan, log: Math.log10, ln: Math.log, exp: Math.exp,
    pow: Math.pow, min: Math.min, max: Math.max, sign: Math.sign, cbrt: Math.cbrt,
  };
  const consts: Record<string, number> = { pi: Math.PI, e: Math.E };
  function expr_(): number { let v = term(); ws(); while (peek() === "+" || peek() === "-") { const op = peek(); i++; const r = term(); v = op === "+" ? v + r : v - r; ws(); } return v; }
  function term(): number { let v = factor(); ws(); while (peek() === "*" || peek() === "/" || peek() === "%") { const op = peek(); i++; const r = factor(); v = op === "*" ? v * r : op === "/" ? v / r : v % r; ws(); } return v; }
  function factor(): number { let v = base(); ws(); if (peek() === "^") { i++; v = Math.pow(v, factor()); } return v; }
  function base(): number {
    ws();
    if (peek() === "(") { i++; const v = expr_(); ws(); if (peek() !== ")") throw new Error("expected )"); i++; return v; }
    if (peek() === "-") { i++; return -base(); }
    if (peek() === "+") { i++; return base(); }
    const num = s.slice(i).match(/^\d*\.?\d+([eE][+-]?\d+)?/);
    if (num) { i += num[0].length; return parseFloat(num[0]); }
    const id = s.slice(i).match(/^[a-zA-Z_]\w*/);
    if (id) {
      i += id[0].length; const name = id[0].toLowerCase(); ws();
      if (peek() === "(") {
        i++; const args: number[] = []; ws();
        if (peek() !== ")") { args.push(expr_()); ws(); while (peek() === ",") { i++; args.push(expr_()); ws(); } }
        if (peek() !== ")") throw new Error("expected )"); i++;
        const fn = funcs[name]; if (!fn) throw new Error(`unknown function "${name}"`); return fn(...args);
      }
      if (name in consts) return consts[name];
      throw new Error(`unknown identifier "${name}"`);
    }
    throw new Error("unexpected token");
  }
  const out = expr_(); ws();
  if (i < s.length) throw new Error(`unexpected input near "${s.slice(i, i + 8)}"`);
  if (!Number.isFinite(out)) throw new Error("non-finite result");
  return out;
}

// ── tool executors ──
function dateTool(input: string): string {
  const now = new Date();
  const m = (input || "").match(/(\d{4}-\d{1,2}-\d{1,2})/);
  if (m) {
    const target = new Date(m[1] + "T00:00:00");
    if (isNaN(target.getTime())) return "Error: could not parse that date.";
    const days = Math.round((target.getTime() - now.getTime()) / 86_400_000);
    return `${m[1]} is ${days >= 0 ? `${days} day(s) from now` : `${-days} day(s) ago`}. Today is ${now.toISOString().slice(0, 10)} (${now.toLocaleDateString(undefined, { weekday: "long" })}).`;
  }
  return `Current date & time: ${now.toString()} — ISO ${now.toISOString()}.`;
}
async function webFetchTool(input: string): Promise<string> {
  const url = (input || "").trim().replace(/^["']|["']$/g, "");
  if (!/^https?:\/\//i.test(url)) return "Error: provide a valid http(s) URL.";
  try {
    const r = await fetch("/api/rag/fetch-url", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }) });
    const j = await r.json();
    if (!r.ok) return "Error: " + (j.error || "fetch failed");
    const t = (j.text || "").replace(/\s+/g, " ").trim();
    return t ? t.slice(0, 1200) + (t.length > 1200 ? " …" : "") : "The page had no extractable text.";
  } catch (e) { return "Error: " + (e as Error).message; }
}
function knowledgeTool(input: string, ctx: ToolCtx): string {
  if (!ctx.knowledgeIndex || !ctx.knowledgeChunks?.length) return "No knowledge base is configured for this agent.";
  const hits = retrieve(ctx.knowledgeIndex, input, "hybrid", 3);
  if (!hits.length) return "No relevant passages found in the knowledge base.";
  return hits.map((h, k) => `[${k + 1}] ${ctx.knowledgeChunks![h.i]}`).join("\n").slice(0, 1400);
}
async function httpTool(input: string): Promise<string> {
  const s = (input || "").trim();
  let spec: { method?: string; url?: string; body?: unknown; headers?: Record<string, string> };
  if (s.startsWith("{")) { try { spec = JSON.parse(s); } catch { return "Error: input must be a URL, or JSON like {\"method\":\"POST\",\"url\":\"…\",\"body\":{…}}."; } }
  else spec = { method: "GET", url: s.replace(/^["']|["']$/g, "") };
  if (!spec.url || !/^https?:\/\//i.test(spec.url)) return "Error: provide a valid http(s) URL.";
  try {
    const r = await fetch("/api/agent/http", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(spec) });
    const j = await r.json();
    if (!r.ok) return "Error: " + (j.error || "request failed");
    return `HTTP ${j.status}\n${(j.text || "").slice(0, 1400)}`;
  } catch (e) { return "Error: " + (e as Error).message; }
}
async function approvalTool(input: string, ctx: ToolCtx): Promise<string> {
  if (!ctx.requestApproval) return "Auto-approved (no human approver connected).";
  return await ctx.requestApproval(input || "Approve this action?");
}
function statsTool(input: string): string {
  const nums = (input.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
  if (!nums.length) return "Error: provide a list of numbers.";
  const n = nums.length, sum = nums.reduce((a, b) => a + b, 0), mean = sum / n;
  const s = [...nums].sort((a, b) => a - b); const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  const std = Math.sqrt(nums.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  return `count=${n} sum=${sum} mean=${mean.toFixed(3)} median=${median} min=${Math.min(...nums)} max=${Math.max(...nums)} stdev=${std.toFixed(3)}`;
}
function unitTool(input: string): string {
  const m = input.toLowerCase().match(/(-?\d+(?:\.\d+)?)\s*°?\s*([a-z]+)\s*(?:to|in|->|→)\s*°?\s*([a-z]+)/);
  if (!m) return 'Error: use like "10 km to mi" or "100 f to c".';
  const val = parseFloat(m[1]), from = m[2], to = m[3];
  const factors: Record<string, number> = { m: 1, km: 1000, cm: 0.01, mm: 0.001, mi: 1609.344, ft: 0.3048, in: 0.0254, yd: 0.9144, g: 1, kg: 1000, mg: 0.001, lb: 453.592, oz: 28.3495 };
  if (factors[from] && factors[to]) return `${val} ${from} = ${(val * factors[from] / factors[to]).toFixed(4)} ${to}`;
  const toC = (v: number, u: string) => u === "c" ? v : u === "f" ? (v - 32) * 5 / 9 : u === "k" ? v - 273.15 : NaN;
  const fromC = (c: number, u: string) => u === "c" ? c : u === "f" ? c * 9 / 5 + 32 : u === "k" ? c + 273.15 : NaN;
  const c = toC(val, from); const out = fromC(c, to);
  if (!isNaN(out)) return `${val}°${from.toUpperCase()} = ${out.toFixed(2)}°${to.toUpperCase()}`;
  return `Error: cannot convert "${from}" to "${to}".`;
}
function jsonExtractTool(input: string): string {
  const nl = input.indexOf("\n");
  let path = "", body = input;
  if (nl > 0) { path = input.slice(0, nl).trim(); body = input.slice(nl + 1); }
  body = body.replace(/^HTTP\s+\d+\s*/i, "").trim();
  let data: unknown; try { data = JSON.parse(body); } catch { return "Error: the JSON part is invalid. Format: first line = dot-path (optional), rest = JSON."; }
  if (!path) return Array.isArray(data) ? `array of ${data.length}` : `top-level keys: ${Object.keys(data as object).join(", ")}`;
  let cur: unknown = data;
  for (const key of path.split(/[.[\]]/).filter(Boolean)) { if (cur == null || typeof cur !== "object") { cur = undefined; break; } cur = (cur as Record<string, unknown>)[key]; }
  if (cur === undefined) return `No value at path "${path}".`;
  return typeof cur === "object" ? JSON.stringify(cur).slice(0, 800) : String(cur);
}

export const AGENT_TOOLS: AgentTool[] = [
  { id: "calculator", name: "calculator", desc: "evaluate an arithmetic / math expression", example: "2*(3+4)^2  or  sqrt(144)+pi", run: async (input) => { try { return String(safeCalc(input)); } catch (e) { return "Error: " + (e as Error).message; } } },
  { id: "datetime", name: "datetime", desc: "current date/time, or days until/since a date", example: "now  |  days until 2026-12-25", run: async (input) => dateTool(input) },
  { id: "web_fetch", name: "web_fetch", desc: "fetch the readable text of a web page by URL", example: "https://en.wikipedia.org/wiki/Agent", run: async (input) => webFetchTool(input) },
  { id: "knowledge", name: "knowledge", desc: "search the agent's knowledge base for relevant passages", example: "what is the refund window?", run: async (input, ctx) => knowledgeTool(input, ctx) },
  { id: "http_request", name: "http_request", desc: "call a REST API — a plain URL (GET) or JSON {method,url,body} — and read the response", example: 'https://api.github.com/repos/vercel/next.js', run: async (input) => httpTool(input) },
  { id: "human_approval", name: "human_approval", desc: "pause and ask a human to approve before a sensitive or irreversible action", example: "Approve a $650 refund for order #1234?", run: async (input, ctx) => approvalTool(input, ctx) },
  { id: "statistics", name: "statistics", desc: "compute mean / median / min / max / stdev of a list of numbers", example: "12, 7, 9, 15, 6", run: async (input) => statsTool(input) },
  { id: "unit_convert", name: "unit_convert", desc: "convert between units — length, mass, or temperature", example: "10 km to mi", run: async (input) => unitTool(input) },
  { id: "json_extract", name: "json_extract", desc: "read a value from JSON by dot-path (line 1 = path, rest = JSON) — pairs with http_request", example: "stargazers_count\\n{ …json… }", run: async (input) => jsonExtractTool(input) },
];

// Build a TF-IDF knowledge base from pasted text (reuses the RAG backend).
export function buildKnowledge(text: string): { index: RagIndex; chunks: string[] } | null {
  const clean = (text || "").trim();
  if (!clean) return null;
  const chunks = chunkText(clean, 60, 12);
  if (!chunks.length) return null;
  return { index: buildIndex(chunks), chunks };
}

// ── ReAct prompt + parse ──
export function reactSystemPrompt(tools: AgentTool[], goal: string): string {
  const list = tools.map((t) => `- ${t.name}: ${t.desc} (example input: ${t.example})`).join("\n");
  return `You are a ReAct agent that solves tasks by reasoning and using tools.${goal ? `\nYour goal / role: ${goal}` : ""}

Available tools:
${list || "(no tools — answer directly)"}

Work step by step. On each turn output EXACTLY one block, nothing else:

Thought: <brief reasoning>
Action: <one tool name from the list above>
Action Input: <the input to pass to that tool>

When you have enough information, instead output:

Thought: <brief reasoning>
Final Answer: <the answer for the user>

Rules: PREFER TOOLS over doing the work yourself. If a tool can compute, look up, or fetch something — arithmetic, dates, web pages, the knowledge base — you MUST call that tool instead of answering from memory (you are unreliable at mental math and date arithmetic). Handle one thing per step. Only give the Final Answer once the tools have given you everything you need. After each Observation, continue the loop. Never write "Observation:" yourself — the system provides it. Keep each Thought to one sentence.`;
}

export interface ReActParse { thought?: string; action?: string; input?: string; final?: string; }
export function parseReAct(text: string): ReActParse {
  const thought = text.match(/Thought:\s*([\s\S]*?)(?=\n\s*(?:Action|Final Answer)\s*:|$)/i)?.[1]?.trim();
  const final = text.match(/Final Answer:\s*([\s\S]*)/i)?.[1]?.trim();
  if (final) return { thought, final };
  const action = text.match(/Action:\s*([^\n]+)/i)?.[1]?.trim().replace(/[.'"]+$/, "");
  const input = text.match(/Action Input:\s*([\s\S]*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|$)/i)?.[1]?.trim().replace(/^["']|["']$/g, "");
  return { thought, action, input };
}
