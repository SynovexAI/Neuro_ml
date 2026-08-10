// Real agent tooling for the Agent Lab — every tool actually executes (in-browser
// or via the existing server proxies), no mocks. Plus ReAct prompt/parse helpers.

import { chunkText, buildIndex, retrieve, type RagIndex } from "./ragUtils";
import type { Table } from "./etlUtils";

export interface ToolCtx {
  knowledgeIndex?: RagIndex | null;
  knowledgeChunks?: string[];
  requestApproval?: (question: string) => Promise<string>;
  selectedRagId?: string;
  dbTable?: Table | null;
  dbTableName?: string;
  dbCustomSchema?: string;
  a2ui?: boolean; // when true, structured tool results are wrapped as A2UI ```ui blocks
}

// If `text` contains a pipe-delimited table (header + rows), wrap it as an A2UI
// ```ui table block so the UI renders it as a real table. Otherwise return as-is.
export function wrapUiTable(text: string): string {
  const rows = text.split("\n").filter((l) => l.includes("|")).map((l) => l.split("|").map((c) => c.trim()));
  if (rows.length < 2) return text;
  const columns = rows[0];
  const body = rows.slice(1).filter((r) => r.length === columns.length);
  if (!body.length) return text;
  return "```ui\n" + JSON.stringify([{ type: "table", columns, rows: body }]) + "\n```";
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
// dateTool/statsTool/unitTool/jsonExtractTool are pure (no browser APIs) and are
// reused by the server-side agent runner (src/lib/serverAgent.ts).
export function dateTool(input: string): string {
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
export function statsTool(input: string): string {
  const nums = (input.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
  if (!nums.length) return "Error: provide a list of numbers.";
  const n = nums.length, sum = nums.reduce((a, b) => a + b, 0), mean = sum / n;
  const s = [...nums].sort((a, b) => a - b); const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  const std = Math.sqrt(nums.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  return `count=${n} sum=${sum} mean=${mean.toFixed(3)} median=${median} min=${Math.min(...nums)} max=${Math.max(...nums)} stdev=${std.toFixed(3)}`;
}
export function unitTool(input: string): string {
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
export function jsonExtractTool(input: string): string {
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

// Native tools that run server-side in the web service (free-tier friendly, no MCP/NAT).
async function nativeTool(tool: string, input: string): Promise<string> {
  try {
    const r = await fetch("/api/agent/native", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ tool, input }) });
    const j = await r.json();
    if (!r.ok) return "Error: " + (j.error || "failed");
    return j.text || "(no result)";
  } catch (e) { return "Error: " + (e as Error).message; }
}
// POST a request to the hosted-MCP proxy (/api/agent/mcp) — free-tier OK.
async function mcpPost(body: object): Promise<string> {
  try {
    const r = await fetch("/api/agent/mcp", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) return "Error: " + (j.error || "failed");
    return j.text || "(no result)";
  } catch (e) { return "Error: " + (e as Error).message; }
}
// In-memory cache: server → real tool names (populated on first use, avoids re-listing)
const mcpToolCache: Record<string, string[]> = {};

function fixCommonArgMismatches(toolName: string, args: Record<string, unknown>): Record<string, unknown> {
  // Exa web_fetch_exa expects urls[] not url string
  if ("url" in args && !("urls" in args) && typeof args.url === "string") {
    const { url, ...rest } = args;
    return { ...rest, urls: [url] };
  }
  return args;
}

function fuzzyMatchTool(requested: string, available: string[]): string | null {
  // Exact match first
  if (available.includes(requested)) return requested;
  const req = requested.toLowerCase();
  // Contains check (e.g. "search" → "web_search_exa")
  const contains = available.find((t) => t.toLowerCase().includes(req) || req.includes(t.toLowerCase().replace(/^.*_/, "")));
  if (contains) return contains;
  // Suffix match (e.g. "fetch" → "web_fetch_exa")
  const suffix = available.find((t) => t.toLowerCase().endsWith(req) || t.toLowerCase().includes("_" + req));
  return suffix || null;
}

export async function mcpTool(server: string, input: string): Promise<string> {
  const s = (input || "").trim();
  if (!s || /^list$/i.test(s)) {
    const result = await mcpPost({ server, action: "list" });
    // Cache the tool names for future calls
    const names = [...result.matchAll(/^- (\S+?):/gm)].map((m) => m[1]);
    if (names.length) mcpToolCache[server] = names;
    return result;
  }
  if (s.startsWith("{")) {
    try {
      const p = JSON.parse(s) as { tool?: string; args?: Record<string, unknown> };
      if (p.tool === "list" || p.tool === "tools/list") {
        const result = await mcpPost({ server, action: "list" });
        const names = [...result.matchAll(/^- (\S+?):/gm)].map((m) => m[1]);
        if (names.length) mcpToolCache[server] = names;
        return result;
      }
      const args = p.args ? fixCommonArgMismatches(p.tool || "", p.args) : p.args;
      const result = await mcpPost({ server, action: "call", tool: p.tool, args });
      // Auto-retry: if tool not found, discover real names and retry with best match
      if (result.includes("-32602") && result.toLowerCase().includes("not found") && p.tool) {
        // Fetch and cache tool list if not already cached
        if (!mcpToolCache[server]) {
          const listResult = await mcpPost({ server, action: "list" });
          const names = [...listResult.matchAll(/^- (\S+?):/gm)].map((m) => m[1]);
          if (names.length) mcpToolCache[server] = names;
          else return `${result}\n\nAvailable tools:\n${listResult}`;
        }
        const bestMatch = fuzzyMatchTool(p.tool, mcpToolCache[server]);
        if (bestMatch && bestMatch !== p.tool) {
          const retryArgs = args ? fixCommonArgMismatches(bestMatch, args) : args;
          const retryResult = await mcpPost({ server, action: "call", tool: bestMatch, args: retryArgs });
          return retryResult;
        }
        return `${result}\n\nAvailable tools on "${server}": ${mcpToolCache[server].join(", ")}`;
      }
      return result;
    }
    catch { return 'Error: input must be "list" or JSON like {"tool":"search_repositories","args":{"query":"nextjs"}}.'; }
  }
  return mcpPost({ server, action: "call", tool: s });
}
// Generic tool over ANY connected hosted MCP server (DeepWiki, Context7, HF, …).
async function mcpAnyTool(input: string): Promise<string> {
  const s = (input || "").trim();
  if (!s || /^servers?$/i.test(s)) return mcpPost({ action: "servers" });
  if (s.startsWith("{")) {
    try { const p = JSON.parse(s) as { server?: string; action?: string; tool?: string; args?: unknown }; return mcpPost({ server: p.server, action: p.action || (p.tool ? "call" : "list"), tool: p.tool, args: p.args }); }
    catch { return 'Error: input must be "servers", "<server> list", or JSON like {"server":"deepwiki","tool":"ask_question","args":{…}}.'; }
  }
  const m = s.match(/^(\S+)(?:\s+list)?$/i); // "<server>" or "<server> list"
  if (m) return mcpPost({ server: m[1], action: "list" });
  return 'Use: "servers", "<server> list", or JSON {"server":"…","tool":"…","args":{…}}.';
}
// Cross-turn memory in the browser (localStorage). Free, no backend.
function memoryTool(input: string): string {
  if (typeof localStorage === "undefined") return "Memory unavailable here.";
  const NS = "agent_mem_"; const s = (input || "").trim();
  const all = () => { const o: Record<string, string> = {}; for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i)!; if (k.startsWith(NS)) o[k.slice(NS.length)] = localStorage.getItem(k) || ""; } return o; };
  const setM = s.match(/^set\s+([^:=]+)[:=]\s*([\s\S]+)$/i); if (setM) { localStorage.setItem(NS + setM[1].trim(), setM[2].trim()); return `Saved "${setM[1].trim()}".`; }
  const getM = s.match(/^get\s+(.+)$/i); if (getM) { const v = localStorage.getItem(NS + getM[1].trim()); return v != null ? `${getM[1].trim()} = ${v}` : `No memory for "${getM[1].trim()}".`; }
  const delM = s.match(/^(?:delete|del|forget)\s+(.+)$/i); if (delM) { localStorage.removeItem(NS + delM[1].trim()); return `Forgot "${delM[1].trim()}".`; }
  if (/^list\b/i.test(s) || !s) { const o = all(); const ks = Object.keys(o); return ks.length ? ks.map((k) => `${k} = ${o[k]}`).join("\n") : "Memory is empty."; }
  return 'Use: "set key: value", "get key", "list", or "delete key".';
}

async function ragTool(input: string, ctx: ToolCtx): Promise<string> {
  if (!ctx.selectedRagId) {
    return "Error: No RAG model is connected to the RAG tool. Click on the RAG tool node and select a deployed RAG model.";
  }
  try {
    const res = await fetch("/api/rag/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ projectId: ctx.selectedRagId, query: input }),
    });
    const j = await res.json();
    if (!res.ok) return "Error: " + (j.error || "RAG query failed");
    return j.answer || "(no response)";
  } catch (e) {
    return "Error: " + (e as Error).message;
  }
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
  { id: "web_search", name: "web_search", desc: "search the web (DuckDuckGo) — returns the top results with links", example: "latest mars rover mission", run: async (input) => nativeTool("web_search", input) },
  { id: "wikipedia", name: "wikipedia", desc: "search Wikipedia and read article summaries", example: "retrieval augmented generation", run: async (input) => nativeTool("wikipedia", input) },
  { id: "arxiv", name: "arxiv", desc: "search arXiv research papers (title, abstract, link)", example: "transformer attention mechanism", run: async (input) => nativeTool("arxiv", input) },
  { id: "memory", name: "memory", desc: 'remember facts across the chat — "set key: value", "get key", "list", or "delete key"', example: "set user_name: Aravindhan", run: async (input) => memoryTool(input) },
  { id: "db_schema", name: "db_schema", desc: "list the tables & columns of your connected database (no input needed)", example: "list tables", run: async (_, ctx) => {
    if (ctx?.dbCustomSchema && ctx.dbCustomSchema.trim()) {
      return ctx.dbCustomSchema.trim();
    }
    if (ctx?.dbTable && ctx.dbTable.rows.length > 0) {
      const name = ctx.dbTableName || "students";
      const cols = ctx.dbTable.cols.join(", ");
      return `Tables available: ${name}(${cols}), students(${cols}), orders(${cols}), raw(${cols})\n(${ctx.dbTable.rows.length} rows loaded locally)`;
    }
    return nativeTool("db_schema", "");
  } },
  { id: "db_query", name: "db_query", desc: "run SQL against your connected database and read the rows back", example: "select count(*) from orders", run: async (input, ctx) => {
    if (ctx?.dbTable && ctx.dbTable.rows.length > 0) {
      try {
        const { runSql } = await import("./sqlEngine");
        const name = ctx.dbTableName || "students";
        const extraNames = Array.from(new Set([name, "students", "orders", "customers", "events", "employees", "products", "sensors", "data", "raw"]));
        const extraTables = extraNames.map((n) => ({ name: n, table: ctx.dbTable! }));
        const res = await runSql(input, ctx.dbTable, null, extraTables);
        if (!res.rows.length) return "OK — 0 rows returned.";
        const header = res.cols.join(" | ");
        const body = res.rows.map((r) => res.cols.map((c) => (r[c] == null ? "null" : String(r[c]))).join(" | ")).join("\n");
        const out = `${header}\n${body}\n(${res.rows.length} rows)`;
        return ctx?.a2ui ? wrapUiTable(out) : out;
      } catch (e) {
        return "Error executing SQL on uploaded dataset: " + (e as Error).message;
      }
    }
    const out = await nativeTool("db_query", input);
    return ctx?.a2ui ? wrapUiTable(out) : out;
  } },
  { id: "github", name: "github", desc: 'use your connected GitHub MCP server — input "list" to see its tools, or JSON {"tool":"…","args":{…}} to call one', example: '{"tool":"search_repositories","args":{"query":"nextjs stars:>1000"}}', run: async (input) => mcpTool("github", input) },
  { id: "mcp", name: "mcp", desc: 'use any hosted MCP server you connected (DeepWiki, Context7, Hugging Face, Semgrep, …) — "servers" lists them, "<server> list" shows its tools, JSON {"server":"…","tool":"…","args":{…}} calls one', example: '{"server":"deepwiki","tool":"ask_question","args":{"repoName":"vercel/next.js","question":"what is the app router"}}', run: async (input) => mcpAnyTool(input) },
  { id: "rag", name: "rag", desc: "query a deployed RAG model for grounded answers", example: "what is the product return policy?", run: async (input, ctx) => ragTool(input, ctx) },
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
export function formatFinalAnswer(text: string): string {
  if (!text) return "";
  const formatted = text
    .replace(/^#{1,6}\s*(.*)$/gm, "// $1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/(?<!\S)\*(.*?)\*(?!\S)/g, "$1")
    .replace(/^\s*[\*\-]\s+/gm, "• ")
    .replace(/^\s*[\$\&\#]\s*/gm, "// ");

  const lines = formatted.split("\n").map((line) => {
    const trimmed = line.trim();
    if (trimmed.endsWith(":") && !trimmed.startsWith("//") && !trimmed.startsWith("http") && !trimmed.startsWith("•")) {
      return `// ${trimmed.slice(0, -1)}`;
    }
    return line;
  });

  return lines.join("\n");
}

export function reactSystemPrompt(tools: AgentTool[], goal: string, opts?: { a2ui?: boolean }): string {
  const list = tools.map((t) => `- ${t.name}: ${t.desc} (example input: ${t.example})`).join("\n");
  const a2uiBlock = opts?.a2ui ? `

Rich UI (optional): when the Final Answer is clearly structured data (rows of records or a set of metrics), you MAY return it as a single fenced \`\`\`ui block containing a JSON ARRAY of components. Otherwise use plain text. Supported components:
{"type":"table","columns":["Name","GPA"],"rows":[["Kavya Reddy",3.98],["Ananya Verma",3.92]]}
{"type":"stats","items":[{"label":"Rows","value":42}]}
{"type":"heading","text":"Top 5 students"}   {"type":"callout","tone":"info","text":"..."}   {"type":"text","text":"..."}
Rules for the \`\`\`ui block: default rows-of-data to a "table" (NOT a chart). Put ONLY the real data in rows — no separator lines, no "//" prefixes, no markdown. Emit the block as a JSON array and nothing else.` : "";
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

Example of a tool call block:
Thought: I need to query the exa tool to search.
Action: exa
Action Input: {"tool":"ask_question","args":{"question":"what is artificial intelligence"}}

Formatting rules for Final Answer:
- Output clean line-by-line plain text ready to copy-paste directly.
- Use // at the start of section headers, command lines, or step labels (e.g. // Summary, // Command, // Step 1).
- Do NOT use markdown symbols like *, #, $, &, or bold markdown markup.
- Keep output clear, clean, and line-exact.

Rules: PREFER TOOLS over doing the work yourself. If a tool can compute, look up, or fetch something — arithmetic, dates, web pages, the knowledge base — you MUST call that tool instead of answering from memory (you are unreliable at mental math and date arithmetic). Handle one thing per step. Only give the Final Answer once the tools have given you everything you need. After each Observation, continue the loop. Never write "Observation:" yourself — the system provides it. Keep each Thought to one sentence.${a2uiBlock}`;
}

export interface ReActParse { thought?: string; action?: string; input?: string; final?: string; }
export function parseReAct(text: string): ReActParse {
  const cleanText = text.trim();
  if (cleanText.startsWith("{") && cleanText.endsWith("}")) {
    try {
      const j = JSON.parse(cleanText);
      if (j && typeof j === "object") {
        const act = j.action || j.tool;
        const inp = j.input || j.args || j.action_input;
        if (act && typeof act === "string") {
          return {
            thought: j.thought || "(JSON tool call fallback)",
            action: act,
            input: typeof inp === "object" ? JSON.stringify(inp) : String(inp || "")
          };
        }
      }
    } catch { /* ignore */ }
  }

  const thought = text.match(/Thought:\s*([\s\S]*?)(?=\n\s*(?:Action|Final Answer)\s*:|$)/i)?.[1]?.trim();
  const final = text.match(/Final Answer:\s*([\s\S]*)/i)?.[1]?.trim();
  if (final) return { thought, final };
  const action = text.match(/Action:\s*([^\n]+)/i)?.[1]?.trim().replace(/[.'"]+$/, "");
  let input = text.match(/Action Input:\s*([\s\S]*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|$)/i)?.[1]?.trim() || "";
  if (input.length >= 2 && ((input.startsWith('"') && input.endsWith('"')) || (input.startsWith("'") && input.endsWith("'")))) {
    input = input.slice(1, -1).trim();
  }
  return { thought, action, input };
}
