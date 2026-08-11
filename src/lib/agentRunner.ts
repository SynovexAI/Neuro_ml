import { and, eq, inArray } from "drizzle-orm";
import { db } from "@/lib/db";
import { agentRuns, mcpServers, knowledgeBases, kbChunks, users } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";
import { retrieve } from "@/lib/kb";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { checkQuota, recordUsage, estimateTokens } from "@/lib/usage";
import { captureError } from "@/lib/monitor";
import { uid } from "@/lib/auth";

export type RunInput = {
  userId: string;
  task: string;
  providerId?: string;
  model?: string;
  systemPrompt?: string;
  agentType?: string;
  tools?: string[];
  temperature?: number;
  mcpServerIds?: string[];
  knowledgeBaseIds?: string[];
  agentName?: string;
  enforceQuota?: boolean; // default true
};

export type RunResult = {
  ok: boolean;
  status: number;
  answer?: string;
  error?: string;
  latency_ms?: number;
  tool_names?: string[];
  unsupported_tools?: string[];
  profiler?: Record<string, unknown>;
  context_used?: boolean;
  usage?: { total_tokens: number };
};

async function resolveKbDocs(userId: string, kbIds: string[], query: string, prov: { baseUrl: string; apiKey: string }): Promise<string[]> {
  const out: string[] = [];
  for (const kbId of kbIds.slice(0, 3)) {
    const [kb] = await db.select().from(knowledgeBases).where(and(eq(knowledgeBases.id, kbId), eq(knowledgeBases.userId, userId)));
    if (!kb || kb.status !== "ready") continue;
    const chunks = await db.select({ text: kbChunks.text, embedding: kbChunks.embedding }).from(kbChunks).where(eq(kbChunks.kbId, kbId));
    out.push(...await retrieve(kb, chunks, query, prov, 4));
  }
  return out;
}

async function resolveMcp(ids: string[], userId: string) {
  if (!ids.length) return [];
  const rows = await db.select().from(mcpServers).where(and(inArray(mcpServers.id, ids), eq(mcpServers.enabled, true), eq(mcpServers.userId, userId)));
  return rows.map((s) => {
    const secret = s.secretEnc ? decrypt(s.secretEnc) : "";
    const cfg: Record<string, unknown> = { name: s.name, transport: s.transport === "http" ? "streamable-http" : s.transport };
    if (s.transport === "stdio") {
      const parts = (s.command || "").split(/\s+/).filter(Boolean);
      cfg.command = parts[0] || "";
      cfg.args = parts.slice(1);
      if (s.envName && secret) cfg.env = { [s.envName]: secret };
    } else {
      cfg.url = s.url || "";
      if (secret) {
        if (s.authType === "bearer") cfg.headers = { Authorization: `Bearer ${secret}` };
        else if (s.authType === "apikey") cfg.headers = { [s.headerName || "X-API-Key"]: secret };
      }
    }
    return cfg;
  });
}

// Runs one agent turn through the NAT sidecar. Shared by the interactive API
// route, the Workroom chat, and channel webhooks (Telegram etc.). Decrypts the
// provider key server-side only. Caller handles auth / rate-limiting.
export async function runAgent(inp: RunInput): Promise<RunResult> {
  const task = (inp.task || "").trim();
  if (!task) return { ok: false, status: 400, error: "A task is required." };

  const serviceUrl = process.env.NAT_SERVICE_URL;
  if (!serviceUrl) return { ok: false, status: 503, error: "NAT runtime isn't configured on the server (set NAT_SERVICE_URL)." };

  if (inp.enforceQuota !== false) {
    const [u] = await db.select().from(users).where(eq(users.id, inp.userId));
    if (u) { const q = await checkQuota(u); if (!q.ok) return { ok: false, status: 429, error: `Monthly token limit reached (${q.used.toLocaleString()} / ${q.limit.toLocaleString()}).` }; }
  }

  let prov = inp.providerId ? await getProviderById(inp.providerId, inp.userId) : await getActiveProvider(inp.userId);
  if (!prov) prov = await getActiveProvider(inp.userId);
  if (!prov || !prov.baseUrl) return { ok: false, status: 400, error: "No LLM provider is configured." };
  const model = inp.model || prov.model;
  if (!model) return { ok: false, status: 400, error: "No model selected for this provider." };

  const tools = (inp.tools || []).filter((t) => typeof t === "string");
  const mcpIds = (inp.mcpServerIds || []).filter((t) => typeof t === "string");
  const kbIds = (inp.knowledgeBaseIds || []).filter((t) => typeof t === "string");
  const mcp_servers = await resolveMcp(mcpIds, inp.userId).catch(() => []);
  const docs = kbIds.length ? await resolveKbDocs(inp.userId, kbIds, task, prov).catch(() => []) : [];

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 90_000);
  try {
    const res = await fetch(serviceUrl.replace(/\/$/, "") + "/run", {
      method: "POST",
      headers: { "content-type": "application/json", ...(process.env.NAT_SHARED_SECRET ? { "x-nat-secret": process.env.NAT_SHARED_SECRET } : {}) },
      body: JSON.stringify({ task, model, base_url: prov.baseUrl, api_key: prov.apiKey, temperature: inp.temperature ?? 0, system_prompt: inp.systemPrompt, agent_type: inp.agentType === "tool_calling_agent" ? "tool_calling_agent" : "react_agent", tools, mcp_servers, ...(docs.length ? { knowledge: { docs } } : {}) }),
      signal: ctrl.signal,
    });
    const j = await res.json().catch(() => null) as { answer?: string; latency_ms?: number; tool_names?: string[]; unsupported_tools?: string[]; profiler?: Record<string, unknown>; context_used?: boolean; detail?: string } | null;
    if (!res.ok) return { ok: false, status: 502, error: j?.detail || `NAT service error ${res.status}` };

    // Prefer real per-step token usage from NAT's profiler (covers every LLM call
    // in the loop + tool round-trips), not just task+answer. Fall back to an estimate.
    const steps = (j?.profiler as { steps?: { tokens?: number }[] } | undefined)?.steps || [];
    const profTokens = steps.reduce((a, s) => a + (Number(s?.tokens) || 0), 0);
    const estAnswer = estimateTokens(j?.answer || "");
    const measured = profTokens > 0;
    const ct = measured ? Math.min(estAnswer, profTokens) : estAnswer;
    const pt = measured ? Math.max(0, profTokens - ct) : estimateTokens(task + (inp.systemPrompt || ""));
    const totalTokens = pt + ct;
    void recordUsage({ userId: inp.userId, lab: "agent-nat", model, promptTokens: pt, completionTokens: ct, estimated: !measured });
    db.insert(agentRuns).values({
      id: uid(), userId: inp.userId, agentName: inp.agentName || "NAT agent", agentType: "nat", runtime: "nat",
      provider: prov.provider, model, iterations: steps.length,
      toolCalls: (j?.tool_names || []).map((t) => ({ tool: t, count: 1 })), toolCallCount: (j?.tool_names || []).length,
      promptTokens: pt, completionTokens: ct, totalTokens,
      latencyMs: Number(j?.latency_ms || 0), outcome: "success",
    }).catch((e) => captureError(e, { where: "agentRunner.log" }));

    return { ok: true, status: 200, answer: j?.answer, latency_ms: j?.latency_ms, tool_names: j?.tool_names, unsupported_tools: j?.unsupported_tools, profiler: j?.profiler, context_used: j?.context_used, usage: { total_tokens: pt + ct } };
  } catch (e) {
    if ((e as Error).name === "AbortError") return { ok: false, status: 504, error: "The agent run timed out." };
    captureError(e, { where: "agentRunner", userId: inp.userId });
    return { ok: false, status: 502, error: `Could not reach the NAT service: ${(e as Error).message}` };
  } finally {
    clearTimeout(timer);
  }
}
