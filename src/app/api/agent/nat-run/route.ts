import { NextResponse } from "next/server";
import { getSessionUser, uid } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { rateLimitDb } from "@/lib/ratelimit";
import { checkQuota, recordUsage, estimateTokens } from "@/lib/usage";
import { captureError } from "@/lib/monitor";
import { db } from "@/lib/db";
import { agentRuns, mcpServers, knowledgeBases, kbChunks } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";
import { retrieve } from "@/lib/kb";
import { and, eq, inArray } from "drizzle-orm";

// Retrieve the most relevant chunks from the user's selected knowledge bases.
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

// Build the resolved MCP configs the NAT service expects (secrets decrypted here,
// headers/env prepared, never sent to the client).
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

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Proxies an agent run to the NAT sidecar (Python service). The provider key is
// decrypted here and passed server-to-server only — never exposed to the client.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("natrun", user.id, 20, 60_000))) return NextResponse.json({ error: "Too many runs — wait a moment." }, { status: 429 });

  const quota = await checkQuota(user);
  if (!quota.ok) return NextResponse.json({ error: `Monthly token limit reached (${quota.used.toLocaleString()} / ${quota.limit.toLocaleString()}). Ask an admin to raise your quota.` }, { status: 429 });

  const serviceUrl = process.env.NAT_SERVICE_URL;
  if (!serviceUrl) return NextResponse.json({ error: "NAT runtime isn't configured on the server (set NAT_SERVICE_URL)." }, { status: 503 });

  const b = await req.json().catch(() => ({}));
  const task = String(b.task || "").trim();
  if (!task) return NextResponse.json({ error: "A task is required." }, { status: 400 });
  const tools: string[] = Array.isArray(b.tools) ? b.tools.filter((t: unknown) => typeof t === "string") : [];
  const systemPrompt = b.systemPrompt ? String(b.systemPrompt) : undefined;
  const mcpIds: string[] = Array.isArray(b.mcpServerIds) ? b.mcpServerIds.filter((t: unknown) => typeof t === "string") : [];
  const kbIds: string[] = Array.isArray(b.knowledgeBaseIds) ? b.knowledgeBaseIds.filter((t: unknown) => typeof t === "string") : [];
  const mcp_servers = await resolveMcp(mcpIds, user.id).catch(() => []);

  const prov = b.providerId ? await getProviderById(String(b.providerId)) : await getActiveProvider();
  if (!prov || !prov.baseUrl) return NextResponse.json({ error: "No LLM provider is configured. An admin must add one under Admin → Providers." }, { status: 400 });
  const model = b.model ? String(b.model) : prov.model;
  if (!model) return NextResponse.json({ error: "No model selected for this provider." }, { status: 400 });

  const docs = kbIds.length ? await resolveKbDocs(user.id, kbIds, task, prov).catch(() => []) : [];

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 90_000);
  try {
    const res = await fetch(serviceUrl.replace(/\/$/, "") + "/run", {
      method: "POST",
      headers: { "content-type": "application/json", ...(process.env.NAT_SHARED_SECRET ? { "x-nat-secret": process.env.NAT_SHARED_SECRET } : {}) },
      body: JSON.stringify({ task, model, base_url: prov.baseUrl, api_key: prov.apiKey, temperature: b.temperature ?? 0, system_prompt: systemPrompt, agent_type: b.agentType === "tool_calling_agent" ? "tool_calling_agent" : "react_agent", tools, mcp_servers, ...(docs.length ? { knowledge: { docs } } : {}) }),
      signal: ctrl.signal,
    });
    const j = await res.json().catch(() => null) as { answer?: string; latency_ms?: number; tool_names?: string[]; unsupported_tools?: string[]; profiler?: Record<string, unknown>; detail?: string } | null;
    if (!res.ok) return NextResponse.json({ error: j?.detail || `NAT service error ${res.status}` }, { status: 502 });

    // NAT doesn't return token counts yet — meter an estimate so cost accounting stays unified.
    const pt = estimateTokens(task + (systemPrompt || "")), ct = estimateTokens(j?.answer || "");
    void recordUsage({ userId: user.id, lab: "agent-nat", model, promptTokens: pt, completionTokens: ct, estimated: true });
    // Log the run for the agent analytics dashboard.
    db.insert(agentRuns).values({
      id: uid(), userId: user.id, agentName: "NAT agent", agentType: "nat", runtime: "nat",
      provider: prov.provider, model, iterations: 0,
      toolCalls: (j?.tool_names || []).map((t) => ({ tool: t, count: 1 })), toolCallCount: (j?.tool_names || []).length,
      promptTokens: pt, completionTokens: ct, totalTokens: pt + ct,
      latencyMs: Number(j?.latency_ms || 0), outcome: "success",
    }).catch((e) => captureError(e, { where: "nat-run.log" }));

    return NextResponse.json({ ...j, usage: { total_tokens: pt + ct } });
  } catch (e) {
    if ((e as Error).name === "AbortError") return NextResponse.json({ error: "The agent run timed out." }, { status: 504 });
    captureError(e, { where: "nat-run", userId: user.id });
    return NextResponse.json({ error: `Could not reach the NAT service: ${(e as Error).message}` }, { status: 502 });
  } finally {
    clearTimeout(timer);
  }
}
