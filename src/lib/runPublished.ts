import "server-only";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { users, agentRuns } from "@/lib/db/schema";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { checkQuota, recordUsage } from "@/lib/usage";
import { captureError } from "@/lib/monitor";
import { uid } from "@/lib/auth";
import { runAgent, type RunResult } from "@/lib/agentRunner";
import { cfgToRunInput } from "@/lib/channels";
import { runReactServer, runWorkflowServer } from "@/lib/serverAgent";

/* eslint-disable @typescript-eslint/no-explicit-any */

// Runs any published agent — NAT (via the sidecar) or an in-browser ReAct/workflow
// agent (server-side loop). Single entry point for the Workroom chat and every
// channel (Telegram / API / widget), so both runtimes work everywhere.
export async function runPublishedAgent(opts: { userId: string; config: unknown; task: string; agentName?: string }): Promise<RunResult> {
  const c = (opts.config || {}) as any;
  const task = (opts.task || "").trim();
  if (!task) return { ok: false, status: 400, error: "A message is required." };

  // NAT runtime → existing sidecar path (handles its own quota/usage/logging).
  if (c.runtime === "nat") {
    return runAgent({ ...cfgToRunInput(c), userId: opts.userId, task, agentName: opts.agentName });
  }

  // In-browser runtime (ReAct / workflow) → run the loop server-side.
  const [u] = await db.select().from(users).where(eq(users.id, opts.userId));
  if (u) { const q = await checkQuota(u); if (!q.ok) return { ok: false, status: 429, error: `Monthly token limit reached (${q.used.toLocaleString()} / ${q.limit.toLocaleString()}).` }; }

  const prov = c.provider ? await getProviderById(String(c.provider)) : await getActiveProvider();
  if (!prov || !prov.baseUrl) return { ok: false, status: 400, error: "No LLM provider is configured." };
  const model = c.model || prov.model;
  if (!model) return { ok: false, status: 400, error: "No model selected for this agent." };

  const start = Date.now();
  try {
    const r = c.type === "workflow"
      ? await runWorkflowServer(c, task, prov, model)
      : await runReactServer(c, task, prov, model);
    const total = r.promptTokens + r.completionTokens;
    void recordUsage({ userId: opts.userId, lab: "agent", model, promptTokens: r.promptTokens, completionTokens: r.completionTokens, estimated: true });
    db.insert(agentRuns).values({
      id: uid(), userId: opts.userId, agentName: opts.agentName || c.name || "agent",
      agentType: c.type === "workflow" ? "workflow" : "react", runtime: "browser",
      provider: prov.provider, model, iterations: r.iterations,
      toolCalls: Object.entries(r.toolCounts).map(([tool, count]) => ({ tool, count })), toolCallCount: r.toolCallCount,
      promptTokens: r.promptTokens, completionTokens: r.completionTokens, totalTokens: total,
      latencyMs: Date.now() - start, outcome: r.outcome,
    }).catch((e) => captureError(e, { where: "runPublished.log" }));
    return { ok: true, status: 200, answer: r.answer, latency_ms: Date.now() - start, tool_names: Object.keys(r.toolCounts), usage: { total_tokens: total } };
  } catch (e) {
    captureError(e, { where: "runPublished", userId: opts.userId });
    return { ok: false, status: 502, error: `Agent run failed: ${(e as Error).message}` };
  }
}
