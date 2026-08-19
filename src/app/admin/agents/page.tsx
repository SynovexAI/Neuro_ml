import { redirect } from "next/navigation";
import { desc } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { agentRuns } from "@/lib/db/schema";
import Shell from "@/components/Shell";
import AgentAnalytics from "@/components/AgentAnalytics";

export const dynamic = "force-dynamic";
const iso = (v: unknown) => v instanceof Date ? v.toISOString() : (v ? String(v) : null);

export default async function AgentAnalyticsPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  let rows: (typeof agentRuns.$inferSelect)[] = [];
  try { rows = await db.select().from(agentRuns).orderBy(desc(agentRuns.ts)).limit(300); }
  catch { rows = []; } // table not migrated yet

  const runs = rows.map((r) => ({
    id: r.id, agentType: r.agentType, runtime: r.runtime, model: r.model || "—", provider: r.provider,
    iterations: r.iterations, toolCalls: (r.toolCalls as { tool: string; count: number }[] | null) || [],
    toolCallCount: r.toolCallCount, totalTokens: r.totalTokens, latencyMs: r.latencyMs,
    outcome: r.outcome || "unknown", ts: iso(r.ts),
  }));

  return (
    <Shell user={user} title="Admin · Agent analytics">
      <div className="eyebrow">Admin</div>
      <h2 className="page-h">Agent analytics</h2>
      <p className="page-sub">Every agent run — in-browser and NAT — with success rate, tool usage, tokens, estimated cost, and latency.</p>
      <AgentAnalytics runs={runs} />
    </Shell>
  );
}
