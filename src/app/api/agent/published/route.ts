import { NextResponse } from "next/server";
import { and, eq, desc, inArray, sql } from "drizzle-orm";
import { db } from "@/lib/db";
import { projects, agentRuns } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const dynamic = "force-dynamic";

// Published agents the current user can use in the Workroom — both NAT
// (lab "agent-nat") and in-browser (lab "agent") agents.
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(projects)
    .where(and(eq(projects.userId, user.id), inArray(projects.lab, ["agent-nat", "agent"]), eq(projects.published, true)))
    .orderBy(desc(projects.updatedAt));

  // Per-agent usage from the run log (matched by name — see runPublishedAgent,
  // which logs agentName = project name).
  const statsRows = await db.select({
    name: agentRuns.agentName,
    runs: sql<number>`count(*)`,
    tokens: sql<number>`coalesce(sum(${agentRuns.totalTokens}), 0)`,
    last: sql<string>`max(${agentRuns.ts})`,
  }).from(agentRuns).where(eq(agentRuns.userId, user.id)).groupBy(agentRuns.agentName);
  const byName = new Map(statsRows.map((s) => [s.name, s]));

  // Config summary only (not the full config), and never any secret.
  const agents = rows.map((r) => {
    const st = byName.get(r.name);
    const c = (r.config || {}) as Record<string, unknown>;
    const nat = c.runtime === "nat";
    const kind = nat
      ? (c.agentType === "tool_calling_agent" ? "Tool-calling" : "ReAct")
      : (c.type === "workflow" ? "Workflow" : "ReAct");
    return {
      id: r.id,
      name: r.name,
      runtime: nat ? "nat" : "browser",
      kind,
      model: typeof c.model === "string" ? c.model : "",
      toolCount: Array.isArray(c.tools) ? c.tools.length : 0,
      kbCount: nat ? (Array.isArray(c.kbIds) ? (c.kbIds as unknown[]).length : 0) : (c.knowledge ? 1 : 0),
      runs: Number(st?.runs || 0),
      tokens: Number(st?.tokens || 0),
      lastUsed: st?.last || null,
      updatedAt: r.updatedAt,
      createdAt: r.createdAt,
    };
  });
  return NextResponse.json({ agents });
}
