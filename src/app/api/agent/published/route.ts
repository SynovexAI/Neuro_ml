import { NextResponse } from "next/server";
import { and, eq, desc, inArray } from "drizzle-orm";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
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
  // Config summary only (not the full config), and never any secret.
  const agents = rows.map((r) => {
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
      updatedAt: r.updatedAt,
    };
  });
  return NextResponse.json({ agents });
}
