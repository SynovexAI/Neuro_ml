import { NextResponse } from "next/server";
import { and, eq, desc } from "drizzle-orm";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const dynamic = "force-dynamic";

// Published agents the current user can use in the Workroom.
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(projects)
    .where(and(eq(projects.userId, user.id), eq(projects.lab, "agent-nat"), eq(projects.published, true)))
    .orderBy(desc(projects.updatedAt));
  // Return only what the Workroom needs — never leak nothing sensitive is here,
  // but keep the payload small (config summary, not full config).
  const agents = rows.map((r) => {
    const c = (r.config || {}) as Record<string, unknown>;
    return {
      id: r.id,
      name: r.name,
      model: typeof c.model === "string" ? c.model : "",
      agentType: typeof c.agentType === "string" ? c.agentType : "react_agent",
      toolCount: Array.isArray(c.tools) ? c.tools.length : 0,
      kbCount: Array.isArray(c.kbIds) ? (c.kbIds as unknown[]).length : 0,
      updatedAt: r.updatedAt,
    };
  });
  return NextResponse.json({ agents });
}
