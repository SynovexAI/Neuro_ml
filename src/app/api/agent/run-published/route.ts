import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import { rateLimitDb } from "@/lib/ratelimit";
import { runPublishedAgent } from "@/lib/runPublished";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Runs one of the user's own published agents (NAT or in-browser) for the
// Workroom chat. Dispatches by runtime — see runPublishedAgent.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("natrun", user.id, 20, 60_000))) return NextResponse.json({ error: "Too many runs — wait a moment." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const projectId = String(b.projectId || "");
  const [proj] = await db.select().from(projects).where(and(eq(projects.id, projectId), eq(projects.userId, user.id)));
  if (!proj || !proj.published) return NextResponse.json({ error: "Agent not found." }, { status: 404 });

  const r = await runPublishedAgent({ userId: user.id, config: proj.config, task: String(b.task || ""), agentName: proj.name });
  if (!r.ok) return NextResponse.json({ error: r.error }, { status: r.status });
  return NextResponse.json({ answer: r.answer, latency_ms: r.latency_ms, tool_names: r.tool_names, usage: r.usage });
}
