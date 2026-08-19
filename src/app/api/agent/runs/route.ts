import { NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { agentRuns } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const num = (v: unknown, cap = 10_000_000) => Math.max(0, Math.min(cap, Math.floor(Number(v) || 0)));

export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(agentRuns)
    .where(eq(agentRuns.userId, user.id)).orderBy(desc(agentRuns.ts)).limit(50);
  return NextResponse.json({ runs: rows });
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("agentlog", user.id, 120, 60_000))) return NextResponse.json({ error: "rate limited" }, { status: 429 });
  const b = await req.json().catch(() => ({}));
  const toolCalls = Array.isArray(b.toolCalls) ? b.toolCalls.slice(0, 30) : [];
  await db.insert(agentRuns).values({
    id: uid(), userId: user.id,
    agentName: b.agentName ? String(b.agentName).slice(0, 120) : null,
    agentType: b.agentType ? String(b.agentType).slice(0, 24) : null,
    runtime: b.runtime ? String(b.runtime).slice(0, 16) : null,
    provider: b.provider ? String(b.provider).slice(0, 40) : null,
    model: b.model ? String(b.model).slice(0, 120) : null,
    iterations: num(b.iterations, 1000),
    toolCalls,
    toolCallCount: num(b.toolCallCount, 1000),
    promptTokens: num(b.promptTokens),
    completionTokens: num(b.completionTokens),
    totalTokens: num(b.totalTokens),
    latencyMs: num(b.latencyMs, 3_600_000),
    outcome: b.outcome ? String(b.outcome).slice(0, 24) : null,
    errorMsg: b.errorMsg ? String(b.errorMsg).slice(0, 300) : null,
  });
  return NextResponse.json({ ok: true });
}
