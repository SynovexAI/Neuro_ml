import { NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { etlRuns } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const num = (v: unknown, cap = 100_000_000) => Math.max(0, Math.min(cap, Math.floor(Number(v) || 0)));
const str = (v: unknown, n: number) => (typeof v === "string" ? v.slice(0, n) : null);

// GET → the current user's recent ETL runs (newest first).
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(etlRuns).where(eq(etlRuns.userId, user.id)).orderBy(desc(etlRuns.ts)).limit(50);
  return NextResponse.json({ runs: rows });
}

// POST → record one run (fire-and-forget from the lab after a load/execute).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("etlrun", user.id, 120, 60_000))) return NextResponse.json({ error: "rate limited" }, { status: 429 });
  const b = await req.json().catch(() => ({}));
  await db.insert(etlRuns).values({
    id: uid(), userId: user.id,
    name: str(b.name, 160), mode: str(b.mode, 24), target: str(b.target, 200),
    rowsIn: num(b.rowsIn), rowsOut: num(b.rowsOut), rowsLoaded: num(b.rowsLoaded),
    durationMs: num(b.durationMs, 3_600_000),
    status: b.status === "error" ? "error" : "ok", error: str(b.error, 300),
  });
  return NextResponse.json({ ok: true });
}
