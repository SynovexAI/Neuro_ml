import { NextResponse } from "next/server";
import { desc, eq, sql } from "drizzle-orm";
import { db } from "@/lib/db";
import { auditLog, users } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// GET /api/admin/audit?event=login&limit=200 → recent audit events (+ the user's email).
export async function GET(req: Request) {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { searchParams } = new URL(req.url);
  const ev = searchParams.get("event");
  const limit = Math.min(1000, Math.max(1, Number(searchParams.get("limit")) || 300));

  const base = db.select({
    id: auditLog.id, event: auditLog.event, userId: auditLog.userId, detail: auditLog.detail, ts: auditLog.ts, email: users.email,
  }).from(auditLog).leftJoin(users, eq(users.id, auditLog.userId));
  const rows = await (ev ? base.where(eq(auditLog.event, ev)) : base).orderBy(desc(auditLog.ts)).limit(limit);

  // Distinct event types for the filter dropdown.
  let events: string[] = [];
  try {
    const evs = await db.select({ event: auditLog.event, n: sql<number>`count(*)` }).from(auditLog).groupBy(auditLog.event).orderBy(desc(sql`count(*)`));
    events = evs.map((e) => e.event);
  } catch { /* ignore */ }

  return NextResponse.json({ rows, events });
}
