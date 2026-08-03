import { NextResponse } from "next/server";
import { gte, sql } from "drizzle-orm";
import { db } from "@/lib/db";
import { users, usage } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { DEFAULT_MONTHLY_TOKEN_LIMIT } from "@/lib/usage";

export async function GET() {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const rows = await db.select({
    id: users.id, email: users.email, name: users.name,
    role: users.role, status: users.status,
    monthlyTokenLimit: users.monthlyTokenLimit, createdAt: users.createdAt,
  }).from(users).orderBy(sql`${users.createdAt} DESC`).limit(1000);

  // Current-month token usage per user (UTC month).
  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  let byUser = new Map<string, number>();
  try {
    const agg = await db.select({ userId: usage.userId, t: sql<number>`COALESCE(SUM(${usage.totalTokens}), 0)` })
      .from(usage).where(gte(usage.ts, monthStart)).groupBy(usage.userId);
    byUser = new Map(agg.map((r) => [r.userId, Number(r.t)]));
  } catch { /* usage table not migrated yet — report zeros */ }

  const withUsage = rows.map((r) => ({ ...r, monthUsage: byUser.get(r.id) ?? 0 }));
  return NextResponse.json({ users: withUsage, defaultLimit: DEFAULT_MONTHLY_TOKEN_LIMIT });
}
