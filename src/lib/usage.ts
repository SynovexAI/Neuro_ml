import "server-only";
import { randomBytes } from "crypto";
import { and, eq, gte, sql } from "drizzle-orm";
import { db } from "./db";
import { usage, type User } from "./db/schema";
import { captureError } from "./monitor";

// Platform default monthly token budget per student (admin-overridable per user).
export const DEFAULT_MONTHLY_TOKEN_LIMIT = Number(process.env.DEFAULT_MONTHLY_TOKEN_LIMIT || 10_000_000);

function monthStartUTC(): Date {
  const n = new Date();
  return new Date(Date.UTC(n.getUTCFullYear(), n.getUTCMonth(), 1));
}

// Effective monthly limit. Admins are unlimited; a NULL per-user value falls
// back to the platform default.
export function limitFor(user: Pick<User, "role" | "monthlyTokenLimit">): number {
  if (user.role === "admin") return Infinity;
  return user.monthlyTokenLimit ?? DEFAULT_MONTHLY_TOKEN_LIMIT;
}

export async function monthlyUsage(userId: string): Promise<number> {
  try {
    const [row] = await db
      .select({ t: sql<number>`COALESCE(SUM(${usage.totalTokens}), 0)` })
      .from(usage)
      .where(and(eq(usage.userId, userId), gte(usage.ts, monthStartUTC())));
    return Number(row?.t ?? 0);
  } catch (e) {
    // Fail-open: a metering read error must not lock users out of the product.
    captureError(e, { where: "monthlyUsage", userId });
    return 0;
  }
}

export interface QuotaStatus { ok: boolean; used: number; limit: number; remaining: number; }

export async function checkQuota(user: User): Promise<QuotaStatus> {
  const limit = limitFor(user);
  if (!isFinite(limit)) return { ok: true, used: 0, limit: Infinity, remaining: Infinity };
  const used = await monthlyUsage(user.id);
  return { ok: used < limit, used, limit, remaining: Math.max(0, limit - used) };
}

export async function recordUsage(p: {
  userId: string; lab?: string | null; model?: string | null;
  promptTokens: number; completionTokens: number; estimated: boolean;
}): Promise<void> {
  try {
    const total = (p.promptTokens || 0) + (p.completionTokens || 0);
    await db.insert(usage).values({
      id: randomBytes(16).toString("hex"),
      userId: p.userId, lab: p.lab ?? null, model: p.model ?? null,
      promptTokens: p.promptTokens || 0, completionTokens: p.completionTokens || 0,
      totalTokens: total, estimated: p.estimated,
    });
  } catch (e) {
    captureError(e, { where: "recordUsage", userId: p.userId });
  }
}

// Rough token estimate (~4 chars/token) used only when a provider omits usage.
export function estimateTokens(text: string): number {
  return Math.ceil((text || "").length / 4);
}
