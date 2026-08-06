import "server-only";
import { eq, sql } from "drizzle-orm";
import { db } from "./db";
import { rateLimits } from "./db/schema";
import { rateLimit as memRateLimit } from "./net";
import { captureError } from "./monitor";

// Persistent fixed-window rate limiter. Returns true if the request is allowed.
// Unlike the in-memory limiter this survives restarts and is shared across
// serverless instances. Falls back to the in-memory limiter if the DB is
// unreachable (still limits per-instance rather than failing open entirely).
export async function rateLimitDb(scope: string, userId: string, max: number, windowMs: number): Promise<boolean> {
  const now = Date.now();
  const windowStart = Math.floor(now / windowMs) * windowMs;
  const id = `${scope}:${userId}:${windowStart}`;
  try {
    await db.insert(rateLimits).values({ id, scope, windowStart, count: 1 })
      .onDuplicateKeyUpdate({ set: { count: sql`${rateLimits.count} + 1` } });
    const [row] = await db.select({ count: rateLimits.count }).from(rateLimits).where(eq(rateLimits.id, id));
    return (row?.count ?? 1) <= max;
  } catch (e) {
    captureError(e, { where: "rateLimitDb", scope });
    return memRateLimit(`${scope}:${userId}`, max, windowMs);
  }
}
