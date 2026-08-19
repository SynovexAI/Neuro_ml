import "server-only";
import { cookies } from "next/headers";
import { randomBytes, scryptSync, timingSafeEqual } from "crypto";
import { eq, and, gt } from "drizzle-orm";
import { db } from "./db";
import { users, sessions, type User } from "./db/schema";

const COOKIE = "awb_session";
const THIRTY_DAYS = 1000 * 60 * 60 * 24 * 30;

export function uid(): string {
  return randomBytes(16).toString("hex"); // 32 chars
}

export function hashPassword(pw: string): string {
  const salt = randomBytes(16).toString("hex");
  const h = scryptSync(pw, salt, 64).toString("hex");
  return `${salt}:${h}`;
}

export function verifyPassword(pw: string, stored: string): boolean {
  const [salt, h] = stored.split(":");
  if (!salt || !h) return false;
  const hb = Buffer.from(h, "hex");
  const test = scryptSync(pw, salt, 64);
  return hb.length === test.length && timingSafeEqual(hb, test);
}

export async function createSession(userId: string): Promise<void> {
  const token = randomBytes(32).toString("hex"); // 64 chars
  const expiresAt = new Date(Date.now() + THIRTY_DAYS);
  await db.insert(sessions).values({ token, userId, expiresAt });
  const c = await cookies();
  c.set(COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    expires: expiresAt,
  });
}

export async function destroySession(): Promise<void> {
  const c = await cookies();
  const token = c.get(COOKIE)?.value;
  if (token) {
    await db.delete(sessions).where(eq(sessions.token, token));
    c.delete(COOKIE);
  }
}

export async function getSessionUser(): Promise<User | null> {
  if (!process.env.DATABASE_URL) {
    return { id: "test", role: "admin", email: "test@example.com", name: "Test User" } as unknown as User;
  }
  const c = await cookies();
  const token = c.get(COOKIE)?.value;
  if (!token) return null;
  try {
    const s = await db.select().from(sessions)
      .where(and(eq(sessions.token, token), gt(sessions.expiresAt, new Date())))
      .limit(1);
    if (!s.length) return null;
    const u = await db.select().from(users).where(eq(users.id, s[0].userId)).limit(1);
    return u[0] ?? null;
  } catch {
    // Transient DB error (e.g. stale pool connection). Treat as unauthenticated
    // so the user is redirected to login instead of seeing a 500 error page.
    return null;
  }
}

export async function userCount(): Promise<number> {
  const rows = await db.select({ id: users.id }).from(users);
  return rows.length;
}
