import "server-only";
import { randomBytes } from "crypto";
import { db } from "./db";
import { auditLog } from "./db/schema";

// Central error capture. Emits one structured JSON line to stderr so any log
// drain (Vercel, CloudWatch, Loki…) can index it. If you later add Sentry,
// forward from here — this is the single choke point for server errors.
export function captureError(err: unknown, context: Record<string, unknown> = {}): void {
  const e = err as Error;
  try {
    console.error(JSON.stringify({
      level: "error",
      ts: new Date().toISOString(),
      msg: e?.message || String(err),
      name: e?.name,
      stack: e?.stack?.split("\n").slice(0, 4).join(" | "),
      ...context,
    }));
  } catch {
    console.error("captureError failed", err);
  }
}

// Fire-and-forget audit event for monitoring (logins, quota hits, admin actions).
// Never throws — a failed audit write must not break the request.
export async function audit(event: string, userId: string | null, detail?: Record<string, unknown>): Promise<void> {
  try {
    await db.insert(auditLog).values({ id: randomBytes(16).toString("hex"), event, userId: userId ?? null, detail: detail ?? null });
  } catch (e) {
    captureError(e, { where: "audit", event });
  }
}
