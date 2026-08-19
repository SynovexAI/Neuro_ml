import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { channels, projects } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { encrypt } from "@/lib/crypto";
import { audit } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Deploy an ETL pipeline as a Bearer-key API. Mints a key (stored encrypted) and,
// for a "full run" (server does extract → transform → load), stores the source /
// target connection URLs ENCRYPTED on the channel (never returned to the client).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  const projectId = String(b.projectId || "");
  const [proj] = await db.select().from(projects).where(and(eq(projects.id, projectId), eq(projects.userId, user.id)));
  if (!proj || proj.lab !== "etl") return NextResponse.json({ error: "Pipeline not found." }, { status: 404 });
  if (!proj.published) return NextResponse.json({ error: "Publish the pipeline first, then deploy it." }, { status: 400 });

  // Optional connection config for full extract → load runs. Only kept when a real
  // source URL is present; otherwise the API stays transform-only (POST rows in).
  const c = b.conn && typeof b.conn === "object" ? b.conn : null;
  const hasFullRun = !!(c && typeof c.sourceUrl === "string" && /^(mysql|postgres(ql)?|libsql):\/\//i.test(c.sourceUrl));
  const conn = hasFullRun ? {
    sourceUrl: String(c.sourceUrl), query: String(c.query || ""), srcType: String(c.srcType || ""),
    targetUrl: typeof c.targetUrl === "string" ? c.targetUrl : "", table: String(c.table || "etl_output"),
    mode: c.mode === "upsert" ? "upsert" : "append", keyCol: String(c.keyCol || ""),
  } : null;

  const id = uid();
  const key = `sk_${uid().replace(/-/g, "")}${uid().replace(/-/g, "").slice(0, 8)}`;
  await db.insert(channels).values({
    id, userId: user.id, projectId, type: "api",
    secretEnc: encrypt(key),
    configEnc: conn ? encrypt(JSON.stringify(conn)) : null,
    dailyLimit: Math.max(1, Math.min(100_000, Math.round(Number(b.dailyLimit) || 200))),
  });
  await audit("etl_deployed", user.id, { pipeline: proj.name, mode: hasFullRun ? "full_run" : "transform_only", hasTarget: !!conn?.targetUrl }).catch(() => {});
  return NextResponse.json({ ok: true, id, apiKey: key, mode: hasFullRun ? "full_run" : "transform_only", hasTarget: !!conn?.targetUrl });
}
