import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { channels, projects } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";
import { runPublishedAgent } from "@/lib/runPublished";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "content-type,authorization", "Access-Control-Allow-Methods": "POST,OPTIONS" };

export function OPTIONS() { return new NextResponse(null, { status: 204, headers: CORS }); }

// Public agent endpoint for `api` (Bearer key) and `widget` (keyless, embeddable)
// channels. Runs the agent as its owner. Rate-limited per channel + IP.
export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const [ch] = await db.select().from(channels).where(eq(channels.id, id));
  if (!ch || !ch.enabled || (ch.type !== "api" && ch.type !== "widget")) return NextResponse.json({ error: "Not found." }, { status: 404, headers: CORS });

  if (ch.type === "api") {
    const auth = req.headers.get("authorization") || "";
    const key = auth.replace(/^Bearer\s+/i, "").trim();
    if (!ch.secretEnc || key !== decrypt(ch.secretEnc)) return NextResponse.json({ error: "Invalid API key." }, { status: 401, headers: CORS });
  }

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0].trim() || "anon";
  if (!(await rateLimitDb(`pub:${id}`, ip, 30, 60_000))) return NextResponse.json({ error: "Rate limit — slow down." }, { status: 429, headers: CORS });
  // Per-channel daily cap so a public link can't run up the owner's bill.
  const cap = ch.dailyLimit ?? 200;
  if (!(await rateLimitDb(`pubday:${id}`, id, cap, 24 * 60 * 60_000))) return NextResponse.json({ error: "This assistant has reached its daily message limit. Try again tomorrow." }, { status: 429, headers: CORS });

  const b = await req.json().catch(() => ({}));
  const task = String(b.message || b.task || "").trim();
  if (!task) return NextResponse.json({ error: "A message is required." }, { status: 400, headers: CORS });

  const [proj] = await db.select().from(projects).where(eq(projects.id, ch.projectId));
  if (!proj || !proj.published) return NextResponse.json({ error: "This agent is unavailable." }, { status: 404, headers: CORS });

  const r = await runPublishedAgent({ userId: ch.userId, config: proj.config, task, agentName: proj.name });
  if (!r.ok) return NextResponse.json({ error: r.error }, { status: r.status, headers: CORS });
  return NextResponse.json({ answer: r.answer, latency_ms: r.latency_ms }, { headers: CORS });
}
