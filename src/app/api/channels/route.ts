import { NextResponse } from "next/server";
import { and, eq, inArray, desc } from "drizzle-orm";
import { db } from "@/lib/db";
import { channels, projects } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { encrypt, decrypt } from "@/lib/crypto";
import { setWebhook, deleteWebhook, getMe, webhookSecret } from "@/lib/telegram";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TYPES = ["telegram", "api", "widget"] as const;

function baseUrl(body: { publicUrl?: string }, req: Request): string {
  const raw = (body.publicUrl || process.env.APP_URL || process.env.NEXT_PUBLIC_APP_URL || new URL(req.url).origin).trim();
  return raw.replace(/\/$/, "");
}

export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(channels).where(eq(channels.userId, user.id)).orderBy(desc(channels.createdAt));
  const pids = [...new Set(rows.map((r) => r.projectId))];
  const projs = pids.length ? await db.select({ id: projects.id, name: projects.name }).from(projects).where(inArray(projects.id, pids)) : [];
  const nameOf = new Map(projs.map((p) => [p.id, p.name]));
  // Never return the raw secret — only whether one is set.
  const out = rows.map((r) => ({ id: r.id, type: r.type, projectId: r.projectId, agentName: nameOf.get(r.projectId) || "(deleted agent)", enabled: r.enabled, dailyLimit: r.dailyLimit ?? null, hasSecret: !!r.secretEnc, createdAt: r.createdAt }));
  return NextResponse.json({ channels: out });
}

const clampCap = (v: unknown) => Math.max(1, Math.min(100_000, Math.round(Number(v) || 200)));

// Toggle enabled or change the daily cap. Telegram webhooks stay registered while
// paused (we just stop processing) — cleaner than re-issuing on every toggle.
export async function PATCH(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  if (!b.id) return NextResponse.json({ error: "id required" }, { status: 400 });
  const set: Record<string, unknown> = {};
  if (typeof b.enabled === "boolean") set.enabled = b.enabled;
  if (b.dailyLimit !== undefined) set.dailyLimit = clampCap(b.dailyLimit);
  if (Object.keys(set).length === 0) return NextResponse.json({ error: "nothing to update" }, { status: 400 });
  await db.update(channels).set(set).where(and(eq(channels.id, String(b.id)), eq(channels.userId, user.id)));
  return NextResponse.json({ ok: true });
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  const type = String(b.type || "");
  if (!TYPES.includes(type as typeof TYPES[number])) return NextResponse.json({ error: "Unknown channel type." }, { status: 400 });
  const projectId = String(b.projectId || "");
  const [proj] = await db.select().from(projects).where(and(eq(projects.id, projectId), eq(projects.userId, user.id)));
  if (!proj) return NextResponse.json({ error: "Agent not found." }, { status: 404 });
  if (!proj.published) return NextResponse.json({ error: "Publish the agent first, then deploy it." }, { status: 400 });

  const id = uid();

  if (type === "telegram") {
    const token = String(b.token || "").trim();
    if (!/^\d+:[\w-]{30,}$/.test(token)) return NextResponse.json({ error: "That doesn't look like a Telegram bot token. Get one from @BotFather." }, { status: 400 });
    const me = await getMe(token);
    if (!me.ok) return NextResponse.json({ error: `Telegram rejected the token: ${me.description}` }, { status: 400 });
    const url = `${baseUrl(b, req)}/api/channels/telegram/${id}`;
    if (/^https:\/\//.test(url) && !/localhost|127\.0\.0\.1/.test(url)) {
      const wh = await setWebhook(token, url, webhookSecret(id));
      if (!wh.ok) return NextResponse.json({ error: `Couldn't register the webhook: ${wh.description}. Make sure the URL is public HTTPS.` }, { status: 400 });
    } else {
      return NextResponse.json({ error: "Telegram needs a public HTTPS URL. Deploy the app (or use a tunnel) and pass its URL, not localhost." }, { status: 400 });
    }
    await db.insert(channels).values({ id, userId: user.id, projectId, type, secretEnc: encrypt(token) });
    return NextResponse.json({ ok: true, id, botUsername: me.username, webhookUrl: url });
  }

  if (type === "api") {
    const key = `sk_${uid().replace(/-/g, "")}${uid().replace(/-/g, "").slice(0, 8)}`;
    await db.insert(channels).values({ id, userId: user.id, projectId, type, secretEnc: encrypt(key), dailyLimit: clampCap(b.dailyLimit) });
    // Returned once, in plaintext — the caller must copy it now.
    return NextResponse.json({ ok: true, id, apiKey: key });
  }

  // widget — no secret
  await db.insert(channels).values({ id, userId: user.id, projectId, type, dailyLimit: clampCap(b.dailyLimit) });
  return NextResponse.json({ ok: true, id });
}

export async function DELETE(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const id = new URL(req.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });
  const [ch] = await db.select().from(channels).where(and(eq(channels.id, id), eq(channels.userId, user.id)));
  if (!ch) return NextResponse.json({ error: "not found" }, { status: 404 });
  if (ch.type === "telegram" && ch.secretEnc) { try { await deleteWebhook(decrypt(ch.secretEnc)); } catch { /* best effort */ } }
  await db.delete(channels).where(and(eq(channels.id, id), eq(channels.userId, user.id)));
  return NextResponse.json({ ok: true });
}
