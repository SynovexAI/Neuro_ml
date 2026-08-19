import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { userProviders } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { encrypt } from "@/lib/crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function owns(userId: string, id: string) {
  const rows = await db.select({ id: userProviders.id }).from(userProviders).where(and(eq(userProviders.id, id), eq(userProviders.userId, userId))).limit(1);
  return !!rows[0];
}

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  if (!(await owns(u.id, id))) return NextResponse.json({ error: "not found" }, { status: 404 });
  const b = await req.json().catch(() => ({}));
  const patch: Record<string, unknown> = {};
  if (b.enabled !== undefined) patch.enabled = !!b.enabled;
  if (b.defaultModel !== undefined) patch.defaultModel = String(b.defaultModel);
  if (b.label !== undefined) patch.label = String(b.label);
  if (b.baseUrl !== undefined) patch.baseUrl = String(b.baseUrl);
  if (b.apiKey) patch.apiKeyEnc = encrypt(String(b.apiKey));
  if (Object.keys(patch).length) await db.update(userProviders).set(patch).where(and(eq(userProviders.id, id), eq(userProviders.userId, u.id)));
  return NextResponse.json({ ok: true });
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  await db.delete(userProviders).where(and(eq(userProviders.id, id), eq(userProviders.userId, u.id)));
  return NextResponse.json({ ok: true });
}
