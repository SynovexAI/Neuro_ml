import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { providers } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { encrypt } from "@/lib/crypto";

async function admin() {
  const u = await getSessionUser();
  return u && u.role === "admin" ? u : null;
}

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!(await admin())) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  const b = await req.json().catch(() => ({}));
  const patch: Record<string, unknown> = {};
  if (b.enabled !== undefined) patch.enabled = !!b.enabled;
  if (b.defaultModel !== undefined) patch.defaultModel = String(b.defaultModel);
  if (b.label !== undefined) patch.label = String(b.label);
  if (b.baseUrl !== undefined) patch.baseUrl = String(b.baseUrl);
  if (b.apiKey) patch.apiKeyEnc = encrypt(String(b.apiKey));
  if (Object.keys(patch).length) await db.update(providers).set(patch).where(eq(providers.id, id));
  return NextResponse.json({ ok: true });
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!(await admin())) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  await db.delete(providers).where(eq(providers.id, id));
  return NextResponse.json({ ok: true });
}
