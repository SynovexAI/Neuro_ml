import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { mcpServers } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { encrypt } from "@/lib/crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  const b = await req.json().catch(() => ({}));
  const patch: Record<string, unknown> = {};
  if (typeof b.enabled === "boolean") patch.enabled = b.enabled;
  if (b.name) patch.name = String(b.name).slice(0, 80);
  if (b.url !== undefined) patch.url = b.url ? String(b.url) : null;
  if (b.command !== undefined) patch.command = b.command ? String(b.command) : null;
  if (b.headerName !== undefined) patch.headerName = b.headerName ? String(b.headerName).slice(0, 80) : null;
  if (b.envName !== undefined) patch.envName = b.envName ? String(b.envName).slice(0, 80) : null;
  // Only re-encrypt when a new secret is actually supplied.
  if (b.secret) patch.secretEnc = encrypt(String(b.secret));
  if (Object.keys(patch).length) await db.update(mcpServers).set(patch).where(eq(mcpServers.id, id));
  return NextResponse.json({ ok: true });
}

export async function DELETE(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  await db.delete(mcpServers).where(eq(mcpServers.id, id));
  return NextResponse.json({ ok: true });
}
