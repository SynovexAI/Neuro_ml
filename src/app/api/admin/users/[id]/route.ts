import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { users, sessions, userProviders, projects, channels, mcpServers, knowledgeBases, agentRuns, usage, etlDatasets } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const me = await getSessionUser();
  if (!me || me.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  const b = await req.json().catch(() => ({}));
  const patch: Record<string, unknown> = {};
  if (b.status && ["pending", "active", "suspended"].includes(b.status)) patch.status = b.status;
  if (b.role && ["admin", "student"].includes(b.role)) patch.role = b.role;
  // Per-user monthly token quota. null / "" resets to the platform default.
  if (b.monthlyTokenLimit !== undefined)
    patch.monthlyTokenLimit = (b.monthlyTokenLimit === null || b.monthlyTokenLimit === "") ? null : Math.max(0, Math.floor(Number(b.monthlyTokenLimit)) || 0);
  // Don't let an admin lock themselves out.
  if (id === me.id && (patch.role === "student" || patch.status === "suspended"))
    return NextResponse.json({ error: "You can't demote or suspend your own account." }, { status: 400 });
  if (Object.keys(patch).length) await db.update(users).set(patch).where(eq(users.id, id));
  return NextResponse.json({ ok: true });
}

// Permanently delete a user account and their data (sessions, keys, projects, KBs, etc.).
export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const me = await getSessionUser();
  if (!me || me.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  if (id === me.id) return NextResponse.json({ error: "You can't delete your own account." }, { status: 400 });

  // Remove sessions first so the user is immediately signed out, then their owned rows.
  const wipes = [
    db.delete(sessions).where(eq(sessions.userId, id)),
    db.delete(userProviders).where(eq(userProviders.userId, id)),
    db.delete(projects).where(eq(projects.userId, id)),
    db.delete(channels).where(eq(channels.userId, id)),
    db.delete(mcpServers).where(eq(mcpServers.userId, id)),
    db.delete(knowledgeBases).where(eq(knowledgeBases.userId, id)),
    db.delete(agentRuns).where(eq(agentRuns.userId, id)),
    db.delete(usage).where(eq(usage.userId, id)),
    db.delete(etlDatasets).where(eq(etlDatasets.userId, id)),
  ];
  for (const w of wipes) { try { await w; } catch { /* table may not exist / no rows */ } }
  await db.delete(users).where(eq(users.id, id));
  return NextResponse.json({ ok: true });
}
