import { NextResponse } from "next/server";
import { and, eq, desc } from "drizzle-orm";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const sp = new URL(req.url).searchParams;
  const id = sp.get("id");
  if (id) {
    const [row] = await db.select().from(projects).where(and(eq(projects.id, id), eq(projects.userId, user.id)));
    if (!row) return NextResponse.json({ error: "not found" }, { status: 404 });
    return NextResponse.json({ project: row });
  }
  const lab = sp.get("lab");
  const where = lab
    ? and(eq(projects.userId, user.id), eq(projects.lab, lab))
    : eq(projects.userId, user.id);
  const rows = await db.select().from(projects).where(where).orderBy(desc(projects.updatedAt));
  return NextResponse.json({ projects: rows });
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  if (!b.lab || !b.name) return NextResponse.json({ error: "lab and name required" }, { status: 400 });
  // Guard against oversized configs (a project row should stay small).
  const bytes = b.config ? JSON.stringify(b.config).length : 0;
  if (bytes > 1_200_000) return NextResponse.json({ error: "This build is too large to save (over ~1.2 MB). Trim the documents/data and try again." }, { status: 413 });
  const id = uid();
  await db.insert(projects).values({ id, userId: user.id, lab: String(b.lab), name: String(b.name).slice(0, 160), config: b.config ?? {}, published: b.published === true });
  return NextResponse.json({ ok: true, id });
}

export async function PATCH(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  if (!b.id) return NextResponse.json({ error: "id required" }, { status: 400 });
  const set: Record<string, unknown> = {};
  if (typeof b.name === "string") set.name = b.name.slice(0, 160);
  if (b.config !== undefined) set.config = b.config;
  if (typeof b.published === "boolean") set.published = b.published;
  if (Object.keys(set).length === 0) return NextResponse.json({ error: "nothing to update" }, { status: 400 });
  const res = await db.update(projects).set(set).where(and(eq(projects.id, String(b.id)), eq(projects.userId, user.id)));
  return NextResponse.json({ ok: true, res });
}

export async function DELETE(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const id = new URL(req.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });
  await db.delete(projects).where(and(eq(projects.id, id), eq(projects.userId, user.id)));
  return NextResponse.json({ ok: true });
}
