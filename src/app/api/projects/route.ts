import { NextResponse } from "next/server";
import { and, eq, desc } from "drizzle-orm";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const lab = new URL(req.url).searchParams.get("lab");
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
  const id = uid();
  await db.insert(projects).values({ id, userId: user.id, lab: String(b.lab), name: String(b.name), config: b.config ?? {} });
  return NextResponse.json({ ok: true, id });
}
