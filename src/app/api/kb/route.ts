import { NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { knowledgeBases } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select({
    id: knowledgeBases.id, name: knowledgeBases.name, status: knowledgeBases.status,
    docCount: knowledgeBases.docCount, chunkCount: knowledgeBases.chunkCount,
    embModel: knowledgeBases.embModel, updatedAt: knowledgeBases.updatedAt,
  }).from(knowledgeBases).where(eq(knowledgeBases.userId, user.id)).orderBy(desc(knowledgeBases.updatedAt));
  return NextResponse.json({ kbs: rows });
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  const name = String(b.name || "").trim();
  if (!name) return NextResponse.json({ error: "A name is required." }, { status: 400 });
  const id = uid();
  await db.insert(knowledgeBases).values({ id, userId: user.id, name: name.slice(0, 120), status: "empty" });
  return NextResponse.json({ ok: true, id });
}
