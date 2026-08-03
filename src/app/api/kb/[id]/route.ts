import { NextResponse } from "next/server";
import { and, eq, sql } from "drizzle-orm";
import { db } from "@/lib/db";
import { knowledgeBases, kbChunks } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  const [kb] = await db.select().from(knowledgeBases).where(and(eq(knowledgeBases.id, id), eq(knowledgeBases.userId, user.id)));
  if (!kb) return NextResponse.json({ error: "not found" }, { status: 404 });
  const docs = await db.select({ docName: kbChunks.docName, chunks: sql<number>`COUNT(*)` })
    .from(kbChunks).where(eq(kbChunks.kbId, id)).groupBy(kbChunks.docName);
  return NextResponse.json({ kb: { id: kb.id, name: kb.name, status: kb.status, docCount: kb.docCount, chunkCount: kb.chunkCount, embModel: kb.embModel }, docs });
}

export async function DELETE(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  const [kb] = await db.select({ id: knowledgeBases.id }).from(knowledgeBases).where(and(eq(knowledgeBases.id, id), eq(knowledgeBases.userId, user.id)));
  if (!kb) return NextResponse.json({ error: "not found" }, { status: 404 });
  await db.delete(kbChunks).where(eq(kbChunks.kbId, id));
  await db.delete(knowledgeBases).where(eq(knowledgeBases.id, id));
  return NextResponse.json({ ok: true });
}
