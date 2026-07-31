import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { knowledgeBases, kbChunks } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { rateLimitDb } from "@/lib/ratelimit";
import { chunkDocs, embedChunks } from "@/lib/kb";
import { captureError } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_CHUNKS = 1500;

// Ingest documents into a KB: chunk -> embed (or TF-IDF fallback) -> store vectors.
export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  if (!(await rateLimitDb("kbsync", user.id, 10, 60_000))) return NextResponse.json({ error: "Too many syncs — wait a minute." }, { status: 429 });

  const [kb] = await db.select().from(knowledgeBases).where(and(eq(knowledgeBases.id, id), eq(knowledgeBases.userId, user.id)));
  if (!kb) return NextResponse.json({ error: "not found" }, { status: 404 });

  const b = await req.json().catch(() => ({}));
  const docs: { name: string; text: string }[] = Array.isArray(b.docs)
    ? b.docs.filter((d: unknown) => d && typeof (d as { text?: unknown }).text === "string" && (d as { text: string }).text.trim())
      .map((d: { name?: string; text: string }) => ({ name: String(d.name || "document").slice(0, 200), text: d.text.slice(0, 400_000) }))
    : [];
  if (!docs.length) return NextResponse.json({ error: "No documents to sync — add files or URLs first." }, { status: 400 });

  const prov = b.providerId ? await getProviderById(String(b.providerId)) : await getActiveProvider();
  if (!prov || !prov.baseUrl) return NextResponse.json({ error: "No LLM provider configured (needed for embeddings)." }, { status: 400 });

  await db.update(knowledgeBases).set({ status: "syncing" }).where(eq(knowledgeBases.id, id));
  try {
    const chunks = chunkDocs(docs).slice(0, MAX_CHUNKS);
    const { embModel, embMeta, vectors } = await embedChunks(chunks.map((c) => c.text), prov);

    await db.delete(kbChunks).where(eq(kbChunks.kbId, id));
    for (let i = 0; i < chunks.length; i += 200) {
      const slice = chunks.slice(i, i + 200);
      await db.insert(kbChunks).values(slice.map((c, j) => ({
        id: uid(), kbId: id, docName: c.docName, idx: i + j, text: c.text, embedding: vectors[i + j] ?? null,
      })));
    }
    await db.update(knowledgeBases).set({
      status: "ready", docCount: docs.length, chunkCount: chunks.length, embModel, embMeta,
    }).where(eq(knowledgeBases.id, id));
    return NextResponse.json({ ok: true, chunkCount: chunks.length, embModel });
  } catch (e) {
    captureError(e, { where: "kb.sync", kbId: id });
    await db.update(knowledgeBases).set({ status: "error" }).where(eq(knowledgeBases.id, id)).catch(() => {});
    return NextResponse.json({ error: `Sync failed: ${(e as Error).message}` }, { status: 500 });
  }
}
