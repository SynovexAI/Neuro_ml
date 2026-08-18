import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { knowledgeBases, kbChunks, kbDocs } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { rateLimitDb } from "@/lib/ratelimit";
import { chunkDocs, embedChunks } from "@/lib/kb";
import { captureError } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

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
    const newChunks = chunkDocs(docs, Number(b.chunkSize) || 60, Number(b.chunkOverlap) ?? 12);
    const emb = await embedChunks(newChunks.map((c) => c.text), prov);

    // Documents ACCUMULATE in a KB: append the new docs rather than replacing.
    // Neural embeddings are per-chunk independent → just append. TF-IDF weights depend
    // on the whole corpus, so recompute over existing + new and rewrite all vectors.
    let embModel = emb.embModel, embMeta = emb.embMeta;
    let rows: { docName: string | null; text: string; vec: number[] | null }[];
    let replaceAll = false, baseIdx = 0;

    if (emb.embModel === "tfidf") {
      const existing = await db.select({ docName: kbChunks.docName, text: kbChunks.text }).from(kbChunks).where(eq(kbChunks.kbId, id)).orderBy(kbChunks.idx);
      const all = [...existing.map((e) => ({ docName: e.docName, text: e.text ?? "" })), ...newChunks.map((c) => ({ docName: c.docName as string | null, text: c.text }))].slice(0, MAX_CHUNKS);
      const re = await embedChunks(all.map((c) => c.text), prov);
      embModel = re.embModel; embMeta = re.embMeta; replaceAll = true;
      rows = all.map((c, k) => ({ docName: c.docName, text: c.text, vec: (re.vectors[k] as number[] | undefined) ?? null }));
    } else {
      baseIdx = kb.chunkCount || 0;
      const nc = newChunks.slice(0, Math.max(0, MAX_CHUNKS - baseIdx));
      rows = nc.map((c, k) => ({ docName: c.docName as string | null, text: c.text, vec: (emb.vectors[k] as number[] | undefined) ?? null }));
    }

    if (replaceAll) await db.delete(kbChunks).where(eq(kbChunks.kbId, id));
    for (let i = 0; i < rows.length; i += 200) {
      const slice = rows.slice(i, i + 200);
      await db.insert(kbChunks).values(slice.map((c, j) => ({
        id: uid(), kbId: id, docName: c.docName, idx: (replaceAll ? 0 : baseIdx) + i + j, text: c.text, embedding: c.vec,
      })));
    }
    // Keep the exact original text (best-effort — table may not exist on older DBs).
    try { await db.insert(kbDocs).values(docs.map((d) => ({ id: uid(), kbId: id, name: d.name, text: d.text }))); } catch { /* kb_docs missing */ }
    const docCount = (kb.docCount || 0) + docs.length;
    const chunkCount = replaceAll ? rows.length : (kb.chunkCount || 0) + rows.length;
    await db.update(knowledgeBases).set({ status: "ready", docCount, chunkCount, embModel, embMeta }).where(eq(knowledgeBases.id, id));
    return NextResponse.json({ ok: true, chunkCount, added: rows.length, embModel });
  } catch (e) {
    captureError(e, { where: "kb.sync", kbId: id });
    await db.update(knowledgeBases).set({ status: "error" }).where(eq(knowledgeBases.id, id)).catch(() => {});
    return NextResponse.json({ error: `Sync failed: ${(e as Error).message}` }, { status: 500 });
  }
}
