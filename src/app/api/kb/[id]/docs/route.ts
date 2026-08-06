import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { knowledgeBases, kbChunks, kbDocs } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// GET /api/kb/[id]/docs → the KB's documents reconstructed from its stored chunks
// (grouped by docName, chunk text joined in order). Used as a RAG source and by agents.
export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  const kbRows = await db.select({ id: knowledgeBases.id, name: knowledgeBases.name }).from(knowledgeBases).where(and(eq(knowledgeBases.id, id), eq(knowledgeBases.userId, u.id))).limit(1);
  const kb = kbRows[0];
  if (!kb) return NextResponse.json({ error: "not found" }, { status: 404 });

  // Prefer the exact stored originals; fall back to reconstructing from chunks (older KBs).
  try {
    const exact = await db.select({ name: kbDocs.name, text: kbDocs.text }).from(kbDocs).where(eq(kbDocs.kbId, id));
    if (exact.length) {
      return NextResponse.json({ kb: kb.name, docs: exact.map((d) => ({ name: d.name || "document", text: (d.text || "").slice(0, 400_000) })) });
    }
  } catch { /* kb_docs missing → reconstruct below */ }

  const rows = await db.select({ docName: kbChunks.docName, idx: kbChunks.idx, text: kbChunks.text }).from(kbChunks).where(eq(kbChunks.kbId, id)).orderBy(kbChunks.idx);
  const byDoc = new Map<string, string[]>();
  for (const c of rows) {
    const name = c.docName || "document";
    if (!byDoc.has(name)) byDoc.set(name, []);
    byDoc.get(name)!.push(c.text || "");
  }
  const docs = [...byDoc.entries()].map(([name, parts]) => ({ name, text: parts.join(" ").slice(0, 400_000) }));
  return NextResponse.json({ kb: kb.name, docs });
}
