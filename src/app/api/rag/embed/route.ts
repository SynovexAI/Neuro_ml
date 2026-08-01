import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider } from "@/lib/providers";
import { rateLimitDb } from "@/lib/ratelimit";
import { embedTexts, pickEmbModel } from "@/lib/kb";
import { captureError } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_TEXTS = 200;

// Real neural embeddings for the RAG lab: chunks + query → dense vectors via the
// active provider's OpenAI-compatible /embeddings endpoint. Used for semantic
// (vector) retrieval so it isn't just lexical TF-IDF.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("ragembed", user.id, 30, 60_000))) return NextResponse.json({ error: "Too many embed requests — wait a minute." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const texts: string[] = Array.isArray(b.texts) ? b.texts.filter((t: unknown) => typeof t === "string").slice(0, MAX_TEXTS).map((t: string) => t.slice(0, 8000)) : [];
  if (!texts.length) return NextResponse.json({ error: "No texts to embed." }, { status: 400 });

  const prov = await getActiveProvider();
  if (!prov || !prov.baseUrl) return NextResponse.json({ error: "No LLM provider configured — an admin must add one under Admin → Providers for neural embeddings." }, { status: 400 });

  try {
    const model = pickEmbModel(prov.baseUrl);
    const vectors = await embedTexts(prov.baseUrl, prov.apiKey, model, texts);
    return NextResponse.json({ ok: true, model, dim: vectors[0]?.length ?? 0, vectors });
  } catch (e) {
    captureError(e, { where: "rag.embed" });
    return NextResponse.json({ error: `Embedding failed: ${(e as Error).message}. Vector retrieval will fall back to TF-IDF.` }, { status: 502 });
  }
}
