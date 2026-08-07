import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { rateLimitDb } from "@/lib/ratelimit";
import { embedTexts, pickEmbModel } from "@/lib/kb";
import { captureError } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

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

  // Honor a provider chosen in the UI; otherwise use the first active provider.
  const providerId = typeof b.providerId === "string" ? b.providerId : "";
  const reqModel = typeof b.model === "string" && b.model.trim() ? b.model.trim() : "";
  // If a specific provider was requested but no longer exists (e.g. deleted & re-added with a new id),
  // fall back to the first active provider instead of failing.
  const inclGlobal = user.role === "admin";
  let prov = providerId ? await getProviderById(providerId, user.id, inclGlobal) : await getActiveProvider(user.id, inclGlobal);
  if (!prov && providerId) prov = await getActiveProvider(user.id, inclGlobal);
  if (!prov || !prov.baseUrl) return NextResponse.json({ error: "No LLM provider configured — add one under Admin → Providers for neural embeddings." }, { status: 400 });

  const model = reqModel || pickEmbModel(prov.baseUrl);
  let host = prov.baseUrl;
  try { host = new URL(prov.baseUrl).host; } catch { /* keep raw */ }
  try {
    const vectors = await embedTexts(prov.baseUrl, prov.apiKey, model, texts);
    return NextResponse.json({ ok: true, model, dim: vectors[0]?.length ?? 0, vectors });
  } catch (e) {
    captureError(e, { where: "rag.embed", model, host });
    const msg = (e as Error).message;
    const is404 = /\b404\b/.test(msg);
    return NextResponse.json({
      error: is404
        ? `Model "${model}" isn't served by ${host} (404). Pick a valid embedding model for this provider — e.g. mistral-embed (Mistral), gemini-embedding-001 (Gemini), text-embedding-3-small (OpenAI/GitHub).`
        : `Embedding failed on ${host} with "${model}": ${msg}. Vector retrieval will fall back to TF-IDF.`,
    }, { status: 502 });
  }
}
