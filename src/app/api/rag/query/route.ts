import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { chunkBy, buildIndex, retrieve, retrieveDense } from "@/lib/ragUtils";
import { embedTexts } from "@/lib/kb";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  const b = await req.json().catch(() => ({}));
  const { projectId, query } = b;
  if (!projectId || !query) return NextResponse.json({ error: "projectId and query required" }, { status: 400 });

  // Fetch project
  const [project] = await db
    .select()
    .from(projects)
    .where(and(eq(projects.id, projectId), eq(projects.userId, user.id)));
  if (!project || project.lab !== "rag") return NextResponse.json({ error: "RAG project not found" }, { status: 404 });

  const c = project.config as any;
  if (!c || !Array.isArray(c.docs) || !c.docs.length) {
    return NextResponse.json({ error: "RAG model has no documents" }, { status: 400 });
  }

  const docs = c.docs;
  const size = c.size ?? 40;
  const overlap = c.overlap ?? 8;
  const strategy = c.strategy ?? "hybrid";
  const metric = c.metric ?? "cosine";
  const topK = c.topK ?? 3;
  const providerId = c.providerId || "";
  const model = c.model || "";
  const embedMode = c.embedMode ?? "tfidf";
  const embModel = c.embModel || "";

  // Chunk documents
  const chunks: string[] = [];
  for (const d of docs) {
    if (d && typeof d.text === "string") {
      chunks.push(...chunkBy(d.text, "fixed", size, overlap));
    }
  }

  if (chunks.length === 0) {
    return NextResponse.json({ error: "No text chunks generated from documents" }, { status: 400 });
  }

  // Build TF-IDF index
  const idx = buildIndex(chunks);

  // Retrieve relevant chunks
  let hits;
  if (embedMode === "neural" && embModel) {
    try {
      const inclGlobal = user.role === "admin";
      let prov = providerId ? await getProviderById(providerId, user.id, inclGlobal) : await getActiveProvider(user.id, inclGlobal);
      if (!prov && providerId) prov = await getActiveProvider(user.id, inclGlobal);
      if (prov && prov.baseUrl) {
        const chunkVectors = await embedTexts(prov.baseUrl, prov.apiKey, embModel, chunks);
        const queryVectors = await embedTexts(prov.baseUrl, prov.apiKey, embModel, [query]);
        const qv = queryVectors[0];
        if (qv && chunkVectors.length === chunks.length) {
          hits = retrieveDense(idx, query, qv, chunkVectors, strategy, topK, metric);
        }
      }
    } catch (e) {
      console.error("Server RAG embedding failed, falling back to TF-IDF:", e);
    }
  }

  if (!hits) {
    hits = retrieve(idx, query, strategy, topK, metric);
  }

  // Construct prompt context
  const context = hits.map((h) => `[chunk ${h.i + 1}] ${chunks[h.i]}`).join("\n\n");

  const messages = [
    { role: "system", content: "You are a helpful assistant. Answer using ONLY the provided context. Cite inline like [chunk N]. If the answer is not in the context, say you don't know." },
    { role: "user", content: `Context:\n${context}\n\nQuestion: ${query}` },
  ];

  // Run chat completion
  try {
    const inclGlobal = user.role === "admin";
    let prov = providerId ? await getProviderById(providerId, user.id, inclGlobal) : await getActiveProvider(user.id, inclGlobal);
    if (!prov && providerId) prov = await getActiveProvider(user.id, inclGlobal);
    if (!prov || !prov.baseUrl) return NextResponse.json({ error: "No provider configured" }, { status: 400 });

    const activeModel = model || prov.model || "gpt-3.5-turbo";
    const payload = {
      model: activeModel,
      messages,
      temperature: 0.2,
      max_tokens: 512,
      stream: false,
    };

    const upstream = await fetch(prov.baseUrl.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", ...(prov.apiKey ? { Authorization: `Bearer ${prov.apiKey}` } : {}) },
      body: JSON.stringify(payload),
    });

    if (!upstream.ok) {
      const t = await upstream.text().catch(() => "");
      return NextResponse.json({ error: `Provider error ${upstream.status}: ${t.slice(0, 300)}` }, { status: 502 });
    }

    const j = await upstream.json().catch(() => null) as { choices?: Array<{ message?: { content?: string } }> } | null;
    const text = j?.choices?.[0]?.message?.content ?? "No response content from model.";
    return NextResponse.json({ answer: text, hits });
  } catch (e) {
    return NextResponse.json({ error: `RAG generation failed: ${(e as Error).message}` }, { status: 500 });
  }
}
