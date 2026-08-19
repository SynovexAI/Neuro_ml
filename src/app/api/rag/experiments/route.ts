import { NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { ragExperiments } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { rateLimitDb } from "@/lib/ratelimit";
import { audit } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const str = (v: unknown, cap: number) => (typeof v === "string" ? v.slice(0, cap) : "");
const num = (v: unknown, cap = 1_000_000) => Math.max(0, Math.min(cap, Math.floor(Number(v) || 0)));

// GET → the current user's saved RAG experiments (newest first).
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(ragExperiments)
    .where(eq(ragExperiments.userId, user.id)).orderBy(desc(ragExperiments.ts)).limit(60);
  return NextResponse.json({ experiments: rows });
}

// POST → save one experiment (config + metrics only, never document/embedding copies).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("ragexp", user.id, 60, 60_000))) return NextResponse.json({ error: "Too many saves — wait a minute." }, { status: 429 });
  const b = await req.json().catch(() => ({}));

  const label = str(b.label, 160).trim() || "Experiment";
  const cfg = b.config && typeof b.config === "object" ? b.config : {};
  // Whitelist the config keys we store — keeps rows tiny and predictable.
  const config = {
    backend: str(cfg.backend, 16) || "vector",
    size: num(cfg.size, 100000),
    overlap: num(cfg.overlap, 100000),
    strategy: str(cfg.strategy, 16) || "hybrid",
    metric: str(cfg.metric, 24) || "cosine",
    topK: num(cfg.topK, 100),
    rerank: str(cfg.rerank, 16) || "none",
    mmrLambda: Math.max(0, Math.min(1, Number(cfg.mmrLambda) || 0)),
    embedMode: str(cfg.embedMode, 16) || "tfidf",
    embModel: str(cfg.embModel, 120),
    kgHops: num(cfg.kgHops, 10),
    alpha: Math.max(0, Math.min(1, Number(cfg.alpha) || 0.5)),
    chunkStrategy: str(cfg.chunkStrategy, 16) || "fixed",
    queryMode: str(cfg.queryMode, 16) || "none",
  };
  const m = b.metrics && typeof b.metrics === "object" ? b.metrics : null;
  const metrics = m ? { p: Number(m.p) || 0, r: Number(m.r) || 0, mrr: Number(m.mrr) || 0, ndcg: Number(m.ndcg) || 0 } : null;

  const row = {
    id: uid(), userId: user.id, label,
    dataset: str(b.dataset, 200) || null,
    question: str(b.question, 2000) || null,
    config, metrics,
    chunkCount: num(b.chunkCount, 1_000_000),
    latencyMs: num(b.latencyMs, 3_600_000),
  };
  await db.insert(ragExperiments).values(row);
  await audit("rag_experiment_saved", user.id, { label, dataset: row.dataset, strategy: config.strategy, chunks: row.chunkCount }).catch(() => {});
  return NextResponse.json({ ok: true, id: row.id });
}
