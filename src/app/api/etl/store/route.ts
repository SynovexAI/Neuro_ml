import { NextResponse } from "next/server";
import { and, eq, desc } from "drizzle-orm";
import { db } from "@/lib/db";
import { etlDatasets, etlDatasetRows } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_ROWS = 5000;

// List the user's stored ETL datasets, or one dataset's rows (?id=).
export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const id = new URL(req.url).searchParams.get("id");
  if (id) {
    const [ds] = await db.select().from(etlDatasets).where(and(eq(etlDatasets.id, id), eq(etlDatasets.userId, user.id)));
    if (!ds) return NextResponse.json({ error: "not found" }, { status: 404 });
    const rows = await db.select({ idx: etlDatasetRows.idx, data: etlDatasetRows.data }).from(etlDatasetRows).where(eq(etlDatasetRows.datasetId, id)).orderBy(etlDatasetRows.idx);
    return NextResponse.json({ dataset: { id: ds.id, name: ds.name, cols: ds.cols, rowCount: ds.rowCount }, rows: rows.map((r) => r.data) });
  }
  const rows = await db.select({ id: etlDatasets.id, name: etlDatasets.name, rowCount: etlDatasets.rowCount, cols: etlDatasets.cols, createdAt: etlDatasets.createdAt }).from(etlDatasets).where(eq(etlDatasets.userId, user.id)).orderBy(desc(etlDatasets.createdAt));
  return NextResponse.json({ datasets: rows });
}

// Persist a pipeline output to a real database table (scoped to the user).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("etlstore", user.id, 20, 60_000))) return NextResponse.json({ error: "Too many writes — wait a moment." }, { status: 429 });
  const b = await req.json().catch(() => ({}));
  const name = String(b.name || "").trim().slice(0, 160) || "ETL output";
  const cols: string[] = Array.isArray(b.cols) ? b.cols.map(String) : [];
  const rows: Record<string, unknown>[] = Array.isArray(b.rows) ? b.rows : [];
  if (!rows.length) return NextResponse.json({ error: "Nothing to store — the output is empty." }, { status: 400 });
  if (rows.length > MAX_ROWS) return NextResponse.json({ error: `Too many rows to store (${rows.length}). Cap is ${MAX_ROWS} — add a Limit transform first.` }, { status: 413 });
  if (JSON.stringify(rows).length > 4_000_000) return NextResponse.json({ error: "Output is too large to store (over ~4 MB)." }, { status: 413 });

  const dsId = uid();
  await db.insert(etlDatasets).values({ id: dsId, userId: user.id, name, cols, rowCount: rows.length });
  // Batch-insert rows.
  const values = rows.map((data, idx) => ({ id: uid(), datasetId: dsId, idx, data }));
  for (let i = 0; i < values.length; i += 500) await db.insert(etlDatasetRows).values(values.slice(i, i + 500));
  return NextResponse.json({ ok: true, id: dsId, rowCount: rows.length });
}

export async function DELETE(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const id = new URL(req.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });
  const [ds] = await db.select().from(etlDatasets).where(and(eq(etlDatasets.id, id), eq(etlDatasets.userId, user.id)));
  if (!ds) return NextResponse.json({ error: "not found" }, { status: 404 });
  await db.delete(etlDatasetRows).where(eq(etlDatasetRows.datasetId, id));
  await db.delete(etlDatasets).where(and(eq(etlDatasets.id, id), eq(etlDatasets.userId, user.id)));
  return NextResponse.json({ ok: true });
}
