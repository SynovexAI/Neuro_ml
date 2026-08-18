import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { channels, projects } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";
type EtlConn = { sourceUrl: string; query: string; srcType?: string; targetUrl?: string; table?: string; mode?: string; keyCol?: string };
import { rateLimitDb } from "@/lib/ratelimit";
import { runPipeline, tableFromRecords, type EtlOp, type Table } from "@/lib/etlUtils";
import { extractRows, loadRows } from "@/lib/etlRun";
import { audit } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "content-type,authorization", "Access-Control-Allow-Methods": "POST,OPTIONS" };
const MAX_ROWS = 5000; // "low-data" API — big jobs must use the exported Python script

export function OPTIONS() { return new NextResponse(null, { status: 204, headers: CORS }); }

type WfNode = { position?: { x?: number }; data?: { kind?: string; op?: EtlOp; table?: Table } };
type Wf = { nodes?: WfNode[] };

// Public ETL endpoint: runs a PUBLISHED etl pipeline's transforms on the JSON rows
// the caller POSTs, and returns the transformed rows. Auth is a Bearer key stored
// in an `api` channel (same model as the agent public API). Small data only —
// there's a hard row cap and a 60s function limit; for real volumes, export Python.
export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const [ch] = await db.select().from(channels).where(eq(channels.id, id));
  if (!ch || !ch.enabled || ch.type !== "api") return NextResponse.json({ error: "Not found." }, { status: 404, headers: CORS });

  const key = (req.headers.get("authorization") || "").replace(/^Bearer\s+/i, "").trim();
  if (!ch.secretEnc || key !== decrypt(ch.secretEnc)) return NextResponse.json({ error: "Invalid API key." }, { status: 401, headers: CORS });

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0].trim() || "anon";
  if (!(await rateLimitDb(`etlpub:${id}`, ip, 30, 60_000))) return NextResponse.json({ error: "Rate limit — slow down." }, { status: 429, headers: CORS });
  if (!(await rateLimitDb(`etlpubday:${id}`, id, ch.dailyLimit ?? 200, 24 * 60 * 60_000))) return NextResponse.json({ error: "Daily run limit reached. Try again tomorrow." }, { status: 429, headers: CORS });

  const [proj] = await db.select().from(projects).where(eq(projects.id, ch.projectId));
  if (!proj || !proj.published || proj.lab !== "etl") return NextResponse.json({ error: "This pipeline is unavailable." }, { status: 404, headers: CORS });

  const cfg = (proj.config || {}) as Wf;
  const nodes = Array.isArray(cfg.nodes) ? cfg.nodes : [];
  const ops = nodes.filter((n) => n.data?.kind === "op" && n.data.op).sort((a, b) => (a.position?.x || 0) - (b.position?.x || 0)).map((n) => n.data!.op!) as EtlOp[];
  // Second source (for join/union) comes from the saved workflow's tables.
  const srcTables = nodes.filter((n) => n.data?.kind === "source" && n.data.table).map((n) => n.data!.table!);
  const secondary = srcTables[1] || null;

  const conn = ch.configEnc ? (JSON.parse(decrypt(ch.configEnc)) as EtlConn) : null;

  // ── FULL RUN: server extracts from the stored source URL, transforms, and (if a
  //    target URL was stored) loads into the warehouse. Trigger with just the key. ──
  if (conn && conn.sourceUrl) {
    const b = await req.json().catch(() => ({}));
    const query = typeof b.query === "string" && /^\s*select\b/i.test(b.query) ? b.query : conn.query; // optional override
    try {
      const src = await extractRows(conn.sourceUrl, query);
      const out = runPipeline(tableFromRecords(src.rows), ops, { secondary }).final;
      let rowsLoaded = 0, loadedTable: string | null = null;
      if (conn.targetUrl) {
        const lr = await loadRows(conn.targetUrl, { table: conn.table || "etl_output", cols: out.cols, rows: out.rows, mode: conn.mode, keyCol: conn.keyCol || out.cols[0] });
        rowsLoaded = lr.rowCount; loadedTable = conn.table || "etl_output";
      }
      await audit("etl_api_run", ch.userId, { pipeline: proj.name, mode: "full_run", rowsExtracted: src.rows.length, rowsOut: out.rows.length, rowsLoaded }).catch(() => {});
      return NextResponse.json({ ok: true, mode: "full_run", pipeline: proj.name, rowsExtracted: src.rows.length, rowsOut: out.rows.length, rowsLoaded, loadedTable, ...(conn.targetUrl ? {} : { cols: out.cols, rows: out.rows }) }, { headers: CORS });
    } catch (e) {
      return NextResponse.json({ error: `Run failed: ${(e as Error).message}` }, { status: 502, headers: CORS });
    }
  }

  // ── TRANSFORM-ONLY: caller POSTs the rows; we run the transforms and return them. ──
  const b = await req.json().catch(() => ({}));
  const records = Array.isArray(b.data) ? b.data : Array.isArray(b.rows) ? b.rows : null;
  if (!records) return NextResponse.json({ error: "Provide `data`: an array of row objects to transform." }, { status: 400, headers: CORS });
  if (records.length > MAX_ROWS) return NextResponse.json({ error: `Too many rows (${records.length}). This API is capped at ${MAX_ROWS}; use the exported Python script for larger jobs.` }, { status: 413, headers: CORS });
  const input = tableFromRecords(records);
  try {
    const out = runPipeline(input, ops, { secondary }).final;
    await audit("etl_api_run", ch.userId, { pipeline: proj.name, mode: "transform_only", rowsIn: records.length, rowsOut: out.rows.length, ops: ops.length }).catch(() => {});
    return NextResponse.json({ ok: true, mode: "transform_only", pipeline: proj.name, cols: out.cols, rows: out.rows, rowsIn: records.length, rowsOut: out.rows.length }, { headers: CORS });
  } catch (e) {
    return NextResponse.json({ error: `Pipeline failed: ${(e as Error).message}` }, { status: 500, headers: CORS });
  }
}
