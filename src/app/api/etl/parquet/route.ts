import { NextResponse } from "next/server";
import { Writable } from "stream";
import { getSessionUser } from "@/lib/auth";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_ROWS = 5000;

// Writes the pipeline output to a real Apache Parquet file (server-side via
// @dsnp/parquetjs — a Node lib; kept out of the client bundle) and streams it
// back as a download. Types are inferred per column (INT64 / DOUBLE / UTF8).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("etlparquet", user.id, 20, 60_000))) return NextResponse.json({ error: "Too many exports — wait a moment." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const cols: string[] = Array.isArray(b.cols) ? b.cols.map(String) : [];
  const rows: Record<string, unknown>[] = Array.isArray(b.rows) ? b.rows : [];
  if (!cols.length || !rows.length) return NextResponse.json({ error: "Nothing to export." }, { status: 400 });
  if (rows.length > MAX_ROWS) return NextResponse.json({ error: `Too many rows (${rows.length}). Cap is ${MAX_ROWS} — add a Limit first.` }, { status: 413 });
  if (JSON.stringify(rows).length > 4_000_000) return NextResponse.json({ error: "Output is too large (over ~4 MB)." }, { status: 413 });

  // Parquet field names can't contain dots (they'd nest); sanitize + dedupe.
  const seen = new Set<string>();
  const keyOf = new Map<string, string>();
  for (const c of cols) { let k = c.replace(/[^A-Za-z0-9_]/g, "_") || "col"; while (seen.has(k)) k += "_"; seen.add(k); keyOf.set(c, k); }

  const infer = (c: string): "INT64" | "DOUBLE" | "UTF8" => {
    let any = false, allNum = true, allInt = true;
    for (const r of rows) { const v = r[c]; if (v == null || v === "") continue; any = true; const n = Number(v); if (typeof v === "number" || (!isNaN(n) && String(v).trim() !== "")) { if (!Number.isInteger(n)) allInt = false; } else { allNum = false; break; } }
    return !any ? "UTF8" : allNum ? (allInt ? "INT64" : "DOUBLE") : "UTF8";
  };
  const kind = new Map<string, "INT64" | "DOUBLE" | "UTF8">();
  const schemaDef: Record<string, { type: "INT64" | "DOUBLE" | "UTF8"; optional: boolean }> = {};
  for (const c of cols) { const t = infer(c); kind.set(c, t); schemaDef[keyOf.get(c)!] = { type: t, optional: true }; }

  try {
    const parquet = await import("@dsnp/parquetjs");
    const schema = new parquet.ParquetSchema(schemaDef);
    const chunks: Buffer[] = [];
    const sink = new Writable({ write(c, _e, cb) { chunks.push(Buffer.from(c)); cb(); } });
    const writer = await parquet.ParquetWriter.openStream(schema, sink as never, { useDataPageV2: false });
    for (const r of rows) {
      const row: Record<string, unknown> = {};
      for (const c of cols) {
        const v = r[c]; if (v == null || v === "") continue;
        const k = keyOf.get(c)!, t = kind.get(c)!;
        row[k] = t === "UTF8" ? String(v) : t === "INT64" ? Math.trunc(Number(v)) : Number(v);
      }
      await writer.appendRow(row);
    }
    await writer.close();
    const buf = Buffer.concat(chunks);
    return new NextResponse(new Uint8Array(buf), { headers: { "content-type": "application/octet-stream", "content-disposition": 'attachment; filename="etl_output.parquet"' } });
  } catch (e) {
    return NextResponse.json({ error: `Parquet export failed: ${(e as Error).message}` }, { status: 500 });
  }
}
