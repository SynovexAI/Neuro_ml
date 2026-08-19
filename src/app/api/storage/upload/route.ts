import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { storageConfigured, putFile } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_BYTES = 25 * 1024 * 1024; // 25 MB per upload (raise once background processing is added)

// POST a file (multipart/form-data field "file") → stored in R2 → { key, url }.
// Returns 501 if R2 isn't configured, so callers can fall back to their existing flow.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!storageConfigured()) return NextResponse.json({ error: "Storage is not configured (set BLOB_READ_WRITE_TOKEN or the R2_* env vars)." }, { status: 501 });

  const form = await req.formData().catch(() => null);
  const file = form?.get("file");
  if (!(file instanceof File)) return NextResponse.json({ error: "No file provided (multipart field 'file')." }, { status: 400 });
  if (file.size > MAX_BYTES) return NextResponse.json({ error: `File too large (max ${MAX_BYTES / 1024 / 1024} MB).` }, { status: 413 });

  const buf = Buffer.from(await file.arrayBuffer());
  // Namespaced key: user/<id>/<timestamp-ish>-<sanitized name>. (No Date.* — use a random-ish suffix.)
  const safeName = (file.name || "file").replace(/[^\w.\-]+/g, "_").slice(0, 120);
  const rand = Math.random().toString(36).slice(2, 10);
  const key = `uploads/${user.id}/${rand}-${safeName}`;
  try {
    const { url } = await putFile(key, buf, file.type || "application/octet-stream");
    return NextResponse.json({ ok: true, key, url, size: file.size, name: file.name });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}
