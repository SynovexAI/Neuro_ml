import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { listFiles, deleteFile, storageBackend, storageConfigured } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

async function admin() { const u = await getSessionUser(); return u && u.role === "admin" ? u : null; }

// GET → { backend, configured, files[], totalBytes }
export async function GET() {
  if (!(await admin())) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const files = await listFiles();
  const totalBytes = files.reduce((a, f) => a + (f.size || 0), 0);
  return NextResponse.json({ backend: storageBackend(), configured: storageConfigured(), files, totalBytes });
}

// DELETE ?url=<blob url>  (or ?key=<r2 key>) → free space
export async function DELETE(req: Request) {
  if (!(await admin())) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("url") || searchParams.get("key") || "";
  if (!target) return NextResponse.json({ error: "url or key required" }, { status: 400 });
  try { await deleteFile(target); return NextResponse.json({ ok: true }); }
  catch (e) { return NextResponse.json({ error: (e as Error).message }, { status: 502 }); }
}
