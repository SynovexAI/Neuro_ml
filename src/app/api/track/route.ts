import { NextResponse } from "next/server";
import { getSessionUser, uid } from "@/lib/auth";
import { db } from "@/lib/db";
import { pageViews } from "@/lib/db/schema";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Records one page view for in-app analytics. Best-effort; never blocks the UI.
export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ ok: false }, { status: 204 });
  const { path } = await req.json().catch(() => ({}));
  if (typeof path !== "string" || !path) return NextResponse.json({ ok: false }, { status: 400 });
  try {
    await db.insert(pageViews).values({ id: uid(), userId: u.id, path: path.split("?")[0].slice(0, 255) });
  } catch { /* table may not exist yet */ }
  return NextResponse.json({ ok: true });
}
