import { NextResponse } from "next/server";
import { getSessionUser, uid } from "@/lib/auth";
import { db } from "@/lib/db";
import { pageViews } from "@/lib/db/schema";
import { audit } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Records one page view for in-app analytics AND the audit log, so the audit
// log is the single "everything" feed: navigation + actions. Best-effort; never blocks the UI.
export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ ok: false }, { status: 204 });
  const { path } = await req.json().catch(() => ({}));
  if (typeof path !== "string" || !path) return NextResponse.json({ ok: false }, { status: 400 });
  const clean = path.split("?")[0].slice(0, 255);
  try {
    await db.insert(pageViews).values({ id: uid(), userId: u.id, path: clean });
  } catch { /* table may not exist yet */ }
  // Also record the navigation in the audit log ("who visited which page").
  await audit("page_view", u.id, { path: clean }).catch(() => {});
  return NextResponse.json({ ok: true });
}
