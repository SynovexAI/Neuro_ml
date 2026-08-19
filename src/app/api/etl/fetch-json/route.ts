import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { resolvesToPrivate } from "@/lib/net";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 30;

// REST/API source: GET a public JSON endpoint and return an array of records.
// Accepts a top-level array, or {data:[…]} / {rows:[…]} / {results:[…]}, or a
// single object (→ one row). SSRF-guarded (no internal/private hosts).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("etljson", user.id, 20, 60_000))) return NextResponse.json({ error: "Too many requests — wait a minute." }, { status: 429 });
  const { url, path } = await req.json().catch(() => ({}));
  if (!/^https?:\/\//i.test(url || "")) return NextResponse.json({ error: "Provide an http(s) URL." }, { status: 400 });
  let host = ""; try { host = new URL(url).hostname; } catch { return NextResponse.json({ error: "Invalid URL." }, { status: 400 }); }
  if (await resolvesToPrivate(host)) return NextResponse.json({ error: "That host is blocked (internal / private address)." }, { status: 400 });

  try {
    const ctrl = new AbortController(); const to = setTimeout(() => ctrl.abort(), 15000);
    const r = await fetch(url, { headers: { accept: "application/json" }, signal: ctrl.signal });
    clearTimeout(to);
    if (!r.ok) return NextResponse.json({ error: `Upstream returned HTTP ${r.status}.` }, { status: 502 });
    let j = await r.json();
    // Optional dot-path into the payload, e.g. "data.items".
    if (path && typeof path === "string") for (const key of path.split(".").map((s: string) => s.trim()).filter(Boolean)) j = j?.[key];
    let records: unknown[];
    if (Array.isArray(j)) records = j;
    else if (Array.isArray(j?.data)) records = j.data;
    else if (Array.isArray(j?.rows)) records = j.rows;
    else if (Array.isArray(j?.results)) records = j.results;
    else if (j && typeof j === "object") records = [j];
    else return NextResponse.json({ error: "Couldn't find an array of records in the response. Try a dot-path (e.g. data.items)." }, { status: 400 });
    if (!records.length) return NextResponse.json({ error: "The response had no records." }, { status: 400 });
    return NextResponse.json({ records: records.slice(0, 5000) });
  } catch (e) {
    return NextResponse.json({ error: `Fetch failed: ${(e as Error).message}` }, { status: 502 });
  }
}
