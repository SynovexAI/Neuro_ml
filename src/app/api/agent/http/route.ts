import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { resolvesToPrivate, rateLimit } from "@/lib/net";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Generic HTTP tool proxy for the Agent Lab. Only GET/POST are allowed (a teaching
// lab shouldn't let an agent PUT/DELETE against arbitrary external services).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!rateLimit(`http:${user.id}`, 30, 60_000)) return NextResponse.json({ error: "Too many requests — wait a minute." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const url = String(b.url || "");
  const method = String(b.method || "GET").toUpperCase();
  if (!/^https?:\/\//i.test(url)) return NextResponse.json({ error: "Provide a valid http(s) URL." }, { status: 400 });
  if (!["GET", "POST"].includes(method)) return NextResponse.json({ error: "Only GET and POST are allowed." }, { status: 400 });
  let host = "";
  try { host = new URL(url).hostname; } catch { return NextResponse.json({ error: "Invalid URL." }, { status: 400 }); }
  if (await resolvesToPrivate(host)) return NextResponse.json({ error: "Blocked host (internal / private address)." }, { status: 400 });

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 12000);
  try {
    const headers: Record<string, string> = { "user-agent": "AI-Workbench-Agent", accept: "application/json, text/*;q=0.9, */*;q=0.8", ...(b.headers && typeof b.headers === "object" ? b.headers : {}) };
    const hasBody = method === "POST" && b.body != null;
    if (hasBody) headers["content-type"] = "application/json";
    const res = await fetch(url, { method, headers, body: hasBody ? (typeof b.body === "string" ? b.body : JSON.stringify(b.body)) : undefined, signal: ctrl.signal });
    const text = await res.text();
    return NextResponse.json({ status: res.status, text: text.slice(0, 4000) });
  } catch (e) {
    if ((e as Error).name === "AbortError") return NextResponse.json({ error: "Request timed out." }, { status: 504 });
    return NextResponse.json({ error: `Could not reach the endpoint: ${(e as Error).message}` }, { status: 502 });
  } finally { clearTimeout(t); }
}
