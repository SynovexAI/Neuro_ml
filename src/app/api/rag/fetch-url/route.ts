import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function stripHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<\/(p|div|li|h[1-6]|br|tr|section|article)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&").replace(/&lt;/gi, "<").replace(/&gt;/gi, ">").replace(/&quot;/gi, '"').replace(/&#39;/gi, "'")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { url } = await req.json().catch(() => ({}));
  if (!/^https?:\/\//i.test(url || "")) return NextResponse.json({ error: "Enter a valid http(s) URL." }, { status: 400 });
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 15000);
    const res = await fetch(url, { signal: ctrl.signal, headers: { "user-agent": "Mozilla/5.0 (AI-Workbench RAG fetcher)" } });
    clearTimeout(t);
    const ct = res.headers.get("content-type") || "";
    const raw = await res.text();
    const isHtml = ct.includes("html") || /<html|<!doctype html/i.test(raw.slice(0, 500));
    const title = (raw.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] || url).trim().slice(0, 120);
    const text = (isHtml ? stripHtml(raw) : raw).slice(0, 200000);
    return NextResponse.json({ title, text });
  } catch (e) {
    return NextResponse.json({ error: `Could not fetch the page: ${(e as Error).message}` }, { status: 502 });
  }
}
