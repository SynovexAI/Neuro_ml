import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider } from "@/lib/providers";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Streams plain-text tokens from the admin-configured provider (OpenAI-compatible).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  const body = await req.json().catch(() => ({}));
  const prov = await getActiveProvider();
  if (!prov || !prov.baseUrl)
    return NextResponse.json({ error: "No LLM provider is configured yet. An admin must add one under Admin → Providers." }, { status: 400 });

  const model = body.model || prov.model;
  if (!model) return NextResponse.json({ error: "No model selected for the active provider." }, { status: 400 });

  const payload = {
    model,
    messages: body.messages || [],
    temperature: body.temperature ?? 0.7,
    max_tokens: body.maxTokens ?? 512,
    top_p: body.topP ?? 1,
    stream: true,
  };

  let upstream: Response;
  try {
    upstream = await fetch(prov.baseUrl.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", ...(prov.apiKey ? { Authorization: `Bearer ${prov.apiKey}` } : {}) },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    return NextResponse.json({ error: `Could not reach provider: ${(e as Error).message}` }, { status: 502 });
  }
  if (!upstream.ok || !upstream.body) {
    const t = await upstream.text().catch(() => "");
    return NextResponse.json({ error: `Provider error ${upstream.status}: ${t.slice(0, 300)}` }, { status: 502 });
  }

  const reader = upstream.body.getReader();
  const dec = new TextDecoder();
  const enc = new TextEncoder();
  let buf = "";
  const stream = new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) { controller.close(); return; }
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const l = line.trim();
        if (!l.startsWith("data:")) continue;
        const data = l.slice(5).trim();
        if (data === "[DONE]") { controller.close(); return; }
        try {
          const j = JSON.parse(data);
          const tok = j.choices?.[0]?.delta?.content;
          if (tok) controller.enqueue(enc.encode(tok));
        } catch { /* skip keepalive / partial */ }
      }
    },
    cancel() { reader.cancel().catch(() => {}); },
  });

  return new Response(stream, { headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-cache" } });
}
