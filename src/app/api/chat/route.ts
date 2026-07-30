import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { rateLimit } from "@/lib/net";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Streams plain-text tokens from the admin-configured provider (OpenAI-compatible).
// Accepts optional `providerId` in the body to use a specific configured provider.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!rateLimit(`chat:${user.id}`, 60, 60_000)) return NextResponse.json({ error: "You're sending requests too fast — wait a moment and try again." }, { status: 429 });

  const body = await req.json().catch(() => ({}));
  let prov: Awaited<ReturnType<typeof getActiveProvider>>;
  try {
    prov = body.providerId
      ? await getProviderById(String(body.providerId))
      : await getActiveProvider();
  } catch {
    return NextResponse.json({ error: "Database error while loading provider config. Please retry." }, { status: 503 });
  }
  if (!prov || !prov.baseUrl)
    return NextResponse.json({ error: "No LLM provider is configured yet. An admin must add one under Admin → Providers." }, { status: 400 });

  const model = body.model || prov.model;
  if (!model) return NextResponse.json({ error: "No model selected for the active provider." }, { status: 400 });

  const isStreaming = body.streaming !== false;

  const payload = {
    model,
    messages: body.messages || [],
    temperature: body.temperature ?? 0.7,
    max_tokens: body.maxTokens ?? 512,
    top_p: body.topP ?? 1,
    ...(body.frequencyPenalty != null && body.frequencyPenalty !== 0 ? { frequency_penalty: Number(body.frequencyPenalty) } : {}),
    ...(body.presencePenalty != null && body.presencePenalty !== 0 ? { presence_penalty: Number(body.presencePenalty) } : {}),
    ...(Array.isArray(body.stop) && body.stop.length ? { stop: body.stop } : {}),
    ...(body.responseFormat && body.responseFormat !== "text" ? { response_format: { type: body.responseFormat as string } } : {}),
    stream: isStreaming,
  };

  let upstream: Response;
  try {
    upstream = await fetch(prov.baseUrl.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", ...(prov.apiKey ? { Authorization: `Bearer ${prov.apiKey}` } : {}) },
      body: JSON.stringify(payload),
      signal: req.signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return new Response(null, { status: 499 });
    return NextResponse.json({ error: `Could not reach provider: ${(e as Error).message}` }, { status: 502 });
  }
  if (!upstream.ok || !upstream.body) {
    const t = await upstream.text().catch(() => "");
    return NextResponse.json({ error: `Provider error ${upstream.status}: ${t.slice(0, 300)}` }, { status: 502 });
  }

  // Non-streaming: return the full completion text.
  if (!isStreaming) {
    const j = await upstream.json().catch(() => null) as { choices?: Array<{ message?: { content?: string } }> } | null;
    const text = j?.choices?.[0]?.message?.content ?? "";
    return new Response(text, { headers: { "content-type": "text/plain; charset=utf-8" } });
  }

  const reader = upstream.body.getReader();
  const dec = new TextDecoder();
  const enc = new TextEncoder();
  let buf = "";

  // Eager reader (avoids pull-backpressure stalls) that ALWAYS terminates:
  // closes on finish_reason (OpenAI-standard), [DONE], upstream end, or an idle timeout.
  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      let idle: ReturnType<typeof setTimeout>;
      const finish = () => {
        if (closed) return;
        closed = true;
        clearTimeout(idle);
        try { controller.close(); } catch { /* already closed */ }
        reader.cancel().catch(() => {});
      };
      // Safety net: close after 8s with no new data if the provider never signals completion.
      const armIdle = () => { clearTimeout(idle); idle = setTimeout(finish, 8000); };

      const emit = (line: string): boolean => {
        const l = line.trim();
        if (!l.startsWith("data:")) return false;
        const data = l.slice(5).trim();
        if (data === "[DONE]") return true;
        try {
          const j = JSON.parse(data);
          if (j.error) {
            const msg = (j.error as { message?: string }).message || "Provider stream error";
            controller.enqueue(enc.encode(`\n⚠ ${msg}`));
            return true;
          }
          const choice = (j as { choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }> }).choices?.[0];
          const tok = choice?.delta?.content;
          if (tok) controller.enqueue(enc.encode(tok));
          if (choice?.finish_reason) return true; // generation complete
        } catch { /* keepalive / partial line */ }
        return false;
      };

      armIdle();
      (async () => {
        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            armIdle();
            buf += dec.decode(value, { stream: true });
            const lines = buf.split("\n");
            buf = lines.pop() ?? "";
            for (const line of lines) { if (emit(line)) { finish(); return; } }
          }
          if (buf.trim()) emit(buf);
        } catch { /* upstream aborted */ }
        finish();
      })();
    },
    cancel() { reader.cancel().catch(() => {}); },
  });

  return new Response(stream, { headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-cache", "x-accel-buffering": "no" } });
}
