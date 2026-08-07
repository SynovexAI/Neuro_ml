import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider, getProviderById } from "@/lib/providers";
import { rateLimitDb } from "@/lib/ratelimit";
import { checkQuota, recordUsage, estimateTokens } from "@/lib/usage";
import { audit } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

// Streams plain-text tokens from the admin-configured provider (OpenAI-compatible).
// Accepts optional `providerId` in the body to use a specific configured provider.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("chat", user.id, 60, 60_000))) return NextResponse.json({ error: "You're sending requests too fast — wait a moment and try again." }, { status: 429 });

  // Cost guardrail: block once the user's monthly token budget is spent.
  const quota = await checkQuota(user);
  if (!quota.ok) {
    await audit("quota_exceeded", user.id, { used: quota.used, limit: quota.limit });
    return NextResponse.json({ error: `Monthly token limit reached (${quota.used.toLocaleString()} / ${quota.limit.toLocaleString()}). Ask an admin to raise your quota.` }, { status: 429 });
  }

  const body = await req.json().catch(() => ({}));
  let prov: Awaited<ReturnType<typeof getActiveProvider>>;
  try {
    prov = body.providerId
      ? await getProviderById(String(body.providerId), user.id, user.role === "admin")
      : await getActiveProvider(user.id, user.role === "admin");
  } catch {
    return NextResponse.json({ error: "Database error while loading provider config. Please retry." }, { status: 503 });
  }
  if (!prov || !prov.baseUrl)
    return NextResponse.json({ error: "No LLM provider is configured yet. An admin must add one under Admin → Providers." }, { status: 400 });

  const model = body.model || prov.model;
  if (!model) return NextResponse.json({ error: "No model selected for the active provider." }, { status: 400 });

  const isStreaming = body.streaming !== false;
  const lab: string | null = body.lab ? String(body.lab) : "chat";
  const promptText = (body.messages || []).map((m: { content?: unknown }) => String(m?.content ?? "")).join(" ");

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
    // Ask OpenAI-compatible providers to report token usage on the final chunk.
    ...(isStreaming ? { stream_options: { include_usage: true } } : {}),
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
    const j = await upstream.json().catch(() => null) as { choices?: Array<{ message?: { content?: string } }>; usage?: { prompt_tokens?: number; completion_tokens?: number } } | null;
    const text = j?.choices?.[0]?.message?.content ?? "";
    const u = j?.usage;
    void recordUsage({
      userId: user.id, lab, model,
      promptTokens: u?.prompt_tokens ?? estimateTokens(promptText),
      completionTokens: u?.completion_tokens ?? estimateTokens(text),
      estimated: !u,
    });
    return new Response(text, { headers: { "content-type": "text/plain; charset=utf-8" } });
  }

  const reader = upstream.body.getReader();
  const dec = new TextDecoder();
  const enc = new TextEncoder();
  let buf = "";

  // Eager reader (avoids pull-backpressure stalls) that ALWAYS terminates:
  // closes on [DONE], upstream end, or an idle timeout. After finish_reason we
  // keep reading briefly to capture the provider's usage chunk for metering.
  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      let graced = false;
      let idle: ReturnType<typeof setTimeout>;
      // token metering (real usage from the provider, else estimated from text)
      let completionChars = 0;
      let promptTok: number | undefined;
      let complTok: number | undefined;
      const finish = () => {
        if (closed) return;
        closed = true;
        clearTimeout(idle);
        void recordUsage({
          userId: user.id, lab, model,
          promptTokens: promptTok ?? estimateTokens(promptText),
          completionTokens: complTok ?? Math.ceil(completionChars / 4),
          estimated: promptTok == null || complTok == null,
        });
        try { controller.close(); } catch { /* already closed */ }
        reader.cancel().catch(() => {});
      };
      // Safety net: close if no new data arrives. Short grace once generation is
      // done (just waiting for the trailing usage chunk), long before that.
      const armIdle = () => { clearTimeout(idle); idle = setTimeout(finish, graced ? 800 : 8000); };

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
          const usg = (j as { usage?: { prompt_tokens?: number; completion_tokens?: number } | null }).usage;
          if (usg) { if (usg.prompt_tokens != null) promptTok = usg.prompt_tokens; if (usg.completion_tokens != null) complTok = usg.completion_tokens; }
          const choice = (j as { choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }> }).choices?.[0];
          const tok = choice?.delta?.content;
          if (tok) { completionChars += tok.length; controller.enqueue(enc.encode(tok)); }
          if (choice?.finish_reason) { graced = true; armIdle(); } // done; briefly await usage chunk
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
