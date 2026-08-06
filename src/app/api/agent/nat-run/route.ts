import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { rateLimitDb } from "@/lib/ratelimit";
import { runAgent } from "@/lib/agentRunner";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

// Proxies an agent run to the NAT sidecar (Python service). The provider key is
// decrypted server-side and passed server-to-server only — never to the client.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("natrun", user.id, 20, 60_000))) return NextResponse.json({ error: "Too many runs — wait a moment." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const r = await runAgent({
    userId: user.id,
    task: String(b.task || ""),
    providerId: b.providerId ? String(b.providerId) : undefined,
    model: b.model ? String(b.model) : undefined,
    systemPrompt: b.systemPrompt ? String(b.systemPrompt) : undefined,
    agentType: b.agentType === "tool_calling_agent" ? "tool_calling_agent" : "react_agent",
    tools: Array.isArray(b.tools) ? b.tools : [],
    temperature: b.temperature,
    mcpServerIds: Array.isArray(b.mcpServerIds) ? b.mcpServerIds : [],
    knowledgeBaseIds: Array.isArray(b.knowledgeBaseIds) ? b.knowledgeBaseIds : [],
  });
  if (!r.ok) return NextResponse.json({ error: r.error }, { status: r.status });
  return NextResponse.json({ answer: r.answer, latency_ms: r.latency_ms, tool_names: r.tool_names, unsupported_tools: r.unsupported_tools, profiler: r.profiler, context_used: r.context_used, usage: r.usage });
}
