import type { RunInput } from "@/lib/agentRunner";

// Maps a saved agent project config (from the Agent Lab) into the fields
// runAgent expects. Callers add userId + task.
export function cfgToRunInput(config: unknown): Omit<RunInput, "userId" | "task"> {
  const c = (config || {}) as Record<string, unknown>;
  const arr = (v: unknown) => (Array.isArray(v) ? v.filter((x) => typeof x === "string") as string[] : []);
  return {
    providerId: typeof c.providerId === "string" ? c.providerId : undefined,
    model: typeof c.model === "string" ? c.model : undefined,
    systemPrompt: typeof c.systemPrompt === "string" ? c.systemPrompt : undefined,
    agentType: c.agentType === "tool_calling_agent" ? "tool_calling_agent" : "react_agent",
    tools: arr(c.tools),
    temperature: typeof c.temperature === "number" ? c.temperature : 0.2,
    mcpServerIds: arr(c.mcpIds),
    knowledgeBaseIds: arr(c.kbIds),
  };
}
