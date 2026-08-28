// Execution core for the multi-agent Orchestration panel.
//
// Two jobs, deliberately kept out of the React component so both the canvas and
// the runner read the SAME structure:
//
//   1. buildGraph()   — turns a topology + node list into an explicit dependency
//                       graph (edges) and an execution plan (waves). The canvas
//                       draws `edges`; the runner executes `waves`. Previously
//                       each derived its own order, so the picture and the run
//                       could disagree — hierarchical drew a fan-out but ran a
//                       straight line.
//   2. runAgentStep() — runs one node as a real ReAct agent against the real
//                       tool registry, returning the trace it actually produced.
//                       Nodes used to be told "Available tools: web_search" and
//                       given no way to call them.

import {
  AGENT_TOOLS,
  reactSystemPrompt,
  parseReAct,
  formatFinalAnswer,
  stripReActScaffolding,
  type AgentTool,
  type ToolCtx,
} from "./agentTools";

export type TopologyType = "linear" | "research" | "hierarchical" | "sequential" | "consensus" | "custom";

export interface GraphNodeLike {
  id: string;
  name: string;
  nodeType?: string;
}

export interface OrchEdge {
  from: string;
  to: string;
}

export interface OrchGraph {
  /** Dependency edges. The canvas renders exactly these. */
  edges: OrchEdge[];
  /** Execution plan: every id in a wave may run concurrently. */
  waves: string[][];
  supervisorId?: string;
  arbiterId?: string;
  /** Structural problems the user should see rather than silently get wrong results from. */
  warnings: string[];
}

/** Terminal aggregator for a topology, if the user kept one. */
function findTerminal(nodes: GraphNodeLike[], preferredId: string): GraphNodeLike | undefined {
  const byId = nodes.find((n) => n.id === preferredId);
  if (byId) return byId;
  // fall back to the last synthesizer-typed node, then to the last node
  for (let i = nodes.length - 1; i >= 0; i--) {
    if (nodes[i].nodeType === "synthesizer") return nodes[i];
  }
  return nodes[nodes.length - 1];
}

/**
 * Build the dependency graph + execution plan for a topology.
 * Node identity is resolved by role (supervisor / arbiter / synthesizer), not by
 * array position, so reordering nodes in the Build step can't silently change
 * which agent is the coordinator.
 */
export function buildGraph(topology: TopologyType, nodes: GraphNodeLike[]): OrchGraph {
  const warnings: string[] = [];
  if (nodes.length === 0) return { edges: [], waves: [], warnings: ["Pipeline has no agents."] };
  if (nodes.length === 1) return { edges: [], waves: [[nodes[0].id]], warnings };

  const ids = nodes.map((n) => n.id);
  if (new Set(ids).size !== ids.length) warnings.push("Duplicate node ids — outputs may overwrite each other.");

  if (topology === "hierarchical") {
    const supervisor = nodes.find((n) => n.id === "supervisor") || nodes[0];
    const terminal = nodes.find((n) => n.nodeType === "synthesizer" && n.id !== supervisor.id);
    const specialists = nodes.filter((n) => n.id !== supervisor.id && n.id !== terminal?.id);

    if (specialists.length === 0) {
      warnings.push("Supervisor has no specialists to delegate to — it will answer alone.");
      const waves = [[supervisor.id], ...(terminal ? [[terminal.id]] : [])];
      const edges = terminal ? [{ from: supervisor.id, to: terminal.id }] : [];
      return { edges, waves, supervisorId: supervisor.id, warnings };
    }

    const edges: OrchEdge[] = specialists.map((s) => ({ from: supervisor.id, to: s.id }));
    if (terminal) edges.push(...specialists.map((s) => ({ from: s.id, to: terminal.id })));
    const waves = [[supervisor.id], specialists.map((s) => s.id), ...(terminal ? [[terminal.id]] : [])];
    return { edges, waves, supervisorId: supervisor.id, warnings };
  }

  if (topology === "consensus") {
    const arbiter = findTerminal(nodes, "referee")!;
    const peers = nodes.filter((n) => n.id !== arbiter.id);
    if (peers.length < 2) {
      warnings.push(
        `Consensus needs at least 2 independent agents to compare — found ${peers.length}. ` +
          "The arbiter has nothing to weigh against."
      );
    }
    return {
      edges: peers.map((p) => ({ from: p.id, to: arbiter.id })),
      waves: peers.length ? [peers.map((p) => p.id), [arbiter.id]] : [[arbiter.id]],
      arbiterId: arbiter.id,
      warnings,
    };
  }

  // linear · sequential · custom — a chain, matching the wires the canvas draws.
  const edges: OrchEdge[] = [];
  for (let i = 0; i < nodes.length - 1; i++) edges.push({ from: nodes[i].id, to: nodes[i + 1].id });
  return { edges, waves: nodes.map((n) => [n.id]), warnings };
}

/**
 * Pull per-specialist assignments out of a supervisor's delegation plan.
 * The supervisor is asked to emit `AGENT <name>: <subtask>` lines; this maps them
 * back onto real node ids by name (case/punctuation-insensitive, prefix match).
 * Returns only what it could actually match — callers fall back to the whole plan
 * for anyone unmatched, so a badly-formatted plan degrades instead of misrouting.
 */
export function parseDelegationPlan(
  plan: string,
  specialists: { id: string; name: string }[]
): Record<string, string> {
  const out: Record<string, string> = {};
  if (!plan) return out;
  const norm = (v: string) => v.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

  const lines = plan.split(/\r?\n/);
  for (const line of lines) {
    const m = line.match(/^\s*(?:[-*\d.)\s]*)?AGENT\s+([^:]{1,60}?)\s*:\s*(.+)$/i);
    if (!m) continue;
    const who = norm(m[1]);
    const task = m[2].trim();
    if (!who || !task) continue;
    const hit =
      specialists.find((s) => norm(s.name) === who) ||
      specialists.find((s) => norm(s.id) === who) ||
      specialists.find((s) => norm(s.name).startsWith(who) || who.startsWith(norm(s.name)));
    if (hit && !out[hit.id]) out[hit.id] = task;
  }
  return out;
}

// ── per-node execution ──────────────────────────────────────────────────────

export interface StepTraceItem {
  kind: "thought" | "action" | "observation" | "final" | "error";
  text?: string;
  tool?: string;
}

export interface StepOutcome {
  output: string;
  trace: StepTraceItem[];
  status: "done" | "error";
  /** "success" | "max_iters" | "error" — fed to the confidence scorer. */
  outcome: string;
  iterations: number;
  maxIters: number;
  toolCalls: number;
  toolErrors: number;
  /** Tools the node was configured with but that don't exist in the registry. */
  unknownTools: string[];
}

export function resolveTools(names: string[] | undefined): { tools: AgentTool[]; unknown: string[] } {
  const wanted = names || [];
  const tools = AGENT_TOOLS.filter((t) => wanted.includes(t.id));
  const known = new Set(tools.map((t) => t.id));
  return { tools, unknown: wanted.filter((n) => !known.has(n)) };
}

export interface RunStepOptions {
  node: { name: string; role: string; systemPrompt: string; tools?: string[] };
  input: string;
  callLLM: (msgs: { role: string; content: string }[], maxTok: number) => Promise<string>;
  toolCtx?: ToolCtx;
  maxIters?: number;
  maxTokens?: number;
  signal?: AbortSignal;
  /** Streamed trace, so the UI can show tool calls as they happen. */
  onTrace?: (item: StepTraceItem) => void;
}

/**
 * Run a single agent node. With tools configured it runs a real ReAct loop
 * against the shared registry; with none it is a single completion. Either way
 * the returned trace reflects what actually happened — nothing is synthesised.
 */
export async function runAgentStep(opts: RunStepOptions): Promise<StepOutcome> {
  const { node, input, callLLM, toolCtx, signal, onTrace } = opts;
  const maxTokens = opts.maxTokens ?? 700;
  const { tools, unknown } = resolveTools(node.tools);
  // Budget steps by how many tools the node has: a collector with 4 search tools
  // needs several calls AND a turn to write the answer. A flat 4 meant it spent
  // every step searching and got marked failed for never concluding.
  const maxIters = opts.maxIters ?? Math.min(12, 4 + tools.length * 2);
  const trace: StepTraceItem[] = [];
  const push = (i: StepTraceItem) => { trace.push(i); onTrace?.(i); };
  const abort = () => { if (signal?.aborted) throw new DOMException("Run cancelled", "AbortError"); };

  for (const u of unknown) push({ kind: "error", text: `Configured tool "${u}" is not available in this workspace.`, tool: u });

  // No tools — a plain completion. Reported honestly as zero tool calls.
  if (tools.length === 0) {
    abort();
    const out = (await callLLM(
      [
        { role: "system", content: `${node.systemPrompt}\nSpecialist Role: ${node.role}\nYou have no tools available — answer from reasoning alone and say so if the task needs live data. Format cleanly with Markdown.` },
        { role: "user", content: input },
      ],
      maxTokens
    )).trim();
    push({ kind: "final", text: out });
    return { output: out, trace, status: "done", outcome: "success", iterations: 1, maxIters: 1, toolCalls: 0, toolErrors: 0, unknownTools: unknown };
  }

  const msgs: { role: string; content: string }[] = [
    { role: "system", content: reactSystemPrompt(tools, `${node.role}. ${node.systemPrompt}`) },
    { role: "user", content: input },
  ];

  let toolCalls = 0;
  let toolErrors = 0;
  const seenCalls = new Map<string, string>();
  let iterations = 0;

  for (let i = 0; i < maxIters; i++) {
    abort();
    iterations = i + 1;
    const raw = await callLLM(msgs, maxTokens);
    const parsed = parseReAct(raw, tools.map((t) => t.id));

    if (parsed.thought) push({ kind: "thought", text: parsed.thought });

    if (parsed.final !== undefined) {
      const out = formatFinalAnswer(parsed.final).trim();
      push({ kind: "final", text: out });
      return { output: out, trace, status: "done", outcome: "success", iterations, maxIters, toolCalls, toolErrors, unknownTools: unknown };
    }

    if (!parsed.action) {
      // Model answered prose instead of the ReAct block — take it as the answer
      // rather than burning iterations on a format it isn't following, but strip
      // any leftover protocol lines so they don't surface in the Final Answer.
      const out = stripReActScaffolding(raw) || raw.trim();
      push({ kind: "final", text: out });
      return { output: out, trace, status: "done", outcome: "success", iterations, maxIters, toolCalls, toolErrors, unknownTools: unknown };
    }

    const tool = tools.find((t) => t.id === parsed.action || t.name === parsed.action);
    push({ kind: "action", text: parsed.input || "", tool: parsed.action });
    toolCalls++;

    const callKey = `${parsed.action}::${(parsed.input || "").trim()}`;
    let observation: string;
    if (seenCalls.has(callKey)) {
      // identical repeat (a planner calling datetime four times) — replay the
      // cached result and push it to move on rather than spending a real call
      observation = `${seenCalls.get(callKey)}

[System: you already ran this exact call. Do not repeat it — use the result and continue, or give your Final Answer.]`;
    } else if (!tool) {
      observation = `Error: unknown tool "${parsed.action}". Available: ${tools.map((t) => t.id).join(", ")}`;
      toolErrors++;
    } else {
      try {
        abort();
        observation = await tool.run(parsed.input || "", toolCtx || {});
        if (/^error\b/i.test(observation.trim())) toolErrors++;
      } catch (e) {
        observation = `Error: ${(e as Error).message}`;
        toolErrors++;
      }
    }
    if (!seenCalls.has(callKey)) seenCalls.set(callKey, observation);
    push({ kind: "observation", text: observation, tool: parsed.action });
    msgs.push({ role: "assistant", content: raw });
    const left = maxIters - (i + 1);
    const nudge = left <= 1
      ? `

[System: this is your LAST step. Reply with "Final Answer:" now, using what you already have.]`
      : left === 2
      ? `

[System: ${left} steps remain — start wrapping up.]`
      : "";
    msgs.push({ role: "user", content: `Observation: ${observation}${nudge}` });
  }

  // Ran out of iterations with no Final Answer — say so instead of passing off
  // the last partial thought as a finished result.
  abort();
  const lastObs = [...trace].reverse().find((t) => t.kind === "observation")?.text || "";
  const out = lastObs
    ? `_Step did not converge within ${maxIters} reasoning steps. Last tool observation:_\n\n${lastObs}`
    : `_Step did not converge within ${maxIters} reasoning steps._`;
  push({ kind: "error", text: `Hit the ${maxIters}-step reasoning limit without a final answer.` });
  return { output: out, trace, status: "error", outcome: "max_iters", iterations, maxIters, toolCalls, toolErrors, unknownTools: unknown };
}

// ── runnable export ─────────────────────────────────────────────────────────

export interface ExportNode {
  id: string; name: string; role: string; systemPrompt: string;
  tools: string[]; temperature: number; nodeType?: string;
}

/**
 * Emit a standalone Python script for the current pipeline.
 * Lives here rather than in the component so it can be compile-checked in tests —
 * it is a template literal producing another language's source, and a single
 * un-escaped "
" silently emits broken Python.
 */
export function buildPythonExport(
  topology: TopologyType,
  title: string,
  nodes: ExportNode[],
  task: string
): string {
  const graph = buildGraph(topology, nodes);
  return `# Multi-Agent Orchestration Pipeline: ${title}
# Topology: ${topology.toUpperCase()} | Active Stages: ${nodes.length}

import asyncio
from typing import TypedDict, Dict, Any, List
from openai import AsyncOpenAI

client = AsyncOpenAI()

class PipelineState(TypedDict):
    initial_task: str
    stage_results: Dict[str, Any]

# ── Agent Node Specifications ──
AGENTS = [
${nodes.map((n) => `    {
        "id": "${n.id}",
        "name": "${n.name}",
        "role": "${n.role}",
        "system_prompt": """${n.systemPrompt.replace(/"/g, '\\"')}""",
        "tools": ${JSON.stringify(n.tools)},
        "temperature": ${n.temperature},
    }`).join(",\n")}
]

async def execute_agent_step(agent: dict, context: str) -> str:
    """Executes a single specialist agent step with LLM reasoning."""
    print(f"[*] Running step: {agent['name']} ({agent['role']})...")
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=agent["temperature"],
        messages=[
            {"role": "system", "content": agent["system_prompt"]},
            {"role": "user", "content": f"Accumulated Context:\\n{context}"}
        ]
    )
    output = response.choices[0].message.content or ""
    return output

# ── Dependency graph (mirrors the wires on the canvas) ──
EDGES = ${JSON.stringify(graph.edges.map((e) => [e.from, e.to]))}
# Execution plan: agents inside a wave are independent and run concurrently.
WAVES = ${JSON.stringify(graph.waves)}

def _deps(node_id: str) -> List[str]:
    return [a for a, b in EDGES if b == node_id]

def _build_input(node_id: str, task: str, results: Dict[str, str], names: Dict[str, str]) -> str:
    deps = _deps(node_id)
    if not deps:
        return f'User Task: "{task}"'
    context = "\\n\\n".join(
        f"### {names.get(d, d)}\\n{results.get(d, '')}" for d in deps
    )
    return f'User Task: "{task}"\\n\\n--- Context from upstream agents ---\\n{context}'

async def run_orchestration_pipeline(task: str):
    """Executes the ${topology} multi-agent workflow wave by wave."""
    by_id = {a["id"]: a for a in AGENTS}
    names = {a["id"]: a["name"] for a in AGENTS}
    results: Dict[str, str] = {}

    print(f"=== Starting Multi-Agent ${topology.toUpperCase()} Flow ===")
    print(f"Goal: {task}\\n")

    for wave in WAVES:
        outs = await asyncio.gather(*[
            execute_agent_step(by_id[nid], _build_input(nid, task, results, names))
            for nid in wave if nid in by_id
        ])
        for nid, out in zip(wave, outs):
            results[nid] = out
            print(f"[done] {by_id[nid]['name']}\\n")

    print("=== Multi-Agent Synthesis Complete ===")
    return {"initial_task": task, "stage_results": results}

if __name__ == "__main__":
    test_task = """${task.replace(/"/g, '\\"')}"""
    asyncio.run(run_orchestration_pipeline(test_task))
`;
}
