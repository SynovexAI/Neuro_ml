import { describe, it, expect } from "vitest";
import { buildGraph, parseDelegationPlan, runAgentStep, resolveTools } from "../src/lib/orchestrator";
import { reactSystemPrompt, parseReAct, stripReActScaffolding } from "../src/lib/agentTools";
import { computeConfidenceScore } from "../src/lib/agentEval";

const n = (id: string, name: string, nodeType?: string) => ({ id, name, nodeType });

describe("buildGraph — execution plan matches the drawn graph", () => {
  it("linear/sequential/custom is a strict chain, one node per wave", () => {
    const nodes = [n("a", "A"), n("b", "B"), n("c", "C")];
    for (const topo of ["linear", "sequential", "custom"] as const) {
      const g = buildGraph(topo, nodes);
      expect(g.waves).toEqual([["a"], ["b"], ["c"]]);
      expect(g.edges).toEqual([{ from: "a", to: "b" }, { from: "b", to: "c" }]);
    }
  });

  it("consensus runs peers concurrently, then the arbiter", () => {
    const g = buildGraph("consensus", [n("agent_a", "Alpha"), n("agent_b", "Beta"), n("referee", "Arbiter", "synthesizer")]);
    expect(g.waves).toEqual([["agent_a", "agent_b"], ["referee"]]);
    expect(g.arbiterId).toBe("referee");
    expect(g.edges).toEqual([{ from: "agent_a", to: "referee" }, { from: "agent_b", to: "referee" }]);
    expect(g.warnings).toHaveLength(0);
  });

  it("consensus warns instead of silently comparing nothing when a peer is deleted", () => {
    // the old engine read stepOutputs["agent_a"] || "" and produced a confident
    // verdict from an empty perspective
    const g = buildGraph("consensus", [n("agent_a", "Alpha"), n("referee", "Arbiter", "synthesizer")]);
    expect(g.warnings.join(" ")).toMatch(/at least 2 independent agents/i);
  });

  it("hierarchical fans out to specialists in one wave, then converges", () => {
    const g = buildGraph("hierarchical", [
      n("supervisor", "Supervisor"), n("researcher", "Researcher"),
      n("analyst", "Analyst"), n("synthesizer", "Final", "synthesizer"),
    ]);
    expect(g.waves).toEqual([["supervisor"], ["researcher", "analyst"], ["synthesizer"]]);
    expect(g.supervisorId).toBe("supervisor");
    expect(g.edges).toContainEqual({ from: "supervisor", to: "researcher" });
    expect(g.edges).toContainEqual({ from: "analyst", to: "synthesizer" });
    // the supervisor must NOT be wired straight to the synthesizer
    expect(g.edges).not.toContainEqual({ from: "supervisor", to: "synthesizer" });
  });

  it("resolves roles by identity, not array position", () => {
    // supervisor moved to the end — it is still the coordinator
    const g = buildGraph("hierarchical", [n("researcher", "R"), n("analyst", "A"), n("supervisor", "S")]);
    expect(g.supervisorId).toBe("supervisor");
    expect(g.waves[0]).toEqual(["supervisor"]);
  });

  it("degenerate pipelines produce warnings, not crashes", () => {
    expect(buildGraph("linear", []).warnings).toHaveLength(1);
    expect(buildGraph("hierarchical", [n("supervisor", "S")]).waves).toEqual([["supervisor"]]);
    expect(buildGraph("linear", [n("a", "A"), n("a", "A2")]).warnings.join(" ")).toMatch(/duplicate/i);
  });
});

describe("parseDelegationPlan — the supervisor's plan actually routes", () => {
  const specialists = [{ id: "researcher", name: "Fact Researcher" }, { id: "analyst", name: "Data Analyst" }];

  it("maps AGENT lines onto node ids by name", () => {
    const plan = [
      "Here is the plan.",
      "AGENT Fact Researcher: find 2024 revenue figures for the top 5 vendors",
      "- AGENT Data Analyst: compute CAGR from those figures",
      "Overall we will then synthesise.",
    ].join("\n");
    expect(parseDelegationPlan(plan, specialists)).toEqual({
      researcher: "find 2024 revenue figures for the top 5 vendors",
      analyst: "compute CAGR from those figures",
    });
  });

  it("matches loosely on case/punctuation and ignores unmatched names", () => {
    const plan = "AGENT fact-researcher: dig\nAGENT Nobody At All: ignored";
    expect(parseDelegationPlan(plan, specialists)).toEqual({ researcher: "dig" });
  });

  it("returns nothing for an unformatted plan rather than misrouting", () => {
    expect(parseDelegationPlan("Just a prose plan with no assignments.", specialists)).toEqual({});
  });
});

describe("runAgentStep — tools actually execute", () => {
  it("calls a real tool and grounds the answer in its observation", async () => {
    const replies = [
      "Thought: I should compute this.\nAction: calculator\nAction Input: 2*(3+4)^2",
      "Thought: I have the number.\nFinal Answer: The result is 98.",
    ];
    let i = 0;
    const res = await runAgentStep({
      node: { name: "Calc", role: "math", systemPrompt: "you do math", tools: ["calculator"] },
      input: "what is 2*(3+4)^2",
      callLLM: async () => replies[i++],
    });
    expect(res.status).toBe("done");
    expect(res.toolCalls).toBe(1);
    expect(res.toolErrors).toBe(0);
    // the observation is the REAL evaluator output, not model-invented
    const obs = res.trace.find((t) => t.kind === "observation");
    expect(obs?.text).toBe("98");
    expect(obs?.tool).toBe("calculator");
  });

  it("counts tool failures instead of passing them off as success", async () => {
    const replies = [
      "Thought: try it.\nAction: calculator\nAction Input: this is not math",
      "Thought: done.\nFinal Answer: could not compute",
    ];
    let i = 0;
    const res = await runAgentStep({
      node: { name: "Calc", role: "math", systemPrompt: "", tools: ["calculator"] },
      input: "x", callLLM: async () => replies[i++],
    });
    expect(res.toolCalls).toBe(1);
    expect(res.toolErrors).toBe(1);
  });

  it("reports non-convergence as an error rather than a finished answer", async () => {
    const res = await runAgentStep({
      node: { name: "Loop", role: "r", systemPrompt: "", tools: ["calculator"] },
      input: "x",
      maxIters: 2,
      callLLM: async () => "Thought: again.\nAction: calculator\nAction Input: 1+1",
    });
    expect(res.status).toBe("error");
    expect(res.outcome).toBe("max_iters");
    expect(res.iterations).toBe(2);
    expect(res.output).toMatch(/did not converge/i);
  });

  it("flags tools the node is configured with but that do not exist", async () => {
    const res = await runAgentStep({
      node: { name: "X", role: "r", systemPrompt: "", tools: ["calculator", "teleporter"] },
      input: "x",
      callLLM: async () => "Thought: no tool needed.\nFinal Answer: done",
    });
    expect(res.unknownTools).toEqual(["teleporter"]);
    expect(res.trace[0]).toMatchObject({ kind: "error", tool: "teleporter" });
  });

  it("a tool-less node reports zero tool calls, not a silent default", async () => {
    const res = await runAgentStep({
      node: { name: "Planner", role: "plan", systemPrompt: "", tools: [] },
      input: "x", callLLM: async () => "a plan",
    });
    expect(res.toolCalls).toBe(0);
    expect(res.trace).toEqual([{ kind: "final", text: "a plan" }]);
  });

  it("aborts mid-run when signalled", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(runAgentStep({
      node: { name: "X", role: "r", systemPrompt: "", tools: [] },
      input: "x", callLLM: async () => "never", signal: ctrl.signal,
    })).rejects.toThrow(/cancelled/i);
  });

  it("preset tool names resolve against the real registry", () => {
    const { tools, unknown } = resolveTools(["web_search", "arxiv", "calculator", "db_query", "wikipedia", "datetime", "statistics"]);
    expect(unknown).toEqual([]);
    expect(tools).toHaveLength(7);
  });
});

describe("confidence reports what it measured", () => {
  const noTools = { finalAnswer: "an answer of reasonable length ".repeat(6), trace: [], iterations: 1, maxIters: 3, task: "t" };

  it("does not invent tool reliability when no tool ran", () => {
    const m = computeConfidenceScore(noTools);
    // previously hardcoded to 90 on every tool-less run
    expect(m.toolReliability).toBeNull();
    expect(m.grounding).toBeNull();
    expect(m.explanation).toMatch(/not measured/i);
  });

  it("measures both once tools have actually run", () => {
    const m = computeConfidenceScore({
      ...noTools,
      trace: [
        { kind: "action", tool: "calculator" },
        { kind: "observation", text: "98 revenue growth figures" },
      ],
    });
    expect(m.toolReliability).toBe(100);
    expect(typeof m.grounding).toBe("number");
  });

  it("penalises tool errors", () => {
    const m = computeConfidenceScore({
      ...noTools,
      trace: [
        { kind: "action", tool: "calculator" }, { kind: "observation", text: "Error: bad expression" },
        { kind: "action", tool: "calculator" }, { kind: "observation", text: "Error: bad expression" },
      ],
    });
    expect(m.toolReliability).toBe(20);
  });

  it("a failed run is Uncertain, not Moderate", () => {
    const m = computeConfidenceScore({ ...noTools, outcome: "error" });
    expect(m.label).toBe("Uncertain");
    expect(m.score).toBeLessThan(45);
  });
});

describe("research topology — the continuous chain", () => {
  const RESEARCH = [
    { id: "research_planner", name: "Research Planner" },
    { id: "source_collector", name: "Source Collector" },
    { id: "organiser", name: "Content Organiser" },
    { id: "summariser", name: "Summariser" },
    { id: "reporter", name: "Report Writer", nodeType: "synthesizer" },
  ];

  it("runs strictly in order — each stage feeds the next", () => {
    const g = buildGraph("research", RESEARCH);
    expect(g.waves).toEqual([["research_planner"], ["source_collector"], ["organiser"], ["summariser"], ["reporter"]]);
    expect(g.edges).toEqual([
      { from: "research_planner", to: "source_collector" },
      { from: "source_collector", to: "organiser" },
      { from: "organiser", to: "summariser" },
      { from: "summariser", to: "reporter" },
    ]);
    expect(g.warnings).toHaveLength(0);
  });

  it("the collector's scraping tools all resolve to real implementations", () => {
    const { tools, unknown } = resolveTools(["web_search", "web_fetch", "wikipedia", "arxiv"]);
    expect(unknown).toEqual([]);
    expect(tools.map((t) => t.id)).toEqual(["web_fetch", "web_search", "wikipedia", "arxiv"]);
  });

  it("stays a chain when the user adds their own stage", () => {
    const withExtra = [...RESEARCH.slice(0, 2), { id: "my_critic", name: "My Critic" }, ...RESEARCH.slice(2)];
    const g = buildGraph("research", withExtra);
    expect(g.waves.flat()).toEqual(withExtra.map((n) => n.id));
    expect(g.edges).toContainEqual({ from: "source_collector", to: "my_critic" });
    expect(g.edges).toContainEqual({ from: "my_critic", to: "organiser" });
  });
});

describe("regressions found by the first live research run", () => {
  it("the ReAct prompt only ever names tools the node actually has", () => {
    const { tools } = resolveTools(["web_search", "arxiv"]);
    const prompt = reactSystemPrompt(tools, "research");
    // the worked example used to call a tool named "exa" that is not in the
    // registry — models copied it verbatim and the call errored
    expect(prompt).not.toMatch(/Action:\s*exa\b/);
    expect(prompt).toMatch(/Action: web_search/);
    expect(prompt).toMatch(/never invent a tool name/i);
  });

  it("gives multi-tool nodes room to search AND conclude", async () => {
    // 4 steps meant a 4-tool collector spent every step searching, then failed
    let calls = 0;
    const res = await runAgentStep({
      node: { name: "Collector", role: "gather", systemPrompt: "", tools: ["web_search", "wikipedia", "arxiv", "calculator"] },
      input: "x",
      callLLM: async () => {
        calls++;
        return calls <= 6
          ? "Thought: searching.\nAction: calculator\nAction Input: 1+1"
          : "Thought: enough.\nFinal Answer: done";
      },
    });
    expect(res.maxIters).toBeGreaterThanOrEqual(10);
    expect(res.status).toBe("done");
    expect(res.toolCalls).toBe(6);
  });

  it("warns the agent when it is about to run out of steps", async () => {
    const seen: string[] = [];
    await runAgentStep({
      node: { name: "X", role: "r", systemPrompt: "", tools: ["calculator"] },
      input: "x",
      maxIters: 3,
      callLLM: async (msgs) => {
        seen.push(msgs[msgs.length - 1].content);
        return "Thought: again.\nAction: calculator\nAction Input: 1+1";
      },
    });
    expect(seen.some((m) => /LAST step/i.test(m))).toBe(true);
  });

  it("a step that never concluded is Uncertain, not High Confidence", () => {
    // this run showed "Source Collector ✕ Failed" beside "85% High Confidence"
    const m = computeConfidenceScore({
      finalAnswer: "Toyota claims 1000 Wh/kg by 2027 with 500 Wh/kg from CATL and 12 other figures".repeat(3),
      trace: [
        { kind: "action", tool: "web_search" }, { kind: "observation", text: "Toyota claims 1000 Wh/kg" },
        { kind: "action", tool: "web_search" }, { kind: "observation", text: "CATL 500 Wh/kg by 2027" },
      ],
      iterations: 4, maxIters: 4, outcome: "max_iters", task: "t",
    });
    expect(m.label).toBe("Uncertain");
    expect(m.score).toBeLessThanOrEqual(40);
    expect(m.explanation).toMatch(/did not converge/i);
  });

  it("the pipeline composite is capped when stages failed", () => {
    // 88% "High Confidence" was reported on a run where 2 of 5 stages failed
    const base = {
      finalAnswer: "A confident report with 12 numbers and 2027 dates ".repeat(6),
      trace: [{ kind: "action", tool: "web_search" }, { kind: "observation", text: "grounded evidence text" }],
      iterations: 5, maxIters: 7, outcome: "success", task: "t",
    };
    const clean = computeConfidenceScore(base);
    const degraded = computeConfidenceScore({ ...base, degradedSteps: 2, totalSteps: 5 });
    expect(clean.score).toBeGreaterThan(degraded.score);
    expect(degraded.score).toBeLessThanOrEqual(60);
    expect(degraded.explanation).toMatch(/2 of 5 stages failed/);
  });
});

describe("regressions from the second live research run", () => {
  it('does not parse "Action: I will search…" as a tool named "I"', () => {
    const raw = "Thought: I need sources.\nAction: I will search for solid-state battery makers";
    // without an allowlist the old parser captured the bare word after "Action:"
    expect(parseReAct(raw).action).toBe("I");
    expect(parseReAct(raw, ["web_search", "wikipedia"]).action).toBeUndefined();
  });

  it("keeps ReAct scaffolding out of the final answer", async () => {
    // this leaked "Action: datetime" straight into the Final Answer card
    const raw = [
      "Key gap: No direct citations for BYD's claim. I'll resolve this.",
      "---",
      "Action: datetime",
    ].join("\n");
    expect(stripReActScaffolding(raw)).toBe("Key gap: No direct citations for BYD's claim. I'll resolve this.");

    // the node has calculator; the model emitted a scaffolding line for a tool it
    // does NOT have, so the allowlist rejects it and the text becomes the answer
    const res = await runAgentStep({
      node: { name: "Reporter", role: "report", systemPrompt: "", tools: ["calculator"] },
      input: "x", callLLM: async () => raw,
    });
    expect(res.output).not.toMatch(/^\s*Action\s*:/im);
    expect(res.output).toContain("Key gap");
  });

  it("replays a repeated identical tool call instead of spending the budget", async () => {
    // the planner called datetime four times in a row
    let calls = 0;
    const res = await runAgentStep({
      node: { name: "P", role: "plan", systemPrompt: "", tools: ["calculator"] },
      input: "x",
      maxIters: 6,
      callLLM: async () => {
        calls++;
        return calls <= 4
          ? "Thought: again.\nAction: calculator\nAction Input: 1+1"
          : "Thought: ok.\nFinal Answer: 2";
      },
    });
    const obs = res.trace.filter((t) => t.kind === "observation");
    expect(obs.length).toBeGreaterThan(1);
    expect(obs[1].text).toMatch(/already ran this exact call/i);
    expect(res.status).toBe("done");
  });

  it("the reporting stages run tool-free so they emit clean prose", async () => {
    const res = await runAgentStep({
      node: { name: "Report Writer", role: "report", systemPrompt: "", tools: [] },
      input: "x", callLLM: async () => "## Key Findings\n- a\n- b",
    });
    expect(res.toolCalls).toBe(0);
    expect(res.output).toContain("## Key Findings");
  });
});
