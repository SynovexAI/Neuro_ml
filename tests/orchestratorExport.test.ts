import { describe, it, expect } from "vitest";
import { writeFileSync, mkdtempSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildPythonExport, type ExportNode } from "../src/lib/orchestrator";

const node = (id: string, name: string, nodeType?: string): ExportNode => ({
  id, name, role: `${name} role`, systemPrompt: `You are ${name}. Say "hi".`,
  tools: ["calculator"], temperature: 0.3, nodeType,
});

const FIXTURES = {
  linear: [node("a", "Alpha"), node("b", "Beta"), node("c", "Gamma", "synthesizer")],
  hierarchical: [node("supervisor", "Supervisor"), node("researcher", "Researcher"), node("analyst", "Analyst"), node("synthesizer", "Final", "synthesizer")],
  consensus: [node("agent_a", "Agent Alpha"), node("agent_b", "Agent Beta"), node("referee", "Arbiter", "synthesizer")],
} as const;

describe("buildPythonExport", () => {
  it.each(Object.keys(FIXTURES) as (keyof typeof FIXTURES)[])(
    "emits syntactically valid Python for %s",
    (topo) => {
      const code = buildPythonExport(topo, `${topo} flow`, [...FIXTURES[topo]], 'Analyse "Q4" revenue');
      const dir = mkdtempSync(join(tmpdir(), "orch-"));
      const file = join(dir, "pipeline.py");
      writeFileSync(file, code, "utf-8");
      // py_compile throws on a SyntaxError — this is what caught the un-escaped \n
      expect(() => execFileSync("python", ["-m", "py_compile", file], { stdio: "pipe" })).not.toThrow();
    }
  );

  it("exports the parallel plan for consensus, not a flat loop", () => {
    const code = buildPythonExport("consensus", "c", [...FIXTURES.consensus], "t");
    expect(code).toContain('WAVES = [["agent_a","agent_b"],["referee"]]');
    expect(code).toContain("asyncio.gather");
    expect(code).toContain('EDGES = [["agent_a","referee"],["agent_b","referee"]]');
  });

  it("keeps real newline escapes literal in the generated source", () => {
    const code = buildPythonExport("linear", "l", [...FIXTURES.linear], "t");
    // the generated Python must contain the two-character escape, not a line break
    expect(code).toContain('"\\n\\n".join(');
    expect(code).not.toContain('"\n\n".join(');
  });
});
