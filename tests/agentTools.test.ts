import { describe, it, expect } from "vitest";
import { safeCalc, parseReAct } from "@/lib/agentTools";

describe("safeCalc", () => {
  it("respects operator precedence and parentheses", () => {
    expect(safeCalc("2 + 3 * 4")).toBe(14);
    expect(safeCalc("(2 + 3) * 4")).toBe(20);
    expect(safeCalc("2 ^ 10")).toBe(1024);
    expect(safeCalc("10 % 3")).toBe(1);
  });

  it("supports functions and constants", () => {
    expect(safeCalc("sqrt(16)")).toBe(4);
    expect(safeCalc("max(2, 9, 5)")).toBe(9);
    expect(safeCalc("round(pi)")).toBe(3);
  });

  it("throws on malformed input", () => {
    expect(() => safeCalc("2 +")).toThrow();
    expect(() => safeCalc("nope(")).toThrow();
  });
});

describe("parseReAct", () => {
  it("parses a Thought + Action + Action Input step", () => {
    const p = parseReAct("Thought: I need to add the numbers\nAction: calculator\nAction Input: 2+2");
    expect(p.action).toBe("calculator");
    expect(p.input).toBe("2+2");
    expect(p.thought).toMatch(/add the numbers/);
    expect(p.final).toBeUndefined();
  });

  it("parses a Final Answer and stops", () => {
    const p = parseReAct("Thought: done\nFinal Answer: The result is 42");
    expect(p.final).toBe("The result is 42");
    expect(p.action).toBeUndefined();
  });
});
