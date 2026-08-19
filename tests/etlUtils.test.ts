import { describe, it, expect } from "vitest";
import { parseRecords, applyOp, runPipeline, tableFromRecords, toCSV } from "@/lib/etlUtils";

describe("parseRecords (RFC-4180 CSV)", () => {
  it("keeps quoted embedded newlines as one field / row", () => {
    const csv = 'id,note,amount\n1,"hello, world",10\n2,"line one\nline two",20\n3,"say ""hi""",30';
    const t = parseRecords(csv);
    expect(t.cols).toEqual(["id", "note", "amount"]);
    expect(t.rows).toHaveLength(3); // the embedded newline must NOT create a 4th row
    expect(t.rows[1].note).toBe("line one\nline two");
    expect(t.rows[0].note).toBe("hello, world"); // comma inside quotes preserved
    expect(t.rows[2].note).toBe('say "hi"');      // escaped quotes unescaped
  });

  it("coerces numeric cells and blanks to null", () => {
    const t = parseRecords("a,b\n1,\n2,x");
    expect(t.rows[0].a).toBe(1);
    expect(t.rows[0].b).toBeNull();
    expect(t.rows[1].b).toBe("x");
  });

  it("auto-detects a semicolon delimiter", () => {
    const t = parseRecords("a;b;c\n1;2;3");
    expect(t.cols).toEqual(["a", "b", "c"]);
    expect(t.rows[0].c).toBe(3);
  });
});

describe("transforms", () => {
  const t = parseRecords("region,amount\nUS,120\nEU,0\nUS,80\nAPAC,240");

  it("filter keeps matching rows (numeric >)", () => {
    const out = applyOp(t, { id: "1", type: "filter", col: "amount", op: ">", value: "50" });
    expect(out.rows).toHaveLength(3);
    expect(out.rows.every((r) => Number(r.amount) > 50)).toBe(true);
  });

  it("select projects columns", () => {
    const out = applyOp(t, { id: "1", type: "select", cols: ["region"] });
    expect(out.cols).toEqual(["region"]);
  });

  it("runPipeline chains ops and reports the final table", () => {
    const res = runPipeline(t, [
      { id: "1", type: "filter", col: "amount", op: ">", value: "50" },
      { id: "2", type: "select", cols: ["region"] },
    ]);
    expect(res.final.cols).toEqual(["region"]);
    expect(res.final.rows).toHaveLength(3);
    expect(res.stages).toHaveLength(3); // source stage (op: null) + one per op
    expect(res.stages[0].op).toBeNull();
  });
});

describe("tableFromRecords / toCSV round-trip", () => {
  it("builds a table from objects and serializes back to CSV", () => {
    const t = tableFromRecords([{ x: 1, y: "a" }, { x: 2, y: "b" }]);
    expect(t.cols).toEqual(["x", "y"]);
    const csv = toCSV(t);
    const back = parseRecords(csv);
    expect(back.rows).toEqual(t.rows);
  });
});
