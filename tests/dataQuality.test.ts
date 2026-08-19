import { describe, it, expect } from "vitest";
import { parseCSV } from "@/lib/mlUtils";
import {
  validateDataset, overallValidationStatus, assessQuality, detectLeakage, snapshotSchema,
} from "@/lib/dataQuality";

describe("validateDataset", () => {
  it("passes clean data with no issues", () => {
    const ds = parseCSV("age,income,churn\n25,50000,no\n30,60000,yes\n45,80000,no\n50,90000,yes\n33,55000,no");
    const issues = validateDataset(ds, "churn");
    expect(overallValidationStatus(issues)).toBe("PASS");
  });

  it("flags missing target values as ERROR", () => {
    const ds = parseCSV("age,churn\n25,no\n30,\n45,no");
    const issues = validateDataset(ds, "churn");
    const err = issues.find((i) => i.column === "churn" && i.severity === "ERROR");
    expect(err).toBeTruthy();
    expect(overallValidationStatus(issues)).toBe("ERROR");
  });

  it("flags impossible negative values for non-negative-hinted columns", () => {
    const ds = parseCSV("age,churn\n-5,no\n30,yes\n45,no\n50,yes");
    const issues = validateDataset(ds, "churn");
    expect(issues.some((i) => i.column === "age" && i.category === "Value" && i.severity === "ERROR")).toBe(true);
  });

  it("flags duplicate rows", () => {
    const ds = parseCSV("a,b\n1,2\n1,2\n3,4");
    const issues = validateDataset(ds, "b");
    expect(issues.some((i) => i.category === "Duplicate")).toBe(true);
  });

  it("flags duplicate values in an id-like column", () => {
    const ds = parseCSV("customer_id,churn\n1,no\n1,yes\n2,no");
    const issues = validateDataset(ds, "churn");
    expect(issues.some((i) => i.column === "customer_id" && i.severity === "ERROR" && i.category === "Constraint")).toBe(true);
  });

  it("detects schema drift against a reference snapshot", () => {
    const dsA = parseCSV("a,b,c\n1,2,x\n3,4,y");
    const ref = snapshotSchema(dsA);
    const dsB = parseCSV("a,c,d\n1,x,9\n3,y,8"); // b dropped, d added
    const issues = validateDataset(dsB, "c", ref);
    expect(issues.some((i) => i.category === "Schema" && i.message.includes('"b"') && i.severity === "ERROR")).toBe(true);
    expect(issues.some((i) => i.category === "Schema" && i.message.includes('"d"') && i.severity === "WARNING")).toBe(true);
  });
});

describe("assessQuality", () => {
  it("scores a clean dataset near 100", () => {
    const rows = Array.from({ length: 50 }, (_, i) => `${20 + (i % 40)},${1000 + i * 7},${i % 2 === 0 ? "no" : "yes"}`).join("\n");
    const ds = parseCSV(`age,income,churn\n${rows}`);
    const q = assessQuality(ds, "churn");
    expect(q.overall).toBeGreaterThanOrEqual(85);
    expect(q.missingPct).toBe(0);
    expect(q.duplicatePct).toBe(0);
  });

  it("penalizes missing values and flags a constant column", () => {
    const ds = parseCSV("age,flag,churn\n25,1,no\n,1,yes\n45,1,no\n,1,yes\n33,1,no");
    const q = assessQuality(ds, "churn");
    expect(q.missingPct).toBeGreaterThan(0);
    expect(q.flags.some((f) => f.column === "flag" && f.kind === "constant")).toBe(true);
    expect(q.overall).toBeLessThan(100);
  });

  it("flags a high-cardinality categorical column", () => {
    const rows = Array.from({ length: 40 }, (_, i) => `user_${i},x`).join("\n");
    const ds = parseCSV(`name,churn\n${rows}`);
    const q = assessQuality(ds, "churn");
    expect(q.flags.some((f) => f.column === "name" && f.kind === "high-cardinality")).toBe(true);
  });
});

describe("detectLeakage", () => {
  it("flags a numeric feature that is a near-perfect copy of a regression target", () => {
    const leakCsv = `idx,price,price_copy\n${Array.from({ length: 30 }, (_, i) => `${i},${i * 2},${i * 2}`).join("\n")}`;
    const dsLeak = parseCSV(leakCsv);
    const warnings = detectLeakage(dsLeak, "price", "regression");
    const w = warnings.find((x) => x.column === "price_copy");
    expect(w).toBeTruthy();
    expect(w?.risk).toBe("HIGH");
  });

  it("flags a categorical feature that perfectly separates a classification target", () => {
    const rows = Array.from({ length: 20 }, (_, i) => `${i},${i % 2 === 0 ? "A" : "B"},${i % 2 === 0 ? "yes" : "no"}`).join("\n");
    const ds = parseCSV(`idx,segment,churn\n${rows}`);
    const warnings = detectLeakage(ds, "churn", "classification");
    const w = warnings.find((x) => x.column === "segment");
    expect(w).toBeTruthy();
    expect(w?.risk).toBe("HIGH");
  });

  it("flags a column whose name references the target and an outcome term", () => {
    const ds = parseCSV("age,churn_date,churn\n25,2024-01-01,yes\n30,,no\n45,2024-02-01,yes\n50,,no");
    const warnings = detectLeakage(ds, "churn", "classification");
    const w = warnings.find((x) => x.column === "churn_date");
    expect(w).toBeTruthy();
    expect(w?.risk).toBe("HIGH");
  });

  it("flags id-like columns as low-risk", () => {
    const ds = parseCSV("customer_id,age,churn\n1,25,no\n2,30,yes\n3,45,no\n4,50,yes");
    const warnings = detectLeakage(ds, "churn", "classification");
    expect(warnings.some((w) => w.column === "customer_id" && w.kind === "id-like")).toBe(true);
  });

  it("does not flag an ordinary weakly-related numeric feature", () => {
    const rows = Array.from({ length: 30 }, (_, i) => `${20 + (i % 15)},${i % 2 === 0 ? "yes" : "no"}`).join("\n");
    const ds = parseCSV(`age,churn\n${rows}`);
    const warnings = detectLeakage(ds, "churn", "classification");
    expect(warnings.find((w) => w.column === "age")).toBeUndefined();
  });
});
