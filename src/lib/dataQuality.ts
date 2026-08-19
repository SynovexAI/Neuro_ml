// Data Validation, Data Quality Assessment, and Leakage Detection.
// Pure, dependency-free functions over the existing `Dataset` shape (see mlUtils.ts)
// so they can be reused from the client UI. No mocked results — every score and
// warning here is computed directly from the loaded data.

import type { Dataset, Column, ColType, Task } from "./mlUtils";
import { colStats, quantile } from "./mlUtils";

// ──────────────────────────────────────────────────────────────────────────
// Schema snapshot — lets the user "pin" the current schema as a reference and
// later detect schema drift/changes against it (points to §3 "Schema changes").
// ──────────────────────────────────────────────────────────────────────────
export interface SchemaSnapshot {
  savedAt: string;
  columns: { name: string; type: ColType }[];
}
export function snapshotSchema(ds: Dataset): SchemaSnapshot {
  return { savedAt: new Date().toISOString(), columns: ds.columns.map((c) => ({ name: c.name, type: c.type })) };
}

// ──────────────────────────────────────────────────────────────────────────
// Data Validation Engine
// ──────────────────────────────────────────────────────────────────────────
export type Severity = "PASS" | "WARNING" | "ERROR";
export type ValidationCategory = "Schema" | "Type" | "Constraint" | "Value" | "Duplicate";
export interface ValidationIssue {
  id: string;
  severity: Severity;
  category: ValidationCategory;
  column?: string;
  message: string;
}

const NON_NEGATIVE_HINTS = /(age|count|amount|price|salary|income|balance|total|qty|quantity|num_|_num|weight|height|distance|duration|length|size|fee|cost|score|rate|hours|days|years|population|revenue|units)/i;
const ID_HINTS = /(^id$|_id$|^uuid$|_uuid$|^key$|_key$|^pk$|_pk$|^index$)/i;

function isIdLike(name: string): boolean { return ID_HINTS.test(name); }

/** Row-level duplicate detection: exact match across every column's stringified value. */
export function findDuplicateRows(ds: Dataset): { count: number; pct: number; sampleRowIdx: number[] } {
  const seen = new Set<string>();
  const dupIdx: number[] = [];
  for (let i = 0; i < ds.nrows; i++) {
    const key = ds.columns.map((c) => String(c.values[i])).join("¦");
    if (seen.has(key)) dupIdx.push(i); else seen.add(key);
  }
  return { count: dupIdx.length, pct: ds.nrows ? dupIdx.length / ds.nrows : 0, sampleRowIdx: dupIdx.slice(0, 20) };
}

/** Duplicate values within a single column (used for ID / primary-key-like columns). */
function findDuplicateValues(col: Column): number {
  const seen = new Map<string, number>();
  let dup = 0;
  for (const v of col.values) {
    if (v == null) continue;
    const k = String(v);
    const n = (seen.get(k) || 0) + 1;
    seen.set(k, n);
    if (n === 2) dup++; // count each duplicated value once, when it's seen the 2nd time
  }
  return dup;
}

export function validateDataset(ds: Dataset, target: string, reference?: SchemaSnapshot | null): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  let n = 0;
  const push = (severity: Severity, category: ValidationCategory, message: string, column?: string) =>
    issues.push({ id: `v${n++}`, severity, category, column, message });

  // ── Schema validation (only when a reference snapshot exists) ──
  if (reference) {
    const curNames = new Set(ds.columns.map((c) => c.name));
    const refNames = new Set(reference.columns.map((c) => c.name));
    for (const rc of reference.columns) {
      if (!curNames.has(rc.name)) push("ERROR", "Schema", `Column "${rc.name}" was present in the reference schema but is missing from the current dataset.`, rc.name);
    }
    for (const c of ds.columns) {
      if (!refNames.has(c.name)) push("WARNING", "Schema", `Column "${c.name}" is new — it was not present in the reference schema.`, c.name);
      const rc = reference.columns.find((x) => x.name === c.name);
      if (rc && rc.type !== c.type) push("ERROR", "Schema", `Column "${c.name}" changed type: was "${rc.type}", now "${c.type}".`, c.name);
    }
  }

  // ── Dataset-level integrity validation ──
  const seenNames = new Set<string>();
  for (const c of ds.columns) {
    if (!c.name.trim()) push("ERROR", "Schema", "A column has an empty name.");
    if (seenNames.has(c.name)) push("ERROR", "Schema", `Duplicate column name "${c.name}" detected.`, c.name);
    seenNames.add(c.name);
    if (c.values.length !== ds.nrows) push("ERROR", "Schema", `Column contains ${c.values.length} values but the dataset declares ${ds.nrows} rows.`, c.name);
    if (c.type !== "num" && c.type !== "cat") push("ERROR", "Type", `Unsupported column type "${String(c.type)}".`, c.name);
  }
  if (!ds.columns.some((c) => c.name === target)) push("ERROR", "Schema", `Target column "${target || "(none)"}" is missing.`);
  if (ds.nrows < 4) push("ERROR", "Constraint", `Dataset has only ${ds.nrows} row(s); at least 4 are required for a train/test workflow.`);

  // ── Per-column type / constraint / value validation ──
  for (const c of ds.columns) {
    const s = colStats(c);
    const nonNull = c.values.filter((v) => v != null).length;

    // Type validation: for categorical columns, flag when most values actually look numeric
    // (often a sign the column should have been parsed as numeric, or has mixed-in text errors).
    if (c.type === "cat" && nonNull > 0) {
      const numericLike = c.values.filter((v) => v != null && v !== "" && !isNaN(Number(v))).length;
      const frac = numericLike / nonNull;
      if (frac > 0.5 && frac < 1) push("WARNING", "Type", `${(frac * 100).toFixed(0)}% of values look numeric but the column is text — likely mixed/dirty data.`, c.name);
    }

    // Constraint: required / non-null fields
    if (s.missing > 0) {
      const isTarget = c.name === target;
      const pct = (s.missing / c.values.length) * 100;
      push(isTarget ? "ERROR" : (pct > 30 ? "WARNING" : "WARNING"), "Constraint",
        isTarget ? `Target column has ${s.missing} missing value(s) (${pct.toFixed(1)}%) — rows with a missing target cannot be used for training.`
          : `${s.missing} missing value(s) (${pct.toFixed(1)}%).`, c.name);
    }

    // Constraint: uniqueness for ID-like columns
    if (isIdLike(c.name)) {
      const dup = findDuplicateValues(c);
      if (dup > 0) push("ERROR", "Constraint", `Looks like an identifier column but has ${dup} duplicate value(s) — IDs are expected to be unique.`, c.name);
    }

    if (c.type === "num" && s.type === "num") {
      const vals = c.values.filter((v): v is number => v != null);
      const nonFinite = vals.filter((v) => !Number.isFinite(v)).length;
      if (nonFinite > 0) push("ERROR", "Value", `${nonFinite} non-finite numeric value(s) (NaN/Infinity) found — these must be cleaned before training.`, c.name);
      // Constraint: impossible negative values
      if (NON_NEGATIVE_HINTS.test(c.name)) {
        const neg = vals.filter((v) => v < 0).length;
        if (neg > 0) push("ERROR", "Value", `${neg} negative value(s) found in a column that should not be negative given its name.`, c.name);
      }
      // Value: out-of-range via z-score (extreme outliers, separate from the EDA outlier chart)
      if (s.std > 0) {
        const extreme = vals.filter((v) => Math.abs((v - s.mean) / s.std) > 4).length;
        if (extreme > 0) push("WARNING", "Value", `${extreme} value(s) more than 4 standard deviations from the mean — check for data-entry errors.`, c.name);
      }
    }

    if (c.type === "cat" && s.type === "cat") {
      const total = s.count || 1;
      const rare = s.top.filter(([, cnt]) => cnt / total < 0.005 && cnt <= 2);
      if (rare.length > 0 && s.top.length > 3) push("WARNING", "Value", `${rare.length} categor${rare.length === 1 ? "y" : "ies"} occur in <0.5% of rows (e.g. "${rare[0][0]}") — could be typos or noise.`, c.name);
    }
  }

  // ── Duplicate row validation ──
  const dupRows = findDuplicateRows(ds);
  if (dupRows.count > 0) {
    const pct = dupRows.pct * 100;
    push(pct > 10 ? "ERROR" : "WARNING", "Duplicate", `${dupRows.count} duplicate row(s) found (${pct.toFixed(1)}% of the dataset).`);
  }

  if (issues.length === 0) push("PASS", "Schema", "No schema, type, constraint, value, or duplicate issues detected.");
  return issues;
}

export function overallValidationStatus(issues: ValidationIssue[]): Severity {
  if (issues.some((i) => i.severity === "ERROR")) return "ERROR";
  if (issues.some((i) => i.severity === "WARNING")) return "WARNING";
  return "PASS";
}

// ──────────────────────────────────────────────────────────────────────────
// Data Quality Assessment
// ──────────────────────────────────────────────────────────────────────────
export interface ColumnQualityFlag { column: string; kind: "constant" | "near-constant" | "high-cardinality"; detail: string; }
export interface QualityScore {
  overall: number; // 0-100
  completeness: number; // 0-1
  validity: number; // 0-1
  uniqueness: number; // 0-1
  consistency: number; // 0-1
  missingPct: number;
  duplicatePct: number;
  invalidPct: number;
  outlierPct: number;
  flags: ColumnQualityFlag[];
  outlierByColumn: { column: string; pct: number }[];
  recommendations: string[];
}

function iqrOutlierCount(vals: number[]): number {
  if (vals.length < 4) return 0;
  const s = [...vals].sort((a, b) => a - b);
  const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), iqr = q3 - q1;
  if (iqr === 0) return 0;
  const lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;
  return vals.filter((v) => v < lo || v > hi).length;
}

export function assessQuality(ds: Dataset, target?: string): QualityScore {
  const totalCells = ds.nrows * (ds.columns.length || 1);
  let missingCells = 0, invalidCells = 0, outlierCells = 0, inconsistentCols = 0;
  const flags: ColumnQualityFlag[] = [];
  const outlierByColumn: { column: string; pct: number }[] = [];

  for (const c of ds.columns) {
    const s = colStats(c);
    missingCells += s.missing;
    const nonNull = c.values.filter((v) => v != null).length;

    if (c.type === "num" && s.type === "num") {
      const vals = c.values.filter((v): v is number => v != null);
      const outliers = iqrOutlierCount(vals);
      outlierCells += outliers;
      if (nonNull) outlierByColumn.push({ column: c.name, pct: outliers / nonNull });
      if (NON_NEGATIVE_HINTS.test(c.name)) invalidCells += vals.filter((v) => v < 0).length;
      const range = s.max - s.min;
      if (range === 0 && nonNull > 0) { flags.push({ column: c.name, kind: "constant", detail: `every value is ${s.min}` }); }
      else if (s.std > 0 && range > 0 && s.std / (Math.abs(s.mean) || 1) < 0.01) { flags.push({ column: c.name, kind: "near-constant", detail: `almost no variance (std ${s.std.toFixed(4)})` }); }
    } else if (c.type === "cat" && s.type === "cat") {
      const total = s.count || 1;
      const topShare = (s.top[0]?.[1] || 0) / total;
      if (s.unique <= 1 && nonNull > 0) flags.push({ column: c.name, kind: "constant", detail: `only value: "${s.top[0]?.[0] ?? ""}"` });
      else if (topShare > 0.95) flags.push({ column: c.name, kind: "near-constant", detail: `"${s.top[0]?.[0]}" makes up ${(topShare * 100).toFixed(1)}% of values` });
      if (nonNull > 20 && s.unique / nonNull > 0.5 && !isIdLike(c.name)) flags.push({ column: c.name, kind: "high-cardinality", detail: `${s.unique} unique values across ${nonNull} rows` });
      invalidCells += s.top.filter(([, cnt]) => cnt / total < 0.005 && cnt <= 2).reduce((a, [, cnt]) => a + cnt, 0);
    }

    // "consistency": mixed numeric-looking text in a categorical column counts as an inconsistency
    if (c.type === "cat" && nonNull > 0) {
      const numericLike = c.values.filter((v) => v != null && v !== "" && !isNaN(Number(v))).length;
      const frac = numericLike / nonNull;
      if (frac > 0.5 && frac < 1) inconsistentCols++;
    }
  }

  const dupRows = findDuplicateRows(ds);
  const completeness = totalCells ? 1 - missingCells / totalCells : 1;
  const validity = totalCells ? Math.max(0, 1 - (invalidCells + outlierCells) / totalCells) : 1;
  const uniqueness = 1 - dupRows.pct;
  const consistency = ds.columns.length ? 1 - inconsistentCols / ds.columns.length : 1;
  const overall = Math.round(((completeness + validity + uniqueness + consistency) / 4) * 100);

  const recommendations: string[] = [];
  if (missingCells > 0) recommendations.push(`Handle missing values (${((missingCells / totalCells) * 100).toFixed(1)}% of all cells) via imputation or row/column removal in Preprocessing.`);
  if (dupRows.count > 0) recommendations.push(`Remove ${dupRows.count} duplicate row(s) before splitting/training.`);
  const constantCols = flags.filter((f) => f.kind === "constant");
  if (constantCols.length) recommendations.push(`Drop constant column(s) — they carry no signal: ${constantCols.map((f) => f.column).join(", ")}.`);
  const nearConstCols = flags.filter((f) => f.kind === "near-constant");
  if (nearConstCols.length) recommendations.push(`Review near-constant column(s), they add little value: ${nearConstCols.map((f) => f.column).join(", ")}.`);
  const highCardCols = flags.filter((f) => f.kind === "high-cardinality");
  if (highCardCols.length) recommendations.push(`High-cardinality categorical column(s) will explode with one-hot encoding — consider frequency/target encoding or dropping: ${highCardCols.map((f) => f.column).join(", ")}.`);
  const worstOutlierCols = outlierByColumn.filter((o) => o.pct > 0.05).sort((a, b) => b.pct - a.pct);
  if (worstOutlierCols.length) recommendations.push(`Column(s) with >5% outliers by IQR: ${worstOutlierCols.map((o) => `${o.column} (${(o.pct * 100).toFixed(1)}%)`).join(", ")} — consider clipping/winsorizing.`);
  if (target) { const tCol = ds.columns.find((c) => c.name === target); if (tCol) { const ts = colStats(tCol); if (ts.missing > 0) recommendations.push(`Target column "${target}" has ${ts.missing} missing value(s) — those rows must be dropped before training.`); } }
  if (recommendations.length === 0) recommendations.push("No major data-quality issues detected — the dataset looks ready for preprocessing.");

  return {
    overall, completeness, validity, uniqueness, consistency,
    missingPct: totalCells ? missingCells / totalCells : 0,
    duplicatePct: dupRows.pct,
    invalidPct: totalCells ? invalidCells / totalCells : 0,
    outlierPct: totalCells ? outlierCells / totalCells : 0,
    flags, outlierByColumn, recommendations,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Leakage Detection
// ──────────────────────────────────────────────────────────────────────────
export interface LeakageWarning {
  column: string;
  risk: "HIGH" | "MEDIUM" | "LOW";
  kind: "target-leakage" | "post-outcome-name" | "duplicate-of-target" | "perfect-separator" | "id-like";
  reason: string;
}

const OUTCOME_NAME_HINTS = /(outcome|result|resolved|closed|cancell?ed|refund|status_after|post_|future_|final_|response_time|_after$|churn_date|closed_date|resolution|settled|decision)/i;

function pearsonSafe(a: number[], b: number[]): number {
  const n = a.length; if (n < 2) return 0;
  const ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  const denom = Math.sqrt(da * db);
  return denom ? num / denom : 0;
}

/** Weighted "purity": for a categorical feature, how well each category predicts one class. */
function categoricalPurity(featVals: (string | null)[], targetVals: string[]): number {
  const groups = new Map<string, Map<string, number>>();
  let total = 0;
  for (let i = 0; i < featVals.length; i++) {
    const f = featVals[i]; if (f == null) continue;
    const t = targetVals[i];
    if (!groups.has(f)) groups.set(f, new Map());
    const g = groups.get(f)!;
    g.set(t, (g.get(t) || 0) + 1);
    total++;
  }
  if (!total) return 0;
  let purityWeighted = 0;
  for (const g of groups.values()) {
    const groupTotal = [...g.values()].reduce((a, b) => a + b, 0);
    let maxClass = 0;
    for (const count of g.values()) {
      if (count > maxClass) maxClass = count;
    }
    purityWeighted += (maxClass / groupTotal) * groupTotal;
  }
  return purityWeighted / total;
}

const RISK_RANK: Record<LeakageWarning["risk"], number> = { HIGH: 2, MEDIUM: 1, LOW: 0 };

export function detectLeakage(ds: Dataset, target: string, task: Task): LeakageWarning[] {
  const warnings: LeakageWarning[] = [];
  const targetCol = ds.columns.find((c) => c.name === target);
  if (!targetCol) return warnings;
  const targetKeywords = target.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(" ").filter((w) => w.length > 3);

  for (const c of ds.columns) {
    if (c.name === target) continue;

    // Collect every candidate signal for this column, then keep only the strongest —
    // a column shouldn't show up twice (e.g. once by name, once by correlation).
    const candidates: LeakageWarning[] = [];

    // 1. Name-based heuristics — post-outcome / target-derived naming
    const nameLower = c.name.toLowerCase();
    const mentionsTarget = targetKeywords.some((kw) => nameLower.includes(kw)) && nameLower !== target.toLowerCase();
    const looksPostOutcome = OUTCOME_NAME_HINTS.test(c.name);
    if (mentionsTarget && looksPostOutcome) {
      candidates.push({ column: c.name, risk: "HIGH", kind: "target-leakage", reason: `Column name references both the target ("${target}") and an outcome/post-event term — likely only known after the target is decided.` });
    } else if (looksPostOutcome) {
      candidates.push({ column: c.name, risk: "MEDIUM", kind: "post-outcome-name", reason: "Column name suggests it records something that happens after/around the outcome (e.g. a resolution or closing event)." });
    } else if (mentionsTarget) {
      candidates.push({ column: c.name, risk: "LOW", kind: "target-leakage", reason: `Column name references the target ("${target}") — verify it isn't derived from it.` });
    }

    // 2. ID-like columns rarely carry real signal and can act as row-identity leakage
    // in-sample (e.g. leaking through train/test overlap of the same entity).
    if (isIdLike(c.name)) {
      candidates.push({ column: c.name, risk: "LOW", kind: "id-like", reason: "Looks like a row/entity identifier — usually should be excluded from features (no generalizable signal, and can leak entity identity across a naive split)." });
    } else {
      const nonNullIdx = c.values.map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0);
      if (nonNullIdx.length >= 4) {
        if (task === "regression" && c.type === "num" && targetCol.type === "num") {
          const a = nonNullIdx.map((i) => Number(c.values[i]));
          const b = nonNullIdx.map((i) => Number(targetCol.values[i]));
          const r = pearsonSafe(a, b);
          if (Math.abs(r) > 0.98) candidates.push({ column: c.name, risk: "HIGH", kind: "duplicate-of-target", reason: `Correlation with the target is ${r.toFixed(3)} — this feature is almost a direct copy/derivative of the target.` });
          else if (Math.abs(r) > 0.9) candidates.push({ column: c.name, risk: "MEDIUM", kind: "perfect-separator", reason: `Correlation with the target is ${r.toFixed(3)} — suspiciously high for a real-world predictor; verify this isn't computed from the target.` });
        }

        if (task === "classification") {
          const targetVals = nonNullIdx.map((i) => String(targetCol.values[i]));
          if (c.type === "cat") {
            const featVals = nonNullIdx.map((i) => c.values[i] as string);
            const uniqueFeat = new Set(featVals).size;
            if (uniqueFeat > 1 && uniqueFeat < featVals.length * 0.9) {
              const purity = categoricalPurity(featVals, targetVals);
              if (purity > 0.98) candidates.push({ column: c.name, risk: "HIGH", kind: "perfect-separator", reason: `Knowing this feature's value predicts the target correctly ${(purity * 100).toFixed(1)}% of the time — almost certainly leaks the label.` });
              else if (purity > 0.92) candidates.push({ column: c.name, risk: "MEDIUM", kind: "perfect-separator", reason: `This feature predicts the target correctly ${(purity * 100).toFixed(1)}% of the time by itself — unusually strong for a raw feature.` });
            }
          } else if (c.type === "num") {
            // Bin the numeric feature and reuse the purity check — catches numeric leak columns too.
            const vals = nonNullIdx.map((i) => Number(c.values[i]));
            const sorted = [...vals].sort((a, b) => a - b);
            const edges = [0.2, 0.4, 0.6, 0.8].map((q) => quantile(sorted, q));
            const binned = vals.map((v) => String(edges.findIndex((e) => v <= e) === -1 ? 4 : edges.findIndex((e) => v <= e)));
            const purity = categoricalPurity(binned, targetVals);
            if (purity > 0.98) candidates.push({ column: c.name, risk: "HIGH", kind: "perfect-separator", reason: `Even after binning into 5 buckets, this feature predicts the target correctly ${(purity * 100).toFixed(1)}% of the time — check whether it's computed from the outcome.` });
          }
        }
      }
    }

    if (candidates.length === 0) continue;
    // Keep the single strongest signal for this column.
    const best = candidates.reduce((a, b) => (RISK_RANK[b.risk] > RISK_RANK[a.risk] ? b : a));
    warnings.push(best);
  }

  return warnings;
}
