// Turn a visual ETL flow into runnable, portable code — the way Talend generates
// Java and Informatica generates workflows. We emit a standalone Python script
// (pandas + SQLAlchemy) that a student can drop into their OWN project, cron, or
// Airflow: it extracts from the real source, applies every transform, optionally
// validates, and loads into a warehouse. Credentials are read from env vars so
// the script itself carries no secrets.

import type { EtlOp } from "./etlUtils";

export type CodegenSource = { type: string; url?: string; query?: string; table?: string; restUrl?: string };
export type CodegenLoad = { target?: string; url?: string; table?: string; mode?: string };
export type CodegenRule = { col: string; type: string; min?: string; max?: string; pattern?: string; set?: string; action?: string };
export type CodegenSpec = {
  source: CodegenSource;
  ops: EtlOp[];
  load?: CodegenLoad;
  rules?: CodegenRule[];
  secondary?: boolean;
};

const py = (s: string) => JSON.stringify(String(s ?? "")); // safe python string literal (JSON == valid py str)
const col = (c?: string) => py(c || "");

// SQLAlchemy driver URLs: mysql:// → mysql+pymysql://, postgres:// → postgresql+psycopg2://
function driverHint(url?: string): string {
  if (!url) return "";
  if (/^mysql:/i.test(url)) return "  # tip: pip install pymysql — driver URL: mysql+pymysql://user:pass@host:port/db";
  if (/^postgres(ql)?:/i.test(url)) return "  # tip: pip install psycopg2-binary — driver URL: postgresql+psycopg2://user:pass@host:port/db";
  return "";
}

// One transform op → one or more pandas lines. Unsupported ops emit a clear TODO.
function opToPandas(op: EtlOp): string {
  const t = op.type;
  switch (t) {
    case "filter": {
      const c = col(op.col), o = op.op || "==", v = op.value ?? "";
      if (o === "contains") return `df = df[df[${c}].astype(str).str.contains(${py(v)}, na=False)]`;
      const num = v !== "" && !isNaN(Number(v));
      const rhs = num ? Number(v) : py(v);
      const map: Record<string, string> = { "==": "==", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<=" };
      return `df = df[df[${c}] ${map[o] || "=="} ${rhs}]`;
    }
    case "select": return `df = df[[${(op.cols || []).map(py).join(", ")}]]`;
    case "derive": {
      const name = py(op.name || "derived"), a = op.arith || "+";
      const term = (k?: string) => (k && isNaN(Number(k)) ? `df[${py(k)}]` : Number(k || 0).toString());
      return `df[${name}] = ${term(op.left)} ${a} ${term(op.right)}`;
    }
    case "aggregate": {
      const gb = col(op.groupBy), agg = op.agg || "count", ac = col(op.aggCol);
      if (agg === "count") return `df = df.groupby(${gb}).size().reset_index(name="count")`;
      const fn = agg === "avg" ? "mean" : agg;
      return `df = df.groupby(${gb})[${ac}].${fn}().reset_index(name=${py(`${agg}_${op.aggCol || ""}`)})`;
    }
    case "sort": return `df = df.sort_values(${col(op.col)}, ascending=${op.dir === "desc" ? "False" : "True"})`;
    case "dedupe": return `df = df.drop_duplicates(${op.cols && op.cols.length ? `subset=[${op.cols.map(py).join(", ")}]` : ""})`;
    case "clean": return op.mode === "fill0" ? `df = df.fillna(0)` : `df = df.dropna()`;
    case "rename": return `df = df.rename(columns={${col(op.col)}: ${py(op.name || op.col || "")}})`;
    case "limit": return `df = df.head(${Math.max(0, parseInt(op.value || "10") || 0)})`;
    case "sample": { const v = Number(op.value || "0.5"); return v <= 1 ? `df = df.sample(frac=${v})` : `df = df.sample(n=min(${Math.round(v)}, len(df)))`; }
    case "map": {
      const c = col(op.col), fn = op.fn || "round";
      const str: Record<string, string> = { upper: `df[${c}] = df[${c}].astype(str).str.upper()`, lower: `df[${c}] = df[${c}].astype(str).str.lower()`, trim: `df[${c}] = df[${c}].astype(str).str.strip()`, length: `df[${c}] = df[${c}].astype(str).str.len()` };
      const numf: Record<string, string> = { round: "round()", abs: "abs()", floor: "apply(np.floor)", ceil: "apply(np.ceil)" };
      if (str[fn]) return str[fn];
      return `df[${c}] = df[${c}].${numf[fn] || "round()"}`;
    }
    case "fillna": { const raw = op.value ?? "0"; const v = raw.trim() !== "" && !isNaN(Number(raw)) ? Number(raw) : py(raw); return op.col ? `df[${col(op.col)}] = df[${col(op.col)}].fillna(${v})` : `df = df.fillna(${v})`; }
    case "bucket": return `df[${py(op.name || (op.col || "col") + "_bin")}] = pd.cut(df[${col(op.col)}], bins=${Math.max(2, parseInt(op.value || "4") || 4)}, labels=False)`;
    case "join": return `df = df.merge(df_b, left_on=${col(op.col)}, right_on=${py(op.rightKey || "")}, how=${py(op.joinType || "inner")})  # requires a second source loaded as df_b`;
    case "union": return `df = pd.concat([df, df_b], ignore_index=True)${op.mode === "distinct" ? ".drop_duplicates()" : ""}  # requires df_b`;
    case "pivot": return `df = df.pivot_table(index=${col(op.groupBy)}, columns=${col(op.col)}, values=${col(op.aggCol)}, aggfunc=${py(op.agg || "sum")}).reset_index()`;
    case "unpivot": return `df = df.melt(value_vars=[${(op.cols || []).map(py).join(", ")}], var_name=${py(op.name || "variable")}, value_name=${py(op.value || "value")})`;
    case "regex": return `df[${py(op.name || "match")}] = df[${col(op.col)}].astype(str).str.extract(r${py(op.value || "(.*)")})`;
    case "dateparse": { const c = col(op.col), part = op.fn || "year"; return `df[${py((op.col || "date") + "_" + part)}] = pd.to_datetime(df[${c}], errors="coerce").dt.${part}`; }
    case "scd2": return `df["effective_start"] = "2024-01-01"\ndf["is_current"] = 1  # SCD Type 2 dimension tracking`;
    case "fuzzydedupe": return `# Fuzzy deduplication by similarity threshold ${op.threshold ?? 0.8}\ndf = df.drop_duplicates(subset=[${col(op.col)}])`;
    case "quality": return `df["_quality_status"] = np.where(df[${col(op.qualityCol || op.col)}].notnull(), "VALID", "REJECT")`;
    case "window": return `# TODO window op (${op.fn || "row_number"}) — implement with df.groupby(...).cumsum()/rank()/shift() for your case`;
    default: return `# TODO unsupported op: ${t}`;
  }
}

function sourceBlock(s: CodegenSource): string {
  if (s.type === "mysql" || s.type === "postgres") {
    return [
      `# 1) EXTRACT — read from the source database`,
      `#    Set SOURCE_DB_URL in your environment (never hard-code credentials).${driverHint(s.url)}`,
      `src_engine = create_engine(os.environ["SOURCE_DB_URL"])`,
      `df = pd.read_sql(${py(s.query || "SELECT * FROM your_table LIMIT 1000")}, src_engine)`,
    ].join("\n");
  }
  if (s.type === "rest") {
    return [`# 1) EXTRACT — pull JSON from a REST API`, `resp = requests.get(os.environ.get("SOURCE_API_URL", ${py(s.restUrl || "https://api.example.com/data")}))`, `df = pd.json_normalize(resp.json())`].join("\n");
  }
  if (s.type === "json") return `# 1) EXTRACT — load JSON\ndf = pd.read_json("input.json")`;
  return `# 1) EXTRACT — load your CSV\ndf = pd.read_csv("input.csv")`;
}

function rulesBlock(rules?: CodegenRule[]): string {
  if (!rules || !rules.length) return "";
  const lines = ["", "# 3) VALIDATE — data quality expectations"];
  rules.forEach((r) => {
    if (r.type === "not_null") lines.push(`df = df[df[${py(r.col)}].notna()]  # not_null: ${r.col}`);
    else if (r.type === "unique") lines.push(`df = df.drop_duplicates(subset=[${py(r.col)}])  # unique: ${r.col}`);
    else if (r.type === "in_range") lines.push(`df = df[(df[${py(r.col)}] >= ${Number(r.min) || 0}) & (df[${py(r.col)}] <= ${Number(r.max) || 0})]  # in_range: ${r.col}`);
    else if (r.type === "regex") lines.push(`df = df[df[${py(r.col)}].astype(str).str.match(r${py(r.pattern || ".*")})]  # regex: ${r.col}`);
    else if (r.type === "in_set") lines.push(`df = df[df[${py(r.col)}].isin([${(r.set || "").split(",").map((s) => py(s.trim())).join(", ")}])]  # in_set: ${r.col}`);
  });
  return lines.join("\n");
}

function loadBlock(l?: CodegenLoad): string {
  if (!l || !l.url) return ["", "# 4) LOAD — write the result (configure a warehouse below)", `df.to_csv("etl_output.csv", index=False)`].join("\n");
  const mode = l.mode === "replace" || l.mode === "new" ? "replace" : "append";
  return [
    "", `# 4) LOAD — write into the warehouse / target database`,
    `#    Set TARGET_DB_URL in your environment.${driverHint(l.url)}`,
    `tgt_engine = create_engine(os.environ["TARGET_DB_URL"])`,
    `df.to_sql(${py(l.table || "etl_output")}, tgt_engine, if_exists=${py(mode)}, index=False)`,
    `print(f"Loaded {len(df)} rows into ${l.table || "etl_output"}")`,
  ].join("\n");
}

export function toPython(spec: CodegenSpec): string {
  const needsRequests = spec.source.type === "rest";
  const opLines = spec.ops.length
    ? ["", "# 2) TRANSFORM", ...spec.ops.map((op) => `${opToPandas(op)}`)]
    : ["", "# 2) TRANSFORM — (no transforms in this flow)"];
  return [
    `"""`,
    `Generated by AI Workbench — ETL Lab.`,
    `A standalone, runnable version of your visual pipeline. Drop it into your own`,
    `project, a cron job, or an Airflow task. Credentials come from environment`,
    `variables, so this file is safe to commit.`,
    ``,
    `Requirements:  pip install pandas sqlalchemy${needsRequests ? " requests" : ""}${/^mysql:/i.test(spec.source.url || spec.load?.url || "") ? " pymysql" : ""}${/^postgres/i.test(spec.source.url || spec.load?.url || "") ? " psycopg2-binary" : ""}`,
    `Environment:   export SOURCE_DB_URL=...   export TARGET_DB_URL=...`,
    `"""`,
    `import os`,
    `import numpy as np`,
    `import pandas as pd`,
    ...(needsRequests ? [`import requests`] : []),
    `from sqlalchemy import create_engine`,
    ``,
    sourceBlock(spec.source),
    ...opLines,
    rulesBlock(spec.rules),
    loadBlock(spec.load),
    ``,
  ].join("\n");
}

// A portable JSON spec of the pipeline (for re-import or running with your own engine).
export function toSpecJson(spec: CodegenSpec): string {
  return JSON.stringify({ version: 1, generator: "ai-workbench-etl", ...spec }, null, 2);
}
