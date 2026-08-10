import React, { useState, useMemo } from "react";
import { EtlOp, Table, runPipeline } from "@/lib/etlUtils";

interface TransformationTimelineProps {
  ops: EtlOp[];
  sourceTable: Table;
  finalTable: Table;
}

export default function TransformationTimeline({ ops, sourceTable, finalTable }: TransformationTimelineProps) {
  const [activeStep, setActiveStep] = useState<number | null>(null);

  // Compute intermediate stages based on ops list
  const stages = useMemo(() => {
    if (ops.length === 0) return [];
    return runPipeline(sourceTable, ops).stages;
  }, [sourceTable, ops]);

  const getOpDescription = (op: any) => {
    switch (op.type) {
      case "filter": return `Kept rows where '${op.col}' ${op.op} '${op.value}'`;
      case "select": return `Selected columns: ${(op.cols || []).join(", ") || "all"}`;
      case "derive": return `Derived new column '${op.name}' from '${op.left}' ${op.arith} '${op.right}'`;
      case "aggregate": return `Grouped by '${op.groupBy}' and calculated ${op.agg?.toUpperCase()} of '${op.aggCol}'`;
      case "sort": return `Sorted by '${op.col}' in ${op.dir}ending order`;
      case "dedupe": return `Removed duplicate rows${(op.cols && op.cols.length > 0) ? ` based on [${op.cols.join(", ")}]` : " across all columns"}`;
      case "clean": return `Cleaned data (Strategy: '${op.mode}')`;
      case "rename": return `Renamed column '${op.col}' to '${op.name}'`;
      case "limit": return `Limited output to ${op.value} rows`;
      case "sample": return `Sampled data (Ratio: ${op.value})`;
      case "map": return `Applied function '${op.fn}' to column '${op.col}'`;
      case "fillna": return `Filled missing values in '${op.col || "all columns"}' with '${op.value}'`;
      case "bucket": return `Created bins '${op.name}' from '${op.col}' (${op.value} buckets)`;
      case "join": return `Joined with another source on '${op.col}' (Type: ${op.joinType})`;
      case "union": return `Unioned with another source (Mode: ${op.mode})`;
      case "pivot": return `Pivoted column '${op.col}', aggregating '${op.aggCol}' by '${op.groupBy}'`;
      case "unpivot": return `Unpivoted columns [${(op.cols || []).join(", ")}] into '${op.name}' & '${op.value}'`;
      case "window": return `Applied window function '${op.fn}' on '${op.col}' partitioned by '${op.groupBy}'`;
      case "regex": return `Extracted regex pattern '${op.value}' from '${op.col}' into '${op.name}'`;
      case "dateparse": return `Parsed date in '${op.col}' using '${op.fn}' into '${op.name}'`;
      default: return `Performed ${op.type} transformation`;
    }
  };

  const getOpIcon = (type: string) => {
    switch (type) {
      case "filter": return "🔍";
      case "select": return "✂️";
      case "derive": return "➕";
      case "aggregate": return "∑";
      case "sort": return "↕️";
      case "dedupe": return "👯";
      case "clean": return "🧹";
      case "rename": return "🏷️";
      case "limit": return "🚧";
      case "join": return "🔗";
      default: return "⚡";
    }
  };

  const renderTablePreview = (table: Table) => {
    if (!table) return null;
    return (
      <div style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
        <h4 style={{ fontSize: 13, marginBottom: 8, color: "#f3f4f6" }}>Data Preview (Top 5 Rows)</h4>
        <div style={{ overflowX: "auto" }}>
          <table className="dtable" style={{ width: "100%", fontSize: 12 }}>
            <thead>
              <tr>
                {table.cols.map(c => <th key={c} style={{ padding: "4px 8px" }}>{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {table.rows.slice(0, 5).map((r, rIdx) => (
                <tr key={rIdx}>
                  {table.cols.map(c => (
                    <td key={c} style={{ padding: "4px 8px", color: "var(--faint)" }}>
                      {r[c] == null ? "—" : String(r[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 6 }}>
          Total Shape: {table.rows.length} rows × {table.cols.length} columns
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: "10px", maxWidth: 900, margin: "0 auto" }}>
      {/* Start Node */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ width: 32, height: 32, borderRadius: "50%", background: "rgba(59, 130, 246, 0.2)", border: "2px solid #3b82f6", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1 }}>
            📥
          </div>
          <div style={{ width: 2, height: activeStep === -1 ? 80 : 30, background: "var(--border)", transition: "height 0.2s" }} />
        </div>
        <div 
          onClick={() => setActiveStep(activeStep === -1 ? null : -1)}
          style={{ background: activeStep === -1 ? "rgba(59, 130, 246, 0.1)" : "rgba(255,255,255,0.02)", border: "1px solid", borderColor: activeStep === -1 ? "#3b82f6" : "var(--border)", borderRadius: 8, padding: "12px 16px", flex: 1, marginTop: -4, cursor: "pointer", transition: "all 0.2s" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontWeight: 600, color: "#60a5fa", marginBottom: 4 }}>Extract Step</div>
            <div style={{ fontSize: 12, color: "var(--faint)" }}>Click to view</div>
          </div>
          <div style={{ fontSize: 13, color: "var(--faint)" }}>
            Source data loaded successfully. Started with <strong style={{ color: "#f3f4f6" }}>{sourceTable.rows.length}</strong> rows.
          </div>
          {activeStep === -1 && renderTablePreview(sourceTable)}
        </div>
      </div>

      {/* Middle Nodes */}
      {ops.map((op, i) => {
        const stageTable = stages[i]?.table;
        const isActive = activeStep === i;
        
        return (
          <div key={op.id || i} style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div style={{ width: 32, height: 32, borderRadius: "50%", background: "rgba(168, 85, 247, 0.2)", border: "2px solid #a855f7", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, zIndex: 1 }}>
                {getOpIcon(op.type)}
              </div>
              <div style={{ width: 2, height: isActive ? 120 : 40, background: "var(--border)", transition: "height 0.2s" }} />
            </div>
            <div 
              onClick={() => setActiveStep(isActive ? null : i)}
              style={{ background: isActive ? "rgba(168, 85, 247, 0.1)" : "rgba(255,255,255,0.02)", border: "1px solid", borderColor: isActive ? "#a855f7" : "var(--border)", borderRadius: 8, padding: "12px 16px", flex: 1, marginTop: -4, cursor: "pointer", transition: "all 0.2s" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontWeight: 600, color: "#c084fc", marginBottom: 4 }}>
                  Step {i + 1}: {op.type.toUpperCase()}
                </div>
                <div style={{ fontSize: 12, color: "var(--faint)" }}>Click to view</div>
              </div>
              <div style={{ fontSize: 13, color: "#e5e7eb", lineHeight: 1.5 }}>
                {getOpDescription(op)}
              </div>
              {isActive && stageTable && renderTablePreview(stageTable)}
            </div>
          </div>
        );
      })}

      {/* End Node */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ width: 32, height: 32, borderRadius: "50%", background: "rgba(16, 185, 129, 0.2)", border: "2px solid #10b981", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1 }}>
            🏁
          </div>
        </div>
        <div 
          onClick={() => setActiveStep(ops.length)}
          style={{ background: activeStep === ops.length ? "rgba(16, 185, 129, 0.1)" : "rgba(255,255,255,0.02)", border: "1px solid", borderColor: activeStep === ops.length ? "#10b981" : "var(--border)", borderRadius: 8, padding: "12px 16px", flex: 1, marginTop: -4, cursor: "pointer", transition: "all 0.2s" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontWeight: 600, color: "#34d399", marginBottom: 4 }}>Final Result</div>
            <div style={{ fontSize: 12, color: "var(--faint)" }}>Click to view</div>
          </div>
          <div style={{ fontSize: 13, color: "var(--faint)" }}>
            Pipeline completed. Final output contains <strong style={{ color: "#f3f4f6" }}>{finalTable.rows.length}</strong> rows.
          </div>
          {activeStep === ops.length && renderTablePreview(finalTable)}
        </div>
      </div>
    </div>
  );
}
